//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/StreamContractionFusion.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <complex>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::compute_graph::passes {

namespace {

/// One fusable einsum: streams tensor S (pattern s_indices) against small
/// operand W into output C, all indices of C and W drawn from s_indices.
struct StreamMember {
    size_t                   node_index;
    TensorId                 out_id;
    TensorId                 w_id;
    std::vector<std::string> s_indices; // this member's index pattern for S
    std::vector<std::string> c_indices;
    std::vector<std::string> w_indices;
    PrefactorScalar          alpha;
    bool                     c_pf_is_one;
    bool                     c_pf_is_zero;
    PrefactorScalar          c_pf;
};

bool no_repeats(std::vector<std::string> const &v) {
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = i + 1; j < v.size(); j++) {
            if (v[i] == v[j]) {
                return false;
            }
        }
    }
    return true;
}

bool subset_of(std::vector<std::string> const &small, std::vector<std::string> const &big) {
    for (auto const &s : small) {
        if (std::ranges::find(big, s) == big.end()) {
            return false;
        }
    }
    return true;
}

/// The streaming kernel for one element type. Walks S once in storage order
/// (axes sorted by descending stride) and feeds every member's accumulator:
///   C_k[rho_k(idx)] += alpha_k * S[idx] * W_k[pi_k(idx)]
/// The coordinate-to-offset maps are affine, so each member carries one
/// offset delta per storage axis of S and the walk maintains running offsets
/// with an odometer.
///
/// With a non-empty allowed_axes (owner-computes chunking) the kernel
/// partitions the highest-stride allowed physical S axis into per-thread
/// blocks - the stride decides because a low-stride partition turns each
/// thread's read into a strided comb through S, while a high-stride one
/// keeps contiguous slabs. Each thread walks only its slab; members whose
/// output pattern contains the axis label write DIRECTLY into their
/// (disjoint) output slices, members without it accumulate into
/// thread-private buffers reduced at the end. With allowed_axes empty every
/// member is privatized and threads split the flattened outer space; the
/// caller guarantees privatized outputs are small (max_output_elems).
template <typename T>
void run_stream(Graph *graph, TensorId s_id, std::vector<StreamMember> const &members, std::vector<TensorId> const &unique_outs,
                std::vector<int> const &allowed_axes) {
    using RT = GeneralRuntimeTensor<T, std::allocator<T>>;

    auto const *S      = static_cast<RT const *>(graph->tensor(s_id).tensor_ptr);
    int const   rank   = static_cast<int>(S->rank());
    size_t      s_size = 1;
    for (int d = 0; d < rank; d++) {
        s_size *= S->dim(d);
    }
    if (s_size == 0) {
        return;
    }

    // Storage-order axes: descending S stride, so the last walks stride-1.
    std::vector<int> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);
    std::ranges::sort(axes, [&](int x, int y) { return S->stride(x) > S->stride(y); });

    std::vector<int64_t> dims(rank), s_delta(rank);
    for (int d = 0; d < rank; d++) {
        dims[d]    = static_cast<int64_t>(S->dim(axes[d]));
        s_delta[d] = static_cast<int64_t>(S->stride(axes[d]));
    }

    // Offset span of a runtime tensor: 1 + sum((dim-1) * stride). Equals the
    // element count for dense layouts and bounds every reachable offset for
    // strided ones, so span-sized private buffers can never overflow.
    auto span_of = [](RT const *X) {
        size_t s = 1;
        for (size_t d = 0; d < X->rank(); d++) {
            if (X->dim(static_cast<int>(d)) == 0) {
                return size_t{0};
            }
            s += (X->dim(static_cast<int>(d)) - 1) * X->stride(static_cast<int>(d));
        }
        return s;
    };

    // Prescale the outputs (member order): 0 -> zero, 1 -> keep, else scale
    // (stride-aware odometer, so strided outputs scale only their own
    // elements). Only the first member touching an output may have
    // c_pf != 1 (validated by the pass), so this is exact.
    std::unordered_set<TensorId> prescaled;
    for (auto const &m : members) {
        if (prescaled.insert(m.out_id).second) {
            auto *C = static_cast<RT *>(graph->tensor(m.out_id).tensor_ptr);
            if (m.c_pf_is_zero) {
                C->zero();
            } else if (!m.c_pf_is_one) {
                T                   *cd = C->data();
                T const              f  = as<T>(m.c_pf);
                int const            cr = static_cast<int>(C->rank());
                std::vector<int64_t> cc(static_cast<size_t>(cr), 0);
                bool                 done = (span_of(C) == 0);
                int64_t              off  = 0;
                while (!done) {
                    cd[off] *= f;
                    int d = cr - 1;
                    for (; d >= 0; d--) {
                        cc[static_cast<size_t>(d)]++;
                        off += static_cast<int64_t>(C->stride(d));
                        if (cc[static_cast<size_t>(d)] < static_cast<int64_t>(C->dim(d))) {
                            break;
                        }
                        off -= cc[static_cast<size_t>(d)] * static_cast<int64_t>(C->stride(d));
                        cc[static_cast<size_t>(d)] = 0;
                    }
                    done = (d < 0);
                }
            }
        }
    }

    // Partition axis: the allowed axis with the largest S stride.
    int part_axis = -1;
    {
        size_t best_stride = 0;
        for (int const axis : allowed_axes) {
            if (S->stride(axis) > best_stride) {
                best_stride = S->stride(axis);
                part_axis   = axis;
            }
        }
    }

    // Per-member affine deltas: for storage axis d (label = member's S
    // pattern at original axis axes[d]), the label's stride in C / W, or 0.
    struct Plan {
        T                    alpha;
        T const             *w;
        T                   *out;
        bool                 direct;
        std::vector<int64_t> cdelta, wdelta;
        size_t               out_slot;
        size_t               out_elems;
    };
    std::vector<Plan> plans;
    plans.reserve(members.size());
    for (auto const &m : members) {
        Plan        p;
        auto const *W = static_cast<RT const *>(graph->tensor(m.w_id).tensor_ptr);
        auto       *C = static_cast<RT *>(graph->tensor(m.out_id).tensor_ptr);
        p.alpha       = as<T>(m.alpha);
        p.w           = W->data();
        p.out         = C->data();
        p.direct      = part_axis >= 0 && std::ranges::find(m.c_indices, m.s_indices[static_cast<size_t>(part_axis)]) != m.c_indices.end();
        p.cdelta.resize(rank, 0);
        p.wdelta.resize(rank, 0);
        for (int d = 0; d < rank; d++) {
            auto const &label = m.s_indices[static_cast<size_t>(axes[d])];
            if (auto it = std::ranges::find(m.c_indices, label); it != m.c_indices.end()) {
                p.cdelta[d] = static_cast<int64_t>(C->stride(static_cast<int>(it - m.c_indices.begin())));
            }
            if (auto it = std::ranges::find(m.w_indices, label); it != m.w_indices.end()) {
                p.wdelta[d] = static_cast<int64_t>(W->stride(static_cast<int>(it - m.w_indices.begin())));
            }
        }
        p.out_slot  = static_cast<size_t>(std::ranges::find(unique_outs, m.out_id) - unique_outs.begin());
        p.out_elems = span_of(C);
        plans.push_back(std::move(p));
    }

    // Storage-order position of the partition axis, if any.
    int pd = -1;
    if (part_axis >= 0) {
        for (int d = 0; d < rank; d++) {
            if (axes[d] == part_axis) {
                pd = d;
                break;
            }
        }
    }

    T const *s_data = S->data();

    std::vector<std::vector<std::vector<T>>> thread_priv; // [thread][out_slot][elem]

#ifdef _OPENMP
    int const nthreads = omp_get_max_threads();
#else
    int const nthreads = 1;
#endif
    thread_priv.resize(static_cast<size_t>(nthreads));

#ifdef _OPENMP
#    pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        int const tid = omp_get_thread_num();
#else
        int const tid = 0;
#endif
        // This thread's iteration space: with a partition axis, one block of
        // that axis and the full extent of every other (bases carry the
        // block offset); otherwise the full volume with the flattened outer
        // space chunked across threads.
        std::vector<int64_t> ldims  = dims;
        int64_t              s_base = 0;
        std::vector<int64_t> c_base(plans.size(), 0), w_base(plans.size(), 0);
        int64_t              q0 = 0, q1 = 0;

        if (pd >= 0) {
            int64_t const extent  = dims[pd];
            int const     nblocks = static_cast<int>(std::min<int64_t>(nthreads, extent));
            if (tid < nblocks) {
                int64_t const b0 = extent * tid / nblocks;
                int64_t const b1 = extent * (tid + 1) / nblocks;
                ldims[pd]        = b1 - b0;
                s_base           = b0 * s_delta[pd];
                for (size_t k = 0; k < plans.size(); k++) {
                    c_base[k] = b0 * plans[k].cdelta[pd];
                    w_base[k] = b0 * plans[k].wdelta[pd];
                }
                q1 = 1;
                for (int d = 0; d < rank - 1; d++) {
                    q1 *= ldims[d];
                }
            }
        } else {
            int64_t outer_total = 1;
            for (int d = 0; d < rank - 1; d++) {
                outer_total *= dims[d];
            }
            int64_t const chunk = (outer_total + nthreads - 1) / nthreads;
            q0                  = std::min<int64_t>(static_cast<int64_t>(tid) * chunk, outer_total);
            q1                  = std::min<int64_t>(q0 + chunk, outer_total);
        }
        int64_t const inner_n = ldims[rank - 1];

        auto &priv = thread_priv[static_cast<size_t>(tid)];
        priv.resize(unique_outs.size());
        for (size_t u = 0; u < unique_outs.size(); u++) {
            size_t elems = 0;
            if (q0 < q1) {
                for (auto const &p : plans) {
                    if (p.out_slot == u && !p.direct) {
                        elems = p.out_elems;
                    }
                }
            }
            priv[u].assign(elems, T{0});
        }

        if (q0 < q1) {
            // Decompose q0 into outer coordinates and compute starting offsets.
            std::vector<int64_t> coord(static_cast<size_t>(rank), 0);
            {
                int64_t rem = q0;
                for (int d = rank - 2; d >= 0; d--) {
                    coord[static_cast<size_t>(d)] = rem % ldims[d];
                    rem /= ldims[d];
                }
            }
            int64_t              s_off = s_base;
            std::vector<int64_t> c_off = c_base, w_off = w_base;
            for (int d = 0; d < rank - 1; d++) {
                s_off += coord[static_cast<size_t>(d)] * s_delta[d];
                for (size_t k = 0; k < plans.size(); k++) {
                    c_off[k] += coord[static_cast<size_t>(d)] * plans[k].cdelta[d];
                    w_off[k] += coord[static_cast<size_t>(d)] * plans[k].wdelta[d];
                }
            }

            for (int64_t q = q0; q < q1; q++) {
                // Innermost walk: constant per-step deltas.
                int64_t  si = s_off;
                T const *sp = s_data;
                for (size_t k = 0; k < plans.size(); k++) {
                    auto const   &p  = plans[k];
                    T            *cb = p.direct ? p.out : priv[p.out_slot].data();
                    int64_t const dc = p.cdelta[rank - 1];
                    int64_t const dw = p.wdelta[rank - 1];
                    int64_t       co = c_off[k];
                    int64_t       wo = w_off[k];
                    si               = s_off;
                    int64_t const ds = s_delta[rank - 1];
                    for (int64_t i = 0; i < inner_n; i++) {
                        cb[co] += p.alpha * sp[si] * p.w[wo];
                        si += ds;
                        co += dc;
                        wo += dw;
                    }
                }

                // Odometer: advance the outer coordinates (bases stay put:
                // the wrap subtracts only the local-axis contribution).
                for (int d = rank - 2; d >= 0; d--) {
                    coord[static_cast<size_t>(d)]++;
                    s_off += s_delta[d];
                    for (size_t k = 0; k < plans.size(); k++) {
                        c_off[k] += plans[k].cdelta[d];
                        w_off[k] += plans[k].wdelta[d];
                    }
                    if (coord[static_cast<size_t>(d)] < ldims[d]) {
                        break;
                    }
                    s_off -= coord[static_cast<size_t>(d)] * s_delta[d];
                    for (size_t k = 0; k < plans.size(); k++) {
                        c_off[k] -= coord[static_cast<size_t>(d)] * plans[k].cdelta[d];
                        w_off[k] -= coord[static_cast<size_t>(d)] * plans[k].wdelta[d];
                    }
                    coord[static_cast<size_t>(d)] = 0;
                }
            }
        }
    }

    // Reduce thread-private accumulators into the real outputs. The private
    // buffers are dense in each output's OWN storage order because cdelta was
    // built from the output's strides... they are not: cdelta indexes the
    // output's real strided layout, so the private buffers use the same
    // (possibly strided) offsets. Allocate them at the output's dense span
    // and add elementwise - spans match because outputs are dense runtime
    // tensors (allocated by capture) and offsets stay within [0, elems).
    for (size_t u = 0; u < unique_outs.size(); u++) {
        auto *C  = static_cast<RT *>(graph->tensor(unique_outs[u]).tensor_ptr);
        T    *cd = C->data();
        for (int t = 0; t < nthreads; t++) {
            auto const &buf = thread_priv[static_cast<size_t>(t)][u];
            for (size_t i = 0; i < buf.size(); i++) {
                cd[i] += buf[i];
            }
        }
    }
}

} // namespace

size_t StreamContractionFusion::max_output_elems(size_t elem_size) const {
    if (!_has_profile || _profile.cpu.caches.empty() || elem_size == 0) {
        return kMaxOutputElemsFallback;
    }
    size_t llc_bytes = 0;
    for (auto const &level : _profile.cpu.caches) {
        llc_bytes = std::max(llc_bytes, level.size_bytes);
    }
    if (llc_bytes == 0) {
        return kMaxOutputElemsFallback;
    }
#ifdef _OPENMP
    auto const threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
#else
    size_t const threads = 1;
#endif
    return std::max(kMinOutputElemsFloor, llc_bytes / threads / elem_size);
}

bool StreamContractionFusion::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();

    _num_groups     = 0;
    _num_eliminated = 0;

    if (nodes.size() < 2) {
        return false;
    }

    auto const handle_of = [&](TensorId tid) -> TensorHandle const * { return graph.find_tensor(tid); };

    // --- Phase 1: collect stream-fusable candidates, grouped by streamed tensor ---
    std::unordered_map<TensorId, std::vector<StreamMember>> groups;

    for (size_t ni = 0; ni < nodes.size(); ni++) {
        auto const &node = nodes[ni];
        if (node.kind != OpKind::Einsum || node.inputs.size() != 2 || node.outputs.size() != 1) {
            continue;
        }
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr || desc->conj_a || desc->conj_b) {
            continue;
        }
        auto const &spec = desc->spec;
        if (!no_repeats(spec.c_indices) || !no_repeats(spec.a_indices) || !no_repeats(spec.b_indices)) {
            continue;
        }

        bool const c_is_one  = is_one(desc->c_prefactor);
        bool const c_is_zero = is_zero(desc->c_prefactor);

        // Try each operand as the streamed tensor S; the other is W. The
        // member qualifies when C's and W's labels are all drawn from S's.
        for (int orient = 0; orient < 2; orient++) {
            TensorId const s_id  = orient == 0 ? node.inputs[0] : node.inputs[1];
            TensorId const w_id  = orient == 0 ? node.inputs[1] : node.inputs[0];
            auto const    &s_idx = orient == 0 ? spec.a_indices : spec.b_indices;
            auto const    &w_idx = orient == 0 ? spec.b_indices : spec.a_indices;

            if (!subset_of(spec.c_indices, s_idx) || !subset_of(w_idx, s_idx)) {
                continue;
            }

            auto const *sh = handle_of(s_id);
            auto const *wh = handle_of(w_id);
            auto const *ch = handle_of(node.outputs[0]);
            if (sh == nullptr || wh == nullptr || ch == nullptr || !sh->is_runtime || !wh->is_runtime || !ch->is_runtime) {
                continue;
            }
            // The kernel casts all three operands to the streamed tensor's
            // element type; a mixed-dtype member must stay unfused.
            if (wh->dtype != sh->dtype || ch->dtype != sh->dtype) {
                continue;
            }
            // Complex prefactors ride through the kernel as element-typed
            // alphas; on a real dtype a nonzero imaginary part has nowhere
            // to go, so such members stay unfused.
            bool const complex_dtype = sh->dtype == packed_gemm::ScalarType::Complex64 || sh->dtype == packed_gemm::ScalarType::Complex128;
            if (!complex_dtype && (!is_real_valued(desc->ab_prefactor) || !is_real_valued(desc->c_prefactor))) {
                continue;
            }
            // Distributed operands belong to the communication passes
            // (InputSlicing / SUMMAExpansion); fusing them into a local
            // Custom node would hide the einsum those passes rewrite.
            if (sh->is_distributed || wh->is_distributed || ch->is_distributed) {
                continue;
            }
            // No output-size gate here: phase 2 admits large outputs when an
            // owner-computes partition axis covers them, and only privatized
            // members are held to max_output_elems.
            size_t const s_elems = sh->total_elems();
            size_t const w_elems = wh->total_elems();
            size_t const c_elems = ch->total_elems();
            if (s_elems < kMinStreamElems || s_elems < kMinSizeRatio * w_elems || s_elems < kMinSizeRatio * c_elems) {
                continue;
            }

            groups[s_id].push_back({.node_index   = ni,
                                    .out_id       = node.outputs[0],
                                    .w_id         = w_id,
                                    .s_indices    = s_idx,
                                    .c_indices    = spec.c_indices,
                                    .w_indices    = w_idx,
                                    .alpha        = desc->ab_prefactor,
                                    .c_pf_is_one  = c_is_one,
                                    .c_pf_is_zero = c_is_zero,
                                    .c_pf         = desc->c_prefactor});
            break; // one orientation per node is enough
        }
    }

    // --- Phase 2: validate groups ---
    size_t const      orig_count = nodes.size();
    std::vector<bool> used(orig_count, false);
    std::vector<bool> remove(orig_count, false);
    bool              modified = false;

    for (auto &[s_id, cands] : groups) {
        std::vector<StreamMember> members;
        for (auto &c : cands) {
            if (!used[c.node_index]) {
                members.push_back(std::move(c));
            }
        }
        if (members.size() < 2) {
            continue;
        }
        std::ranges::sort(members, [](StreamMember const &a, StreamMember const &b) { return a.node_index < b.node_index; });

        // Owner-computes chunking, engaged ONLY when some output exceeds the
        // privatization cap: a physical S axis whose label lands in a
        // member's output pins one output coordinate, so threads owning
        // disjoint blocks of that axis write disjoint output slices DIRECTLY
        // (no thread-private copy, no reduction) - fusing outputs the flat
        // walk would have to decline. An axis is allowed when every member
        // it does NOT cover fits under the cap; the kernel picks the
        // highest-stride allowed axis at execute time (partitioning a
        // low-stride axis turns each thread's read into a strided comb -
        // measured ~5x slower than contiguous slabs - so layout, which only
        // the kernel knows, drives the choice). Over-cap members no axis can
        // cover drop out of the group and stay ordinary einsums. When
        // nothing is over the cap the flat privatized walk is kept: its
        // contiguous per-thread slabs are the measured-fastest layout, and
        // the private buffers are cache-resident by construction.
        size_t const cap    = max_output_elems(handle_of(s_id)->element_size);
        auto const  &s_dims = handle_of(s_id)->dims;
        size_t const s_rank = s_dims.size();

        auto const out_elems_of = [&](StreamMember const &m) {
            auto const *h = handle_of(m.out_id);
            return h == nullptr ? size_t{0} : h->total_elems();
        };
        auto const covered_by = [](StreamMember const &m, size_t axis) {
            return std::ranges::find(m.c_indices, m.s_indices[axis]) != m.c_indices.end();
        };

        std::vector<int> allowed_axes;
        if (std::ranges::any_of(members, [&](StreamMember const &m) { return out_elems_of(m) > cap; })) {
            for (size_t d = 0; d < s_rank; d++) {
                if (s_dims[d] < 2) {
                    continue; // a length-1 block feeds one thread
                }
                bool const valid =
                    std::ranges::all_of(members, [&](StreamMember const &m) { return covered_by(m, d) || out_elems_of(m) <= cap; });
                if (valid) {
                    allowed_axes.push_back(static_cast<int>(d));
                }
            }
            if (allowed_axes.empty()) {
                std::erase_if(members, [&](StreamMember const &m) { return out_elems_of(m) > cap; });
                if (members.size() < 2) {
                    continue;
                }
            }
        }

        // Shared-output tail rule: only the first member touching an output
        // may have c_pf != 1 (contributions interleave in the fused stream,
        // so a later overwrite/scale is not reproducible).
        {
            std::unordered_set<TensorId> seen;
            bool                         ok = true;
            for (auto const &m : members) {
                if (!seen.insert(m.out_id).second && !m.c_pf_is_one) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
        }

        // Interference guard: between the first and last member, no other
        // node may write S, any W, or any output, and none may read an
        // output (it would observe a partial sum).
        size_t const lo = members.front().node_index;
        size_t const hi = members.back().node_index;

        std::unordered_set<TensorId> outs, ws;
        for (auto const &m : members) {
            outs.insert(m.out_id);
            ws.insert(m.w_id);
        }

        std::vector<bool> is_member(nodes.size(), false);
        for (auto const &m : members) {
            is_member[m.node_index] = true;
        }
        bool interference = false;
        for (size_t n = lo + 1; n < hi && !interference; n++) {
            if (is_member[n]) {
                continue;
            }
            for (auto const &out : nodes[n].outputs) {
                if (out == s_id || outs.contains(out) || ws.contains(out)) {
                    interference = true;
                    break;
                }
            }
            if (interference) {
                break;
            }
            for (auto const &in : nodes[n].inputs) {
                if (outs.contains(in)) {
                    interference = true;
                    break;
                }
            }
        }
        if (interference) {
            report(3, fmt::format("skip stream fusion over tensor {}: an intervening node touches an operand or output",
                                  handle_of(s_id) != nullptr ? handle_of(s_id)->name : "?"));
            continue;
        }

        // --- Phase 3: build the fused node ---
        auto const dtype = handle_of(s_id)->dtype;

        std::vector<TensorId> unique_outs;
        for (auto const &m : members) {
            if (std::ranges::find(unique_outs, m.out_id) == unique_outs.end()) {
                unique_outs.push_back(m.out_id);
            }
        }

        Graph *graph_ptr = &graph;
        auto   fused_fn  = [graph_ptr, s_id, members, unique_outs, dtype, allowed_axes]() {
            switch (dtype) {
            case packed_gemm::ScalarType::Float32:
                run_stream<float>(graph_ptr, s_id, members, unique_outs, allowed_axes);
                break;
            case packed_gemm::ScalarType::Float64:
                run_stream<double>(graph_ptr, s_id, members, unique_outs, allowed_axes);
                break;
            case packed_gemm::ScalarType::Complex64:
                run_stream<std::complex<float>>(graph_ptr, s_id, members, unique_outs, allowed_axes);
                break;
            case packed_gemm::ScalarType::Complex128:
                run_stream<std::complex<double>>(graph_ptr, s_id, members, unique_outs, allowed_axes);
                break;
            default:
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "StreamContractionFusion: unknown ScalarType");
            }
        };

        Node fused;
        fused.kind = OpKind::Custom;
        fused.label =
            fmt::format("stream_fusion({} members over '{}')", members.size(), handle_of(s_id) != nullptr ? handle_of(s_id)->name : "?");
        fused.execute = std::move(fused_fn);
        fused.inputs  = {s_id};
        for (auto const &m : members) {
            fused.inputs.push_back(m.w_id);
            if (!m.c_pf_is_zero) {
                fused.inputs.push_back(m.out_id); // RMW: order after prior writers
            }
        }
        for (auto const &out : unique_outs) {
            fused.outputs.push_back(out);
        }

        fused.id                     = nodes[members[0].node_index].id;
        nodes[members[0].node_index] = std::move(fused);
        used[members[0].node_index]  = true;

        for (size_t mi = 1; mi < members.size(); mi++) {
            remove[members[mi].node_index] = true;
            used[members[mi].node_index]   = true;
            _num_eliminated++;
        }
        _num_groups++;
        modified = true;

        report(2, fmt::format(
                      "fused {} contractions into one stream over '{}' feeding {} output(s){}", members.size(),
                      handle_of(s_id) != nullptr ? handle_of(s_id)->name : "?", unique_outs.size(),
                      allowed_axes.empty() ? "" : fmt::format(" (owner-computes chunking, {} candidate axis/axes)", allowed_axes.size())));
    }

    if (!modified) {
        return false;
    }

    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i >= orig_count || !remove[i]) {
            filtered.push_back(std::move(nodes[i]));
        }
    }
    nodes = std::move(filtered);
    graph.topological_sort();

    EINSUMS_LOG_INFO("StreamContractionFusion: fused {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("fused {} stream group(s), replacing {} contractions with {} fused node(s)", _num_groups,
                          _num_eliminated + _num_groups, _num_groups));
    return true;
}

} // namespace einsums::compute_graph::passes
