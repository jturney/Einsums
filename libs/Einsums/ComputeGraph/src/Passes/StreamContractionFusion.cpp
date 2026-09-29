//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/StreamContractionFusion.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/Stream.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <complex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

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

/// Execute one fused group: hand every member to packed_gemm::stream_contract,
/// which walks S once in storage order and feeds each member's output. The
/// graph's part is only resolving TensorIds to live storage and index labels to
/// axis positions; the walk, the privatized accumulators and the owner-computes
/// partition all live in the kernel, which the eager string dispatch shares.
template <typename T>
void run_stream(Graph *graph, TensorId s_id, std::vector<StreamMember> const &members, std::vector<int> const &allowed_axes) {
    using Impl = ::einsums::detail::TensorImpl<T>;

    // Geometry comes from the handle's LIVE rank-erased impl, never from a cast
    // of the tensor object itself. Both spellings resolve the object the graph
    // keeps alive (capture adopts an operand's storage into a stand-in so the
    // caller's wrapper may die before execute()), but a runtime OPERAND is a
    // GeneralRuntimeTensor<T> or a RuntimeTensorView<T> and those are unrelated
    // classes with unrelated vtable layouts. Casting one to the other and
    // calling a virtual jumped through the wrong slot: a view-operand group
    // reached rank() and landed in set_name(), which is a segfault at the first
    // fused node. TensorImpl<T> is the one representation both types agree on,
    // and impl_fn() re-reads it at call time so a re-emplaced view is seen.
    auto const impl_of = [graph](TensorId id) -> Impl * {
        auto const *handle = graph->find_tensor(id);
        if (handle == nullptr || !handle->impl_fn) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "StreamContractionFusion: fused operand {} has no live tensor impl", id);
        }
        return static_cast<Impl *>(handle->impl_fn());
    };
    auto const layout_of = [](Impl const *X) {
        packed_gemm::StreamLayout l;
        for (size_t d = 0; d < X->rank(); d++) {
            l.dims.push_back(static_cast<int64_t>(X->dim(static_cast<int>(d))));
            l.strides.push_back(static_cast<int64_t>(X->stride(static_cast<int>(d))));
        }
        return l;
    };
    auto const axis_map = [](std::vector<std::string> const &s_indices, std::vector<std::string> const &indices) {
        std::vector<int> map(s_indices.size(), -1);
        for (size_t d = 0; d < s_indices.size(); d++) {
            if (auto it = std::ranges::find(indices, s_indices[d]); it != indices.end()) {
                map[d] = static_cast<int>(it - indices.begin());
            }
        }
        return map;
    };

    auto const *S = impl_of(s_id);

    std::vector<packed_gemm::StreamTerm<T>> terms;
    terms.reserve(members.size());
    for (auto const &m : members) {
        auto       *C = impl_of(m.out_id);
        auto const *W = impl_of(m.w_id);
        terms.push_back({.c        = C->data(),
                         .c_layout = layout_of(C),
                         .w        = W->data(),
                         .w_layout = layout_of(W),
                         .c_axis   = axis_map(m.s_indices, m.c_indices),
                         .w_axis   = axis_map(m.s_indices, m.w_indices),
                         .alpha    = as<T>(m.alpha),
                         .c_pf     = m.c_pf_is_one ? T{1} : (m.c_pf_is_zero ? T{0} : as<T>(m.c_pf))});
    }

    packed_gemm::stream_contract<T>(S->data(), layout_of(S), terms, allowed_axes);
}

} // namespace

size_t StreamContractionFusion::max_output_elems(size_t elem_size) const {
    if (!_has_cost_model || _cost_model.cpu.caches.empty() || elem_size == 0) {
        return kMaxOutputElemsFallback;
    }
    size_t llc_bytes = 0;
    for (auto const &level : _cost_model.cpu.caches) {
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

std::vector<std::string> StreamContractionFusion::explain() const {
    if (_num_groups == 0) {
        return {};
    }
    return {fmt::format("StreamContractionFusion: loop-fused {} group(s), eliminating {} node(s)", _num_groups, _num_eliminated)};
}

void StreamContractionFusion::reset_stats() {
    _num_groups     = 0;
    _num_eliminated = 0;
}

bool StreamContractionFusion::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();

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
        if (!understands(graph, node)) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", node.label, describe_features(features_of(graph, node))));
            continue;
        }
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr || live_conj_a(*desc) || live_conj_b(*desc)) {
            continue;
        }
        auto const &spec = desc->spec;
        if (!no_repeats(spec.c_indices) || !no_repeats(spec.a_indices) || !no_repeats(spec.b_indices)) {
            continue;
        }

        bool const c_is_one  = is_one(live_c_prefactor(*desc));
        bool const c_is_zero = is_zero(live_c_prefactor(*desc));

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
            // The kernel reads every operand's geometry through the handle's
            // live TensorImpl, which is the one representation an owning runtime
            // tensor and a runtime view share. A handle without one (a tile-wise
            // sparse tensor) has no such geometry to read.
            if (!sh->impl_fn || !wh->impl_fn || !ch->impl_fn) {
                note_skip("operand has no live tensor impl",
                          fmt::format("streamed tensor '{}' or an operand of node {} is tile-wise sparse", sh->name, ni));
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
            if (!complex_dtype && (!is_real_valued(live_ab_prefactor(*desc)) || !is_real_valued(live_c_prefactor(*desc)))) {
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
                                    .alpha        = live_ab_prefactor(*desc),
                                    .c_pf_is_one  = c_is_one,
                                    .c_pf_is_zero = c_is_zero,
                                    .c_pf         = live_c_prefactor(*desc)});
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

        // Every id below is resolved to the BUFFER it names
        // (Graph::buffer_of), never compared raw. A view and its parent are
        // different TensorIds over the same memory, and in a real capture views
        // are the common case (a DLPNO-CCSD iteration registers ~5800 of them
        // against ~1200 whole tensors), so an id-only comparison answers "these
        // do not interact" about two accesses to one buffer. That is what makes
        // fusing unsound rather than merely suboptimal: fusion interleaves the
        // members, and an interleaving is only equivalent to the original order
        // when no member's write is observable by another member or by any node
        // it moves across.
        auto const root_of = [&](TensorId id) { return graph.buffer_of(id); };

        TensorId const               s_root = root_of(s_id);
        std::unordered_set<TensorId> out_roots, w_roots;
        for (auto const &m : members) {
            out_roots.insert(root_of(m.out_id));
            w_roots.insert(root_of(m.w_id));
        }

        // Intra-group aliasing. Writing into the buffer being streamed changes
        // what later elements of the same stream read; writing into a weight
        // changes what later members multiply by; and two outputs in one buffer
        // cannot both be prescaled up front and then accumulated, because the
        // original order let the second output's prefactor overwrite the first
        // one's contribution. All three are decided per buffer, so a view of
        // the streamed tensor counts as the streamed tensor.
        {
            std::unordered_set<TensorId> distinct_outs;
            for (auto const &m : members) {
                distinct_outs.insert(m.out_id);
            }
            bool const outs_share_buffer = distinct_outs.size() != out_roots.size();
            if (outs_share_buffer || out_roots.contains(s_root) ||
                std::ranges::any_of(w_roots, [&](TensorId w) { return out_roots.contains(w); })) {
                note_skip("group operands share storage",
                          fmt::format("stream over '{}': an output aliases the streamed tensor, a weight, or another output",
                                      handle_of(s_id) != nullptr ? handle_of(s_id)->name : "?"));
                continue;
            }
        }

        // Interference guard: between the first and last member, no other node
        // may write the streamed buffer, any weight buffer or any output
        // buffer, and none may read an output buffer (it would observe a
        // partial sum, since the fused node moves every member's write to the
        // first member's position).
        size_t const lo = members.front().node_index;
        size_t const hi = members.back().node_index;

        std::vector<bool> is_member(nodes.size(), false);
        for (auto const &m : members) {
            is_member[m.node_index] = true;
        }
        // A control-flow node lists none of what its sub-graphs read or write,
        // so the lists scanned below cannot clear it: a Loop between members
        // whose body writes an operand would go unseen. Decline any group that
        // spans one.
        bool spans_control_flow = false;
        for (size_t n = lo + 1; n < hi; n++) {
            if (!is_member[n] && is_control_flow(nodes[n].kind)) {
                spans_control_flow = true;
                break;
            }
        }
        if (spans_control_flow) {
            note_skip("a control-flow node lies between the members, and what its sub-graphs touch is not in its node lists",
                      fmt::format("stream over '{}'", handle_of(s_id) != nullptr ? handle_of(s_id)->name : "?"));
            continue;
        }

        bool interference = false;
        for (size_t n = lo + 1; n < hi && !interference; n++) {
            if (is_member[n]) {
                continue;
            }
            for (auto const &out : nodes[n].outputs) {
                TensorId const r = root_of(out);
                if (r == s_root || out_roots.contains(r) || w_roots.contains(r)) {
                    interference = true;
                    break;
                }
            }
            if (interference) {
                break;
            }
            for (auto const &in : nodes[n].inputs) {
                if (out_roots.contains(root_of(in))) {
                    interference = true;
                    break;
                }
            }
        }
        if (interference) {
            note_skip("an intervening node touches an operand or output",
                      fmt::format("stream over '{}': a node between the members shares a buffer with one of them",
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

        auto const anchor   = graph.anchor();
        auto       fused_fn = [anchor, s_id, members, dtype, allowed_axes]() {
            Graph *const graph_now = &anchor->graph();
            detail::dispatch_scalar_type(dtype, [&](auto tag) { run_stream<decltype(tag)>(graph_now, s_id, members, allowed_axes); });
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

    graph.erase_nodes(remove);
    graph.topological_sort();

    EINSUMS_LOG_INFO("StreamContractionFusion: fused {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("fused {} stream group(s), replacing {} contractions with {} fused node(s)", _num_groups,
                          _num_eliminated + _num_groups, _num_groups));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
