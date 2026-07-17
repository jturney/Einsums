//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <complex>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Groups einsum nodes that fold into one contraction. All members share the
/// output, the shared operand (+ its index pattern), the C index pattern, and
/// the *same* non-shared tensor read with a permuted index pattern.
struct FoldKey {
    TensorId                 output_id;
    std::vector<std::string> c_indices;
    TensorId                 shared_id;
    bool                     shared_is_first;
    std::vector<std::string> shared_indices;
    TensorId                 non_shared_id;
    std::vector<std::string> non_shared_sorted; // canonicalizes the permutation class

    bool operator==(FoldKey const &o) const {
        return output_id == o.output_id && c_indices == o.c_indices && shared_id == o.shared_id && shared_is_first == o.shared_is_first &&
               shared_indices == o.shared_indices && non_shared_id == o.non_shared_id && non_shared_sorted == o.non_shared_sorted;
    }
};

struct FoldKeyHash {
    size_t operator()(FoldKey const &k) const {
        size_t h = std::hash<TensorId>{}(k.output_id);
        h ^= std::hash<TensorId>{}(k.shared_id) * 2654435761ULL;
        h ^= std::hash<TensorId>{}(k.non_shared_id) * 40503ULL;
        h ^= std::hash<bool>{}(k.shared_is_first) * 19349663ULL;
        for (auto const &v : {std::cref(k.c_indices), std::cref(k.shared_indices), std::cref(k.non_shared_sorted)}) {
            for (auto const &s : v.get()) {
                h ^= std::hash<std::string>{}(s)*16777619ULL + (h << 6) + (h >> 2);
            }
        }
        return h;
    }
};

struct FoldCandidate {
    size_t                   node_index;
    std::vector<std::string> non_shared_indices; // actual (permuted) order on this node
    double                   ab_prefactor;
    bool                     c_pf_is_one;  // true if this node purely accumulates (c_pf == 1)
    bool                     c_pf_is_zero; // true if this node overwrites the output (c_pf == 0)
};

} // namespace

bool LinearCombinationContractionFolding::run(Graph &graph) {
    graph.topological_sort();

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    _num_groups     = 0;
    _num_eliminated = 0;

    if (nodes.size() < 2) {
        return false;
    }

    // --- Phase 1: collect candidates under both orientations ---
    std::unordered_map<FoldKey, std::vector<FoldCandidate>, FoldKeyHash> groups;

    for (size_t ni = 0; ni < nodes.size(); ni++) {
        auto const &node = nodes[ni];
        if (node.kind != OpKind::Einsum) {
            continue;
        }
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            continue;
        }
        // Conjugated contractions are left untouched: this fold doesn't track
        // conj_a/conj_b through the rewrite, so skip them. They still execute
        // correctly, just unfolded. TODO: make conj-aware if it shows up hot.
        if (desc->conj_a || desc->conj_b) {
            continue;
        }
        // Unlike DistributiveFactoring we DON'T skip c_pf==0 nodes: in the 2J-K
        // idiom the first contraction is often an overwrite seed (c_pf=0) and the
        // rest accumulate (c_pf=1). Phase 2 requires every *non-first* member to
        // accumulate, so a stray overwrite in the tail correctly rejects the group.
        if (node.inputs.size() != 2 || node.outputs.size() != 1) {
            continue;
        }
        // Complex prefactors would lose their imaginary part in the real-valued
        // linear combination below, skip to never silently miscompute.
        auto const ab_pf = as_real<double>(desc->ab_prefactor);
        // c_pf must be exactly 1 (pure accumulate) for all but the first node;
        // record it and enforce in Phase 2.
        bool const c_is_one  = is_one(desc->c_prefactor);
        bool const c_is_zero = is_zero(desc->c_prefactor);

        auto const &spec = desc->spec;

        auto sorted_of = [](std::vector<std::string> v) {
            std::ranges::sort(v);
            return v;
        };

        // Orientation A: first operand shared, second (B) is the folded operand.
        {
            FoldKey const key{.output_id         = node.outputs[0],
                              .c_indices         = spec.c_indices,
                              .shared_id         = node.inputs[0],
                              .shared_is_first   = true,
                              .shared_indices    = spec.a_indices,
                              .non_shared_id     = node.inputs[1],
                              .non_shared_sorted = sorted_of(spec.b_indices)};
            groups[key].push_back({.node_index         = ni,
                                   .non_shared_indices = spec.b_indices,
                                   .ab_prefactor       = ab_pf,
                                   .c_pf_is_one        = c_is_one,
                                   .c_pf_is_zero       = c_is_zero});
        }
        // Orientation B: second operand shared, first (A) is the folded operand.
        {
            FoldKey const key{.output_id         = node.outputs[0],
                              .c_indices         = spec.c_indices,
                              .shared_id         = node.inputs[1],
                              .shared_is_first   = false,
                              .shared_indices    = spec.b_indices,
                              .non_shared_id     = node.inputs[0],
                              .non_shared_sorted = sorted_of(spec.a_indices)};
            groups[key].push_back({.node_index         = ni,
                                   .non_shared_indices = spec.a_indices,
                                   .ab_prefactor       = ab_pf,
                                   .c_pf_is_one        = c_is_one,
                                   .c_pf_is_zero       = c_is_zero});
        }
    }

    // --- Phase 2: keep foldable groups (>=2 members, a genuine permutation) ---
    struct ValidGroup {
        FoldKey                    key;
        std::vector<FoldCandidate> candidates;
    };
    std::vector<ValidGroup> valid;

    for (auto &[key, cands] : groups) {
        if (cands.size() < 2) {
            continue;
        }
        // Members run in execution (node-index) order; the first is canonical.
        std::ranges::sort(cands, [](FoldCandidate const &a, FoldCandidate const &b) { return a.node_index < b.node_index; });
        // At least one member must read the operand in a *different* order than
        // the canonical (else this is duplicate-contraction territory, not ours).
        bool any_permuted = false;
        for (size_t i = 1; i < cands.size(); i++) {
            if (cands[i].non_shared_indices != cands[0].non_shared_indices) {
                any_permuted = true;
                break;
            }
        }
        if (!any_permuted) {
            continue;
        }
        // Every non-first member must purely accumulate (c_pf == 1) so the
        // reassociation into a single contraction is exact.
        bool tail_accumulates = true;
        for (size_t i = 1; i < cands.size(); i++) {
            if (!cands[i].c_pf_is_one) {
                tail_accumulates = false;
                break;
            }
        }
        if (!tail_accumulates) {
            continue;
        }
        if (tensors.find(key.non_shared_id) == tensors.end()) {
            continue;
        }
        valid.push_back({.key = key, .candidates = std::move(cands)});
    }

    if (valid.empty()) {
        return false;
    }

    // Largest groups first; greedy so each node folds once.
    std::ranges::sort(valid, [](ValidGroup const &a, ValidGroup const &b) { return a.candidates.size() > b.candidates.size(); });

    size_t const      orig_count = nodes.size(); // appended Alloc nodes (>= this) are always kept
    std::vector<bool> used(orig_count, false);
    std::vector<bool> remove(orig_count, false);
    bool              modified = false;

    for (auto &vg : valid) {
        std::vector<FoldCandidate> members;
        for (auto const &c : vg.candidates) {
            if (!used[c.node_index]) {
                members.push_back(c);
            }
        }
        if (members.size() < 2) {
            continue;
        }

        // --- Phase 3: interference guard ---
        // Between the first and last folded node (execution order), no OTHER
        // node may read or write the output (would observe/clobber a partial
        // sum) or write the shared / non-shared operand (would change a factor
        // mid-fold). This makes the reassociation provably safe.
        size_t lo = members.front().node_index;
        size_t hi = members.back().node_index;
        for (auto const &m : members) {
            lo = std::min(lo, m.node_index);
            hi = std::max(hi, m.node_index);
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
            auto const &nd = nodes[n];
            for (auto const &out : nd.outputs) {
                if (out == vg.key.output_id || out == vg.key.shared_id || out == vg.key.non_shared_id) {
                    interference = true;
                    break;
                }
            }
            if (interference) {
                break;
            }
            for (auto const &in : nd.inputs) {
                if (in == vg.key.output_id) { // someone reads the partial sum
                    interference = true;
                    break;
                }
            }
        }
        if (interference) {
            if (_verbosity >= 3) {
                auto on = tensors.find(vg.key.output_id);
                report(3, fmt::format("skip fold into '{}': an intervening node reads/writes the output or an operand",
                                      on != tensors.end() ? on->second.name : "?"));
            }
            continue;
        }

        // --- Phase 4: rewrite ---
        auto nb_it = tensors.find(vg.key.non_shared_id);
        if (nb_it == tensors.end()) {
            continue;
        }
        auto const &b_handle = nb_it->second;

        // The combine executor casts the user operands (shared, non-shared,
        // output) to GeneralRuntimeTensor<T>. Statically-typed Tensor<T, Rank>
        // captures produce the same handle shape, and a blind cast is type
        // confusion (a segfault in the fused axpy). Fold only when
        // all three really are runtime tensors.
        auto const is_rt = [&](TensorId tid) {
            auto it = tensors.find(tid);
            return it != tensors.end() && it->second.is_runtime;
        };
        if (!is_rt(vg.key.output_id) || !is_rt(vg.key.shared_id) || !is_rt(vg.key.non_shared_id)) {
            continue;
        }

        auto const dtype = b_handle.dtype;

        // L = sum_k (ab_k / ab_0) * P_k(B), in operand-0's canonical layout, and a
        // reused scratch T for permuted contributions. Both RUNTIME tensors so the
        // combine below can cast operands uniformly to GeneralRuntimeTensor<T>.
        auto l_res = graph.create_zero_runtime_tensor_dynamic(fmt::format("_lccf_L_{}", _num_groups), dtype, b_handle.dims);
        if (!l_res) {
            continue;
        }
        TensorId const l_id  = l_res.value().first;
        auto           t_res = graph.create_zero_runtime_tensor_dynamic(fmt::format("_lccf_T_{}", _num_groups), dtype, b_handle.dims);
        if (!t_res) {
            continue;
        }
        TensorId const t_id = t_res.value().first;

        double ab0 = members[0].ab_prefactor;
        if (ab0 == 0.0) {
            ab0 = 1.0;
        }
        auto const &canonical = members[0].non_shared_indices;

        // Per-member contribution descriptor, resolved against runtime tensors at
        // execution time (no static-rank assumptions).
        struct Contribution {
            bool              is_permute;
            double            scale;
            ParsedPermuteSpec spec;
        };
        std::vector<Contribution> contribs;
        contribs.reserve(members.size());
        for (auto const &m : members) {
            double const scale = m.ab_prefactor / ab0;
            if (m.non_shared_indices == canonical) {
                contribs.push_back({.is_permute = false, .scale = scale, .spec = {}});
            } else {
                ParsedPermuteSpec pspec;
                pspec.c_indices = canonical;
                pspec.a_indices = m.non_shared_indices;
                pspec.raw       = fmt::format("{} <- {}", fmt::join(canonical, ","), fmt::join(m.non_shared_indices, ","));
                contribs.push_back({.is_permute = true, .scale = scale, .spec = std::move(pspec)});
            }
        }

        // The fused contraction is node-0's einsum with its non-shared operand
        // replaced by L. Build a ParsedEinsumSpec from node-0's index lists and run
        // string_einsum reading L DIRECTLY (no slot mutation, thread-safe and
        // unambiguous when the operands are shared across the graph).
        auto const *n0_desc = std::get_if<EinsumDescriptor>(&nodes[members[0].node_index].op_data);
        if (n0_desc == nullptr) {
            continue;
        }
        ParsedEinsumSpec einspec;
        einspec.c_indices           = n0_desc->spec.c_indices;
        einspec.a_indices           = n0_desc->spec.a_indices;
        einspec.b_indices           = n0_desc->spec.b_indices;
        einspec.raw                 = fmt::format("{} <- {} ; {}", fmt::join(einspec.c_indices, ","), fmt::join(einspec.a_indices, ","),
                                                  fmt::join(einspec.b_indices, ","));
        auto const     c_pf0        = as_real<double>(n0_desc->c_prefactor);
        TensorId const nonshared_id = vg.key.non_shared_id;
        TensorId const shared_id    = vg.key.shared_id;
        TensorId const out_id       = vg.key.output_id;
        bool const     shared_first = vg.key.shared_is_first;
        Graph         *graph_ptr    = &graph;

        // Captured for the verbosity report below, before einspec is moved into the lambda.
        std::string const fold_out_indices = fmt::format("{}", fmt::join(einspec.c_indices, ","));

        auto combined = [contribs = std::move(contribs), einspec = std::move(einspec), c_pf0, ab0, shared_first, graph_ptr, nonshared_id,
                         shared_id, out_id, l_id, t_id, dtype]() {
            auto build = [&]<typename T>(T /*tag*/) {
                using RT = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto *L  = static_cast<RT *>(graph_ptr->tensor(l_id).tensor_ptr);
                auto *B  = static_cast<RT *>(graph_ptr->tensor(nonshared_id).tensor_ptr);
                auto *Tt = static_cast<RT *>(graph_ptr->tensor(t_id).tensor_ptr);
                L->zero();
                for (auto const &c : contribs) {
                    if (c.is_permute) {
                        dispatch::string_permute<RT, RT>(c.spec, T{0}, Tt, T{1}, *B); // T = P_k(B)
                        linear_algebra::axpy(static_cast<T>(c.scale), *Tt, L);        // L += scale * T
                    } else {
                        linear_algebra::axpy(static_cast<T>(c.scale), *B, L); // L += scale * B
                    }
                }
                // Single fused contraction: out = c_pf0*out + ab0 * (shared op L).
                auto     *out = static_cast<RT *>(graph_ptr->tensor(out_id).tensor_ptr);
                auto     *sh  = static_cast<RT *>(graph_ptr->tensor(shared_id).tensor_ptr);
                RT const *Aop = shared_first ? sh : L;
                RT const *Bop = shared_first ? L : sh;
                dispatch::string_einsum<RT, RT, RT>(einspec, static_cast<T>(c_pf0), out, static_cast<T>(ab0), *Aop, *Bop);
            };
            switch (dtype) {
            case packed_gemm::ScalarType::Float32:
                build(float{});
                break;
            case packed_gemm::ScalarType::Float64:
                build(double{});
                break;
            case packed_gemm::ScalarType::Complex64:
                build(std::complex<float>{});
                break;
            case packed_gemm::ScalarType::Complex128:
                build(std::complex<double>{});
                break;
            default:
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "LinearCombinationContractionFolding: unknown ScalarType");
            }
        };

        Node fused;
        fused.kind    = OpKind::Custom;
        fused.label   = fmt::format("lccf({} terms via _lccf_L_{})", members.size(), _num_groups);
        fused.execute = std::move(combined);
        fused.inputs  = {vg.key.shared_id, nonshared_id};
        // When node-0 accumulates (c_pf != 0) the fused op is a read-modify-write of
        // the output: declare the output as an input too, so the scheduler keeps it
        // ordered after the seed and the other accumulations into that tensor.
        if (!members[0].c_pf_is_zero) {
            fused.inputs.push_back(vg.key.output_id);
        }
        fused.outputs = {vg.key.output_id, l_id, t_id};
        // Place the fused node AT node-0's vector position (not appended). The
        // topological sort derives RAW/WAR/WAW edges from scan order, so the fused
        // node must occupy node-0's slot to remain the last writer of the output
        // before its consumer (appending would schedule it after the consumer).
        fused.id                     = nodes[members[0].node_index].id;
        nodes[members[0].node_index] = std::move(fused);
        used[members[0].node_index]  = true; // kept (now the fused node)

        for (size_t mi = 1; mi < members.size(); mi++) { // members[0] reused for the fused node
            remove[members[mi].node_index] = true;
            used[members[mi].node_index]   = true;
            _num_eliminated++;
        }
        _num_groups++;
        modified = true;

        if (_verbosity >= 2) {
            auto        on = tensors.find(out_id);
            std::string member_desc;
            for (auto const &m : members) {
                member_desc +=
                    fmt::format("{}node {}=[{}]", member_desc.empty() ? "" : ", ", m.node_index, fmt::join(m.non_shared_indices, ","));
            }
            report(2, fmt::format("fold {} contractions into '{}' [{}] via L=Σ(ab/{})·perm(operand); members {}", members.size(),
                                  on != tensors.end() ? on->second.name : "?", fold_out_indices, ab0, member_desc));
        }
    }

    if (!modified) {
        return false;
    }

    // Keep appended fused nodes (index >= orig_count); drop folded originals.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i >= orig_count || !remove[i]) {
            filtered.push_back(std::move(nodes[i]));
        }
    }
    nodes = std::move(filtered);
    graph.topological_sort();

    EINSUMS_LOG_INFO("LinearCombinationContractionFolding: folded {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("folded {} group(s), replacing {} contractions with {} fused node(s)", _num_groups, _num_eliminated + _num_groups,
                          _num_groups));
    return true;
}

} // namespace einsums::compute_graph::passes
