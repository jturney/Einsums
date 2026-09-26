//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <complex>
#include <tuple>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

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

    bool operator==(FoldKey const &) const = default;

    /// A total order over every field, so two distinct keys always compare
    /// unequal in one direction or the other. Used only to break a tie in the
    /// fold order; the values it compares carry no meaning of their own.
    bool operator<(FoldKey const &o) const {
        return std::tie(output_id, shared_id, non_shared_id, shared_is_first, c_indices, shared_indices, non_shared_sorted) <
               std::tie(o.output_id, o.shared_id, o.non_shared_id, o.shared_is_first, o.c_indices, o.shared_indices, o.non_shared_sorted);
    }
};

struct FoldKeyHash {
    size_t operator()(FoldKey const &k) const {
        size_t h = 0;
        hash_combine(h, k.output_id);
        hash_combine(h, k.shared_id);
        hash_combine(h, k.non_shared_id);
        hash_combine(h, k.shared_is_first);
        hash_range(h, k.c_indices);
        hash_range(h, k.shared_indices);
        hash_range(h, k.non_shared_sorted);
        return h;
    }
};

struct FoldCandidate {
    size_t                   node_index;
    std::vector<std::string> non_shared_indices; // actual (permuted) order on this node
    PrefactorScalar          ab_prefactor;
    bool                     c_pf_is_one;  // true if this node purely accumulates (c_pf == 1)
    bool                     c_pf_is_zero; // true if this node overwrites the output (c_pf == 0)
};

} // namespace

void LinearCombinationContractionFolding::reset_stats() {
    _num_groups     = 0;
    _num_eliminated = 0;
}

std::vector<std::string> LinearCombinationContractionFolding::explain() const {
    if (_num_groups == 0) {
        return {};
    }
    return {fmt::format("LinearCombinationContractionFolding: folded {} group(s), replacing {} contraction(s) with {} fused contraction(s) "
                        "plus {} operand-combination node(s)",
                        _num_groups, _num_eliminated + _num_groups, _num_groups, _num_groups)};
}

bool LinearCombinationContractionFolding::run(Graph &graph) {
    graph.topological_sort();

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

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
        if (!understands(graph, node)) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", node.label, describe_features(features_of(graph, node))));
            continue;
        }
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr) {
            continue;
        }
        // Conjugated contractions are left untouched: this fold doesn't track
        // conj_a/conj_b through the rewrite, so skip them. They still execute
        // correctly, just unfolded. TODO: make conj-aware if it shows up hot.
        if (live_conj_a(*desc) || live_conj_b(*desc)) {
            continue;
        }
        // Unlike DistributiveFactoring we DON'T skip c_pf==0 nodes: in the 2J-K
        // idiom the first contraction is often an overwrite seed (c_pf=0) and the
        // rest accumulate (c_pf=1). Phase 2 requires every *non-first* member to
        // accumulate, so a stray overwrite in the tail correctly rejects the group.
        if (node.inputs.size() != 2 || node.outputs.size() != 1) {
            continue;
        }
        // Prefactors stay type-erased; the fold kernel converts with as<T>()
        // so complex prefactors fold exactly on complex tensors. Real dtypes
        // are gated in Phase 4 (a nonzero imaginary part cannot land in a
        // real linear combination).
        auto const &ab_pf = live_ab_prefactor(*desc);
        // c_pf must be exactly 1 (pure accumulate) for all but the first node;
        // record it and enforce in Phase 2.
        bool const c_is_one  = is_one(live_c_prefactor(*desc));
        bool const c_is_zero = is_zero(live_c_prefactor(*desc));

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
            // Same operand read the same way by every member: that is duplicate
            // work for CSE to collapse, not a linear combination to fold.
            note_skip("group members all read the folded operand in the same order (no transpose to fold)",
                      fmt::format("{} members into tensor {}", cands.size(), key.output_id));
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
            note_skip("a non-first group member overwrites the output instead of accumulating (c_prefactor != 1)",
                      fmt::format("{} members into tensor {}", cands.size(), key.output_id));
            continue;
        }
        if (!tensors.contains(key.non_shared_id)) {
            continue;
        }
        valid.push_back({.key = key, .candidates = std::move(cands)});
    }

    if (valid.empty()) {
        return false;
    }

    // Largest groups first; greedy so each node folds once. Equal-sized groups are
    // ordered by their SMALLEST member node index (their members were sorted by
    // node index above, so that is candidates.front()), and identically-placed
    // groups by the FoldKey itself, which is unique because it is the map key.
    // Together those make the order a total one that depends only on the graph.
    //
    // The size comparison alone was not: `groups` is an unordered_map keyed by a
    // hash, so equal-sized groups came out of it in hash-iteration order, which
    // differs between standard libraries (libc++ and libstdc++ iterate a bucket
    // newest-first, the MSVC STL in insertion order) and moves whenever the hash
    // does. Fold order decides which group gets which ordinal in the emitted node
    // labels and `_lccf_L_<n>` tensor names, so one and the same input produced a
    // different graph, and different saved IR bytes, per platform.
    std::ranges::sort(valid, [](ValidGroup const &a, ValidGroup const &b) {
        if (a.candidates.size() != b.candidates.size()) {
            return a.candidates.size() > b.candidates.size();
        }
        if (a.candidates.front().node_index != b.candidates.front().node_index) {
            return a.candidates.front().node_index < b.candidates.front().node_index;
        }
        return a.key < b.key;
    });

    size_t const      orig_count = nodes.size(); // appended Alloc nodes (>= this) are always kept
    std::vector<bool> used(orig_count, false);
    std::vector<bool> remove(orig_count, false);
    // (position in the ORIGINAL numbering, nodes to splice immediately before it).
    // Collected while folding and applied after erase_nodes, with the positions
    // remapped for the erased nodes below each one.
    std::vector<std::pair<std::size_t, std::vector<Node>>> pending_inserts;
    bool                                                   modified = false;

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
        // vg.candidates was sorted by node index in Phase 2 and `members` is an
        // order-preserving filter of it, so the first and last members ARE the
        // ends of the span.
        size_t const      lo = members.front().node_index;
        size_t const      hi = members.back().node_index;
        std::vector<bool> is_member(nodes.size(), false);
        for (auto const &m : members) {
            is_member[m.node_index] = true;
        }
        std::unordered_set<TensorId> const operand_ids{vg.key.shared_id, vg.key.non_shared_id};
        bool const interference = span_interferes(graph, lo, hi, is_member, {vg.key.output_id}, operand_ids, /*reject_control_flow=*/true);
        if (interference) {
            auto on = tensors.find(vg.key.output_id);
            note_skip("an intervening node reads or writes the output or a folded operand",
                      fmt::format("fold into '{}' over nodes [{}..{}]", on != tensors.end() ? on->second.name : "?", lo, hi));
            continue;
        }

        // --- Phase 4: rewrite ---
        auto const *b_handle = graph.find_tensor(vg.key.non_shared_id);
        if (b_handle == nullptr) {
            continue;
        }

        // L takes the folded operand's element type, and the fused contraction
        // reads it in that operand's place, so all three must share it. Any
        // tensor with a rank-erased geometry qualifies: the L build and the fused
        // contraction both read their operands through it, strides included, so
        // a view or a statically-typed tensor is as good as a runtime one.
        auto const dtype   = b_handle->dtype;
        auto const same_dt = [&](TensorId tid) {
            auto const *h = graph.find_tensor(tid);
            return h != nullptr && h->dtype == dtype && static_cast<bool>(h->impl_fn);
        };
        if (!same_dt(vg.key.output_id) || !same_dt(vg.key.shared_id) || !same_dt(vg.key.non_shared_id)) {
            note_skip("the output and both operands must share one element type and a rank-erased geometry",
                      fmt::format("group of {} into tensor {}", members.size(), vg.key.output_id));
            continue;
        }

        // L = sum_k (ab_k / ab_0) * P_k(B), in operand-0's canonical layout.
        // The fused contraction is node-0's einsum; grab its descriptor up
        // front so the prefactor gate below can see c_prefactor too.
        auto const *n0_desc = nodes[members[0].node_index].op_data.get_if<EinsumDescriptor>();
        if (n0_desc == nullptr) {
            continue;
        }

        // Complex prefactors fold exactly on complex dtypes (the kernel
        // carries them as T); on a real dtype a nonzero imaginary part has
        // nowhere to go, so such groups stay unfolded.
        bool const complex_dtype = dtype == packed_gemm::ScalarType::Complex64 || dtype == packed_gemm::ScalarType::Complex128;
        if (!complex_dtype) {
            bool const all_real = std::ranges::all_of(members, [](FoldCandidate const &m) { return is_real_valued(m.ab_prefactor); }) &&
                                  is_real_valued(live_c_prefactor(*n0_desc));
            if (!all_real) {
                continue;
            }
        }

        auto l_res = graph.create_zero_runtime_tensor_dynamic(fmt::format("_lccf_L_{}", _num_groups), dtype, b_handle->dims);
        if (!l_res) {
            continue;
        }
        TensorId const l_id = l_res.value().first;

        // The create_* calls above append Alloc nodes and may reallocate the
        // node vector, dangling the descriptor pointer fetched for the gate;
        // re-resolve it before building the fused spec from it.
        n0_desc = nodes[members[0].node_index].op_data.get_if<EinsumDescriptor>();

        PrefactorScalar ab0 = members[0].ab_prefactor;
        if (is_zero(ab0)) {
            ab0 = 1.0;
        }
        auto const &canonical = members[0].non_shared_indices;

        // One permute per member, L = beta*L + (ab/ab0) * P_k(B), with beta zero
        // on the first so it overwrites. A member reading B in the canonical
        // order is the identity permute. The ab prefactor stays type-erased;
        // the kernel forms the ratio ab/ab0 in T so complex prefactors fold
        // exactly.
        struct Contribution {
            PrefactorScalar   ab;
            ParsedPermuteSpec spec;
        };
        std::vector<Contribution> contribs;
        contribs.reserve(members.size());
        for (auto const &m : members) {
            ParsedPermuteSpec pspec;
            pspec.c_indices = canonical;
            pspec.a_indices = m.non_shared_indices;
            pspec.raw       = pspec.render();
            contribs.push_back({.ab = m.ab_prefactor, .spec = std::move(pspec)});
        }

        // The fused contraction is node-0's einsum with its non-shared operand
        // replaced by L. Build a ParsedEinsumSpec from node-0's index lists and run
        // string_einsum reading L DIRECTLY (no slot mutation, thread-safe and
        // unambiguous when the operands are shared across the graph).
        ParsedEinsumSpec einspec;
        einspec.c_indices                  = n0_desc->spec.c_indices;
        einspec.a_indices                  = n0_desc->spec.a_indices;
        einspec.b_indices                  = n0_desc->spec.b_indices;
        einspec.raw                        = einspec.render();
        PrefactorScalar const c_pf0        = live_c_prefactor(*n0_desc);
        TensorId const        nonshared_id = vg.key.non_shared_id;
        TensorId const        shared_id    = vg.key.shared_id;
        TensorId const        out_id       = vg.key.output_id;
        bool const            shared_first = vg.key.shared_is_first;

        // Captured for the verbosity report below, before einspec is moved into the lambda.
        std::string const fold_out_indices = fmt::format("{}", fmt::join(einspec.c_indices, ","));

        // Emit TWO nodes, not one fused blob:
        //
        //   A: L = sum_k (ab_k/ab0) * P_k(B)      -- reads only the non-shared operand
        //   B: out = c_pf0*out + ab0 * (shared op L)
        //
        // Building L inside the contraction's executor made it invisible: the
        // combined node reads the varying shared operand, so nothing could tell
        // that the L half depends only on B. In a loop that meant L was rebuilt
        // from scratch on every replay even when B was loop-invariant, which is
        // the common case (B is an integral block, one-time setup). Split out,
        // node A has invariant inputs, a single writer, and a destination it does
        // not read, so LoopInvariantHoisting's existing criteria lift it out of
        // the loop with no changes to that pass.
        //
        // Both operands are read through accessors bound here, which go through the
        // slot first: a rebind or a redirected slot of B lands on the next replay, and
        // a view or a statically-typed operand is read through its own geometry.
        OperandAccessor const b_access = resolve_operand(graph, nonshared_id, "LinearCombinationContractionFolding", "B");
        OperandAccessor const l_access = resolve_operand(graph, l_id, "LinearCombinationContractionFolding", "L");
        auto                  build_l  = [contribs = std::move(contribs), ab0, b_access, l_access, dtype]() {
            detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
                auto   *L     = l_access.impl<T>();
                auto   *B     = b_access.impl<T>();
                T const ab0_t = as<T>(ab0);
                bool    first = true;
                for (auto const &c : contribs) {
                    T const scale = as<T>(c.ab) / ab0_t; // exact in T, complex included
                    dispatch::string_permute_impl<T>(c.spec, first ? T{0} : T{1}, L, scale, *B);
                    first = false;
                }
            });
        };

        // Node A. Writes L and does not read it, so it is a pure producer as far as
        // the hoisting and liveness analyses are concerned.
        Node lbuild;
        lbuild.kind    = OpKind::Custom;
        lbuild.label   = fmt::format("lccf_build_L({} terms -> _lccf_L_{})", members.size(), _num_groups);
        lbuild.execute = std::move(build_l);
        lbuild.inputs  = {nonshared_id};
        lbuild.outputs = {l_id};
        lbuild.id      = graph.reserve_node_id();

        // Node B, the contraction: node-0's einsum with its non-shared operand
        // replaced by L, so the index lists and prefactors are unchanged and the
        // spec carries over verbatim. Built through make_einsum_node rather than as
        // a baked closure so it stays a REAL OpKind::Einsum with a live descriptor,
        // which is what lets CSE, DeadNodeElimination, ScaleAbsorption,
        // PermuteFusion, StreamContractionFusion and the placement passes see it
        // instead of stepping over an opaque Custom node. The factory also applies
        // the RMW input convention for the accumulating case.
        //
        // Operand ORDER matters now: a real einsum node's descriptor implies
        // inputs[0] is operand A, so L has to land in the slot the non-shared
        // operand occupied.
        TensorId const a_operand = shared_first ? shared_id : l_id;
        TensorId const b_operand = shared_first ? l_id : shared_id;
        Node fused = graph.make_einsum_node(a_operand, b_operand, out_id, einspec, c_pf0, ab0, /*conj_a=*/false,
                                            /*conj_b=*/false, fmt::format("lccf({} terms via _lccf_L_{})", members.size(), _num_groups));
        // Keep node-0's id: state keyed by NodeId (profiler payload strings, the
        // program-order validator's observed-writes map) refers to it, and node-0 is
        // the node being replaced, so no duplicate arises.
        fused.id = nodes[members[0].node_index].id;
        // Place the fused node AT node-0's vector position (not appended). The
        // topological sort derives RAW/WAR/WAW edges from scan order, so the fused
        // node must occupy node-0's slot to remain the last writer of the output
        // before its consumer (appending would schedule it after the consumer).
        nodes[members[0].node_index] = std::move(fused);
        used[members[0].node_index]  = true; // kept (now the fused node)
        // A goes immediately before it; collected and spliced after the removal
        // bookkeeping so the recorded indices stay in the original numbering.
        pending_inserts.emplace_back(members[0].node_index, std::vector<Node>{std::move(lbuild)});

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
                                  on != tensors.end() ? on->second.name : "?", fold_out_indices, to_string(ab0), member_desc));
        }
    }

    if (!modified) {
        return false;
    }

    // Keep appended fused nodes (index >= orig_count); drop folded originals, and
    // splice each L-builder immediately before its contraction. A contraction's own
    // slot is never erased (it took over node-0's), so the shifted position still
    // points at it and inserting "before" lands the builder directly ahead of its
    // consumer. @see Graph::replace_nodes for the position bookkeeping.
    graph.replace_nodes(remove, std::move(pending_inserts));
    graph.topological_sort();

    EINSUMS_LOG_INFO("LinearCombinationContractionFolding: folded {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("folded {} group(s), replacing {} contractions with {} fused node(s)", _num_groups, _num_eliminated + _num_groups,
                          _num_groups));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
