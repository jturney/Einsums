//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Key for grouping factorable einsum nodes.
struct FactorKey {
    TensorId                 output_id;
    TensorId                 shared_input_id;
    bool                     shared_is_first;
    std::vector<std::string> non_shared_indices;

    bool operator==(FactorKey const &o) const {
        return output_id == o.output_id && shared_input_id == o.shared_input_id && shared_is_first == o.shared_is_first &&
               non_shared_indices == o.non_shared_indices;
    }
};

struct FactorKeyHash {
    size_t operator()(FactorKey const &k) const {
        size_t h = std::hash<TensorId>{}(k.output_id);
        h ^= std::hash<TensorId>{}(k.shared_input_id) * 2654435761ULL;
        h ^= std::hash<bool>{}(k.shared_is_first) * 40503ULL;
        for (auto const &s : k.non_shared_indices) {
            h ^= std::hash<std::string>{}(s)*16777619ULL;
        }
        return h;
    }
};

struct FactorCandidate {
    size_t   node_index;
    TensorId non_shared_input;
    double   ab_prefactor;
};

/// A real prefactor exactly equal to one.
///
/// The factored form applies the output prefactor ONCE, so every member after
/// the first has to be a pure accumulation. A member with ``c_pf != 1`` rescales
/// the partial sum its predecessors already wrote, which the single combined
/// contraction cannot reproduce: `W = 2*(W + A*B0) + A*B1` is not
/// `W = 2*W + A*(B0 + B1)`.
bool is_unit_real(PrefactorScalar const &p) {
    return is_real_valued(p) && as_real<double>(p) == 1.0;
}

} // namespace

void DistributiveFactoring::reset_stats() {
    _num_groups     = 0;
    _num_eliminated = 0;
}

bool DistributiveFactoring::run(Graph &graph) {
    // Own the recursion (like LoopInvariantHoisting): reset the counters once
    // here at the root, then descend ourselves. If we instead opted into
    // PassManager auto-recursion, run() would be re-invoked per subgraph and
    // reset the top-level tally each time. recurse_into_subgraphs()
    // returns false so the PassManager does not double-walk.
    _groups.clear();
    return run_recursive(graph);
}

bool DistributiveFactoring::run_recursive(Graph &graph) {
    bool modified = factor_one_level(graph);
    // Factor loop bodies and conditional branches too; counters accumulate
    // across the whole tree (no per-level reset).
    graph.for_each_subgraph([&](Graph &sub) {
        if (run_recursive(sub)) {
            modified = true;
        }
    });
    return modified;
}

bool DistributiveFactoring::factor_one_level(Graph &graph) {
    graph.topological_sort();

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    if (nodes.size() < 2) {
        return false;
    }

    // --- Phase 1: Collect candidates ---
    std::unordered_map<FactorKey, std::vector<FactorCandidate>, FactorKeyHash> candidate_groups;

    for (size_t ni = 0; ni < nodes.size(); ni++) {
        auto const &node = nodes[ni];
        if (node.kind != OpKind::Einsum)
            continue;

        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (!desc)
            continue;
        if (desc->conj_a || desc->conj_b)
            continue; // conjugated contractions aren't factored (conj not threaded through the rewrite)
        if (is_zero(desc->c_prefactor))
            continue;
        if (node.inputs.size() != 2 || node.outputs.size() != 1)
            continue;

        TensorId const out_id = node.outputs[0];
        TensorId const in_a   = node.inputs[0];
        TensorId const in_b   = node.inputs[1];
        auto const    &spec   = desc->spec;

        // The factoring math below is real-valued; a prefactor with nonzero
        // imaginary part would silently lose it, so skip those nodes.
        if (!is_real_valued(desc->ab_prefactor)) {
            continue;
        }
        auto const ab_pf_d = as_real<double>(desc->ab_prefactor);

        // Try first input as shared
        {
            FactorKey const key{
                .output_id = out_id, .shared_input_id = in_a, .shared_is_first = true, .non_shared_indices = spec.b_indices};
            candidate_groups[key].push_back({.node_index = ni, .non_shared_input = in_b, .ab_prefactor = ab_pf_d});
        }
        // Try second input as shared
        {
            FactorKey const key{
                .output_id = out_id, .shared_input_id = in_b, .shared_is_first = false, .non_shared_indices = spec.a_indices};
            candidate_groups[key].push_back({.node_index = ni, .non_shared_input = in_a, .ab_prefactor = ab_pf_d});
        }
    }

    // --- Phase 2: Filter to valid groups ---
    struct ValidGroup {
        FactorKey                    key;
        std::vector<FactorCandidate> candidates;
    };
    std::vector<ValidGroup> valid_groups;

    for (auto &[key, candidates] : candidate_groups) {
        if (candidates.size() < 2)
            continue;

        // All non-shared inputs must have same shape and dtype
        auto it0 = tensors.find(candidates[0].non_shared_input);
        if (it0 == tensors.end())
            continue;
        auto const &ref_dims  = it0->second.dims;
        auto const  ref_dtype = it0->second.dtype;

        bool all_compatible = true;
        for (size_t ci = 1; ci < candidates.size(); ci++) {
            auto it = tensors.find(candidates[ci].non_shared_input);
            if (it == tensors.end() || it->second.dims != ref_dims || it->second.dtype != ref_dtype) {
                all_compatible = false;
                break;
            }
        }
        if (!all_compatible)
            continue;

        // All non-shared inputs must be different tensors
        bool has_duplicate = false;
        for (size_t ci = 0; ci < candidates.size() && !has_duplicate; ci++) {
            for (size_t cj = ci + 1; cj < candidates.size(); cj++) {
                if (candidates[ci].non_shared_input == candidates[cj].non_shared_input) {
                    has_duplicate = true;
                    break;
                }
            }
        }
        if (has_duplicate)
            continue;

        // The rewrite redirects the non-shared operand's SLOT (keyed by tensor id)
        // to the summed intermediate T. If the shared operand is the SAME tensor as
        // a summed operand, they share that slot, so a self-contraction like A*A
        // would read T for BOTH factors (T*T instead of A*T). Reject aliased groups;
        // the slot-redirect trick cannot separate the two reads.
        bool shared_aliases_nonshared = false;
        for (auto const &c : candidates) {
            if (c.non_shared_input == key.shared_input_id) {
                shared_aliases_nonshared = true;
                break;
            }
        }
        if (shared_aliases_nonshared)
            continue;

        valid_groups.push_back({.key = key, .candidates = std::move(candidates)});
    }

    if (valid_groups.empty()) {
        return false;
    }

    // --- Phase 3: Deduplicate (largest group first, greedy) ---
    std::ranges::sort(valid_groups, [](ValidGroup const &a, ValidGroup const &b) { return a.candidates.size() > b.candidates.size(); });

    // create_*_tensor_dynamic appends Alloc nodes (index >= orig_count) that are
    // always kept; the removal / used sets cover only the original nodes.
    size_t const      orig_count = nodes.size();
    std::vector<bool> node_used(orig_count, false);
    std::vector<bool> remove(orig_count, false);
    bool              modified = false;

    // Replacement nodes, keyed by the pre-erase position they belong at.
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;

    for (auto &vg : valid_groups) {
        std::vector<FactorCandidate> available;
        for (auto const &c : vg.candidates) {
            if (!node_used[c.node_index]) {
                available.push_back(c);
            }
        }
        if (available.size() < 2)
            continue;

        // Candidates are collected in ascending node-index (execution) order and
        // that order survives filtering, so available.front() is the earliest
        // member and available.back() the latest.
        size_t const first_pos = available.front().node_index;
        size_t const last_pos  = available.back().node_index;

        // --- Pure-accumulation gate ---
        // Only the first member's output prefactor survives into the combined
        // contraction, so the rest must accumulate with exactly 1. See
        // @ref is_unit_real for why anything else changes the result.
        {
            bool unit_accumulate = true;
            for (size_t ci = 1; ci < available.size(); ci++) {
                auto const *d = std::get_if<EinsumDescriptor>(&nodes[available[ci].node_index].op_data);
                if (d == nullptr || !is_unit_real(d->c_prefactor)) {
                    unit_accumulate = false;
                    break;
                }
            }
            if (!unit_accumulate) {
                auto on = tensors.find(vg.key.output_id);
                report(3, fmt::format("skip factoring into '{}': a member accumulates with a prefactor other than 1, which would rescale "
                                      "the partial sum",
                                      on != tensors.end() ? on->second.name : "?"));
                continue;
            }
        }

        // --- Placement / interference gate (mirrors GEMMBatching / LCCF) ---
        // The combined node takes the FIRST member's slot so it stays scan-before
        // any consumer of the factored output. Position IS program order in this
        // IR, so appending the writer would let a later reader of the output run
        // first and observe a stale buffer, and the PassManager's program-order
        // verifier (check_observed_writes) would reject the rewrite.
        // The slot move is only sound when no OTHER node between the first and last
        // member reads/writes the output (would observe/clobber the partial sum)
        // or writes the shared / non-shared operands (would change a factor
        // mid-fold). Control-flow nodes hide their I/O in sub-graphs, so their
        // presence in the span disqualifies the group outright.
        {
            std::vector<bool>            is_member(nodes.size(), false);
            std::unordered_set<TensorId> operand_ids;
            operand_ids.insert(vg.key.shared_input_id);
            for (auto const &c : available) {
                is_member[c.node_index] = true;
                operand_ids.insert(c.non_shared_input);
            }
            bool const interference =
                span_interferes(nodes, first_pos, last_pos, is_member, vg.key.output_id, operand_ids, /*reject_control_flow=*/true);
            if (interference) {
                auto on = tensors.find(vg.key.output_id);
                report(3, fmt::format("skip factoring into '{}': an intervening node reads/writes the output or a factor operand",
                                      on != tensors.end() ? on->second.name : "?"));
                continue;
            }
        }

        // --- Phase 4: Rewrite the graph ---

        // Get the reference tensor handle for the non-shared operands
        auto ref_it = tensors.find(available[0].non_shared_input);
        if (ref_it == tensors.end())
            continue;
        auto const &ref_handle = ref_it->second;

        // The combined executor redirects the einsum's non-shared operand slot to
        // the accumulator T, so T must be the SAME tensor kind the operands are: a
        // runtime graph reads a GeneralRuntimeTensor at that slot, a compile-time
        // graph a Tensor<T, Rank>. Require every summed operand to share that kind
        // and build a matching accumulator, so make_zero/make_axpy dispatch
        // correctly - a compile-time accumulator fed runtime operands rank-errors
        // at execute on any Python-captured graph.
        bool const operands_runtime = ref_handle.is_runtime;
        bool       kinds_uniform    = true;
        for (auto const &c : available) {
            auto it = tensors.find(c.non_shared_input);
            if (it == tensors.end() || it->second.is_runtime != operands_runtime) {
                kinds_uniform = false;
                break;
            }
        }
        if (!kinds_uniform)
            continue;

        // Create intermediate tensor T = sum of non-shared operands, of the same
        // kind as the operands. This APPENDS an Alloc node (kept below).
        std::string t_name        = fmt::format("_df_sum_{}", _num_groups);
        auto        create_result = operands_runtime ? graph.create_zero_runtime_tensor_dynamic(t_name, ref_handle.dtype, ref_handle.dims)
                                                     : graph.create_tensor_dynamic(t_name, ref_handle.dtype, ref_handle.dims);
        if (!create_result)
            continue; // Skip this factoring group if tensor creation fails
        auto [t_id, t_ptr] = create_result.value();

        // Emit the factorization as ORDINARY nodes: zero T, accumulate each
        // non-shared operand into it, contract once against the shared operand.
        //
        // An earlier version fused all of that into one OpKind::Custom node whose
        // executor swapped a slot pointer so the first member's baked einsum
        // executor would read T. That worked, but it made the factorization opaque:
        // T's construction was invisible to every other pass, so two consumers of
        // the same sum each built their own copy (CSE has no nodes to match) and a
        // sum that is loop-invariant could not be hoisted out of a CC iteration.
        // Explicit nodes cost nothing at execute and let those passes do their job.
        auto const *first_desc = std::get_if<EinsumDescriptor>(&nodes[first_pos].op_data);
        if (first_desc == nullptr) {
            continue;
        }
        // make_einsum_node wants the parsed form; the descriptor keeps the
        // PackedGemm topology. Only the raw index lists carry over, which is all
        // that changes here: T takes the non-shared operand's place, so the
        // contraction's indices are exactly the ones the members already used.
        ParsedEinsumSpec spec;
        spec.c_indices = first_desc->spec.c_indices;
        spec.a_indices = first_desc->spec.a_indices;
        spec.b_indices = first_desc->spec.b_indices;
        spec.raw =
            fmt::format("{} <- {} ; {}", fmt::join(spec.c_indices, ","), fmt::join(spec.a_indices, ","), fmt::join(spec.b_indices, ","));
        auto const c_pf = first_desc->c_prefactor;

        std::vector<Node> emitted;
        emitted.reserve(available.size() + 2);

        // T = 0. Recorded as a Scale by zero, which is what it means, so the
        // algebraic passes can reason about it; the executor is a true zero-fill
        // rather than a multiply, so a freshly allocated T holding garbage cannot
        // turn into a NaN here.
        {
            Node nd;
            nd.id      = graph.reserve_node_id();
            nd.kind    = OpKind::Scale;
            nd.label   = fmt::format("zero({})", t_name);
            nd.inputs  = {t_id};
            nd.outputs = {t_id};
            nd.op_data = ScaleDescriptor{.factor = 0.0};
            nd.execute = graph.make_zero_executor(t_id);
            emitted.push_back(std::move(nd));
        }

        // T += ab_i * B_i, carrying each member's own product prefactor, so the
        // combined contraction needs no prefactor of its own.
        for (auto const &c : available) {
            auto nit = tensors.find(c.non_shared_input);
            Node nd;
            nd.id      = graph.reserve_node_id();
            nd.kind    = OpKind::Axpy;
            nd.label   = fmt::format("axpy({} -> {}, alpha={})", nit != tensors.end() ? nit->second.name : "?", t_name, c.ab_prefactor);
            nd.inputs  = {c.non_shared_input, t_id};
            nd.outputs = {t_id};
            nd.execute = graph.make_axpy_executor(c.ab_prefactor, c.non_shared_input, t_id);
            emitted.push_back(std::move(nd));
        }

        // One contraction, with T in the position the non-shared operand held, so
        // the spec is unchanged. ab_pf is 1: the prefactors moved into the axpys.
        {
            TensorId const a_id = vg.key.shared_is_first ? vg.key.shared_input_id : t_id;
            TensorId const b_id = vg.key.shared_is_first ? t_id : vg.key.shared_input_id;
            emitted.push_back(graph.make_einsum_node(a_id, b_id, vg.key.output_id, spec, c_pf, PrefactorScalar{double{1}},
                                                     /*conj_a=*/false, /*conj_b=*/false,
                                                     fmt::format("df_factored({} terms via {})", available.size(), t_name)));
        }

        // Splice in at the first member's position so the combined writer stays
        // scan-before every consumer of the output, and drop all the members; see
        // the placement gate above for why a later position is unsound.
        inserts.emplace_back(first_pos, std::move(emitted));
        for (auto const &c : available) {
            remove[c.node_index]    = true;
            node_used[c.node_index] = true;
        }
        _num_eliminated += available.size() - 1;

        // Record the group for reporting
        FactoringGroup fg;
        auto           sh_it = tensors.find(vg.key.shared_input_id);
        fg.shared_tensor     = sh_it != tensors.end() ? sh_it->second.name : "?";
        auto out_it          = tensors.find(vg.key.output_id);
        fg.output_tensor     = out_it != tensors.end() ? out_it->second.name : "?";
        fg.num_terms         = available.size();
        for (auto const &c : available) {
            auto nit = tensors.find(c.non_shared_input);
            fg.summed_tensors.push_back(nit != tensors.end() ? nit->second.name : "?");
        }
        report(2, fmt::format("factor {} contractions into '{}' sharing operand '{}' (sum operands, contract once)", available.size(),
                              out_it != tensors.end() ? out_it->second.name : "?", sh_it != tensors.end() ? sh_it->second.name : "?"));
        _groups.push_back(std::move(fg));
        _num_groups++;
        modified = true;
    }

    if (!modified)
        return false;

    // create_*_tensor_dynamic appended Alloc nodes past orig_count; pad the mask so
    // it covers them (always kept) and so the shift table below is complete.
    remove.resize(nodes.size(), false);

    // Keep the appended Alloc nodes; drop the subsumed originals.
    graph.erase_nodes(remove);

    // Positions were recorded pre-erase; shift each down by the number of erased
    // nodes below it. A group's own first-member slot IS erased, so the shifted
    // position lands exactly where it used to be and the replacements take its
    // place (same accounting as TiledExpansion).
    std::vector<size_t> erased_below(remove.size() + 1, 0);
    for (size_t i = 0; i < remove.size(); i++) {
        erased_below[i + 1] = erased_below[i] + (remove[i] ? 1 : 0);
    }
    for (auto &[position, group] : inserts) {
        position -= erased_below[position];
    }
    graph.insert_node_groups(std::move(inserts));

    // Re-sort since we added/removed nodes
    graph.topological_sort();

    EINSUMS_LOG_INFO("DistributiveFactoring: rewrote {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("factored {} group(s), eliminated {} contraction node(s)", _num_groups, _num_eliminated));
    return true;
}

} // namespace einsums::compute_graph::passes
