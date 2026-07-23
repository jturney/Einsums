//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp>
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

} // namespace

bool DistributiveFactoring::run(Graph &graph) {
    // Own the recursion (like LoopInvariantHoisting): reset the counters once
    // here at the root, then descend ourselves. If we instead opted into
    // PassManager auto-recursion, run() would be re-invoked per subgraph and
    // reset the top-level tally each time. recurse_into_subgraphs()
    // returns false so the PassManager does not double-walk.
    _num_groups     = 0;
    _num_eliminated = 0;
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
            bool interference = false;
            for (size_t n = first_pos + 1; n < last_pos && !interference; n++) {
                if (is_member[n]) {
                    continue;
                }
                Node const &other = nodes[n];
                if (other.kind == OpKind::Loop || other.kind == OpKind::Conditional) {
                    interference = true;
                    break;
                }
                for (auto const out : other.outputs) {
                    if (out == vg.key.output_id || operand_ids.count(out) != 0) { // clobbers the sum or a factor
                        interference = true;
                        break;
                    }
                }
                if (interference) {
                    break;
                }
                for (auto const in : other.inputs) {
                    if (in == vg.key.output_id) { // observes the partial sum
                        interference = true;
                        break;
                    }
                }
            }
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

        // Build a single Custom node that:
        // 1. Zeros T
        // 2. Axpy each non-shared operand into T: T = sum(alpha_i * Bi)
        // 3. Runs the original einsum with T substituted for the non-shared input
        //
        // We capture the first einsum's original executor and the original non-shared
        // input's slot. By updating the slot to point to T before calling the executor,
        // the einsum reads from T.
        Node combined_node;
        combined_node.kind    = OpKind::Custom;
        combined_node.label   = fmt::format("df_factored({} terms via {})", available.size(), t_name);
        combined_node.outputs = {vg.key.output_id, t_id};
        {
            auto zero_exec = graph.make_zero_executor(t_id);

            // Build axpy executors for each non-shared operand.
            // Scale by (c.ab_prefactor / first_ab_prefactor) so the original einsum's
            // baked-in ab_prefactor produces the correct combined result.
            double first_ab = available[0].ab_prefactor;
            if (first_ab == 0.0)
                first_ab = 1.0; // Avoid division by zero

            std::vector<std::function<void()>> axpy_execs;
            for (auto const &c : available) {
                double const scale = c.ab_prefactor / first_ab;
                axpy_execs.push_back(graph.make_axpy_executor(scale, c.non_shared_input, t_id));
            }

            // Capture the first einsum's executor and its slot
            auto           original_executor = nodes[available[0].node_index].execute;
            TensorId const non_shared_id     = available[0].non_shared_input;
            bool const     shared_is_first   = vg.key.shared_is_first;

            Graph *graph_ptr = &graph;
            // NOLINTNEXTLINE
            auto combined_executor = [zero_exec, axpy_execs = std::move(axpy_execs), original_executor, graph_ptr, t_id, non_shared_id,
                                      shared_is_first]() {
                // 1. Zero T
                zero_exec();

                // 2. Sum all non-shared operands into T
                for (auto const &axpy : axpy_execs) {
                    axpy();
                }

                // 3. Redirect the slot so the original einsum reads from T
                auto *slot         = graph_ptr->find_slot(non_shared_id);
                void *original_ptr = nullptr;
                if (slot) {
                    original_ptr = slot->ptr;
                    slot->ptr    = graph_ptr->tensor(t_id).tensor_ptr;
                }

                // 4. Run the original einsum (now reads T instead of B1)
                original_executor();

                // 5. Restore the slot (so other operations that read B1 still work)
                if (slot) {
                    slot->ptr = original_ptr;
                }
            };

            // Collect all input TensorIds for dependency tracking
            std::vector<TensorId> all_inputs;
            all_inputs.push_back(vg.key.shared_input_id);
            for (auto const &c : available) {
                all_inputs.push_back(c.non_shared_input);
            }

            combined_node.execute = std::move(combined_executor);
            combined_node.inputs  = std::move(all_inputs);
        }

        // Occupy the first member's slot (reusing its id) instead of appending, so
        // the combined writer stays scan-before every consumer of the output; see
        // the placement gate above for why appending is unsound.
        combined_node.id     = nodes[first_pos].id;
        nodes[first_pos]     = std::move(combined_node);
        node_used[first_pos] = true;

        // The remaining members are subsumed by the combined node; drop them.
        for (auto const &c : available) {
            if (c.node_index != first_pos) {
                remove[c.node_index] = true;
                _num_eliminated++;
            }
            node_used[c.node_index] = true;
        }

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

    // Keep appended Alloc nodes (index >= orig_count); drop the subsumed originals.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i >= orig_count || !remove[i]) {
            filtered.push_back(std::move(nodes[i]));
        }
    }
    nodes = std::move(filtered);

    // Re-sort since we added/removed nodes
    graph.topological_sort();

    EINSUMS_LOG_INFO("DistributiveFactoring: rewrote {} groups, eliminated {} nodes", _num_groups, _num_eliminated);
    report(1, fmt::format("factored {} group(s), eliminated {} contraction node(s)", _num_groups, _num_eliminated));
    return true;
}

} // namespace einsums::compute_graph::passes
