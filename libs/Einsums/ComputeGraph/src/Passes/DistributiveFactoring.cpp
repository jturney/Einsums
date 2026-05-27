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
    graph.topological_sort();

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    _num_groups     = 0;
    _num_eliminated = 0;
    _groups.clear();

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
        if (is_zero(desc->c_prefactor))
            continue;
        if (node.inputs.size() != 2 || node.outputs.size() != 1)
            continue;

        TensorId const out_id = node.outputs[0];
        TensorId const in_a   = node.inputs[0];
        TensorId const in_b   = node.inputs[1];
        auto const    &spec   = desc->spec;

        // TODO: when ab_prefactor is complex with non-zero imag, the
        // factoring math below loses the imaginary part. Skip the pass for
        // complex prefactors so we never silently miscompute.
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

        valid_groups.push_back({.key = key, .candidates = std::move(candidates)});
    }

    if (valid_groups.empty()) {
        return false;
    }

    // --- Phase 3: Deduplicate (largest group first, greedy) ---
    std::ranges::sort(valid_groups, [](ValidGroup const &a, ValidGroup const &b) { return a.candidates.size() > b.candidates.size(); });

    std::vector<bool> node_used(nodes.size(), false);
    std::vector<bool> remove(nodes.size(), false);
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

        // --- Phase 4: Rewrite the graph ---

        // Get the reference tensor handle for the non-shared operands
        auto ref_it = tensors.find(available[0].non_shared_input);
        if (ref_it == tensors.end())
            continue;
        auto const &ref_handle = ref_it->second;

        // Create intermediate tensor T = sum of non-shared operands
        std::string t_name        = fmt::format("_df_sum_{}", _num_groups);
        auto        create_result = graph.create_tensor_dynamic(t_name, ref_handle.dtype, ref_handle.dims);
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

            Node combined_node;
            combined_node.kind    = OpKind::Custom;
            combined_node.label   = fmt::format("df_factored({} terms via {})", available.size(), t_name);
            combined_node.execute = std::move(combined_executor);
            combined_node.inputs  = std::move(all_inputs);
            combined_node.outputs = {vg.key.output_id, t_id};
            graph.add_node(std::move(combined_node));
        }

        // Mark ALL original einsum nodes for removal (replaced by the combined node)
        for (auto const &c : available) {
            remove[c.node_index] = true;
        }
        _num_eliminated += available.size();

        // Mark all nodes as used
        for (auto const &c : available) {
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

    // Remove eliminated nodes
    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!remove[i]) {
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
