//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <variant>
#include <vector>

namespace einsums::compute_graph::passes {

bool InputSlicing::run(Graph &graph) {
    _num_sliced = 0;

    if (comm::world_size() <= 1)
        return false;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();
    int         rank    = comm::world_rank();

    struct SliceInfo {
        TensorId tid;
        size_t   dim;
        size_t   start;
        size_t   count;
    };
    struct SliceInsertion {
        size_t                 einsum_idx;
        std::vector<SliceInfo> slices;
    };
    std::vector<SliceInsertion> insertions;

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        // Extract output index list and per-input index lists based on node kind
        std::vector<std::string> const               *c_indices_ptr = nullptr;
        std::vector<std::vector<std::string> const *> input_indices;

        if (node.kind == OpKind::Einsum) {
            auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
            if (!desc)
                continue;
            c_indices_ptr = &desc->spec.c_indices;
            input_indices.push_back(&desc->spec.a_indices);
            if (node.inputs.size() > 1)
                input_indices.push_back(&desc->spec.b_indices);
        } else if (node.kind == OpKind::Permute || node.kind == OpKind::Transpose) {
            auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
            if (!desc || desc->c_indices.empty())
                continue;
            c_indices_ptr = &desc->c_indices;
            input_indices.push_back(&desc->a_indices);
        } else {
            // BatchedGemm and other kinds: distributed batched contractions
            // aren't supported (see docs/gemm_batching.rst); the generic
            // fallback below would have no index list to reason about.
            continue;
        }

        // Check each output for a DistributionDescriptor
        for (auto out_tid : node.outputs) {
            auto out_it = tensors.find(out_tid);
            if (out_it == tensors.end())
                continue;

            auto const &out_handle = out_it->second;
            if (!out_handle.is_distributed || out_handle.is_replicated || !out_handle.distribution_info)
                continue;

            auto out_desc = std::static_pointer_cast<comm::DistributionDescriptor>(out_handle.distribution_info);

            SliceInsertion ins;
            ins.einsum_idx = idx;

            // For each distributed dimension of the output, find matching input dimensions
            for (size_t out_d = 0; out_d < out_desc->dim_to_axis.size(); out_d++) {
                if (out_desc->dim_to_axis[out_d] == comm::GridAxis::None)
                    continue;

                if (out_d >= c_indices_ptr->size())
                    continue;
                std::string dist_index = (*c_indices_ptr)[out_d];

                // Compute local range from the descriptor
                auto [start, end]  = out_desc->local_range(out_d, rank);
                size_t const count = end - start;

                // Find matching dimensions in each input
                for (size_t inp_idx = 0; inp_idx < node.inputs.size() && inp_idx < input_indices.size(); inp_idx++) {
                    auto inp_tid = node.inputs[inp_idx];
                    auto inp_it  = tensors.find(inp_tid);
                    if (inp_it == tensors.end())
                        continue;

                    auto const &inp_handle = inp_it->second;

                    // Only slice pre-allocated (materialized, non-distributed) inputs
                    if (inp_handle.is_distributed && !inp_handle.is_replicated)
                        continue;
                    if (inp_handle.alloc_state == AllocState::Deferred)
                        continue;

                    auto const &inp_indices = *input_indices[inp_idx];

                    for (size_t d = 0; d < inp_indices.size(); d++) {
                        if (inp_indices[d] == dist_index) {
                            ins.slices.push_back({.tid = inp_tid, .dim = d, .start = start, .count = count});

                            EINSUMS_LOG_INFO("InputSlicing: slicing '{}' dim {} [{}, {}) for index '{}' (rank {})", inp_handle.name, d,
                                             start, end, dist_index, rank);
                            report(2, fmt::format("slice '{}' dim {} -> [{}, {}) for distributed index '{}' (rank {})", inp_handle.name, d,
                                                  start, end, dist_index, rank));
                            break;
                        }
                    }
                }
            }

            if (!ins.slices.empty()) {
                insertions.push_back(std::move(ins));
            }
        }
    }

    if (insertions.empty())
        return false;

    // Process in reverse order so indices don't shift
    std::ranges::sort(insertions, [](SliceInsertion const &a, SliceInsertion const &b) { return a.einsum_idx > b.einsum_idx; });

    for (auto const &ins : insertions) {
        auto  tokens     = std::make_shared<std::vector<std::pair<TensorId, size_t>>>();
        auto &tensor_map = graph.tensors_map();

        // EndSlice AFTER the node
        Node end_node;
        end_node.kind    = OpKind::Custom;
        end_node.label   = "end_slice";
        end_node.execute = [tokens, &tensor_map]() {
            for (auto const &[tid, token] : *tokens) {
                auto it = tensor_map.find(tid);
                if (it != tensor_map.end() && it->second.end_local_view_fn) {
                    it->second.end_local_view_fn(token);
                }
            }
        };
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.einsum_idx + 1), std::move(end_node));

        // BeginSlice BEFORE the node
        auto slices_copy = ins.slices;
        Node begin_node;
        begin_node.kind    = OpKind::Custom;
        begin_node.label   = "begin_slice";
        begin_node.execute = [slices_copy, tokens, &tensor_map]() {
            tokens->clear();
            for (auto const &s : slices_copy) {
                auto it = tensor_map.find(s.tid);
                if (it != tensor_map.end() && it->second.begin_local_view_fn) {
                    size_t token = it->second.begin_local_view_fn(s.dim, s.start, s.count);
                    tokens->emplace_back(s.tid, token);
                }
            }
        };
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.einsum_idx), std::move(begin_node));
        _num_sliced += ins.slices.size();
    }

    graph.mark_sorted();
    return _num_sliced > 0;
}

} // namespace einsums::compute_graph::passes
