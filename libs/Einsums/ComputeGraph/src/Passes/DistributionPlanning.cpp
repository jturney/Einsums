//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/ProcessGrid.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DistributionPlanning.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace einsums::compute_graph::passes {

DistributionPlanning::DistributionPlanning(size_t threshold, bool enable_summa) : _threshold(threshold), _enable_summa(enable_summa) {
}

namespace {

/// Classify which output-tensor indices come from A only, B only, or both.
/// For C[i,j] = A[i,k] * B[k,j]: target_a = {i}, target_b = {j}, link = {k}.
struct IndexClassification {
    std::vector<std::string> target_a; ///< C indices from A only → grid rows
    std::vector<std::string> target_b; ///< C indices from B only → grid cols
    std::vector<std::string> shared;   ///< C indices in both A and B (batch dims)
    std::vector<std::string> link;     ///< Contraction indices (in A and B, not in C)
};

IndexClassification classify_indices(packed_gemm::ContractionSpec const &spec) {
    IndexClassification result;

    std::unordered_set<std::string> const a_set(spec.a_indices.begin(), spec.a_indices.end());
    std::unordered_set<std::string> const b_set(spec.b_indices.begin(), spec.b_indices.end());

    for (auto const &idx : spec.c_indices) {
        bool const in_a = a_set.count(idx) > 0;
        bool const in_b = b_set.count(idx) > 0;
        if (in_a && !in_b)
            result.target_a.push_back(idx);
        else if (!in_a && in_b)
            result.target_b.push_back(idx);
        else if (in_a && in_b)
            result.shared.push_back(idx);
    }

    result.link = spec.link_indices;
    return result;
}

/// Build a DistributionDescriptor for a tensor given its role in a contraction.
/// @param role "C" (output), "A" (left input), or "B" (right input).
/// @param summa If true, link indices are distributed too (A: link→Col, B: link→Row).
comm::DistributionDescriptor build_descriptor(std::vector<std::string> const &tensor_indices, std::vector<size_t> const &dims,
                                              IndexClassification const &cls, comm::ProcessGrid const *grid, std::string const &role,
                                              bool summa) {
    comm::DistributionDescriptor desc;
    desc.dim_to_axis.resize(tensor_indices.size(), comm::GridAxis::None);
    desc.global_dims = dims;
    desc.grid        = grid;
    desc.summa       = summa;

    std::unordered_set<std::string> const target_a_set(cls.target_a.begin(), cls.target_a.end());
    std::unordered_set<std::string> const target_b_set(cls.target_b.begin(), cls.target_b.end());
    std::unordered_set<std::string> const shared_set(cls.shared.begin(), cls.shared.end());
    std::unordered_set<std::string> const link_set(cls.link.begin(), cls.link.end());

    for (size_t d = 0; d < tensor_indices.size(); d++) {
        auto const &idx = tensor_indices[d];
        if (target_a_set.count(idx) > 0) {
            desc.dim_to_axis[d] = comm::GridAxis::Row;
        } else if (target_b_set.count(idx) > 0) {
            desc.dim_to_axis[d] = comm::GridAxis::Col;
        } else if (shared_set.count(idx) > 0) {
            // Shared (batch) indices appear in all three tensors and are independent.
            // Distribute along whichever axis has fewer dims so far for balance.
            int const row_count = static_cast<int>(std::count(desc.dim_to_axis.begin(), desc.dim_to_axis.end(), comm::GridAxis::Row));
            int const col_count = static_cast<int>(std::count(desc.dim_to_axis.begin(), desc.dim_to_axis.end(), comm::GridAxis::Col));
            desc.dim_to_axis[d] = (row_count <= col_count) ? comm::GridAxis::Row : comm::GridAxis::Col;
        } else if (summa && link_set.count(idx) > 0) {
            // SUMMA: link indices distributed across the grid too.
            if (role == "A") {
                desc.dim_to_axis[d] = comm::GridAxis::Col;
            } else if (role == "B") {
                desc.dim_to_axis[d] = comm::GridAxis::Row;
            }
        }
    }

    return desc;
}

} // namespace

bool DistributionPlanning::run(Graph &graph) {
    _num_distributed = 0;
    _num_replicated  = 0;

    int num_ranks = comm::world_size();

    // Single rank: everything is replicated, nothing to plan.
    if (num_ranks <= 1) {
        for (auto const &[tid, handle] : graph.tensors_map()) {
            if (handle.alloc_state == AllocState::Deferred) {
                _num_replicated++;
            }
        }
        return false;
    }

    auto &grid = comm::ProcessGrid::default_grid();

    // Build a map: TensorId → list of einsum nodes that reference it, and which role (C, A, B)
    struct TensorRole {
        size_t                              node_idx;
        std::string                         role; // "C", "A", "B"
        packed_gemm::ContractionSpec const *spec;
    };
    std::unordered_map<TensorId, std::vector<TensorRole>> tensor_usage;

    // Synthetic ContractionSpecs for Permute nodes (all indices are "shared")
    std::vector<packed_gemm::ContractionSpec> permute_specs;

    auto const &nodes = graph.nodes();
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        // Only classify Einsum nodes. BatchedGemm replaces einsums after
        // GEMMBatching runs; mixing batched and distributed dispatch is
        // documented as unsupported (docs/gemm_batching.rst), so we skip
        // BatchedGemm here rather than inventing a role for it.
        if (node.kind == OpKind::Einsum) {
            auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
            if (!desc)
                continue;
            if (!node.outputs.empty())
                tensor_usage[node.outputs[0]].push_back({.node_idx = idx, .role = "C", .spec = &desc->spec});
            if (node.inputs.size() > 0)
                tensor_usage[node.inputs[0]].push_back({.node_idx = idx, .role = "A", .spec = &desc->spec});
            if (node.inputs.size() > 1)
                tensor_usage[node.inputs[1]].push_back({.node_idx = idx, .role = "B", .spec = &desc->spec});
        } else if (node.kind == OpKind::Permute || node.kind == OpKind::Transpose) {
            auto const *pdesc = std::get_if<PermuteDescriptor>(&node.op_data);
            if (!pdesc || pdesc->c_indices.empty())
                continue;
            // Create a synthetic ContractionSpec treating permute as C[...] = A[...]
            // All indices are "shared" (in both A and C, no B), no link indices.
            packed_gemm::ContractionSpec spec;
            spec.c_indices = pdesc->c_indices;
            spec.a_indices = pdesc->a_indices;
            // b_indices empty → target_b is all C indices not in A, but for permute
            // all C indices ARE in A. So target_a={}, target_b={}, shared=all.
            // But that means nothing gets distributed! Instead, treat A indices
            // as if they're all target_a (since there's no B tensor).
            // Use c_indices as b_indices too so shared = c_indices.
            spec.b_indices = pdesc->c_indices;
            permute_specs.push_back(std::move(spec));
            auto *spec_ptr = &permute_specs.back();

            if (!node.outputs.empty())
                tensor_usage[node.outputs[0]].push_back({.node_idx = idx, .role = "C", .spec = spec_ptr});
            if (!node.inputs.empty())
                tensor_usage[node.inputs[0]].push_back({.node_idx = idx, .role = "A", .spec = spec_ptr});
        }
    }

    // Collect deferred tensor IDs
    std::vector<TensorId> deferred_tids;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.alloc_state == AllocState::Deferred) {
            deferred_tids.push_back(tid);
        }
    }

    // Sort: process tensors that appear as outputs first (producers before consumers).
    // This ensures chain intermediates get their distribution set when they're outputs,
    // and consumers inherit that distribution via constraint propagation.
    std::ranges::sort(deferred_tids, [&nodes](TensorId a, TensorId b) {
        auto first_output = [&](TensorId tid) -> size_t {
            for (size_t idx = 0; idx < nodes.size(); idx++)
                for (auto out : nodes[idx].outputs)
                    if (out == tid)
                        return idx;
            return SIZE_MAX;
        };
        return first_output(a) < first_output(b);
    });

    bool modified = false;

    for (auto tid : deferred_tids) {
        auto &handle = graph.tensor(tid);

        // Skip if already distributed (set when processed as output of a prior einsum in this loop)
        if (handle.is_distributed && handle.distribution_info) {
            _num_distributed++;
            continue;
        }

        size_t total_bytes = handle.total_bytes();

        if (total_bytes <= _threshold) {
            handle.is_distributed = false;
            handle.is_replicated  = true;
            _num_replicated++;
            EINSUMS_LOG_DEBUG("DistributionPlanning: '{}' ({} bytes) → replicated", handle.name, total_bytes);
            continue;
        }

        // Find the first einsum that uses this tensor and build a descriptor
        auto it = tensor_usage.find(tid);
        if (it == tensor_usage.end()) {
            handle.is_distributed = false;
            handle.is_replicated  = true;
            _num_replicated++;
            continue;
        }

        // Use the first einsum's index classification
        auto const &usage = it->second[0];
        auto        cls   = classify_indices(*usage.spec);

        // Determine which indices this tensor carries
        std::vector<std::string> const *tensor_indices = nullptr;
        if (usage.role == "C")
            tensor_indices = &usage.spec->c_indices;
        else if (usage.role == "A")
            tensor_indices = &usage.spec->a_indices;
        else
            tensor_indices = &usage.spec->b_indices;

        // Use SUMMA when enabled and we have a true 2D square grid
        bool const use_summa = _enable_summa && (grid.rows() > 1 && grid.cols() > 1 && grid.rows() == grid.cols());
        auto       desc      = build_descriptor(*tensor_indices, handle.dims, cls, &grid, usage.role, use_summa);

        // Conflict resolution: check ALL usages of this tensor. If a dimension's
        // index is a link (contraction) index in ANY other einsum, it must NOT be
        // distributed — distributing a link dim produces partial sums that need
        // SUMMA-style communication. For the outer-product strategy, downgrade to None.
        for (size_t u = 1; u < it->second.size(); u++) {
            auto const                           &other_usage = it->second[u];
            auto                                  other_cls   = classify_indices(*other_usage.spec);
            std::unordered_set<std::string> const other_link(other_cls.link.begin(), other_cls.link.end());

            // Get this tensor's indices in the other einsum
            std::vector<std::string> const *other_tensor_indices = nullptr;
            if (other_usage.role == "C")
                other_tensor_indices = &other_usage.spec->c_indices;
            else if (other_usage.role == "A")
                other_tensor_indices = &other_usage.spec->a_indices;
            else
                other_tensor_indices = &other_usage.spec->b_indices;

            // For each dimension, check if its index is a link in the other einsum
            for (size_t d = 0; d < tensor_indices->size() && d < other_tensor_indices->size(); d++) {
                if (desc.dim_to_axis[d] == comm::GridAxis::None)
                    continue;
                // The tensor's dim d has index name tensor_indices[d].
                // In the other einsum, the same dim d has index name other_tensor_indices[d].
                // If that index is a link index there, we can't distribute this dim.
                if (other_link.count((*other_tensor_indices)[d]) > 0) {
                    EINSUMS_LOG_INFO("DistributionPlanning: '{}' dim {} ({}) conflicts — link index '{}' in another einsum, downgrading to "
                                     "None",
                                     handle.name, d, (*tensor_indices)[d], (*other_tensor_indices)[d]);
                    desc.dim_to_axis[d] = comm::GridAxis::None;
                }
            }
        }

        // Constraint propagation: if any input to this einsum is already distributed,
        // inherit its axis assignment for matching indices. This ensures chain
        // intermediates maintain consistent distributions.
        auto const &node = nodes[usage.node_idx];
        for (size_t inp_idx = 0; inp_idx < node.inputs.size(); inp_idx++) {
            auto inp_it = graph.tensors_map().find(node.inputs[inp_idx]);
            if (inp_it == graph.tensors_map().end())
                continue;
            auto const &inp_handle = inp_it->second;
            if (!inp_handle.is_distributed || !inp_handle.distribution_info)
                continue;

            auto        inp_desc    = std::static_pointer_cast<comm::DistributionDescriptor>(inp_handle.distribution_info);
            auto const &inp_indices = (inp_idx == 0) ? usage.spec->a_indices : usage.spec->b_indices;

            for (size_t id = 0; id < inp_desc->dim_to_axis.size() && id < inp_indices.size(); id++) {
                if (inp_desc->dim_to_axis[id] == comm::GridAxis::None)
                    continue;
                // Find this index in the current tensor's indices
                for (size_t td = 0; td < tensor_indices->size(); td++) {
                    if ((*tensor_indices)[td] == inp_indices[id]) {
                        desc.dim_to_axis[td] = inp_desc->dim_to_axis[id];
                    }
                }
            }
        }

        // Check if any dimension is actually distributed
        if (desc.is_fully_replicated()) {
            // No target indices matched → replicate
            handle.is_distributed = false;
            handle.is_replicated  = true;
            _num_replicated++;
            EINSUMS_LOG_DEBUG("DistributionPlanning: '{}' → replicated (no distributable indices)", handle.name);
            continue;
        }

        handle.is_distributed    = true;
        handle.is_replicated     = false;
        handle.distribution_info = std::make_shared<comm::DistributionDescriptor>(std::move(desc));
        _num_distributed++;
        modified = true;

        // Log the assignment
        std::string axis_str;
        auto const &stored = std::static_pointer_cast<comm::DistributionDescriptor>(handle.distribution_info);
        for (size_t d = 0; d < stored->dim_to_axis.size(); d++) {
            if (d > 0)
                axis_str += ",";
            switch (stored->dim_to_axis[d]) {
            case comm::GridAxis::None:
                axis_str += "None";
                break;
            case comm::GridAxis::Row:
                axis_str += "Row";
                break;
            case comm::GridAxis::Col:
                axis_str += "Col";
                break;
            }
        }
        EINSUMS_LOG_INFO("DistributionPlanning: '{}' ({} bytes) → [{}] on {}x{} grid", handle.name, total_bytes, axis_str, grid.rows(),
                         grid.cols());
    }

    if (_num_distributed > 0 || _num_replicated > 0) {
        EINSUMS_LOG_INFO("DistributionPlanning: {} distributed, {} replicated ({}x{} grid, {} ranks)", _num_distributed, _num_replicated,
                         grid.rows(), grid.cols(), num_ranks);
    }

    return modified;
}

} // namespace einsums::compute_graph::passes
