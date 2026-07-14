//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <complex>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Does `target` fully overwrite its output without reading its prior
/// contents (c_prefactor / beta == 0)? If so, a Scale of that tensor
/// immediately preceding it (with no intervening reader) is dead: its
/// result is discarded wholesale.
///
/// The target node must NOT be mutated here. CPU einsum executors read
/// their prefactors live from the shared EinsumParams, while GPU dispatch
/// reads the descriptor; editing the descriptor alone desyncs the two
/// backends and makes every later descriptor-reading pass misclassify
/// this node as accumulating.
bool overwrites_without_reading(Node const &target) {
    switch (target.kind) {
    case OpKind::Einsum: {
        auto const *desc = std::get_if<EinsumDescriptor>(&target.op_data);
        return desc != nullptr && is_zero(desc->c_prefactor);
    }
    case OpKind::Permute: {
        auto const *desc = std::get_if<PermuteDescriptor>(&target.op_data);
        return desc != nullptr && desc->beta == 0.0;
    }
    case OpKind::BatchedGemm: {
        auto const *desc = std::get_if<BatchedGemmDescriptor>(&target.op_data);
        return desc != nullptr && desc->beta == std::complex<double>{0.0, 0.0};
    }
    default:
        return false;
    }
}

} // namespace

bool ScaleAbsorption::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        _num_absorbed = 0;
        return false;
    }

    _num_absorbed = 0;
    std::vector<bool> remove(nodes.size(), false);

    for (size_t sc = 0; sc + 1 < nodes.size(); sc++) {
        if (remove[sc])
            continue;

        auto &scale_node = nodes[sc];
        if (scale_node.kind != OpKind::Scale)
            continue;

        auto *scale_desc = std::get_if<ScaleDescriptor>(&scale_node.op_data);
        if (!scale_desc)
            continue;
        if (scale_node.outputs.size() != 1)
            continue;

        TensorId const scaled_tensor = scale_node.outputs[0];
        double         scale_factor  = scale_desc->factor;

        // Look for the next node that writes to the same tensor
        for (size_t tgt = sc + 1; tgt < nodes.size(); tgt++) {
            if (remove[tgt])
                continue;

            auto &target = nodes[tgt];

            bool const writes_scaled = std::ranges::find(target.outputs, scaled_tensor) != target.outputs.end();

            if (!writes_scaled) {
                bool const reads_scaled = std::ranges::find(target.inputs, scaled_tensor) != target.inputs.end();
                if (reads_scaled)
                    break;
                continue;
            }

            // The next writer overwrites the scaled tensor without reading
            // it, so the scale's result is discarded: drop the Scale node.
            if (overwrites_without_reading(target)) {
                remove[sc] = true;
                _num_absorbed++;

                EINSUMS_LOG_INFO("ScaleAbsorption: removed dead scale({}) of a tensor overwritten by {} node {}", scale_factor,
                                 op_kind_name(target.kind), target.id);
                report(2, fmt::format("remove dead scale({}); {} node {} overwrites the tensor without reading it", scale_factor,
                                      op_kind_name(target.kind), target.id));
            }
            break;
        }
    }

    if (_num_absorbed == 0)
        return false;
    report(1, fmt::format("removed {} dead scale(s) whose result the following op overwrites", _num_absorbed));

    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (!remove[idx]) {
            filtered.push_back(std::move(nodes[idx]));
        }
    }
    nodes = std::move(filtered);
    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
