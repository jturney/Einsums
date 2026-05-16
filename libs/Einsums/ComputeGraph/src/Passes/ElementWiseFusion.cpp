//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp>
#include <Einsums/Logging.hpp>

#include <vector>

namespace einsums::compute_graph::passes {

bool ElementWiseFusion::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        _num_fused = 0;
        return false;
    }

    _num_fused = 0;
    std::vector<bool> remove(nodes.size(), false);

    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        if (remove[i])
            continue;

        // Look for consecutive Scale ops on the same tensor
        if (nodes[i].kind != OpKind::Scale)
            continue;
        if (nodes[i].outputs.size() != 1)
            continue;

        auto *desc_i = std::get_if<ScaleDescriptor>(&nodes[i].op_data);
        if (!desc_i)
            continue;

        TensorId const target = nodes[i].outputs[0];

        // Scan forward for more Scale ops on the same tensor
        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (remove[j])
                continue;

            // If this node reads the target but isn't a scale on it, stop
            if (nodes[j].kind != OpKind::Scale)
                break;
            if (nodes[j].outputs.size() != 1 || nodes[j].outputs[0] != target)
                break;

            auto *desc_j = std::get_if<ScaleDescriptor>(&nodes[j].op_data);
            if (!desc_j)
                break;

            // Fuse: multiply factors, compose executors
            desc_i->factor *= desc_j->factor;

            auto exec_i      = std::move(nodes[i].execute);
            auto exec_j      = std::move(nodes[j].execute);
            nodes[i].execute = [exec_i = std::move(exec_i), exec_j = std::move(exec_j)]() {
                exec_i();
                exec_j();
            };

            nodes[i].label = fmt::format("scale({}) [fused]", desc_i->factor);

            remove[j] = true;
            _num_fused++;

            EINSUMS_LOG_INFO("ElementWiseFusion: merged scale({}) into scale({})", desc_j->factor, desc_i->factor);
        }
    }

    if (_num_fused == 0)
        return false;

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
