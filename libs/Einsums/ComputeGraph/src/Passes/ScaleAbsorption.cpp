//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Try to absorb a scale factor into a node's prefactor.
/// Returns true if absorbed (the scale node can be removed).
bool try_absorb(double scale_factor, Node &target) {
    switch (target.kind) {
    case OpKind::Einsum: {
        auto *desc = std::get_if<EinsumDescriptor>(&target.op_data);
        if (desc && is_zero(desc->c_prefactor)) {
            // Preserve the descriptor's existing dtype alternative — the
            // executor lambda extracts via as<T> and a type mismatch here
            // would surface as a runtime conversion at execute time.
            desc->c_prefactor = std::visit(
                [&](auto x) -> PrefactorScalar {
                    using U = decltype(x);
                    if constexpr (std::is_arithmetic_v<U>) {
                        return static_cast<U>(scale_factor);
                    } else {
                        return U{static_cast<typename U::value_type>(scale_factor), typename U::value_type{0}};
                    }
                },
                desc->c_prefactor);
            return true;
        }
        return false;
    }
    case OpKind::Permute: {
        auto *desc = std::get_if<PermuteDescriptor>(&target.op_data);
        if (desc && desc->beta == 0.0) {
            desc->beta = scale_factor;
            return true;
        }
        return false;
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

            // Try to absorb
            if (try_absorb(scale_factor, target)) {
                // Create composite executor
                auto scale_exec  = std::move(scale_node.execute);
                auto target_exec = std::move(target.execute);
                target.execute   = [scale_exec = std::move(scale_exec), target_exec = std::move(target_exec)]() {
                    scale_exec();
                    target_exec();
                };

                target.label = fmt::format("[absorbed scale={}] {}", scale_factor, target.label);
                remove[sc]   = true;
                _num_absorbed++;

                EINSUMS_LOG_INFO("ScaleAbsorption: absorbed scale({}) into {} node {}", scale_factor, op_kind_name(target.kind), target.id);
                report(2, fmt::format("absorb scale({}) into {} node {} (drops the standalone Scale)", scale_factor,
                                      op_kind_name(target.kind), target.id));
            }
            break;
        }
    }

    if (_num_absorbed == 0)
        return false;
    report(1, fmt::format("absorbed {} scale(s) into a following op", _num_absorbed));

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
