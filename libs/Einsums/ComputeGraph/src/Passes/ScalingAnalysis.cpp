//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ScalingAnalysis.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "LetterBindings.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Merge the bindings re-derived from the operands' current annotations with the map frozen into
/// the node at capture. The derived side wins wherever the two disagree: a handle's annotation is
/// what the graph says now, while the captured map is what it said then. The captured side still
/// contributes the letters the derivation could not reach, which is how a node a pass rebuilt
/// without operand annotations keeps whatever its author knew.
std::vector<std::pair<std::string, SpaceId>> effective_letter_spaces(Graph const &graph, Node const &node, EinsumDescriptor const &desc) {
    auto merged = detail::agreed_letter_spaces(detail::letter_bindings(graph, node, desc));
    for (auto const &captured : desc.letter_spaces) {
        auto const found = std::ranges::find_if(merged, [&captured](auto const &entry) { return entry.first == captured.first; });
        if (found == merged.end()) {
            merged.push_back(captured);
        }
    }
    std::ranges::sort(merged, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
    return merged;
}

/// Whether a polynomial mentions a letter no annotation reached.
bool has_anonymous_variable(SymbolicPoly const &poly) {
    auto const variables = poly.variables();
    return std::ranges::any_of(variables, [](SymbolicVar const &variable) { return variable.is_anonymous(); });
}

/// Render one polynomial, tolerating a null registry.
std::string render(SymbolicPoly const &poly, SpaceRegistry const *registry) {
    return poly.to_string(registry);
}

} // namespace

void ScalingAnalysis::reset_stats() {
    _node_costs.clear();
    _intermediate_sizes.clear();
    _rate_limiting.clear();
    _total_flops           = SymbolicPoly::zero();
    _total_traffic         = SymbolicPoly::zero();
    _memory_bound          = SymbolicPoly::zero();
    _num_unannotated_nodes = 0;
    _registry              = nullptr;
}

void ScalingAnalysis::rank_nodes(SpaceRegistry const *registry) {
    _rate_limiting.clear();
    if (_node_costs.empty()) {
        return;
    }

    ComparisonContext const ctx{.registry = registry};

    // The comparison is total and returns `equal` only for identical canonical forms, so the set
    // of maxima is exactly the set of nodes sharing one polynomial. That keeps the verdict a
    // statement about the program rather than about container order.
    NodeCost const *best = &_node_costs.front();
    for (auto const &candidate : _node_costs) {
        if (compare(candidate.cost.flops, best->cost.flops, ctx) > 0) {
            best = &candidate;
        }
    }
    for (auto const &candidate : _node_costs) {
        if (candidate.cost.flops == best->cost.flops) {
            _rate_limiting.push_back(candidate);
        }
    }
}

bool ScalingAnalysis::run(Graph &graph) {
    // Per-apply state: the recursive driver calls run() once per subgraph and reset_stats() once
    // per apply(), so everything below accumulates across levels rather than starting over.
    std::size_t const analyzed_at_entry = _node_costs.size();

    auto const *registry = &graph.space_registry();
    _registry            = registry;

    // Intermediates are recorded once per graph: two nodes writing one tensor must not size it
    // twice, and a tensor id is only unique within its own graph.
    std::vector<TensorId> sized;

    for (auto const &node : graph.nodes()) {
        if (node.kind != OpKind::Einsum) {
            note_skip("not a contraction node", fmt::format("node {} is a {}", node.id, op_kind_name(node.kind)));
            continue;
        }
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            note_skip("contraction node carries no descriptor", fmt::format("node {} ('{}')", node.id, node.label));
            continue;
        }

        EinsumDescriptor effective;
        effective.spec          = desc->spec;
        effective.letter_spaces = effective_letter_spaces(graph, node, *desc);

        SymbolicCost const cost = symbolic_cost_for(effective);
        _node_costs.push_back(NodeCost{.graph_name = graph.name(), .node_id = node.id, .label = node.label, .cost = cost});
        _total_flops += cost.flops;
        _total_traffic += cost.traffic;
        if (has_anonymous_variable(cost.flops)) {
            ++_num_unannotated_nodes;
        }

        if (node.outputs.empty()) {
            continue;
        }
        TensorId const out    = node.outputs.front();
        auto const    *handle = graph.find_tensor(out);
        if (handle == nullptr || !handle->is_intermediate) {
            continue;
        }
        if (std::ranges::find(sized, out) != sized.end()) {
            note_skip("intermediate already sized by an earlier writer", fmt::format("tensor '{}'", handle->name));
            continue;
        }
        sized.push_back(out);

        LetterVars const   vars = letter_vars_for(effective);
        SymbolicPoly const size = symbolic_size_for(effective.spec.c_indices, vars);
        _intermediate_sizes.push_back(IntermediateSize{.graph_name = graph.name(), .tensor_id = out, .name = handle->name, .size = size});
        _memory_bound += size;
    }

    rank_nodes(registry);

    if (_node_costs.size() > analyzed_at_entry) {
        EINSUMS_LOG_INFO("ScalingAnalysis: total flops {}", render(_total_flops, registry));
        report(1, fmt::format("costed {} contraction(s), total flops {}", _node_costs.size(), render(_total_flops, registry)));
    }

    // Analysis pass, never changes the node list.
    return false;
}

std::vector<std::string> ScalingAnalysis::explain() const {
    if (_node_costs.empty()) {
        return {};
    }

    std::vector<std::string> lines;
    lines.push_back(fmt::format("ScalingAnalysis: {} contraction(s), total flops {}", _node_costs.size(), render(_total_flops, _registry)));

    if (!_rate_limiting.empty()) {
        std::string names;
        for (std::size_t i = 0; i < _rate_limiting.size(); ++i) {
            if (i != 0) {
                names += ", ";
            }
            names += fmt::format("'{}' (#{})", _rate_limiting[i].label, _rate_limiting[i].node_id);
        }
        lines.push_back(
            fmt::format("ScalingAnalysis: rate-limiting node(s) {} at {}", names, render(_rate_limiting.front().cost.flops, _registry)));
    }

    if (!_intermediate_sizes.empty()) {
        lines.push_back(fmt::format("ScalingAnalysis: intermediate footprint at most {} elements over {} intermediate(s), "
                                    "a sum of sizes rather than a liveness-aware high-water mark",
                                    render(_memory_bound, _registry), _intermediate_sizes.size()));
    }

    if (_num_unannotated_nodes != 0) {
        lines.push_back(fmt::format("ScalingAnalysis: {} of {} contraction(s) carry letters no index-space annotation reaches, "
                                    "so those terms are named after their letters and cannot be ranked by scale order",
                                    _num_unannotated_nodes, _node_costs.size()));
    }

    return lines;
}

void ScalingAnalysis::print_report(std::ostream &os) const {
    os << "=== ScalingAnalysis ===\n";

    if (_node_costs.empty()) {
        os << "  no contraction nodes analysed\n";
        return;
    }

    os << fmt::format("  contractions costed: {}\n", _node_costs.size());
    os << "  per-node cost (flops | traffic | resident):\n";
    for (auto const &entry : _node_costs) {
        os << fmt::format("    [{}#{}] {}\n", entry.graph_name, entry.node_id, entry.label);
        os << fmt::format("      flops    {}\n", render(entry.cost.flops, _registry));
        os << fmt::format("      traffic  {}\n", render(entry.cost.traffic, _registry));
        os << fmt::format("      resident {}\n", render(entry.cost.resident, _registry));
    }

    if (!_intermediate_sizes.empty()) {
        os << "  intermediate sizes (elements):\n";
        for (auto const &entry : _intermediate_sizes) {
            os << fmt::format("    [{}#{}] {}: {}\n", entry.graph_name, entry.tensor_id, entry.name, render(entry.size, _registry));
        }
    }

    os << fmt::format("  total flops:   {}\n", render(_total_flops, _registry));
    os << fmt::format("  total traffic: {}\n", render(_total_traffic, _registry));
    os << fmt::format("  memory bound:  {} (sum of intermediate sizes, an upper bound on the high-water mark)\n",
                      render(_memory_bound, _registry));

    for (auto const &entry : _rate_limiting) {
        os << fmt::format("  rate-limiting: [{}#{}] {} at {}\n", entry.graph_name, entry.node_id, entry.label,
                          render(entry.cost.flops, _registry));
    }

    if (_num_unannotated_nodes != 0) {
        os << fmt::format("  {} contraction(s) carry unannotated letters; annotate tensor slots for a stronger report\n",
                          _num_unannotated_nodes);
    }
}

std::string ScalingAnalysis::report_string() const {
    std::ostringstream out;
    print_report(out);
    return std::move(out).str();
}

std::string ScalingAnalysis::total_flops_str() const {
    return render(_total_flops, _registry);
}

std::string ScalingAnalysis::total_traffic_str() const {
    return render(_total_traffic, _registry);
}

std::string ScalingAnalysis::memory_bound_str() const {
    return render(_memory_bound, _registry);
}

std::vector<std::string> ScalingAnalysis::node_labels() const {
    std::vector<std::string> out;
    out.reserve(_node_costs.size());
    for (auto const &entry : _node_costs) {
        out.push_back(entry.label);
    }
    return out;
}

std::vector<std::string> ScalingAnalysis::node_flops() const {
    std::vector<std::string> out;
    out.reserve(_node_costs.size());
    for (auto const &entry : _node_costs) {
        out.push_back(render(entry.cost.flops, _registry));
    }
    return out;
}

std::vector<std::string> ScalingAnalysis::rate_limiting_labels() const {
    std::vector<std::string> out;
    out.reserve(_rate_limiting.size());
    for (auto const &entry : _rate_limiting) {
        out.push_back(entry.label);
    }
    return out;
}

std::vector<std::string> ScalingAnalysis::intermediate_names() const {
    std::vector<std::string> out;
    out.reserve(_intermediate_sizes.size());
    for (auto const &entry : _intermediate_sizes) {
        out.push_back(entry.name);
    }
    return out;
}

std::vector<std::string> ScalingAnalysis::intermediate_sizes_str() const {
    std::vector<std::string> out;
    out.reserve(_intermediate_sizes.size());
    for (auto const &entry : _intermediate_sizes) {
        out.push_back(render(entry.size, _registry));
    }
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
