//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <string_view>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

void RegionRewrite::reset_stats() {
    _dumps.clear();
    _decline_reasons.clear();
    _cost_mismatches.clear();
    _regions_formed    = 0;
    _regions_rewritten = 0;
    _regions_declined  = 0;
}

namespace {

/// The flops of every node in @p nodes that carries a contraction, through the SAME formula the
/// raise gives a term. Nothing else in the algebra is priced, so nothing else is priced here.
SymbolicPoly node_flops(std::vector<Node> const &nodes, auto &&include) {
    SymbolicPoly out;
    for (auto const &node : nodes) {
        if (!include(node)) {
            continue;
        }
        if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data); desc != nullptr) {
            out += symbolic_cost_for(*desc).flops;
        }
    }
    return out;
}

} // namespace

void RegionRewrite::decline(std::string_view reason, std::string_view detail) {
    ++_regions_declined;
    note_skip(reason, detail);
    auto const hit = std::ranges::find_if(_decline_reasons, [&reason](auto const &e) { return e.first == reason; });
    if (hit == _decline_reasons.end()) {
        _decline_reasons.emplace_back(std::string(reason), 1);
    } else {
        ++hit->second;
    }
}

std::vector<std::string> RegionRewrite::explain() const {
    std::vector<std::string> out;

    // Formed, rewritten, declined - reported together, because "formed twelve regions and
    // rewrote none" and "formed none" are different findings and only the first one means the
    // region rule is doing its job on this graph.
    if (_regions_formed != 0) {
        out.push_back(fmt::format("{}: formed {} region(s), rewrote {}", name(), _regions_formed, _regions_rewritten));
    }
    for (auto const &[reason, count] : _decline_reasons) {
        out.push_back(fmt::format("{}: declined {} region(s): {}", name(), count, reason));
    }

    // What each accepted rewrite cost, before and after. Printed at every verbosity because it
    // is one line and it answers the question a rewrite exists to answer; the algebra behind it
    // is what the dump is for.
    for (auto const &dump : _dumps) {
        if (!dump.changed || dump.cost_before.empty()) {
            continue;
        }
        out.push_back(fmt::format("{}: region {} ({} node(s)) cost {} -> {}", name(), dump.region_index, dump.node_count, dump.cost_before,
                                  dump.cost_after));
        if (!dump.before.empty()) {
            out.push_back(fmt::format("      before: {}\n      after:  {}", dump.before, dump.after));
        }
    }

    // A disagreement between the two derivations is a finding rather than a diagnostic, so it is
    // printed wherever the cost line it contradicts is.
    for (auto const &line : _cost_mismatches) {
        out.push_back(fmt::format("{}: {}", name(), line));
    }

    for (auto &line : describe()) {
        out.push_back(std::move(line));
    }
    return out;
}

std::string RegionRewrite::dump_text() const {
    std::string out;
    for (auto const &dump : _dumps) {
        out += fmt::format("region {} ({} node(s)){}\n  before:\n{}  after:\n{}", dump.region_index, dump.node_count,
                           dump.changed ? "" : " [unchanged]", dump.before, dump.after);
    }
    return out;
}

bool RegionRewrite::applicable(Graph const & /*graph*/) const {
    return true;
}

bool RegionRewrite::run(Graph &graph) {
    bool const dump = _dump || config::get(option::GraphDumpRegions);

    // Before anything expensive. See the header: a pass in the default pipeline runs on every
    // graph anyone optimizes, and most of them have nothing for it.
    if (!applicable(graph)) {
        note_skip("the graph holds nothing this pass acts on");
        return false;
    }

    auto const    escapes = EscapeAnalysis::over(graph);
    RegionOptions options;
    options.min_nodes  = min_region_nodes();
    auto const regions = form_regions(graph, escapes, options);
    _regions_formed    = regions.size();
    if (regions.empty()) {
        note_skip("the graph holds no run of raisable nodes");
        return false;
    }

    SpaceRegistry const *registry = &graph.space_registry();

    // Rewritten back to front. A lowering splices over a region's positions, and
    // every earlier region's positions are unaffected by a splice that happens
    // after them; going forwards would invalidate every later region's recorded
    // positions on the first rewrite, and re-forming after each one would make
    // the pass quadratic in a graph with many regions for no benefit.
    bool modified = false;
    for (std::size_t index = regions.size(); index-- > 0;) {
        auto const &region = regions[index];

        auto raised = raise_region(graph, region);
        if (!raised) {
            decline(raised.error().reason, raised.error().detail);
            continue;
        }

        RegionDump record;
        record.region_index = index;
        record.node_count   = region.size();
        if (dump) {
            record.before = raised->to_string(registry);
        }

        TensorExpr rewritten = *raised;
        bool const changed   = rewrite(graph, region, rewritten);
        record.changed       = changed;
        if (dump) {
            record.after = changed ? rewritten.to_string(registry) : record.before;
            report(2, fmt::format("region {} ({} nodes)\n  before:\n{}  after:\n{}", index, region.size(), record.before, record.after));
        }
        SymbolicPoly expr_before;
        SymbolicPoly expr_after;
        if (changed) {
            // Only for an accepted rewrite: summing a polynomial per term is cheap, and doing
            // it for regions nobody touched would be paid on every graph for nothing.
            expr_before        = raised->total_cost().flops;
            expr_after         = rewritten.total_cost().flops;
            record.cost_before = expr_before.to_string(registry);
            record.cost_after  = expr_after.to_string(registry);
        }
        // Recorded when dumping OR when something changed, so the structural report has a line
        // per rewrite without the option being on; an untouched region with dumping off is not
        // worth a record.
        if (dump || changed) {
            _dumps.push_back(record);
        }
        if (!changed) {
            continue;
        }

        // The node-side derivation of the BEFORE number, taken while the region's own nodes are
        // still in the graph. Compared rather than assumed: it is what says the raise reached
        // every statement and that `total_cost`'s reachability pruning kept the live ones.
        SymbolicPoly               nodes_before;
        std::unordered_set<NodeId> before_ids;
        if (_verify_costs) {
            auto const &all  = graph.nodes();
            auto const  span = [&](Node const &node) { return std::ranges::find(region.nodes, node.id) != region.nodes.end(); };
            nodes_before     = node_flops(all, span);
            for (auto const &node : all) {
                before_ids.insert(node.id);
            }
        }

        if (auto lowered = lower_region(graph, region, rewritten); !lowered) {
            // The graph is untouched: lower_region builds every node before it
            // erases anything, so a refusal costs the rewrite and nothing else.
            _dumps.pop_back();
            decline(lowered.error().reason, lowered.error().detail);
            continue;
        }

        if (_verify_costs) {
            // The emitted nodes are the ones whose id the graph did not hold before. Keyed by id
            // rather than by position, because lowering sorts the graph and the splice does not
            // stay where it was put.
            SymbolicPoly const nodes_after = node_flops(graph.nodes(), [&](Node const &node) { return !before_ids.contains(node.id); });
            if (!(nodes_before == expr_before)) {
                _cost_mismatches.push_back(fmt::format("region {} reports a before-cost of {} and its nodes cost {}", index,
                                                       expr_before.to_string(registry), nodes_before.to_string(registry)));
            }
            if (!(nodes_after == expr_after)) {
                _cost_mismatches.push_back(fmt::format("region {} reports an after-cost of {} and the nodes it emitted cost {}", index,
                                                       expr_after.to_string(registry), nodes_after.to_string(registry)));
            }
        }

        ++_regions_rewritten;
        modified = true;
        report(2, fmt::format("rewrote region {} ({} nodes)", index, region.size()));
    }

    if (modified) {
        report(1, fmt::format("rewrote {} of {} region(s)", _regions_rewritten, _regions_formed));
        EINSUMS_LOG_INFO("{}: rewrote {} of {} region(s)", name(), _regions_rewritten, _regions_formed);
    }
    // The dumps are in reverse region order because the rewrite is; put them
    // back so a report reads in program order, which is how a person reads a
    // graph.
    std::ranges::reverse(_dumps);
    return modified;
}

bool RegionIdentity::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    // Deliberately empty. Returning true is what makes the lowering happen, and
    // the lowering is the thing under test.
    (void)graph;
    (void)region;
    (void)expr;
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
