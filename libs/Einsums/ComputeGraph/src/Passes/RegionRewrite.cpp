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

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

void RegionRewrite::reset_stats() {
    _dumps.clear();
    _regions_formed    = 0;
    _regions_rewritten = 0;
    _regions_declined  = 0;
}

std::vector<std::string> RegionRewrite::explain() const {
    std::vector<std::string> out;
    if (_regions_rewritten != 0) {
        out.push_back(fmt::format("{}: rewrote {} of {} region(s)", name(), _regions_rewritten, _regions_formed));
    }
    // Which regions formed, and which declined.
    // Reported even when nothing was rewritten, because "formed twelve regions
    // and rewrote none" and "formed none" are different findings and only the
    // first one means the region rule is working.
    if (_regions_rewritten == 0 && _regions_formed != 0) {
        out.push_back(fmt::format("{}: formed {} region(s), rewrote none", name(), _regions_formed));
    }
    if (_regions_declined != 0) {
        out.push_back(fmt::format("{}: {} region(s) could not be raised or lowered; see the skip tally", name(), _regions_declined));
    }
    for (auto const &dump : _dumps) {
        if (!dump.changed) {
            continue;
        }
        out.push_back(fmt::format("{}: region {} ({} nodes)\n      before: {}\n      after:  {}", name(), dump.region_index,
                                  dump.node_count, dump.before, dump.after));
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

bool RegionRewrite::run(Graph &graph) {
    bool const dump = _dump || config::get(option::GraphDumpRegions);

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
            ++_regions_declined;
            note_skip(raised.error().reason, raised.error().detail);
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
        if (dump) {
            record.after   = changed ? rewritten.to_string(registry) : record.before;
            record.changed = changed;
            _dumps.push_back(record);
            report(2, fmt::format("region {} ({} nodes)\n  before:\n{}  after:\n{}", index, region.size(), record.before, record.after));
        }
        if (!changed) {
            continue;
        }

        if (auto lowered = lower_region(graph, region, rewritten); !lowered) {
            // The graph is untouched: lower_region builds every node before it
            // erases anything, so a refusal costs the rewrite and nothing else.
            ++_regions_declined;
            note_skip(lowered.error().reason, lowered.error().detail);
            continue;
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

bool RegionIdentity::rewrite(Graph const &graph, Region const &region, TensorExpr &expr) {
    // Deliberately empty. Returning true is what makes the lowering happen, and
    // the lowering is the thing under test.
    (void)graph;
    (void)region;
    (void)expr;
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
