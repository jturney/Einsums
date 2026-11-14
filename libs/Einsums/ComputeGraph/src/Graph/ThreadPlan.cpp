//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ThreadPlan.cpp
/// @brief The measured trial between a cold thread plan and its re-plan from timings.

#include <Einsums/ComputeGraph/Detail/ThreadPlan.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

namespace {

using Snapshot = std::vector<std::pair<std::uint16_t, std::int64_t>>;

/// Record every node's planned width and admission priority, container bodies
/// included, in one deterministic walk order shared with apply_plan.
///
/// Setup bodies are deliberately left out of both walks: the widths these two
/// snapshot and restore are the ones ThreadPlanning::plan_graph writes, and that
/// planner descends into loop bodies and conditional branches only.
void collect_plan(Graph const &graph, Snapshot &out) {
    for (auto const &node : graph.nodes()) {
        out.emplace_back(node.thread_width, node.admission_priority);
        for_each_child_graph(node, [&out](Graph const &sub) { collect_plan(sub, out); }, /*include_setup=*/false);
    }
}

void apply_plan(Graph &graph, Snapshot const &plan, std::size_t &pos) {
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        if (pos >= plan.size()) {
            return; // structure changed under the trial; leave the rest alone
        }
        NodeSchedule schedule       = schedule_of(graph.nodes()[i]);
        schedule.thread_width       = plan[pos].first;
        schedule.admission_priority = plan[pos].second;
        graph.set_schedule(i, schedule);
        pos++;
        graph.for_each_child(i, [&](Graph &sub) { apply_plan(sub, plan, pos); }, /*include_setup=*/false);
    }
}

[[nodiscard]] bool same_widths(Snapshot const &a, Snapshot const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); i++) {
        if (a[i].first != b[i].first) {
            return false;
        }
    }
    return true;
}

/// How much faster the candidate's replay must be before it replaces the cold
/// plan. Consecutive replays of one graph differ by a few percent on a machine
/// doing anything else, and the trial reads exactly one replay of each side,
/// so a candidate inside this band is indistinguishable from noise - and the
/// incumbent is the plan a calibrated model chose on purpose.
constexpr double kTrialMargin = 0.05;

} // namespace

void ThreadPlan::drop_snapshots() noexcept {
    _incumbent.clear();
    _candidate.clear();
}

void ThreadPlan::arm() noexcept {
    _trial = Trial::Armed;
    drop_snapshots();
}

void ThreadPlan::settle() noexcept {
    _trial = Trial::None;
    drop_snapshots();
}

bool ThreadPlan::begin_replan(Graph const &graph) {
    if (_trial != Trial::Measured) {
        return false;
    }
    // The re-plan runs here, as the second replay starts, and not where the
    // first replay ends. That is the same point in a graph's life whenever a
    // second replay comes, and it never comes for a graph built and executed
    // once (one per chunk of a sweep): a full planner run there bought nothing
    // and took longer than the graph's own work.
    //
    // Disarmed BEFORE planning, so a planner that throws does not leave the
    // graph re-planning at the start of every replay from here on.
    _trial = Trial::None;
    _incumbent.clear();
    collect_plan(graph, _incumbent);
    return true;
}

void ThreadPlan::offer_candidate(Graph const &graph) {
    _candidate.clear();
    collect_plan(graph, _candidate);
    if (same_widths(_candidate, _incumbent)) {
        // The timings agree with the model about the widths, so there is
        // nothing to referee; the re-plan's measured admission priorities
        // are kept and the plan is final.
        drop_snapshots();
        return;
    }
    // The candidate's widths are live now; the replay starting times them.
    _trial = Trial::Candidate;
}

void ThreadPlan::finish_replay(Graph &graph, double replay_ms) {
    switch (_trial) {
    case Trial::None:
    case Trial::Measured:
        return;

    case Trial::Armed:
        // This replay's timings are what the re-plan reads, when the next one starts.
        _trial = Trial::Measured;
        return;

    case Trial::Candidate: {
        _candidate_ms   = replay_ms;
        std::size_t pos = 0;
        apply_plan(graph, _incumbent, pos);
        _trial = Trial::Incumbent;
        return;
    }

    case Trial::Incumbent: {
        _trial               = Trial::None;
        bool const candidate = _candidate_ms < (1.0 - kTrialMargin) * replay_ms;
        if (candidate) {
            std::size_t pos = 0;
            apply_plan(graph, _candidate, pos);
        }
        if (config::get(option::PassVerbosity) >= 1) {
            einsums::detail::get_passes_logger().info(
                "[ThreadPlanning] trial on '{}': candidate replay {:.1f} ms vs incumbent replay {:.1f} ms; keeping the {}", graph.name(),
                _candidate_ms, replay_ms, candidate ? "re-planned widths" : "cold plan");
        }
        _incumbent.clear();
        _incumbent.shrink_to_fit();
        _candidate.clear();
        _candidate.shrink_to_fit();
        return;
    }
    }
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
