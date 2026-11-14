//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ThreadPlan.hpp
/// @brief A graph's thread plan: the thread count its node widths assume, and the measured trial
///        that decides whether a re-plan from real timings replaces the cold model plan.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

namespace detail {

/**
 * @brief What a graph knows about the thread plan on its nodes, beyond the widths themselves.
 *
 * The widths live on the nodes, where the executors read them. This holds the two things that
 * are not per-node: the thread count the plan divided, and where a cold plan is in its trial.
 *
 * The trial runs over the graph's next replays. A cold plan is a model's guess; the first
 * replay's timings let the planner re-plan, but a plan computed from timings taken under the
 * plan it would replace cannot be judged by estimates, so the two are raced on the wall clock:
 * the candidate is timed for one replay, the incumbent for the next, and the candidate stays
 * only when it wins by more than the noise band.
 *
 * The methods that walk the graph take it as an argument rather than holding it, so a graph
 * that is moved carries its plan with nothing to re-point.
 */
class EINSUMS_EXPORT ThreadPlan {
  public:
    /// Threads the node widths were planned for; 0 when nothing recorded them.
    [[nodiscard]] unsigned planned_threads() const noexcept { return _planned_threads; }

    /// Record the thread count the current widths assume.
    void record_planned_threads(unsigned threads) noexcept { _planned_threads = static_cast<std::uint16_t>(threads); }

    /// Whether a trial is open: the widths may still change at a coming replay.
    [[nodiscard]] bool trial_open() const noexcept { return _trial != Trial::None; }

    /// Start a trial for a cold plan just made, so the next replay records the timings a
    /// re-plan reads.
    void arm() noexcept;

    /// Close any open trial and keep the widths on the nodes as they are. For a plan made from
    /// timings, a frozen plan, and a width set by hand, each of which is a decision a trial
    /// would otherwise re-plan over.
    void settle() noexcept;

    /**
     * @brief The first half of the re-plan, as a replay starts.
     *
     * When the previous replay recorded the timings the trial waits for, closes the trial
     * (so a planner that throws leaves no re-plan pending at every replay from here on),
     * snapshots the cold widths, and returns true: the caller then runs the planner and
     * hands the graph back to @ref offer_candidate. Returns false, touching nothing, at
     * every other replay.
     */
    [[nodiscard]] bool begin_replan(Graph const &graph);

    /// The second half: snapshot the re-planned widths now on the nodes and, when they differ
    /// from the cold ones, open the race with the candidate live for the starting replay.
    void offer_candidate(Graph const &graph);

    /// Advance the trial at the end of a completed replay that took @p replay_ms.
    void finish_replay(Graph &graph, double replay_ms);

  private:
    /// Where a cold plan is in its trial.
    enum class Trial : std::uint8_t {
        None,      ///< No decision pending: the widths on the nodes are final.
        Armed,     ///< Cold model plan is live; the next replay records the timings the re-plan reads.
        Measured,  ///< A replay has recorded timings; the re-plan runs when the next replay starts.
        Candidate, ///< The re-planned widths are live and this replay is timing them.
        Incumbent, ///< The cold widths are back and this replay is timing them.
    };

    /// Every planned width and admission priority in the graph tree, in walk order: the
    /// graph's nodes, then each container body's, recursively.
    using Snapshot = std::vector<std::pair<std::uint16_t, std::int64_t>>;

    void drop_snapshots() noexcept;

    std::uint16_t _planned_threads{0};
    Trial         _trial{Trial::None};
    Snapshot      _incumbent;         ///< The cold model plan, held during the trial.
    Snapshot      _candidate;         ///< The timings re-plan, held during the trial.
    double        _candidate_ms{0.0}; ///< Wall clock of the candidate's replay.
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
