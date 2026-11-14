//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file OptimizationRecord.hpp
/// @brief What the optimizer has done to a graph: the passes that shaped it, the last
///        optimize() report, and the lossy rewrites it carries with the budget they spend.

#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/ComputeGraph/Decision.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

/**
 * @brief A graph's optimization history, and the accuracy contract its lossy rewrites keep.
 *
 * The pass lists and the approximation records are saved structure (a load restores them);
 * the report, the decisions and the budget are not. Nothing here reads the graph, so a rule
 * about what a lossy pass may spend is decided from the records alone.
 */
class EINSUMS_EXPORT OptimizationRecord {
  public:
    /// Record a structural-algebraic pass, once however many times it ran.
    void note_structural_pass(std::string pass_name);

    /// Record a structural-resource or tuning pass, once however many times it ran.
    void note_machine_dependent_pass(std::string pass_name);

    [[nodiscard]] std::vector<std::string> const &structural_passes() const noexcept { return _structural_passes; }
    [[nodiscard]] std::vector<std::string> const &machine_dependent_passes() const noexcept { return _machine_dependent_passes; }

    /// Keep what the last optimize() run reported, replacing the one before.
    void record_optimize(std::string report, std::vector<Decision> decisions) {
        _last_report    = std::move(report);
        _last_decisions = std::move(decisions);
    }

    [[nodiscard]] std::string const           &last_report() const noexcept { return _last_report; }
    [[nodiscard]] std::vector<Decision> const &last_decisions() const noexcept { return _last_decisions; }

    [[nodiscard]] std::vector<ApproximationRecord> const &approximations() const noexcept { return _approximations; }

    /// Why @p candidate may not be applied, or empty when it may. @see Graph::can_approximate
    [[nodiscard]] std::string refusal(ApproximationRecord const &candidate) const;

    /// Append a record @ref refusal has accepted.
    void add_approximation(ApproximationRecord record) { _approximations.push_back(std::move(record)); }

    /// Install a whole history, unchecked, as a load does.
    void restore_approximations(std::vector<ApproximationRecord> records) { _approximations = std::move(records); }

    /// The bound already composed on @p output for @p effect; empty @p output is every output.
    [[nodiscard]] double accuracy_spent(ApproximationEffect effect, std::string const &output) const;

    /// Both sides of the composed tolerance on @p output.
    [[nodiscard]] ApproximationTolerance approximation_tolerance(std::string const &output) const;

    [[nodiscard]] std::optional<std::pair<ApproximationEffect, double>> const &budget() const noexcept { return _budget; }
    void set_budget(ApproximationEffect effect, double value) noexcept { _budget = std::pair{effect, value}; }
    void clear_budget() noexcept { _budget.reset(); }

  private:
    /// Structural-algebraic passes that rewrote the graph, in the order they did.
    std::vector<std::string> _structural_passes;
    /// Structural-resource and tuning passes that changed the graph.
    std::vector<std::string> _machine_dependent_passes;

    /// Summary of the last optimize() run, and the decisions behind it.
    std::string           _last_report;
    std::vector<Decision> _last_decisions;

    /// Every lossy rewrite applied to the graph, in order. Saved with the structure.
    std::vector<ApproximationRecord> _approximations;

    /// The cap lossy passes compose against, or unset for no cap. NOT saved: a budget is
    /// what a caller was willing to spend, which is a property of the run rather than of
    /// the graph, and a loaded graph's caller gets to state their own.
    std::optional<std::pair<ApproximationEffect, double>> _budget;
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
