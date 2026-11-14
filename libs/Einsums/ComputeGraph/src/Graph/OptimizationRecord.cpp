//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file OptimizationRecord.cpp
/// @brief The pass history a graph carries, and the rule a lossy pass's bound is checked by.

#include <Einsums/ComputeGraph/Detail/OptimizationRecord.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

void OptimizationRecord::note_structural_pass(std::string pass_name) {
    // Once per pass, however many times it ran. The list says what shaped this graph; a count
    // of applies would say something about the caller's pipeline instead, and the two get
    // confused the moment anyone applies a manager twice.
    if (std::ranges::find(_structural_passes, pass_name) != _structural_passes.end()) {
        return;
    }
    _structural_passes.push_back(std::move(pass_name));
}

void OptimizationRecord::note_machine_dependent_pass(std::string pass_name) {
    if (std::ranges::find(_machine_dependent_passes, pass_name) != _machine_dependent_passes.end()) {
        return;
    }
    _machine_dependent_passes.push_back(std::move(pass_name));
}

// ── The accuracy contract ───────────────────────────────────────────────────

namespace {

/// Whether @p record's bound applies to @p output.
///
/// A record naming no outputs applies to all of them, which is the honest answer for a pass
/// that rewrote something feeding every result; and an EMPTY question means "the graph-wide
/// worst case", which every record answers.
bool record_covers(ApproximationRecord const &record, std::string_view output) {
    if (output.empty() || record.outputs.empty()) {
        return true;
    }
    return std::find(record.outputs.begin(), record.outputs.end(), output) != record.outputs.end();
}

/// The outputs @p candidate could collide with an existing record over.
bool records_overlap(ApproximationRecord const &a, ApproximationRecord const &b) {
    if (a.outputs.empty() || b.outputs.empty()) {
        return true;
    }
    return std::any_of(a.outputs.begin(), a.outputs.end(),
                       [&b](std::string const &name) { return std::find(b.outputs.begin(), b.outputs.end(), name) != b.outputs.end(); });
}

} // namespace

std::string OptimizationRecord::refusal(ApproximationRecord const &candidate) const {
    if (!std::isfinite(candidate.bound) || candidate.bound < 0) {
        return fmt::format("pass '{}' states a bound of {}, which is not a number an accuracy budget can be measured against; a lossy "
                           "rewrite has to say how large its effect is",
                           candidate.pass_name, candidate.bound);
    }

    // Records of DIFFERENT effects are allowed to coexist, and deliberately. They do not
    // convert into one another, but neither do they need to: composition is per effect
    // (@ref accuracy_spent counts one kind at a time) and @ref approximation_tolerance
    // carries the two sides separately, which is the honest representation of "this result
    // is off by so much in norm and so much per element". Refusing the second one would
    // make an ordinary pipeline, a factorization followed by a precision change,
    // unexpressible for no gain.
    //
    // Whether a pass's own error model still holds once something else has perturbed its
    // inputs is a question only that pass can answer, so it is asked of the pass rather
    // than decided here: @ref approximations is readable, and a pass that finds its bound
    // no longer defensible declines through @ref OptimizerPass::approximate like any other
    // refusal.
    if (!_budget.has_value()) {
        return {};
    }
    auto const [budget_effect, budget] = *_budget;
    if (budget_effect != candidate.effect) {
        return fmt::format("pass '{}' bounds itself {}, and this graph's accuracy budget is stated {}; a budget in other units is not a "
                           "budget this pass can spend against",
                           candidate.pass_name, candidate.effect, budget_effect);
    }

    // A budget IS the one place mixed effects have to be refused, and only because of what a
    // budget claims to be. It caps one kind of error; a record of another kind over the same
    // output is not counted by it and cannot be, so letting this through would leave a
    // caller with a cap they reasonably read as covering everything and that covers part.
    // Saying so is better than capping half of it in silence.
    for (auto const &existing : _approximations) {
        if (existing.effect == budget_effect || !records_overlap(existing, candidate)) {
            continue;
        }
        return fmt::format("pass '{}' would be spent against a {} budget, but '{}' has already been applied to the same outputs with a "
                           "{} bound, which that budget does not cap and cannot; clear the budget or state it in the other units",
                           candidate.pass_name, budget_effect, existing.pass_name, existing.effect);
    }

    // Checked per named output rather than graph-wide, so two passes over DISJOINT outputs
    // each get the whole budget, which is what a budget on one output means.
    std::vector<std::string> const targets = candidate.outputs.empty() ? std::vector<std::string>{std::string{}} : candidate.outputs;
    for (auto const &target : targets) {
        double const composed = compose_approximation(candidate.effect, accuracy_spent(candidate.effect, target), candidate.bound);
        if (composed > budget) {
            return fmt::format("pass '{}' would take {} to {:g} on {}, over this graph's budget of {:g}", candidate.pass_name,
                               candidate.effect, composed, target.empty() ? std::string{"every output"} : fmt::format("'{}'", target),
                               budget);
        }
    }
    return {};
}

double OptimizationRecord::accuracy_spent(ApproximationEffect effect, std::string const &output) const {
    double spent = 0;
    for (auto const &record : _approximations) {
        if (record.effect == effect && record_covers(record, output)) {
            spent = compose_approximation(effect, spent, record.bound);
        }
    }
    return spent;
}

ApproximationTolerance OptimizationRecord::approximation_tolerance(std::string const &output) const {
    ApproximationTolerance out;
    for (auto const &record : _approximations) {
        if (!record_covers(record, output)) {
            continue;
        }
        double &side = is_absolute_effect(record.effect) ? out.absolute : out.relative;
        side         = compose_approximation(record.effect, side, record.bound);
    }
    return out;
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
