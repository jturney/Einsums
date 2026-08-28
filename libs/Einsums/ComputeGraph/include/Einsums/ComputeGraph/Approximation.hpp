//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief What a lossy pass's tolerance is a bound ON.
 *
 * A number without units is not a bound, and the three kinds below do not convert into one
 * another: an element-wise bound says nothing about a norm, and a norm bound says nothing
 * about the worst element. So a record carries which one it means and composition happens
 * within a kind, never across.
 *
 * Records of different kinds coexist on one output rather than being refused. There is
 * nothing that needs them as a single number: @ref Graph::accuracy_spent counts one kind at
 * a time and @ref ApproximationTolerance reports the two sides separately, which is the
 * honest description of a result that is off by so much in norm and so much per element.
 * A declared BUDGET is the exception, because a budget claims to be a cap and can only cap
 * its own units; see @ref Graph::can_approximate.
 *
 * @see ApproximationRecord
 * @see compose_approximation
 * @versionadded{2.0.0}
 */
enum class APIARY_EXPOSE APIARY_MODULE("graph") ApproximationEffect : std::uint8_t{
    /// Worst absolute difference over the elements of the named outputs.
    ElementWise,
    /// Difference in norm, RELATIVE to the exact result's norm.
    NormRelative,
    /// Absolute error in a scalar the outputs reduce to, an energy being the case this
    /// exists for. Composes with @ref ElementWise, since both are absolute.
    EnergyLike,
};

/**
 * @brief Where a bound's number came from.
 *
 * A bound and a guess are the same double, and months later a reader of a saved graph cannot
 * tell them apart from the number alone. They deserve different trust, so the file carries
 * which it is.
 *
 * @ref Asserted is not a lesser answer, it is the only one available wherever measuring would
 * mean computing the exact result the approximation exists to avoid computing. A metric fit is
 * that case: its error is the difference from the exact tensor, and a caller holding that had
 * no reason to fit. @ref Measured belongs to a provider that genuinely knows, a truncated
 * decomposition holding its own discarded singular values being the plain example.
 * @versionadded{2.0.0}
 */
enum class APIARY_EXPOSE APIARY_MODULE("graph") ApproximationOrigin : std::uint8_t{
    /// The pass computed the error it actually had.
    Measured,
    /// A caller stated the error, and nothing here checked it.
    Asserted,
};

/**
 * @brief The name of an approximation origin, for diagnostics and for the saved form.
 * @param[in] origin The origin to name.
 * @return A stable spelling ("measured", "asserted").
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view approximation_origin_name(ApproximationOrigin origin) noexcept;

/**
 * @brief The @ref ApproximationOrigin spelled @p name, if there is one.
 * @param[in] name A spelling @ref approximation_origin_name produces.
 * @return The origin, or an empty optional when nothing is spelled that way.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<ApproximationOrigin> approximation_origin_from_name(std::string_view name) noexcept;

/**
 * @brief The name of an approximation effect, for diagnostics and for the saved form.
 * @param[in] effect The effect to name.
 * @return A stable spelling ("element-wise", "norm-relative", "energy-like"). Written by
 *         NAME into any saved artifact, never by numeric value, per the compatibility policy.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string_view approximation_effect_name(ApproximationEffect effect) noexcept;

/**
 * @brief The @ref ApproximationEffect spelled @p name, if there is one.
 * @param[in] name A spelling @ref approximation_effect_name produces.
 * @return The effect, or an empty optional when nothing is spelled that way.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::optional<ApproximationEffect> approximation_effect_from_name(std::string_view name) noexcept;

/**
 * @brief Whether @p effect is an ABSOLUTE bound rather than a relative one.
 * @param[in] effect The effect to classify.
 * @return True for @ref ApproximationEffect::ElementWise and @ref ApproximationEffect::EnergyLike.
 *
 * The one place the absolute/relative split is written down, because a comparison has to
 * decide which of its two knobs a bound widens and two answers to that would drift.
 * @versionadded{2.0.0}
 */
[[nodiscard]] constexpr bool is_absolute_effect(ApproximationEffect effect) noexcept {
    return effect == ApproximationEffect::ElementWise || effect == ApproximationEffect::EnergyLike;
}

/**
 * @brief The bound that holds after applying an approximation of size @p second on top of one
 *        of size @p first.
 *
 * @param[in] effect The units both bounds are in.
 * @param[in] first The bound already standing.
 * @param[in] second The bound being added.
 * @return The composed bound, never smaller than either input.
 *
 * @par Why the relative case is not a sum
 * For an ABSOLUTE effect this is the triangle inequality and the answer is the sum: if the
 * first rewrite moved the result by at most @p first and the second by at most @p second,
 * the pair moved it by at most their sum.
 *
 * For a RELATIVE one, summing is not a bound, it is an under-estimate. The second pass's
 * error is relative to the ALREADY PERTURBED result, so with
 * @f$\|x_1 - x_0\| \le e_1 \|x_0\|@f$ and @f$\|x_2 - x_1\| \le e_2 \|x_1\|@f$,
 * @f$\|x_2 - x_0\| \le (e_1 + e_2 + e_1 e_2)\|x_0\|@f$. The product term is what a sum
 * drops, and dropping it makes a composed budget quietly optimistic, which is the one
 * direction an accuracy contract must never fail in.
 * @versionadded{2.0.0}
 */
[[nodiscard]] constexpr double compose_approximation(ApproximationEffect effect, double first, double second) noexcept {
    return is_absolute_effect(effect) ? first + second : first + second + first * second;
}

/**
 * @brief One lossy rewrite that has been applied to a graph, and what it cost in accuracy.
 *
 * Accumulated on the @ref Graph and SAVED with its structure, because it is a statement about
 * what the graph now computes rather than about how it runs: a caller reading a result off a
 * loaded graph is entitled to know that the energy is DF at 1e-5, and no machine can recover
 * that from the node list.
 *
 * It is three things at once. Provenance for the user, the input to the composition rule that
 * lets a second lossy pass decide whether it may apply, and the trigger that widens a
 * differential test's tolerance from exact to whatever the records say.
 *
 * @see Graph::note_approximation
 * @see Graph::approximation_tolerance
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") ApproximationRecord {
    /// The pass that applied the rewrite, by @ref OptimizerPass::name.
    ///
    /// Not spelled ``pass``, which is what it wants to be called and is a Python keyword:
    /// the field would bind to a name no Python source can write down. The saved key matches
    /// this spelling rather than the shorter one, so the file and the type do not need a
    /// translation nobody would remember was there.
    APIARY_EXPOSE std::string pass_name;

    /// The knob the pass was ASKED for, e.g. a ``thc_epsilon`` of 1e-5. Recorded beside
    /// @ref bound rather than instead of it: a pass may well meet its target with room to
    /// spare, and a composition rule that read the target would spend accuracy nobody used.
    APIARY_EXPOSE double tolerance{0};

    /// What @ref bound is a bound on.
    APIARY_EXPOSE ApproximationEffect effect{ApproximationEffect::NormRelative};

    /// The effect the pass STATES it had, in @ref effect's units. This is what composes and
    /// what a tolerance-aware comparison widens by.
    APIARY_EXPOSE double bound{0};

    /// Whether @ref bound was computed or claimed. Defaults to @ref ApproximationOrigin::Asserted,
    /// which is also how a file written before this field existed is read: a number cannot be
    /// promoted to evidence by a newer build reading it.
    APIARY_EXPOSE ApproximationOrigin origin{ApproximationOrigin::Asserted};

    /// Manifest names the bound applies to. EMPTY means every output, which is the honest
    /// answer for a pass that rewrote something feeding all of them.
    APIARY_EXPOSE std::vector<std::string> outputs;

    /// Index-space names the rewrite involved, by NAME for the reason every other saved
    /// annotation holds names: an id means nothing in another process.
    APIARY_EXPOSE std::vector<std::string> spaces;

    /// Label of the setup node the rewrite created, or empty when it created none. A
    /// factorization emits its fitting into one, and the record is where the two are tied
    /// together for anyone asking what produced these factors.
    APIARY_EXPOSE std::string setup;
};

/**
 * @brief The comparison tolerance a set of approximation records implies.
 *
 * Two numbers rather than one because the effects do not share units: an absolute bound
 * widens the absolute side of a comparison and a relative one widens the relative side.
 * Which is which is @ref is_absolute_effect, in one place.
 *
 * Both zero means the graph carries no records, which is exact mode and today's behavior.
 *
 * @see Graph::approximation_tolerance
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") ApproximationTolerance {
    /// Composed bound of every relative record. Widens a comparison's relative tolerance.
    APIARY_EXPOSE APIARY_READONLY double relative{0};

    /// Composed bound of every absolute record. Widens a comparison's absolute tolerance.
    APIARY_EXPOSE APIARY_READONLY double absolute{0};
};

/**
 * @brief Build an @ref ApproximationRecord in one call.
 *
 * @param[in] pass_name The pass applying the rewrite.
 * @param[in] effect What @p bound is a bound on.
 * @param[in] tolerance The knob the pass was asked for.
 * @param[in] bound The effect the pass states it had.
 * @param[in] outputs Manifest names the bound applies to; empty means all of them.
 * @param[in] spaces Index-space names the rewrite involved.
 * @param[in] setup Label of the setup node it created, or empty.
 * @param[in] origin Whether @p bound was computed or claimed.
 * @return The record.
 *
 * A free function rather than a constructor, so @ref ApproximationRecord stays an AGGREGATE
 * and C++ callers keep designated initializers, which is how a struct of seven fields stays
 * readable at a call site. Declaring even a defaulted constructor would take that away.
 * Python has no aggregate initialization and needs some entry point, and this is a better
 * one than a default-constructed record with seven assignments after it.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("approximation_record") [[nodiscard]] EINSUMS_EXPORT ApproximationRecord
    make_approximation_record(std::string pass_name, ApproximationEffect effect, double tolerance, double bound,
                              std::vector<std::string> outputs = std::vector<std::string>{},
                              std::vector<std::string> spaces = std::vector<std::string>{}, std::string setup = "",
                              ApproximationOrigin origin = ApproximationOrigin::Asserted);

EINSUMS_NAMESPACE_END(compute_graph)
