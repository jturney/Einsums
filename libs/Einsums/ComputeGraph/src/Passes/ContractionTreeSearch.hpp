//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ContractionTreeSearch.hpp
 * @brief The subset program that brackets one product, and the letter table it prices against.
 *
 * Private to the pass sources. Two passes ask the same question of a product of several
 * tensors: which binary bracketing of it is cheapest under @ref SymbolicCost. `MultiTermFactorization`
 * asks it of a captured chain it flattened, and `FactorizationPass` asks it of the chain a
 * provider's factors leave behind once a tagged operand has been substituted away. One
 * implementation, because two would rank the same candidates by two rules and a plan found by
 * one could not be checked against the report written by the other.
 *
 * The program is the standard one: the best way to contract a set of factors is the best split
 * of it into two sets, each contracted optimally. It is @c 3^N in the factor count, so a caller
 * caps that count rather than letting a pathological product decide how long a pass runs.
 *
 * Masks and submasks are walked in a fixed integer order and an improvement must be STRICT, so
 * the first optimum encountered is the one kept and two runs over one product pick the same
 * tree. A search whose answer varies between runs makes every measurement against it noise.
 */

#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::search)

/// A factor set, as a bit per factor. Capped by the caller so the subset program stays bounded;
/// the type only fixes the width of the mask.
using Mask = std::uint32_t;

/// @brief One leaf factor of a product the search brackets.
struct Factor {
    TensorId               tensor{};
    std::vector<ExprIndex> indices;
    bool                   conjugate{false};
};

/// @brief Everything the cost model needs to know about an index letter.
///
/// The extents are here because the graph is BOUND: a region raised from a live graph knows how
/// big every operand is, and throwing that away would leave the search ranking most of its
/// candidates by the comparison's last rung, which is a deterministic tie-break rather than an
/// answer. They are fed in as @ref ComparisonContext::bound_extent, which sits BELOW the
/// scale-order rung, so an annotated program's asymptotic verdict still decides and the extents
/// only settle what would otherwise be arbitrary.
///
/// That is a machine-independent input in the sense the phase rule cares about, since the extents
/// come from the problem rather than from the hardware; what it is not is size-independent, and a
/// plan chosen at one size stays CORRECT at another rather than necessarily optimal. That is the
/// same bargain @c LayoutAssignment strikes with its permute costs.
struct LetterTable {
    std::unordered_map<std::string, SymbolicVar> var;
    std::unordered_map<std::string, std::size_t> extent;
    std::unordered_map<std::string, double>      anonymous_extent;
    std::map<std::uint64_t, double>              space_extent;

    /// Letters whose extent is a CONSTANT of the rewrite rather than a dimension of the problem.
    ///
    /// A quadrature index is the case this exists for: its length is fixed by a tolerance and
    /// does not move when the molecule does, so a cost model that gave it a scale variable
    /// would say the arithmetic grows with it and would rank a decoupled form as one scale
    /// order worse than it is. Such a letter multiplies the polynomial by a number instead.
    std::unordered_map<std::string, double> constant;

    void observe(ExprIndex const &index, std::size_t extent_value) {
        if (auto const [it, fresh] = var.try_emplace(index.letter); fresh) {
            it->second = index.space.valid() ? SymbolicVar::space(index.space) : SymbolicVar::anonymous(index.letter);
        }
        extent.try_emplace(index.letter, extent_value);
        if (index.space.valid()) {
            space_extent.try_emplace(static_cast<std::uint64_t>(index.space.value()), static_cast<double>(extent_value));
        } else {
            anonymous_extent.try_emplace(index.letter, static_cast<double>(extent_value));
        }
    }

    /// @brief Note a letter whose extent is a constant of a rewrite rather than a scale.
    /// @param[in] letter The index letter.
    /// @param[in] value  Its length.
    void observe_constant(std::string const &letter, std::size_t value) {
        constant[letter] = static_cast<double>(value);
        extent[letter]   = value;
        var.try_emplace(letter, SymbolicVar::anonymous(letter));
    }

    /// @brief The lookup @ref ComparisonContext::bound_extent wants.
    /// @return A resolver over every letter this table has observed.
    [[nodiscard]] ExtentLookup lookup() const {
        return [this](SymbolicVar const &variable) -> std::optional<double> {
            if (variable.is_anonymous()) {
                auto const it = anonymous_extent.find(variable.letter());
                return it == anonymous_extent.end() ? std::nullopt : std::optional<double>{it->second};
            }
            if (variable.is_space()) {
                auto const it = space_extent.find(static_cast<std::uint64_t>(variable.space_id().value()));
                return it == space_extent.end() ? std::nullopt : std::optional<double>{it->second};
            }
            return std::nullopt;
        };
    }
};

/// @brief The product of the given letters' extent variables, coefficient one.
/// @param[in] letters Distinct letters, in any order; the polynomial is canonical either way.
/// @param[in] table   The letter lookup.
/// @return The polynomial.
inline SymbolicPoly poly_over(std::set<std::string> const &letters, LetterTable const &table) {
    SymbolicPoly poly = SymbolicPoly::constant(1.0);
    for (auto const &letter : letters) {
        if (auto const fixed = table.constant.find(letter); fixed != table.constant.end()) {
            poly *= SymbolicPoly::constant(fixed->second);
            continue;
        }
        auto const it = table.var.find(letter);
        poly *= SymbolicPoly::variable(it == table.var.end() ? SymbolicVar::anonymous(letter) : it->second);
    }
    return poly;
}

/// @brief The cost of one binary contraction, in the same shape @ref symbolic_cost_for uses.
///
/// Deliberately the same conventions rather than a second opinion: flops is twice the loop space
/// for the multiply and the add, traffic is the three operand sizes, and resident equals traffic.
/// A search that ranked candidates by a different rule than the one @c ScalingAnalysis reports
/// would produce a plan nobody could check against the report.
///
/// @param[in] left  The left operand's distinct letters.
/// @param[in] right The right operand's distinct letters.
/// @param[in] out   The result's distinct letters.
/// @param[in] table The letter lookup.
/// @return The cost bundle.
inline SymbolicCost contraction_cost(std::set<std::string> const &left, std::set<std::string> const &right,
                                     std::set<std::string> const &out, LetterTable const &table) {
    std::set<std::string> loop = left;
    loop.insert(right.begin(), right.end());

    SymbolicCost cost;
    cost.flops    = poly_over(loop, table) * 2.0;
    cost.traffic  = poly_over(out, table) + poly_over(left, table) + poly_over(right, table);
    cost.resident = cost.traffic;
    return cost;
}

/// @brief The sum of two cost bundles.
/// @param[in] lhs Left operand.
/// @param[in] rhs Right operand.
/// @return The bundle whose every field is the sum of theirs.
inline SymbolicCost add_cost(SymbolicCost const &lhs, SymbolicCost const &rhs) {
    return SymbolicCost{.flops = lhs.flops + rhs.flops, .traffic = lhs.traffic + rhs.traffic, .resident = lhs.resident + rhs.resident};
}

/// @brief The distinct letters of one factor.
/// @param[in] factor The factor.
/// @return Its letters, deduplicated.
inline std::set<std::string> letters_of(Factor const &factor) {
    std::set<std::string> out;
    for (auto const &index : factor.indices) {
        out.insert(index.letter);
    }
    return out;
}

/// @brief The distinct letters of an index list.
/// @param[in] indices The list.
/// @return Its letters, deduplicated.
inline std::set<std::string> letters_of(std::vector<ExprIndex> const &indices) {
    std::set<std::string> out;
    for (auto const &index : indices) {
        out.insert(index.letter);
    }
    return out;
}

/// @brief The result of the subset program over one product.
struct TreePlan {
    bool                      ok{false};
    SymbolicCost              cost;
    std::vector<Mask>         split;    ///< split[mask] = the left half chosen for that mask.
    std::vector<std::uint8_t> resolved; ///< Whether split[mask] is meaningful.
};

/**
 * @brief The optimal binary contraction tree over @p factors producing @p output.
 *
 * @param[in] factors The leaves, in a fixed order the caller also uses to read the plan.
 * @param[in] output  The index list the whole product must expose.
 * @param[in] table   The letter lookup, complete over every letter any factor mentions.
 * @param[in] ctx     What the comparison may consult.
 * @return The plan. @ref TreePlan::ok is false when no tree was found.
 */
inline TreePlan solve_tree(std::vector<Factor> const &factors, std::vector<ExprIndex> const &output, LetterTable const &table,
                           ComparisonContext const &ctx) {
    std::size_t const count = factors.size();
    Mask const        full  = static_cast<Mask>((Mask{1} << count) - 1);

    std::vector<std::set<std::string>> factor_letters;
    factor_letters.reserve(count);
    for (auto const &factor : factors) {
        factor_letters.push_back(letters_of(factor));
    }
    std::set<std::string> const output_letters = letters_of(output);

    // The letters a subset must still expose: those it shares with a factor outside it, plus
    // those the product's own output asks for. Everything else is summed away when the subset is
    // formed, which is exactly what makes an early contraction cheap.
    auto external_of = [&](Mask mask) {
        std::set<std::string> inside;
        for (std::size_t f = 0; f < count; f++) {
            if ((mask & (Mask{1} << f)) != 0) {
                inside.insert(factor_letters[f].begin(), factor_letters[f].end());
            }
        }
        std::set<std::string> outside = output_letters;
        for (std::size_t f = 0; f < count; f++) {
            if ((mask & (Mask{1} << f)) == 0) {
                outside.insert(factor_letters[f].begin(), factor_letters[f].end());
            }
        }
        std::set<std::string> kept;
        for (auto const &letter : inside) {
            if (outside.count(letter) != 0) {
                kept.insert(letter);
            }
        }
        return kept;
    };

    std::vector<std::set<std::string>> external(static_cast<std::size_t>(full) + 1);
    for (Mask mask = 1; mask <= full; mask++) {
        external[mask] = external_of(mask);
    }

    TreePlan plan;
    plan.split.assign(static_cast<std::size_t>(full) + 1, 0);
    plan.resolved.assign(static_cast<std::size_t>(full) + 1, 0);
    std::vector<SymbolicCost> best(static_cast<std::size_t>(full) + 1);
    std::vector<std::uint8_t> have(static_cast<std::size_t>(full) + 1, 0);

    for (std::size_t f = 0; f < count; f++) {
        Mask const single = Mask{1} << f;
        best[single]      = SymbolicCost{};
        have[single]      = 1;
    }

    for (Mask mask = 1; mask <= full; mask++) {
        if (std::popcount(mask) < 2) {
            continue;
        }
        // Proper non-empty submasks, each pair visited once: taking only halves whose lowest set
        // bit is the mask's lowest set bit avoids visiting (S1,S2) and (S2,S1).
        Mask const anchor = mask & (~mask + 1);
        for (Mask left = (mask - 1) & mask; left != 0; left = (left - 1) & mask) {
            if ((left & anchor) == 0) {
                continue;
            }
            Mask const right = mask ^ left;
            if (right == 0 || have[left] == 0 || have[right] == 0) {
                continue;
            }
            SymbolicCost const combined =
                add_cost(add_cost(best[left], best[right]), contraction_cost(external[left], external[right], external[mask], table));
            if (have[mask] == 0 || compare(combined, best[mask], ctx) < 0) {
                best[mask]          = combined;
                have[mask]          = 1;
                plan.split[mask]    = left;
                plan.resolved[mask] = 1;
            }
        }
    }

    if (have[full] == 0) {
        return plan;
    }
    plan.ok   = true;
    plan.cost = best[full];
    return plan;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::search)
