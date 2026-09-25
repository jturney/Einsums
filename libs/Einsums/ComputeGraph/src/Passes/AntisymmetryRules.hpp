//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file AntisymmetryRules.hpp
 * @brief How a permutation operator's letters become symmetry generators.
 *
 * Private to the pass sources. AntisymmetryDetection probes for these generators,
 * AntisymmetryInference tags them, and AntisymmetrizerFolding looks them up, so all three must
 * spell a group the same way: adjacent transpositions of sorted output axes. Folding compares
 * generator sets by lookup rather than closing them under composition, which is only sound
 * because the spelling here is the only one.
 */

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::antisymmetry)

/// Where a letter sits in a list, when it sits there exactly once and within the rank a
/// SymmetryDescriptor can address.
///
/// "Exactly once" is not fussiness. A letter appearing twice in an operand is a DIAGONAL access,
/// and swapping two output axes then does not correspond to swapping two of that operand's
/// slots, so a rule reasoning from it would be about a permutation the contraction does not
/// perform.
[[nodiscard]] inline std::optional<int> sole_position(std::vector<std::string> const &list, std::string const &letter) {
    if (std::ranges::count(list, letter) != 1) {
        return std::nullopt;
    }
    auto const at = static_cast<int>(std::ranges::find(list, letter) - list.begin());
    return at < kMaxSymmetryRank ? std::optional<int>{at} : std::nullopt;
}

/// The output axes @p letters name, sorted, or nothing when one is not a sole position.
[[nodiscard]] inline std::optional<std::vector<int>> sorted_axes(std::vector<std::string> const &letters,
                                                                 std::vector<std::string> const &c_indices) {
    std::vector<int> axes;
    for (auto const &letter : letters) {
        auto const at = sole_position(c_indices, letter);
        if (!at.has_value()) {
            return std::nullopt;
        }
        axes.push_back(*at);
    }
    std::ranges::sort(axes);
    return axes;
}

/// Each of @p op's letter groups as sorted output axes.
[[nodiscard]] inline std::optional<std::vector<std::vector<int>>> axis_groups(PermutationOperator const      &op,
                                                                              std::vector<std::string> const &c_indices) {
    std::vector<std::vector<int>> groups;
    for (auto const &group : op.groups) {
        auto axes = sorted_axes(group, c_indices);
        if (!axes.has_value()) {
            return std::nullopt;
        }
        groups.push_back(std::move(*axes));
    }
    return groups;
}

/// Add the adjacent transpositions of @p axes, which generate the symmetric group on them, each
/// with @p sign.
inline void add_adjacent_swaps(SymmetryDescriptor &desc, std::vector<int> const &axes, std::int8_t sign) {
    for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
        desc.add(SymmetryOp::swap(axes[k], axes[k + 1], sign));
    }
}

/// Antisymmetry WITHIN each group, which is what a coset operator's operand has to carry before
/// the operator's output carries anything. Empty for singleton groups.
[[nodiscard]] inline SymmetryDescriptor within_groups(std::vector<std::vector<int>> const &groups) {
    SymmetryDescriptor desc;
    for (auto const &axes : groups) {
        add_adjacent_swaps(desc, axes, -1);
    }
    return desc;
}

/// @ref within_groups over every operator, or nothing when a letter is not a sole position.
[[nodiscard]] inline std::optional<SymmetryDescriptor> within_groups(std::span<PermutationOperator const> operators,
                                                                     std::vector<std::string> const      &c_indices) {
    SymmetryDescriptor desc;
    for (auto const &op : operators) {
        auto const groups = axis_groups(op, c_indices);
        if (!groups.has_value()) {
            return std::nullopt;
        }
        for (auto const &axes : *groups) {
            add_adjacent_swaps(desc, axes, -1);
        }
    }
    return desc;
}

/// Full antisymmetry in all the letters each operator names: what an antisymmetrizer's output
/// carries. Nothing when a letter is not a sole position, an operator names fewer than two axes,
/// or there are no operators.
[[nodiscard]] inline std::optional<SymmetryDescriptor> full_antisymmetry(std::span<PermutationOperator const> operators,
                                                                         std::vector<std::string> const      &c_indices) {
    SymmetryDescriptor desc;
    for (auto const &op : operators) {
        auto const axes = sorted_axes(op.letters(), c_indices);
        if (!axes.has_value() || axes->size() < 2) {
            return std::nullopt;
        }
        add_adjacent_swaps(desc, *axes, -1);
    }
    if (desc.empty()) {
        return std::nullopt;
    }
    return desc;
}

/// Whether every generator of @p need appears in @p have, by lookup.
[[nodiscard]] inline bool contains_all(SymmetryDescriptor const &have, SymmetryDescriptor const &need) {
    return std::ranges::all_of(need.ops, [&](SymmetryOp const &op) { return std::ranges::find(have.ops, op) != have.ops.end(); });
}

/// The one contraction operand that carries both of two output letters, and where.
struct Carrier {
    std::size_t operand; ///< 0 for A, 1 for B
    int         at_p;    ///< the slot carrying the first letter
    int         at_q;    ///< the slot carrying the second letter
};

/// The operand that carries both @p p and @p q, each once, while the other carries neither.
///
/// Only then does swapping p and q in the output swap exactly two slots of one operand, so that
/// operand's antisymmetry in those slots negates the product and the other operand is untouched.
/// At most one operand can qualify, since the other must carry neither letter.
[[nodiscard]] inline std::optional<Carrier> sole_carrier(std::vector<std::string> const &a, std::vector<std::string> const &b,
                                                         std::string const &p, std::string const &q) {
    for (std::size_t which = 0; which < 2; ++which) {
        auto const &carrier = which == 0 ? a : b;
        auto const &other   = which == 0 ? b : a;
        auto const  at_p    = sole_position(carrier, p);
        auto const  at_q    = sole_position(carrier, q);
        if (!at_p.has_value() || !at_q.has_value() || std::ranges::find(other, p) != other.end() ||
            std::ranges::find(other, q) != other.end()) {
            continue;
        }
        return Carrier{.operand = which, .at_p = *at_p, .at_q = *at_q};
    }
    return std::nullopt;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::antisymmetry)
