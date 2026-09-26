//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NodeFeatures.hpp
 * @brief What a node carries beyond a plain dense operation, and what a pass says it understands.
 *
 * Features are added to nodes over time: permutation operators, views, conjugation, complex
 * prefactors, grouped families. A pass written before a feature existed reads a node carrying it
 * as the plain operation it knows and rewrites it into something else, silently: seven passes did
 * exactly that with permutation operators. So each pass declares the features it understands
 * (@ref OptimizerPass::understood_features), leaves every other node alone
 * (@ref OptimizerPass::understands), and a test build checks it did.
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <string>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;
struct Node;

/// @brief One thing a node can carry that a pass must know about to rewrite it correctly.
enum class NodeFeature : std::uint32_t {
    PermutationOperators = 1U << 0, ///< Applies P(..) to its result: a signed sum of permuted terms.
    Views                = 1U << 1, ///< Names a view: a tensor laid over part of another's storage.
    Conjugation          = 1U << 2, ///< Conjugates an operand.
    ComplexPrefactor     = 1U << 3, ///< Scales by a prefactor with a nonzero imaginary part.
    MixedPrecision       = 1U << 4, ///< Its operands hold different element types.
    Grouped              = 1U << 5, ///< Computes a family of members in one node.
    ControlFlow          = 1U << 6, ///< Owns sub-graphs (a loop, a branch, a setup).
    Tiled                = 1U << 7, ///< Names a tile-wise sparse tensor, which has no single buffer.
    RawScalar            = 1U << 8, ///< Names a bare scalar written through a pointer.
    RedirectedSlot       = 1U << 9, ///< Names a tensor whose slot a pass redirected to another's storage.
};

/// @brief A set of @ref NodeFeature values.
class NodeFeatures {
  public:
    /// @brief The empty set.
    constexpr NodeFeatures() = default;

    /// @brief The set holding @p feature alone.
    /// @param[in] feature The feature.
    constexpr NodeFeatures(NodeFeature feature) : _bits(static_cast<std::uint32_t>(feature)) {} // NOLINT(google-explicit-constructor)

    /// @brief Every feature there is, which is what a pass that never interprets a node declares.
    /// @return The full set.
    [[nodiscard]] static constexpr NodeFeatures all() { return NodeFeatures{kAll}; }

    /// @brief The union of two sets.
    /// @param[in] other The other set.
    /// @return Every feature in either.
    [[nodiscard]] constexpr NodeFeatures operator|(NodeFeatures other) const { return NodeFeatures{_bits | other._bits}; }

    /// @brief This set less @p other.
    /// @param[in] other The features to drop.
    /// @return Every feature here that is not in @p other.
    [[nodiscard]] constexpr NodeFeatures without(NodeFeatures other) const { return NodeFeatures{_bits & ~other._bits}; }

    /// @brief The features of @p other this set does not hold.
    /// @param[in] other The set to compare against.
    /// @return @p other less this set.
    [[nodiscard]] constexpr NodeFeatures missing_from(NodeFeatures other) const { return NodeFeatures{other._bits & ~_bits}; }

    /// @brief Whether this set holds every feature of @p other.
    /// @param[in] other The set to test.
    /// @return True when nothing in @p other is missing here.
    [[nodiscard]] constexpr bool covers(NodeFeatures other) const { return (other._bits & ~_bits) == 0; }

    /// @brief Whether the set is empty.
    /// @return True for the empty set.
    [[nodiscard]] constexpr bool empty() const { return _bits == 0; }

    /// @brief Whether two sets hold the same features.
    /// @return True when they do.
    constexpr bool operator==(NodeFeatures const &) const = default;

  private:
    static constexpr std::uint32_t kAll = (1U << 10) - 1;

    constexpr explicit NodeFeatures(std::uint32_t bits) : _bits(bits) {}

    std::uint32_t _bits{0};
};

/// @brief The union of two features.
/// @param[in] a One feature.
/// @param[in] b Another.
/// @return The set holding both.
[[nodiscard]] constexpr NodeFeatures operator|(NodeFeature a, NodeFeature b) {
    return NodeFeatures{a} | NodeFeatures{b};
}

/**
 * @brief The features @p node carries in @p graph.
 * @param[in] graph The graph the node belongs to, for its operands' handles and slot redirects.
 * @param[in] node  The node.
 * @return Every feature the node carries; empty for a plain dense operation.
 */
[[nodiscard]] EINSUMS_EXPORT NodeFeatures features_of(Graph const &graph, Node const &node);

/**
 * @brief The features in @p features, by name, for a diagnostic.
 * @param[in] features The set.
 * @return The names joined with ``", "``, or ``"none"``.
 */
[[nodiscard]] EINSUMS_EXPORT std::string describe_features(NodeFeatures features);

EINSUMS_NAMESPACE_END(compute_graph)
