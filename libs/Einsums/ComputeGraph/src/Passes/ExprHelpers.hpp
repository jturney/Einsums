//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ExprHelpers.hpp
 * @brief Letters and scratch tensors for the passes that rewrite through TensorExpr.
 *
 * Private to the pass sources. FactorizationPass, LaplaceTransform, DeltaElimination and
 * MultiTermFactorization each rename letters and declare the intermediates their rewrites
 * introduce, and all of them have to do it the same way.
 */

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::expr)

/// The letters of an index list, in order, without their spaces.
[[nodiscard]] inline std::vector<std::string> letter_list(std::vector<ExprIndex> const &indices) {
    std::vector<std::string> out;
    out.reserve(indices.size());
    for (auto const &index : indices) {
        out.push_back(index.letter);
    }
    return out;
}

[[nodiscard]] inline bool contains(std::vector<std::string> const &haystack, std::string const &needle) {
    return std::ranges::find(haystack, needle) != haystack.end();
}

/// A letter nothing in @p used spells, seeded from @p wanted.
[[nodiscard]] inline std::string fresh_letter(std::vector<std::string> const &used, std::string const &wanted) {
    if (!contains(used, wanted)) {
        return wanted;
    }
    for (int suffix = 1;; ++suffix) {
        std::string candidate = fmt::format("{}{}", wanted, suffix);
        if (!contains(used, candidate)) {
            return candidate;
        }
    }
}

/// Add a leaf term reading @p id under @p indices, named as the graph names it.
[[nodiscard]] inline TermId add_leaf(TensorExpr &expr, Graph const &graph, TensorId id, std::vector<ExprIndex> indices) {
    ExprTerm            leaf;
    TensorHandle const *held = graph.find_tensor(id);
    leaf.kind                = TermKind::Leaf;
    leaf.tensor              = id;
    leaf.name                = held != nullptr ? held->name : std::string{};
    leaf.indices             = std::move(indices);
    return expr.add(std::move(leaf));
}

/// Declare a graph-owned deferred intermediate of the given shape, or return 0 when the graph
/// could not register it.
///
/// Deferred rather than eager, and runtime-rank rather than typed, for the two reasons every
/// pass-created tensor has both: the memory passes can only manage storage they are allowed to
/// place, and a later bind can only move an extent whose storage has not been committed.
[[nodiscard]] inline TensorId declare_scratch(Graph &graph, std::string name, packed_gemm::ScalarType dtype,
                                              std::vector<std::size_t> const &dims) {
    return compute_graph::detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> TensorId {
        auto &tensor = graph.declare_runtime_tensor<T>(std::move(name), dims, /*intermediate=*/true);
        return graph.live_tensor_id_by_ptr(&tensor, {});
    });
}

EINSUMS_NAMESPACE_END(compute_graph::passes::expr)
