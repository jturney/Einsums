//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LaplaceRewrite.hpp
 * @brief The quadrature substitution, as a function over a @ref TensorExpr.
 *
 * Private to the pass sources. `LaplaceTransform` applies it to a live region and
 * `FactorizationPass` applies it to a TRIAL expression, because on the canonical MP2 shape
 * neither rewrite pays on its own and both together do: the fit has no candidate until the
 * integral is a contraction operand, and the transform's numerator is a leaf until the fit has
 * made one. A veto taken on either alone refuses a rewrite the pair makes profitable, so the
 * profitability question has to be asked of the pair, which means one rewrite has to be
 * callable on an expression that is not yet in the graph.
 *
 * The two callers differ in exactly one thing, and it is a callback rather than a mode: where
 * the tensors the rewrite introduces come from. A live run declares them on the graph; a trial
 * invents ids and records the shapes, so costing a rewrite that is then declined leaves no
 * shells behind. Everything else, the recognizer, the refusals and the algebra, is one
 * implementation.
 */

#include <Einsums/ComputeGraph/Passes/LaplaceTransform.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::quadrature)

/// @brief What the rewrite needs from its caller.
struct RewriteOptions {
    /// Target accuracy the quadrature is fitted to.
    double epsilon{1.0e-5};

    /// An explicit point count, or zero to derive one from @ref epsilon.
    std::int64_t points{0};

    /// The orbital-energy vector registered under a name, or null.
    std::function<LaplaceTransform::EnergyVector const *(std::string const &)> energy;

    /// Make a tensor of the given name, type and shape, and answer with its id.
    ///
    /// A live run declares it on the graph. A trial invents an id and records the shape, which
    /// is what lets a rewrite be costed without leaving a declaration behind when it is
    /// declined.
    std::function<TensorId(std::string const &, packed_gemm::ScalarType, std::vector<std::size_t> const &)> declare;

    /// Whether to build the setup emitter. A trial that is only costed does not need one.
    bool want_setup{true};

    /// Last word before one denominator's region is spliced, or empty to take every offer.
    ///
    /// Called with the rule that was fitted and nothing rewritten yet, which is where an
    /// accuracy budget belongs: a refusal leaves the region exactly as it was and leaves no
    /// tensor behind, because nothing has been made at that point.
    std::function<bool(struct RewriteOutcome const &)> accept;
};

/// @brief What became of one tagged denominator.
struct RewriteOutcome {
    /// Whether the region was rewritten for this denominator.
    bool applied{false};

    /// Why not, phrased for a skip tally, and the specifics. Empty when @ref applied.
    std::string reason;
    std::string detail;

    /// The denominator this is about, and its name.
    TensorId    denominator{};
    std::string name;

    /// The rule that was fitted, for the approximation record. Meaningless unless applied.
    std::size_t points{0};
    double      measured{0};
    double      tolerance{0};

    /// Whether the tagged tensor is written by a chain this rewrite verified and may dissolve.
    ///
    /// A caller that owns the node set erases those writers once the region loop is over, at
    /// which point nothing reads the denominator: the rewrite that made this true dissolved its
    /// only reader. A caller costing a trial leaves them alone.
    bool dissolvable_writers{false};

    /// The setup body's label and the callback that captures it, when one was asked for.
    std::string                           setup_label;
    std::function<void(Graph &, Graph &)> emit_setup;

    /// Shapes of the tensors this rewrite invented, for a caller that has to price an
    /// expression naming tensors no graph holds.
    std::unordered_map<TensorId, std::vector<std::size_t>> shapes;

    /// Letters whose extent is a constant of the rule rather than a dimension of the problem,
    /// which for this rewrite is the quadrature index. A caller pricing the rewritten
    /// expression has to know, or it ranks every decoupled form one scale order worse than it
    /// is; see @ref search::LetterTable::observe_constant.
    std::vector<std::pair<std::string, std::size_t>> constant_letters;
};

/**
 * @brief Replace every tagged denominator in @p expr with a quadrature, in place.
 *
 * @param[in]     graph    The graph, read for tensor shapes, tags and provenance.
 * @param[in]     internal Tensors the region owns and a rewrite may dissolve.
 * @param[in,out] expr     The algebra, rewritten in place.
 * @param[in]     options  What the rewrite needs; see @ref RewriteOptions.
 * @param[in,out] claimed  Denominators already dealt with, appended to.
 * @return One outcome per denominator considered, applied or declined.
 *
 * Nothing here consults the accuracy budget: the caller does that, because a budget refusal is
 * a statement about the graph rather than about the algebra and the two callers refuse at
 * different moments.
 */
[[nodiscard]] std::vector<RewriteOutcome> rewrite_denominators(Graph &graph, std::vector<TensorId> const &internal, TensorExpr &expr,
                                                               RewriteOptions const &options, std::vector<TensorId> &claimed);

/// @brief Whether any operand of @p expr carries the denominator tag.
/// @param[in] graph The graph holding the tensors.
/// @param[in] expr  The algebra to scan.
/// @return True when at least one does.
[[nodiscard]] bool expression_carries_denominator(Graph const &graph, TensorExpr const &expr);

EINSUMS_NAMESPACE_END(compute_graph::passes::quadrature)
