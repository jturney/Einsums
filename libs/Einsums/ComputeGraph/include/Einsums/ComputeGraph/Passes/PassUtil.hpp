//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <complex>
#include <cstddef>
#include <unordered_set>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief The live ``beta`` of an Axpby node, or null when @p nd is not an Axpby
 *        or carries no descriptor (a pass-built node) and beta is unknowable.
 *
 * Prefer the shared params over the descriptor snapshot: a pass that folded a
 * scale into this axpby wrote beta through the params handle, and that is the
 * value the executor will read.
 */
[[nodiscard]] inline PrefactorScalar const *axpby_beta(Node const &nd) {
    // A pass-built Axpby with no descriptor returns null, which every caller
    // already treats conservatively.
    if (nd.kind != OpKind::Axpby) {
        return nullptr;
    }
    auto const *ad = std::get_if<AxpbyDescriptor>(&nd.op_data);
    if (ad == nullptr) {
        return nullptr;
    }
    return &live_beta(*ad);
}

/**
 * @brief ``pf * a``, preserving the PrefactorScalar's element type.
 *
 * Shared by the passes that push a real scalar into an existing prefactor
 * (ScaleAbsorption's operand fold, CSE's proportional-duplicate merge).
 */
[[nodiscard]] inline PrefactorScalar scale_prefactor(PrefactorScalar const &pf, double a) {
    return std::visit(
        [a](auto x) -> PrefactorScalar {
            using T = decltype(x);
            return PrefactorScalar{x * static_cast<T>(a)};
        },
        pf);
}

/**
 * @brief True when @p nd overwrites its destination without reading it.
 *
 * A prefactor-bearing op (Einsum/Permute/BatchedGemm) whose destination
 * prefactor is zero writes C fresh, making any prior write to that tensor dead.
 * Shared by ScaleAbsorption (drop a scale the next op overwrites) and CSE
 * (only a pure-overwrite result is a safe common-subexpression survivor).
 */
[[nodiscard]] inline bool pure_overwrite(Node const &nd) {
    // Axpby: Y = alpha*X + beta*Y overwrites Y exactly when beta == 0.
    if (auto const *beta = axpby_beta(nd)) {
        return is_zero(*beta);
    }
    // The LIVE prefactor, not the at-capture snapshot beside it. Every writer in the library
    // currently keeps the two in step -- ScaleAbsorption::apply_fold, CSE::fold_reader,
    // ElementWiseFusion and Graph::update_prefactors each write both, deliberately -- so this
    // reads the same value either way TODAY. It is spelled this way so that stays true without
    // depending on every future writer remembering: what this predicate is asked about is what
    // the next execute() will do, and that is the params block by definition.
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return is_zero(live_c_prefactor(*e));
    }
    // A tiled einsum keeps its destination prefactor in the shared params rather
    // than in an EinsumDescriptor, so it needs its own arm. Without one it fell
    // through to `false` here and in reads_destination below, and an op that
    // answers "no" to both looks neither overwriting nor accumulating.
    if (auto const *t = std::get_if<TiledEinsumDescriptor>(&nd.op_data)) {
        return t->params && is_zero(t->params->c_pf);
    }
    // Permute has no live accessor (its snapshot and its params block disagree on type; see the
    // note in Node.hpp), so the preference is spelled out here.
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->params != nullptr ? is_zero(p->params->beta) : p->beta == 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta == std::complex<double>{0.0, 0.0};
    }
    return false;
}

/**
 * @brief True when @p nd reads its destination (accumulates into it).
 *
 * The always-accumulating kinds (Scale/ElementTransform), an Axpby with a
 * non-zero beta, or a
 * prefactor-bearing op with a non-zero destination prefactor. Used by
 * LoopInvariantHoisting to refuse to hoist a self-modifying update out of a loop.
 *
 * @note This is NOT the negation of pure_overwrite: an op with no prefactor
 *       descriptor (e.g. Gemm, Dot) returns false from both.
 */
[[nodiscard]] inline bool reads_destination(Node const &nd) {
    switch (nd.kind) {
    case OpKind::Scale:
    case OpKind::ElementTransform:
        return true;
    default:
        break;
    }
    // Axpby reads its destination only when beta != 0. It used to be lumped in
    // with the always-accumulating ops above, which predates AxpbyDescriptor
    // carrying beta and made a pure-overwrite `Y = alpha*X` look self-modifying.
    // LoopInvariantHoisting then refused to hoist it, even though the identical
    // pure-overwrite Permute (checked precisely below) hoists fine -- so an
    // invariant `L = alpha*g` rebuilt every iteration stayed in the loop.
    // A pass-built Axpby with no descriptor stays conservative (true).
    if (nd.kind == OpKind::Axpby) {
        auto const *beta = axpby_beta(nd);
        return beta == nullptr || !is_zero(*beta);
    }
    // The LIVE prefactor; see the note in pure_overwrite above.
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return !is_zero(live_c_prefactor(*e));
    }
    // Tiled einsum: same rule as the dense one, read through the shared params.
    // A descriptor with no params is unknowable, so assume it accumulates; that
    // only costs a missed hoist, where the other answer loses the accumulation.
    if (auto const *t = std::get_if<TiledEinsumDescriptor>(&nd.op_data)) {
        return t->params == nullptr || !is_zero(t->params->c_pf);
    }
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->params != nullptr ? !is_zero(p->params->beta) : p->beta != 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta != std::complex<double>{0.0, 0.0};
    }
    return false;
}

/**
 * @brief Interference gate shared by the reassociation passes
 *        (DistributiveFactoring, LinearCombinationContractionFolding).
 *
 * These passes move a combined op into the first member's slot, which is sound
 * only if nothing between the first and last member disturbs the partial sum or
 * a factor. Returns true when some node strictly between positions @p first and
 * @p last (exclusive), other than the group members flagged in @p is_member:
 *   - writes @p output_id or any tensor in @p operand_ids (clobbering the partial
 *     sum or a factor mid-fold),
 *   - reads @p output_id (observing the partial sum before it is complete), or
 *   - when @p reject_control_flow is set, is a control-flow node whose hidden
 *     sub-graph I/O cannot be inspected.
 *
 * @p reject_control_flow lets DistributiveFactoring disqualify a span containing
 * a Loop/Conditional while LinearCombinationContractionFolding keeps its existing
 * (control-flow-agnostic) behavior.
 */
[[nodiscard]] inline bool span_interferes(std::vector<Node> const &nodes, std::size_t first, std::size_t last,
                                          std::vector<bool> const &is_member, TensorId output_id,
                                          std::unordered_set<TensorId> const &operand_ids, bool reject_control_flow) {
    for (std::size_t n = first + 1; n < last; ++n) {
        if (is_member[n]) {
            continue;
        }
        Node const &other = nodes[n];
        if (reject_control_flow && is_control_flow(other.kind)) {
            return true;
        }
        for (auto const out : other.outputs) {
            if (out == output_id || operand_ids.count(out) != 0) {
                return true;
            }
        }
        for (auto const in : other.inputs) {
            if (in == output_id) {
                return true;
            }
        }
    }
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
