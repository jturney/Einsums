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

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
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
 * @brief ``lhs * rhs`` over two type-erased prefactors.
 *
 * Needed wherever a pass collapses several scaled operations into one and has to
 * carry the scalars it removed: ContractionPlanning folds a chain of
 * contractions into a tree of GEMMs and owes the product of the prefactors the
 * chain carried. The arithmetic is done in ``complex<double>`` and the result
 * narrowed back, so precision is lost only where the inputs were already single
 * and an imaginary part only appears where one of the inputs had one.
 */
[[nodiscard]] inline PrefactorScalar multiply_prefactors(PrefactorScalar const &lhs, PrefactorScalar const &rhs) {
    // Alternative order is float, double, complex<float>, complex<double>: index
    // >= 2 is complex, and an even index is single precision.
    bool const complex_result = lhs.index() >= 2 || rhs.index() >= 2;
    bool const single_result  = (lhs.index() % 2 == 0) && (rhs.index() % 2 == 0);

    std::complex<double> const product = as<std::complex<double>>(lhs) * as<std::complex<double>>(rhs);
    if (complex_result) {
        if (single_result) {
            return PrefactorScalar{std::complex<float>{static_cast<float>(product.real()), static_cast<float>(product.imag())}};
        }
        return PrefactorScalar{product};
    }
    if (single_result) {
        return PrefactorScalar{static_cast<float>(product.real())};
    }
    return PrefactorScalar{product.real()};
}

/**
 * @brief Whether @p v is exactly a power of two, sign included.
 *
 * Multiplying by such a value only shifts an exponent, so the arithmetic is
 * bit-for-bit invariant under moving the factor around: `(r*x)*y` equals
 * `x*(r*y)`, and `r*(a+b)` equals `r*a + r*b`. That is what lets CSE move a
 * ratio off a producer and onto its readers, and DistributiveFactoring reuse one
 * assembled sum for a proportional one, without changing any result.
 *
 * Ratios that are exactly representable but not powers of two are declined:
 * they would make the result depend on which of two proportional nodes the pass
 * happened to keep, a bad property for a pass that runs by default.
 */
[[nodiscard]] inline bool is_exact_power_of_two(double v) {
    if (!std::isfinite(v) || v == 0.0) {
        return false;
    }
    int          exp = 0;
    double const m   = std::frexp(v, &exp);
    return m == 0.5 || m == -0.5;
}

/**
 * @brief Fold @p v into the running hash @p h.
 *
 * The golden-ratio mix every pass that keys an ``unordered_map`` on a composite
 * struct needs. Each of them used to spell its own, and two of those spellings
 * combined terms with a bare XOR, which is commutative: a key holding index
 * lists hashed the same under any permutation of them, so exactly the groups
 * these passes exist to tell apart landed in one bucket.
 *
 * Only bucketing depends on this. Two keys that compare equal still hash equal,
 * because the fold is a pure function of the values fed to it in order.
 */
template <typename T>
inline void hash_combine(std::size_t &h, T const &v) {
    h ^= std::hash<T>{}(v) + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (h << 6) + (h >> 2);
}

/// Fold every element of @p range into @p h, in order. @see hash_combine
template <typename Range>
inline void hash_range(std::size_t &h, Range const &range) {
    for (auto const &element : range) {
        hash_combine(h, element);
    }
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
 * @brief Interference gate shared by the passes that collapse a run of nodes
 *        into one (DistributiveFactoring, LinearCombinationContractionFolding,
 *        GEMMBatching).
 *
 * These passes move a combined op into the FIRST member's slot, because position
 * is program order in this IR and the combined node has to stay ahead of every
 * consumer of a member's output. That is sound only if nothing between the first
 * and last member disturbs what the group reads or writes. Returns true when
 * some node strictly between positions @p first and @p last (exclusive), other
 * than the group members flagged in @p is_member:
 *   - writes anything the group reads or writes (clobbering a factor, or a
 *     partial result, mid-collapse),
 *   - reads anything the group writes (observing a partial result before it is
 *     complete), or
 *   - when @p reject_control_flow is set, is a control-flow node whose hidden
 *     sub-graph I/O cannot be inspected.
 *
 * @p reject_control_flow lets DistributiveFactoring and GEMMBatching disqualify
 * a span containing a Loop/Conditional while
 * LinearCombinationContractionFolding keeps its existing (control-flow-agnostic)
 * behavior.
 *
 * @param[in] nodes               The graph's node list.
 * @param[in] first               Position of the first group member.
 * @param[in] last                Position of the last group member.
 * @param[in] is_member           Membership mask over @p nodes.
 * @param[in] writes              Every tensor the group writes.
 * @param[in] reads               Every tensor the group reads.
 * @param[in] reject_control_flow Whether a control-flow node in the span disqualifies it.
 */
[[nodiscard]] inline bool span_interferes(std::vector<Node> const &nodes, std::size_t first, std::size_t last,
                                          std::vector<bool> const &is_member, std::unordered_set<TensorId> const &writes,
                                          std::unordered_set<TensorId> const &reads, bool reject_control_flow) {
    for (std::size_t n = first + 1; n < last; ++n) {
        if (is_member[n]) {
            continue;
        }
        Node const &other = nodes[n];
        if (reject_control_flow && is_control_flow(other.kind)) {
            return true;
        }
        for (auto const out : other.outputs) {
            if (writes.count(out) != 0 || reads.count(out) != 0) {
                return true;
            }
        }
        for (auto const in : other.inputs) {
            if (writes.count(in) != 0) {
                return true;
            }
        }
    }
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
