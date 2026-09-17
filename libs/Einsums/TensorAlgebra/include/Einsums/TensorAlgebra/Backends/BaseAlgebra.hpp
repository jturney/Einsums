//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_algebra::detail)

// A and B arrive as raw element pointers rather than tensors so the caller can
// hand over a private snapshot when an operand aliases C; the strides are
// precomputed from the original tensors and index the copy identically. See
// einsum_generic_algorithm below for why the snapshot is needed.
template <size_t __I, typename T, bool ConjA, bool ConjB, typename... LinkDims, typename AValue, typename BValue>
std::remove_cvref_t<T> einsums_generic_link_loop(std::tuple<LinkDims...> const                 &link_dims,
                                                 std::array<size_t, sizeof...(LinkDims)> const &A_link_strides,
                                                 std::array<size_t, sizeof...(LinkDims)> const &B_link_strides, size_t A_index,
                                                 size_t B_index, AValue const *A_data, BValue const *B_data) {
    if constexpr (sizeof...(LinkDims) == __I) {
        auto A_val = A_data[A_index];
        auto B_val = B_data[B_index];

        if constexpr (IsComplexV<std::remove_cvref_t<decltype(A_val)>> && ConjA) {
            A_val = std::conj(A_val);
        }
        if constexpr (IsComplexV<std::remove_cvref_t<decltype(B_val)>> && ConjB) {
            B_val = std::conj(B_val);
        }

        return A_val * B_val;
    } else {
        size_t const curr_dim = std::get<__I>(link_dims);
        size_t const A_stride = A_link_strides[__I];
        size_t const B_stride = B_link_strides[__I];

        T sum{0.0};

        // No OMP here; the link loop is always nested inside the parallel target loop.
        for (size_t i = 0; i < curr_dim; i++) {
            sum += einsums_generic_link_loop<__I + 1, T, ConjA, ConjB>(link_dims, A_link_strides, B_link_strides, A_index + i * A_stride,
                                                                       B_index + i * B_stride, A_data, B_data);
        }
        return sum;
    }
}

/// @brief The target loops, ordered for the layout and merged where they compose.
///
/// The loops are held outermost-first in @c extent / @c c_stride / ..., of which
/// the first @c outer_count entries are driven by an odometer; the innermost
/// loop is @c inner_extent steps of @c inner_c / @c inner_a / @c inner_b.
template <size_t N>
struct GenericLoopPlan {
    std::array<size_t, N> extent{};
    std::array<size_t, N> c_stride{};
    std::array<size_t, N> a_stride{};
    std::array<size_t, N> b_stride{};
    size_t                outer_count{0};
    size_t                inner_extent{1};
    size_t                inner_c{0};
    size_t                inner_a{0};
    size_t                inner_b{0};
};

/// @brief Order the target loops by C's stride and merge the ones that compose.
///
/// The loops used to run in the order the caller spelled C's indices, outermost
/// first, which puts C's LAST axis - its widest stride - on the innermost loop.
/// On a first-index-fastest layout that walks C, A and B a full row apart per
/// step, wider than a page, so the hardware prefetcher never engages and every
/// element is a demand miss. On a 200^3 double contraction that cost 3.2x
/// against the same loops ordered for the layout, and 5.4x for the one
/// permutation whose page working set also outgrew the TLB.
///
/// C decides the order because C is written: a scattered store pays
/// read-for-ownership on top of the miss, which a scattered load does not.
///
/// Two adjacent loops merge when the outer step is exactly one full sweep of the
/// inner one in C, A and B alike. A zero stride composes with a zero stride,
/// which is how an index absent from an operand folds in. An elementwise
/// contraction merges all the way down to a single flat sweep.
///
/// Only the TARGET loops move. The link loops keep the caller's order, so every
/// output element accumulates its terms in the sequence it always did and the
/// results stay bit-identical rather than merely close.
template <size_t N>
GenericLoopPlan<N> plan_generic_target_loops(std::array<size_t, N> const &dims, std::array<size_t, N> const &cs,
                                             std::array<size_t, N> const &as, std::array<size_t, N> const &bs) {
    GenericLoopPlan<N> plan;

    if constexpr (N == 0) {
        return plan;
    } else {
        std::array<size_t, N> order;
        for (size_t i = 0; i < N; i++) {
            order[i] = i;
        }
        // Widest C stride outermost. Stable, so equal strides keep the caller's
        // order and a contraction already spelled for the layout is left alone.
        std::stable_sort(order.begin(), order.end(), [&](size_t l, size_t r) { return cs[l] > cs[r]; });

        for (size_t k = 0; k < N; k++) {
            plan.extent[k]   = dims[order[k]];
            plan.c_stride[k] = cs[order[k]];
            plan.a_stride[k] = as[order[k]];
            plan.b_stride[k] = bs[order[k]];
        }

        size_t inner_e = plan.extent[N - 1];
        size_t inner_c = plan.c_stride[N - 1];
        size_t inner_a = plan.a_stride[N - 1];
        size_t inner_b = plan.b_stride[N - 1];
        size_t first   = N - 1;
        while (first > 0) {
            size_t const o = first - 1;
            if (plan.c_stride[o] != inner_c * inner_e || plan.a_stride[o] != inner_a * inner_e || plan.b_stride[o] != inner_b * inner_e) {
                break;
            }
            inner_e *= plan.extent[o];
            first = o;
        }

        plan.outer_count  = first;
        plan.inner_extent = inner_e;
        plan.inner_c      = inner_c;
        plan.inner_a      = inner_a;
        plan.inner_b      = inner_b;
        return plan;
    }
}

/// @brief Walk the planned target loops, innermost loop tight.
///
/// The outer loops are flattened into one counter so the parallel region always
/// has the full outer trip count to divide, instead of whatever the first
/// dimension happened to be.
template <bool ConjA, bool ConjB, size_t N, typename... LinkDims, CoreBasicTensorConcept CType, typename AValue, typename BValue,
          typename T>
void einsums_generic_target_walk(GenericLoopPlan<N> const &plan, std::tuple<LinkDims...> const &link_dims,
                                 std::array<size_t, sizeof...(LinkDims)> const &A_link_strides,
                                 std::array<size_t, sizeof...(LinkDims)> const &B_link_strides, T const AB_prefactor, CType *C,
                                 AValue const *A_data, BValue const *B_data) {
    auto        *C_data = C->data();
    size_t const n      = plan.inner_extent;
    size_t const cS = plan.inner_c, aS = plan.inner_a, bS = plan.inner_b;

    size_t const link_total = extent_product(link_dims);

    if (plan.outer_count == 0) {
        // Everything merged into one sweep, so the innermost loop is the only
        // place left to take the parallelism from.
        EINSUMS_OMP_PARALLEL_FOR_IF(generic_walk_wants_threads(n, link_total))
        for (size_t i = 0; i < n; i++) {
            C_data[i * cS] += AB_prefactor * einsums_generic_link_loop<0, T, ConjA, ConjB>(link_dims, A_link_strides, B_link_strides,
                                                                                           i * aS, i * bS, A_data, B_data);
        }
        return;
    }

    size_t outer_total = 1;
    for (size_t k = 0; k < plan.outer_count; k++) {
        outer_total *= plan.extent[k];
    }

    EINSUMS_OMP_PARALLEL_FOR_IF(generic_walk_wants_threads(outer_total * n, link_total))
    for (size_t o = 0; o < outer_total; o++) {
        size_t rem = o, offC = 0, offA = 0, offB = 0;
        // Low-order digit is the loop nearest the inner one, so consecutive o
        // stay as close together in memory as the ordering allows.
        for (size_t k = plan.outer_count; k-- > 0;) {
            size_t const i = rem % plan.extent[k];
            rem /= plan.extent[k];
            offC += i * plan.c_stride[k];
            offA += i * plan.a_stride[k];
            offB += i * plan.b_stride[k];
        }

        for (size_t i = 0; i < n; i++) {
            C_data[offC + i * cS] +=
                AB_prefactor * einsums_generic_link_loop<0, T, ConjA, ConjB>(link_dims, A_link_strides, B_link_strides, offA + i * aS,
                                                                             offB + i * bS, A_data, B_data);
        }
    }
}

template <bool ConjA, bool ConjB, typename... CUniqueIndices, typename... AUniqueIndices, typename... BUniqueIndices,
          typename... LinkUniqueIndices, typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims,
          typename... LinkDims, typename... TargetPositionInC, typename... LinkPositionInLink, typename CType, CoreBasicTensorConcept AType,
          CoreBasicTensorConcept BType>
    requires(CoreBasicTensorConcept<CType> || (!TensorConcept<CType> && sizeof...(CIndices) == 0))
void einsum_generic_algorithm(std::tuple<CUniqueIndices...> const &C_unique, std::tuple<AUniqueIndices...> const & /*A_unique*/,
                              std::tuple<BUniqueIndices...> const & /*B_unique*/, std::tuple<LinkUniqueIndices...> const &link_unique,
                              std::tuple<CIndices...> const & /*C_indices*/, std::tuple<AIndices...> const               &A_indices,
                              std::tuple<BIndices...> const &B_indices, std::tuple<TargetDims...> const &target_dims,
                              std::tuple<LinkDims...> const &link_dims, std::tuple<TargetPositionInC...> const &target_position_in_C,
                              std::tuple<LinkPositionInLink...> const & /*link_position_in_link*/, ValueTypeT<CType> const C_prefactor,
                              CType                                                                         *C,
                              std::conditional_t<(sizeof(typename AType::ValueType) > sizeof(typename BType::ValueType)),
                                                 typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                              AType const &A, BType const &B) {
    LabeledSection0();

    using ADataType        = typename AType::ValueType;
    using BDataType        = typename BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    auto const target_position_in_A = find_type_with_position(C_unique, A_indices);
    auto const target_position_in_B = find_type_with_position(C_unique, B_indices);
    auto const link_position_in_A   = find_type_with_position(link_unique, A_indices);
    auto const link_position_in_B   = find_type_with_position(link_unique, B_indices);

    auto const A_target_strides = tensor_algebra::get_stride_for(A, target_position_in_A, C_unique);
    auto const B_target_strides = tensor_algebra::get_stride_for(B, target_position_in_B, C_unique);
    auto const A_link_strides   = tensor_algebra::get_stride_for(A, link_position_in_A, link_unique);
    auto const B_link_strides   = tensor_algebra::get_stride_for(B, link_position_in_B, link_unique);

    // The dispatcher lets C alias an operand whose index list is IDENTICAL to
    // C's, on the grounds that each element is then read immediately before its
    // own overwrite. True of the elementwise kernels, false here: both branches
    // below clear or rescale C before reading anything, so an aliased operand
    // would be read back already zeroed and the result would be silently wrong
    // ("ij <- ij ; j" with C aliasing A produced all zeros). Snapshot any
    // operand that overlaps C and read the copy; the precomputed strides index
    // it identically, so the loops and their summation order are untouched.
    ADataType const       *A_data = A.data();
    BDataType const       *B_data = B.data();
    std::vector<ADataType> A_snapshot;
    std::vector<BDataType> B_snapshot;
    if constexpr (CoreBasicTensorConcept<CType> && IsTensorV<CType>) {
        auto const span_of = [](auto const &t) -> size_t {
            using TT    = std::remove_cvref_t<decltype(t)>;
            size_t last = 0;
            for (size_t d = 0; d < TT::Rank; d++) {
                if (t.dim(d) == 0) {
                    return 0;
                }
                last += (t.dim(d) - 1) * t.stride(d);
            }
            return last + 1;
        };
        // Byte intervals, since A, B and C may have different element types.
        // Deliberately conservative: unlike the dispatcher's guard this only
        // decides whether to take a copy, so a false positive costs an
        // allocation rather than a spurious throw.
        auto const *c_lo     = reinterpret_cast<char const *>(C->data());
        auto const *c_hi     = c_lo + span_of(*C) * sizeof(CDataType);
        auto const  overlaps = [&](auto const &t, size_t elem_size) {
            auto const *lo = reinterpret_cast<char const *>(t.data());
            return lo < c_hi && c_lo < lo + span_of(t) * elem_size;
        };
        if (span_of(*C) != 0) {
            if (overlaps(A, sizeof(ADataType))) {
                A_snapshot.assign(A_data, A_data + span_of(A));
                A_data = A_snapshot.data();
            }
            if (overlaps(B, sizeof(BDataType))) {
                B_snapshot.assign(B_data, B_data + span_of(B));
                B_data = B_snapshot.data();
            }
        }
    }

    if constexpr (sizeof...(CIndices) == 0 && sizeof...(LinkDims) != 0) {
        if (C_prefactor == CDataType{0.0}) {
            *C = CDataType{0.0};
        } else {
            *C *= C_prefactor;
        }

        *C += AB_prefactor *
              einsums_generic_link_loop<0, CDataType, ConjA, ConjB>(link_dims, A_link_strides, B_link_strides, 0, 0, A_data, B_data);
    } else {
        auto const C_target_strides = tensor_algebra::get_stride_for(*C, target_position_in_C, C_unique);

        if (C_prefactor == CDataType{0.0}) {
            C->zero();
        } else {
            *C *= C_prefactor;
        }

        constexpr size_t NT = sizeof...(TargetDims);
        auto const       target_extents =
            std::apply([](auto const &...d) { return std::array<size_t, NT>{static_cast<size_t>(d)...}; }, target_dims);

        auto const plan = plan_generic_target_loops<NT>(target_extents, C_target_strides, A_target_strides, B_target_strides);

        einsums_generic_target_walk<ConjA, ConjB>(plan, link_dims, A_link_strides, B_link_strides, (CDataType)AB_prefactor, C, A_data,
                                                  B_data);
    }
}

EINSUMS_NAMESPACE_END(tensor_algebra::detail)
