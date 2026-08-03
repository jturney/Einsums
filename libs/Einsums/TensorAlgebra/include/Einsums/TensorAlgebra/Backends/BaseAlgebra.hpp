//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace einsums::tensor_algebra::detail {

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

template <size_t __I, bool ConjA, bool ConjB, typename... TargetDims, typename... LinkDims, CoreBasicTensorConcept CType, typename AValue,
          typename BValue, typename T>
void einsums_generic_target_loop(std::tuple<TargetDims...> const &target_dims, std::tuple<LinkDims...> const &link_dims,
                                 std::array<size_t, sizeof...(TargetDims)> const &C_target_strides,
                                 std::array<size_t, sizeof...(TargetDims)> const &A_target_strides,
                                 std::array<size_t, sizeof...(TargetDims)> const &B_target_strides,
                                 std::array<size_t, sizeof...(LinkDims)> const   &A_link_strides,
                                 std::array<size_t, sizeof...(LinkDims)> const &B_link_strides, size_t C_index, size_t A_index,
                                 size_t B_index, T &&C_prefactor, CType *C, T &&AB_prefactor, AValue const *A_data, BValue const *B_data) {
    if constexpr (sizeof...(TargetDims) == __I) {
        C->data()[C_index] += AB_prefactor * einsums_generic_link_loop<0, T, ConjA, ConjB>(link_dims, A_link_strides, B_link_strides,
                                                                                           A_index, B_index, A_data, B_data);
    } else {
        size_t const curr_dim = std::get<__I>(target_dims);
        size_t const A_stride = A_target_strides[__I];
        size_t const B_stride = B_target_strides[__I];
        size_t const C_stride = C_target_strides[__I];

        // Only parallelize the outermost target dimension to avoid nested OMP overhead and stack overflow.
        if constexpr (__I == 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < curr_dim; i++) {
                einsums_generic_target_loop<__I + 1, ConjA, ConjB>(
                    target_dims, link_dims, C_target_strides, A_target_strides, B_target_strides, A_link_strides, B_link_strides,
                    C_index + i * C_stride, A_index + i * A_stride, B_index + i * B_stride, std::forward<T>(C_prefactor), C,
                    std::forward<T>(AB_prefactor), A_data, B_data);
            }
        } else {
            for (size_t i = 0; i < curr_dim; i++) {
                einsums_generic_target_loop<__I + 1, ConjA, ConjB>(
                    target_dims, link_dims, C_target_strides, A_target_strides, B_target_strides, A_link_strides, B_link_strides,
                    C_index + i * C_stride, A_index + i * A_stride, B_index + i * B_stride, std::forward<T>(C_prefactor), C,
                    std::forward<T>(AB_prefactor), A_data, B_data);
            }
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

        einsums_generic_target_loop<0, ConjA, ConjB>(target_dims, link_dims, C_target_strides, A_target_strides, B_target_strides,
                                                     A_link_strides, B_link_strides, 0, 0, 0, (CDataType)C_prefactor, C,
                                                     (CDataType)AB_prefactor, A_data, B_data);
    }
}

} // namespace einsums::tensor_algebra::detail
