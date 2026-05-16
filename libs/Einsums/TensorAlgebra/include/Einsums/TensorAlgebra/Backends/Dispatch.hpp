//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#ifndef DOXYGEN

#    include <Einsums/Config.hpp>

#    include <Einsums/Concepts/SubscriptChooser.hpp>
#    include <Einsums/Concepts/TensorConcepts.hpp>
#    include <Einsums/Errors/Error.hpp>
#    include <Einsums/LinearAlgebra.hpp>
#    include <Einsums/Logging.hpp>
#    include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#    include <Einsums/Print.hpp>
#    include <Einsums/Profile.hpp>
#    include <Einsums/Tensor/BlockTensor.hpp>
#    include <Einsums/Tensor/Tensor.hpp>
#    include <Einsums/TensorAlgebra/Backends/BaseAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/BlockAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/BlockTileAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Backends/GenericAlgorithm.hpp>
#    include <Einsums/TensorAlgebra/Backends/TileAlgebra.hpp>
#    include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#    include <Einsums/TensorAlgebra/Permute.hpp> // Required for einsum_do_sort_gemm
#    include <Einsums/TensorBase/Common.hpp>

#    include <algorithm>
#    include <cmath>
#    include <cstddef>
#    include <memory>
#    include <stdexcept>
#    include <string>
#    include <tuple>
#    include <type_traits>
#    include <utility>

#    if defined(EINSUMS_USE_CATCH2)
#        include <catch2/catch_all.hpp>
#    endif

namespace einsums::tensor_algebra {
namespace detail {

template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B);

/**
 * @brief Perform runtime checks for the tensor dimensions.
 */
template <bool OnlyUseGenericAlgorithm, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
void einsum_runtime_check(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                          BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                          std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                          BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;
    bool             runtime_indices_abort{false};

    for_sequence<ARank>([&](auto a) {
        size_t dimA = A.dim(a);
        for_sequence<BRank>([&](auto b) {
            size_t dimB = B.dim(b);
            if (std::get<a>(A_indices).letter == std::get<b>(B_indices).letter) {
                if (dimA != dimB) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
        for_sequence<CRank>([&](auto c) {
            size_t dimC;
            if constexpr (IsTensorV<CType>) {
                dimC = C->dim(c);
            } else {
                dimC = 0;
            }
            if (std::get<a>(A_indices).letter == std::get<c>(C_indices).letter) {
                if (dimA != dimC) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
    });
    for_sequence<BRank>([&](auto b) {
        size_t dimB = B.dim(b);
        for_sequence<CRank>([&](auto c) {
            size_t dimC;
            if constexpr (IsTensorV<CType>) {
                dimC = C->dim(c);
            } else {
                dimC = 0;
            }
            if (std::get<b>(B_indices).letter == std::get<c>(C_indices).letter) {
                if (dimB != dimC) {
#    if !defined(EINSUMS_IS_TESTING)
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));
#    endif
                    runtime_indices_abort = true;
                }
            }
        });
    });

    if (runtime_indices_abort) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Inconsistent dimensions found!");
    }
}

/**
 * @brief Perform the generic fallback algorithm.
 *
 * This will either call einsum_generic_algorithm or einsum_special_dispatch, depending on whether the tensors
 * have special dispatching. The template argument will be passed onto einsum_special_dispatch.
 */
template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
AlgorithmChoice einsum_generic_default(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                                       BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                                       std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                                       BType const &B) {
    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    if constexpr (IsAlgebraTensorV<AType> && IsAlgebraTensorV<BType> &&
                  (IsAlgebraTensorV<CType> || !IsTensorV<CType>)&&(!IsBasicTensorV<AType> || !IsBasicTensorV<BType> ||
                                                                   (!IsBasicTensorV<CType> && IsTensorV<CType>))) {
        if constexpr (!DryRun) {
            einsum_special_dispatch<OnlyUseGenericAlgorithm, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices,
                                                                           B);
        }
        Tensor<typename AType::ValueType, AType::Rank> dry_A;
        Tensor<typename BType::ValueType, BType::Rank> dry_B;
        if constexpr (TensorConcept<CType>) {
            Tensor<typename CType::ValueType, CType::Rank> dry_C;
            return einsum<OnlyUseGenericAlgorithm, true, ConjA, ConjB>(C_prefactor, C_indices, &dry_C, AB_prefactor, A_indices, dry_A,
                                                                       B_indices, dry_B);
        } else {
            return einsum<OnlyUseGenericAlgorithm, true, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, dry_A, B_indices,
                                                                       dry_B);
        }
    } else {
        constexpr auto A_unique              = UniqueT<std::tuple<AIndices...>>();
        constexpr auto B_unique              = UniqueT<std::tuple<BIndices...>>();
        constexpr auto C_unique              = UniqueT<std::tuple<CIndices...>>();
        constexpr auto linksAB               = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
        constexpr auto links                 = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
        constexpr auto link_unique           = CUniqueT<decltype(links)>();
        constexpr auto link_position_in_A    = detail::find_type_with_position(link_unique, A_indices);
        constexpr auto link_position_in_link = detail::find_type_with_position(link_unique, links);
        constexpr auto target_position_in_C  = detail::find_type_with_position(C_unique, C_indices);
        auto           unique_target_dims    = detail::get_dim_for(*C, detail::unique_find_type_with_position(C_unique, C_indices));
        auto           unique_link_dims      = detail::get_dim_for(A, link_position_in_A);

        EINSUMS_LOG_TRACE("Performing the generic algorithm.");

        if constexpr (!DryRun) {
            // Attempt the packed GEMM backend for BasicTensor contractions that
            // were not handled by any BLAS specialisation.  The backend is
            // skipped when OnlyUseGenericAlgorithm is true (e.g. during
            // correctness-regression tests of the generic loops themselves).
            if constexpr (!OnlyUseGenericAlgorithm && IsBasicTensorV<AType> && IsBasicTensorV<BType> &&
                          std::is_same_v<typename AType::ValueType, typename BType::ValueType> && TensorConcept<CType> &&
                          std::is_same_v<ValueTypeT<CType>, typename AType::ValueType> && TensorRank<CType> >= 2) {
                if (einsums::packed_gemm::try_packed_gemm<ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices,
                                                                        B)) {
                    return PACKED_GEMM;
                }
            }
            einsum_generic_algorithm<ConjA, ConjB>(C_unique, A_unique, B_unique, link_unique, C_indices, A_indices, B_indices,
                                                   unique_target_dims, unique_link_dims, target_position_in_C, link_position_in_link,
                                                   C_prefactor, C, AB_prefactor, A, B);
        }
        return GENERIC;
    }
}

/**
 * @brief Check to see if the index pack is Hadamard.
 *
 * This checks to see if there are duplicate indices within the pack.
 */
template <typename... Indices>
constexpr bool einsum_is_hadamard_found() {
    constexpr auto unique = UniqueT<std::tuple<Indices...>>();
    return std::tuple_size_v<std::tuple<Indices...>> != std::tuple_size_v<decltype(unique)>;
}

/**
 * @brief Check to see if any of the index packs are Hadamard.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_all_hadamard_found(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &,
                                            std::tuple<BIndices...> const &) {
    return einsum_is_hadamard_found<CIndices...>() || einsum_is_hadamard_found<AIndices...>() || einsum_is_hadamard_found<BIndices...>();
}

/**
 * @brief Checks to see if the indices passed can be turned into a dot product.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_dot_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {

    constexpr auto A_exactly_matches_B = same_indices<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    return sizeof...(CIndices) == 0 && A_exactly_matches_B;
}

/**
 * @brief Checks to see if the indices passed can be turned into a direct product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_direct_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_unique = UniqueT<std::tuple<AIndices...>>();
    constexpr auto B_unique = UniqueT<std::tuple<BIndices...>>();
    constexpr auto C_unique = UniqueT<std::tuple<CIndices...>>();
    constexpr auto C_exactly_matches_A =
        sizeof...(CIndices) == sizeof...(AIndices) && same_indices<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto C_exactly_matches_B =
        sizeof...(CIndices) == sizeof...(BIndices) && same_indices<std::tuple<CIndices...>, std::tuple<BIndices...>>();

    return C_exactly_matches_A && C_exactly_matches_B && !ConjA && !ConjB;
}

/**
 * @brief Checks to see if the indices passed can be turned into an outer product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_outer_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                       = std::tuple<AIndices...>();
    constexpr auto B_indices                       = std::tuple<BIndices...>();
    constexpr auto C_indices                       = std::tuple<CIndices...>();
    constexpr auto linksAB                         = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto C_unique                        = UniqueT<std::tuple<CIndices...>>();
    constexpr auto target_position_in_A            = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B            = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto contiguous_target_position_in_A = detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B = detail::contiguous_positions(target_position_in_B);
    constexpr auto A_target_position_in_C          = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C          = detail::find_type_with_position(B_indices, C_indices);

    constexpr bool condition =
        std::tuple_size_v<decltype(linksAB)> == 0 && contiguous_target_position_in_A && contiguous_target_position_in_B;

    if constexpr (condition) {
        constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;

        constexpr bool straight_conjugation = !ConjA && !swap_AB;
        constexpr bool swapped_conjugation  = !ConjB && swap_AB;
        return straight_conjugation || swapped_conjugation;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs the outer product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
bool einsum_do_outer_product(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                             BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = detail::find_type_with_position(B_indices, C_indices);

    EINSUMS_LOG_TRACE("outer_product");
    if (!A.is_totally_vectorable() || !B.is_totally_vectorable()) {
        EINSUMS_LOG_TRACE("do not have full view of underlying data A {} B{}", !A.is_totally_vectorable(), !B.is_totally_vectorable());
        return false;
    }

    constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;
    EINSUMS_LOG_TRACE("swap_AB {}", swap_AB);

    Dim<2>    dC;
    Stride<2> sC;
    dC[0] = product_dims(A_target_position_in_C, *C);
    dC[1] = product_dims(B_target_position_in_C, *C);
    sC[0] = last_stride(A_target_position_in_C, *C);
    sC[1] = last_stride(B_target_position_in_C, *C);
    if constexpr (CRank == 2) {
        if (C->stride(0) != 1 && C->stride(1) != 1) {
            EINSUMS_LOG_TRACE("do not have gerable view of C.");
            return false;
        }
    } else {
        if (C->is_row_major()) {
            size_t cutoff_stride = std::max(sC[0], sC[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = CRank - 1; i >= 0; i--) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (C->stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (C->stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gerable view of C.");
                    return false;
                }
                check_stride *= C->dim(i);
            }
        } else {
            size_t cutoff_stride = std::max(sC[0], sC[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = 0; i < CRank; i++) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (C->stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (C->stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gerable view of C.");
                    return false;
                }
                check_stride *= C->dim(i);
            }
        }
    }

    if constexpr (swap_AB) {
        std::swap(dC[0], dC[1]);
    }

    if constexpr (DryRun) {
        return true;
    }

    TensorView<CDataType, 2> tC{*C, dC};
    if (C_prefactor != CDataType{1.0}) {
        EINSUMS_LOG_TRACE("scaling C");
        linear_algebra::scale(C_prefactor, C);
    }
    try {
        EINSUMS_LOG_TRACE("calling ger");
        if constexpr (swap_AB) {
            if constexpr (ConjA) {
                linear_algebra::gerc(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
            } else {
                linear_algebra::ger(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
            }
        } else {
            if constexpr (ConjB) {
                linear_algebra::gerc(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
            } else {
                linear_algebra::ger(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
            }
        }
    } catch (std::runtime_error &e) {
#    if defined(EINSUMS_SHOW_WARNING)
        EINSUMS_LOG_WARN("Optimized outer product failed. Likely from a non-contiguous "
                         "TensorView. Attempting to perform generic algorithm.");
#    endif
        if constexpr (IsComplexV<CDataType>) {
            if (C_prefactor == CDataType{0.0, 0.0}) {
#    if defined(EINSUMS_SHOW_WARNING)
                EINSUMS_LOG_WARN("WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor,
                                 C->name());
#    endif
            } else {
                linear_algebra::scale(CDataType{1.0, 0.0} / C_prefactor, C);
            }
        } else {
            if (C_prefactor == CDataType{0.0}) {
#    if defined(EINSUMS_SHOW_WARNING)
                EINSUMS_LOG_WARN("WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor,
                                 C->name());
#    endif
            } else {
                linear_algebra::scale(CDataType{1.0} / C_prefactor, C);
            }
        }
        return false;
    }
    return true;
}

/**
 * @brief Checks to see if the indices passed can be turned into a matrix-vector product where the second pack contains the indices for the
 * matrix.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_matrix_vector(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                           = std::tuple<AIndices...>();
    constexpr auto B_indices                           = std::tuple<BIndices...>();
    constexpr auto C_indices                           = std::tuple<CIndices...>();
    constexpr auto C_unique                            = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB                             = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                               = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto link_unique                         = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A                  = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B                  = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A                = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B                = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C              = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C              = detail::find_type_with_position(B_indices, C_indices);
    constexpr auto contiguous_link_position_in_A       = detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B       = detail::contiguous_positions(link_position_in_B);
    constexpr auto contiguous_target_position_in_A     = detail::contiguous_positions(target_position_in_A);
    constexpr auto same_ordering_link_position_in_AB   = detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr bool condition = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                               same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                               !same_ordering_target_position_in_CB && std::tuple_size_v<decltype(B_target_position_in_C)> == 0 && !ConjB;

    if constexpr (condition) {
        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
        return transpose_A || !ConjA;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs a matrix-vector product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
bool einsum_do_matrix_vector(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                             BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto C_unique               = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB                = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                  = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto link_unique            = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A     = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B     = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A   = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);

    constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;

    if (!B.is_totally_vectorable() || !C->is_totally_vectorable()) {
        return false;
    }

    Dim<2>    dA;
    Dim<1>    dB, dC;
    Stride<2> sA;
    Stride<1> sB, sC;

    dA[0] = product_dims(A_target_position_in_C, *C);
    dA[1] = product_dims(link_position_in_A, A);
    sA[0] = last_stride(target_position_in_A, A);
    sA[1] = last_stride(link_position_in_A, A);

    if constexpr (ARank == 2) {
        if (A.stride(0) != 1 && A.stride(1) != 1) {
            EINSUMS_LOG_TRACE("do not have gemvable view of A.");
            return false;
        }
    } else {
        if (A.is_row_major()) {
            size_t cutoff_stride = std::max(sA[0], sA[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = ARank - 1; i >= 0; i--) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (A.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (A.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemvable view of A.");
                    return false;
                }
                check_stride *= A.dim(i);
            }
        } else {
            size_t cutoff_stride = std::max(sA[0], sA[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = 0; i < ARank; i++) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (A.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (A.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemvable view of A.");
                    return false;
                }
                check_stride *= A.dim(i);
            }
        }
    }

    if constexpr (DryRun) {
        return true;
    }

    if constexpr (transpose_A) {
        std::swap(dA[0], dA[1]);
        std::swap(sA[0], sA[1]);
    }

    dB[0] = product_dims(link_position_in_B, B);
    sB[0] = last_stride(link_position_in_B, B);

    dC[0] = product_dims(A_target_position_in_C, *C);
    sC[0] = last_stride(A_target_position_in_C, *C);

    TensorView<ADataType, 2> const tA{const_cast<AType &>(A), dA, sA};
    TensorView<BDataType, 1> const tB{const_cast<BType &>(B), dB, sB};
    TensorView<CDataType, 1>       tC{*C, dC, sC};

    if constexpr (transpose_A) {
        linear_algebra::gemv((ConjA) ? 'c' : 't', AB_prefactor, tA, tB, C_prefactor, &tC);
    } else {
        linear_algebra::gemv<false>(AB_prefactor, tA, tB, C_prefactor, &tC);
    }

    return true;
}

/**
 * @brief Checks to see if the indices passed can be turned into a matrix-matrix product.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_matrix_product(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices                           = std::tuple<AIndices...>();
    constexpr auto B_indices                           = std::tuple<BIndices...>();
    constexpr auto C_indices                           = std::tuple<CIndices...>();
    constexpr auto CminusA                             = DifferenceT<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto CminusB                             = DifferenceT<std::tuple<CIndices...>, std::tuple<BIndices...>>();
    constexpr bool have_remaining_indices_in_CminusA   = std::tuple_size_v<decltype(CminusA)>;
    constexpr bool have_remaining_indices_in_CminusB   = std::tuple_size_v<decltype(CminusB)>;
    constexpr auto linksAB                             = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                               = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto C_unique                            = UniqueT<std::tuple<CIndices...>>();
    constexpr auto link_unique                         = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A                  = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B                  = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A                = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B                = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C              = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C              = detail::find_type_with_position(B_indices, C_indices);
    constexpr auto contiguous_link_position_in_A       = detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B       = detail::contiguous_positions(link_position_in_B);
    constexpr auto contiguous_target_position_in_A     = detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B     = detail::contiguous_positions(target_position_in_B);
    constexpr auto contiguous_A_targets_in_C           = detail::contiguous_positions(A_target_position_in_C);
    constexpr auto contiguous_B_targets_in_C           = detail::contiguous_positions(B_target_position_in_C);
    constexpr auto same_ordering_link_position_in_AB   = detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr bool condition = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB && contiguous_link_position_in_A &&
                               contiguous_link_position_in_B && contiguous_target_position_in_A && contiguous_target_position_in_B &&
                               contiguous_A_targets_in_C && contiguous_B_targets_in_C && same_ordering_link_position_in_AB &&
                               same_ordering_target_position_in_CA && same_ordering_target_position_in_CB;

    if constexpr (condition) {
        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
        constexpr bool transpose_B = std::get<1>(link_position_in_B) != 0;
        constexpr bool transpose_C = std::get<1>(A_target_position_in_C) != 0;
        // BLAS only supports conjugate-transpose ('c'), not conjugate-only.
        // For !transpose_C: transA = (transpose_A) ? 'c'/'t' : 'n'  → ConjA needs transpose_A
        // For  transpose_C: transA = (!transpose_A) ? 'c'/'t' : 'n' → ConjA needs !transpose_A
        constexpr bool conjugate_works = (transpose_C && (!transpose_A || !ConjA) && (!transpose_B || !ConjB)) ||
                                         (!transpose_C && (transpose_A || !ConjA) && (transpose_B || !ConjB));
        return conjugate_works;
    } else {
        return false;
    }
}

/**
 * @brief Sets up tensor views and performs a matrix-vector product on them.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
bool einsum_do_matrix_product(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                              BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                              std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/,
                              BType const &B) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices              = std::tuple<AIndices...>();
    constexpr auto B_indices              = std::tuple<BIndices...>();
    constexpr auto C_indices              = std::tuple<CIndices...>();
    constexpr auto linksAB                = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links                  = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    constexpr auto C_unique               = UniqueT<std::tuple<CIndices...>>();
    constexpr auto link_unique            = CUniqueT<decltype(links)>();
    constexpr auto link_position_in_A     = detail::find_type_with_position(link_unique, A_indices);
    constexpr auto link_position_in_B     = detail::find_type_with_position(link_unique, B_indices);
    constexpr auto target_position_in_A   = detail::find_type_with_position(C_unique, A_indices);
    constexpr auto target_position_in_B   = detail::find_type_with_position(C_unique, B_indices);
    constexpr auto A_target_position_in_C = detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = detail::find_type_with_position(B_indices, C_indices);

    constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
    constexpr bool transpose_B = std::get<1>(link_position_in_B) != 0;
    constexpr bool transpose_C = std::get<1>(A_target_position_in_C) != 0;

    Dim<2>    dA, dB, dC;
    Stride<2> sA, sB, sC;

    dA[0] = product_dims(A_target_position_in_C, *C);
    dA[1] = product_dims(link_position_in_A, A);
    sA[0] = last_stride(target_position_in_A, A);
    sA[1] = last_stride(link_position_in_A, A);

    if constexpr (ARank == 2) {
        if (A.stride(0) != 1 && A.stride(1) != 1) {
            EINSUMS_LOG_TRACE("do not have gemmable view of A.");
            return false;
        }
    } else {
        if (A.is_row_major()) {
            size_t cutoff_stride = std::max(sA[0], sA[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = ARank - 1; i >= 0; i--) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (A.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (A.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of A.");
                    return false;
                }
                check_stride *= A.dim(i);
            }
        } else {
            size_t cutoff_stride = std::max(sA[0], sA[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = 0; i < ARank; i++) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (A.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (A.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of A.");
                    return false;
                }
                check_stride *= A.dim(i);
            }
        }
    }

    if constexpr (transpose_A) {
        std::swap(dA[0], dA[1]);
        std::swap(sA[0], sA[1]);
    }

    dB[0] = product_dims(link_position_in_B, B);
    dB[1] = product_dims(B_target_position_in_C, *C);
    sB[0] = last_stride(link_position_in_B, B);
    sB[1] = last_stride(target_position_in_B, B);

    if constexpr (BRank == 2) {
        if (B.stride(0) != 1 && B.stride(1) != 1) {
            EINSUMS_LOG_TRACE("do not have gemmable view of B.");
            return false;
        }
    } else {
        if (B.is_row_major()) {
            size_t cutoff_stride = std::max(sB[0], sB[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = BRank - 1; i >= 0; i--) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (B.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (B.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of B.");
                    return false;
                }
                check_stride *= B.dim(i);
            }
        } else {
            size_t cutoff_stride = std::max(sB[0], sB[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = 0; i < BRank; i++) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (B.stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (B.stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of B.");
                    return false;
                }
                check_stride *= B.dim(i);
            }
        }
    }

    if constexpr (transpose_B) {
        std::swap(dB[0], dB[1]);
        std::swap(sB[0], sB[1]);
    }

    dC[0] = product_dims(A_target_position_in_C, *C);
    dC[1] = product_dims(B_target_position_in_C, *C);
    sC[0] = last_stride(A_target_position_in_C, *C);
    sC[1] = last_stride(B_target_position_in_C, *C);

    if constexpr (CRank == 2) {
        if (C->stride(0) != 1 && C->stride(1) != 1) {
            EINSUMS_LOG_TRACE("do not have gemmable view of C.");
            return false;
        }
    } else {
        if (C->is_row_major()) {
            size_t cutoff_stride = std::max(sC[0], sC[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = CRank - 1; i >= 0; i--) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (C->stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (C->stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of C.");
                    return false;
                }
                check_stride *= C->dim(i);
            }
        } else {
            size_t cutoff_stride = std::max(sC[0], sC[1]);
            size_t check_stride  = 1;

            for (ptrdiff_t i = 0; i < CRank; i++) {
                // Check to see if we reach the cutoff between the two sets of strides.
                // If we did, check against a new set of strides.
                if (C->stride(i) == cutoff_stride) {
                    check_stride = cutoff_stride;
                }
                if (C->stride(i) != check_stride) {
                    EINSUMS_LOG_TRACE("do not have gemmable view of C.");
                    return false;
                }
                check_stride *= C->dim(i);
            }
        }
    }

    if constexpr (transpose_C) {
        std::swap(dC[0], dC[1]);
        std::swap(sC[0], sC[1]);
    }

    TensorView<ADataType, 2> const tA{const_cast<AType &>(A), dA, sA};
    TensorView<BDataType, 2> const tB{const_cast<BType &>(B), dB, sB};
    TensorView<CDataType, 2>       tC{*C, dC, sC};
    if constexpr (CoreTensorConcept<decltype(tA)>) {
        if (!tA.impl().is_gemmable() || !tB.impl().is_gemmable() || !tC.impl().is_gemmable()) {
            return false;
        }
    } else {
        if (tA.stride(1) != 1 || tB.stride(1) != 1 || tC.stride(1) != 1) {
            return false;
        }
    }

    if constexpr (DryRun) {
        return true;
    }

    if constexpr (!transpose_C) {
        constexpr char transA = (transpose_A) ? ((ConjA) ? 'c' : 't') : 'n', transB = (transpose_B) ? ((ConjB) ? 'c' : 't') : 'n';
        linear_algebra::gemm(transA, transB, AB_prefactor, tA, tB, C_prefactor, &tC);
    } else {
        constexpr char transA = (!transpose_A) ? ((ConjA) ? 'c' : 't') : 'n', transB = (!transpose_B) ? ((ConjB) ? 'c' : 't') : 'n';
        linear_algebra::gemm(transB, transA, AB_prefactor, tB, tA, C_prefactor, &tC);
    }

    return true;
}

/**
 * @brief Checks to see if the indices can be rearranged to form a GEMM.
 *
 * This is a relaxed version of einsum_is_matrix_product that only checks
 * structural requirements (has M dims, N dims, K dims, all unique) but
 * NOT contiguity or ordering.  If this returns true, we can permute the
 * tensors so that einsum_is_matrix_product succeeds.
 */
template <bool ConjA, bool ConjB, typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_sort_gemm_candidate(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &,
                                             std::tuple<BIndices...> const &) {
    // Need all BasicTensor types (checked at call site), and ranks >= 2.
    constexpr size_t CRank = sizeof...(CIndices);
    constexpr size_t ARank = sizeof...(AIndices);
    constexpr size_t BRank = sizeof...(BIndices);
    if constexpr (CRank < 2 || ARank < 2 || BRank < 2) {
        return false;
    }

    // No Hadamard (duplicate indices within any pack).
    if constexpr (einsum_is_hadamard_found<CIndices...>() || einsum_is_hadamard_found<AIndices...>() ||
                  einsum_is_hadamard_found<BIndices...>()) {
        return false;
    }

    // Must have link indices (A∩B \ C non-empty).
    constexpr auto linksAB = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto links   = DifferenceT<decltype(linksAB), std::tuple<CIndices...>>();
    if constexpr (std::tuple_size_v<decltype(links)> == 0) {
        return false;
    }

    // Must have target indices from both A-only and B-only in C.
    // CminusB = indices in C but not in B = "M dims" (from A).
    // CminusA = indices in C but not in A = "N dims" (from B).
    using CminusA_check = DifferenceT<std::tuple<CIndices...>, std::tuple<AIndices...>>;
    using CminusB_check = DifferenceT<std::tuple<CIndices...>, std::tuple<BIndices...>>;
    if constexpr (std::tuple_size_v<CminusA_check> == 0 || std::tuple_size_v<CminusB_check> == 0) {
        return false;
    }

    // Every C index must appear in A or B. If CminusA ∩ CminusB is non-empty,
    // there is an index in C that is in neither A nor B — not a valid GEMM.
    using ExternalIndices = IntersectT<CminusA_check, CminusB_check>;
    if constexpr (std::tuple_size_v<ExternalIndices> > 0) {
        return false;
    }

    // If it already IS a matrix product, no need for sort+GEMM.
    if constexpr (einsum_is_matrix_product<ConjA, ConjB>(std::tuple<CIndices...>{}, std::tuple<AIndices...>{}, std::tuple<BIndices...>{})) {
        return false;
    }

    return true;
}

/**
 * @brief Returns the HPTT selection method from the runtime config.
 *
 * Reads `"hptt-selection-method"` from GlobalConfigMap on first call, caches the result.
 * Set via command line: `--einsums:hptt:selection-method measure`
 */
inline hptt::SelectionMethod hptt_selection_method() {
    static hptt::SelectionMethod method = [] {
        auto &config = GlobalConfigMap::get_singleton();
        config.lock();
        auto val = config.get_string("hptt-selection-method", "estimate");
        config.unlock();

        if (val == "measure")
            return hptt::MEASURE;
        if (val == "patient")
            return hptt::PATIENT;
        if (val == "crazy")
            return hptt::CRAZY;
        return hptt::ESTIMATE;
    }();
    return method;
}

/**
 * @brief Cached HPTT permute: reuses the plan when tensor dimensions/strides match.
 *
 * Template parameters (including index types) make each call site a distinct instantiation,
 * so the thread_local cache is per-permutation-pattern per-thread.
 */
template <bool ConjA, typename T, size_t SrcRank, typename SrcType, typename DstType, typename... DstIndices, typename... SrcIndices>
void cached_permute(std::tuple<DstIndices...> const &dst_indices, DstType *dst, std::tuple<SrcIndices...> const &src_indices,
                    SrcType const &src) {
    thread_local Dim<SrcRank>                        prev_dims{};
    thread_local Stride<SrcRank>                     prev_strides{};
    thread_local std::shared_ptr<hptt::Transpose<T>> plan{};

    bool hit = (plan != nullptr);
    if (hit) {
        for (size_t d = 0; d < SrcRank; d++) {
            if (src.dim(d) != prev_dims[d] || src.stride(d) != prev_strides[d]) {
                hit = false;
                break;
            }
        }
    }

    if (!hit) {
        plan = tensor_algebra::compile_permute<ConjA>(T{0}, dst_indices, dst, T{1}, src_indices, src, hptt_selection_method());
        for (size_t d = 0; d < SrcRank; d++) {
            prev_dims[d]    = src.dim(d);
            prev_strides[d] = src.stride(d);
        }
    } else {
        plan->set_input_ptr(src.data());
        plan->set_output_ptr(dst->data());
    }
    plan->execute();
}

/**
 * @brief Cached HPTT permute with explicit alpha/beta prefactors.
 */
template <bool ConjA, typename T, size_t SrcRank, typename SrcType, typename DstType, typename... DstIndices, typename... SrcIndices>
void cached_permute(T beta, std::tuple<DstIndices...> const &dst_indices, DstType *dst, T alpha,
                    std::tuple<SrcIndices...> const &src_indices, SrcType const &src) {
    thread_local Dim<SrcRank>                        prev_dims{};
    thread_local Stride<SrcRank>                     prev_strides{};
    thread_local std::shared_ptr<hptt::Transpose<T>> plan{};

    bool hit = (plan != nullptr);
    if (hit) {
        for (size_t d = 0; d < SrcRank; d++) {
            if (src.dim(d) != prev_dims[d] || src.stride(d) != prev_strides[d]) {
                hit = false;
                break;
            }
        }
    }

    if (!hit) {
        plan = tensor_algebra::compile_permute<ConjA>(beta, dst_indices, dst, alpha, src_indices, src, hptt_selection_method());
        for (size_t d = 0; d < SrcRank; d++) {
            prev_dims[d]    = src.dim(d);
            prev_strides[d] = src.stride(d);
        }
    } else {
        plan->set_input_ptr(src.data());
        plan->set_output_ptr(dst->data());
        plan->set_alpha(alpha);
        plan->set_beta(beta);
    }
    plan->execute();
}

/**
 * @brief Performs a sort (permute) + GEMM contraction.
 *
 * When einsum_is_matrix_product fails because indices are scrambled, this
 * function automatically permutes A and B (and optionally C) so that the
 * contraction maps to a standard GEMM, then calls einsum() on the permuted
 * tensors.
 *
 * Canonical ordering (with optional batch dims):
 *   A_sorted: [Batch..., M dims (from CminusB, in C order), K dims (link, in A order)]
 *   B_sorted: [Batch..., N dims (from CminusA, in C order), K dims (link, same order as A_sorted)]
 *   C_sorted: [Batch..., M dims, N dims]
 *
 * When batch dims are present (indices in all three of A, B, C), we loop over
 * batch indices and call GEMM on per-batch TensorView slices.
 *
 * If C already has the canonical ordering, no C permutation is needed.
 */
template <bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires TensorConcept<CType>
bool einsum_do_sort_gemm(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
                         BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                         std::tuple<AIndices...> const & /*As*/, AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B) {
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    // Compute the index groups at compile time using type aliases to avoid
    // const-qualification issues that arise from constexpr auto variables.
    // M dims: in C but not in B. N dims: in C but not in A. K dims: in A∩B \ C.
    // Batch dims: in all three of A, B, C.
    using M_indices_t     = DifferenceT<std::tuple<CIndices...>, std::tuple<BIndices...>>;
    using N_indices_t     = DifferenceT<std::tuple<CIndices...>, std::tuple<AIndices...>>;
    using LinksAB_t       = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>;
    using Links_t         = DifferenceT<LinksAB_t, std::tuple<CIndices...>>;
    using K_indices_t     = UniqueT<Links_t>;
    using Batch_indices_t = DecayElementsT<IntersectT<std::tuple<CIndices...>, LinksAB_t>>;

    constexpr size_t NumBatch   = std::tuple_size_v<Batch_indices_t>;
    constexpr size_t InnerARank = ARank - NumBatch;
    constexpr size_t InnerBRank = BRank - NumBatch;
    constexpr size_t InnerCRank = CRank - NumBatch;

    // Inner index tuples (without batch):
    using A_inner_t = DecayElementsT<decltype(std::tuple_cat(std::declval<M_indices_t>(), std::declval<K_indices_t>()))>;
    using B_inner_t = DecayElementsT<decltype(std::tuple_cat(std::declval<N_indices_t>(), std::declval<K_indices_t>()))>;
    using C_inner_t = DecayElementsT<decltype(std::tuple_cat(std::declval<M_indices_t>(), std::declval<N_indices_t>()))>;

    // Full sorted index tuples (with batch at front):
    using A_sorted_t =
        DecayElementsT<decltype(std::tuple_cat(std::declval<Batch_indices_t>(), std::declval<M_indices_t>(), std::declval<K_indices_t>()))>;
    using B_sorted_t =
        DecayElementsT<decltype(std::tuple_cat(std::declval<Batch_indices_t>(), std::declval<N_indices_t>(), std::declval<K_indices_t>()))>;
    using C_sorted_t =
        DecayElementsT<decltype(std::tuple_cat(std::declval<Batch_indices_t>(), std::declval<M_indices_t>(), std::declval<N_indices_t>()))>;

    constexpr auto A_sorted_indices = A_sorted_t{};
    constexpr auto B_sorted_indices = B_sorted_t{};
    constexpr auto C_sorted_indices = C_sorted_t{};

    // Check if C is already in canonical order (Batch..., M..., N...).
    constexpr bool c_needs_permute = !same_indices<std::tuple<CIndices...>, decltype(C_sorted_indices)>() ||
                                     !detail::is_same_ordering(detail::find_type_with_position(C_sorted_indices, C_indices),
                                                               detail::find_type_with_position(C_indices, C_sorted_indices));

    // Runtime: compute dimensions for the sorted tensors.
    constexpr auto a_sorted_pos_in_A = detail::find_type_with_position(A_sorted_t{}, A_indices);
    constexpr auto b_sorted_pos_in_B = detail::find_type_with_position(B_sorted_t{}, B_indices);

    Dim<ARank> a_sorted_dims;
    for_sequence<ARank>([&](auto i) {
        constexpr size_t pos = std::get<2 * i + 1>(a_sorted_pos_in_A);
        a_sorted_dims[i]     = A.dim(pos);
    });

    Dim<BRank> b_sorted_dims;
    for_sequence<BRank>([&](auto i) {
        constexpr size_t pos = std::get<2 * i + 1>(b_sorted_pos_in_B);
        b_sorted_dims[i]     = B.dim(pos);
    });

    if constexpr (DryRun) {
        return true;
    }

    // Allocate sorted temporaries.
    Tensor<CDataType, ARank> A_s("_sort_gemm_A", a_sorted_dims);
    Tensor<CDataType, BRank> B_s("_sort_gemm_B", b_sorted_dims);

    // Permute A and B into canonical order, applying conjugation during permute.
    // Uses cached HPTT plans to avoid recompiling when tensor shapes are unchanged.
    cached_permute<ConjA, CDataType, ARank>(A_sorted_indices, &A_s, A_indices, A);
    cached_permute<ConjB, CDataType, BRank>(B_sorted_indices, &B_s, B_indices, B);

    if constexpr (NumBatch == 0) {
        // No batch dims — original path.
        if constexpr (!c_needs_permute) {
            detail::einsum<false, false, false, false>(C_prefactor, C_indices, C, AB_prefactor, A_sorted_indices, A_s, B_sorted_indices,
                                                       B_s);
        } else {
            constexpr auto c_sorted_pos_in_C = detail::find_type_with_position(C_sorted_t{}, C_indices);
            Dim<CRank>     c_sorted_dims;
            for_sequence<CRank>([&](auto i) {
                constexpr size_t pos = std::get<2 * i + 1>(c_sorted_pos_in_C);
                c_sorted_dims[i]     = C->dim(pos);
            });

            Tensor<CDataType, CRank> C_s_temp("_sort_gemm_C", c_sorted_dims);

            if (C_prefactor != CDataType{0}) {
                cached_permute<false, CDataType, CRank>(C_sorted_indices, &C_s_temp, C_indices, *C);
            }

            detail::einsum<false, false, false, false>(C_prefactor, C_sorted_indices, &C_s_temp, AB_prefactor, A_sorted_indices, A_s,
                                                       B_sorted_indices, B_s);

            cached_permute<false, CDataType, CRank>(CDataType{0}, C_indices, C, CDataType{1}, C_sorted_indices, C_s_temp);
        }
    } else {
        // Batch dims present — loop over batch indices and call GEMM on slices.
        constexpr auto A_inner_indices = A_inner_t{};
        constexpr auto B_inner_indices = B_inner_t{};
        constexpr auto C_inner_indices = C_inner_t{};

        // Compute inner dims and strides from sorted tensors (positions NumBatch..Rank-1).
        Dim<InnerARank>    a_inner_dims;
        Stride<InnerARank> a_inner_strides;
        for_sequence<InnerARank>([&](auto i) {
            a_inner_dims[i]    = A_s.dim(NumBatch + i);
            a_inner_strides[i] = A_s.stride(NumBatch + i);
        });

        Dim<InnerBRank>    b_inner_dims;
        Stride<InnerBRank> b_inner_strides;
        for_sequence<InnerBRank>([&](auto i) {
            b_inner_dims[i]    = B_s.dim(NumBatch + i);
            b_inner_strides[i] = B_s.stride(NumBatch + i);
        });

        // Compute batch dims and total batch count.
        size_t        total_batches = 1;
        Dim<NumBatch> batch_dims;
        for_sequence<NumBatch>([&](auto i) {
            batch_dims[i] = A_s.dim(i);
            total_batches *= batch_dims[i];
        });

        if constexpr (!c_needs_permute) {
            // C is already in canonical (Batch..., M..., N...) order.
            // The sorted C order matches C_indices, so we work directly on C.
            Dim<InnerCRank>    c_inner_dims;
            Stride<InnerCRank> c_inner_strides;
            for_sequence<InnerCRank>([&](auto i) {
                c_inner_dims[i]    = C->dim(NumBatch + i);
                c_inner_strides[i] = C->stride(NumBatch + i);
            });

            for (size_t batch = 0; batch < total_batches; batch++) {
                // Decompose flat batch index into per-dim indices and compute data offsets.
                size_t a_offset = 0, b_offset = 0, c_offset = 0;
                size_t remaining = batch;
                for (size_t d = NumBatch; d > 0; d--) {
                    size_t idx = remaining % batch_dims[d - 1];
                    remaining /= batch_dims[d - 1];
                    a_offset += idx * A_s.stride(d - 1);
                    b_offset += idx * B_s.stride(d - 1);
                    c_offset += idx * C->stride(d - 1);
                }

                TensorView<CDataType, InnerARank> a_slice(A_s.data() + a_offset, a_inner_dims, a_inner_strides);
                TensorView<CDataType, InnerBRank> b_slice(B_s.data() + b_offset, b_inner_dims, b_inner_strides);
                TensorView<CDataType, InnerCRank> c_slice(C->data() + c_offset, c_inner_dims, c_inner_strides);

                detail::einsum<false, false, false, false>(C_prefactor, C_inner_indices, &c_slice, AB_prefactor, A_inner_indices, a_slice,
                                                           B_inner_indices, b_slice);
            }
        } else {
            // C needs permutation: work in canonical order, then permute back.
            constexpr auto c_sorted_pos_in_C = detail::find_type_with_position(C_sorted_t{}, C_indices);
            Dim<CRank>     c_sorted_dims;
            for_sequence<CRank>([&](auto i) {
                constexpr size_t pos = std::get<2 * i + 1>(c_sorted_pos_in_C);
                c_sorted_dims[i]     = C->dim(pos);
            });

            Tensor<CDataType, CRank> C_s_temp("_sort_gemm_C", c_sorted_dims);

            if (C_prefactor != CDataType{0}) {
                cached_permute<false, CDataType, CRank>(C_sorted_indices, &C_s_temp, C_indices, *C);
            }

            Dim<InnerCRank>    c_inner_dims;
            Stride<InnerCRank> c_inner_strides;
            for_sequence<InnerCRank>([&](auto i) {
                c_inner_dims[i]    = C_s_temp.dim(NumBatch + i);
                c_inner_strides[i] = C_s_temp.stride(NumBatch + i);
            });

            for (size_t batch = 0; batch < total_batches; batch++) {
                size_t a_offset = 0, b_offset = 0, c_offset = 0;
                size_t remaining = batch;
                for (size_t d = NumBatch; d > 0; d--) {
                    size_t idx = remaining % batch_dims[d - 1];
                    remaining /= batch_dims[d - 1];
                    a_offset += idx * A_s.stride(d - 1);
                    b_offset += idx * B_s.stride(d - 1);
                    c_offset += idx * C_s_temp.stride(d - 1);
                }

                TensorView<CDataType, InnerARank> a_slice(A_s.data() + a_offset, a_inner_dims, a_inner_strides);
                TensorView<CDataType, InnerBRank> b_slice(B_s.data() + b_offset, b_inner_dims, b_inner_strides);
                TensorView<CDataType, InnerCRank> c_slice(C_s_temp.data() + c_offset, c_inner_dims, c_inner_strides);

                detail::einsum<false, false, false, false>(C_prefactor, C_inner_indices, &c_slice, AB_prefactor, A_inner_indices, a_slice,
                                                           B_inner_indices, b_slice);
            }

            // Permute result back to original C ordering.
            cached_permute<false, CDataType, CRank>(CDataType{0}, C_indices, C, CDataType{1}, C_sorted_indices, C_s_temp);
        }
    }

    return true;
}

/**
 * @brief Checks to see if there are indices that appear in all three index packs.
 *
 * The indices that appear in all three index packs can be used to batch einsum calls.
 */
template <typename... CIndices, typename... AIndices, typename... BIndices>
constexpr bool einsum_is_batchable(std::tuple<CIndices...> const &, std::tuple<AIndices...> const &, std::tuple<BIndices...> const &) {
    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();
    constexpr auto A_unique  = UniqueT<std::tuple<AIndices...>>();
    constexpr auto B_unique  = UniqueT<std::tuple<BIndices...>>();
    constexpr auto C_unique  = UniqueT<std::tuple<CIndices...>>();
    constexpr auto linksAB   = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    constexpr auto batches   = IntersectT<std::tuple<CIndices...>, decltype(linksAB)>();

    return std::tuple_size_v<decltype(batches)> > 0;
}

// CType has typename to allow for interoperability with scalar types.
template <bool OnlyUseGenericAlgorithm, bool DryRun, bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(TensorConcept<CType> || (ScalarConcept<CType> && sizeof...(CIndices) == 0))
auto einsum(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const & /*Cs*/, CType *C,
            BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor, std::tuple<AIndices...> const & /*As*/,
            AType const &A, std::tuple<BIndices...> const & /*Bs*/, BType const &B) -> AlgorithmChoice {
    // profile::Timer const _timer();
    print::Indent const _indent;

    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    // Ensure the ranks are correct. (Compile-time check.)
    static_assert(sizeof...(CIndices) == CRank, "Rank of C does not match Indices given for C.");
    static_assert(sizeof...(AIndices) == ARank, "Rank of A does not match Indices given for A.");
    static_assert(sizeof...(BIndices) == BRank, "Rank of B does not match Indices given for B.");

    // Runtime check of sizes
#    if defined(EINSUMS_RUNTIME_INDICES_CHECK)
    if constexpr (!DryRun) {
        einsum_runtime_check(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
    }
#    endif

    bool            has_performed_contraction = false;
    AlgorithmChoice retval                    = INDETERMINATE;

    if constexpr (OnlyUseGenericAlgorithm) {
        // Skip to the generic algorithm.
    } else if constexpr (einsum_is_all_hadamard_found(C_indices, A_indices, B_indices) || !std::is_same_v<CDataType, ADataType> ||
                         !std::is_same_v<CDataType, BDataType> ||
                         (!IsAlgebraTensorV<AType> || !IsAlgebraTensorV<BType> || (!IsAlgebraTensorV<CType> && !IsScalarV<CType>))) {
        // Mixed datatypes and poorly behaved tensor types go directly to the generic algorithm.
    } else if constexpr (einsum_is_dot_product(C_indices, A_indices, B_indices)) {
        if constexpr (!DryRun) {
            if constexpr (ConjA == ConjB || (!IsComplexV<ADataType> && !IsComplex<BDataType>)) {
                CDataType temp = linear_algebra::dot(A, B);
                if constexpr (ConjA && IsComplexV<CDataType>) {
                    temp = std::conj(temp);
                }
                (*C) *= C_prefactor;
                (*C) += AB_prefactor * temp;
            } else {
                CDataType temp = linear_algebra::true_dot(A, B);
                if constexpr (ConjB && IsComplexV<CDataType>) {
                    temp = std::conj(temp);
                }
                (*C) *= C_prefactor;
                (*C) += AB_prefactor * temp;
            }
        }

        has_performed_contraction = true;
        retval                    = DOT;
    } else if constexpr (einsum_is_direct_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        if constexpr (!DryRun) {
            linear_algebra::direct_product(AB_prefactor, A, B, C_prefactor, C);
        }

        has_performed_contraction = true;
        retval                    = DIRECT;
    } else if constexpr (!IsBasicTensorV<AType> || !IsBasicTensorV<BType> || !IsBasicTensorV<CType>) {
        retval = einsum_generic_default<false, DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        has_performed_contraction = true;
    } else if constexpr (einsum_is_outer_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        has_performed_contraction =
            einsum_do_outer_product<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        retval = GER;
    } else if constexpr (einsum_is_matrix_vector<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
        has_performed_contraction =
            einsum_do_matrix_vector<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
        retval = GEMV;
    } else if constexpr (einsum_is_matrix_vector<ConjA, ConjB>(C_indices, B_indices, A_indices)) {
        has_performed_contraction =
            einsum_do_matrix_vector<DryRun, ConjB, ConjA>(C_prefactor, C_indices, C, AB_prefactor, B_indices, B, A_indices, A);
        retval = GEMV;
    }
    // To use a gemm the input tensors need to be at least rank 2
    else if constexpr (CRank >= 2 && ARank >= 2 && BRank >= 2) {
        if constexpr (einsum_is_matrix_product<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
            has_performed_contraction =
                einsum_do_matrix_product<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
            retval = GEMM;
        }
    }

    // Sort+GEMM: auto-permute scrambled indices to enable GEMM dispatch.
    if (!has_performed_contraction) {
        if constexpr (IsBasicTensorV<AType> && IsBasicTensorV<BType> && IsBasicTensorV<CType> &&
                      einsum_is_sort_gemm_candidate<ConjA, ConjB>(C_indices, A_indices, B_indices)) {
            has_performed_contraction =
                einsum_do_sort_gemm<DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);
            retval = SORT_GEMM;
        }
    }

    if (!has_performed_contraction) {
        return einsum_generic_default<OnlyUseGenericAlgorithm, DryRun, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A,
                                                                                     B_indices, B);
    }
    return retval;
}
} // namespace detail

template <bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename CType, typename U, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires InSamePlace<AType, CType> || !TensorConcept<CType>;
        requires !SmartPointer<CType>;
    }
void einsum(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UAB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices, BType const &B,
            detail::AlgorithmChoice *algorithm_choice) {
    using ADataType        = AType::ValueType;
    using BDataType        = BType::ValueType;
    using CDataType        = ValueTypeT<CType>;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = TensorRank<CType>;

    using ABDataType = std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType>;

    EINSUMS_LOG_TRACE("BEGIN: einsum");
#    if defined(EINSUMS_HAVE_PROFILER)
    std::unique_ptr<profile::ScopedZone> _section;
#    endif
    if constexpr (IsTensorV<CType>) {
        EINSUMS_LOG_DEBUG(
            std::abs(UC_prefactor) > EINSUMS_ZERO
                ? fmt::format(R"(einsum: "{}"{} = {} {}"{}"{}{} * {}"{}"{}{} + {} "{}"{})", C->name(), C_indices, UAB_prefactor,
                              (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "", B.name(), B_indices,
                              (ConjB) ? ")" : "", UC_prefactor, C->name(), C_indices)
                : fmt::format(R"(einsum: "{}"{} = {} {}"{}"{}{} * {}"{}"{}{})", C->name(), C_indices, UAB_prefactor, (ConjA) ? "conj(" : "",
                              A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "", B.name(), B_indices, (ConjB) ? ")" : ""));
#    if defined(EINSUMS_HAVE_PROFILER)
        _section = std::make_unique<profile::ScopedZone>(std::abs(UC_prefactor) > EINSUMS_ZERO
                                                             ? fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(),
                                                                           C_indices, UAB_prefactor, A.name(), A_indices, B.name(),
                                                                           B_indices, UC_prefactor, C->name(), C_indices)
                                                             : fmt::format(R"(einsums: "{}"{} = {} "{}"{} * "{}"{})", C->name(), C_indices,
                                                                           UAB_prefactor, A.name(), A_indices, B.name(), B_indices),
                                                         __FILE__, __LINE__, __func__);
#    endif
    } else {
        EINSUMS_LOG_DEBUG(std::abs(UC_prefactor) > EINSUMS_ZERO
                              ? fmt::format(R"(einsum: "C"{} = {} {}"{}"{}{} * {}"{}"{}{} + {} "C"{})", C_indices, UAB_prefactor,
                                            (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "",
                                            B.name(), B_indices, (ConjB) ? ")" : "", UC_prefactor, C_indices)
                              : fmt::format(R"(einsum: "C"{} = {} {}"{}"{}{} * {}"{}"{}{})", C_indices, UAB_prefactor,
                                            (ConjA) ? "conj(" : "", A.name(), A_indices, (ConjA) ? ")" : "", (ConjB) ? "conj(" : "",
                                            B.name(), B_indices, (ConjB) ? ")" : ""));
#    if defined(EINSUMS_HAVE_PROFILER)
        _section = std::make_unique<profile::ScopedZone>(
            std::abs(UC_prefactor) > EINSUMS_ZERO
                ? fmt::format(R"(einsum: "C"{} = {} "{}"{} * "{}"{} + {} "C"{})", C_indices, UAB_prefactor, A.name(), A_indices, B.name(),
                              B_indices, UC_prefactor, C_indices)
                : fmt::format(R"(einsum: "C"{} = {} "{}"{} * "{}"{})", C_indices, UAB_prefactor, A.name(), A_indices, B.name(), B_indices),
            __FILE__, __LINE__, __func__);
#    endif
    }

    CDataType const  C_prefactor  = UC_prefactor;
    ABDataType const AB_prefactor = UAB_prefactor;

#    if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    auto testC = Tensor<CDataType, CRank>(*C);
    {
        Section t1("testing");
        if constexpr (!einsums::detail::IsBasicTensorV<AType> && !einsums::detail::IsBasicTensorV<BType>) {
            auto testA = Tensor<ADataType, ARank>(A);
            auto testB = Tensor<BDataType, BRank>(B);
            { detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, testB); }
        } else if constexpr (!einsums::detail::IsBasicTensorV<AType>) {
            auto testA = Tensor<ADataType, ARank>(A);
            { detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, testA, B_indices, B); }
        } else if constexpr (!einsums::detail::IsBasicTensorV<BType>) {
            auto testB = Tensor<BDataType, BRank>(B);
            { detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, testB); }
        } else {
            // Perform the einsum using only the generic algorithm
            // #pragma omp task depend(in: A, B) depend(inout: testC)
            { detail::einsum<true, false, ConjA, ConjB>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B); }
            // #pragma omp taskwait depend(in: testC)
        }
    }
#    endif

    // Default einsums.
    detail::AlgorithmChoice retval =
        detail::einsum<false, false, ConjA, ConjB>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

#    if defined(EINSUMS_TEST_NANS)
    // The tests need a wait.
    // #pragma omp taskwait depend(in: *C, testC)
    if constexpr (CRank != 0) {
        Stride<CRank> index_strides;

        size_t elements = dims_to_strides(C->dims(), index_strides);

        for (size_t item = 0; item < elements; item++) {
            std::array<int64_t, CRank> target_combination;

            sentinel_to_indices(item, index_strides, target_combination);

            CDataType Cvalue{subscript_tensor(*C, target_combination)};
            if constexpr (!IsComplexV<CDataType>) {
                if (std::isnan(Cvalue)) {
                    EINSUMS_LOG_ERROR("NaN DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "NAN detected in resulting tensor.");
                }

                if (std::isinf(Cvalue)) {
                    EINSUMS_LOG_ERROR("Infinity DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Infinity detected in resulting tensor.");
                }

                if (std::abs(Cvalue) > 100000000) {
                    EINSUMS_LOG_ERROR("Large value DETECTED!");
                    EINSUMS_LOG_ERROR("{:f} {}({:}) += {:f} {}({:}) * {}({:})", C_prefactor, C->name(), print_tuple_no_type(C_indices),
                                      AB_prefactor, A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices));

                    println(*C);
                    println(A);
                    println(B);

                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Large value detected in resulting tensor.");
                }
            }
        }
    }
#    endif

#    if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    if constexpr (CRank != 0) {
        // Need to walk through the entire C and testC comparing values and reporting differences.
        bool print_info_and_abort{false};

        Stride<CRank> index_strides;

        size_t elements = dims_to_strides(C->dims(), index_strides);

        for (size_t item = 0; item < elements; item++) {
            std::array<int64_t, CRank> target_combination;

            sentinel_to_indices(item, index_strides, target_combination);

            CDataType Cvalue{subscript_tensor(*C, target_combination)};
            CDataType Ctest{subscript_tensor(testC, target_combination)};

            if constexpr (!IsComplexV<CDataType>) {
                if (std::isnan(Cvalue) || std::isnan(Ctest)) {
                    EINSUMS_LOG_ERROR("NAN DETECTED!");
                    println("Source tensors");
                    println(A);
                    println(B);
                    if (std::isnan(Cvalue)) {
                        EINSUMS_LOG_ERROR("NAN detected in C");
                        EINSUMS_LOG_ERROR("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(*C);
                    }
                    if (std::isnan(Ctest)) {
                        EINSUMS_LOG_ERROR("NAN detected in reference Ctest");
                        EINSUMS_LOG_ERROR("location of detected NAN {}", print_tuple_no_type(target_combination));
                        println(testC);
                    }

                    print_info_and_abort = true;
                }
            }

#        if defined(EINSUMS_USE_CATCH2)
            if constexpr (!IsComplexV<CDataType>) {
                REQUIRE_THAT(Cvalue,
                             Catch::Matchers::WithinRel(Ctest, static_cast<CDataType>(0.001)) || Catch::Matchers::WithinAbs(0, 0.0001));
                CHECK(print_info_and_abort == false);
            }
#        endif

            if (std::abs(Cvalue - Ctest) > 1.0E-6) {
                print_info_and_abort = true;
            }

            if (print_info_and_abort) {
                EINSUMS_LOG_ERROR(emphasis::bold, "!!! EINSUM ERROR !!!");
                if constexpr (IsComplexV<CDataType>) {
                    EINSUMS_LOG_ERROR("Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                    EINSUMS_LOG_ERROR("Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
                } else {
                    EINSUMS_LOG_ERROR("Expected {:20.14f}", Ctest);
                    EINSUMS_LOG_ERROR("Obtained {:20.14f}", Cvalue);
                }
                EINSUMS_LOG_ERROR("tensor element ({:})", print_tuple_no_type(target_combination));
                std::string C_prefactor_string;
                if constexpr (IsComplexV<CDataType>) {
                    C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
                } else {
                    C_prefactor_string = fmt::format("{:f}", C_prefactor);
                }
                std::string AB_prefactor_string;
                if constexpr (IsComplexV<ABDataType>) {
                    AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
                } else {
                    AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
                }
                EINSUMS_LOG_ERROR("{} C({:}) += {:f} A({:}) * B({:})", C_prefactor_string, print_tuple_no_type(C_indices),
                                  AB_prefactor_string, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

                println("Expected:");
                println(testC);
                println("Obtained");
                println(*C);
                println(A);
                println(B);
#        if defined(EINSUMS_TEST_EINSUM_ABORT)
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Continuous test failed!");
#        endif
            }
        }
    } else {
        CDataType const Cvalue = static_cast<CDataType const>(*C);
        CDataType const Ctest  = static_cast<CDataType const>(testC);

        // testC could be a Tensor<CDataType, 0> type. Cast it to the underlying data type.
        if (std::abs(Cvalue - (CDataType)testC) > 1.0E-6) {
            println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "!!! EINSUM ERROR !!!");
            if constexpr (IsComplexV<CDataType>) {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f} + {:20.14f}i", Ctest.real(), Ctest.imag());
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f} + {:20.14f}i", Cvalue.real(), Cvalue.imag());
            } else {
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);
            }

            println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ()");
            std::string C_prefactor_string;
            if constexpr (IsComplexV<CDataType>) {
                C_prefactor_string = fmt::format("({:f} + {:f}i)", C_prefactor.real(), C_prefactor.imag());
            } else {
                C_prefactor_string = fmt::format("{:f}", C_prefactor);
            }
            std::string AB_prefactor_string;
            if constexpr (IsComplexV<ABDataType>) {
                AB_prefactor_string = fmt::format("({:f} + {:f}i)", AB_prefactor.real(), AB_prefactor.imag());
            } else {
                AB_prefactor_string = fmt::format("{:f}", AB_prefactor);
            }
            println(bg(fmt::color::red) | fg(fmt::color::white), "    {} C() += {} A({:}) * B({:})", C_prefactor_string,
                    AB_prefactor_string, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

            println("Expected:");
            println(testC);
            println("Obtained");
            println(*C);
            println(A);
            println(B);

#        if defined(EINSUMS_TEST_EINSUM_ABORT)
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Continuous test failed!");
#        endif
        }
    }
#    endif
    // Annotate the profiling zone with the algorithm choice and tensor ranks.
    {
        static constexpr char const *algo_names[] = {"GENERIC", "DOT",         "DIRECT",    "GER",          "GEMV",
                                                     "GEMM",    "PACKED_GEMM", "SORT_GEMM", "INDETERMINATE"};
#    if defined(EINSUMS_HAVE_PROFILER)
        if (retval >= 0 && retval < static_cast<detail::AlgorithmChoice>(sizeof(algo_names) / sizeof(algo_names[0]))) {
            ProfileAnnotate("algorithm", algo_names[retval]);
        }
        ProfileAnnotate("C_rank", static_cast<int64_t>(CRank));
        ProfileAnnotate("A_rank", static_cast<int64_t>(ARank));
        ProfileAnnotate("B_rank", static_cast<int64_t>(BRank));
#    endif
        // Warn once per unique contraction pattern when falling back to the
        // generic nested-loop algorithm.  The `static bool` ensures each
        // template instantiation (= unique index pattern) warns only once.
        if (retval == detail::GENERIC) {
            thread_local static bool warned = false;
            if (!warned) {
                warned = true;
                if constexpr (IsTensorV<CType>) {
                    EINSUMS_LOG_WARN("einsum dispatch: GENERIC fallback for \"{}\"{} = \"{}\"{} * \"{}\"{} "
                                     "(ranks {}/{}/{}).  This contraction is not accelerated by BLAS.",
                                     C->name(), C_indices, A.name(), A_indices, B.name(), B_indices, CRank, ARank, BRank);
                } else {
                    EINSUMS_LOG_WARN("einsum dispatch: GENERIC fallback for scalar{} = \"{}\"{} * \"{}\"{} "
                                     "(ranks {}/{}/{}).  This contraction is not accelerated by BLAS.",
                                     C_indices, A.name(), A_indices, B.name(), B_indices, CRank, ARank, BRank);
                }
            }
        }
    }

    EINSUMS_LOG_TRACE("END: einsum");

    if (algorithm_choice != nullptr) {
        *algorithm_choice = retval;
    }
}

template <bool ConjA, bool ConjB, Container CType, Container AType, Container BType, typename CPrefactorType, typename ABPrefactorType,
          typename... AIndices, typename... BIndices, typename... CIndices>
void einsum(CPrefactorType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C_list, ABPrefactorType const AB_prefactor,
            std::tuple<AIndices...> const &A_indices, AType const &A_list, std::tuple<BIndices...> const &B_indices, BType const &B_list,
            detail::AlgorithmChoice *algorithm_choice) {
    if (C_list->size() != A_list.size() || C_list->size() != B_list.size()) {
        EINSUMS_THROW_EXCEPTION(bad_logic, "Lists passed to batched einsum call do not have the same size!");
    }

    if (C_list->size() == 0) {
        return;
    }

    size_t tensors = C_list->size();

    *algorithm_choice = detail::einsum<false, true, ConjA, ConjB>(C_prefactor, C_indices, &(C_list->at(0)), AB_prefactor, A_indices,
                                                                  A_list.at(0), B_indices, B_list.at(0));

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < tensors; i++) {
        einsum<ConjA, ConjB>(C_prefactor, C_indices, &(C_list->at(i)), AB_prefactor, A_indices, A_list.at(i), B_indices, B_list.at(i));
    }
}

} // namespace einsums::tensor_algebra

#endif
