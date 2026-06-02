//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/StringUtil/StringOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/HpttPlanCache.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include <memory>

namespace einsums::tensor_algebra {

#if !defined(EINSUMS_WINDOWS)
namespace detail {

template <bool ConjA = false, typename T, typename... CIndices, typename... AIndices>
std::shared_ptr<hptt::Transpose<T>> compile_permute(T beta, std::tuple<CIndices...> const &C_indices, einsums::detail::TensorImpl<T> *C,
                                                    T alpha, std::tuple<AIndices...> const &A_indices,
                                                    einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod                 method = hptt::ESTIMATE) {
    constexpr size_t ARank = sizeof...(AIndices);
    constexpr size_t CRank = sizeof...(CIndices);

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    einsums::for_sequence<ARank>([&](auto n) {
        if (C->dim((size_t)n) < A.dim(std::get<2 * (size_t)n + 1>(target_position_in_A))) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The {} dimension of the output tensor is smaller than the input tensor!",
                                    print::ordinal((size_t)n));
        }

        if (C->dim((size_t)n) == 0) {
            return;
        }
    });
    // Calculate reversed indices.
    using ReverseC                   = ReverseT<CIndices...>;
    constexpr auto reverse_C_indices = ReverseC();

    std::array<int, ARank>    perms{};
    std::array<size_t, ARank> size{};
    std::array<size_t, ARank> outerSizeA{};
    std::array<size_t, ARank> offsetA{};
    std::array<size_t, ARank> outerSizeC{};
    std::array<size_t, ARank> offsetC{};

    if (A.is_row_major() && C->is_row_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(C_indices, A_indices);
        size_t innerStrideA             = A.stride(ARank - 1);
        size_t innerStrideC             = C->stride(CRank - 1);
        perms[0]                        = arguments::get_from_tuple<size_t>(new_target_position_in_A, 1);
        size[0]                         = A.dim(0);
        outerSizeA[0]                   = A.dim(0);
        offsetA[0]                      = 0;
        outerSizeC[0]                   = C->dim(0);
        offsetC[0]                      = 0;
        for (int i0 = 1; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 - 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), ARank, alpha, A.data(), size.data(), outerSizeA.data(), offsetA.data(),
                                                       innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(), innerStrideC, true,
                                                       method);
        plan->set_conj_a(ConjA);

        return plan;
    } else if (A.is_row_major() && C->is_column_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(reverse_C_indices, A_indices);
        auto   C_swap                   = C->to_row_major();
        size_t innerStrideA             = A.stride(ARank - 1);
        size_t innerStrideC             = C_swap.stride(CRank - 1);
        perms[0]                        = arguments::get_from_tuple<size_t>(new_target_position_in_A, 1);
        size[0]                         = A.dim(0);
        outerSizeA[0]                   = A.dim(0);
        offsetA[0]                      = 0;
        outerSizeC[0]                   = C_swap.dim(0);
        offsetC[0]                      = 0;
        for (int i0 = 1; i0 < ARank; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 - 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), ARank, alpha, A.data(), size.data(), outerSizeA.data(), offsetA.data(),
                                                       innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(), innerStrideC, true,
                                                       method);
        plan->set_conj_a(ConjA);

        return plan;
    } else if (A.is_column_major() && C->is_column_major()) {
        auto   new_target_position_in_A = detail::find_type_with_position(C_indices, A_indices);
        size_t innerStrideA             = A.stride(0);
        size_t innerStrideC             = C->stride(0);
        perms[ARank - 1]                = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * ARank) - 1);
        size[ARank - 1]                 = A.dim(-1);
        outerSizeA[ARank - 1]           = A.dim(-1);
        offsetA[ARank - 1]              = 0;
        outerSizeC[ARank - 1]           = C->dim(-1);
        offsetC[ARank - 1]              = 0;
        for (int i0 = 0; i0 < ARank - 1; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 + 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), ARank, alpha, A.data(), size.data(), outerSizeA.data(), offsetA.data(),
                                                       innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(), innerStrideC,
                                                       false, method);
        plan->set_conj_a(ConjA);

        return plan;
    } else {
        auto   new_target_position_in_A = detail::find_type_with_position(reverse_C_indices, A_indices);
        auto   C_swap                   = C->to_column_major();
        size_t innerStrideA             = A.stride(0);
        size_t innerStrideC             = C_swap.stride(0);
        perms[ARank - 1]                = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * ARank) - 1);
        size[ARank - 1]                 = A.dim(-1);
        outerSizeA[ARank - 1]           = A.dim(-1);
        offsetA[ARank - 1]              = 0;
        outerSizeC[ARank - 1]           = C_swap.dim(-1);
        offsetC[ARank - 1]              = 0;
        for (int i0 = 0; i0 < ARank - 1; i0++) {
            perms[i0]      = arguments::get_from_tuple<size_t>(new_target_position_in_A, (2 * i0) + 1);
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 + 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), ARank, alpha, A.data(), size.data(), outerSizeA.data(), offsetA.data(),
                                                       innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(), innerStrideC,
                                                       false, method);
        plan->set_conj_a(ConjA);

        return plan;
    }
}

template <bool ConjA = false, typename T>
std::shared_ptr<hptt::Transpose<T>> compile_permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha,
                                                    std::string const &A_indices, einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod method = hptt::ESTIMATE) {
    LabeledSection((std::abs(beta) > EINSUMS_ZERO)
                       ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), C_indices, alpha, A.name(), A_indices, beta,
                                     C->name(), C_indices)
                       : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), C_indices, alpha, A.name(), A_indices));

    // Error check:  If there are any remaining indices then we cannot perform a permute
    auto check = difference(A_indices, C_indices);
    if (check.size() != 0) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The number of unique indices needs to be the same on the input and output for permute!");
    }

    // Calculate reversed indices.
    BufferVector<int>    perms(A.rank());
    BufferVector<size_t> size(A.rank());
    BufferVector<size_t> outerSizeA(A.rank());
    BufferVector<size_t> offsetA(A.rank());
    BufferVector<size_t> outerSizeC(A.rank());
    BufferVector<size_t> offsetC(A.rank());

    if (A.is_row_major() && C->is_row_major()) {
        size_t innerStrideA = A.stride(-1);
        size_t innerStrideC = C->stride(-1);
        size[0]             = A.dim(0);
        outerSizeA[0]       = A.dim(0);
        offsetA[0]          = 0;
        outerSizeC[0]       = C->dim(0);
        offsetC[0]          = 0;
        for (int i0 = 1; i0 < A.rank(); i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 - 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }

        find_char_with_position(A_indices, C_indices, &perms);
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), A.rank(), alpha, A.data(), size.data(), outerSizeA.data(),
                                                       offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                       innerStrideC, true, method);
        plan->set_conj_a(ConjA);

        return plan;
    } else if (A.is_row_major() && C->is_column_major()) {
        auto   C_swap       = C->to_row_major();
        size_t innerStrideA = A.stride(-1);
        size_t innerStrideC = C_swap.stride(-1);
        size[0]             = A.dim(0);
        outerSizeA[0]       = A.dim(0);
        offsetA[0]          = 0;
        outerSizeC[0]       = C_swap.dim(0);
        offsetC[0]          = 0;
        for (int i0 = 1; i0 < A.rank(); i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 - 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 - 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        find_char_with_position(A_indices, reverse(C_indices), &perms);
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), A.rank(), alpha, A.data(), size.data(), outerSizeA.data(),
                                                       offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                       innerStrideC, true, method);
        plan->set_conj_a(ConjA);

        return plan;
    } else if (A.is_column_major() && C->is_column_major()) {
        size_t innerStrideA      = A.stride(0);
        size_t innerStrideC      = C->stride(0);
        size[A.rank() - 1]       = A.dim(-1);
        outerSizeA[A.rank() - 1] = A.dim(-1);
        offsetA[A.rank() - 1]    = 0;
        outerSizeC[A.rank() - 1] = C->dim(-1);
        offsetC[A.rank() - 1]    = 0;
        for (int i0 = 0; i0 < A.rank() - 1; i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C->stride(i0 + 1) / (C->stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }

        find_char_with_position(A_indices, C_indices, &perms);
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), A.rank(), alpha, A.data(), size.data(), outerSizeA.data(),
                                                       offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                       innerStrideC, false, method);
        plan->set_conj_a(ConjA);

        return plan;
    } else {
        auto   C_swap            = C->to_column_major();
        size_t innerStrideA      = A.stride(0);
        size_t innerStrideC      = C_swap.stride(0);
        size[A.rank() - 1]       = A.dim(-1);
        outerSizeA[A.rank() - 1] = A.dim(-1);
        offsetA[A.rank() - 1]    = 0;
        outerSizeC[A.rank() - 1] = C_swap.dim(-1);
        offsetC[A.rank() - 1]    = 0;
        for (int i0 = 0; i0 < A.rank() - 1; i0++) {
            size[i0]       = A.dim(i0);
            outerSizeA[i0] = A.stride(i0 + 1) / (A.stride(i0) * innerStrideA);
            offsetA[i0]    = 0;
            outerSizeC[i0] = C_swap.stride(i0 + 1) / (C_swap.stride(i0) * innerStrideC);
            offsetC[i0]    = 0;
        }
        find_char_with_position(A_indices, C_indices, &perms);
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), A.rank(), alpha, A.data(), size.data(), outerSizeA.data(),
                                                       offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                       innerStrideC, false, method);
        plan->set_conj_a(ConjA);

        return plan;
    }
}

template <typename T>
void permute(einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A, std::shared_ptr<hptt::Transpose<T>> plan) {
    plan->set_input_ptr(A.data());
    plan->set_output_ptr(C->data());

    plan->execute();
}

template <bool ConjA = false, typename T, typename... CIndices, typename... AIndices>
void permute(T beta, std::tuple<CIndices...> const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha,
             std::tuple<AIndices...> const &A_indices, einsums::detail::TensorImpl<T> const &A) {
    auto plan = compile_permute<ConjA>(beta, C_indices, C, alpha, A_indices, A);
    plan->execute();
}

template <bool ConjA = false, typename T>
void permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha, std::string const &A_indices,
             einsums::detail::TensorImpl<T> const &A) {
    auto plan = compile_permute<ConjA>(beta, C_indices, C, alpha, A_indices, A);
    plan->execute();
}

} // namespace detail
#endif

/**
 * @brief Permutes the elements of a tensor and puts it into an output tensor.
 *
 * This function uses HPTT, which can only handle out-of-place tensor transpositions. The tensors passed in should not
 * have overlapping storage.
 *
 * @param UC_prefactor The prefactor for mixing the output tensor.
 * @param C_indices The indices for the output tensor.
 * @param C The output tensor.
 * @param UA_prefactor The prefactor for the input tensor.
 * @param A_indices The indices for the input tensor.
 * @param A The input tensor.
 * @tparam ConjA If true, conjugate the values of A as it is being permuted.
 */
template <bool ConjA = false, CoreTensorConcept AType, CoreTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {
    using T                = typename AType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    std::string description = std::abs(UC_prefactor) > EINSUMS_ZERO
                                  ? fmt::format(R"(permute: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(),
                                                A_indices, UC_prefactor, C->name(), C_indices)
                                  : fmt::format(R"(permute: "{}"{} = {} "{}"{})", C->name(), C_indices, UA_prefactor, A.name(), A_indices);
    LabeledSection(fmt::runtime(description));

    T const C_prefactor = UC_prefactor;
    T const A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a permute
    constexpr auto check = DifferenceT<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    einsums::for_sequence<ARank>([&](auto n) {
        if (C->dim((size_t)n) < A.dim(std::get<2 * (size_t)n + 1>(target_position_in_A))) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The {} dimension of the output tensor is smaller than the input tensor!",
                                    print::ordinal((size_t)n));
        }

        if (C->dim((size_t)n) == 0) {
            return;
        }
    });

#if !defined(EINSUMS_WINDOWS)
    if (CoreBasicTensorConcept<AType> && CoreBasicTensorConcept<CType>) {
        detail::permute<ConjA>(C_prefactor, C_indices, &C->impl(), A_prefactor, A_indices, A.impl());
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)> && !(ConjA && IsComplexV<T>)) {
        // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
        if (C_prefactor == T{0.0}) {
            *C = T{0.0};
        }
        linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
        if (C_prefactor == T{0.0}) {
            *C = T{0.0};
        }
        Stride<ARank> index_strides;
        size_t        elements = dims_to_strides(A.dims(), index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < elements; i++) {
            thread_local std::array<int64_t, ARank> index;

            sentinel_to_indices(i, index_strides, index);

            auto A_order = detail::construct_indices(A_indices, index, target_position_in_A, index, target_position_in_A);

            T &target_value = subscript_tensor(*C, index);
            T  A_value      = subscript_tensor(A, A_order);

            if constexpr (ConjA && IsComplexV<T>) {
                target_value = C_prefactor * target_value + A_prefactor * std::conj(A_value);
            } else {
                target_value = C_prefactor * target_value + A_prefactor * A_value;
            }
        }
    }
}

/**
 * Compiles a permutation so that it can be used multiple times.
 *
 * @param UC_prefactor The prefactor for the output.
 * @param C_indices The indices for the output.
 * @param C The output tensor.
 * @param UA_prefactor The prefactor for the input.
 * @param A_indices The indices for the input.
 * @param A The input tensor.
 * @param method Indicates which method HPTT should use to optimize the permutation. By default, it is hptt::ESTIMATE , which is the
 * simplest of the options.
 *
 * @return The compiled plan.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, CoreTensorConcept AType, CoreTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
std::shared_ptr<hptt::Transpose<typename AType::ValueType>>
compile_permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
                std::tuple<AIndices...> const &A_indices, AType const &A, hptt::SelectionMethod method = hptt::ESTIMATE) {
    using T = typename AType::ValueType;
    return detail::compile_permute<ConjA>((T)UC_prefactor, C_indices, &C->impl(), (T)UA_prefactor, A_indices, A.impl(), method);
}

/**
 * Perform a permuation using the provided plan.
 *
 * @param C The output tensor.
 * @param A The input tensor.
 * @param plan The pre-compiled plan.
 *
 * @versionadded{2.0.0}
 */
template <CoreTensorConcept AType, CoreTensorConcept CType>
    requires SameUnderlyingAndRank<AType, CType>
void permute(CType *C, AType const &A, std::shared_ptr<hptt::Transpose<typename AType::ValueType>> plan) {
    detail::permute(&C->impl(), A.impl(), plan);
}

// Sort with default values, no smart pointers
/**
 * @brief Permutes the elements of a tensor and puts it into an output tensor.
 *
 * This function uses HPTT, which can only handle out-of-place tensor transpositions. The tensors passed in should not
 * have overlapping storage.
 *
 * @param C_indices The indices for the output tensor.
 * @param C The output tensor.
 * @param A_indices The indices for the input tensor.
 * @param A The input tensor.
 * @tparam ConjA If true, conjugate the values of A as it is being permuted.
 */
template <bool ConjA = false, NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, ObjectC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute<ConjA>(0, C_indices, C, 1, A_indices, A);
}

// Sort with default values, two smart pointers
template <bool ConjA = false, SmartPointer SmartPointerA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute<ConjA>(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <bool ConjA = false, SmartPointer SmartPointerA, NotASmartPointer PointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, PointerC *C, std::tuple<AIndices...> const &A_indices, SmartPointerA const &A) {
    permute<ConjA>(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <bool ConjA = false, NotASmartPointer ObjectA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void permute(std::tuple<CIndices...> const &C_indices, SmartPointerC *C, std::tuple<AIndices...> const &A_indices, ObjectA const &A) {
    permute<ConjA>(0, C_indices, C->get(), 1, A_indices, A);
}

template <bool ConjA = false, BlockTensorConcept AType, BlockTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        auto       &C_block = (*C)[i];
        auto const &A_block = A[i];
        permute<ConjA>(UC_prefactor, C_indices, &C_block, UA_prefactor, A_indices, A_block);
    }
}

template <bool ConjA = false, CoreTensorConcept AType, CoreTensorConcept CType, typename... CIndices, typename... AIndices>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
    }
std::shared_ptr<hptt::Transpose<typename AType::ValueType>> compile_permute(std::tuple<CIndices...> const &C_indices, CType *C,
                                                                            std::tuple<AIndices...> const &A_indices, AType const &A,
                                                                            hptt::SelectionMethod method = hptt::ESTIMATE) {
    return compile_permute(0.0, C_indices, C, 1.0, A_indices, A, method);
}

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, TensorConcept AType, TensorConcept BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
inline auto get_grid_ranges_for_many(CType const &C, std::tuple<CIndices...> const &C_indices, AType const &A,
                                     std::tuple<AIndices...> const &A_indices,
                                     std::tuple<AllUniqueIndices...> const & /*All_unique_indices*/) {
    return std::array{get_grid_ranges_for_many_a<AllUniqueIndices, 0>(C, C_indices, A, A_indices)...};
}

template <bool ConjA = false, TiledTensorConcept AType, TiledTensorConcept CType, typename... CIndices, typename... AIndices, typename U>
    requires requires {
        requires sizeof...(CIndices) == sizeof...(AIndices);
        requires sizeof...(CIndices) == CType::Rank;
        requires sizeof...(AIndices) == AType::Rank;
        requires SameUnderlyingAndRank<AType, CType>;
        requires InSamePlace<AType, CType>;
        requires std::is_arithmetic_v<U> || IsComplexV<U>;
    }
void permute(U const UC_prefactor, std::tuple<CIndices...> const &C_indices, CType *C, U const UA_prefactor,
             std::tuple<AIndices...> const &A_indices, AType const &A) {

    using ADataType        = typename AType::ValueType;
    using CDataType        = typename CType::ValueType;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t CRank = CType::Rank;

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size_v<decltype(unique_indices)>>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size_v<decltype(unique_indices)>> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, CRank> C_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < CRank; i++) {
            C_tile_index[i] = unique_index_table[C_index_table[i]];
        }

        if (!A.has_tile(A_tile_index) || A.has_zero_size(A_tile_index) || C->has_zero_size(C_tile_index)) {
            continue;
        }

        C->lock();
        auto &C_tile = C->tile(C_tile_index);
        C->unlock();
        C_tile.lock();
        permute<ConjA>(UC_prefactor, C_indices, &C_tile, UA_prefactor, A_indices, A.tile(A_tile_index));
        C_tile.unlock();
    }
}

template <bool ConjA = false, MatrixConcept CType, MatrixConcept AType>
void transpose(CType *C, AType const &A) {
    permute<ConjA>(0.0, einsums::Indices{index::i, index::j}, C, 1.0, einsums::Indices{index::j, index::i}, A);
}

} // namespace einsums::tensor_algebra
