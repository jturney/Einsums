//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Permute.hpp
/// @brief Tensor permutation with the axes named by characters, on rank-erased tensors.
///
/// The axes of each operand are spelled as a string, one character per axis: ``"ji"`` for the
/// output and ``"ij"`` for the input is a matrix transpose. Nothing here is templated on the
/// index letters, so a permutation can be built at run time and rewritten as data, which is what
/// the ComputeGraph string specs need. The kernel is HPTT, through a per-thread plan cache.

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/StringUtil/StringOps.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorPermute/Detail/HpttPlanCache.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_permute)

/**
 * @brief Build the HPTT plan for @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$ without running it.
 *
 * @p C_indices and @p A_indices name the axes of each operand, one character per axis; the same
 * characters must appear in both. The plan comes from a per-thread cache keyed on the shapes, so
 * building the same permutation again is cheap. Run it with the plan's own ``execute()``, or rebind
 * it to new data of the same shapes with the three-argument ``permute``.
 *
 * @tparam ConjA If true, conjugate the elements of @p A as they are permuted.
 * @param beta The scale applied to the existing contents of @p C.
 * @param C_indices The axes of @p C.
 * @param C The output tensor.
 * @param alpha The scale applied to the permuted @p A.
 * @param A_indices The axes of @p A.
 * @param A The input tensor. It must not share storage with @p C.
 * @param method How hard HPTT searches for a fast plan.
 * @return The plan, or null when either operand is empty: there is nothing to permute, and running
 *         the null plan through the three-argument ``permute`` does nothing.
 * @throws RankError when an index string does not have one letter per axis of its operand, or the
 *         two strings do not name the same axes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
std::shared_ptr<hptt::Transpose<T>> compile_permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha,
                                                    std::string const &A_indices, einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod method = hptt::ESTIMATE) {
    LabeledSection("permute: {} <- {}", C_indices, A_indices);

    // One letter per axis, and the same letters on both sides. Checked in both directions: a letter
    // on either side alone would reach HPTT as an invalid permutation.
    if (C_indices.size() != C->rank() || A_indices.size() != A.rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "permute: '{}' names {} axes of a rank-{} output and '{}' names {} axes of a rank-{} input",
                                C_indices, C_indices.size(), C->rank(), A_indices, A_indices.size(), A.rank());
    }
    if (!difference(A_indices, C_indices).empty() || !difference(C_indices, A_indices).empty()) {
        EINSUMS_THROW_EXCEPTION(RankError, "permute: '{}' and '{}' do not name the same axes", C_indices, A_indices);
    }

    // An empty operand has nothing to permute, and HPTT rejects a zero extent. There is no plan.
    if (A.size() == 0 || C->size() == 0) {
        return nullptr;
    }

    // Calculate reversed indices.
    BufferVector<int>   perms(A.rank());
    std::vector<size_t> size(A.rank());
    std::vector<size_t> outerSizeA(A.rank());
    std::vector<size_t> offsetA(A.rank());
    std::vector<size_t> outerSizeC(A.rank());
    std::vector<size_t> offsetC(A.rank());

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

        // perms[i] = position of C's i-th index in A, matching the tuple
        // overload's find_type_with_position(C_indices, A_indices). The
        // argument order used to be swapped, producing the INVERSE
        // permutation - invisible for involutions (self-inverse), wrong for
        // any cyclic permutation.
        find_char_with_position(C_indices, A_indices, &perms);
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
        find_char_with_position(reverse(C_indices), A_indices, &perms);
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

        find_char_with_position(C_indices, A_indices, &perms);
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
        find_char_with_position(reverse(C_indices), A_indices, &perms);
        auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), A.rank(), alpha, A.data(), size.data(), outerSizeA.data(),
                                                       offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                       innerStrideC, false, method);
        plan->set_conj_a(ConjA);

        return plan;
    }
}

/**
 * @brief Run a plan from @ref compile_permute on new data of the same shapes. A null plan, which
 * compile_permute returns for empty operands, does nothing.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void permute(einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A, std::shared_ptr<hptt::Transpose<T>> plan) {
    if (plan == nullptr) {
        return; // compile_permute's answer for empty operands
    }
    plan->set_input_ptr(A.data());
    plan->set_output_ptr(C->data());

    plan->execute();
}

/**
 * @brief Compute @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$, with the axes named by characters.
 *
 * @code
 * tensor_permute::permute(0.0, "kij", &C, 1.0, "ijk", A);  // C(k, i, j) = A(i, j, k)
 * @endcode
 *
 * Empty operands are valid and leave @p C as it is.
 *
 * @tparam ConjA If true, conjugate the elements of @p A as they are permuted.
 * @throws RankError when an index string does not have one letter per axis of its operand, or the
 *         two strings do not name the same axes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
void permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha, std::string const &A_indices,
             einsums::detail::TensorImpl<T> const &A) {
    auto plan = compile_permute<ConjA>(beta, C_indices, C, alpha, A_indices, A);
    if (plan != nullptr) {
        plan->execute();
    }
}

/**
 * @brief Compute @f$ C = A^T @f$ for rank-2 tensors.
 *
 * @throws RankError when either operand is not rank 2.
 * @throws DimensionError when @p C is smaller than the transposed @p A.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
void transpose(einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A) {
    if (C->rank() != 2 || A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(RankError, "transpose needs rank-2 tensors, got output rank {} and input rank {}", C->rank(), A.rank());
    }
    if (C->dim(0) < A.dim(1) || C->dim(1) < A.dim(0)) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "transpose: the output tensor is smaller than the transposed input");
    }
    permute<ConjA>(T{0}, "ij", C, T{1}, "ji", A);
}

/**
 * @brief A tensor type that exposes the TensorImpl it wraps, as every dense tensor, view and
 *        RuntimeTensor does.
 *
 * The overloads below take such tensors directly, so callers never reach for ``impl()``, while
 * this module still depends on nothing above TensorImpl.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
concept HasTensorImpl = requires(T &t, T const &ct) {
    typename T::ValueType;
    t.impl();
    ct.impl();
};

/**
 * @brief Compute @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$ on tensors, with the axes named
 *        by characters.
 *
 * @code
 * tensor_permute::permute(0.0, "kij", &C, 1.0, "ijk", A);  // C(k, i, j) = A(i, j, k)
 * @endcode
 *
 * @throws RankError when an index string does not have one letter per axis of its operand, or the
 *         two strings do not name the same axes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void permute(typename AType::ValueType beta, std::string const &C_indices, CType *C, typename AType::ValueType alpha,
             std::string const &A_indices, AType const &A) {
    permute<ConjA>(beta, C_indices, &C->impl(), alpha, A_indices, A.impl());
}

/**
 * @brief Compute @f$ C = A^T @f$ for rank-2 tensors.
 *
 * @throws RankError when either operand is not rank 2.
 * @throws DimensionError when @p C is smaller than the transposed @p A.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void transpose(CType *C, AType const &A) {
    transpose<ConjA>(&C->impl(), A.impl());
}

EINSUMS_NAMESPACE_END(tensor_permute)
