//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Unfold.hpp
/// @brief The mode-n unfolding (matricization) of a tensor, on rank-erased tensors.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorPermute/Permute.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_permute)

/**
 * @brief Write the mode-@p mode unfolding of @p A into @p C.
 *
 * @f$ C(a_{mode}, z) = A(a_0, \ldots, a_{N-1}) @f$, where the column @f$ z @f$ runs over the
 * remaining modes in their order with the first varying fastest:
 * @f$ z = a_{r_1} + d_{r_1} (a_{r_2} + d_{r_2} (\ldots)) @f$. This is the unfolding of Kolda and
 * Bader, and the column order is fixed: it does not depend on either tensor's memory layout.
 *
 * It is a permutation that brings @p mode to the front, written through a view of @p C that splits
 * its column axis into the remaining modes, so no reshaped copy is made.
 *
 * @code
 * tensor_permute::unfold(1, &C, A);  // A is 3 x 4 x 5, C is 4 x 15
 * @endcode
 *
 * @throws RankError when @p C is not rank 2, or @p mode is not an axis of @p A.
 * @throws DimensionError when @p C is not @p A's extent along @p mode by the product of the others.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void unfold(std::size_t mode, einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A) {
    LabeledSection("mode-{} unfold", mode);

    // One character per axis for the permute kernel: a-z then A-Z.
    constexpr std::size_t max_rank = 52;
    std::size_t const     rank     = A.rank();
    if (C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(RankError, "unfold: the output must be rank 2, not rank {}", C->rank());
    }
    if (mode >= rank) {
        EINSUMS_THROW_EXCEPTION(RankError, "unfold: mode {} is not an axis of a rank-{} tensor", mode, rank);
    }
    if (rank > max_rank) {
        EINSUMS_THROW_EXCEPTION(RankError, "unfold: a rank-{} tensor has more axes than the {} this unfolds", rank, max_rank);
    }

    std::size_t columns = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (axis != mode) {
            columns *= A.dim(axis);
        }
    }
    if (C->dim(0) != A.dim(mode) || C->dim(1) != columns) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "unfold: the mode-{} unfolding of a tensor with {} along that mode is {} x {}, not {} x {}",
                                mode, A.dim(mode), A.dim(mode), columns, C->dim(0), C->dim(1));
    }

    auto const letter = [](std::size_t axis) { return static_cast<char>(axis < 26 ? 'a' + axis : 'A' + (axis - 26)); };

    // The view: C's row axis is the mode, and its column axis is split into the remaining modes,
    // the first fastest, by giving each the stride its position in z implies.
    struct Axis {
        char        letter;
        std::size_t dim;
        std::size_t stride;
    };
    std::string       a_letters;
    std::vector<Axis> remaining;
    std::size_t       column_stride = C->stride(1);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        a_letters.push_back(letter(axis));
        if (axis == mode) {
            continue;
        }
        remaining.push_back({letter(axis), A.dim(axis), column_stride});
        column_stride *= A.dim(axis);
    }
    // The kernel takes a padded column-major or row-major layout, its strides monotone in axis
    // order. A column-major C gives the remaining modes increasing strides as listed; a row-major
    // C needs them listed last first, which moves no element, only the order the view names them.
    if (C->stride(0) > C->stride(1)) {
        std::ranges::reverse(remaining);
    }

    std::string              view_letters(1, letter(mode));
    std::vector<std::size_t> view_dims{A.dim(mode)};
    std::vector<std::size_t> view_strides{C->stride(0)};
    for (auto const &axis : remaining) {
        view_letters.push_back(axis.letter);
        view_dims.push_back(axis.dim);
        view_strides.push_back(axis.stride);
    }

    einsums::detail::TensorImpl<T> view(C->data(), view_dims, view_strides);
    detail::permute(T{0}, view_letters, &view, T{1}, a_letters, A);
}

/**
 * @brief Write the mode-@p mode unfolding of @p A into @p C, on tensors.
 *
 * @versionadded{2.0.0}
 */
template <HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void unfold(std::size_t mode, CType *C, AType const &A) {
    unfold(mode, &C->impl(), A.impl());
}

EINSUMS_NAMESPACE_END(tensor_permute)
