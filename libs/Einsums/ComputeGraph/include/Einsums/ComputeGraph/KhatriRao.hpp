//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file KhatriRao.hpp
/// @brief The Khatri-Rao product, with its operands' axes named by string index lists.

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief The Khatri-Rao product of @p A and @p B, as a matrix.
 *
 * The indices the two operands share are matched, and every other index is an outer product:
 * @f$ R(a, b, c) = A(a, c)\,B(b, c) @f$ for the indices @f$ a @f$ only @p A names, @f$ b @f$ only
 * @p B names, and @f$ c @f$ both name. The result is that tensor as a matrix: its rows run over
 * @f$ (a, b) @f$ and its columns over @f$ c @f$, each with its first index varying fastest.
 * With one shared index this is the column-wise Kronecker product.
 *
 * The index lists follow the grammar of an einsum operand: ``"ir"`` names two axes, one per
 * character, and ``"mu,r"`` names two with a comma between them.
 *
 * @code
 * auto KR = cg::khatri_rao("ir", A, "mr", B);  // A is I x R, B is M x R; KR is (I * M) x R
 * @endcode
 *
 * @throws std::invalid_argument when an index list does not parse or names an axis twice.
 * @throws std::logic_error during graph capture, where a returning form cannot be recorded.
 * @throws RankError when an index list does not name every axis of its operand.
 * @throws DimensionError when a shared index has different extents in @p A and @p B.
 *
 * @versionadded{2.0.0}
 */
template <typename AType, typename BType>
    requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>
auto khatri_rao(std::string_view a_indices, AType const &A, std::string_view b_indices, BType const &B)
    -> Tensor<typename AType::ValueType, 2> {
    using T = typename AType::ValueType;
    detail::reject_if_capturing("cg::khatri_rao returning form cannot be used during graph capture.");
    LabeledSection0();

    // Tokenized by the spec parser itself, so an index list means what it would as an operand.
    auto parsed = parse_einsum_spec(fmt::format("{} <- {} ; {}", a_indices, a_indices, b_indices));
    if (!parsed) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::khatri_rao: {}", parsed.error().message);
    }
    std::vector<std::string> const &a = parsed->a_indices;
    std::vector<std::string> const &b = parsed->b_indices;
    if (a.size() != detail::tensor_rank(A) || b.size() != detail::tensor_rank(B)) {
        EINSUMS_THROW_EXCEPTION(RankError, "cg::khatri_rao: '{}' and '{}' name {} and {} axes of operands of rank {} and {}", a_indices,
                                b_indices, a.size(), b.size(), detail::tensor_rank(A), detail::tensor_rank(B));
    }
    for (auto const *list : {&a, &b}) {
        for (auto const &index : *list) {
            if (std::ranges::count(*list, index) > 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::khatri_rao: index '{}' names two axes of one operand", index);
            }
        }
    }

    auto const position = [](std::vector<std::string> const &list, std::string const &index) {
        return static_cast<std::size_t>(std::ranges::distance(list.begin(), std::ranges::find(list, index)));
    };

    // The result's axes: A's own, then B's own, then the shared ones, each group in operand order.
    // The first two groups are the matrix's rows, the last its columns.
    std::vector<std::string> rows_part;
    std::vector<std::string> columns_part;
    std::vector<std::size_t> row_dims;
    std::vector<std::size_t> column_dims;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::size_t const j = position(b, a[i]);
        if (j == b.size()) {
            rows_part.push_back(a[i]);
            row_dims.push_back(A.dim(i));
        } else {
            if (A.dim(i) != B.dim(j)) {
                EINSUMS_THROW_EXCEPTION(DimensionError, "cg::khatri_rao: shared index '{}' has extent {} in A and {} in B", a[i], A.dim(i),
                                        B.dim(j));
            }
            columns_part.push_back(a[i]);
            column_dims.push_back(A.dim(i));
        }
    }
    for (std::size_t j = 0; j < b.size(); ++j) {
        if (position(a, b[j]) == a.size()) {
            rows_part.push_back(b[j]);
            row_dims.push_back(B.dim(j));
        }
    }

    std::size_t rows = 1;
    for (auto const d : row_dims) {
        rows *= d;
    }
    std::size_t columns = 1;
    for (auto const d : column_dims) {
        columns *= d;
    }
    Tensor<T, 2> result{"KR product", rows, columns};

    // A view of the matrix with its row axis split into the row indices and its column axis into
    // the shared ones, so the contraction writes the matrix directly.
    std::vector<std::size_t> view_dims;
    std::vector<std::size_t> view_strides;
    std::size_t              stride = result.stride(0);
    for (auto const d : row_dims) {
        view_dims.push_back(d);
        view_strides.push_back(stride);
        stride *= d;
    }
    stride = result.stride(1);
    for (auto const d : column_dims) {
        view_dims.push_back(d);
        view_strides.push_back(stride);
        stride *= d;
    }
    RuntimeTensorView<T> view(einsums::detail::TensorImpl<T>(result.data(), view_dims, view_strides));

    std::vector<std::string> output = rows_part;
    output.insert(output.end(), columns_part.begin(), columns_part.end());
    std::string const spec = fmt::format("{} <- {} ; {}", fmt::join(output, ","), fmt::join(a, ","), fmt::join(b, ","));
    einsum(EinsumFormatString(std::string_view{spec}), T{0}, &view, T{1}, A, B);
    return result;
}

EINSUMS_NAMESPACE_END(compute_graph)
