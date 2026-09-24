//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The mode-n unfolding on rank-erased tensors. Every expected value comes from index arithmetic
// over the raw storage: the source's multi-index is walked directly and its column computed from
// the definition, so nothing above this module is used to check it.

#include <Einsums/Errors/Error.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorPermute/Unfold.hpp>

#include <complex>
#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::detail::TensorImpl;
namespace tp = einsums::tensor_permute;

namespace {

/// Column-major strides for @p dims.
std::vector<size_t> column_major(std::vector<size_t> const &dims) {
    std::vector<size_t> strides(dims.size());
    size_t              stride = 1;
    for (size_t axis = 0; axis < dims.size(); ++axis) {
        strides[axis] = stride;
        stride *= dims[axis];
    }
    return strides;
}

/// Row-major strides for @p dims.
std::vector<size_t> row_major(std::vector<size_t> const &dims) {
    std::vector<size_t> strides(dims.size());
    size_t              stride = 1;
    for (size_t axis = dims.size(); axis-- > 0;) {
        strides[axis] = stride;
        stride *= dims[axis];
    }
    return strides;
}

template <typename T>
T value_for(size_t n) {
    if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
        return T{static_cast<typename T::value_type>(n + 1), static_cast<typename T::value_type>(n) - 2};
    } else {
        return static_cast<T>(n + 1);
    }
}

/// Unfold @p dims along every mode into an output of each memory order, and compare every element
/// with the definition: C(a_mode, z), z = a_r1 + d_r1 * (a_r2 + d_r2 * ...).
template <typename T>
void check_all_modes(std::vector<size_t> const &dims, std::vector<size_t> const &a_strides) {
    size_t size = 1;
    for (auto const d : dims) {
        size *= d;
    }
    std::vector<T> a_data(size);
    for (size_t n = 0; n < size; ++n) {
        a_data[n] = value_for<T>(n);
    }
    TensorImpl<T> const A(a_data.data(), dims, a_strides);

    for (size_t mode = 0; mode < dims.size(); ++mode) {
        size_t const rows    = dims[mode];
        size_t const columns = size / rows;
        for (bool const c_row_major : {false, true}) {
            CAPTURE(mode, c_row_major);
            std::vector<size_t> const c_dims{rows, columns};
            std::vector<T>            c_data(size, T{-99});
            TensorImpl<T>             C(c_data.data(), c_dims, c_row_major ? row_major(c_dims) : column_major(c_dims));

            tp::unfold(mode, &C, A);

            // Walk A's multi-index, first axis fastest.
            std::vector<size_t> index(dims.size(), 0);
            for (size_t n = 0; n < size; ++n) {
                size_t a_offset = 0;
                size_t column   = 0;
                size_t weight   = 1;
                for (size_t axis = 0; axis < dims.size(); ++axis) {
                    a_offset += index[axis] * a_strides[axis];
                    if (axis != mode) {
                        column += index[axis] * weight;
                        weight *= dims[axis];
                    }
                }
                size_t const c_offset = index[mode] * C.stride(0) + column * C.stride(1);
                REQUIRE(c_data[c_offset] == a_data[a_offset]);

                for (size_t axis = 0; axis < dims.size(); ++axis) {
                    if (++index[axis] < dims[axis]) {
                        break;
                    }
                    index[axis] = 0;
                }
            }
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("unfold - every mode matches the definition", "[TensorPermute][unfold]", float, double, std::complex<float>,
                   std::complex<double>) {
    SECTION("rank 1") {
        check_all_modes<TestType>({5}, {1});
    }
    SECTION("rank 2") {
        check_all_modes<TestType>({3, 4}, column_major({3, 4}));
    }
    SECTION("rank 3, column-major source") {
        check_all_modes<TestType>({3, 4, 5}, column_major({3, 4, 5}));
    }
    SECTION("rank 3, row-major source") {
        // The column order is the definition's, not the source's memory order.
        check_all_modes<TestType>({3, 4, 5}, row_major({3, 4, 5}));
    }
    SECTION("rank 4") {
        check_all_modes<TestType>({2, 3, 2, 4}, column_major({2, 3, 2, 4}));
    }
}

TEST_CASE("unfold - a zero extent is an empty unfolding", "[TensorPermute][unfold]") {
    std::vector<double>      a_data;
    std::vector<double>      c_data;
    TensorImpl<double> const A(a_data.data(), std::vector<size_t>{3, 0, 2}, column_major({3, 0, 2}));
    TensorImpl<double>       C(c_data.data(), std::vector<size_t>{3, 0}, column_major({3, 0}));
    CHECK_NOTHROW(tp::unfold(0, &C, A));
}

TEST_CASE("unfold - rejects a mismatched output", "[TensorPermute][unfold]") {
    std::vector<double>      a_data(24);
    std::vector<double>      c_data(24);
    TensorImpl<double> const A(a_data.data(), std::vector<size_t>{2, 3, 4}, column_major({2, 3, 4}));

    SECTION("a mode past the last axis") {
        TensorImpl<double> C(c_data.data(), std::vector<size_t>{2, 12}, column_major({2, 12}));
        CHECK_THROWS_AS(tp::unfold(3, &C, A), einsums::RankError);
    }
    SECTION("an output that is not a matrix") {
        TensorImpl<double> C(c_data.data(), std::vector<size_t>{2, 3, 4}, column_major({2, 3, 4}));
        CHECK_THROWS_AS(tp::unfold(0, &C, A), einsums::RankError);
    }
    SECTION("an output of the wrong shape") {
        // The mode-1 unfolding is 3 x 8; 2 x 12 is the mode-0 one.
        TensorImpl<double> C(c_data.data(), std::vector<size_t>{2, 12}, column_major({2, 12}));
        CHECK_THROWS_AS(tp::unfold(1, &C, A), einsums::DimensionError);
    }
}
