//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// strided_for_each, the walk behind the elementwise fallbacks for operands of different layouts.
//
// The fallbacks used to recurse over the axes in index order with one scalar operation per leaf
// call, which for column-major tensors puts the largest stride innermost: a dense output beside a
// strided view ran at about 8 ns per element. The walk orders the axes by the written operand's
// strides and hands each innermost run to a flat loop.

#include <Einsums/TensorImpl/StridedForEach.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::detail::strided_for_each;
using einsums::detail::StridedOperand;

namespace {

using Strides = std::vector<std::size_t>;

/// A buffer holding, at each element a layout reaches, that element's index-order position.
std::vector<int> coded(std::vector<std::size_t> const &dims, Strides const &strides) {
    std::size_t span = 1;
    for (std::size_t d = 0; d < dims.size(); ++d) {
        span += (dims[d] - 1) * strides[d];
    }
    std::vector<int>         buffer(span, -1);
    std::vector<std::size_t> index(dims.size(), 0);
    for (int code = 0;; ++code) {
        std::size_t offset = 0;
        for (std::size_t d = 0; d < dims.size(); ++d) {
            offset += index[d] * strides[d];
        }
        buffer[offset] = code;
        std::size_t d  = dims.size();
        while (d > 0) {
            --d;
            if (++index[d] < dims[d]) {
                break;
            }
            index[d] = 0;
            if (d == 0) {
                return buffer;
            }
        }
    }
}

template <typename T>
StridedOperand<T, Strides> op(T *data, Strides const &strides) {
    return {.data = data, .strides = &strides};
}

} // namespace

TEST_CASE("strided_for_each pairs the same element of every layout, once each", "[tensor][strided]") {
    // Extents 3, 1, 4, 5. The written operand is column-major and dense; the second is row-major;
    // the third is a view taking every other row of a (6, 2, 9, 5) column-major parent, so it has
    // gaps; the fourth shares the first's layout on the first two non-unit axes and differs after,
    // which exercises the merge.
    std::vector<std::size_t> const dims{3, 1, 4, 5};
    Strides const                  col{1, 3, 3, 12};
    Strides const                  row{20, 20, 5, 1};
    Strides const                  gaps{2, 6, 12, 108};
    Strides const                  partly{1, 3, 3, 15};

    auto out = coded(dims, col);
    auto a   = coded(dims, row);
    auto b   = coded(dims, gaps);
    auto c   = coded(dims, partly);

    std::vector<int> visits(60, 0);
    strided_for_each(
        dims,
        [&](int &o, int const &x, int const &y, int const &z) {
            REQUIRE(o >= 0);
            REQUIRE(x == o);
            REQUIRE(y == o);
            REQUIRE(z == o);
            ++visits[static_cast<std::size_t>(o)];
        },
        op(out.data(), col), op(a.data(), row), op(b.data(), gaps), op(c.data(), partly));

    for (int const v : visits) {
        REQUIRE(v == 1);
    }

    // Without the row-major operand, the first two non-unit axes are contiguous in both layouts and
    // merge into one run of 12; the last axis is not, and stays an outer loop.
    std::fill(visits.begin(), visits.end(), 0);
    strided_for_each(
        dims,
        [&](int &o, int const &z) {
            REQUIRE(z == o);
            ++visits[static_cast<std::size_t>(o)];
        },
        op(out.data(), col), op(c.data(), partly));
    for (int const v : visits) {
        REQUIRE(v == 1);
    }
}

TEST_CASE("strided_for_each walks the written operand in its own memory order", "[tensor][strided]") {
    // The output is row-major, so its fastest axis is the LAST index, and the input is column-major,
    // so neither index order nor the input's layout lines up with it. The output must still be
    // visited at consecutive addresses, which is what keeps the walk off its largest stride.
    std::vector<std::size_t> const dims{4, 3, 5};
    Strides const                  col{1, 4, 12};
    Strides const                  row{15, 5, 1};
    std::vector<double>            out(60, 0.0);
    std::vector<double> const      in(60, 1.0);

    std::vector<std::ptrdiff_t> order;
    strided_for_each(dims, [&](double &o, double const &) { order.push_back(&o - out.data()); }, op(out.data(), row), op(in.data(), col));

    REQUIRE(order.size() == 60);
    for (std::size_t i = 0; i < order.size(); ++i) {
        REQUIRE(std::cmp_equal(order[i], i));
    }
}

TEST_CASE("strided_for_each on empty and rank-zero shapes", "[tensor][strided]") {
    double                         x = 2.0;
    Strides const                  none{};
    Strides const                  one{1, 1};
    std::vector<double>            buffer(4, 0.0);
    int                            calls = 0;
    std::vector<std::size_t> const scalar{};
    strided_for_each(
        scalar,
        [&](double &v) {
            v += 1.0;
            ++calls;
        },
        op(&x, none));
    REQUIRE(calls == 1);
    REQUIRE(x == 3.0);

    std::vector<std::size_t> const empty{4, 0};
    strided_for_each(empty, [&](double &) { ++calls; }, op(buffer.data(), one));
    REQUIRE(calls == 1);
}
