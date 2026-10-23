//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Testing/Layouts.hpp>

#include <algorithm>
#include <complex>
#include <set>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums::testing;

// Tests that compare an operation across layouts trust three things: every layout holds the same
// logical values, the layouts really differ, and a write outside a layout shows in its gaps.
TEMPLATE_TEST_CASE("layouts hold the same values in distinct memory orders", "[testing][layouts]", float, std::complex<double>, int) {
    using T = TestType;
    for (auto const &dims : layout_shapes()) {
        INFO("shape " << shape_name(dims));
        auto layouts_of_shape = layouts<T>(dims, 7);

        std::set<std::vector<size_t>> distinct;
        for (auto &layout : layouts_of_shape) {
            INFO(layout.name);
            distinct.insert(layout.strides);
            CHECK(layout.gaps_untouched());
            auto const impl = layout.impl();
            REQUIRE(impl.rank() == dims.size());
            for (size_t d = 0; d < dims.size(); d++) {
                CHECK(impl.dim(d) == dims[d]);
            }
            for_each_index(dims, [&](std::vector<size_t> const &index) {
                T const want = layout_value<T>(7, logical_index(dims, index));
                CHECK(layout.at(index) == want);
                CHECK(want != T(0));
                CHECK(want != Layout<T>::sentinel());
            });
        }
        CHECK(distinct.size() == layouts_of_shape.size());
    }
}

TEST_CASE("a write outside a layout shows in its gaps", "[testing][layouts]") {
    auto all = layouts<double>({3, 4}, 1);
    for (auto &layout : all) {
        INFO(layout.name);
        if (layout.buffer.size() == 12) {
            continue; // dense: no gaps
        }
        auto gap = std::find(layout.buffer.begin(), layout.buffer.end(), Layout<double>::sentinel());
        REQUIRE(gap != layout.buffer.end());
        *gap = 0.5;
        CHECK_FALSE(layout.gaps_untouched());
    }
}

TEST_CASE("for_each_layout_combination visits every combination once", "[testing][layouts]") {
    size_t calls = 0, expected = 0;
    for (auto const &dims : layout_shapes()) {
        expected += layouts<float>(dims, 1).size() * layouts<double>(dims, 2).size();
    }
    for_each_layout_combination<float, double>([&](std::vector<size_t> const &, Layout<float>, Layout<double>) { calls++; });
    CHECK(calls == expected);
}
