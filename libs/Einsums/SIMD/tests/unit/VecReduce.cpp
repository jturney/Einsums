//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Horizontal reductions. The data are small integers, so every partial sum is exact and the
// result does not depend on the order a backend combines lanes in.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <algorithm>
#include <cstdint>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("reduce_add, reduce_min and reduce_max over every lane", "[simd][reduce]", float, double, int32_t) {
    constexpr int L = Vec<TestType>::lanes;
    // Each rotation puts the minimum and the maximum in a different lane, so a reduction that
    // skips a lane position fails for some rotation.
    for (int rot = 0; rot < L; ++rot) {
        TestType data[L];
        for (int i = 0; i < L; ++i) {
            data[(i + rot) % L] = TestType((i * 7) % 11) - TestType(5);
        }
        data[rot] = TestType(-40);
        data[(rot + L - 1) % L] += TestType(L == 1 ? 0 : 60);

        TestType sum = 0;
        for (int i = 0; i < L; ++i) {
            sum += data[i];
        }
        auto const v = loadu(data);
        INFO("rotation " << rot);
        CHECK(reduce_add(v) == sum);
        CHECK(reduce_min(v) == *std::min_element(data, data + L));
        CHECK(reduce_max(v) == *std::max_element(data, data + L));
    }
}

TEST_CASE("reduce_add of an int32 Vec wraps like the scalar sum", "[simd][reduce]") {
    constexpr int L = Vec<int32_t>::lanes;
    int32_t       data[L];
    uint32_t      sum = 0;
    for (int i = 0; i < L; ++i) {
        data[i] = INT32_MAX - i;
        sum += static_cast<uint32_t>(data[i]);
    }
    CHECK(reduce_add(loadu(data)) == static_cast<int32_t>(sum));
}
