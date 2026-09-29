//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// int32 <-> float conversion: rounding to nearest one way, truncation toward zero the other.

#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Operations.hpp>

#include <cstdint>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEST_CASE("convert<float> of int32 rounds to nearest, ties to even", "[simd][convert]") {
    constexpr int L = Vec<int32_t>::lanes;
    int32_t       in[L];
    float         out[L];
    for (int i = 0; i < L; ++i) {
        in[i] = (i % 2 ? -1 : 1) * (i * 1000003 + 17);
    }
    storeu(out, convert<float>(loadu(in)));
    for (int i = 0; i < L; ++i) {
        CHECK(out[i] == static_cast<float>(in[i]));
    }
    // 2^24 + 1 and 2^24 + 3 are the first odd integers float cannot hold: the ties go to the even
    // neighbours 2^24 and 2^24 + 4.
    storeu(out, convert<float>(broadcast(int32_t{16777217})));
    CHECK(out[0] == 16777216.0f);
    storeu(out, convert<float>(broadcast(int32_t{16777219})));
    CHECK(out[0] == 16777220.0f);
}

TEST_CASE("convert<int32_t> of float truncates toward zero", "[simd][convert]") {
    constexpr int L = Vec<float>::lanes;
    float         in[L];
    int32_t       out[L];
    for (int i = 0; i < L; ++i) {
        in[i] = (i % 2 ? -1.0f : 1.0f) * (static_cast<float>(i) + 0.75f);
    }
    storeu(out, convert<int32_t>(loadu(in)));
    for (int i = 0; i < L; ++i) {
        INFO("lane " << i << ": " << in[i]);
        CHECK(out[i] == static_cast<int32_t>(in[i]));
    }
    storeu(out, convert<int32_t>(broadcast(-0.5f)));
    CHECK(out[0] == 0);
    storeu(out, convert<int32_t>(broadcast(2147483520.0f))); // the largest float below 2^31
    CHECK(out[0] == 2147483520);
}
