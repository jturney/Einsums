//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A transpose built from bad arguments throws. It used to log and call exit(-1), which ended the
// caller's process, a Python interpreter included, and it indexed the size arrays with the
// permutation before checking it, so an out-of-range entry was an out-of-bounds read first.

#include <Einsums/HPTT/HPTT.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

void make_plan(std::vector<int> const &perm, std::vector<size_t> const &size, bool row_major = false) {
    size_t count = 1;
    for (auto s : size)
        count *= s;
    std::vector<float> A(count == 0 ? 1 : count), B(count == 0 ? 1 : count);
    (void)einsums::hptt::create_plan(perm.data(), static_cast<int>(perm.size()), 1.0f, A.data(), size.data(), nullptr, 0.0f, B.data(),
                                     nullptr, einsums::hptt::ESTIMATE, 1, nullptr, row_major);
}

} // namespace

TEST_CASE("HPTT rejects a permutation that is not one", "[hptt]") {
    for (bool row_major : {false, true}) {
        CAPTURE(row_major);
        CHECK_THROWS_AS(make_plan({0, 0}, {3, 4}, row_major), std::invalid_argument);  // repeated
        CHECK_THROWS_AS(make_plan({0, -1}, {3, 4}, row_major), std::invalid_argument); // what a missing letter becomes
        CHECK_THROWS_AS(make_plan({0, 2}, {3, 4}, row_major), std::invalid_argument);  // out of range
    }
}

TEST_CASE("HPTT rejects a zero extent", "[hptt]") {
    CHECK_THROWS_AS(make_plan({1, 0}, {3, 0}), std::invalid_argument);
}

TEST_CASE("HPTT rejects a zero-rank transpose", "[hptt]") {
    CHECK_THROWS_AS(make_plan({}, {}), std::invalid_argument);
}

TEST_CASE("HPTT still builds a valid plan", "[hptt]") {
    CHECK_NOTHROW(make_plan({1, 0}, {3, 4}));
    CHECK_NOTHROW(make_plan({2, 0, 1}, {2, 3, 4}, true));
}
