//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The test main reseeds the einsums random engine at every pass through a test
// case. Before it did, the engine was seeded once per binary, so the tensors a
// test drew depended on every test that ran before it: a tolerance-edge failure
// in CI could not be reproduced by rerunning the failing test with the printed
// seed. These cases pin that the engine state at the start of a test is a pure
// function of the run seed and the test name.

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <string>

#include <Einsums/Testing.hpp>

namespace {

auto draws_after_seeding(std::uint32_t seed) -> std::vector<double> {
    einsums::seed_random(seed);
    auto                A = einsums::create_random_tensor<double>("A", 3, 4);
    std::vector<double> out(A.data(), A.data() + A.size());
    return out;
}

auto current_test_seed() -> std::uint32_t {
    return einsums::testing::test_case_seed(Catch::getSeed(), Catch::getResultCapture().getCurrentTestName());
}

} // namespace

TEST_CASE("test case seed is FNV-1a over the run seed and the name", "[testing][random]") {
    // The Python suite computes the same function; these values pin both.
    STATIC_REQUIRE(einsums::testing::test_case_seed(0, "") == 1268118805U);
    REQUIRE(einsums::testing::test_case_seed(12345, "a test") != einsums::testing::test_case_seed(12345, "another test"));
    REQUIRE(einsums::testing::test_case_seed(12345, "a test") != einsums::testing::test_case_seed(12346, "a test"));
}

TEST_CASE("the engine starts each test from the run seed and the test name", "[testing][random]") {
    // Draw first: nothing else in this test has touched the engine yet.
    auto                A = einsums::create_random_tensor<double>("A", 3, 4);
    std::vector<double> first(A.data(), A.data() + A.size());

    REQUIRE(first == draws_after_seeding(current_test_seed()));
    REQUIRE(first != draws_after_seeding(current_test_seed() + 1));
}

TEST_CASE("every section pass starts from the same engine state", "[testing][random]") {
    // Code above the sections runs once per pass. Drawing here and comparing
    // against the seed shows each pass starts fresh, so running one section on
    // its own with -c draws the same tensors it drew in the full run.
    auto                A = einsums::create_random_tensor<double>("A", 3, 4);
    std::vector<double> first(A.data(), A.data() + A.size());

    SECTION("first") {
        REQUIRE(first == draws_after_seeding(current_test_seed()));
    }
    SECTION("second") {
        REQUIRE(first == draws_after_seeding(current_test_seed()));
    }
}
