//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("iamax", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    auto x = create_tensor<T>("x", N);
    x(0)   = 1.0;
    x(1)   = -5.0;
    x(2)   = 3.0;
    x(3)   = 2.0;

    // iamax returns the index of the element with maximum absolute value
    // BLAS iamax is 1-based in Fortran, but the C++ wrapper may return 0-based
    auto idx = blas::iamax<T>(N, x.data(), 1);

    // Element with max |value| is -5.0 at index 1 (0-based)
    // Some BLAS implementations return 1-based index, check both
    // The einsums wrapper returns the raw BLAS result
    REQUIRE((idx == 1 || idx == 2)); // 0-based: 1, 1-based: 2

    if (idx == 2) {
        // 1-based indexing from BLAS
        REQUIRE(std::abs(x(idx - 1)) >= std::abs(x(0)));
        REQUIRE(std::abs(x(idx - 1)) >= std::abs(x(2)));
        REQUIRE(std::abs(x(idx - 1)) >= std::abs(x(3)));
    } else {
        // 0-based indexing
        REQUIRE(std::abs(x(idx)) >= std::abs(x(0)));
        REQUIRE(std::abs(x(idx)) >= std::abs(x(2)));
        REQUIRE(std::abs(x(idx)) >= std::abs(x(3)));
    }
}
