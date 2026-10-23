//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/CompilerSpecific.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

// The + reduction declared for std::complex starts each thread's private sum at zero. It started them
// from the variable's own value, which added that value once per thread: a block contraction into a
// complex scalar that already held a value came out with the value counted as many times as there
// were threads.
TEMPLATE_TEST_CASE("an OpenMP + reduction over complex adds the starting value once", "[config][openmp]", std::complex<float>,
                   std::complex<double>) {
    using T = TestType;
    T sum(5, -2);
    EINSUMS_OMP_PRAGMA(parallel for num_threads(4) reduction(+: sum))
    for (int i = 0; i < 100; i++) {
        sum += T(1, 1);
    }
    CHECK(sum == T(105, 98));
}
