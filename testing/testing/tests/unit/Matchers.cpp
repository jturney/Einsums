//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The comparison matchers are shared by every module's tests. A defect here
// does not fail: it silently changes what several hundred assertions mean, in
// whichever direction the defect leans. These cases pin the contract itself.

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("WithinRel tolerance is relative with a floor", "[testing][matchers]", float, double) {
    // An ordinary target: the tolerance is a fraction of the target itself.
    {
        TestType const target = TestType{100};
        REQUIRE_THAT(TestType(100.05), CheckWithinRel(target, 0.001));
        REQUIRE_THAT(TestType(99.95), CheckWithinRel(target, 0.001));
        REQUIRE_THAT(TestType(101.0), !CheckWithinRel(target, 0.001));
    }

    // The regression. A sum carries the rounding of the terms that went into
    // it, not of what they added up to, so a contraction of order-one operands
    // that cancels to near zero is still only accurate to the absolute error of
    // those operands. Before the floor existed this comparison demanded nine
    // significant digits of a float32 result that has none of them, and the
    // TensorAlgebra Transpose suite failed roughly one run in nine hundred,
    // whenever the random draw put an element this close to zero.
    {
        TestType const target   = TestType{1e-6};
        TestType const computed = TestType{1e-6} + TestType{6e-8};
        REQUIRE_THAT(computed, CheckWithinRel(target, 0.001));
    }

    // The floor does not become a licence to be wrong. An absolute miss larger
    // than the tolerance still fails, however small the target.
    {
        TestType const target = TestType{1e-6};
        REQUIRE_THAT(TestType(0.5), !CheckWithinRel(target, 0.001));
    }

    // A target of exactly zero and a target just beside it are the same rule.
    // The discontinuity between them is what the defect was.
    {
        REQUIRE_THAT(TestType(5e-4), CheckWithinRel(TestType{0}, 0.001));
        REQUIRE_THAT(TestType(5e-4), CheckWithinRel(TestType{1e-9}, 0.001));
        REQUIRE_THAT(TestType(2.0), !CheckWithinRel(TestType{0}, 0.001));
    }
}

TEMPLATE_TEST_CASE("WithinRel compares complex by magnitude", "[testing][matchers]", std::complex<float>, std::complex<double>) {
    using Real = RemoveComplexT<TestType>;

    REQUIRE_THAT(TestType(Real{3}, Real{4}), CheckWithinRel(TestType(Real{3}, Real{4}), 0.001));
    // Same modulus, different argument: the difference is what is measured, not
    // the difference of the moduli, so this must not pass.
    REQUIRE_THAT(TestType(Real{4}, Real{3}), !CheckWithinRel(TestType(Real{3}, Real{4}), 0.001));
    // The near-zero floor applies to the complex overload too.
    REQUIRE_THAT(TestType(Real{1e-6}, Real{6e-8}), CheckWithinRel(TestType(Real{1e-6}, Real{0}), 0.001));
}

TEMPLATE_TEST_CASE("WithinMagnitude scales with the terms, not the result", "[testing][matchers]", float, double) {
    // The caller accumulated terms of order one; an error of that size is
    // expected no matter how completely they cancelled.
    REQUIRE_THAT(TestType(1e-6), CheckWithinMagnitude(TestType{0}, 4.0, 0.001));
    // A magnitude below one does not tighten the check past the floor, so a
    // caller that happens to sum very small terms is not held to an impossible
    // standard.
    REQUIRE_THAT(TestType(5e-4), CheckWithinMagnitude(TestType{0}, 1e-9, 0.001));
    // It is still a real bound: an error larger than tolerance * magnitude fails.
    REQUIRE_THAT(TestType(1.0), !CheckWithinMagnitude(TestType{0}, 4.0, 0.001));
}
