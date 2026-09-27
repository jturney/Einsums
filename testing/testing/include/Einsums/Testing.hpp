//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#if defined(EINSUMS_WINDOWS)
#    define CATCH_CONFIG_WINDOWS_SEH
#endif
#include <algorithm>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <catch2/catch_all.hpp>

// Wraps Catch2's TEST_CASE to automatically insert a profiler zone named
// after the current test case.  Usage is identical to TEST_CASE:
//
//   EINSUMS_TEST_CASE("my test", "[tag]") { ... }
//
// clang-format off
#define EINSUMS_TEST_CASE2_(impl_name, ...)                                                                                                \
    static void impl_name();                                                                                                               \
    TEST_CASE(__VA_ARGS__) {                                                                                                               \
        LabeledSectionRuntime((Catch::getResultCapture().getCurrentTestName()));                                                       \
        impl_name();                                                                                                                       \
    }                                                                                                                                      \
    static void impl_name()
#define EINSUMS_TEST_CASE(...) EINSUMS_TEST_CASE2_(INTERNAL_CATCH_UNIQUE_NAME(einsums_test_impl_), __VA_ARGS__)

// Wraps Catch2's TEMPLATE_TEST_CASE to automatically insert a profiler zone.
// The zone name includes the type (e.g., "my test - double") since Catch2
// registers template tests as "Name - TypeName".
//
//   EINSUMS_TEMPLATE_TEST_CASE("my test", "[tag]", float, double) { ... }
//
#define EINSUMS_TEMPLATE_TEST_CASE2_(impl_name, Name, Tags, ...)                                                                           \
    template <typename TestType>                                                                                                           \
    static void impl_name();                                                                                                               \
    TEMPLATE_TEST_CASE(Name, Tags, __VA_ARGS__) {                                                                                          \
        LabeledSection("{} <{}>", Catch::getResultCapture().getCurrentTestName(), einsums::type_name<TestType>());                          \
        impl_name<TestType>();                                                                                                             \
    }                                                                                                                                      \
    template <typename TestType>                                                                                                           \
    static void impl_name()
#define EINSUMS_TEMPLATE_TEST_CASE(Name, Tags, ...) EINSUMS_TEMPLATE_TEST_CASE2_(INTERNAL_CATCH_UNIQUE_NAME(einsums_tmpl_test_impl_), Name, Tags, __VA_ARGS__)
// clang-format on

EINSUMS_NAMESPACE_BEGIN()

/// Default relative tolerance of the comparison matchers for element type @p T.
///
/// Single precision carries about seven digits, so a short contraction of order-one terms is good to
/// a few parts in 1e6 and 1e-4 leaves room for longer sums. A looser bound would pass a dropped term
/// worth a percent of the result. A test that needs more than this should compute the rounding it
/// expects rather than raise the default.
template <typename T>
constexpr double tolerance() {
    return 1e-6;
}

template <>
constexpr double tolerance<float>() {
    return 1e-4;
}

template <>
constexpr double tolerance<std::complex<float>>() {
    return 1e-4;
}

/**
 * @struct WithinStrictMatcher
 *
 * Catch2 matcher that matches the strictest range for floating point operations.
 */
template <typename T>
struct WithinStrictMatcher : public Catch::Matchers::MatcherGenericBase {};

template <>
struct WithinStrictMatcher<float> : public Catch::Matchers::MatcherGenericBase {
  private:
    float _value, _scale;

  public:
    WithinStrictMatcher(float value, float scale) : _value(value), _scale(scale) {}

    bool match(float other) const {
        // Minimum error is 5.96e-8, according to LAPACK docs.
        if (_value == 0.0f) {
            return std::abs(other) <= 5.960464477539063e-08f * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 5.960464477539063e-08f * _scale;
        }
    }

    float get_error() const { return 5.960464477539063e-08f * _scale; }

  protected:
    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<float>::convert(5.960464477539063e-08f * _scale) + " to " +
               Catch::StringMaker<float>::convert(_value);
    }
};

template <>
struct WithinStrictMatcher<double> : public Catch::Matchers::MatcherGenericBase {
  private:
    double _value, _scale;

  public:
    WithinStrictMatcher(double value, double scale) : _value(value), _scale(scale) {}

    bool match(double other) const {
        // Minimum error is 1.1e-16, according to LAPACK docs.
        if (_value == 0.0) {
            return std::abs(other) <= 1.1102230246251565e-16 * _scale;
        } else {
            return std::abs((other - _value) / _value) <= 1.1102230246251565e-16 * _scale;
        }
    }

    double get_error() const { return 1.1102230246251565e-16 * _scale; }

  protected:
    std::string describe() const override {
        return "is within a fraction of " + Catch::StringMaker<double>::convert(1.1102230246251565e-16 * _scale) + " to " +
               Catch::StringMaker<double>::convert(_value);
    }
};

template <typename T>
auto WithinStrict(T value, T scale = T{1.0}) -> WithinStrictMatcher<T> { // NOLINT
    return WithinStrictMatcher<T>{value, scale};
}

template <typename TestType>
class WithinRelMatcher : public Catch::Matchers::MatcherGenericBase {
  public:
    WithinRelMatcher(TestType value, double eps) : _target{value}, _eps{eps} {}

    // The tolerance is relative to the target, with a floor of one. A pure
    // relative test is meaningless once the target approaches zero: the
    // rounding in a sum scales with the terms that went into it, not with the
    // result, so a contraction that cancels to 1e-6 still carries the absolute
    // error of its operands and fails a relative check it has no way to meet.
    // The floor is what the zero target has always been compared against, so
    // this is the same rule either side of zero rather than a strict relative
    // test that becomes an absolute one the instant the target lands exactly on
    // it.
    bool match(TestType value) const {
        auto const scale = std::max(static_cast<double>(std::abs(_target)), 1.0);
        return static_cast<double>(std::abs(value - _target)) <= _eps * scale;
    }

  protected:
    std::string describe() const override {
        if constexpr (IsComplexV<std::remove_cvref_t<TestType>>) {
            return "and " + Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(_target.real()) +
                   ((_target.imag() < 0) ? "-" : "+") +
                   Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(std::abs(_target.imag())) + "i are within " +
                   Catch::StringMaker<double>::convert(_eps * 100) + "% of each other";
        } else {
            return "and " + Catch::StringMaker<std::remove_cvref_t<TestType>>::convert(_target) + " are within " +
                   Catch::StringMaker<double>::convert(_eps * 100) + "% of each other";
        }
    }

  private:
    TestType _target;
    double   _eps;
};

#ifdef __cpp_deduction_guides
template <typename TestType>
WithinRelMatcher(TestType, double) -> WithinRelMatcher<TestType>;
#endif

template <typename TestType>
// NOLINTNEXTLINE
WithinRelMatcher<std::remove_cvref_t<TestType>> CheckWithinRel(TestType reference, double tolerance = ::einsums::tolerance<TestType>()) {
    return WithinRelMatcher(reference, tolerance);
}

// Compares against a caller-supplied magnitude rather than against the result.
// A sum carries the rounding of the terms that went into it, so a contraction
// whose terms are order one is accurate to the tolerance times one no matter
// how completely those terms cancel. Tests that compute a reference sum can
// accumulate the magnitude of the terms alongside it and hand that in, which
// keeps the check as strict as the arithmetic allows without turning a
// cancellation into a failure.
template <typename TestType>
class WithinMagnitudeMatcher : public Catch::Matchers::MatcherGenericBase {
  public:
    WithinMagnitudeMatcher(TestType value, double magnitude, double eps) : _target{value}, _magnitude{magnitude}, _eps{eps} {}

    bool match(TestType value) const { return static_cast<double>(std::abs(value - _target)) <= _eps * std::max(_magnitude, 1.0); }

  protected:
    std::string describe() const override {
        if constexpr (IsComplexV<std::remove_cvref_t<TestType>>) {
            return "and " + Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(_target.real()) +
                   ((_target.imag() < 0) ? "-" : "+") +
                   Catch::StringMaker<RemoveComplexT<std::remove_cvref_t<TestType>>>::convert(std::abs(_target.imag())) +
                   "i differ by at most " + Catch::StringMaker<double>::convert(_eps * std::max(_magnitude, 1.0));
        } else {
            return "and " + Catch::StringMaker<std::remove_cvref_t<TestType>>::convert(_target) + " differ by at most " +
                   Catch::StringMaker<double>::convert(_eps * std::max(_magnitude, 1.0));
        }
    }

  private:
    TestType _target;
    double   _magnitude;
    double   _eps;
};

template <typename TestType>
// NOLINTNEXTLINE
WithinMagnitudeMatcher<std::remove_cvref_t<TestType>> CheckWithinMagnitude(TestType reference, double magnitude,
                                                                           double tolerance = ::einsums::tolerance<TestType>()) {
    return WithinMagnitudeMatcher<std::remove_cvref_t<TestType>>(reference, magnitude, tolerance);
}

namespace testing {

/// The four element types every tensor operation supports, as a type list for Catch2's
/// TEMPLATE_LIST_TEST_CASE:
///
///     TEMPLATE_LIST_TEST_CASE("round trip", "[tag]", einsums::testing::AllScalarTypes) { ... }
using AllScalarTypes = std::tuple<float, double, std::complex<float>, std::complex<double>>;

/// The two real element types, as a type list for Catch2's TEMPLATE_LIST_TEST_CASE.
using RealScalarTypes = std::tuple<float, double>;

/// The two complex element types, as a type list for Catch2's TEMPLATE_LIST_TEST_CASE.
using ComplexScalarTypes = std::tuple<std::complex<float>, std::complex<double>>;

/**
 * @brief A scalar of element type @p T built from a real and an imaginary part.
 *
 * Complex types keep both parts and real types keep only @p re, so one templated test exercises a
 * genuinely complex prefactor wherever the element type allows it. A real-valued prefactor on a
 * complex tensor cannot tell a kernel that drops the imaginary part of the scalar, or conjugates it,
 * from a correct one. Choosing parts with no exact binary representation (0.1, not 0.5) also makes
 * two kernels that round differently land on different bits.
 *
 * @param re The real part.
 * @param im The imaginary part, ignored for a real @p T.
 */
template <typename T>
constexpr T prefactor(double re, double im) {
    if constexpr (IsComplexV<T>) {
        using R = RemoveComplexT<T>;
        return T{static_cast<R>(re), static_cast<R>(im)};
    } else {
        return static_cast<T>(re);
    }
}

/**
 * @brief The seed the einsums random engine starts each test case from.
 *
 * The test main reseeds @ref einsums::random_engine with this value at the start of every run
 * through a test case, including each pass Catch2 makes to reach another leaf section. The
 * result depends only on the run seed Catch2 prints as "Randomness seeded to: N" and on the
 * test case name, so a failure is reproduced by rerunning that one test with ``--rng-seed N``:
 * the draws do not depend on which tests ran before it or on which sections are selected.
 *
 * The mix is 32-bit FNV-1a over the seed's four little-endian bytes followed by the name. The
 * Python test suite uses the same function, so one seed means the same thing in both.
 *
 * @param run_seed The seed of the whole test run.
 * @param test_name The name of the test case.
 *
 * @return The seed for the named test case.
 */
constexpr std::uint32_t test_case_seed(std::uint32_t run_seed, std::string_view test_name) {
    std::uint32_t hash  = 2166136261U;
    auto          mixin = [&hash](unsigned char byte) {
        hash ^= byte;
        hash *= 16777619U;
    };
    for (int shift = 0; shift < 32; shift += 8) {
        mixin(static_cast<unsigned char>((run_seed >> shift) & 0xffU));
    }
    for (char const c : test_name) {
        mixin(static_cast<unsigned char>(c));
    }
    return hash;
}

} // namespace testing

EINSUMS_NAMESPACE_END()