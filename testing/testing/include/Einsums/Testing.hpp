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

template <typename T>
constexpr double tolerance() {
    return 1e-6;
}

template <>
constexpr double tolerance<float>() {
    return 1e-2;
}

template <>
constexpr double tolerance<std::complex<float>>() {
    return 1e-2;
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

EINSUMS_NAMESPACE_END()