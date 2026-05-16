//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#if defined(EINSUMS_WINDOWS)
#    define CATCH_CONFIG_WINDOWS_SEH
#endif
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
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
        LabeledSection(fmt::runtime(Catch::getResultCapture().getCurrentTestName()));                                                       \
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

namespace einsums {

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
    bool match(TestType value) const {
        if (_target == RemoveComplexT<std::remove_cvref_t<TestType>>{0.0}) {
            return std::abs(value) <= _eps;
        } else {
            return std::abs((value - _target) / _target) <= _eps;
        }
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

} // namespace einsums