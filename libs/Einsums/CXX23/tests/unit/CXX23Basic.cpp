//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file CXX23Basic.cpp
/// @brief Tests for C++23 backport library.

#include <Einsums/CXX23.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

// ═══════════════════════════════════════════════════════════════════════════════
// expected<T, E>
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("expected - construct with value", "[CXX23][expected]") {
    einsums::expected<int, std::string> e(42);
    CHECK(e.has_value());
    CHECK(static_cast<bool>(e));
    CHECK(e.value() == 42);
    CHECK(*e == 42);
}

TEST_CASE("expected - construct with error", "[CXX23][expected]") {
    einsums::expected<int, std::string> e(einsums::unexpected(std::string("oops")));
    CHECK_FALSE(e.has_value());
    CHECK_FALSE(static_cast<bool>(e));
    CHECK(e.error() == "oops");
}

TEST_CASE("expected - default construct", "[CXX23][expected]") {
    einsums::expected<int, std::string> e;
    CHECK(e.has_value());
    CHECK(e.value() == 0);
}

TEST_CASE("expected - value_or", "[CXX23][expected]") {
    einsums::expected<int, std::string> val(42);
    einsums::expected<int, std::string> err(einsums::unexpected(std::string("fail")));

    CHECK(val.value_or(99) == 42);
    CHECK(err.value_or(99) == 99);
}

TEST_CASE("expected - arrow operator", "[CXX23][expected]") {
    einsums::expected<std::string, int> e(std::string("hello"));
    CHECK(e->size() == 5);
}

TEST_CASE("expected - transform", "[CXX23][expected]") {
    einsums::expected<int, std::string> e(10);
    auto                                doubled = e.transform([](int x) { return x * 2; });
    CHECK(doubled.has_value());
    CHECK(doubled.value() == 20);
}

TEST_CASE("expected - transform propagates error", "[CXX23][expected]") {
    einsums::expected<int, std::string> e(einsums::unexpected(std::string("bad")));
    auto                                doubled = e.transform([](int x) { return x * 2; });
    CHECK_FALSE(doubled.has_value());
    CHECK(doubled.error() == "bad");
}

TEST_CASE("expected - and_then", "[CXX23][expected]") {
    auto safe_div = [](int x) -> einsums::expected<double, std::string> {
        if (x == 0)
            return einsums::unexpected(std::string("div by zero"));
        return 100.0 / x;
    };

    einsums::expected<int, std::string> e(5);
    auto                                result = e.and_then(safe_div);
    CHECK(result.has_value());
    CHECK(result.value() == Catch::Approx(20.0));
}

TEST_CASE("expected<void, E> - success", "[CXX23][expected]") {
    einsums::expected<void, std::string> e;
    CHECK(e.has_value());
}

TEST_CASE("expected<void, E> - error", "[CXX23][expected]") {
    einsums::expected<void, std::string> e(einsums::unexpected(std::string("fail")));
    CHECK_FALSE(e.has_value());
    CHECK(e.error() == "fail");
}

TEST_CASE("expected - move semantics", "[CXX23][expected]") {
    einsums::expected<std::vector<int>, std::string> e(std::vector<int>{1, 2, 3});
    auto                                             moved = std::move(e);
    CHECK(moved.has_value());
    CHECK(moved.value().size() == 3);
}

TEST_CASE("unexpected - deduction guide", "[CXX23][expected]") {
    auto u = einsums::unexpected(42);
    CHECK(u.error() == 42);

    auto u2 = einsums::unexpected(std::string("err"));
    CHECK(u2.error() == "err");
}
