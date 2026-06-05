//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file CXX23Basic.cpp
/// @brief Tests for C++23 backport library.

#include <Einsums/CXX23.hpp>

#include <complex>
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

// ═══════════════════════════════════════════════════════════════════════════════
// flat_set
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("flat_set - default construction", "[CXX23][flat_set]") {
    einsums::flat_set<int> s;
    CHECK(s.empty());
    CHECK(s.size() == 0);
}

TEST_CASE("flat_set - initializer list", "[CXX23][flat_set]") {
    einsums::flat_set<int> s{3, 1, 4, 1, 5, 9, 2, 6};
    CHECK(s.size() == 7); // Duplicates removed
    // Should be sorted
    auto it = s.begin();
    CHECK(*it++ == 1);
    CHECK(*it++ == 2);
    CHECK(*it++ == 3);
    CHECK(*it++ == 4);
    CHECK(*it++ == 5);
    CHECK(*it++ == 6);
    CHECK(*it++ == 9);
}

TEST_CASE("flat_set - insert", "[CXX23][flat_set]") {
    einsums::flat_set<int> s;
    auto [it1, ok1] = s.insert(5);
    CHECK(ok1);
    CHECK(*it1 == 5);

    auto [it2, ok2] = s.insert(5); // Duplicate
    CHECK_FALSE(ok2);
    CHECK(*it2 == 5);
    CHECK(s.size() == 1);
}

TEST_CASE("flat_set - contains and find", "[CXX23][flat_set]") {
    einsums::flat_set<int> s{10, 20, 30};
    CHECK(s.contains(20));
    CHECK_FALSE(s.contains(25));
    CHECK(s.find(30) != s.end());
    CHECK(s.find(99) == s.end());
    CHECK(s.count(10) == 1);
    CHECK(s.count(99) == 0);
}

TEST_CASE("flat_set - erase", "[CXX23][flat_set]") {
    einsums::flat_set<int> s{1, 2, 3, 4, 5};
    CHECK(s.erase(3) == 1);
    CHECK(s.size() == 4);
    CHECK_FALSE(s.contains(3));
    CHECK(s.erase(99) == 0); // Not present
}

TEST_CASE("flat_set - string keys", "[CXX23][flat_set]") {
    einsums::flat_set<std::string> s{"beta", "alpha", "gamma"};
    CHECK(s.size() == 3);
    CHECK(*s.begin() == "alpha");
    CHECK(s.contains("beta"));
}

TEST_CASE("flat_set - equality", "[CXX23][flat_set]") {
    einsums::flat_set<int> a{1, 2, 3};
    einsums::flat_set<int> b{3, 2, 1};
    CHECK(a == b); // Same elements, order shouldn't matter for equality
}

TEST_CASE("flat_set - range-for", "[CXX23][flat_set]") {
    einsums::flat_set<int> s{5, 3, 1};
    std::vector<int>       collected;
    for (int x : s)
        collected.push_back(x);
    CHECK(collected == std::vector<int>{1, 3, 5});
}

// ═══════════════════════════════════════════════════════════════════════════════
// flat_map
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("flat_map - default construction", "[CXX23][flat_map]") {
    einsums::flat_map<int, std::string> m;
    CHECK(m.empty());
    CHECK(m.size() == 0);
}

TEST_CASE("flat_map - initializer list", "[CXX23][flat_map]") {
    einsums::flat_map<int, std::string> m{{3, "three"}, {1, "one"}, {2, "two"}};
    CHECK(m.size() == 3);
    CHECK(m.begin()->first == 1); // Sorted by key
}

TEST_CASE("flat_map - operator[]", "[CXX23][flat_map]") {
    einsums::flat_map<std::string, int> m;
    m["foo"] = 42;
    m["bar"] = 99;
    CHECK(m.size() == 2);
    CHECK(m["foo"] == 42);
    CHECK(m["bar"] == 99);
    m["foo"] = 100; // Overwrite
    CHECK(m["foo"] == 100);
}

TEST_CASE("flat_map - insert and find", "[CXX23][flat_map]") {
    einsums::flat_map<int, double> m;
    auto [it1, ok1] = m.insert({10, 3.14});
    CHECK(ok1);
    CHECK(it1->second == 3.14);

    auto [it2, ok2] = m.insert({10, 2.71}); // Duplicate key
    CHECK_FALSE(ok2);
    CHECK(it2->second == 3.14); // Original value preserved
}

TEST_CASE("flat_map - insert_or_assign", "[CXX23][flat_map]") {
    einsums::flat_map<int, std::string> m;
    auto [it1, ins1] = m.insert_or_assign(1, std::string("first"));
    CHECK(ins1);
    CHECK(it1->second == "first");

    auto [it2, ins2] = m.insert_or_assign(1, std::string("updated"));
    CHECK_FALSE(ins2);
    CHECK(it2->second == "updated");
}

TEST_CASE("flat_map - contains and count", "[CXX23][flat_map]") {
    einsums::flat_map<int, int> m{{1, 10}, {2, 20}, {3, 30}};
    CHECK(m.contains(2));
    CHECK_FALSE(m.contains(4));
    CHECK(m.count(1) == 1);
    CHECK(m.count(99) == 0);
}

TEST_CASE("flat_map - erase", "[CXX23][flat_map]") {
    einsums::flat_map<int, int> m{{1, 10}, {2, 20}, {3, 30}};
    CHECK(m.erase(2) == 1);
    CHECK(m.size() == 2);
    CHECK_FALSE(m.contains(2));
}

TEST_CASE("flat_map - at", "[CXX23][flat_map]") {
    einsums::flat_map<int, std::string> m{{1, "one"}, {2, "two"}};
    CHECK(m.at(1) == "one");
    CHECK(m.at(2) == "two");
}

TEST_CASE("flat_map - iteration order is sorted by key", "[CXX23][flat_map]") {
    einsums::flat_map<int, int> m{{30, 3}, {10, 1}, {20, 2}};
    std::vector<int>            keys;
    for (auto const &[k, v] : m)
        keys.push_back(k);
    CHECK(keys == std::vector<int>{10, 20, 30});
}

TEST_CASE("flat_map - equality", "[CXX23][flat_map]") {
    einsums::flat_map<int, int> a{{1, 10}, {2, 20}};
    einsums::flat_map<int, int> b{{2, 20}, {1, 10}};
    CHECK(a == b);
}

TEST_CASE("flat_map - reserve", "[CXX23][flat_map]") {
    einsums::flat_map<int, int> m;
    REQUIRE_NOTHROW(m.reserve(100));
    CHECK(m.empty()); // Reserve doesn't add elements
}
