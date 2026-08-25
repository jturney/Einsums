//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Json.cpp
/// @brief The strict document model the graph IR is written on.
///
/// The properties asserted here are the ones the IR's compatibility policy
/// rests on, and each is a failure this module has actually seen:
///
///  1. **Consumed-key audit.** A reader that does not understand a field must
///     be told, not left to ignore it. Glaze was rejected for this feature
///     because it silently dropped unknown fields, and `CostModel`'s hand
///     parser silently skips unrecognised rows to this day.
///  2. **Number fidelity.** A double must read back BIT-exactly, denormals and
///     boundary values included, and an integer must stay an integer. A saved
///     prefactor that comes back one ulp out is a wrong number nothing reports.
///  3. **Fixpoint.** `emit(parse(x)) == x` byte for byte for canonical text, so
///     the canonical form is a function of the document and can serve as a hash
///     domain.
///  4. **Hostile input.** Truncation, garbage and a depth bomb must fail with a
///     position and must never partially succeed or overrun the stack.

#include <Einsums/ComputeGraph/Detail/Json.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

namespace json = einsums::compute_graph::json;

namespace {

/// Parse @p text, requiring success, and return the value.
json::Value must_parse(std::string const &text) {
    auto parsed = json::parse(text);
    REQUIRE(parsed.has_value());
    return *parsed;
}

/// Parse @p text, requiring failure, and return the error.
json::ParseError must_fail(std::string const &text, json::ParseOptions options = {}) {
    auto parsed = json::parse(text, options);
    REQUIRE_FALSE(parsed.has_value());
    return parsed.error();
}

std::string canonical(json::Value const &value) {
    return json::emit(value, json::EmitOptions{.style = json::EmitStyle::Canonical});
}

} // namespace

TEST_CASE("json: value model round-trips every alternative", "[computegraph][json]") {
    SECTION("scalars keep their type") {
        REQUIRE(must_parse("null").is_null());
        REQUIRE(must_parse("true").as_bool());
        REQUIRE(must_parse("false").is_bool());
        REQUIRE(must_parse("17").is_int());
        REQUIRE(must_parse("17").as_int() == 17);
        REQUIRE(must_parse("-17").as_int() == -17);
        REQUIRE(must_parse("\"text\"").as_string() == "text");
    }

    SECTION("an integer is not a double and a double is not an integer") {
        // The distinction is the whole reason the model has both: an extent, an
        // id and an iteration count are integers, and a model that stores them
        // as doubles cannot say so.
        REQUIRE(must_parse("2").is_int());
        REQUIRE(must_parse("2.0").is_double());
        REQUIRE(must_parse("2e0").is_double());
        REQUIRE(canonical(json::Value{2.0}) == "2.0");
        REQUIRE(canonical(json::Value{std::int64_t{2}}) == "2");
    }

    SECTION("containers keep insertion order") {
        json::Object object;
        object.set("zebra", json::Value{1});
        object.set("apple", json::Value{2});
        object.set("mango", json::Value{3});
        json::Value const document{std::move(object)};
        // NOT sorted: a canonical order is the WRITER's job, so that a hash over
        // canonical bytes cannot depend on a hash table's iteration order.
        REQUIRE(canonical(document) == R"({"zebra":1,"apple":2,"mango":3})");
    }

    SECTION("strings escape what the grammar requires and nothing else") {
        json::Value const value{std::string("a\"b\\c\nd\te\x01 \xc3\xa9")};
        std::string const text = canonical(value);
        REQUIRE(text == "\"a\\\"b\\\\c\\nd\\te\\u0001 \xc3\xa9\"");
        REQUIRE(must_parse(text).as_string() == value.as_string());
    }

    SECTION("escapes read back") {
        REQUIRE(must_parse(R"("\u0041\u00e9\ud83d\ude00")").as_string() == "A\xc3\xa9\xf0\x9f\x98\x80");
        REQUIRE(must_parse(R"("\/\b\f")").as_string() == std::string("/\b\f"));
    }
}

TEST_CASE("json: doubles round-trip bit-exactly", "[computegraph][json]") {
    std::vector<double> const values{0.0,
                                     -0.0,
                                     1.0,
                                     -1.0,
                                     0.1,
                                     1.0 / 3.0,
                                     std::numbers::pi,
                                     2.2250738585072014e-308,                   // smallest normal
                                     std::numeric_limits<double>::denorm_min(), // smallest denormal
                                     std::numeric_limits<double>::min(),
                                     std::numeric_limits<double>::max(),
                                     -std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::epsilon(),
                                     1e308,
                                     1e-308,
                                     123456789.123456789};

    for (double const value : values) {
        std::string const text = canonical(json::Value{value});
        json::Value const back = must_parse(text);
        INFO("value " << value << " emitted as " << text);
        REQUIRE(back.is_double());
        // Bit equality, not approximate equality: a prefactor that comes back
        // one ulp out is a wrong number nothing reports.
        REQUIRE(std::bit_cast<std::uint64_t>(back.as_double()) == std::bit_cast<std::uint64_t>(value));
    }
}

TEST_CASE("json: int64 bounds", "[computegraph][json]") {
    REQUIRE(must_parse("9223372036854775807").as_int() == std::numeric_limits<std::int64_t>::max());
    REQUIRE(must_parse("-9223372036854775808").as_int() == std::numeric_limits<std::int64_t>::min());

    SECTION("an integer too large for int64 is refused, never silently widened") {
        auto const error = must_fail("9223372036854775808");
        REQUIRE_THAT(error.message, Catch::Matchers::ContainsSubstring("64-bit"));
    }
}

TEST_CASE("json: emit refuses non-finite doubles", "[computegraph][json]") {
    // JSON has no NaN and no infinity. Writing `null` in their place is the
    // silent-corruption class this model exists to refuse; a schema that must
    // carry a special tags it as a string inside its own typed object.
    REQUIRE_THROWS_AS(json::emit(json::Value{std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
    REQUIRE_THROWS_AS(json::emit(json::Value{std::numeric_limits<double>::infinity()}), std::invalid_argument);
    REQUIRE_THROWS_AS(json::emit(json::Value{-std::numeric_limits<double>::infinity()}), std::invalid_argument);

    SECTION("and the literals are not accepted on the way in either") {
        REQUIRE_FALSE(json::parse("NaN").has_value());
        REQUIRE_FALSE(json::parse("Infinity").has_value());
        REQUIRE_FALSE(json::parse("-Infinity").has_value());
    }
}

TEST_CASE("json: the consumed-key audit is what a strict loader stands on", "[computegraph][json]") {
    json::Value const document = must_parse(R"({"a":1,"b":{"c":2,"d":3},"e":[{"f":4}]})");
    auto const       &root     = document.as_object();

    SECTION("nothing is consumed until a reader takes it") {
        std::vector<std::string> unconsumed;
        json::collect_unconsumed(document, "$", unconsumed);
        REQUIRE(unconsumed == std::vector<std::string>{"$.a", "$.b", "$.e"});
    }

    SECTION("peek does not consume; take does") {
        REQUIRE(root.peek("a") != nullptr);
        REQUIRE(root.unconsumed_keys().size() == 3);
        REQUIRE(root.take("a") != nullptr);
        REQUIRE(root.unconsumed_keys() == std::vector<std::string>{"b", "e"});
    }

    SECTION("a partially understood object reports exactly the leftover key") {
        REQUIRE(root.take("a") != nullptr);
        REQUIRE(root.take("e") != nullptr);
        auto const *branch = root.take("b");
        REQUIRE(branch != nullptr);
        REQUIRE(branch->as_object().take("c") != nullptr);
        // "d" was never read, and neither was the object inside "e".
        std::vector<std::string> unconsumed;
        json::collect_unconsumed(document, "$", unconsumed);
        REQUIRE(unconsumed == std::vector<std::string>{"$.b.d", "$.e[0].f"});
    }

    SECTION("an absent key consumes nothing") {
        REQUIRE(root.take("missing") == nullptr);
        REQUIRE(root.unconsumed_keys().size() == 3);
    }

    SECTION("consuming a container key does not excuse the keys inside it") {
        // The audit descends: taking "b" says the reader knew there was a "b",
        // not that it understood everything in it. That is the property that
        // makes the policy worth having, since an unknown field is far more
        // likely to be nested than top-level.
        root.mark_consumed("a");
        root.mark_consumed("b");
        root.mark_consumed("e");
        std::vector<std::string> unconsumed;
        json::collect_unconsumed(document, "$", unconsumed);
        REQUIRE(unconsumed == std::vector<std::string>{"$.b.c", "$.b.d", "$.e[0].f"});

        root.peek("b")->as_object().mark_consumed("c");
        root.peek("b")->as_object().mark_consumed("d");
        root.peek("e")->as_array()[0].as_object().mark_consumed("f");
        unconsumed.clear();
        json::collect_unconsumed(document, "$", unconsumed);
        REQUIRE(unconsumed.empty());
    }
}

TEST_CASE("json: canonical emission is a fixpoint", "[computegraph][json]") {
    std::vector<std::string> const canonical_documents{
        R"({})",
        R"([])",
        R"(null)",
        R"({"a":1,"b":2.5,"c":"x","d":[1,2,3],"e":{"f":true,"g":null}})",
        R"([[[1]],[2],[],{}])",
        R"({"nested":{"deep":{"deeper":[1.5,-2,"three"]}}})",
    };

    for (auto const &text : canonical_documents) {
        INFO("document " << text);
        REQUIRE(canonical(must_parse(text)) == text);
        // And a second pass changes nothing, which is what "fixpoint" means.
        REQUIRE(canonical(must_parse(canonical(must_parse(text)))) == text);
    }
}

TEST_CASE("json: pretty emission is diff-tuned but says the same thing", "[computegraph][json]") {
    std::string const source = R"({"version":"1.0.0","nodes":[{"id":0,"kind":"Einsum"},{"id":1,"kind":"Scale"}]})";
    json::Value const value  = must_parse(source);

    std::string const pretty = json::emit(value, json::EmitOptions{.style = json::EmitStyle::Pretty, .inline_width = 40});
    // One node per line: each small node object fits inline (and is then written
    // canonically, so the inline form is the same bytes a hash would see), while
    // the array around them expands.
    REQUIRE_THAT(pretty, Catch::Matchers::ContainsSubstring(R"({"id":0,"kind":"Einsum"})"));
    REQUIRE_THAT(pretty, Catch::Matchers::ContainsSubstring(R"({"id":1,"kind":"Scale"})"));
    REQUIRE_THAT(pretty, Catch::Matchers::ContainsSubstring("\"nodes\": [\n"));

    // Whatever the layout, the content is identical: pretty text parses back to
    // the same canonical bytes, which is what lets a hash live on the canonical
    // form while a human reads the pretty one.
    REQUIRE(canonical(must_parse(pretty)) == source);
}

TEST_CASE("json: hostile input fails cleanly and says where", "[computegraph][json]") {
    SECTION("truncation") {
        for (std::string const full = R"({"a":[1,2,{"b":"c"}]})"; auto const cut : {1U, 4U, 6U, 9U, 14U, 18U, 20U}) {
            std::string const text = full.substr(0, cut);
            INFO("truncated to '" << text << "'");
            auto const error = must_fail(text);
            REQUIRE_FALSE(error.message.empty());
        }
    }

    SECTION("garbage") {
        for (char const *text : {"{",       "}",           "[",        "]",           ",",           ":",
                                 "{,}",     "[,]",         R"({"a"})", R"({"a":})",   "{a:1}",       "{'a':1}",
                                 "[1,]",    R"({"a":1,})", "01",       "+1",          ".5",          "1.",
                                 "1e",      "1e+",         "--1",      "tru",         "nul",         R"("unterminated)",
                                 R"("\q")", "{} extra",    "1 2",      R"("\ud800")", R"("\udc00")", R"("\ud800\u0041")"}) {
            INFO("input '" << text << "'");
            auto const error = must_fail(text);
            REQUIRE_FALSE(error.message.empty());
        }
    }

    SECTION("a duplicate key is an error, not last-one-wins") {
        auto const error = must_fail(R"({"a":1,"a":2})");
        REQUIRE_THAT(error.message, Catch::Matchers::ContainsSubstring("duplicate key 'a'"));
    }

    SECTION("positions point at the offending byte") {
        auto const error = must_fail("{\n  \"a\": 1,\n  \"a\": 2\n}");
        REQUIRE(error.where.line == 3);
        REQUIRE(error.where.column == 3);
        REQUIRE_THAT(error.to_string(), Catch::Matchers::ContainsSubstring("line 3 column 3"));
    }

    SECTION("a depth bomb is a diagnostic, not a stack overflow") {
        std::string bomb;
        for (int i = 0; i < 5000; ++i) {
            bomb += '[';
        }
        for (int i = 0; i < 5000; ++i) {
            bomb += ']';
        }
        auto const error = must_fail(bomb);
        REQUIRE_THAT(error.message, Catch::Matchers::ContainsSubstring("nesting deeper"));
    }

    SECTION("the depth limit admits exactly what it says") {
        json::ParseOptions const options{.max_depth = 3};
        REQUIRE(json::parse("[[[1]]]", options).has_value());
        REQUIRE_FALSE(json::parse("[[[[1]]]]", options).has_value());
    }

    SECTION("a bit flip in a valid document never partially loads") {
        std::string const source = R"({"a":1,"b":[2,3],"c":{"d":"e"}})";
        for (std::size_t i = 0; i < source.size(); ++i) {
            std::string mutated = source;
            mutated[i]          = static_cast<char>(mutated[i] ^ 0x20);
            if (mutated == source) {
                continue;
            }
            auto const parsed = json::parse(mutated);
            if (!parsed.has_value()) {
                REQUIRE_FALSE(parsed.error().message.empty());
                continue;
            }
            // A flip that still parses must have produced a WHOLE document, not
            // a half-read one: re-emitting it and reading it back agrees.
            REQUIRE(canonical(must_parse(canonical(*parsed))) == canonical(*parsed));
        }
    }
}

TEST_CASE("json: Object::set refuses a duplicate key", "[computegraph][json]") {
    json::Object object;
    object.set("a", json::Value{1});
    REQUIRE_THROWS_AS(object.set("a", json::Value{2}), std::invalid_argument);
}
