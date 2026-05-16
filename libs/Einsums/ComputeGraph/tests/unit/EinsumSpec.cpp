//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums::compute_graph;

// ─── Arrow notation tests ───────────────────────────────────────────────────

TEST_CASE("parse_einsum_spec - arrow notation, single-char", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij <- ik ; kj");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "k"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"k", "j"});
}

TEST_CASE("parse_einsum_spec - arrow notation, no whitespace", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij<-ik;kj");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "k"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"k", "j"});
}

TEST_CASE("parse_einsum_spec - arrow notation, multi-char", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("mu,nu <- mu,rho ; rho,nu");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"mu", "nu"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"mu", "rho"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"rho", "nu"});
}

TEST_CASE("parse_einsum_spec - arrow notation, numbered indices", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("i1,i2 <- i1,i3 ; i3,i2");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"i1", "i2"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i1", "i3"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"i3", "i2"});
}

// ─── NumPy notation tests ───────────────────────────────────────────────────

TEST_CASE("parse_einsum_spec - numpy notation, single-char", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ik;kj -> ij");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "k"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"k", "j"});
}

TEST_CASE("parse_einsum_spec - numpy notation, multi-char", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("mu,rho;rho,nu -> mu,nu");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"mu", "nu"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"mu", "rho"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"rho", "nu"});
}

// ─── Special cases ──────────────────────────────────────────────────────────

TEST_CASE("parse_einsum_spec - dot product (empty output)", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec(" <- i ; i");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices.empty());
    REQUIRE(spec.a_indices == std::vector<std::string>{"i"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"i"});
}

TEST_CASE("parse_einsum_spec - rank-3 contraction", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ijk <- ijl ; lk");
    REQUIRE(result.has_value());
    auto &spec = result.value();

    REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j", "k"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "j", "l"});
    REQUIRE(spec.b_indices == std::vector<std::string>{"l", "k"});
}

// ─── Link and target index computation ──────────────────────────────────────

TEST_CASE("ParsedEinsumSpec - link_indices", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij <- ik ; kj");
    REQUIRE(result.has_value());

    auto links = result.value().link_indices();
    REQUIRE(links.size() == 1);
    REQUIRE(links[0] == "k");
}

TEST_CASE("ParsedEinsumSpec - target_indices", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij <- ik ; kj");
    REQUIRE(result.has_value());

    auto targets = result.value().target_indices();
    REQUIRE(targets.size() == 2);
    REQUIRE(targets[0] == "i");
    REQUIRE(targets[1] == "j");
}

TEST_CASE("ParsedEinsumSpec - link_indices multi-char", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("mu,nu <- mu,rho ; rho,nu");
    REQUIRE(result.has_value());

    auto links = result.value().link_indices();
    REQUIRE(links.size() == 1);
    REQUIRE(links[0] == "rho");
}

// ─── Error handling (now uses expected instead of exceptions) ──────────────

TEST_CASE("parse_einsum_spec - error: no arrow", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij ik ; kj");
    CHECK_FALSE(result.has_value());
    CHECK(result.error().kind == GraphError::Kind::Parse);
    CHECK(result.error().message.find("arrow") != std::string::npos);
}

TEST_CASE("parse_einsum_spec - error: both arrows", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij <- ik;kj -> ij");
    CHECK_FALSE(result.has_value());
    CHECK(result.error().kind == GraphError::Kind::Parse);
}

TEST_CASE("parse_einsum_spec - error: no semicolon", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_einsum_spec("ij <- ikkj");
    CHECK_FALSE(result.has_value());
    CHECK(result.error().kind == GraphError::Kind::Parse);
}

// ─── Constexpr validation ───────────────────────────────────────────────────

TEST_CASE("validate_einsum_spec - valid specs", "[ComputeGraph][EinsumSpec]") {
    STATIC_REQUIRE(validate_einsum_spec("ij <- ik ; kj"));
    STATIC_REQUIRE(validate_einsum_spec("ik;kj -> ij"));
    STATIC_REQUIRE(validate_einsum_spec("ij<-ik;kj"));
    STATIC_REQUIRE(validate_einsum_spec(" <- i ; i"));
}

TEST_CASE("validate_einsum_spec - invalid specs", "[ComputeGraph][EinsumSpec]") {
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ij ik ; kj"));
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ij <- ik kj"));
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ij <- ik;kj->ij"));
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ij <- ik ; kj!"));
}
