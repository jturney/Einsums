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

TEST_CASE("parse_einsum_spec - conj(...) operand wrapper", "[ComputeGraph][EinsumSpec]") {
    SECTION("no conjugation") {
        auto r = parse_einsum_spec("ij <- ik ; kj");
        REQUIRE(r.has_value());
        REQUIRE_FALSE(r.value().conj_a);
        REQUIRE_FALSE(r.value().conj_b);
    }
    SECTION("conj A only — wrapper stripped, indices intact") {
        auto r = parse_einsum_spec("ij <- conj(ki) ; kj");
        REQUIRE(r.has_value());
        auto &s = r.value();
        REQUIRE(s.conj_a);
        REQUIRE_FALSE(s.conj_b);
        REQUIRE(s.a_indices == std::vector<std::string>{"k", "i"});
        REQUIRE(s.b_indices == std::vector<std::string>{"k", "j"});
    }
    SECTION("conj B only") {
        auto r = parse_einsum_spec("ij <- ik ; conj(jk)");
        REQUIRE(r.has_value());
        REQUIRE_FALSE(r.value().conj_a);
        REQUIRE(r.value().conj_b);
        REQUIRE(r.value().b_indices == std::vector<std::string>{"j", "k"});
    }
    SECTION("conj both operands") {
        auto r = parse_einsum_spec("ij <- conj(ki) ; conj(jk)");
        REQUIRE(r.has_value());
        REQUIRE(r.value().conj_a);
        REQUIRE(r.value().conj_b);
    }
    SECTION("conj with multi-char comma indices") {
        auto r = parse_einsum_spec("mu,nu <- conj(mu,rho) ; rho,nu");
        REQUIRE(r.has_value());
        auto &s = r.value();
        REQUIRE(s.conj_a);
        REQUIRE(s.a_indices == std::vector<std::string>{"mu", "rho"});
    }
    SECTION("conj in the -> arrow form") {
        auto r = parse_einsum_spec("conj(ki) ; kj -> ij");
        REQUIRE(r.has_value());
        REQUIRE(r.value().conj_a);
        REQUIRE(r.value().a_indices == std::vector<std::string>{"k", "i"});
    }
}

// Compile-time path (EinsumFormatString literal ctor): the consteval validator
// must accept the conj(...) parens, and the index counter must count the wrapped
// indices only (so a C++ literal like cg::einsum("ij <- conj(ki) ; kj", ...) both
// compiles and gets the right operand ranks).
static_assert(validate_einsum_spec("ij <- conj(ki) ; kj"));
static_assert(parse_index_counts("ij <- conj(ki) ; conj(kj)").a == 2);
static_assert(parse_index_counts("ij <- conj(ki) ; conj(kj)").b == 2);
static_assert(parse_index_counts("ij <- conj(ki) ; conj(kj)").c == 2);
static_assert(parse_index_counts("mu,nu <- conj(mu,rho) ; rho,nu").a == 2);

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

// Mixed delimiters: a comma-less operand alongside comma'd operands must be
// char-split per operand, not mis-read as one multi-char index (bug-1026).
TEST_CASE("parse_einsum_spec - mixed comma/no-comma operands", "[ComputeGraph][EinsumSpec]") {
    SECTION("comma-less output, comma'd inputs") {
        auto result = parse_einsum_spec("ijab <- Q,a,i,j,f ; Q,b,f");
        REQUIRE(result.has_value());
        auto &spec = result.value();
        REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j", "a", "b"});
        REQUIRE(spec.a_indices == std::vector<std::string>{"Q", "a", "i", "j", "f"});
        REQUIRE(spec.b_indices == std::vector<std::string>{"Q", "b", "f"});
    }
    SECTION("comma'd output, comma-less input") {
        auto result = parse_einsum_spec("i,j,a,b <- ijef ; abef");
        REQUIRE(result.has_value());
        auto &spec = result.value();
        REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j", "a", "b"});
        REQUIRE(spec.a_indices == std::vector<std::string>{"i", "j", "e", "f"});
        REQUIRE(spec.b_indices == std::vector<std::string>{"a", "b", "e", "f"});
    }
}

TEST_CASE("parse_permute_spec - mixed comma/no-comma operands", "[ComputeGraph][EinsumSpec]") {
    auto result = parse_permute_spec("jiba <- i,j,a,b");
    REQUIRE(result.has_value());
    auto &spec = result.value();
    REQUIRE(spec.c_indices == std::vector<std::string>{"j", "i", "b", "a"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "j", "a", "b"});
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
