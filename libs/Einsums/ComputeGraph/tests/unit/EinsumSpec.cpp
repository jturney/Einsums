//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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
// must accept the conj(...) parens.
static_assert(validate_einsum_spec("ij <- conj(ki) ; kj"));

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
// char-split per operand, not mis-read as one multi-char index.
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

TEST_CASE("parse_einsum_spec - error: non-alphanumeric index character", "[ComputeGraph][EinsumSpec]") {
    // A comma-less operand is char-split, so a stray '@' / '$' / '.' used to
    // become a silent index label and run a malformed contraction. The runtime
    // parser now rejects it, matching the constexpr validate_einsum_spec (which
    // already rejects e.g. "... kj!"). Numbered indices (digits) stay valid.
    for (auto const *bad : {"i@ <- ij ; jk", "ij <- i$ ; $j", "i.. <- ij ; jk"}) {
        auto result = parse_einsum_spec(bad);
        CHECK_FALSE(result.has_value());
        if (!result.has_value()) {
            CHECK(result.error().kind == GraphError::Kind::Parse);
        }
    }
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

// ─── Permutation operators ──────────────────────────────────────────────────

namespace {

/// The expansion as `{"j,i,k", -1.0}` pairs, which is how the literature prints
/// it and how these tests are read against a printed equation.
std::vector<std::pair<std::string, double>> expanded(std::vector<std::string> const         &c_indices,
                                                     std::vector<PermutationOperator> const &operators) {
    std::vector<std::pair<std::string, double>> out;
    for (auto const &term : expand_permutation_operators(c_indices, operators)) {
        std::string joined;
        for (std::size_t i = 0; i < term.c_indices.size(); ++i) {
            if (i != 0) {
                joined += ',';
            }
            joined += term.c_indices[i];
        }
        out.emplace_back(std::move(joined), term.sign);
    }
    return out;
}

/// Same, sorted, for the cases where only the SET of terms is contractual. The
/// enumeration order is deterministic but it is an implementation choice
/// everywhere except the identity term, which is always first.
std::vector<std::pair<std::string, double>> expanded_sorted(std::vector<std::string> const         &c_indices,
                                                            std::vector<PermutationOperator> const &operators) {
    auto out = expanded(c_indices, operators);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<PermutationOperator> ops_of(std::string_view spec) {
    auto parsed = parse_einsum_spec(spec);
    REQUIRE(parsed.has_value());
    return parsed.value().operators;
}

} // namespace

// This is the test that pins the CONVENTION, and it is the reason the others can
// be read at a glance. A coset of the Young subgroup holds members of different
// parity, so which representative is taken changes the value on anything that is
// not already antisymmetric within each group. The sets below are the ones the
// coupled-cluster literature prints; the natural "place each group in increasing
// order" enumeration instead yields {e, (ij), (ikj)} for P(i/jk), whose third
// term carries the OPPOSITE sign.
TEST_CASE("expand_permutation_operators - literature expansions", "[ComputeGraph][EinsumSpec]") {
    using Terms = std::vector<std::pair<std::string, double>>;

    SECTION("P(i/j): the plain pair antisymmetrizer, identity first") {
        REQUIRE(expanded({"i", "j", "a", "b"}, ops_of("i,j,a,b <- P(i/j) i,k,a,c ; k,c,j,b")) ==
                Terms{{"i,j,a,b", 1.0}, {"j,i,a,b", -1.0}});
    }

    SECTION("P(i/jk) = 1 - (ij) - (ik), as f(ijk) - f(jik) - f(kji)") {
        REQUIRE(expanded({"i", "j", "k"}, ops_of("i,j,k <- P(i/jk) i,m ; m,j,k")) ==
                Terms{{"i,j,k", 1.0}, {"j,i,k", -1.0}, {"k,j,i", -1.0}});
    }

    SECTION("P(ij/k) = 1 - (ik) - (jk)") {
        REQUIRE(expanded_sorted({"i", "j", "k"}, ops_of("i,j,k <- P(ij/k) i,m ; m,j,k")) ==
                Terms{{"i,j,k", 1.0}, {"i,k,j", -1.0}, {"k,j,i", -1.0}});
    }

    SECTION("P(i/j/k): the full antisymmetrizer, six terms with parity signs") {
        REQUIRE(expanded_sorted({"i", "j", "k"}, ops_of("i,j,k <- P(i/j/k) i,m ; m,j,k")) ==
                Terms{{"i,j,k", 1.0}, {"i,k,j", -1.0}, {"j,i,k", -1.0}, {"j,k,i", 1.0}, {"k,i,j", 1.0}, {"k,j,i", -1.0}});
    }

    SECTION("P(ij/kl) = 1 - (ik) - (il) - (jk) - (jl) + (ik)(jl), six terms not seven") {
        auto const terms = expanded_sorted({"i", "j", "k", "l"}, ops_of("i,j,k,l <- P(ij/kl) i,m ; m,j,k,l"));
        REQUIRE(terms.size() == 6);
        REQUIRE(terms ==
                Terms{{"i,j,k,l", 1.0}, {"i,k,j,l", -1.0}, {"i,l,k,j", -1.0}, {"k,j,i,l", -1.0}, {"k,l,i,j", 1.0}, {"l,j,k,i", -1.0}});
    }

    SECTION("no operators expands to exactly the identity term") {
        REQUIRE(expanded({"i", "j"}, {}) == Terms{{"i,j", 1.0}});
    }
}

// The nine (spec, sign) pairs of examples/toy/ccsd_t_spinorbital_toy.py:134,
// which that file maintains by hand and checks against psi4's CCSD(T). If this
// ever disagrees, the operator is computing a different quantity than the
// working (T) code does.
TEST_CASE("expand_permutation_operators - the toy (T) nine-term table", "[ComputeGraph][EinsumSpec]") {
    std::vector<std::pair<std::string, double>> want{{"i,j,k,a,b,c", 1.0},  {"j,i,k,a,b,c", -1.0}, {"k,j,i,a,b,c", -1.0},
                                                     {"i,j,k,b,a,c", -1.0}, {"i,j,k,c,b,a", -1.0}, {"j,i,k,b,a,c", 1.0},
                                                     {"j,i,k,c,b,a", 1.0},  {"k,j,i,b,a,c", 1.0},  {"k,j,i,c,b,a", 1.0}};
    std::sort(want.begin(), want.end());

    auto const got = expanded_sorted({"i", "j", "k", "a", "b", "c"}, ops_of("i,j,k,a,b,c <- P(i/jk) P(a/bc) i,j,k,m ; m,a,b,c"));
    REQUIRE(got.size() == 9);
    REQUIRE(got == want);
}

// The slide's term. Note the output ordering is i,a,j,b rather than i,j,a,b, so
// this also checks that the operator permutes LETTERS and does not assume the
// output groups its occupied and virtual axes together.
TEST_CASE("expand_permutation_operators - P(ij)P(ab) on an interleaved output", "[ComputeGraph][EinsumSpec]") {
    using Terms      = std::vector<std::pair<std::string, double>>;
    auto const terms = expanded_sorted({"i", "a", "j", "b"}, ops_of("i,a,j,b <- P(ij) P(ab) i,k,a,c ; k,c,j,b"));
    REQUIRE(terms == Terms{{"i,a,j,b", 1.0}, {"i,b,j,a", -1.0}, {"j,a,i,b", -1.0}, {"j,b,i,a", 1.0}});
}

TEST_CASE("parse_einsum_spec - permutation operator parsing", "[ComputeGraph][EinsumSpec]") {
    using Groups = std::vector<std::vector<std::string>>;

    SECTION("P(ij) is sugar for P(i/j) in character mode") {
        auto const ops = ops_of("i,j,a,b <- P(ij) i,k,a,c ; k,c,j,b");
        REQUIRE(ops.size() == 1);
        REQUIRE(ops[0].groups == Groups{{"i"}, {"j"}});
    }

    SECTION("two operators, in source order") {
        auto const ops = ops_of("i,j,a,b <- P(i/j) P(a/b) i,k,a,c ; k,c,j,b");
        REQUIRE(ops.size() == 2);
        REQUIRE(ops[0].groups == Groups{{"i"}, {"j"}});
        REQUIRE(ops[1].groups == Groups{{"a"}, {"b"}});
    }

    SECTION("numpy notation puts the operator at the head of the term") {
        auto const ops = ops_of("P(i/j) i,k,a,c ; k,c,j,b -> i,j,a,b");
        REQUIRE(ops.size() == 1);
        REQUIRE(ops[0].groups == Groups{{"i"}, {"j"}});
    }

    SECTION("whitespace around and inside the operator is ignored") {
        REQUIRE(ops_of("i,j,k <- P( i / j k )  i,m ; m,j,k")[0].groups == Groups{{"i"}, {"j", "k"}});
        REQUIRE(ops_of("i,j,k <-P(i/jk)i,m ; m,j,k")[0].groups == Groups{{"i"}, {"j", "k"}});
    }

    SECTION("comma mode is decided across the whole operator, not per group") {
        // 'rho' must stay one index; a per-group rule would char-split it into three
        // because that group has no comma of its own.
        auto const ops = ops_of("mu,nu,rho <- P(mu,nu/rho) mu,x ; x,nu,rho");
        REQUIRE(ops.size() == 1);
        REQUIRE(ops[0].groups == Groups{{"mu", "nu"}, {"rho"}});
    }

    SECTION("the operand list still parses normally underneath") {
        auto parsed = parse_einsum_spec("i,a,j,b <- P(i/j) P(a/b) i,k,a,c ; k,c,j,b");
        REQUIRE(parsed.has_value());
        auto const &spec = parsed.value();
        REQUIRE(spec.c_indices == std::vector<std::string>{"i", "a", "j", "b"});
        REQUIRE(spec.a_indices == std::vector<std::string>{"i", "k", "a", "c"});
        REQUIRE(spec.b_indices == std::vector<std::string>{"k", "c", "j", "b"});
        REQUIRE(spec.link_indices() == std::vector<std::string>{"c", "k"});
    }

    SECTION("an operator composes with a conj(...) operand") {
        auto parsed = parse_einsum_spec("i,j,a,b <- P(i/j) conj(i,k,a,c) ; k,c,j,b");
        REQUIRE(parsed.has_value());
        REQUIRE(parsed.value().conj_a);
        REQUIRE_FALSE(parsed.value().conj_b);
        REQUIRE(parsed.value().operators.size() == 1);
        REQUIRE(parsed.value().a_indices == std::vector<std::string>{"i", "k", "a", "c"});
    }

    SECTION("a spec with no operator parses to an empty list") {
        REQUIRE(ops_of("ij <- ik ; kj").empty());
    }
}

TEST_CASE("parse_permute_spec - permutation operator parsing", "[ComputeGraph][EinsumSpec]") {
    using Groups = std::vector<std::vector<std::string>>;

    auto parsed = parse_permute_spec("i,j,k,a,b,c <- P(i/jk) P(a/bc) i,j,k,a,b,c");
    REQUIRE(parsed.has_value());
    auto const &spec = parsed.value();
    REQUIRE(spec.operators.size() == 2);
    REQUIRE(spec.operators[0].groups == Groups{{"i"}, {"j", "k"}});
    REQUIRE(spec.operators[1].groups == Groups{{"a"}, {"b", "c"}});
    REQUIRE(spec.c_indices == std::vector<std::string>{"i", "j", "k", "a", "b", "c"});
    REQUIRE(spec.a_indices == std::vector<std::string>{"i", "j", "k", "a", "b", "c"});
    REQUIRE(expand_permutation_operators(spec.c_indices, spec.operators).size() == 9);
}

TEST_CASE("parse_einsum_spec - permutation operator errors", "[ComputeGraph][EinsumSpec]") {
    auto const fails = [](std::string_view spec, std::string_view needle) {
        auto const r = parse_einsum_spec(spec);
        REQUIRE_FALSE(r.has_value());
        INFO("spec: " << spec << "\nmessage: " << r.error().message);
        REQUIRE(r.error().message.find(needle) != std::string::npos);
    };

    SECTION("a letter that is not an output index") {
        fails("i,j,a,b <- P(i/k) i,k,a,c ; k,c,j,b", "not an output index");
    }
    SECTION("a letter the output repeats") {
        fails("i,i,a,b <- P(i/a) i,k ; k,i,a,b", "the output repeats");
    }
    SECTION("overlapping groups within one operator") {
        fails("i,j,a,b <- P(i/i) i,k,a,c ; k,c,j,b", "twice");
    }
    SECTION("two operators naming the same letter") {
        fails("i,j,a,b <- P(i/j) P(i/a) i,k,a,c ; k,c,j,b", "already permutes");
    }
    SECTION("a single group in character mode is not two groups") {
        fails("i,j,a,b <- P(i) i,k,a,c ; k,c,j,b", "at least two groups");
    }
    SECTION("a single multi-char group is ambiguous rather than guessed") {
        fails("mu,nu <- P(mu,nu) mu,x ; x,nu", "ambiguous");
    }
    SECTION("an empty group") {
        fails("i,j,a,b <- P(i//j) i,k,a,c ; k,c,j,b", "empty group");
    }
    SECTION("an unterminated operator") {
        fails("i,j,a,b <- P(i/j i,k,a,c ; k,c,j,b", "unterminated");
    }
    SECTION("an operator inside an operand, rather than in front of the term") {
        fails("i,j,a,b <- i,k,a,c ; P(i/j) k,c,j,b", "prefixes the whole term");
    }
}

TEST_CASE("parse_permute_spec - permutation operator errors", "[ComputeGraph][EinsumSpec]") {
    auto const r = parse_permute_spec("i,j,k <- P(i/m) i,j,k");
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().message.find("not an output index") != std::string::npos);
}

TEST_CASE("ParsedEinsumSpec::render - permutation operators round-trip", "[ComputeGraph][EinsumSpec]") {
    SECTION("a spec without operators renders exactly as before") {
        REQUIRE(parse_einsum_spec("ij <- ik ; kj").value().render() == "i,j <- i,k ; k,j");
    }

    SECTION("operators render in canonical slash/comma form") {
        auto const spec = parse_einsum_spec("i,a,j,b <- P(ij) P(ab) i,k,a,c ; k,c,j,b").value();
        REQUIRE(spec.render() == "i,a,j,b <- P(i/j) P(a/b) i,k,a,c ; k,c,j,b");
    }

    SECTION("the rendered form parses back to the same operators") {
        auto const once  = parse_einsum_spec("i,j,k <- P(i/jk) i,m ; m,j,k").value();
        auto const twice = parse_einsum_spec(once.render());
        REQUIRE(twice.has_value());
        REQUIRE(twice.value().operators[0].groups == once.operators[0].groups);
        REQUIRE(twice.value().render() == once.render());
    }

    SECTION("permute specs render their operators too") {
        auto const spec = parse_permute_spec("i,j,k <- P(i/jk) i,j,k").value();
        REQUIRE(spec.render() == "i,j,k <- P(i/j,k) i,j,k");
        REQUIRE(parse_permute_spec(spec.render()).has_value());
    }
}

// An output index that no operand carries has no meaning the engine agrees on:
// the generic loop broadcast the result along it while the rank-1 DOT route wrote
// one element, so "j <- i ; i" gave [x, 0, 0]. The parser rejects it, as numpy does.
TEST_CASE("parse_einsum_spec - an output index must come from an operand", "[ComputeGraph][EinsumSpec]") {
    auto const bad = parse_einsum_spec("j <- i ; i");
    REQUIRE_FALSE(bad.has_value());
    REQUIRE_THAT(bad.error().message, Catch::Matchers::ContainsSubstring("output index 'j' appears in neither operand"));

    REQUIRE_FALSE(parse_einsum_spec("i,x <- i,k ; k").has_value());
    REQUIRE_FALSE(parse_einsum_spec("ik ; kj -> ix").has_value());

    // an index carried by only one operand is still fine
    REQUIRE(parse_einsum_spec("ij <- ik ; kj").has_value());
    REQUIRE(parse_einsum_spec("ij <- i ; j").has_value());
    REQUIRE(parse_einsum_spec(" <- i ; i").has_value());
}

// The permute parser took any character as an index, so "j@ <- @j" ran as a
// transpose; einsum specs already rejected it.
TEST_CASE("parse_permute_spec - index labels are alphanumeric and come from the input", "[ComputeGraph][EinsumSpec]") {
    auto const bad_char = parse_permute_spec("j@ <- @j");
    REQUIRE_FALSE(bad_char.has_value());
    REQUIRE_THAT(bad_char.error().message, Catch::Matchers::ContainsSubstring("non-letter character"));

    auto const stray = parse_permute_spec("ik <- ij");
    REQUIRE_FALSE(stray.has_value());
    REQUIRE_THAT(stray.error().message, Catch::Matchers::ContainsSubstring("output index 'k' does not appear in the input"));

    REQUIRE(parse_permute_spec("ji <- ij").has_value());
    REQUIRE(parse_permute_spec("i1,i2 <- i2,i1").has_value());
}

TEST_CASE("validate_einsum_spec - permutation operators", "[ComputeGraph][EinsumSpec]") {
    STATIC_REQUIRE(validate_einsum_spec("i,j,a,b <- P(i/j) P(a/b) i,k,a,c ; k,c,j,b"));
    STATIC_REQUIRE(validate_einsum_spec("i,j,k <- P(i/jk) i,m ; m,j,k"));
    STATIC_REQUIRE(validate_permute_spec("i,j,k <- P(i/jk) i,j,k"));

    STATIC_REQUIRE_FALSE(validate_einsum_spec("ijab <- P(i/j ikac ; kcjb"));   // unbalanced
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ijab <- P() ikac ; kcjb"));     // empty
    STATIC_REQUIRE_FALSE(validate_einsum_spec("ijab <- conj(P(i/j)) ; kcjb")); // nested
}
