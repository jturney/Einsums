//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>

#include <compare>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

namespace cg = einsums::compute_graph;

namespace {

/// Build a space with the fields these tests care about.
cg::IndexSpace make_space(std::string name, std::string symbol, double extent = 0.0) {
    return cg::IndexSpace{
        .name = std::move(name), .scale_symbol = std::move(symbol), .typical_extent = extent, .growth = cg::GrowthClass::linear()};
}

/// A registry with the scale order and typical extents every case below reuses.
///
/// `occ < virt < aux` is declared, and the typical extents respect that ordering, which is the
/// consistency the comparison's transitivity depends on. `grid` and `mesh` are deliberately related
/// to nothing and carry no typical extent, so a pair of them is the case where the first two rungs
/// both have to give up.
struct SpaceFixture {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ{registry.register_space(make_space("occ", "o", 10.0))};
    cg::SpaceId       virt{registry.register_space(make_space("virt", "v", 100.0))};
    cg::SpaceId       aux{registry.register_space(make_space("aux", "x", 500.0))};
    cg::SpaceId       grid{registry.register_space(make_space("grid", "g"))};
    cg::SpaceId       mesh{registry.register_space(make_space("mesh", "m"))};

    SpaceFixture() {
        registry.declare_less(occ, virt);
        registry.declare_less(virt, aux);
    }
};

/// Shorthand for a single power of a single variable.
[[nodiscard]] cg::SymbolicPoly pv(cg::SymbolicVar const &variable, int exponent = 1) {
    return cg::SymbolicPoly::variable(variable, exponent);
}

/// Collapse an ordering to -1, 0 or 1 so a failure prints something readable.
[[nodiscard]] int order_sign(std::strong_ordering order) {
    if (order < 0) {
        return -1;
    }
    if (order > 0) {
        return 1;
    }
    return 0;
}

} // namespace

TEST_CASE("SymbolicCost - canonical form", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    auto const o    = cg::SymbolicVar::space(fixture.occ);
    auto const v    = cg::SymbolicVar::space(fixture.virt);
    auto const anon = cg::SymbolicVar::anonymous("a");

    SECTION("like terms merge") {
        auto const merged = cg::SymbolicPoly::from_terms({
            cg::SymbolicTerm{.coefficient = 2.0, .factors = {cg::SymbolicFactor{.variable = v, .exponent = 1}}},
            cg::SymbolicTerm{.coefficient = 3.0, .factors = {cg::SymbolicFactor{.variable = v, .exponent = 1}}},
        });
        REQUIRE(merged.terms().size() == 1);
        CHECK(merged.terms()[0].coefficient == 5.0);
        CHECK(merged == 5.0 * pv(v));
    }

    SECTION("cancelling terms drop out") {
        auto const cancelled = cg::SymbolicPoly::from_terms({
            cg::SymbolicTerm{.coefficient = 1.0, .factors = {cg::SymbolicFactor{.variable = o, .exponent = 1}}},
            cg::SymbolicTerm{.coefficient = -1.0, .factors = {cg::SymbolicFactor{.variable = o, .exponent = 1}}},
        });
        CHECK(cancelled.is_zero());
        CHECK(cancelled == cg::SymbolicPoly::zero());
        CHECK(cancelled.terms().empty());
    }

    SECTION("repeated factors merge and zero exponents vanish") {
        auto const squared = cg::SymbolicPoly::from_terms({
            cg::SymbolicTerm{
                .coefficient = 1.0,
                .factors     = {cg::SymbolicFactor{.variable = o, .exponent = 1}, cg::SymbolicFactor{.variable = o, .exponent = 1}}},
        });
        REQUIRE(squared.terms().size() == 1);
        REQUIRE(squared.terms()[0].factors.size() == 1);
        CHECK(squared.terms()[0].factors[0].exponent == 2);
        CHECK(squared == pv(o, 2));

        CHECK(pv(o, 0) == cg::SymbolicPoly::constant(1.0));
        CHECK(cg::SymbolicPoly::constant(0.0) == cg::SymbolicPoly::zero());
    }

    SECTION("term order does not depend on construction order") {
        cg::SymbolicTerm const first{.coefficient = 1.0, .factors = {cg::SymbolicFactor{.variable = v, .exponent = 2}}};
        cg::SymbolicTerm const second{.coefficient = 3.0, .factors = {cg::SymbolicFactor{.variable = o, .exponent = 1}}};
        cg::SymbolicTerm const third{
            .coefficient = 1.0,
            .factors     = {cg::SymbolicFactor{.variable = o, .exponent = 1}, cg::SymbolicFactor{.variable = anon, .exponent = 1}}};

        auto const forward  = cg::SymbolicPoly::from_terms({first, second, third});
        auto const backward = cg::SymbolicPoly::from_terms({third, second, first});
        CHECK(forward == backward);
        REQUIRE(forward.terms().size() == 3);

        // Space variables sort before anonymous ones, and within a kind by registration order, so
        // the canonical sequence is o, then o*?a, then v^2.
        CHECK(forward.terms()[0].factors[0].variable == o);
        CHECK(forward.terms()[1].factors.size() == 2);
        CHECK(forward.terms()[1].factors[1].variable == anon);
        CHECK(forward.terms()[2].factors[0].variable == v);
    }

    SECTION("arithmetic identities") {
        auto const a = pv(o);
        auto const b = pv(v);
        auto const c = cg::SymbolicPoly::constant(2.0);

        CHECK(a + b == b + a);
        CHECK((a * b) * c == a * (b * c));
        CHECK(a * cg::SymbolicPoly::constant(1.0) == a);
        CHECK(a * cg::SymbolicPoly::constant(0.0) == cg::SymbolicPoly::zero());
        CHECK(a + cg::SymbolicPoly::zero() == a);
        CHECK(2.0 * a == a * 2.0);
        CHECK(a + a == 2.0 * a);
    }

    SECTION("degree queries") {
        auto const poly = (pv(o, 2) * pv(v)) + pv(anon, 5);
        CHECK(poly.total_degree() == 5);
        CHECK(poly.degree_in(o) == 2);
        CHECK(poly.degree_in(v) == 1);
        CHECK(poly.degree_in(anon) == 5);
        CHECK(poly.degree_in(cg::SymbolicVar::space(fixture.aux)) == 0);

        auto const variables = poly.variables();
        REQUIRE(variables.size() == 3);
        CHECK(variables[0] == o);
        CHECK(variables[1] == v);
        CHECK(variables[2] == anon);

        CHECK(cg::SymbolicPoly::zero().total_degree() == 0);
    }
}

TEST_CASE("SymbolicCost - rendering and evaluation", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    auto const o    = cg::SymbolicVar::space(fixture.occ);
    auto const v    = cg::SymbolicVar::space(fixture.virt);
    auto const anon = cg::SymbolicVar::anonymous("a");

    SECTION("scale symbols and anonymous placeholders") {
        auto const poly = 2.0 * pv(o, 2) * pv(v, 2) * pv(anon);
        CHECK(poly.to_string(&fixture.registry) == "2*o^2*v^2*?a");

        CHECK(cg::SymbolicPoly::zero().to_string(&fixture.registry) == "0");
        CHECK(cg::SymbolicPoly::constant(3.0).to_string(&fixture.registry) == "3");
        CHECK(pv(v).to_string(&fixture.registry) == "v");
        CHECK((pv(o) + pv(v)).to_string(&fixture.registry) == "o + v");
        CHECK((0.5 * pv(o)).to_string(&fixture.registry) == "0.5*o");

        // Without a registry a space variable falls back to its id, and an anonymous one does not
        // need one at all.
        CHECK(pv(v).to_string() == "s1");
        CHECK(pv(anon, 3).to_string() == "?a^3");
    }

    SECTION("substitution") {
        auto const poly    = 2.0 * pv(o, 2) * pv(v);
        auto const resolve = [&](cg::SymbolicVar const &variable) -> std::optional<double> {
            if (variable == o) {
                return 3.0;
            }
            if (variable == v) {
                return 5.0;
            }
            return std::nullopt;
        };

        CHECK(poly.evaluate(resolve) == std::optional{90.0});
        CHECK((poly + pv(anon)).evaluate(resolve) == std::nullopt);
        CHECK(cg::SymbolicPoly::zero().evaluate(resolve) == std::optional{0.0});
        CHECK(cg::SymbolicPoly::constant(7.0).evaluate(cg::ExtentLookup{}) == std::optional{7.0});
        CHECK(pv(o).evaluate(cg::ExtentLookup{}) == std::nullopt);
    }
}

TEST_CASE("SymbolicCost - rung 1 decides by declared scale order", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    cg::ComparisonContext const ctx{.registry = &fixture.registry};

    auto const o    = cg::SymbolicVar::space(fixture.occ);
    auto const v    = cg::SymbolicVar::space(fixture.virt);
    auto const g    = cg::SymbolicVar::space(fixture.grid);
    auto const anon = cg::SymbolicVar::anonymous("a");

    SECTION("the design's own example") {
        auto const lhs = pv(o, 3) * pv(v, 3);
        auto const rhs = pv(o, 2) * pv(v, 4);

        auto const forward = cg::compare_explain(lhs, rhs, ctx);
        CHECK(cg::compare_rung_name(forward.rung) == "ScaleOrder");
        CHECK(order_sign(forward.order) == -1);

        auto const backward = cg::compare_explain(rhs, lhs, ctx);
        CHECK(cg::compare_rung_name(backward.rung) == "ScaleOrder");
        CHECK(order_sign(backward.order) == 1);
    }

    SECTION("extra factors dominate, even anonymous ones") {
        auto const fewer = cg::compare_explain(pv(anon, 2), pv(anon, 3), ctx);
        CHECK(cg::compare_rung_name(fewer.rung) == "ScaleOrder");
        CHECK(order_sign(fewer.order) == -1);

        auto const empty = cg::compare_explain(cg::SymbolicPoly::zero(), pv(o), ctx);
        CHECK(cg::compare_rung_name(empty.rung) == "ScaleOrder");
        CHECK(order_sign(empty.order) == -1);

        auto const unit = cg::compare_explain(cg::SymbolicPoly::constant(1.0), pv(o, 2), ctx);
        CHECK(cg::compare_rung_name(unit.rung) == "ScaleOrder");
        CHECK(order_sign(unit.order) == -1);
    }

    SECTION("an unknown relation falls through") {
        // grid and mesh are related to nothing and have no typical extent, so nothing above the
        // last rung can separate them.
        auto const m         = cg::SymbolicVar::space(fixture.mesh);
        auto const unrelated = cg::compare_explain(pv(g, 2), pv(m, 2), ctx);
        CHECK(cg::compare_rung_name(unrelated.rung) == "Lexicographic");
        CHECK(order_sign(unrelated.order) == -1);
    }

    SECTION("a negative coefficient makes the rung abstain") {
        auto const negative = cg::SymbolicPoly::from_terms({
            cg::SymbolicTerm{.coefficient = -1.0, .factors = {cg::SymbolicFactor{.variable = o, .exponent = 1}}},
        });
        auto const result   = cg::compare_explain(negative, pv(v), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "TypicalExtent");
        CHECK(order_sign(result.order) == -1);
    }

    SECTION("a null registry leaves only self-domination") {
        cg::ComparisonContext const bare;
        auto const                  result = cg::compare_explain(pv(o, 3) * pv(v, 3), pv(o, 2) * pv(v, 4), bare);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");
    }
}

TEST_CASE("SymbolicCost - rung 2 decides by typical extent", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    cg::ComparisonContext const ctx{.registry = &fixture.registry};

    auto const o = cg::SymbolicVar::space(fixture.occ);
    auto const v = cg::SymbolicVar::space(fixture.virt);
    auto const g = cg::SymbolicVar::space(fixture.grid);
    auto const m = cg::SymbolicVar::space(fixture.mesh);

    SECTION("incomparable degrees are settled numerically") {
        // o^3 = 1000 against v = 100: no injective matching either way, so the scale-order rung has
        // nothing to say and the advisory extents decide.
        auto const result = cg::compare_explain(pv(o, 3), pv(v), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "TypicalExtent");
        CHECK(order_sign(result.order) == 1);

        auto const mirrored = cg::compare_explain(pv(v), pv(o, 3), ctx);
        CHECK(cg::compare_rung_name(mirrored.rung) == "TypicalExtent");
        CHECK(order_sign(mirrored.order) == -1);
    }

    SECTION("what substitutes ranks ahead of what does not") {
        auto const missing = cg::compare_explain(pv(o, 3), pv(g), ctx);
        CHECK(cg::compare_rung_name(missing.rung) == "TypicalExtent");
        CHECK(order_sign(missing.order) == -1);

        auto const anonymous = cg::compare_explain(cg::SymbolicPoly::variable(cg::SymbolicVar::anonymous("a")), pv(o, 3), ctx);
        CHECK(cg::compare_rung_name(anonymous.rung) == "TypicalExtent");
        CHECK(order_sign(anonymous.order) == 1);
    }

    SECTION("two unsubstitutable polynomials fall through") {
        auto const result = cg::compare_explain(pv(g, 3), pv(m), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");

        auto const anonymous = cg::compare_explain(pv(g), cg::SymbolicPoly::variable(cg::SymbolicVar::anonymous("a")), ctx);
        CHECK(cg::compare_rung_name(anonymous.rung) == "Lexicographic");
    }
}

TEST_CASE("SymbolicCost - rung 3 decides by bound extents", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    // Only polynomials the typical extents could not touch ever reach this rung, so both operands
    // below are built from the two spaces that carry no typical extent.
    auto const g = cg::SymbolicVar::space(fixture.grid);
    auto const m = cg::SymbolicVar::space(fixture.mesh);

    SECTION("a complete table decides") {
        cg::ComparisonContext const ctx{.registry     = &fixture.registry,
                                        .bound_extent = [&](cg::SymbolicVar const &variable) -> std::optional<double> {
                                            if (variable == g) {
                                                return 3.0;
                                            }
                                            if (variable == m) {
                                                return 4.0;
                                            }
                                            return std::nullopt;
                                        }};

        auto const result = cg::compare_explain(pv(g, 3), pv(m), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "BoundExtent");
        CHECK(order_sign(result.order) == 1);

        auto const flipped = cg::compare_explain(pv(g), pv(m), ctx);
        CHECK(cg::compare_rung_name(flipped.rung) == "BoundExtent");
        CHECK(order_sign(flipped.order) == -1);
    }

    SECTION("a table that resolves only one ranks that one first") {
        cg::ComparisonContext const ctx{.registry     = &fixture.registry,
                                        .bound_extent = [&](cg::SymbolicVar const &variable) -> std::optional<double> {
                                            if (variable == m) {
                                                return 4.0;
                                            }
                                            return std::nullopt;
                                        }};

        auto const result = cg::compare_explain(pv(g, 3), pv(m), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "BoundExtent");
        CHECK(order_sign(result.order) == 1);
    }

    SECTION("a table that resolves neither falls through") {
        cg::ComparisonContext const ctx{.registry     = &fixture.registry,
                                        .bound_extent = [&](cg::SymbolicVar const &variable) -> std::optional<double> {
                                            if (variable == cg::SymbolicVar::space(fixture.occ)) {
                                                return 4.0;
                                            }
                                            return std::nullopt;
                                        }};

        auto const result = cg::compare_explain(pv(g, 3), pv(m), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");
    }
}

TEST_CASE("SymbolicCost - rung 4 is arbitrary but deterministic", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    cg::ComparisonContext const ctx{.registry = &fixture.registry};

    auto const o = cg::SymbolicVar::space(fixture.occ);
    auto const v = cg::SymbolicVar::space(fixture.virt);
    auto const a = cg::SymbolicVar::anonymous("a");
    auto const b = cg::SymbolicVar::anonymous("b");

    SECTION("numerically equal but structurally different") {
        // o^2 and v both come to 100 at the typical extents, and neither dominates the other, so
        // the last rung picks by canonical form: occ was registered before virt.
        auto const forward = cg::compare_explain(pv(o, 2), pv(v), ctx);
        CHECK(cg::compare_rung_name(forward.rung) == "Lexicographic");
        CHECK(order_sign(forward.order) == -1);

        auto const backward = cg::compare_explain(pv(v), pv(o, 2), ctx);
        CHECK(cg::compare_rung_name(backward.rung) == "Lexicographic");
        CHECK(order_sign(backward.order) == 1);
    }

    SECTION("identical canonical forms compare equal") {
        auto const left  = pv(o, 2) * pv(v);
        auto const right = pv(v) * pv(o) * pv(o);
        REQUIRE(left == right);

        auto const result = cg::compare_explain(left, right, ctx);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");
        CHECK(order_sign(result.order) == 0);
    }

    SECTION("coefficients break a monomial tie") {
        // Anonymous variables keep the numeric rungs out of it, so the coefficients are all the
        // last rung has left to look at.
        auto const result = cg::compare_explain(pv(a), 2.0 * pv(a), ctx);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");
        CHECK(order_sign(result.order) == -1);
    }

    SECTION("a shorter term list orders first") {
        // The negative coefficient makes the scale-order rung abstain, which is the only way to
        // reach the last rung with one term list a prefix of the other.
        auto const longer = cg::SymbolicPoly::from_terms({
            cg::SymbolicTerm{.coefficient = 1.0, .factors = {cg::SymbolicFactor{.variable = a, .exponent = 1}}},
            cg::SymbolicTerm{.coefficient = -1.0, .factors = {cg::SymbolicFactor{.variable = b, .exponent = 1}}},
        });
        auto const result = cg::compare_explain(pv(a), longer, ctx);
        CHECK(cg::compare_rung_name(result.rung) == "Lexicographic");
        CHECK(order_sign(result.order) == -1);
    }
}

TEST_CASE("SymbolicCost - node cost derivation", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    auto const o = cg::SymbolicVar::space(fixture.occ);
    auto const v = cg::SymbolicVar::space(fixture.virt);

    SECTION("ij,jk->ik fully annotated") {
        cg::EinsumDescriptor const desc{
            .spec          = {.c_indices      = {"i", "k"},
                              .a_indices      = {"i", "j"},
                              .b_indices      = {"j", "k"},
                              .all_indices    = {"i", "k", "j"},
                              .link_indices   = {"j"},
                              .target_indices = {"i", "k"}},
            .letter_spaces = {{"i", fixture.occ}, {"j", fixture.occ}, {"k", fixture.virt}},
        };

        auto const cost = cg::symbolic_cost_for(desc);

        // Two flops per loop iteration over the distinct loop letters i, j, k, which map to o, o, v.
        CHECK(cost.flops == 2.0 * pv(o, 2) * pv(v));
        CHECK(cost.flops.to_string(&fixture.registry) == "2*o^2*v");

        // C is o*v, A is o*o, B is o*v.
        CHECK(cost.traffic == pv(o, 2) + (2.0 * pv(o) * pv(v)));
        CHECK(cost.resident == cost.traffic);

        CHECK(cg::symbolic_size_for(desc.spec.c_indices, cg::letter_vars_for(desc)) == pv(o) * pv(v));
        CHECK(cg::letter_vars_for(desc).annotated_count() == 3);
    }

    SECTION("half annotated letters go anonymous") {
        cg::EinsumDescriptor const desc{
            .spec          = {.c_indices      = {"i", "k"},
                              .a_indices      = {"i", "j"},
                              .b_indices      = {"j", "k"},
                              .all_indices    = {"i", "k", "j"},
                              .link_indices   = {"j"},
                              .target_indices = {"i", "k"}},
            .letter_spaces = {{"i", fixture.occ}},
        };

        auto const cost = cg::symbolic_cost_for(desc);
        auto const j    = cg::SymbolicVar::anonymous("j");
        auto const k    = cg::SymbolicVar::anonymous("k");

        CHECK(cost.flops == 2.0 * pv(o) * pv(j) * pv(k));
        CHECK(cost.flops.to_string(&fixture.registry) == "2*o*?j*?k");
        CHECK(cost.traffic == (pv(o) * pv(k)) + (pv(o) * pv(j)) + (pv(j) * pv(k)));
    }

    SECTION("an unannotated node is all anonymous") {
        cg::EinsumDescriptor const desc{
            .spec = {.c_indices      = {"i"},
                     .a_indices      = {"i", "j"},
                     .b_indices      = {"j"},
                     .all_indices    = {"i", "j"},
                     .link_indices   = {"j"},
                     .target_indices = {"i"}},
        };

        auto const cost = cg::symbolic_cost_for(desc);
        auto const i    = cg::SymbolicVar::anonymous("i");
        auto const j    = cg::SymbolicVar::anonymous("j");

        CHECK(cost.flops == 2.0 * pv(i) * pv(j));
        CHECK(cg::letter_vars_for(desc).annotated_count() == 0);
    }

    SECTION("a repeated letter counts once") {
        cg::EinsumDescriptor const desc{
            .spec          = {.c_indices      = {"j"},
                              .a_indices      = {"i", "i"},
                              .b_indices      = {"i", "j"},
                              .all_indices    = {"j", "i"},
                              .link_indices   = {"i"},
                              .target_indices = {"j"}},
            .letter_spaces = {{"i", fixture.occ}, {"j", fixture.virt}},
        };

        auto const cost = cg::symbolic_cost_for(desc);
        CHECK(cost.flops == 2.0 * pv(o) * pv(v));
        CHECK(cg::symbolic_size_for(desc.spec.a_indices, cg::letter_vars_for(desc)) == pv(o));
        CHECK(cost.traffic == pv(v) + pv(o) + (pv(o) * pv(v)));
    }

    SECTION("a scalar index list is the constant one") {
        cg::LetterVars const empty;
        CHECK(cg::symbolic_size_for({}, empty) == cg::SymbolicPoly::constant(1.0));
        CHECK(empty.var_for("i") == cg::SymbolicVar::anonymous("i"));
    }
}

TEST_CASE("SymbolicCost - bundle comparison ranks flops first", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    cg::ComparisonContext const ctx{.registry = &fixture.registry};

    auto const o = cg::SymbolicVar::space(fixture.occ);
    auto const v = cg::SymbolicVar::space(fixture.virt);

    cg::SymbolicCost const cheap{.flops = 2.0 * pv(o, 3) * pv(v, 3), .traffic = pv(v, 4), .resident = pv(v, 4)};
    cg::SymbolicCost const dear{.flops = 2.0 * pv(o, 2) * pv(v, 4), .traffic = pv(o), .resident = pv(o)};

    SECTION("flops win even when traffic disagrees") {
        auto const result = cg::compare_explain(cheap, dear, ctx);
        CHECK(cg::cost_component_name(result.component) == "Flops");
        CHECK(cg::compare_rung_name(result.rung) == "ScaleOrder");
        CHECK(order_sign(result.order) == -1);
    }

    SECTION("traffic breaks an exact flop tie") {
        cg::SymbolicCost const other{.flops = cheap.flops, .traffic = pv(v, 5), .resident = cheap.resident};

        auto const result = cg::compare_explain(cheap, other, ctx);
        CHECK(cg::cost_component_name(result.component) == "Traffic");
        CHECK(order_sign(result.order) == -1);
    }

    SECTION("resident breaks the last tie") {
        cg::SymbolicCost const other{.flops = cheap.flops, .traffic = cheap.traffic, .resident = pv(v, 5)};

        auto const result = cg::compare_explain(cheap, other, ctx);
        CHECK(cg::cost_component_name(result.component) == "Resident");
        CHECK(order_sign(result.order) == -1);

        CHECK(order_sign(cg::compare(cheap, cheap, ctx)) == 0);
        CHECK(cheap.to_string(&fixture.registry) == "flops=2*o^3*v^3, traffic=v^4, resident=v^4");
    }
}

TEST_CASE("SymbolicCost - comparison is a total order", "[ComputeGraph][Spaces][SymbolicCost]") {
    SpaceFixture const fixture;

    cg::ComparisonContext const ctx{.registry = &fixture.registry};

    auto const o    = cg::SymbolicVar::space(fixture.occ);
    auto const v    = cg::SymbolicVar::space(fixture.virt);
    auto const x    = cg::SymbolicVar::space(fixture.aux);
    auto const anon = cg::SymbolicVar::anonymous("a");

    std::vector<cg::SymbolicPoly> const polys{
        cg::SymbolicPoly::zero(),
        cg::SymbolicPoly::constant(1.0),
        pv(o),
        pv(v),
        pv(x),
        pv(o, 2),
        pv(v, 2),
        pv(o) * pv(v),
        2.0 * pv(o) * pv(v),
        pv(o, 2) * pv(v, 2),
        pv(o, 3) * pv(v, 3),
        pv(o, 2) * pv(v, 4),
        pv(o) + pv(v),
        pv(anon),
        pv(anon, 2),
        pv(anon) * pv(o),
    };

    // No duplicates, so "compares equal" and "is the same polynomial" must coincide.
    for (std::size_t i = 0; i < polys.size(); ++i) {
        for (std::size_t j = 0; j < polys.size(); ++j) {
            INFO("i=" << i << " j=" << j);
            CHECK((polys[i] == polys[j]) == (i == j));
        }
    }

    for (auto const &poly : polys) {
        INFO(poly.to_string(&fixture.registry));
        CHECK(order_sign(cg::compare(poly, poly, ctx)) == 0);
    }

    for (std::size_t i = 0; i < polys.size(); ++i) {
        for (std::size_t j = 0; j < polys.size(); ++j) {
            INFO(polys[i].to_string(&fixture.registry) << " vs " << polys[j].to_string(&fixture.registry));
            int const forward  = order_sign(cg::compare(polys[i], polys[j], ctx));
            int const backward = order_sign(cg::compare(polys[j], polys[i], ctx));
            CHECK(forward == -backward);
            CHECK((forward == 0) == (i == j));
        }
    }

    for (std::size_t i = 0; i < polys.size(); ++i) {
        for (std::size_t j = 0; j < polys.size(); ++j) {
            if (order_sign(cg::compare(polys[i], polys[j], ctx)) > 0) {
                continue;
            }
            for (std::size_t k = 0; k < polys.size(); ++k) {
                if (order_sign(cg::compare(polys[j], polys[k], ctx)) > 0) {
                    continue;
                }
                INFO(polys[i].to_string(&fixture.registry)
                     << " <= " << polys[j].to_string(&fixture.registry) << " <= " << polys[k].to_string(&fixture.registry));
                CHECK(order_sign(cg::compare(polys[i], polys[k], ctx)) <= 0);
            }
        }
    }
}
