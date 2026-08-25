//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>

#include <stdexcept>
#include <unordered_map>

#include <Einsums/Testing.hpp>

namespace cg = einsums::compute_graph;

namespace {

/// Build a space with the fields the tests care about, keeping the cases short.
cg::IndexSpace make_space(std::string name, std::string symbol, double extent = 0.0, cg::GrowthClass growth = cg::GrowthClass::linear()) {
    return cg::IndexSpace{.name = std::move(name), .scale_symbol = std::move(symbol), .typical_extent = extent, .growth = growth};
}

} // namespace

TEST_CASE("SpaceRegistry - registration and lookup", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    REQUIRE(registry.size() == 0);

    auto const occ  = registry.register_space(make_space("occ", "o", 10.0));
    auto const virt = registry.register_space(make_space("virt", "v", 100.0));

    REQUIRE(registry.size() == 2);
    CHECK(occ.valid());
    CHECK(occ != virt);

    // Re-registering identical content is idempotent and hands back the same id.
    auto const occ_again = registry.register_space(make_space("occ", "o", 10.0));
    CHECK(occ_again == occ);
    CHECK(registry.size() == 2);

    // Any difference in content is a conflict, not a silent overwrite.
    CHECK_THROWS_AS(registry.register_space(make_space("occ", "o", 12.0)), std::invalid_argument);
    CHECK_THROWS_AS(registry.register_space(make_space("occ", "i", 10.0)), std::invalid_argument);
    CHECK_THROWS_AS(registry.register_space(make_space("occ", "o", 10.0, cg::GrowthClass::constant())), std::invalid_argument);
    CHECK(registry.size() == 2);

    // An empty name is rejected outright.
    CHECK_THROWS_AS(registry.register_space(make_space("", "q")), std::invalid_argument);

    // Lookup by name.
    auto const found = registry.find("virt");
    REQUIRE(found.has_value());
    CHECK(found == std::optional{virt});
    CHECK_FALSE(registry.find("nope").has_value());

    // Lookup by id.
    CHECK(registry.space(occ).name == "occ");
    CHECK(registry.space(occ).scale_symbol == "o");
    CHECK(registry.space(virt).typical_extent == 100.0);
    CHECK(registry.space(virt).growth == cg::GrowthClass::linear());

    // An invalid id is an error everywhere, not a silent default.
    cg::SpaceId const nothing;
    CHECK_FALSE(nothing.valid());
    CHECK_THROWS_AS(registry.space(nothing), std::invalid_argument);
    CHECK_THROWS_AS(registry.is_less(nothing, occ), std::invalid_argument);
    CHECK_THROWS_AS(registry.declare_disjoint(occ, nothing), std::invalid_argument);

    // Ids come back in registration order.
    auto const ids = registry.ids();
    REQUIRE(ids.size() == 2);
    CHECK(ids[0] == occ);
    CHECK(ids[1] == virt);
}

TEST_CASE("SpaceRegistry - growth classes and space value semantics", "[ComputeGraph][Spaces]") {
    CHECK(cg::GrowthClass::constant().exponent == 0.0);
    CHECK(cg::GrowthClass::linear().exponent == 1.0);
    CHECK(cg::GrowthClass::power(2.5).exponent == 2.5);
    CHECK(cg::GrowthClass::power(1.0) == cg::GrowthClass::linear());
    CHECK_FALSE(cg::GrowthClass::constant() == cg::GrowthClass::linear());

    // A default-constructed space grows linearly: that is the common case for orbital spaces.
    cg::IndexSpace const fresh;
    CHECK(fresh.growth == cg::GrowthClass::linear());
    CHECK(fresh.typical_extent == 0.0);

    CHECK(make_space("aux", "x") == make_space("aux", "x"));
    CHECK_FALSE(make_space("aux", "x") == make_space("aux", "y"));

    // A grid can be declared with a user-defined exponent rather than an enum the library owns.
    cg::SpaceRegistry registry;
    auto const        grid = registry.register_space(make_space("grid", "g", 50000.0, cg::GrowthClass::power(1.5)));
    CHECK(registry.space(grid).growth.exponent == 1.5);
}

TEST_CASE("SpaceRegistry - scale ordering is transitive and acyclic", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ  = registry.register_space(make_space("occ", "o"));
    auto const virt = registry.register_space(make_space("virt", "v"));
    auto const aux  = registry.register_space(make_space("aux", "x"));
    auto const grid = registry.register_space(make_space("grid", "g"));

    // Nothing declared yet: no ordering is knowable.
    CHECK(registry.is_less(occ, virt) == cg::Tristate::Unknown);

    registry.declare_less(occ, virt);
    registry.declare_less(virt, aux);

    CHECK(registry.is_less(occ, virt) == cg::Tristate::Yes);
    CHECK(registry.is_less(virt, occ) == cg::Tristate::No);

    // Transitivity: occ < aux was never declared.
    CHECK(registry.is_less(occ, aux) == cg::Tristate::Yes);
    CHECK(registry.is_less(aux, occ) == cg::Tristate::No);

    // grid is unrelated to all of them.
    CHECK(registry.is_less(grid, occ) == cg::Tristate::Unknown);
    CHECK(registry.is_less(occ, grid) == cg::Tristate::Unknown);

    // A space is not smaller than itself, and saying so is an error.
    CHECK(registry.is_less(occ, occ) == cg::Tristate::No);
    CHECK_THROWS_AS(registry.declare_less(occ, occ), std::invalid_argument);

    // Direct and transitive cycles are both rejected.
    CHECK_THROWS_AS(registry.declare_less(virt, occ), std::invalid_argument);
    CHECK_THROWS_AS(registry.declare_less(aux, occ), std::invalid_argument);

    // The rejected declarations left nothing behind.
    CHECK(registry.is_less(occ, aux) == cg::Tristate::Yes);

    // Re-declaring something already true is harmless.
    REQUIRE_NOTHROW(registry.declare_less(occ, virt));
    CHECK(registry.is_less(occ, virt) == cg::Tristate::Yes);
}

TEST_CASE("SpaceRegistry - disjointness is symmetric", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ  = registry.register_space(make_space("occ", "o"));
    auto const virt = registry.register_space(make_space("virt", "v"));
    auto const aux  = registry.register_space(make_space("aux", "x"));

    CHECK(registry.is_disjoint(occ, virt) == cg::Tristate::Unknown);

    registry.declare_disjoint(occ, virt);

    CHECK(registry.is_disjoint(occ, virt) == cg::Tristate::Yes);
    CHECK(registry.is_disjoint(virt, occ) == cg::Tristate::Yes);

    // Undeclared pairs stay unknown.
    CHECK(registry.is_disjoint(occ, aux) == cg::Tristate::Unknown);

    // A non-empty space overlaps itself, and declaring otherwise is an error.
    CHECK(registry.is_disjoint(occ, occ) == cg::Tristate::No);
    CHECK_THROWS_AS(registry.declare_disjoint(occ, occ), std::invalid_argument);

    // Re-declaring is harmless.
    REQUIRE_NOTHROW(registry.declare_disjoint(virt, occ));
    CHECK(registry.is_disjoint(occ, virt) == cg::Tristate::Yes);
}

TEST_CASE("SpaceRegistry - containment is transitive and reflexive", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const virt = registry.register_space(make_space("virt", "v"));
    auto const pao  = registry.register_space(make_space("pao", "r"));
    auto const pno  = registry.register_space(make_space("pno", "a"));
    auto const occ  = registry.register_space(make_space("occ", "o"));

    CHECK(registry.is_contained(pno, virt) == cg::Tristate::Unknown);

    registry.declare_contained(pno, pao);
    registry.declare_contained(pao, virt);

    CHECK(registry.is_contained(pno, pao) == cg::Tristate::Yes);
    CHECK(registry.is_contained(pao, virt) == cg::Tristate::Yes);

    // Transitivity: pno within virt was never declared.
    CHECK(registry.is_contained(pno, virt) == cg::Tristate::Yes);

    // The reverse direction is definitively false, not unknown.
    CHECK(registry.is_contained(virt, pno) == cg::Tristate::No);

    // Reflexive containment is derived, but declaring it is rejected as a mistake.
    CHECK(registry.is_contained(pno, pno) == cg::Tristate::Yes);
    CHECK_THROWS_AS(registry.declare_contained(pno, pno), std::invalid_argument);

    // Unrelated spaces stay unknown.
    CHECK(registry.is_contained(pno, occ) == cg::Tristate::Unknown);
    CHECK(registry.is_contained(occ, pno) == cg::Tristate::Unknown);

    // Containment cycles are rejected in both the direct and the transitive form.
    CHECK_THROWS_AS(registry.declare_contained(pao, pno), std::invalid_argument);
    CHECK_THROWS_AS(registry.declare_contained(virt, pno), std::invalid_argument);

    // Containment implies nothing about scale order: that has to be declared separately.
    CHECK(registry.is_less(pno, virt) == cg::Tristate::Unknown);

    // Re-declaring is harmless.
    REQUIRE_NOTHROW(registry.declare_contained(pno, pao));
    CHECK(registry.is_contained(pno, virt) == cg::Tristate::Yes);
}

TEST_CASE("SpaceRegistry - containment and disjointness cannot contradict", "[ComputeGraph][Spaces]") {
    SECTION("containment first, then the conflicting disjointness") {
        cg::SpaceRegistry registry;
        auto const        virt = registry.register_space(make_space("virt", "v"));
        auto const        pno  = registry.register_space(make_space("pno", "a"));

        registry.declare_contained(pno, virt);
        CHECK_THROWS_AS(registry.declare_disjoint(pno, virt), std::invalid_argument);
        CHECK_THROWS_AS(registry.declare_disjoint(virt, pno), std::invalid_argument);

        // The rejection did not corrupt what was already held.
        CHECK(registry.is_contained(pno, virt) == cg::Tristate::Yes);
        CHECK(registry.is_disjoint(pno, virt) == cg::Tristate::No);
    }

    SECTION("disjointness first, then the conflicting containment") {
        cg::SpaceRegistry registry;
        auto const        virt = registry.register_space(make_space("virt", "v"));
        auto const        pno  = registry.register_space(make_space("pno", "a"));

        registry.declare_disjoint(pno, virt);
        CHECK_THROWS_AS(registry.declare_contained(pno, virt), std::invalid_argument);
        CHECK_THROWS_AS(registry.declare_contained(virt, pno), std::invalid_argument);

        CHECK(registry.is_disjoint(pno, virt) == cg::Tristate::Yes);
        CHECK(registry.is_contained(pno, virt) == cg::Tristate::No);
    }

    SECTION("a space cannot end up inside two disjoint spaces, however indirectly") {
        cg::SpaceRegistry registry;
        auto const        occ  = registry.register_space(make_space("occ", "o"));
        auto const        virt = registry.register_space(make_space("virt", "v"));
        auto const        pno  = registry.register_space(make_space("pno", "a"));

        registry.declare_disjoint(occ, virt);
        registry.declare_contained(pno, virt);

        // pno is already inside virt, which is disjoint from occ.
        CHECK_THROWS_AS(registry.declare_contained(pno, occ), std::invalid_argument);

        // And the other way around: declaring the disjointness last is caught too.
        cg::SpaceRegistry other;
        auto const        occ2  = other.register_space(make_space("occ", "o"));
        auto const        virt2 = other.register_space(make_space("virt", "v"));
        auto const        pno2  = other.register_space(make_space("pno", "a"));
        other.declare_contained(pno2, virt2);
        other.declare_contained(pno2, occ2);
        CHECK_THROWS_AS(other.declare_disjoint(occ2, virt2), std::invalid_argument);
    }
}

TEST_CASE("SpaceRegistry - disjointness derived through containment", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ  = registry.register_space(make_space("occ", "o"));
    auto const virt = registry.register_space(make_space("virt", "v"));
    auto const pao  = registry.register_space(make_space("pao", "r"));
    auto const pno  = registry.register_space(make_space("pno", "a"));
    auto const froz = registry.register_space(make_space("frozen", "f"));

    registry.declare_disjoint(occ, virt);
    registry.declare_contained(pao, virt);
    registry.declare_contained(pno, pao);
    registry.declare_contained(froz, occ);

    // One level of containment on one side.
    CHECK(registry.is_disjoint(pao, occ) == cg::Tristate::Yes);
    CHECK(registry.is_disjoint(occ, pao) == cg::Tristate::Yes);

    // Two levels on one side.
    CHECK(registry.is_disjoint(pno, occ) == cg::Tristate::Yes);

    // Containment on both sides at once.
    CHECK(registry.is_disjoint(pno, froz) == cg::Tristate::Yes);
    CHECK(registry.is_disjoint(froz, pno) == cg::Tristate::Yes);

    // Derived disjointness settles containment in the negative as well.
    CHECK(registry.is_contained(pno, occ) == cg::Tristate::No);

    // Siblings inside one parent are NOT disjoint by construction; nobody declared they are.
    auto const other_pao = registry.register_space(make_space("pao2", "r2"));
    registry.declare_contained(other_pao, virt);
    CHECK(registry.is_disjoint(pao, other_pao) == cg::Tristate::Unknown);
}

TEST_CASE("SpaceRegistry - policy table", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ = registry.register_space(make_space("occ", "o"));
    registry.register_space(make_space("virt", "v"));

    // Unset names answer with the undecided default.
    cg::SpacePolicy const fallback = registry.policy("occ");
    CHECK(fallback.dist == cg::DistributionHint::None);
    CHECK(fallback.storage == cg::StorageHint::None);
    CHECK(registry.policy("never-registered") == cg::SpacePolicy{});

    registry.set_policy("occ", cg::SpacePolicy{.dist = cg::DistributionHint::Replicate, .storage = cg::StorageHint::PreferMemory});
    registry.set_policy("virt", cg::SpacePolicy{.dist = cg::DistributionHint::Tile, .storage = cg::StorageHint::SpillOk});

    CHECK(registry.policy("occ").dist == cg::DistributionHint::Replicate);
    CHECK(registry.policy("occ").storage == cg::StorageHint::PreferMemory);
    CHECK(registry.policy("virt").dist == cg::DistributionHint::Tile);

    // Setting again replaces.
    registry.set_policy("occ", cg::SpacePolicy{.dist = cg::DistributionHint::Tile, .storage = cg::StorageHint::None});
    CHECK(registry.policy("occ").dist == cg::DistributionHint::Tile);
    CHECK(registry.policy("occ").storage == cg::StorageHint::None);

    // Policy is keyed by name and does not need the space to exist: it is configuration, and it
    // never travels inside the IndexSpace itself.
    REQUIRE_NOTHROW(registry.set_policy("not-yet-registered",
                                        cg::SpacePolicy{.dist = cg::DistributionHint::Replicate, .storage = cg::StorageHint::None}));
    CHECK(registry.policy("not-yet-registered").dist == cg::DistributionHint::Replicate);
    CHECK_THROWS_AS(registry.set_policy("", cg::SpacePolicy{}), std::invalid_argument);

    // Registering a space did not invent a policy for it.
    CHECK(registry.space(occ).name == "occ");
}

TEST_CASE("SpaceRegistry - ids hash and order for use as container keys", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ  = registry.register_space(make_space("occ", "o"));
    auto const virt = registry.register_space(make_space("virt", "v"));

    std::unordered_map<cg::SpaceId, int> counts;
    counts[occ]  = 1;
    counts[virt] = 2;
    counts[occ] += 10;

    CHECK(counts.size() == 2);
    CHECK(counts.at(occ) == 11);
    CHECK(counts.at(virt) == 2);
    CHECK(occ < virt);
}

TEST_CASE("SpaceRegistry - chemistry spaces", "[ComputeGraph][Spaces]") {
    cg::SpaceRegistry registry;

    auto const occ  = registry.register_space(make_space("occ", "o", 20.0));
    auto const virt = registry.register_space(make_space("virt", "v", 200.0));
    auto const aux  = registry.register_space(make_space("aux", "x", 800.0));
    auto const pao  = registry.register_space(make_space("pao", "r", 60.0));
    auto const pno  = registry.register_space(make_space("pno", "a", 12.0, cg::GrowthClass::constant()));
    auto const grid = registry.register_space(make_space("grid", "g", 1.0e5, cg::GrowthClass::power(1.5)));

    registry.declare_less(occ, virt);
    registry.declare_less(occ, aux);
    registry.declare_contained(pno, virt);
    registry.declare_contained(pao, virt);
    registry.declare_disjoint(occ, virt);

    // Scale order, declared and transitive-by-absence.
    CHECK(registry.is_less(occ, virt) == cg::Tristate::Yes);
    CHECK(registry.is_less(occ, aux) == cg::Tristate::Yes);
    CHECK(registry.is_less(virt, aux) == cg::Tristate::Unknown);

    // A local space is inside the virtual space, so it is disjoint from the occupied space even
    // though nobody said so.
    CHECK(registry.is_contained(pno, virt) == cg::Tristate::Yes);
    CHECK(registry.is_contained(pao, virt) == cg::Tristate::Yes);
    CHECK(registry.is_disjoint(pno, occ) == cg::Tristate::Yes);
    CHECK(registry.is_disjoint(pao, occ) == cg::Tristate::Yes);

    // pno and pao both sit in virt and are not known to be disjoint from each other.
    CHECK(registry.is_disjoint(pno, pao) == cg::Tristate::Unknown);

    // The grid is unrelated to the orbital spaces in every relation.
    CHECK(registry.is_less(grid, occ) == cg::Tristate::Unknown);
    CHECK(registry.is_disjoint(grid, occ) == cg::Tristate::Unknown);
    CHECK(registry.is_contained(grid, occ) == cg::Tristate::Unknown);

    // Semantics survive the round trip through the registry, growth exponents included.
    CHECK(registry.space(pno).scale_symbol == "a");
    CHECK(registry.space(pno).growth == cg::GrowthClass::constant());
    CHECK(registry.space(grid).growth.exponent == 1.5);
    CHECK(registry.space(aux).typical_extent == 800.0);
}

TEST_CASE("SpaceRegistry - the process-wide registry is one object", "[ComputeGraph][Spaces]") {
    auto &registry = cg::global_space_registry();
    CHECK(&registry == &cg::global_space_registry());

    // Register under names unlikely to clash with anything a later task puts there.
    auto const occ  = registry.register_space(make_space("test.global.occ", "o", 10.0));
    auto const virt = registry.register_space(make_space("test.global.virt", "v", 100.0));
    registry.declare_less(occ, virt);

    auto const found = cg::global_space_registry().find("test.global.occ");
    REQUIRE(found.has_value());
    CHECK(found == std::optional{occ});
    CHECK(cg::global_space_registry().is_less(occ, virt) == cg::Tristate::Yes);
}
