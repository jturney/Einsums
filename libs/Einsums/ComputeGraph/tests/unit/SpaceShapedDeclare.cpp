//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;

    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .dim_symbol = "no", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .dim_symbol = "nv", .typical_extent = 8.0});
    }
};

} // namespace

TEST_CASE("SpaceShapedDeclare - an annotated operand teaches the graph what a space measures", "[ComputeGraph][Spaces][SpaceShape]") {
    Spaces spaces;

    cg::Graph graph("learn");
    graph.set_space_registry(spaces.registry);

    CHECK_FALSE(graph.space_extent(spaces.occ).has_value());

    RuntimeTensor<double> const t2("t2", {4, 8});
    graph.annotate_spaces(t2, {spaces.occ, spaces.virt});

    REQUIRE(graph.space_extent(spaces.occ).has_value());
    CHECK(*graph.space_extent(spaces.occ) == 4);
    CHECK(*graph.space_extent(spaces.virt) == 8);
}

TEST_CASE("SpaceShapedDeclare - a space-shaped declare sizes and annotates in one call", "[ComputeGraph][Spaces][SpaceShape]") {
    Spaces spaces;

    cg::Graph graph("declare");
    graph.set_space_registry(spaces.registry);

    RuntimeTensor<double> const t2("t2", {4, 8});
    graph.annotate_spaces(t2, {spaces.occ, spaces.virt});

    // THE case the overload exists for: the shape is stated in what the axes MEAN, and the
    // extents, the space annotation and the dim symbols all follow from that one statement.
    auto &scratch = graph.declare_zero_runtime_tensor<double>("scratch", {spaces.occ, spaces.virt});

    CHECK(scratch.dim(0) == 4);
    CHECK(scratch.dim(1) == 8);

    auto const id = graph.find_tensor_id_by_ptr(&scratch);
    REQUIRE(id != 0);
    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"no", "nv"});

    // The symbol carries the tie, so the graph agrees the two name one space each.
    auto const &ties = graph.symbol_spaces();
    REQUIRE(ties.contains("no"));
    CHECK(ties.at("no") == spaces.occ);
    CHECK(ties.at("nv") == spaces.virt);
}

TEST_CASE("SpaceShapedDeclare - a literal axis mixes with a space-typed one", "[ComputeGraph][Spaces][SpaceShape]") {
    Spaces spaces;

    cg::Graph graph("mixed");
    graph.set_space_registry(spaces.registry);
    graph.pin_space_extent(spaces.occ, 4);

    // A DIIS history axis is genuinely a number: it means nothing chemically and must not be
    // rebound with the problem. It stays literal, and only the space-typed axis gets a symbol.
    auto &history = graph.declare_zero_runtime_tensor<double>("history", {cg::fixed(8), spaces.occ});

    CHECK(history.dim(0) == 8);
    CHECK(history.dim(1) == 4);

    auto const id = graph.find_tensor_id_by_ptr(&history);
    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"", "no"});
    CHECK(graph.tensor_spaces(id)[0] == cg::SpaceId{});
    CHECK(graph.tensor_spaces(id)[1] == spaces.occ);
}

TEST_CASE("SpaceShapedDeclare - an unpinned space is refused by name", "[ComputeGraph][Spaces][SpaceShape]") {
    Spaces spaces;

    cg::Graph graph("unpinned");
    graph.set_space_registry(spaces.registry);

    // Nothing has said how big 'virt' is, and guessing from typical_extent would be exactly
    // the advisory-value-used-as-truth mistake that field's doc comment warns against.
    CHECK_THROWS_WITH(graph.declare_zero_runtime_tensor<double>("scratch", {spaces.virt}),
                      Catch::Matchers::ContainsSubstring("virt") && Catch::Matchers::ContainsSubstring("how big it is"));
}

TEST_CASE("SpaceShapedDeclare - a ragged family unpins the space rather than erroring", "[ComputeGraph][Spaces][SpaceShape][Ragged]") {
    Spaces spaces;

    cg::Graph graph("ragged");
    graph.set_space_registry(spaces.registry);

    // Two operands over one space with different extents is what a PNO domain looks like. It
    // is legal, so annotating the second must NOT throw - but the space stops being something
    // an axis can be sized from, and the message has to say which of the two problems it is.
    RuntimeTensor<double> const pair_one("pair_one", {4, 8});
    RuntimeTensor<double> const pair_two("pair_two", {4, 5});
    graph.annotate_spaces(pair_one, {spaces.occ, spaces.virt});
    REQUIRE_NOTHROW(graph.annotate_spaces(pair_two, {spaces.occ, spaces.virt}));

    CHECK(graph.space_extent(spaces.occ).has_value()); // agreed at 4
    CHECK_FALSE(graph.space_extent(spaces.virt).has_value());

    CHECK_THROWS_WITH(graph.declare_zero_runtime_tensor<double>("scratch", {spaces.occ, spaces.virt}),
                      Catch::Matchers::ContainsSubstring("ragged family") && Catch::Matchers::ContainsSubstring("annotate_ragged_dim"));
}

TEST_CASE("SpaceShapedDeclare - an all-literal shape annotates nothing", "[ComputeGraph][Spaces][SpaceShape]") {
    Spaces spaces;

    cg::Graph graph("literal");
    graph.set_space_registry(spaces.registry);

    // Indistinguishable from the dims-based overload, which is the point: a caller who names
    // no space has opted out, and must not find symbols invented for them.
    auto      &plain = graph.declare_zero_runtime_tensor<double>("plain", {3, 3});
    auto const id    = graph.find_tensor_id_by_ptr(&plain);

    CHECK(plain.dim(0) == 3);
    CHECK(graph.tensor_spaces(id).empty());
    CHECK(graph.tensor_dim_symbols(id).empty());
}

TEST_CASE("SpaceShapedDeclare - the declared tensor is rebindable at a new extent", "[ComputeGraph][Spaces][SpaceShape][Bind]") {
    Spaces spaces;

    cg::Graph graph("rebindable");
    graph.set_space_registry(spaces.registry);

    RuntimeTensor<double> const t2("t2", {4, 8});
    graph.annotate_spaces(t2, {spaces.occ, spaces.virt});
    graph.annotate_dims(t2, {"no", "nv"});

    auto &scratch = graph.declare_zero_runtime_tensor<double>("scratch", {spaces.occ, spaces.virt}, true);

    // The payoff: the scratch got its dim symbols without anyone asking for them, so it is
    // already the reusable form rather than pinned to the capture geometry. That is what the
    // manifest reads when a bind moves the problem.
    auto const id = graph.find_tensor_id_by_ptr(&scratch);
    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"no", "nv"});
    CHECK(graph.tensor_dim_symbols(graph.find_tensor_id_by_ptr(&t2)) == std::vector<std::string>{"no", "nv"});
}

// ══════════════════════════════════════════════════════════════════════════════
// Tiled tensors: the space fixes the axis total, the partition says how it is cut
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SpaceShapedDeclare - a tiled tensor takes its partition from the space", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_canonical");
    graph.set_space_registry(spaces.registry);

    // Said ONCE, at the space. Every occupied axis in the program lands on these boundaries,
    // and retiling is this line rather than an edit at each declaration.
    graph.pin_space_tiling(spaces.occ, {2, 2});
    graph.pin_space_tiling(spaces.virt, {4, 4});

    // A partition states an extent, so pinning one pins the other.
    REQUIRE(graph.space_extent(spaces.occ).has_value());
    CHECK(*graph.space_extent(spaces.occ) == 4);
    CHECK(*graph.space_extent(spaces.virt) == 8);

    auto &T = graph.declare_zero_tiled_tensor<double>("T", {spaces.occ, spaces.virt});

    auto const id = graph.find_tensor_id_by_ptr(&T);
    REQUIRE(id != 0);
    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    // A PLAIN symbol: the space fixes the total, and the tiling is layout, not raggedness.
    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"no", "nv"});
    CHECK_FALSE(cg::is_ragged_symbol(graph.tensor_dim_symbols(id)[0]));
}

TEST_CASE("SpaceShapedDeclare - one axis overrides the canonical tiling", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_override");
    graph.set_space_registry(spaces.registry);
    graph.pin_space_tiling(spaces.occ, {2, 2});
    graph.pin_space_tiling(spaces.virt, {4, 4});

    // Same space, same total, different boundaries - which is the whole reason the partition
    // is not simply a property of the space.
    auto      &U  = graph.declare_zero_tiled_tensor<double>("U", {{spaces.occ, {1, 3}}, spaces.virt});
    auto const id = graph.find_tensor_id_by_ptr(&U);

    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"no", "nv"});
}

TEST_CASE("SpaceShapedDeclare - a partition that contradicts the space is refused", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_contradiction");
    graph.set_space_registry(spaces.registry);
    graph.pin_space_tiling(spaces.occ, {2, 2}); // occ measures 4

    CHECK_THROWS_WITH(graph.declare_zero_tiled_tensor<double>("bad", {{spaces.occ, {2, 2, 2}}}),
                      Catch::Matchers::ContainsSubstring("summing to 6") && Catch::Matchers::ContainsSubstring("measuring 4"));
}

TEST_CASE("SpaceShapedDeclare - an untiled space says so", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_unpinned");
    graph.set_space_registry(spaces.registry);

    CHECK_THROWS_WITH(graph.declare_zero_tiled_tensor<double>("T", {spaces.occ}),
                      Catch::Matchers::ContainsSubstring("how that space is cut up") &&
                          Catch::Matchers::ContainsSubstring("pin_space_tiling"));
}

TEST_CASE("SpaceShapedDeclare - a space-less tiled axis mixes in", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_mixed");
    graph.set_space_registry(spaces.registry);
    graph.pin_space_tiling(spaces.occ, {2, 2});

    auto      &T  = graph.declare_zero_tiled_tensor<double>("T", {cg::tiles({3, 3}), spaces.occ});
    auto const id = graph.find_tensor_id_by_ptr(&T);

    CHECK(graph.tensor_dim_symbols(id) == std::vector<std::string>{"", "no"});
    CHECK(graph.tensor_spaces(id)[0] == cg::SpaceId{});
    CHECK(graph.tensor_spaces(id)[1] == spaces.occ);
}

TEST_CASE("SpaceShapedDeclare - a bare tile list still selects the tile-sizes overload", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_plain");
    graph.set_space_registry(spaces.registry);

    // Unambiguous, and annotates nothing: a caller who named no space opted out.
    auto      &plain = graph.declare_zero_tiled_tensor<double>("plain", {{2, 2}, {4, 4}});
    auto const id    = graph.find_tensor_id_by_ptr(&plain);

    CHECK(graph.tensor_spaces(id).empty());
    CHECK(graph.tensor_dim_symbols(id).empty());
}

TEST_CASE("SpaceShapedDeclare - two different canonical tilings leave the space without one", "[ComputeGraph][Spaces][SpaceShape][Tiled]") {
    Spaces spaces;

    cg::Graph graph("tiled_conflict");
    graph.set_space_registry(spaces.registry);

    graph.pin_space_tiling(spaces.occ, {2, 2});
    REQUIRE(graph.space_tiling(spaces.occ).has_value());
    // Same total, so the extent stays consistent and this is not an error - but there is no
    // longer one answer to "how is occ tiled", so an axis has to bring its own.
    REQUIRE_NOTHROW(graph.pin_space_tiling(spaces.occ, {1, 3}));
    CHECK_FALSE(graph.space_tiling(spaces.occ).has_value());
    CHECK(*graph.space_extent(spaces.occ) == 4);

    CHECK_THROWS_WITH(graph.declare_zero_tiled_tensor<double>("T", {spaces.occ}), Catch::Matchers::ContainsSubstring("two different ones"));
}

TEST_CASE("SpaceShapedDeclare - a space with no dim symbol is refused rather than given one", "[ComputeGraph][Spaces][SpaceShape]") {
    cg::SpaceRegistry registry;
    auto const        grid = registry.register_space(cg::IndexSpace{.name = "grid", .scale_symbol = "g"});

    cg::Graph graph("no_symbol");
    graph.set_space_registry(registry);
    graph.pin_space_extent(grid, 12);

    // Inventing one from the space's name would make the (symbol, space) tie a tautology and
    // would have a plain symbol claim a single extent for a space that may yet be ragged.
    CHECK_THROWS_WITH(graph.declare_zero_runtime_tensor<double>("T", {grid}),
                      Catch::Matchers::ContainsSubstring("without a") && Catch::Matchers::ContainsSubstring("dim symbol"));
}

TEST_CASE("SpaceShapedDeclare - create_* takes spaces but writes no dim symbols", "[ComputeGraph][Spaces][SpaceShape][Create]") {
    Spaces spaces;

    cg::Graph graph("create");
    graph.set_space_registry(spaces.registry);
    graph.pin_space_extent(spaces.occ, 4);
    graph.pin_space_extent(spaces.virt, 8);

    auto &T = graph.create_zero_runtime_tensor<double>("T", {spaces.occ, spaces.virt});

    CHECK(T.dim(0) == 4);
    CHECK(T.dim(1) == 8);

    auto const id = graph.find_tensor_id_by_ptr(&T);
    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    // Deliberately absent: this tensor is allocated NOW, and a bind cannot resize it, so a
    // symbol promising otherwise would be a bind-time error dressed up as an annotation.
    CHECK(graph.tensor_dim_symbols(id).empty());
}
