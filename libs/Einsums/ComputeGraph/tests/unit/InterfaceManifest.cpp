//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file InterfaceManifest.cpp
/// @brief The graph's named interface: what it declares, and what binding to it enforces.
///
/// ``is_intermediate`` already partitions a graph's handles into "storage the graph made
/// for itself" and everything else. @ref einsums::compute_graph::Graph::manifest turns the
/// second half into a contract with names, types, spaces, and ownership scopes, and
/// @ref einsums::compute_graph::Graph::bind is what a caller satisfies it with.
///
/// The manifest is a TOP-LEVEL feature. A Loop body and a Conditional branch hold
/// deliberately fresh, default handles (see MetadataBoundary.cpp), so they have no
/// interface of their own and nothing here descends into one.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The entry named @p name, or a failed REQUIRE naming what the manifest does hold.
cg::ManifestEntry const &entry_named(cg::InterfaceManifest const &contract, std::string const &name) {
    cg::ManifestEntry const *entry = contract.find(name);
    if (entry == nullptr) {
        std::string known;
        for (auto const &n : contract.names()) {
            known += (known.empty() ? "" : ", ") + n;
        }
        INFO("manifest holds: [" << known << "]");
        FAIL("no manifest entry named '" << name << "'");
    }
    return *entry;
}

/// Whether the manifest holds an entry of that name at all.
bool has_entry(cg::InterfaceManifest const &contract, std::string const &name) {
    return contract.find(name) != nullptr;
}

} // namespace

// ── The interface a captured graph declares ────────────────────────────────

TEST_CASE("Manifest - a captured graph names its operands and hides its intermediates", "[ComputeGraph][Manifest]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("manifest_basic");
    auto     &tmp = graph.create_tensor<double, 2>("tmp", 4, 5);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::permute("ij <- ij", 0.0, &C, 1.0, tmp);
    }

    auto const contract = graph.manifest();

    REQUIRE(has_entry(contract, "A"));
    REQUIRE(has_entry(contract, "B"));
    REQUIRE(has_entry(contract, "C"));

    // The graph made ``tmp`` for itself. It is not part of anybody's contract.
    REQUIRE_FALSE(has_entry(contract, "tmp"));

    auto const &a = entry_named(contract, "A");
    REQUIRE(a.direction == cg::ManifestDirection::Input);
    REQUIRE(a.dtype == packed_gemm::ScalarType::Float64);
    REQUIRE(a.rank == 2);
    REQUIRE(a.dims == std::vector<std::size_t>{4, 3});
    REQUIRE(a.scope == cg::TensorOwnership::Graph);
    REQUIRE(a.spaces.empty());
    REQUIRE_FALSE(a.spaces_inferred);

    REQUIRE(entry_named(contract, "C").direction == cg::ManifestDirection::Output);

    // Inputs and outputs are the two halves of one set, and an Output is not in
    // the input half.
    REQUIRE(contract.size() == 3);
    REQUIRE_FALSE(std::ranges::any_of(contract.inputs, [](auto const &e) { return e.name == "C"; }));
}

TEST_CASE("Manifest - a slot both read and written is InOut and appears in both halves", "[ComputeGraph][Manifest]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("manifest_inout");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
        cg::scale(2.0, &C); // reads C and writes it back
    }

    auto const  contract = graph.manifest();
    auto const &c        = entry_named(contract, "C");
    REQUIRE(c.direction == cg::ManifestDirection::InOut);

    REQUIRE(std::ranges::any_of(contract.inputs, [](auto const &e) { return e.name == "C"; }));
    REQUIRE(std::ranges::any_of(contract.outputs, [](auto const &e) { return e.name == "C"; }));

    // An InOut slot is one slot, however many halves hold a copy of it.
    REQUIRE(contract.size() == 2);
    REQUIRE(contract.entries().size() == 2);
}

TEST_CASE("Manifest - a registered tensor no node touches is not part of the interface", "[ComputeGraph][Manifest]") {
    auto A         = create_random_tensor<double>("A", 4, 4);
    auto C         = create_zero_tensor<double>("C", 4, 4);
    auto spectator = create_random_tensor<double>("spectator", 4, 4);

    cg::Graph graph("manifest_spectator");
    // Registering a handle is not using it: this is how a caller attaches
    // metadata (spaces) before any op exists, and it must not manufacture a
    // contract entry out of nothing.
    (void)graph.register_operand(spectator);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    auto const contract = graph.manifest();
    REQUIRE(has_entry(contract, "A"));
    REQUIRE_FALSE(has_entry(contract, "spectator"));
}

TEST_CASE("Manifest - entries come back in a deterministic name order", "[ComputeGraph][Manifest]") {
    auto zulu  = create_random_tensor<double>("zulu", 4, 4);
    auto alpha = create_random_tensor<double>("alpha", 4, 4);
    auto mike  = create_zero_tensor<double>("mike", 4, 4);

    cg::Graph graph("manifest_order");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &mike, zulu, alpha);
    }

    // Capture order was zulu, alpha, mike; the manifest is by name regardless,
    // because a serializer and a test both need one answer and not an
    // id-numbering artifact.
    REQUIRE(graph.manifest().names() == std::vector<std::string>{"alpha", "mike", "zulu"});

    // And it is stable across calls.
    REQUIRE(graph.manifest().names() == graph.manifest().names());
}

TEST_CASE("Manifest - two interface tensors of one name is an error at manifest time", "[ComputeGraph][Manifest]") {
    auto first  = create_random_tensor<double>("A", 4, 4);
    auto second = create_random_tensor<double>("A", 4, 4);
    auto C      = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("manifest_dupes");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, first, second);
    }

    // Binding is by name. An ambiguity has to be reported where it is
    // diagnosable, not silently resolved to whichever entry a lookup reaches.
    REQUIRE_THROWS_WITH(graph.manifest(), Catch::Matchers::ContainsSubstring("both named 'A'"));
}

// ── Ownership scope ────────────────────────────────────────────────────────

TEST_CASE("Manifest - a workspace tensor is workspace-scoped in a stage's manifest", "[ComputeGraph][Manifest]") {
    cg::Workspace ws("scopes");
    auto         &eri = ws.declare_zero_tensor<double, 2>("eri", 4, 4);
    auto         &out = ws.declare_zero_tensor<double, 2>("out", 4, 4);

    // The declaring scope stamps its own canonical handle...
    REQUIRE(ws.tensor_handles().front().ownership == cg::TensorOwnership::Workspace);

    cg::Pipeline pipe("scoped");
    pipe.set_workspace(ws);
    auto &carried = pipe.declare_zero_tensor<double, 2>("carried", 4, 4);

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::einsum("ik;kj->ij", &out, eri, carried);
    }
    ws.materialize_all();

    auto const contract = stage->manifest();
    // ... and the scope survives the boundary where the stage graph builds its
    // OWN handle for the same tensor, which is the only reason a scope table exists.
    REQUIRE(entry_named(contract, "eri").scope == cg::TensorOwnership::Workspace);
    REQUIRE(entry_named(contract, "out").scope == cg::TensorOwnership::Workspace);
    REQUIRE(entry_named(contract, "carried").scope == cg::TensorOwnership::Pipeline);
}

TEST_CASE("Manifest - a workspace associated after the stage was added still reaches it", "[ComputeGraph][Manifest]") {
    cg::Workspace ws("late");
    auto         &eri = ws.declare_zero_tensor<double, 2>("eri", 4, 4);
    auto         &out = ws.declare_zero_tensor<double, 2>("out", 4, 4);

    cg::Pipeline pipe("late_pipe");

    cg::Graph *stage = nullptr;
    {
        stage = &pipe.add_stage("s");
        cg::CaptureGuard const guard(*stage);
        cg::permute("ij <- ij", 0.0, &out, 1.0, eri);
    }

    // Order of setup calls is not something a caller should have to get right.
    pipe.set_workspace(ws);
    ws.materialize_all();

    REQUIRE(stage->manifest().find("eri")->scope == cg::TensorOwnership::Workspace);
}

TEST_CASE("Manifest - a graph-declared tensor is graph-scoped and an intermediate is absent", "[ComputeGraph][Manifest]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::Graph graph("graph_scope");
    auto     &result  = graph.declare_tensor<double, 2>("result", 4, 4);
    auto     &scratch = graph.scratch<double, 2>("scratch", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &scratch, 1.0, A);
        cg::permute("ij <- ij", 0.0, &result, 1.0, scratch);
    }

    auto const contract = graph.manifest();
    REQUIRE(entry_named(contract, "result").scope == cg::TensorOwnership::Graph);
    REQUIRE_FALSE(has_entry(contract, "scratch"));
}

// ── Index spaces ───────────────────────────────────────────────────────────

TEST_CASE("Manifest - spaces round-trip as names, with the inferred flag", "[ComputeGraph][Manifest][Spaces]") {
    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::make_index_space("occ", "no", 8.0));
    auto const        virt = registry.register_space(cg::make_index_space("virt", "nv", 40.0));

    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("manifest_spaces");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.annotate_spaces(A, {occ, virt});

    // An inference is only as good as the declarations it came from, and the
    // manifest has to carry that distinction across a save.
    cg::TensorId const b_id            = graph.live_tensor_id_by_ptr(&B, {});
    graph.tensor(b_id).spaces          = {virt, occ};
    graph.tensor(b_id).spaces_inferred = true;

    auto const contract = graph.manifest();

    auto const &a = entry_named(contract, "A");
    // NAMES, not SpaceIds: an id is a handle into the registry that issued it.
    REQUIRE(a.spaces == std::vector<std::string>{"occ", "virt"});
    REQUIRE_FALSE(a.spaces_inferred);

    auto const &b = entry_named(contract, "B");
    REQUIRE(b.spaces == std::vector<std::string>{"virt", "occ"});
    REQUIRE(b.spaces_inferred);

    // Unannotated is a legal state and reads as empty, not as an error.
    REQUIRE(entry_named(contract, "C").spaces.empty());
}

TEST_CASE("Manifest - a SpaceId the registry cannot resolve is an error naming the tensor", "[ComputeGraph][Manifest][Spaces]") {
    cg::SpaceRegistry registry;
    auto const        occ = registry.register_space(cg::make_index_space("occ", "no", 8.0));

    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("manifest_bad_space");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }
    graph.annotate_spaces(A, {occ, occ});

    // A default-constructed SpaceId on a populated annotation is a corrupted
    // annotation, not an "unannotated axis". Reading it as the latter is how a
    // wrong space silently becomes no space.
    cg::TensorId const a_id      = graph.live_tensor_id_by_ptr(&A, {});
    graph.tensor(a_id).spaces[1] = cg::SpaceId{};

    REQUIRE_THROWS_WITH(graph.manifest(), Catch::Matchers::ContainsSubstring("'A'"));
}

// ── bind ───────────────────────────────────────────────────────────────────

TEST_CASE("Manifest - bind repoints the graph at fresh storage and it replays the same numbers", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("bind_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();

    auto A2 = create_zero_tensor<double>("A", 4, 3);
    auto B2 = create_zero_tensor<double>("B", 3, 5);
    auto C2 = create_zero_tensor<double>("C", 4, 5);
    A2      = A;
    B2      = B;

    graph.bind("A", A2, "B", B2, "C", C2);
    graph.execute();

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            REQUIRE(std::abs(C2(i, j) - C(i, j)) < 1e-14);
        }
    }
}

TEST_CASE("Manifest - bind rejects a name the interface does not hold, and lists what it does", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind_unknown");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    auto other = create_random_tensor<double>("other", 4, 4);
    REQUIRE_THROWS_WITH(graph.bind("nope", other), Catch::Matchers::ContainsSubstring("Known names:") &&
                                                       Catch::Matchers::ContainsSubstring("A") && Catch::Matchers::ContainsSubstring("C"));
}

TEST_CASE("Manifest - bind rejects a dtype, rank, or dim the interface does not declare", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("bind_shape");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    SECTION("dtype") {
        auto wrong = create_random_tensor<float>("A", 4, 3);
        REQUIRE_THROWS_WITH(graph.bind("A", wrong), Catch::Matchers::ContainsSubstring("dtype mismatch") &&
                                                        Catch::Matchers::ContainsSubstring("float32") &&
                                                        Catch::Matchers::ContainsSubstring("float64"));
    }
    SECTION("rank") {
        auto wrong = create_random_tensor<double>("A", 4, 3, 2);
        REQUIRE_THROWS_WITH(graph.bind("A", wrong), Catch::Matchers::ContainsSubstring("rank mismatch"));
    }
    SECTION("dim") {
        auto wrong = create_random_tensor<double>("A", 4, 7);
        REQUIRE_THROWS_WITH(graph.bind("A", wrong), Catch::Matchers::ContainsSubstring("dim 1 mismatch"));
    }
}

TEST_CASE("Manifest - bind rejects an index-space annotation the interface disagrees with", "[ComputeGraph][Manifest][Bind][Spaces]") {
    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::make_index_space("occ", "no", 8.0));
    auto const        virt = registry.register_space(cg::make_index_space("virt", "nv", 40.0));

    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind_spaces");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }
    graph.annotate_spaces(A, {occ, virt});

    auto wrong = create_random_tensor<double>("A", 4, 4);
    graph.annotate_spaces(wrong, {virt, occ});
    REQUIRE_THROWS_WITH(graph.bind("A", wrong), Catch::Matchers::ContainsSubstring("index-space mismatch"));

    // A tensor the graph has never annotated has nothing to check: annotations
    // live on handles, not on tensors, and silence is the documented behaviour.
    auto unannotated = create_random_tensor<double>("A", 4, 4);
    REQUIRE_NOTHROW(graph.bind("A", unannotated));
}

TEST_CASE("Manifest - bind may be partial and reports what is still unbound", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("bind_partial");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    REQUIRE(graph.unbound_manifest_entries() == std::vector<std::string>{"A", "B", "C"});

    auto A2 = create_random_tensor<double>("A", 4, 3);
    graph.bind("A", A2);
    REQUIRE(graph.unbound_manifest_entries() == std::vector<std::string>{"B", "C"});

    // Partial rebinding is a supported thing to do to a live graph, so bind
    // does not insist; a loader is the component that will.
    REQUIRE_NOTHROW(graph.execute());

    graph.clear_bindings();
    REQUIRE(graph.unbound_manifest_entries().size() == 3);
}

TEST_CASE("Manifest - a bound slot keeps its interface name, not the bound tensor's", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind_names");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    // rebind renames the HANDLE after the tensor it now points at, which is
    // right for a handle and wrong for a contract: a replay loop that binds the
    // same slot twice would find the name gone after the first pass.
    auto first = create_random_tensor<double>("something_else", 4, 4);
    graph.bind("A", first);
    REQUIRE(graph.manifest().names() == std::vector<std::string>{"A", "C"});

    auto second = create_random_tensor<double>("yet_another", 4, 4);
    REQUIRE_NOTHROW(graph.bind("A", second));
    graph.execute();

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            REQUIRE(std::abs(C(i, j) - second(i, j)) < 1e-14);
        }
    }
}

// ── Aliasing across a bind ─────────────────────────────────────────────────

TEST_CASE("Manifest - binding one tensor to two input slots is rejected as undeclared aliasing", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind_alias_undeclared");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const contract = graph.manifest();
    REQUIRE(entry_named(contract, "A").aliases_input == 0);
    REQUIRE(entry_named(contract, "B").aliases_input == 0);

    auto shared = create_random_tensor<double>("shared", 4, 4);
    graph.bind("A", shared);
    // Accepting this silently is exactly how a hazard edge between two slots
    // goes missing, which is the failure mode the full-cover alias bug had.
    REQUIRE_THROWS_WITH(graph.bind("B", shared), Catch::Matchers::ContainsSubstring("overlaps the storage") &&
                                                     Catch::Matchers::ContainsSubstring("declares no alias"));
}

TEST_CASE("Manifest - a declared alias between two interface slots survives a bind", "[ComputeGraph][Manifest][Bind]") {
    // A view sliced OUTSIDE a capture reaches the graph as an ordinary operand
    // and is linked to its parent by link_alias_storage. Both ends are
    // caller-supplied, so both are manifest entries and the relation between
    // them is declarable.
    auto parent = create_random_tensor<double>("parent", 4, 4);
    auto slice  = parent(AllT{}, Range{0, 2});
    auto out    = create_zero_tensor<double>("out", 4, 2);

    cg::Graph graph("bind_alias_declared");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, slice);
        cg::scale(1.0, &parent);
    }

    auto const  contract   = graph.manifest();
    auto const &slice_name = slice.name();
    REQUIRE(has_entry(contract, slice_name));
    auto const &slice_entry  = entry_named(contract, slice_name);
    auto const &parent_entry = entry_named(contract, "parent");
    REQUIRE(slice_entry.aliases_input == parent_entry.id);

    auto parent2 = create_random_tensor<double>("parent", 4, 4);
    auto slice2  = parent2(AllT{}, Range{0, 2});
    // The two share a base address, and the manifest says they are allowed to.
    REQUIRE(static_cast<void const *>(slice2.data()) == static_cast<void const *>(parent2.data()));
    REQUIRE_NOTHROW(graph.bind("parent", parent2));
    REQUIRE_NOTHROW(graph.bind(slice_name, slice2));

    // Span OVERLAP, not base-address identity, is what the check now tests; two
    // overlapping slices with different bases are caught too. See
    // AliasStructural.cpp for that case and for the link the acceptance installs.
}

// ── Control flow: the manifest stops at the body boundary ──────────────────

TEST_CASE("Manifest - a loop body's own intermediates do not leak into the parent's interface", "[ComputeGraph][Manifest]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("manifest_loop");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    auto &body       = graph.add_loop("halve", 2, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{1}));
    auto &body_local = body.create_tensor<double, 2>("body_local", 4, 4);
    {
        cg::CaptureGuard const guard(body);
        cg::permute("ij <- ij", 0.0, &body_local, 1.0, C);
        cg::scale(0.5, &C);
    }

    auto const contract = graph.manifest();

    // The parent hands C to the body, so C stays part of the parent's contract...
    REQUIRE(has_entry(contract, "C"));
    REQUIRE(entry_named(contract, "C").direction == cg::ManifestDirection::InOut);

    // ... but storage the body made for itself is the body's business. The
    // handle effective-IO registers for it in the parent is a copy of the
    // body's, is_intermediate and all.
    REQUIRE_FALSE(has_entry(contract, "body_local"));

    graph.execute();
}

TEST_CASE("Manifest - a bind reaches the slots inside a loop body", "[ComputeGraph][Manifest][Bind]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind_into_body");
    // Every operation lives in the BODY, so a bind that does not reach through the boundary
    // repoints the parent's slots and changes nothing about what actually runs.
    auto &body = graph.add_loop("once", 1, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{0}));
    {
        cg::CaptureGuard const guard(body);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }
    graph.execute();

    auto A2 = create_random_tensor<double>("A", 4, 4);
    auto C2 = create_zero_tensor<double>("C", 4, 4);

    graph.bind("A", A2, "C", C2);
    graph.execute();

    // C2 holds the new problem's answer, and C still holds the old one. Before the identity
    // fix in rebind_impl this wrote the OLD C on every replay: capture ADOPTS an operand, so
    // the parent and the body hold two different stand-ins for one caller tensor, and the
    // descent compared stand-in addresses that can never be equal across the boundary.
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            REQUIRE(std::abs(C2(i, j) - A2(i, j)) < 1e-14);
            REQUIRE(std::abs(C(i, j) - A(i, j)) < 1e-14);
        }
    }
}

// ── serializability_report descends into bodies ────────────────────────────

TEST_CASE("Manifest - serializability_report reports a blocker inside a loop body", "[ComputeGraph][ExecutorBuilder]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("report_recursive");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    // A data-shaped condition, so the Loop node itself reports clean. Under the
    // flat report that made the body unreachable AND unreported, which is the
    // thing this recursion exists to fix.
    auto &body = graph.add_loop("inner", 1, cg::PredExpr::always(false));
    {
        cg::CaptureGuard const guard(body);
        cg::element_transform(&C, [](double v) { return v + 1.0; });
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "ElementTransform");
    REQUIRE_THAT(report.front().subgraph_path, Catch::Matchers::ContainsSubstring("loop(") && Catch::Matchers::ContainsSubstring("inner"));
}

TEST_CASE("Manifest - serializability_report reports a blocker inside a conditional branch", "[ComputeGraph][ExecutorBuilder]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("report_branch");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    auto [then_g, else_g] = graph.add_conditional("branch", cg::PredExpr::always(true));
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &C);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::element_transform(&C, [](double v) { return v * 3.0; });
    }

    auto const report = graph.serializability_report();
    REQUIRE(report.size() == 1);
    REQUIRE(report.front().kind_name == "ElementTransform");
    REQUIRE_THAT(report.front().subgraph_path, Catch::Matchers::ContainsSubstring("else("));

    // A node of the graph itself carries no path, which is what makes the field
    // readable as "where, relative to what I asked about".
    cg::Graph flat("report_flat");
    {
        cg::CaptureGuard const guard(flat);
        cg::element_transform(&C, [](double v) { return v; });
    }
    REQUIRE(flat.serializability_report().front().subgraph_path.empty());
}
