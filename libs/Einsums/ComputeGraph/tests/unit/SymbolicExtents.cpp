//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SymbolicExtents.cpp
/// @brief Dim symbols, the bind solver, and the rebind staleness they made live.
///
/// ``rebind`` rejects any dim mismatch, which is right for a same-problem pointer swap and
/// fatal for the cross-problem reuse a saved graph exists to allow. Part 3.7 of the
/// algebraic-optimizer design answers that with symbols: a dimension is either literal
/// (matched exactly, as before), a symbol every slot naming it must agree on, or ragged
/// over an index space. ``bind`` solves the symbols, re-derives the graph's own
/// intermediates from them, and refuses what it cannot honestly do.
///
/// The core case here is differential: a chain captured at one problem size, bound to
/// another, must produce exactly what capturing it at the second size produces. Everything
/// else in the file is a refusal - the cases where accepting the bind would mean executing
/// garbage - plus the staleness fixes that only became reachable once extents could move.

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

/// The widest dependency level, which is where a missing hazard edge shows up.
/// A missing edge is a race that reproduces probabilistically; the level it belongs to is
/// deterministic, so the schedule is what gets asserted.
size_t widest_level(cg::Graph &graph) {
    graph.topological_sort();
    size_t widest = 0;
    for (auto const &level : graph.dependencies().levels) {
        widest = std::max(widest, level.size());
    }
    return widest;
}

/// ``C = (A * B) contracted back against A``, over two einsums and one deferred scratch.
///
/// The shape under test: ``tmp``'s extents are not written anywhere, they follow from the
/// letters of the contraction that produces it, which is exactly what a bind has to
/// re-derive when ``no`` and ``nv`` change.
struct Chain {
    cg::Graph          graph;
    Tensor<double, 2> *tmp{nullptr};

    Chain(std::string name, Tensor<double, 2> const &amp, Tensor<double, 2> const &fock, Tensor<double, 2> &out, size_t no, size_t nv)
        : graph(std::move(name)) {
        tmp = &graph.scratch<double, 2>("tmp", no, nv);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ia;ab->ib", tmp, amp, fock);
            cg::einsum("ia;ja->ij", &out, *tmp, amp);
        }
        cg::PassManager pm;
        pm.add<cg::passes::Materialization>();
        graph.apply(pm);
    }
};

/// Declare the chain's operands over the symbols ``no`` and ``nv``.
void annotate_chain(cg::Graph &graph, Tensor<double, 2> const &amp, Tensor<double, 2> const &fock, Tensor<double, 2> const &out) {
    graph.annotate_dims(amp, {"no", "nv"});
    graph.annotate_dims(fock, {"nv", "nv"});
    graph.annotate_dims(out, {"no", "no"});
}

} // namespace

// ── Declaring symbols ──────────────────────────────────────────────────────

TEST_CASE("Symbolic extents - annotate_dims round-trips through the manifest", "[ComputeGraph][Manifest][Symbolic]") {
    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto fock = create_random_tensor<double>("fock", 6, 6);
    auto out  = create_zero_tensor<double>("out", 4, 4);

    Chain chain("symbols_roundtrip", amp, fock, out, 4, 6);
    annotate_chain(chain.graph, amp, fock, out);

    auto const  contract = chain.graph.manifest();
    auto const *entry    = contract.find("amp");
    REQUIRE(entry != nullptr);
    CHECK(entry->dim_symbols == std::vector<std::string>{"no", "nv"});
    CHECK(entry->dims == std::vector<std::size_t>{4, 6});

    // An unannotated slot still says so by staying empty, which is what makes "no symbols"
    // and "every axis literal" one state rather than two.
    auto      wide = create_zero_tensor<double>("wide", 4, 6);
    cg::Graph plain("symbols_absent");
    {
        cg::CaptureGuard const guard(plain);
        cg::einsum("ia;ab->ib", &wide, amp, fock);
    }
    auto const bare = plain.manifest();
    CHECK(bare.find("amp")->dim_symbols.empty());

    CHECK(chain.graph.tensor_dim_symbols(chain.graph.find_tensor_id_by_ptr(&fock)) == std::vector<std::string>{"nv", "nv"});
}

TEST_CASE("Symbolic extents - one symbol cannot name two index spaces", "[ComputeGraph][Manifest][Symbolic][Spaces]") {
    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
    auto const        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 6.0});

    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto swap = create_random_tensor<double>("swap", 6, 4);
    auto out  = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("symbol_space_conflict");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;aj->ij", &out, amp, swap);
    }

    graph.annotate_spaces(amp, {occ, virt});
    graph.annotate_dims(amp, {"no", "nv"});
    CHECK(graph.symbol_spaces().at("nv") == virt);

    // ``swap`` axis 0 really is the virtual space, so calling it ``no`` ties ``no`` to
    // ``virt`` while it already stands for ``occ``. A symbol names one extent, so it
    // cannot range over two sets.
    graph.annotate_spaces(swap, {virt, occ});
    REQUIRE_THROWS_WITH(graph.annotate_dims(swap, {"no", "nv"}), Catch::Matchers::ContainsSubstring("dim symbol 'no'") &&
                                                                     Catch::Matchers::ContainsSubstring("occ") &&
                                                                     Catch::Matchers::ContainsSubstring("virt"));

    // Reachable from the other side too: dims first, spaces second.
    cg::Graph reversed("symbol_space_conflict_reversed");
    reversed.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(reversed);
        cg::einsum("ia;aj->ij", &out, amp, swap);
    }
    reversed.annotate_dims(amp, {"no", "nv"});
    reversed.annotate_spaces(amp, {occ, virt});
    reversed.annotate_dims(swap, {"nv", "no"});
    REQUIRE_THROWS_WITH(reversed.annotate_spaces(swap, {occ, virt}), Catch::Matchers::ContainsSubstring("dim symbol 'nv'"));

    // A refused annotation leaves the handle exactly as it was.
    CHECK(reversed.tensor_spaces(reversed.find_tensor_id_by_ptr(&swap)).empty());
}

TEST_CASE("Symbolic extents - a rank mismatch and an unknown ragged space are refused", "[ComputeGraph][Manifest][Symbolic]") {
    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto fock = create_random_tensor<double>("fock", 6, 6);
    auto out  = create_zero_tensor<double>("out", 4, 6);

    cg::Graph graph("annotate_dims_refusals");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ab->ib", &out, amp, fock);
    }

    REQUIRE_THROWS_WITH(graph.annotate_dims(amp, {"no"}), Catch::Matchers::ContainsSubstring("rank-2"));
    REQUIRE_THROWS_WITH(graph.annotate_ragged_dim(amp, 1, "pno"), Catch::Matchers::ContainsSubstring("registry does not hold"));
    REQUIRE_THROWS_WITH(graph.annotate_ragged_dim(amp, 7, "pno"), Catch::Matchers::ContainsSubstring("past its rank"));
}

// ── The solver ─────────────────────────────────────────────────────────────

TEST_CASE("Symbolic extents - a bound graph computes what capturing at that size computes", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // THE case the feature exists for. One capture at (no=4, nv=6), bound to a (3, 5)
    // problem, has to be numerically identical to a capture at (3, 5) - not close, since
    // both run the same kernels over the same values in the same order.
    auto amp0  = create_random_tensor<double>("amp", 4, 6);
    auto fock0 = create_random_tensor<double>("fock", 6, 6);
    auto out0  = create_zero_tensor<double>("out", 4, 4);

    Chain chain("bind_family", amp0, fock0, out0, 4, 6);
    annotate_chain(chain.graph, amp0, fock0, out0);

    // Execute at the captured size first, so the deferred scratch really is holding
    // storage sized for the old problem when the bind arrives. Re-binding a graph that
    // has already run is the replay case, not an exotic one.
    chain.graph.execute();
    CHECK(chain.tmp->dim(0) == 4);

    auto amp  = create_random_tensor<double>("amp_small", 3, 5);
    auto fock = create_random_tensor<double>("fock_small", 5, 5);
    auto out  = create_zero_tensor<double>("out_small", 3, 3);

    REQUIRE_NOTHROW(chain.graph.bind("amp", amp, "fock", fock, "out", out));

    // The intermediate nobody declared followed the symbols: its letters tie it to ``no``
    // and ``nv``, and letter propagation is what turns that into extents.
    CHECK(chain.tmp->dim(0) == 3);
    CHECK(chain.tmp->dim(1) == 5);

    chain.graph.execute();

    auto  ref_out = create_zero_tensor<double>("ref_out", 3, 3);
    Chain reference("fresh_small", amp, fock, ref_out, 3, 5);
    reference.graph.execute();

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            INFO("element (" << i << ", " << j << ")");
            REQUIRE(out(i, j) == ref_out(i, j));
        }
    }
}

TEST_CASE("Symbolic extents - two slots disagreeing about one symbol is an error naming both", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    auto amp0  = create_random_tensor<double>("amp", 4, 6);
    auto fock0 = create_random_tensor<double>("fock", 6, 6);
    auto out0  = create_zero_tensor<double>("out", 4, 4);

    Chain chain("bind_symbol_conflict", amp0, fock0, out0, 4, 6);
    annotate_chain(chain.graph, amp0, fock0, out0);

    auto amp  = create_random_tensor<double>("amp_small", 3, 5);
    auto fock = create_random_tensor<double>("fock_wrong", 7, 7); // ``nv`` is 5, not 7
    auto out  = create_zero_tensor<double>("out_small", 3, 3);

    REQUIRE_THROWS_WITH(chain.graph.bind("amp", amp, "fock", fock, "out", out),
                        Catch::Matchers::ContainsSubstring("dim symbol 'nv'") && Catch::Matchers::ContainsSubstring("amp") &&
                            Catch::Matchers::ContainsSubstring("7") && Catch::Matchers::ContainsSubstring("5"));

    // The refusal happens before anything is repointed, so the graph still describes the
    // problem it described a moment ago.
    CHECK(chain.tmp->dim(0) == 4);
    CHECK(chain.tmp->dim(1) == 6);
}

TEST_CASE("Symbolic extents - a literal axis still has to match exactly", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    auto amp0  = create_random_tensor<double>("amp", 4, 6);
    auto fock0 = create_random_tensor<double>("fock", 6, 6);
    auto out0  = create_zero_tensor<double>("out", 4, 4);

    Chain chain("bind_literal_axis", amp0, fock0, out0, 4, 6);
    // ``amp``'s occupied axis is declared, its virtual axis deliberately is not.
    chain.graph.annotate_dims(amp0, {"no", ""});

    auto amp = create_random_tensor<double>("amp_wrong", 3, 5);
    REQUIRE_THROWS_WITH(chain.graph.bind("amp", amp),
                        Catch::Matchers::ContainsSubstring("dim 1 mismatch") && Catch::Matchers::ContainsSubstring("That axis is literal"));

    // And a slot with no annotation at all is unchanged from before symbols existed.
    auto fock = create_random_tensor<double>("fock_wrong", 5, 5);
    REQUIRE_THROWS_AS(chain.graph.bind("fock", fock), std::invalid_argument);
}

TEST_CASE("Symbolic extents - a materialized intermediate cannot be reshaped", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // ``create_tensor`` allocates eagerly and for the graph's whole lifetime, so its
    // extents are a fact about storage rather than a consequence of the symbols. Saying so
    // is the only honest answer; silently keeping the old shape would contract the wrong
    // extents and silently reallocating would invalidate every pointer a pass planned on.
    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto fock = create_random_tensor<double>("fock", 6, 6);
    auto out  = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("eager_intermediate");
    auto     &tmp = graph.create_tensor<double, 2>("tmp", 4, 6);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ab->ib", &tmp, amp, fock);
        cg::einsum("ia;ja->ij", &out, tmp, amp);
    }
    annotate_chain(graph, amp, fock, out);

    auto amp2  = create_random_tensor<double>("amp_small", 3, 5);
    auto fock2 = create_random_tensor<double>("fock_small", 5, 5);
    auto out2  = create_zero_tensor<double>("out_small", 3, 3);

    REQUIRE_THROWS_WITH(graph.bind("amp", amp2, "fock", fock2, "out", out2),
                        Catch::Matchers::ContainsSubstring("tmp") && Catch::Matchers::ContainsSubstring("already materialized") &&
                            Catch::Matchers::ContainsSubstring("deferred intermediates"));
}

TEST_CASE("Symbolic extents - an under-annotated graph fails loudly at bind", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // ``other`` indexes the same virtual axis the symbols move, but nothing says so. After
    // the solve its extents describe the previous problem, and executing would contract a
    // length-5 index against a length-6 one. The node that cannot be satisfied is named.
    auto amp   = create_random_tensor<double>("amp", 4, 6);
    auto fock  = create_random_tensor<double>("fock", 6, 6);
    auto other = create_random_tensor<double>("other", 4, 6);
    auto out   = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("under_annotated");
    auto     &tmp = graph.scratch<double, 2>("tmp", 4, 6);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ab->ib", &tmp, amp, fock);
        cg::einsum("ia;ja->ij", &out, tmp, other);
    }
    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    graph.apply(pm);

    graph.annotate_dims(amp, {"no", "nv"});
    graph.annotate_dims(fock, {"nv", "nv"});
    graph.annotate_dims(out, {"no", "no"});

    auto amp2  = create_random_tensor<double>("amp_small", 3, 5);
    auto fock2 = create_random_tensor<double>("fock_small", 5, 5);
    auto out2  = create_zero_tensor<double>("out_small", 3, 3);

    REQUIRE_THROWS_WITH(graph.bind("amp", amp2, "fock", fock2, "out", out2),
                        Catch::Matchers::ContainsSubstring("two different problems") && Catch::Matchers::ContainsSubstring("other"));
}

// ── Ragged axes ────────────────────────────────────────────────────────────

TEST_CASE("Symbolic extents - a ragged axis declares a space and takes an extent table", "[ComputeGraph][Manifest][Symbolic][Ragged]") {
    cg::SpaceRegistry registry;
    (void)registry.register_space(cg::IndexSpace{.name = "pno", .scale_symbol = "d", .typical_extent = 6.0});

    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto fock = create_random_tensor<double>("fock", 6, 6);
    auto out  = create_zero_tensor<double>("out", 4, 6);

    cg::Graph graph("ragged_table");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ab->ib", &out, amp, fock);
    }

    graph.annotate_ragged_dim(amp, 1, "pno");
    auto const  contract = graph.manifest();
    auto const *entry    = contract.find("amp");
    REQUIRE(entry != nullptr);
    REQUIRE(entry->dim_symbols.size() == 2);
    CHECK(entry->dim_symbols[0].empty());
    CHECK(entry->dim_symbols[1] == "ragged:pno");
    CHECK(cg::is_ragged_symbol(entry->dim_symbols[1]));
    CHECK(cg::ragged_symbol_space(entry->dim_symbols[1]) == "pno");
    CHECK_FALSE(cg::is_ragged_symbol(entry->dim_symbols[0]));

    graph.bind_ragged_extents("amp", 1, {3, 5, 7});
    auto const *table = graph.find_ragged_extents("amp", 1);
    REQUIRE(table != nullptr);
    CHECK(table->space == "pno");
    CHECK(table->extents == std::vector<std::size_t>{3, 5, 7});
    CHECK(graph.ragged_extent_tables().size() == 1);

    // Arity: the length is the instance count, which belongs to the space rather than to
    // one operand, so a second table over ``pno`` cannot claim a different number of them.
    graph.annotate_ragged_dim(fock, 0, "pno");
    REQUIRE_THROWS_WITH(graph.bind_ragged_extents("fock", 0, {3, 5}),
                        Catch::Matchers::ContainsSubstring("instance count is a property of the space"));
    REQUIRE_NOTHROW(graph.bind_ragged_extents("fock", 0, {2, 4, 6}));
    CHECK(graph.ragged_extent_tables().size() == 2);

    // An axis nobody declared ragged has no per-instance extents to supply.
    REQUIRE_THROWS_WITH(graph.bind_ragged_extents("amp", 0, {1, 2, 3}), Catch::Matchers::ContainsSubstring("not declared ragged"));
    REQUIRE_THROWS_WITH(graph.bind_ragged_extents("amp", 1, {}), Catch::Matchers::ContainsSubstring("names no instance"));

    graph.clear_bindings();
    CHECK(graph.ragged_extent_tables().empty());
}

TEST_CASE("Symbolic extents - a moved extent is refused on a graph whose GEMMs are already batched",
          "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // Batching is a resource decision over ONE set of shapes: it is never saved, and the
    // load path re-runs it against the new extents. Rebinding a batch whose group table
    // describes the previous problem is therefore not this layer's job, and pretending
    // otherwise would hand the kernel a group list that no longer matches its operands.
    auto amp  = create_random_tensor<double>("amp", 4, 6);
    auto fock = create_random_tensor<double>("fock", 6, 6);
    auto out  = create_zero_tensor<double>("out", 4, 6);

    auto ba = create_random_tensor<double>("ba", 3, 3);
    auto bb = create_random_tensor<double>("bb", 3, 3);
    auto bc = create_zero_tensor<double>("bc", 3, 3);

    cg::Graph graph("post_batching");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ab->ib", &out, amp, fock);
        cg::grouped_batched_gemm(1.0, std::vector<Tensor<double, 2> const *>{&ba}, std::vector<Tensor<double, 2> const *>{&bb}, 0.0,
                                 std::vector<Tensor<double, 2> *>{&bc});
    }
    REQUIRE(std::ranges::any_of(graph.nodes(), [](cg::Node const &n) { return n.kind == cg::OpKind::GroupedBatchedGemm; }));

    graph.annotate_dims(amp, {"no", "nv"});
    graph.annotate_dims(fock, {"nv", "nv"});
    graph.annotate_dims(out, {"no", "nv"});

    auto amp2  = create_random_tensor<double>("amp_small", 3, 5);
    auto fock2 = create_random_tensor<double>("fock_small", 5, 5);
    auto out2  = create_zero_tensor<double>("out_small", 3, 5);

    REQUIRE_THROWS_WITH(graph.bind("amp", amp2, "fock", fock2, "out", out2),
                        Catch::Matchers::ContainsSubstring("GroupedBatchedGemm") &&
                            Catch::Matchers::ContainsSubstring("re-run the batching passes"));

    // A bind at the SAME extents moves nothing, so a batched graph still replays.
    auto amp3  = create_random_tensor<double>("amp_same", 4, 6);
    auto fock3 = create_random_tensor<double>("fock_same", 6, 6);
    REQUIRE_NOTHROW(graph.bind("amp", amp3, "fock", fock3));
}

// ── The staleness the solver made live ─────────────────────────────────────

TEST_CASE("Symbolic extents - rebind refreshes the handle's geometry snapshots", "[ComputeGraph][Rebind][Symbolic]") {
    // ``dims``, ``strides`` and ``data_ptr`` are what the alias derivation, the manifest
    // and every extent check read. Nothing refreshed them, which was latent only while
    // extents could not change and pointer swaps rarely aliased differently.
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_snapshots");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    cg::TensorId const id = graph.find_tensor_id_by_ptr(&A1);
    REQUIRE(id != 0);
    CHECK(graph.tensor(id).data_ptr == static_cast<void const *>(A1.data()));
    auto const captured_strides = graph.tensor(id).strides;

    // Same extents, different strides: a window of a wider buffer.
    auto big = create_random_tensor<double>("big", 8, 3);
    auto A2  = big(Range{0, 4}, AllT{});
    REQUIRE(A2.stride(1) != captured_strides[1]);

    graph.rebind(id, A2);

    CHECK(graph.tensor(id).data_ptr == static_cast<void const *>(A2.data()));
    CHECK(graph.tensor(id).dims == std::vector<std::size_t>{4, 3});
    CHECK(graph.tensor(id).strides[1] == A2.stride(1));

    // The graph still computes with the new operand, which is what the snapshots are
    // supposed to be describing.
    graph.execute();
    auto ref = create_zero_tensor<double>("ref", 4, 5);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                acc += A2(i, k) * B(k, j);
            }
            REQUIRE(std::abs(C(i, j) - acc) < 1e-12);
        }
    }
}

TEST_CASE("Symbolic extents - rebind relinks the alias relation against the new address", "[ComputeGraph][Rebind][Alias][Symbolic]") {
    // The latent bug made visible. ``link_alias_storage`` keys on the registration-time
    // ``data_ptr``, so a rebind that moves a slice from one region of its parent to an
    // overlapping one used to leave the OLD box in place: the two writes stayed on one
    // level and the hazard edge between them was silently absent.
    constexpr size_t n      = 6;
    auto             parent = create_zero_tensor<double>("parent", n, n);
    auto             left   = parent(AllT{}, Range{0, 2});
    auto             right  = parent(AllT{}, Range{4, 6});
    auto             source = create_random_tensor<double>("source", n, 2);

    cg::Graph graph("rebind_relink");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(1.0, &parent);
        cg::axpby(1.0, source, 0.0, &left);
        cg::axpby(1.0, source, 0.0, &right);
    }

    // Disjoint windows of one buffer share no byte, so the two writes may share a level.
    CHECK(widest_level(graph) == 2);

    auto overlapping = parent(AllT{}, Range{1, 3});
    graph.rebind(right, overlapping);

    // They now share a column, so they must not.
    CHECK(widest_level(graph) == 1);
}

TEST_CASE("Symbolic extents - rebind moves analysis_version and leaves structure_version alone",
          "[ComputeGraph][Rebind][Phases][Symbolic]") {
    // The two counters mean different things on purpose: ``analysis_version`` invalidates
    // position-keyed caches, ``structure_version`` invalidates plans. A rebind changes
    // storage, never the node set, so exactly one of them moves.
    auto A1 = create_random_tensor<double>("A1", 4, 4);
    auto A2 = create_random_tensor<double>("A2", 4, 4);
    auto B  = create_random_tensor<double>("B", 4, 4);
    auto C  = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("rebind_versions");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    auto const analysis  = graph.analysis_version();
    auto const structure = graph.structure_version();

    graph.rebind(A1, A2);

    CHECK(graph.analysis_version() > analysis);
    CHECK(graph.structure_version() == structure);
}

TEST_CASE("Symbolic extents - a RUNTIME-RANK deferred intermediate resizes at bind", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // The runtime-rank declare is the whole Python surface, and it shipped without the
    // resize hook that only the static-rank declare_tensor installed. The graph was refused
    // with "its storage is already materialized", which named the wrong cause: the tensor WAS
    // deferred, the callback was missing. Asserting on the dim symbols would not have caught
    // it, because the symbols were written correctly and did nothing.
    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::IndexSpace{.name = "rr_occ", .scale_symbol = "o", .dim_symbol = "no"});
    auto const        virt = registry.register_space(cg::IndexSpace{.name = "rr_virt", .scale_symbol = "v", .dim_symbol = "nv"});

    RuntimeTensor<double> amp("amp", std::vector<std::size_t>{4, 6});
    RuntimeTensor<double> out("out", std::vector<std::size_t>{4, 4});

    cg::Graph graph("runtime_rank_resize");
    graph.set_space_registry(registry);
    graph.annotate_spaces(amp, {occ, virt});
    graph.annotate_dims(amp, {"no", "nv"});
    graph.annotate_spaces(out, {occ, occ});
    graph.annotate_dims(out, {"no", "no"});

    auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {cg::SpaceDim{occ}, cg::SpaceDim{virt}}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ia->ia", &tmp, amp, amp);
        cg::einsum("ia;ja->ij", &out, tmp, amp);
    }

    CHECK(tmp.dim(0) == 4);
    CHECK(tmp.dim(1) == 6);

    RuntimeTensor<double> amp2("amp2", std::vector<std::size_t>{3, 5});
    RuntimeTensor<double> out2("out2", std::vector<std::size_t>{3, 3});
    REQUIRE_NOTHROW(graph.bind("amp", amp2, "out", out2));

    // The behaviour, not the declaration: the intermediate followed the symbols.
    CHECK(tmp.dim(0) == 3);
    CHECK(tmp.dim(1) == 5);

    // Deferred scratch needs its Materialize node before the graph can run, which is the
    // resource phase's job and deliberately not bind's.
    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    graph.apply(pm);
    REQUIRE_NOTHROW(graph.execute());
}

TEST_CASE("Symbolic extents - bind_begin/add/commit is one transaction", "[ComputeGraph][Manifest][Symbolic][Bind]") {
    // The spelling a caller without variadics has to use. A dim symbol constrains ACROSS
    // slots, so binding one slot at a time solves the second against an interface the first
    // has already moved; these three calls are the variadic bind opened up.
    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::IndexSpace{.name = "tx_occ", .scale_symbol = "o", .dim_symbol = "no"});
    auto const        virt = registry.register_space(cg::IndexSpace{.name = "tx_virt", .scale_symbol = "v", .dim_symbol = "nv"});

    RuntimeTensor<double> amp("amp", std::vector<std::size_t>{4, 6});
    RuntimeTensor<double> out("out", std::vector<std::size_t>{4, 4});

    cg::Graph graph("transaction");
    graph.set_space_registry(registry);
    graph.annotate_spaces(amp, {occ, virt});
    graph.annotate_dims(amp, {"no", "nv"});
    graph.annotate_spaces(out, {occ, occ});
    graph.annotate_dims(out, {"no", "no"});
    auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {cg::SpaceDim{occ}, cg::SpaceDim{virt}}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ia->ia", &tmp, amp, amp);
        cg::einsum("ia;ja->ij", &out, tmp, amp);
    }

    RuntimeTensor<double> amp2("amp2", std::vector<std::size_t>{3, 5});
    RuntimeTensor<double> out2("out2", std::vector<std::size_t>{3, 3});

    graph.bind_begin();
    graph.bind_add("amp", amp2);
    graph.bind_add("out", out2);
    REQUIRE_NOTHROW(graph.bind_commit());

    CHECK(tmp.dim(0) == 3);
    CHECK(tmp.dim(1) == 5);

    // Binding the SAME pair one slot at a time is what the transaction exists to avoid: the
    // second slot is reconciled against an interface the first already moved.
    cg::Graph other("one_at_a_time");
    other.set_space_registry(registry);
    RuntimeTensor<double> amp3("amp3", std::vector<std::size_t>{4, 6});
    RuntimeTensor<double> out3("out3", std::vector<std::size_t>{4, 4});
    other.annotate_spaces(amp3, {occ, virt});
    other.annotate_dims(amp3, {"no", "nv"});
    other.annotate_spaces(out3, {occ, occ});
    other.annotate_dims(out3, {"no", "no"});
    auto &tmp3 = other.declare_zero_runtime_tensor<double>("tmp", {cg::SpaceDim{occ}, cg::SpaceDim{virt}}, true);
    {
        cg::CaptureGuard const guard(other);
        cg::einsum("ia;ia->ia", &tmp3, amp3, amp3);
        cg::einsum("ia;ja->ij", &out3, tmp3, amp3);
    }
    RuntimeTensor<double> amp4("amp4", std::vector<std::size_t>{3, 5});
    CHECK_THROWS(other.bind("amp", amp4));
}
