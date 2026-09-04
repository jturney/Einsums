//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DeltaElimination.cpp
/// @brief Contraction against a declared Kronecker delta is a rename, and the rename is exact.
///
/// The first real client of the region rewrite framework, so these cases carry two burdens: that
/// the arithmetic is right, and that the framework it is built on does what the identity gate
/// only showed for a no-op.
///
/// Every numeric case compares BITWISE against the un-rewritten graph. That is not strictness for
/// its own sake. ``sum_k A[i,k] * I[k,j]`` has one nonzero term, and every other product is
/// exactly ``0.0``, so the rewritten form must produce the same float rather than a nearby one.
/// A tolerance here would hide precisely the class of mistake worth catching: a dropped
/// prefactor, a transposed operand, a letter renamed the wrong way round.
///
/// The one place bitwise equality is NOT expected has its own case at the end: on non-finite
/// input the two forms genuinely differ, because the original computes ``0.0 * inf`` and the
/// rewrite never performs that multiply at all.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

template <typename T>
std::vector<T> flatten(Tensor<T, 2> const &tensor) {
    std::vector<T> out;
    out.reserve(tensor.dim(0) * tensor.dim(1));
    for (std::size_t i = 0; i < tensor.dim(0); ++i) {
        for (std::size_t j = 0; j < tensor.dim(1); ++j) {
            out.push_back(tensor(i, j));
        }
    }
    return out;
}

/// Run @p build with no passes, then again through DeltaElimination, hand both results to
/// @p compare, and require that the pass actually fired.
///
/// One derivation for the two comparisons below, because the only thing that differs between
/// them is what agreement MEANS for the rewrite under test; everything about running the graph
/// twice and proving the pass fired is common.
template <typename Build, typename Reset, typename Read, typename Compare>
std::shared_ptr<cg::passes::DeltaElimination> require_rewrite(Build &&build, Reset &&reset, Read &&read, Compare &&compare) {
    reset();
    cg::Graph plain("plain");
    build(plain);
    plain.execute();
    auto const expected = read();

    reset();
    cg::Graph rewritten("rewritten");
    build(rewritten);

    auto pass = std::make_shared<cg::passes::DeltaElimination>();
    pass->set_dump(true);
    cg::PassManager pm;
    pm.add(pass);
    bool const modified = pm.run(rewritten);
    INFO(pm.explain());
    REQUIRE(modified);

    rewritten.execute();
    auto const actual = read();

    REQUIRE(actual.size() == expected.size());
    compare(expected, actual);
    return pass;
}

/// Require the rewrite to reproduce the unrewritten result to the last bit.
///
/// The right assertion wherever the rewrite leaves the SAME kernel computing the value: index
/// substitution against a delta removes multiplications by one and does not reorder a sum, so
/// anything short of identical is a defect.
template <typename Build, typename Reset, typename Read>
std::shared_ptr<cg::passes::DeltaElimination> require_same_bits(Build &&build, Reset &&reset, Read &&read) {
    return require_rewrite(build, reset, read, [](auto const &expected, auto const &actual) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            INFO("element " << i);
            CHECK(actual[i] == expected[i]); // bitwise
        }
    });
}

/// Require the rewrite to reproduce the unrewritten result to within a few ulps.
///
/// The right assertion wherever the rewrite REPLACES the kernel, which is what happens as soon
/// as prefactors are involved: the unrewritten form is a vendor GEMM computing
/// ``beta*C + alpha*(A delta)`` and the rewritten one is a scaled permute computing
/// ``beta*C + alpha*A``. Those are two implementations of one expression, and a platform is
/// free to contract one of them into an FMA and not the other, which is a last-bit difference
/// and not a defect. Demanding bit equality across them asserted something the design's own
/// reproducibility rule does not offer - it promises the same graph on the same machine in the
/// same configuration, not two different node sets - and it passed here only because
/// Accelerate and this compiler happened to agree. GCC does not.
///
/// The bound is eight ulps of the result. What these cases exist to catch is a dropped or
/// mangled prefactor, which moves a value by a FACTOR: fifteen orders of magnitude above this
/// line, so nothing that matters fits underneath it.
template <typename Build, typename Reset, typename Read>
std::shared_ptr<cg::passes::DeltaElimination> require_within_ulps(Build &&build, Reset &&reset, Read &&read) {
    return require_rewrite(build, reset, read, [](auto const &expected, auto const &actual) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            INFO("element " << i);
            double const bound = 8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(expected[i]));
            CHECK(std::abs(actual[i] - expected[i]) <= bound);
        }
    });
}

} // namespace

TEST_CASE("a delta contraction feeding another contraction disappears", "[ComputeGraph][DeltaElimination]") {
    // The shape the pass exists for. tmp is a full GEMM against an identity matrix and its only
    // reader is the contraction below it, so both the node and the intermediate go.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto D     = create_random_tensor<double>("D", 5, 3);
    auto C     = create_zero_tensor<double>("C", 4, 3);

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &tmp, A, delta);
            cg::einsum("ij;jl->il", &C, tmp, D);
        }
    };

    auto const pass = require_same_bits(
        build, [&] { C.zero(); }, [&] { return flatten(C); });
    CHECK(pass->num_eliminated() == 1);
    CHECK(pass->num_dissolved() == 1);

    // And the node really is gone rather than merely cheaper.
    cg::Graph after("after");
    build(after);
    auto const      before_nodes = after.num_nodes();
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    pm.run(after);
    CHECK(after.num_nodes() < before_nodes);
}

TEST_CASE("a delta on the left of the contraction is eliminated too", "[ComputeGraph][DeltaElimination]") {
    // Operand order is not part of the mathematics, and a pass that only looked at slot 1 would
    // silently miss half the candidates a real equation set produces.
    auto delta = create_identity_tensor<double>("delta", 4, 4);
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto D     = create_random_tensor<double>("D", 5, 3);
    auto C     = create_zero_tensor<double>("C", 4, 3);

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &tmp, delta, A);
            cg::einsum("ij;jl->il", &C, tmp, D);
        }
    };

    auto const pass = require_same_bits(
        build, [&] { C.zero(); }, [&] { return flatten(C); });
    CHECK(pass->num_eliminated() == 1);
}

TEST_CASE("a chain of deltas collapses in one visit", "[ComputeGraph][DeltaElimination]") {
    // Dissolving one intermediate is what makes the next contraction eliminable, so a pass that
    // swept once would need to be run twice to finish. The fixpoint loop is what makes one apply
    // enough, and this is the case that would notice if it were removed.
    auto A  = create_random_tensor<double>("A", 4, 5);
    auto d1 = create_identity_tensor<double>("d1", 5, 5);
    auto d2 = create_identity_tensor<double>("d2", 5, 5);
    auto D  = create_random_tensor<double>("D", 5, 3);
    auto C  = create_zero_tensor<double>("C", 4, 3);

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(d1, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        graph.annotate_tag(d2, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        auto &t1 = graph.create_zero_runtime_tensor<double>("t1", {4, 5}, true);
        auto &t2 = graph.create_zero_runtime_tensor<double>("t2", {4, 5}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &t1, A, d1);
            cg::einsum("ik;kj->ij", &t2, t1, d2);
            cg::einsum("ij;jl->il", &C, t2, D);
        }
    };

    auto const pass = require_same_bits(
        build, [&] { C.zero(); }, [&] { return flatten(C); });
    CHECK(pass->num_eliminated() == 2);
    CHECK(pass->num_dissolved() == 2);
}

TEST_CASE("a delta contraction writing a user tensor keeps a permute", "[ComputeGraph][DeltaElimination]") {
    // The escape rule deciding the outcome. C is the caller's, so it must still be written; the
    // contraction becomes a permute rather than disappearing, and the numbers stay identical.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto C     = create_zero_tensor<double>("C", 4, 5);

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, delta);
    };

    auto const pass = require_same_bits(
        build, [&] { C.zero(); }, [&] { return flatten(C); });
    CHECK(pass->num_eliminated() == 1);
    CHECK(pass->num_dissolved() == 0); // the target escapes, so it is still written

    cg::Graph after("after");
    build(after);
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    pm.run(after);
    bool has_permute = false;
    for (auto const &node : after.nodes()) {
        has_permute = has_permute || node.kind == cg::OpKind::Permute;
    }
    CHECK(has_permute);
}

TEST_CASE("prefactors survive the rewrite exactly", "[ComputeGraph][DeltaElimination]") {
    // A dropped prefactor is the single most likely way to get this wrong, and it produces a
    // result that is plausible, wrong, and off by a constant nobody notices in a residual.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto C     = create_random_tensor<double>("C", 4, 5);
    auto seed  = create_random_tensor<double>("seed", 4, 5);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            seed(i, j) = C(i, j);
        }
    }

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        cg::CaptureGuard const guard(graph);
        // Accumulating, with a nonunit product prefactor: C = 1.5*C + 2.5*(A delta).
        cg::einsum("ik;kj->ij", 1.5, &C, 2.5, A, delta);
    };

    auto const reset = [&] {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                C(i, j) = seed(i, j);
            }
        }
    };
    auto const pass = require_within_ulps(build, reset, [&] { return flatten(C); });
    CHECK(pass->num_eliminated() == 1);

    // And the claim this case is really about, asserted where it is DISCRETE. A dropped
    // prefactor is a property of the node the pass emitted, not of the bits that come out of
    // it, so reading it off the rewritten graph says exactly what the test set out to say and
    // says it without asking two kernels to round alike. This half stays exact on every
    // platform; the numeric half above is the sanity check that the node is also WIRED up.
    reset();
    cg::Graph structural("structural");
    build(structural);
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    REQUIRE(pm.run(structural));

    bool found_permute = false;
    for (auto const &node : structural.nodes()) {
        if (node.kind != cg::OpKind::Permute) {
            continue;
        }
        auto const *desc = std::get_if<cg::PermuteDescriptor>(&node.op_data);
        REQUIRE(desc != nullptr);
        CHECK(desc->alpha == std::complex<double>{2.5, 0.0}); // the product prefactor
        CHECK(desc->beta == std::complex<double>{1.5, 0.0});  // the accumulation prefactor
        found_permute = true;
    }
    CHECK(found_permute);
}

TEST_CASE("an untagged identity matrix is left alone", "[ComputeGraph][DeltaElimination]") {
    // Recognition is DECLARED. A tensor that happens to hold an identity today is not a delta,
    // because this pass's output is what a saved graph keeps and a later bind may put something
    // else behind the same name. Reading the values would be the bug; this is the case that says
    // the pass does not.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5); // an identity, but unannounced
    auto C     = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("untagged");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, delta);
    }
    auto const before = graph.num_nodes();

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(graph.num_nodes() == before);
    CHECK(pass->num_eliminated() == 0);
}

TEST_CASE("a delta whose letters are both free is declined", "[ComputeGraph][DeltaElimination]") {
    // C[i,j] = A[i,j] delta[i,j] is a diagonal extraction, not a rename. Attempting it as one
    // would produce A verbatim and silently drop the masking.
    auto A     = create_random_tensor<double>("A", 4, 4);
    auto delta = create_identity_tensor<double>("delta", 4, 4);
    auto C     = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("diagonal");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij;ij->ij", &C, A, delta);
    }

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_eliminated() == 0);

    auto const reasons = pass->skip_reasons();
    REQUIRE_FALSE(reasons.empty());
    INFO("skip reason: " << reasons[0].first);
    CHECK(reasons[0].first.find("diagonal or a trace") != std::string::npos);
}

TEST_CASE("the rewrite is not bitwise on non-finite input, and only ever improves it", "[ComputeGraph][DeltaElimination][NonFinite]") {
    // The documented hole in the exactness claim, pinned rather than left to be discovered.
    //
    // The captured form computes sum_k A[i,k] * I[k,j]. Where A holds an infinity in a column
    // that the delta zeroes, that term is 0.0 * inf = NaN and the whole sum is poisoned. The
    // rewritten form never performs that multiply, so it returns the finite element untouched.
    //
    // The difference is one-directional: the rewrite removes a NaN the arithmetic had no reason
    // to produce and can never introduce one. That is why it is accepted rather than guarded.
    auto A     = create_zero_tensor<double>("A", 2, 2);
    auto delta = create_identity_tensor<double>("delta", 2, 2);
    auto C     = create_zero_tensor<double>("C", 2, 2);

    A(0, 0) = 1.0;
    A(0, 1) = std::numeric_limits<double>::infinity();
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    auto const build = [&](cg::Graph &graph) {
        graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, delta);
    };

    C.zero();
    cg::Graph plain("plain");
    build(plain);
    plain.execute();
    double const captured_00 = C(0, 0);

    C.zero();
    cg::Graph rewritten("rewritten");
    build(rewritten);
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    REQUIRE(pm.run(rewritten));
    rewritten.execute();
    double const rewritten_00 = C(0, 0);

    // The captured form poisons C(0,0): its sum includes A(0,1) * delta(1,0) = inf * 0.
    INFO("captured C(0,0) = " << captured_00 << ", rewritten = " << rewritten_00);
    CHECK(std::isnan(captured_00));
    // The rewrite returns the element the mathematics actually asks for.
    CHECK(rewritten_00 == 1.0);
    // Never the other way round: the rewrite does not turn a finite answer into a NaN.
    CHECK_FALSE(std::isnan(rewritten_00));
}

TEST_CASE("a tag survives a save and a load", "[ComputeGraph][DeltaElimination][SaveLoad]") {
    // A tag is a statement about the mathematics that nothing can re-derive, so it is saved
    // structure rather than an annotation a load re-computes. A file that dropped it would come
    // back eliminable-in-principle and un-eliminable in fact.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto C     = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("tagged");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity), .attributes = {{"over", "occ"}}});
    REQUIRE(graph.tensor_tag(delta).name == cg::provenance_identity);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, delta);
    }

    // delta is a tensor a caller binds, so it gets a MANIFEST entry as well as a tensor record,
    // and the file describes it through the manifest one. The first version of the writer filled
    // in only the tensor record, which meant every tensor anyone would actually want to tag
    // saved without its tag. Asserted here so a future reader knows which of the two records
    // this case is exercising.
    REQUIRE(graph.manifest().find("delta") != nullptr);

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());
    INFO(*text);
    CHECK(text->find("identity") != std::string::npos);

    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());

    cg::ProvenanceTag const *tag = nullptr;
    for (auto const &[id, handle] : loaded->tensors_map()) {
        if (handle.name == "delta") {
            tag = &handle.tag;
        }
    }
    REQUIRE(tag != nullptr);
    CHECK(tag->name == cg::provenance_identity);
    REQUIRE(tag->attribute("over").has_value());
    CHECK(*tag->attribute("over") == "occ");
}

TEST_CASE("a tag rides along a permute but not a slice", "[ComputeGraph][DeltaElimination][Provenance]") {
    // The propagation rule, and the trap in it. A transposed delta is still a delta. A SLICE of
    // one is an identity only when it is a square block on the diagonal, which nothing here can
    // know, so the tag deliberately stops at a view.
    auto delta   = create_identity_tensor<double>("delta", 4, 4);
    auto swapped = create_zero_tensor<double>("swapped", 4, 4);

    cg::Graph graph("propagate");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &swapped, delta);
    }

    auto            pass = std::make_shared<cg::passes::ProvenancePropagation>();
    cg::PassManager pm;
    pm.add(pass);
    pm.run(graph);
    CHECK(pass->num_propagated() == 1);

    bool tagged = false;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == "swapped") {
            tagged = handle.tag.name == cg::provenance_identity;
        }
    }
    CHECK(tagged);
}

TEST_CASE("propagation never overrules a declaration", "[ComputeGraph][DeltaElimination][Provenance]") {
    // A declaration is authoritative. Two ends tagged differently means one of them is wrong,
    // and picking a winner silently would bury the mistake rather than report it.
    auto delta = create_identity_tensor<double>("delta", 4, 4);
    auto other = create_zero_tensor<double>("other", 4, 4);

    cg::Graph graph("conflict");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
    graph.annotate_tag(other, cg::ProvenanceTag{.name = "fock"});
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &other, delta);
    }

    auto            pass = std::make_shared<cg::passes::ProvenancePropagation>();
    cg::PassManager pm;
    pm.add(pass);
    pm.run(graph);
    CHECK(pass->num_propagated() == 0);

    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == "other") {
            CHECK(handle.tag.name == "fock"); // the declaration stands
        }
    }

    auto const reasons = pass->skip_reasons();
    REQUIRE_FALSE(reasons.empty());
    CHECK(reasons[0].first.find("already carries a different provenance tag") != std::string::npos);
}

TEST_CASE("the structural report says what the rewrite cost, before and after", "[ComputeGraph][DeltaElimination][Report]") {
    // Two defects live here, both found by reading the report rather than by a failing test.
    //
    // The first: a client that overrode `explain()` silently dropped the framework's half of it,
    // so the pass reported its own counters and the structural section - the one that exists to
    // diagnose the pass - vanished. `explain()` is final now and clients add through `describe`.
    //
    // The second: `total_cost()` summed the whole term arena, and a rewrite leaves the term it
    // replaced in the arena because nothing renumbers indices mid-rewrite. So the before and
    // after were always equal, which is worse than reporting nothing: it is evidence that the
    // rewrite achieved nothing, printed on a rewrite that removed a contraction.
    auto A     = create_random_tensor<double>("A", 4, 5);
    auto delta = create_identity_tensor<double>("delta", 5, 5);
    auto D     = create_random_tensor<double>("D", 5, 3);
    auto C     = create_zero_tensor<double>("C", 4, 3);

    cg::Graph graph("report");
    graph.annotate_tag(delta, cg::ProvenanceTag{.name = std::string(cg::provenance_identity)});
    auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, delta);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::DeltaElimination>());
    REQUIRE(pm.run(graph));

    auto const report = pm.explain();
    INFO("report:\n" << report);

    // The framework's half survives the client's.
    CHECK(report.find("formed 1 region(s), rewrote 1") != std::string::npos);
    // The client's half is still there.
    CHECK(report.find("dissolving 1 intermediate") != std::string::npos);

    // And the cost really moved. Read the two sides of the arrow and require them different:
    // asserting the exact polynomial would pin the cost model's spelling rather than the
    // property worth pinning, which is that eliminating a contraction is visible as one.
    auto const arrow = report.find("cost ");
    REQUIRE(arrow != std::string::npos);
    auto const line  = report.substr(arrow, report.find('\n', arrow) - arrow);
    auto const split = line.find(" -> ");
    REQUIRE(split != std::string::npos);
    auto const before = line.substr(std::string("cost ").size(), split - std::string("cost ").size());
    auto const after  = line.substr(split + 4);
    INFO("before='" << before << "' after='" << after << "'");
    CHECK(before != after);
    CHECK(before.size() > after.size()); // a term went away
}

// ── The zero-block half ───────────────────────────────────────────────────────────────────────
//
// The second thing a declaration can make provable. A letter summed over two spaces that share no
// element has no term to sum, so the contraction contributes exactly nothing and only the
// destination's own prefactor is left.
//
// The numeric cases below hold the declaration TRUE in the data, by giving the block that would be
// summed a zero operand. That is the same discipline the delta cases follow with a real identity
// matrix: a rewrite justified by a declaration can only be compared against arithmetic that
// honours it, and comparing against arithmetic that does not would measure the annotation's
// falsehood rather than the rewrite.

namespace {

/// The spaces the zero-block cases draw from, in a registry each case owns.
struct DisjointSpaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ, virt, aux, pno;

    DisjointSpaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 4.0});
        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 4.0});
        pno  = registry.register_space(cg::IndexSpace{.name = "pno", .scale_symbol = "p", .typical_extent = 4.0});
        registry.declare_disjoint(occ, virt);
        registry.declare_contained(pno, virt);
    }
};

bool skip_mentions(std::vector<std::pair<std::string, std::size_t>> const &reasons, std::string_view text) {
    return std::ranges::any_of(reasons, [text](auto const &entry) { return entry.first.find(text) != std::string::npos; });
}

} // namespace

TEST_CASE("a contraction over disjoint spaces keeps only its prefactor", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    // A's second axis ranges over occ and B's first over virt, and the two share no element, so
    // the sum has no terms. B is zero, which is what makes that declaration true of the data and
    // lets the rewritten graph be compared bitwise against the captured one.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_random_tensor<double>("C", 4, 4);

    auto build = [&](cg::Graph &graph) {
        graph.set_space_registry(spaces.registry);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", 2.0, &C, 1.0, A, B);
        }
        graph.annotate_spaces(A, {spaces.aux, spaces.occ});
        graph.annotate_spaces(B, {spaces.virt, spaces.aux});
    };

    std::vector<double> const seed(C.data(), C.data() + C.size());

    cg::Graph plain("plain");
    build(plain);
    plain.execute();
    auto const expected = flatten(C);

    std::ranges::copy(seed, C.data());
    cg::Graph rewritten("rewritten");
    build(rewritten);

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(rewritten));
    CHECK(pass->num_zero_blocks() == 1);
    CHECK(pass->num_eliminated() == 0);

    rewritten.execute();
    auto const actual = flatten(C);

    // Bitwise against the captured program, which is what the claim is worth: both forms leave
    // 2*C and the rewritten one gets there without the GEMM.
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        INFO("element " << i);
        CHECK(actual[i] == expected[i]);
    }
    // And the value really is the destination's prefactor applied to what was there, not a
    // coincidence between two runs of the same wrong thing.
    for (std::size_t i = 0; i < seed.size(); ++i) {
        INFO("element " << i);
        CHECK(C.data()[i] == 2.0 * seed[i]);
    }

    // The GEMM really is gone rather than merely cheaper.
    std::size_t contractions = 0;
    for (auto const &node : rewritten.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
    }
    CHECK(contractions == 0);
}

TEST_CASE("a zero block into a buffer nothing reads leaves no node", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);

    cg::Graph graph("dead-zero");
    graph.set_space_registry(spaces.registry);
    auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 4}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
    }
    graph.annotate_spaces(A, {spaces.aux, spaces.occ});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(graph));

    CHECK(pass->num_zero_blocks() == 1);
    // The destination is overwritten, is the region's own, and nothing reads it, so there is no
    // value left to produce and no node left to emit.
    std::size_t contractions = 0, scales = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
        scales += node.kind == cg::OpKind::Scale ? 1 : 0;
    }
    CHECK(contractions == 0);
    CHECK(scales == 0);
}

TEST_CASE("an overwriting zero block assigns zero rather than multiplying by it", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_random_tensor<double>("C", 4, 4);
    // The destination holds an infinity going in. A rewrite that emitted `C *= 0` would leave a
    // NaN where the captured program leaves a clean zero, which is the direction this pass is
    // never allowed to move in.
    C(0, 0) = std::numeric_limits<double>::infinity();

    cg::Graph graph("assign-zero");
    graph.set_space_registry(spaces.registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }
    graph.annotate_spaces(A, {spaces.aux, spaces.occ});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(graph));
    CHECK(pass->num_zero_blocks() == 1);

    graph.execute();
    for (auto const value : flatten(C)) {
        CHECK(value == 0.0);
    }
}

TEST_CASE("an unrelated pair of spaces is declined", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    // occ and aux: nothing declared relates them, and Unknown is treated exactly as No.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("unrelated");
    graph.set_space_registry(spaces.registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.annotate_spaces(A, {spaces.virt, spaces.occ});
    graph.annotate_spaces(B, {spaces.aux, spaces.virt});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_zero_blocks() == 0);
    CHECK(skip_mentions(pass->skip_reasons(), "nothing declared makes a shared letter's two spaces disjoint"));
}

TEST_CASE("a letter annotated on one operand only is declined", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("half-annotated");
    graph.set_space_registry(spaces.registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    // B says nothing at all, so nothing is provable about the letter the two share.
    graph.annotate_spaces(A, {spaces.aux, spaces.occ});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_zero_blocks() == 0);
    CHECK(skip_mentions(pass->skip_reasons(), "annotated on one operand and not the other"));
}

TEST_CASE("a batched letter over disjoint spaces is declined", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    // 'b' is on both operands AND on the output, so it is walked rather than summed. Disjointness
    // then says the pairing is meaningless, which is a diagnosis, and NOT that the answer is zero.
    auto A = create_random_tensor<double>("A", 3, 4, 4);
    auto B = create_random_tensor<double>("B", 3, 4, 4);
    auto C = create_zero_tensor<double>("C", 3, 4, 4);

    cg::Graph graph("batched");
    graph.set_space_registry(spaces.registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("b,i,j <- b,i,k ; b,k,j", 0.0, &C, 1.0, A, B);
    }
    graph.annotate_spaces(A, {spaces.occ, spaces.aux, spaces.aux});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux, spaces.aux});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_zero_blocks() == 0);
    CHECK(skip_mentions(pass->skip_reasons(), "batched rather than summed"));
}

TEST_CASE("a contained pair of spaces is not disjoint", "[ComputeGraph][DeltaElimination][Spaces]") {
    DisjointSpaces spaces;

    // pno lives inside virt, which is a restriction rather than an empty intersection.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("contained");
    graph.set_space_registry(spaces.registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.annotate_spaces(A, {spaces.aux, spaces.pno});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_zero_blocks() == 0);
}

TEST_CASE("a graph with no declared disjointness never forms a region", "[ComputeGraph][DeltaElimination][Spaces]") {
    // The gate that keeps this pass free on every graph in the default pipeline. A registry with
    // spaces but no disjointness declared cannot produce a candidate, so nothing is raised.
    cg::SpaceRegistry registry;
    auto const        occ = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
    auto const        aux = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 4.0});

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("no-relations");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.annotate_spaces(A, {aux, occ});
    graph.annotate_spaces(B, {occ, aux});

    auto            pass = std::make_shared<cg::passes::DeltaElimination>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->regions_formed() == 0);
}
