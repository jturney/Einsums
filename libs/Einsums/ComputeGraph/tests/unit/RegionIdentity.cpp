//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file RegionIdentity.cpp
/// @brief Raise every region, rewrite nothing, lower, and demand the same bits.
///
/// The gate everything else in the region framework rests on, and the only thing
/// that makes the rest of it trustworthy. Nothing about a round-trip through an algebraic IR is
/// guaranteed by construction: `lower_region` rebuilds every node from the
/// expression rather than reusing what it raised, so anything the IR fails to
/// carry - a conjugation flag, a destination prefactor, an index list a pass had
/// rewritten through the live block rather than the snapshot - comes back as a
/// different number rather than as a missing field nobody notices.
///
/// Bitwise rather than close, deliberately. A lowered node runs the same kernel
/// over the same values in the same order as the node it replaced, so anything
/// short of identical is a defect and not a tolerance question. The same
/// argument the CCSD reuse gate makes for a replay.
///
/// The shapes here are the ones a hand-written file can state precisely; the
/// breadth comes from `test_fuzz_diff_region_identity_python.py`, which drives
/// the same pass over the differential corpus.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Every element of a rank-2 tensor, flattened, so two runs can be compared
/// exactly rather than approximately.
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

/// Build a graph with @p build, run it once as captured, then run a fresh copy
/// through RegionIdentity, and require the two results to agree bit for bit.
///
/// The graph is built TWICE rather than executed twice, because a graph that
/// accumulates into its output would otherwise be compared against its own
/// second iteration. Each build gets its own destination.
template <typename Build, typename Reset, typename Read>
void require_identity(Build &&build, Reset &&reset, Read &&read) {
    reset();
    cg::Graph plain("plain");
    build(plain);
    plain.execute();
    auto const expected = read();

    reset();
    cg::Graph rewritten("rewritten");
    build(rewritten);

    auto pass = std::make_shared<cg::passes::RegionIdentity>();
    pass->set_dump(true);
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(rewritten));
    REQUIRE(pass->regions_formed() >= 1);
    REQUIRE(pass->regions_rewritten() == pass->regions_formed());

    rewritten.execute();
    auto const actual = read();

    // The dump is asserted on too, because a dump nobody reads is a dump that
    // rots: an identity rewrite must render identically before and after.
    for (auto const &dump : pass->last_dumps()) {
        INFO("region " << dump.region_index << " before:\n" << dump.before << "after:\n" << dump.after);
        CHECK(dump.before == dump.after);
        CHECK_FALSE(dump.before.empty());
    }

    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        INFO("element " << i);
        CHECK(actual[i] == expected[i]); // bitwise, not Approx
    }
}

} // namespace

TEST_CASE("identity round-trip - a single contraction", "[ComputeGraph][RegionRewrite][Identity]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &C, A, B);
        },
        [&] { C.zero(); }, [&] { return flatten(C); });
}

TEST_CASE("identity round-trip - a chain through an intermediate", "[ComputeGraph][RegionRewrite][Identity]") {
    // The shape a factorization pass exists to rewrite: an intermediate written
    // by one contraction and read by the next, dissolvable by the escape rule.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    require_identity(
        [&](cg::Graph &graph) {
            auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
            {
                cg::CaptureGuard const guard(graph);
                cg::einsum("ik;kj->ij", &tmp, A, B);
                cg::einsum("ij;jl->il", &C, tmp, D);
            }
        },
        [&] { C.zero(); }, [&] { return flatten(C); });
}

TEST_CASE("identity round-trip - an accumulating contraction", "[ComputeGraph][RegionRewrite][Identity]") {
    // The destination prefactor is semantics, not decoration: a round-trip that
    // dropped it would turn an accumulation into an overwrite and lose whatever
    // the first contraction put there.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
            cg::einsum("ik;kj->ij", 1.0, &C, 2.5, A2, B2);
        },
        [&] { C.zero(); }, [&] { return flatten(C); });
}

TEST_CASE("identity round-trip - elementwise ops between contractions", "[ComputeGraph][RegionRewrite][Identity]") {
    // Scale and axpby raise as NAMED elementwise terms carrying their
    // descriptors. The round-trip has to put the prefactors back exactly, and a
    // scale whose factor came back as its real part is precisely the bug the
    // ScaleDescriptor's PrefactorScalar exists to prevent.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto X = create_random_tensor<double>("X", 4, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &C, A, B);
            cg::scale(0.5, &C);
            cg::axpby(2.0, X, 1.5, &C);
        },
        [&] { C.zero(); }, [&] { return flatten(C); });
}

TEST_CASE("identity round-trip - a complex contraction with conjugation", "[ComputeGraph][RegionRewrite][Identity]") {
    // conj_a and conj_b live in the live params block, not only in the snapshot.
    // A raise that read the snapshot would be right here and wrong after any
    // pass that rewrote the flags, so raising the live block is asserted by
    // making the flags matter at all.
    auto A = create_random_tensor<std::complex<double>>("A", 4, 3);
    auto B = create_random_tensor<std::complex<double>>("B", 3, 5);
    auto C = create_zero_tensor<std::complex<double>>("C", 4, 5);

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            cg::einsum("conj(ik);kj->ij", &C, A, B);
        },
        [&] { C.zero(); }, [&] { return flatten(C); });
}

TEST_CASE("a LAPACK node is a barrier that splits the regions around it", "[ComputeGraph][RegionRewrite][Identity]") {
    // The barrier rule, made observable. A region must not swallow a node
    // whose arithmetic the IR cannot express, and the evidence that it does not
    // is that the run BREAKS at the barrier rather than continuing past it.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto S = create_zero_tensor<double>("S", 4, 4);
    auto W = create_zero_tensor<double>("W", 4);

    cg::Graph graph("barrier");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &S, A, A); // symmetric, so syev has something valid to do
        cg::syev(&S, &W);
        cg::scale(2.0, &C);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const regions = cg::form_regions(graph, escapes);

    INFO("formed " << regions.size() << " region(s)");
    REQUIRE(regions.size() == 2);
    CHECK(regions[0].size() == 2); // the two contractions
    CHECK(regions[1].size() == 1); // the scale after the barrier
    // And no region holds the syev.
    for (auto const &region : regions) {
        for (auto const node_id : region.nodes) {
            for (auto const &node : graph.nodes()) {
                if (node.id == node_id) {
                    CHECK(cg::is_raisable(node.kind));
                }
            }
        }
    }
}

TEST_CASE("the escape rule separates a region's outputs from its temporaries", "[ComputeGraph][RegionRewrite][Identity]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    cg::Graph graph("classify");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const regions = cg::form_regions(graph, escapes);
    REQUIRE(regions.size() == 1);

    auto const &region = regions[0];
    // A and B and D are read and never written here, so they are operands.
    CHECK(region.inputs.size() == 3);
    // C is what the region produces and the caller holds it.
    REQUIRE(region.outputs.size() == 1);
    // tmp is written and read only inside, and is a graph intermediate, so a
    // rewrite may make it disappear.
    REQUIRE(region.internal.size() == 1);
    CHECK(graph.tensor(region.internal[0]).name == "tmp");
    CHECK(graph.tensor(region.outputs[0]).name == "C");
}

TEST_CASE("region formation is the same on every run", "[ComputeGraph][RegionRewrite][Identity]") {
    // A region set that varied between runs would make every downstream rewrite
    // vary with it, and the Kahn FIFO bug is the standing reminder that "some
    // valid order" and "the same valid order every time" are different
    // requirements. Checked by re-deriving, because the containers this walks
    // include unordered ones and an accidental iteration over one would show up
    // here rather than as an unreproducible benchmark.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    cg::Graph graph("determinism");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const first   = cg::form_regions(graph, escapes);
    auto const raised  = cg::raise_region(graph, first[0]);
    REQUIRE(raised.has_value());
    auto const rendering = raised->to_string();

    for (int trial = 0; trial < 8; ++trial) {
        auto const again = cg::form_regions(graph, cg::EscapeAnalysis::over(graph));
        REQUIRE(again.size() == first.size());
        for (std::size_t r = 0; r < again.size(); ++r) {
            CHECK(again[r].nodes == first[r].nodes);
            CHECK(again[r].inputs == first[r].inputs);
            CHECK(again[r].outputs == first[r].outputs);
            CHECK(again[r].internal == first[r].internal);
        }
        auto const reraised = cg::raise_region(graph, again[0]);
        REQUIRE(reraised.has_value());
        CHECK(reraised->to_string() == rendering);
    }
}

TEST_CASE("a refused lowering leaves the graph exactly as it was", "[ComputeGraph][RegionRewrite][Identity]") {
    // The framework's safety property. A client that produces something the
    // lowering cannot build must cost a rewrite and nothing else - not a
    // half-spliced node list, which is a wrong answer rather than a missed
    // optimization.
    struct BreakIt : cg::passes::RegionRewrite {
        [[nodiscard]] std::string name() const override { return "BreakIt"; }

      protected:
        bool rewrite(cg::Graph & /*graph*/, cg::Region const & /*region*/, cg::TensorExpr &expr) override {
            // A three-operand contraction: representable in the IR, and with no
            // node form to lower to, which is exactly what lower_region declines.
            for (auto &term : expr.terms) {
                if (term.kind == cg::TermKind::Contraction) {
                    term.operands.push_back(term.operands.front());
                    term.operand_indices.push_back(term.operand_indices.front());
                    term.conjugate.push_back(false);
                    return true;
                }
            }
            return false;
        }
    };

    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("refusal");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    auto const nodes_before = graph.num_nodes();

    auto            pass = std::make_shared<BreakIt>();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(graph.num_nodes() == nodes_before);

    auto const reasons = pass->skip_reasons();
    REQUIRE_FALSE(reasons.empty());
    INFO("skip reason: " << reasons[0].first);
    CHECK(reasons[0].first.find("two operands") != std::string::npos);

    // And it still computes the right thing.
    graph.execute();
    auto ref = create_zero_tensor<double>("ref", 4, 5);
    using namespace einsums::index;
    tensor_algebra::einsum(Indices{i, j}, &ref, Indices{i, k}, A, Indices{k, j}, B);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 5; ++col) {
            CHECK(C(row, col) == ref(row, col));
        }
    }
}
