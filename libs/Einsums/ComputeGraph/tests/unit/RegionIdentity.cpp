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
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <complex>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

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

    // And it still computes the right thing. Within a tolerance rather than
    // bitwise: the graph's GEMM and the reference loop sum in different orders,
    // so their last bits depend on how the compiler contracts multiplies and
    // adds (clang 23 differs from 22 here). The refusal itself is checked
    // exactly above, by the node count.
    graph.execute();
    auto ref = create_zero_tensor<double>("ref", 4, 5);
    reference_einsum("ij <- ik ; kj", &ref, A, B);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 5; ++col) {
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(ref(row, col), 1e-12));
        }
    }
}

// ── The grouped family ──────────────────────────────────────────────────────
//
// A grouped node is one operation over a family of members whose extents
// differ, and it raises to a term carrying one more free letter. What these
// cases pin is the MECHANISM as well as the numbers: that a family raises at
// all, that the member letter is outermost on every operand, that the ragged
// extents on the family are the members' own, and that the lowering emits the
// grouped kind again rather than a per-member loop of ordinary nodes, which is
// the one thing the grouped family exists to avoid.

namespace {

/// A run of rank-two tensors whose extents differ member by member, which is
/// the shape a local-correlation method's per-pair work has and the reason the
/// grouped family exists.
std::vector<Tensor<double, 2>> ragged_pool(std::string const &stem, std::vector<std::size_t> const &rows,
                                           std::vector<std::size_t> const &cols) {
    std::vector<Tensor<double, 2>> out;
    out.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        out.push_back(create_random_tensor<double>(fmt::format("{}{}", stem, i), rows[i], cols[i]));
    }
    return out;
}

std::vector<Tensor<double, 2> const *> as_inputs(std::vector<Tensor<double, 2>> const &pool) {
    std::vector<Tensor<double, 2> const *> out;
    out.reserve(pool.size());
    for (auto const &tensor : pool) {
        out.push_back(&tensor);
    }
    return out;
}

std::vector<Tensor<double, 2> *> as_outputs(std::vector<Tensor<double, 2>> &pool) {
    std::vector<Tensor<double, 2> *> out;
    out.reserve(pool.size());
    for (auto &tensor : pool) {
        out.push_back(&tensor);
    }
    return out;
}

std::vector<double> flatten_pool(std::vector<Tensor<double, 2>> const &pool) {
    std::vector<double> out;
    for (auto const &tensor : pool) {
        auto const one = flatten(tensor);
        out.insert(out.end(), one.begin(), one.end());
    }
    return out;
}

/// Every region of @p graph, formed with the grouped kinds admitted.
std::vector<cg::Region> grouped_regions(cg::Graph const &graph) {
    cg::RegionOptions options;
    options.grouped = true;
    return cg::form_regions(graph, cg::EscapeAnalysis::over(graph), options);
}

/// How many nodes of @p kind the graph holds.
std::size_t count_kind(cg::Graph const &graph, cg::OpKind kind) {
    std::size_t total = 0;
    for (auto const &node : graph.nodes()) {
        total += node.kind == kind ? 1 : 0;
    }
    return total;
}

} // namespace

TEST_CASE("a grouped batch raises to a contraction over a member letter", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    std::vector<std::size_t> const rows{2, 3, 4, 3};
    std::vector<std::size_t> const links{5, 2, 3, 6};
    std::vector<std::size_t> const cols{4, 4, 2, 5};
    auto                           A = ragged_pool("A", rows, links);
    auto                           B = ragged_pool("B", links, cols);
    auto                           C = ragged_pool("C", rows, cols);

    cg::Graph graph("grouped raise");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(2.0, as_inputs(A), as_inputs(B), 0.0, as_outputs(C));
    }

    auto const regions = grouped_regions(graph);
    REQUIRE(regions.size() == 1);
    auto const raised = cg::raise_region(graph, regions[0]);
    REQUIRE(raised.has_value());

    REQUIRE(raised->families.size() == 1);
    REQUIRE(raised->statements.size() == 1);
    auto const &family    = raised->families[0];
    auto const &statement = raised->statements[0];
    CHECK(family.letter == "#m0");
    CHECK(family.members == rows.size());
    CHECK(family.kind == cg::OpKind::GroupedBatchedGemm);
    CHECK(statement.family == 0);
    CHECK(statement.targets.size() == rows.size());

    // The member letter is OUTERMOST on the output and on both operands, which
    // is the rule that makes a batched letter cheap to carry: the search reads
    // it as a letter neither operand sums, exactly as it reads an ordinary
    // batched one.
    auto const &term = raised->at(statement.value);
    REQUIRE(term.kind == cg::TermKind::Contraction);
    REQUIRE(statement.target_indices.size() == 3);
    CHECK(statement.target_indices[0].letter == family.letter);
    REQUIRE(term.operand_indices.size() == 2);
    for (auto const &operand : term.operand_indices) {
        REQUIRE(operand.size() == 3);
        CHECK(operand[0].letter == family.letter);
    }
    // Both operands are RAGGED leaves whose identity is the member list.
    for (auto const leaf : term.operands) {
        CHECK(raised->at(leaf).ragged());
        CHECK(raised->at(leaf).members.size() == rows.size());
    }

    // The per-instance extent table is the members' own extents, and the
    // typical extent the cost model reads is their mean.
    std::string const row_letter = statement.target_indices[1].letter;
    std::string const col_letter = statement.target_indices[2].letter;
    bool              saw_rows   = false;
    for (auto const &[letter, values] : family.extents) {
        if (letter == row_letter) {
            saw_rows = true;
            CHECK(values == rows);
        }
        if (letter == col_letter) {
            CHECK(values == cols);
        }
    }
    CHECK(saw_rows);
    CHECK(family.typical_extent(row_letter) == 3); // (2 + 3 + 4 + 3) / 4
}

TEST_CASE("identity round-trip - a grouped batch", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    std::vector<std::size_t> const rows{2, 3, 4, 3};
    std::vector<std::size_t> const links{5, 2, 3, 6};
    std::vector<std::size_t> const cols{4, 4, 2, 5};
    auto                           A    = ragged_pool("A", rows, links);
    auto                           B    = ragged_pool("B", links, cols);
    auto                           C    = ragged_pool("C", rows, cols);
    auto const                     seed = C;

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            // Two calls, one transposing and accumulating, so the round trip has
            // a transpose flag and a destination prefactor to carry as well as
            // the shapes.
            cg::grouped_batched_gemm(2.0, as_inputs(A), as_inputs(B), 0.0, as_outputs(C));
            cg::grouped_batched_gemm(0.5, as_inputs(A), as_inputs(B), -1.5, as_outputs(C));
        },
        [&] { C = seed; }, [&] { return flatten_pool(C); });
}

TEST_CASE("identity round-trip - the grouped scalar and element-wise kinds", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    std::vector<std::size_t> const rows{3, 2, 4};
    std::vector<std::size_t> const cols{2, 5, 3};
    auto                           X      = ragged_pool("X", rows, cols);
    auto                           Y      = ragged_pool("Y", rows, cols);
    auto                           Z      = ragged_pool("Z", rows, cols);
    auto                           P      = ragged_pool("P", cols, rows);
    auto                           r      = ragged_pool("r", {1, 1, 1}, {1, 1, 1});
    auto const                     y_seed = Y;
    auto const                     z_seed = Z;
    auto const                     p_seed = P;
    auto const                     r_seed = r;

    require_identity(
        [&](cg::Graph &graph) {
            cg::CaptureGuard const guard(graph);
            cg::grouped_axpby({2.0, -1.0, 0.5}, as_inputs(X), {0.0, 1.0, -2.0}, as_outputs(Y));
            cg::grouped_dot(as_outputs(r), as_inputs(X), as_inputs(Y));
            cg::grouped_permute("ba <- ab", as_outputs(P), as_inputs(Y), {0.0, 0.0, 1.0}, {3.0, 1.0, -1.0});
            cg::grouped_direct_product(std::vector<double>{1.5, 1.0, -0.5}, as_inputs(X), as_inputs(Y), std::vector<double>{0.0, 2.0, 1.0},
                                       as_outputs(Z));
            cg::grouped_direct_division(std::vector<double>{1.0, 2.0, 0.25}, as_inputs(X), as_inputs(Z), std::vector<double>{1.0, 0.0, 3.0},
                                        as_outputs(Z));
        },
        [&] {
            Y = y_seed;
            Z = z_seed;
            P = p_seed;
            r = r_seed;
        },
        [&] {
            auto       out = flatten_pool(Y);
            auto const z   = flatten_pool(Z);
            auto const p   = flatten_pool(P);
            auto const s   = flatten_pool(r);
            out.insert(out.end(), z.begin(), z.end());
            out.insert(out.end(), p.begin(), p.end());
            out.insert(out.end(), s.begin(), s.end());
            return out;
        });
}

TEST_CASE("a lowered grouped region is grouped nodes, not a per-member loop", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    // The rule the design states and the reason it states it: a member loop is
    // what the grouped family exists to avoid, so a rewrite must emit the
    // grouped kind or decline. Counted on the node set rather than believed
    // from the report.
    std::vector<std::size_t> const rows{2, 3, 4, 3, 2};
    std::vector<std::size_t> const links{5, 2, 3, 6, 4};
    std::vector<std::size_t> const cols{4, 4, 2, 5, 3};
    auto                           A = ragged_pool("A", rows, links);
    auto                           B = ragged_pool("B", links, cols);
    auto                           C = ragged_pool("C", rows, cols);
    auto                           D = ragged_pool("D", rows, cols);

    cg::Graph graph("grouped lowering");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.0, as_inputs(A), as_inputs(B), 0.0, as_outputs(C));
        cg::grouped_axpby({1.0, 1.0, 1.0, 1.0, 1.0}, as_inputs(C), {0.0, 0.0, 0.0, 0.0, 0.0}, as_outputs(D));
    }

    auto            pass = std::make_shared<cg::passes::RegionIdentity>();
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(graph));
    CHECK(graph.num_nodes() == 2);
    CHECK(count_kind(graph, cg::OpKind::GroupedBatchedGemm) == 1);
    CHECK(count_kind(graph, cg::OpKind::GroupedAxpby) == 1);
    CHECK(count_kind(graph, cg::OpKind::Einsum) == 0);
    CHECK(count_kind(graph, cg::OpKind::Axpby) == 0);
}

TEST_CASE("a blocked grouped batch stays a barrier", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    // Its destinations are column ranges of shared bases and the offsets that
    // say where each member lands are held by the executor, not by the node, so
    // the member list of destinations is not there to raise. Left whole.
    std::vector<std::size_t> const rows{3, 3};
    std::vector<std::size_t> const links{2, 4};
    auto                           A    = ragged_pool("A", rows, links);
    auto                           B    = ragged_pool("B", links, {2, 2});
    auto                           base = create_zero_tensor<double>("base", 3, 4);

    cg::Graph graph("blocked");
    {
        cg::CaptureGuard const                 guard(graph);
        std::vector<Tensor<double, 2> *> const bases{&base, &base};
        cg::grouped_batched_gemm_blocked(1.0, as_inputs(A), as_inputs(B), 0.0, bases, std::vector<size_t>{0, 6});
    }
    CHECK(grouped_regions(graph).empty());
}

TEST_CASE("a grouped family whose members disagree on a shape contract is declined", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    // The capture API writes one transpose pair for the whole call, so this is
    // reached by rewriting the group table the way a future producer of these
    // nodes might. A term has one index list per operand and cannot say that
    // one member transposes and another does not, so the family is left whole
    // with that reason rather than raised into an algebra that is wrong for
    // half of it.
    std::vector<std::size_t> const rows{2, 3};
    std::vector<std::size_t> const links{5, 2};
    std::vector<std::size_t> const cols{4, 4};
    auto                           A = ragged_pool("A", rows, links);
    auto                           B = ragged_pool("B", links, cols);
    auto                           C = ragged_pool("C", rows, cols);

    cg::Graph graph("disagreement");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.0, as_inputs(A), as_inputs(B), 0.0, as_outputs(C));
    }
    for (auto &node : graph.nodes()) {
        if (auto *desc = node.op_data.get_if<cg::GroupedBatchedGemmDescriptor>(); desc != nullptr) {
            REQUIRE(desc->groups.size() == 2);
            desc->groups[1].trans_a = 'T';
        }
    }

    auto const regions = grouped_regions(graph);
    REQUIRE(regions.size() == 1);
    auto const raised = cg::raise_region(graph, regions[0]);
    REQUIRE_FALSE(raised.has_value());
    INFO("decline: " << raised.error().reason << " (" << raised.error().detail << ")");
    CHECK(raised.error().reason.find("disagree on a shape contract") != std::string::npos);
}

TEST_CASE("the fused grouped kernels are barriers", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    // A sandwich dresses a slice and accumulates a symmetric product per
    // auxiliary tile, and a gather-rotate gathers a block out of a shared
    // parent before rotating it. Neither is one algebraic term, so neither has
    // a term to raise to nor a grouped kind a rewritten term could lower onto.
    CHECK_FALSE(cg::is_grouped_raisable(cg::OpKind::GroupedSandwich));
    CHECK_FALSE(cg::is_grouped_raisable(cg::OpKind::GroupedGatherRotate));
    CHECK_FALSE(cg::is_raisable(cg::OpKind::GroupedBatchedGemm));
}

// ── What prices a ragged letter ─────────────────────────────────────────────

TEST_CASE("a ragged letter is priced by its typical extent, below the scale rung", "[ComputeGraph][RegionRewrite][Identity][Grouped]") {
    // The rule, and which rung of the comparison it puts the decision on.
    //
    // A grouped family's member letter is the family's registered space, so a
    // dump names it and two families over one member count compare by scale
    // order. Every other letter it introduces is RAGGED: one extent per member
    // and no space says which, so it is an anonymous variable. One anonymous
    // variable is enough to make the typical-extent rung abstain over the whole
    // polynomial, which leaves the BOUND-extent rung, and what a client feeds
    // that rung for a ragged letter is the family's typical extent. That is
    // exactly where a bound extent for an ordinary letter already sits.
    std::vector<std::size_t> const rows{2, 3, 4, 3};
    std::vector<std::size_t> const links{5, 2, 3, 6};
    std::vector<std::size_t> const cols{4, 4, 2, 5};
    auto                           A = ragged_pool("A", rows, links);
    auto                           B = ragged_pool("B", links, cols);
    auto                           C = ragged_pool("C", rows, cols);

    cg::Graph graph("grouped cost");
    {
        cg::CaptureGuard const guard(graph);
        cg::grouped_batched_gemm(1.0, as_inputs(A), as_inputs(B), 0.0, as_outputs(C));
    }
    auto const regions = grouped_regions(graph);
    REQUIRE(regions.size() == 1);
    auto const raised = cg::raise_region(graph, regions[0]);
    REQUIRE(raised.has_value());
    REQUIRE(raised->families.size() == 1);
    auto const &family = raised->families[0];

    // The flops polynomial names the member letter's SPACE and every other
    // letter anonymously, which is what forces the rung below.
    auto const flops = raised->total_cost().flops;
    REQUIRE_FALSE(flops.is_zero());
    bool saw_space = false;
    bool saw_anon  = false;
    for (auto const &variable : flops.variables()) {
        saw_space = saw_space || (variable.is_space() && variable.space_id() == family.space);
        saw_anon  = saw_anon || variable.is_anonymous();
    }
    CHECK(saw_space);
    CHECK(saw_anon);

    // The rung, asked of the comparison rather than assumed from the shape. The
    // two candidates are this contraction and the same one with its link axis
    // one longer, which nothing but a number can separate.
    cg::ComparisonContext ctx;
    ctx.registry = &graph.space_registry();
    std::map<std::string, double> extents;
    extents[family.letter] = static_cast<double>(family.members);
    for (auto const &[letter, values] : family.extents) {
        extents[letter] = static_cast<double>(family.typical_extent(letter));
    }
    ctx.bound_extent = [&extents](cg::SymbolicVar const &variable) -> std::optional<double> {
        if (variable.is_anonymous()) {
            auto const hit = extents.find(std::string(variable.letter()));
            return hit == extents.end() ? std::nullopt : std::optional<double>{hit->second};
        }
        return std::optional<double>{4.0}; // the member count, which the family's space stands for
    };

    // The link letter is the one neither the row nor the column letter is.
    auto const &target = raised->statements[0].target_indices;
    auto const &term   = raised->at(raised->statements[0].value);
    std::string link;
    for (auto const &index : term.operand_indices[0]) {
        if (index.letter != target[0].letter && index.letter != target[1].letter && index.letter != target[2].letter) {
            link = index.letter;
        }
    }
    REQUIRE_FALSE(link.empty());

    auto const cheaper = flops;
    auto       dearer  = flops;
    dearer *= cg::SymbolicPoly::constant(1.5); // the same shape at a longer link
    auto const verdict = cg::compare_explain(cheaper, dearer, ctx);
    INFO("rung: " << cg::compare_rung_name(verdict.rung));
    CHECK(cg::compare_rung_name(verdict.rung) == "BoundExtent");
    CHECK(verdict.order == std::strong_ordering::less);

    // And the typical extent IS the mean of the members', which is the number
    // that rung was handed.
    CHECK(family.typical_extent(target[1].letter) == 3); // (2 + 3 + 4 + 3) / 4
    CHECK(family.typical_extent(link) == 4);             // (5 + 2 + 3 + 6) / 4
}

TEST_CASE("raise_region refuses a contraction whose operands differ in element type", "[ComputeGraph][TensorExpr][MixedPrecision]") {
    // The region of the determinism case above, with A in single precision. The algebra has one
    // element type and lowering rebuilds every node from its destination's, so raising the
    // mixed-precision contraction would lower it as a double GEMM reading A's floats. RegionRewrite,
    // FactorizationPass, LaplaceTransform and MultiTermFactorization all raise through here.
    auto A = create_random_tensor<float>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    cg::Graph graph("mixed");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    auto const regions = cg::form_regions(graph, cg::EscapeAnalysis::over(graph));
    REQUIRE_FALSE(regions.empty());
    auto const raised = cg::raise_region(graph, regions[0]);
    REQUIRE_FALSE(raised.has_value());
    CHECK(raised.error().reason == "a contraction's operands hold different element types");
}
