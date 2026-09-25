//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryDetection.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryInference.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

std::shared_ptr<cg::passes::AntisymmetryInference> infer(cg::Graph &graph) {
    auto            pass = std::make_shared<cg::passes::AntisymmetryInference>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    return pass;
}

/// The hint the pass left on the graph's only intermediate, or nullptr.
SymmetryDescriptor const *hint_on(cg::Graph const &graph, std::string_view name) {
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.name == name) {
            return handle.symmetry_hint.get();
        }
    }
    return nullptr;
}

} // namespace

// THE test for this pass. A tag is a promise about the contents, so the way to
// know the rule is right is to run the graph and ask the data, not to compare
// the descriptor against what the rule was written to produce. The source is
// random, with no symmetry of its own, which is the only input on which a wrong
// rule and a right one differ.
TEST_CASE("AntisymmetryInference - the tagged fact holds in the data", "[ComputeGraph][AntisymmetryInference]") {
    size_t const                n         = 4;
    auto                        src_typed = create_random_tensor<double>("src", n, n, n);
    RuntimeTensor<double> const src(src_typed);

    cg::Graph graph("full_antisymmetrizer");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, /*intermediate=*/true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/j/k) i,j,k", 0.0, &dst, 1.0, src);
    }

    auto const pass = infer(graph);
    CHECK(pass->num_candidates() == 1);
    REQUIRE(pass->num_tagged() == 1);

    auto const *hint = hint_on(graph, "dst");
    REQUIRE(hint != nullptr);
    REQUIRE(hint->size() == 2); // adjacent transpositions of three axes

    graph.execute();
    REQUIRE(check_symmetry(dst, *hint));

    // And the promise is not vacuous: the source itself does not have it.
    REQUIRE_FALSE(check_symmetry(src, *hint));
}

TEST_CASE("AntisymmetryInference - P(ij)P(ab) tags both pairs", "[ComputeGraph][AntisymmetryInference]") {
    size_t const                no = 3, nv = 4;
    auto                        t2_typed = create_random_tensor<double>("t2", no, no, nv, nv);
    auto                        F_typed  = create_random_tensor<double>("F", no, no);
    RuntimeTensor<double> const t2(t2_typed);
    RuntimeTensor<double> const F(F_typed);

    cg::Graph graph("doubles");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {no, no, nv, nv}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,b ; m,j", 0.0, &dst, 1.0, t2, F);
    }

    auto const pass = infer(graph);
    REQUIRE(pass->num_tagged() == 1);
    auto const *hint = hint_on(graph, "dst");
    REQUIRE(hint != nullptr);
    REQUIRE(hint->size() == 2); // one adjacent transposition per operator

    graph.execute();
    REQUIRE(check_symmetry(dst, *hint));
}

// The case the measurement in DESIGN-permutation-operator-folding.md section 1.1
// found. A coset form carries no antisymmetry of its own, so the pass must
// DECLINE rather than tag; a rule that tagged it would be stating something the
// data does not satisfy, and the fold built on it would be wrong.
TEST_CASE("AntisymmetryInference - a coset operator is declined", "[ComputeGraph][AntisymmetryInference]") {
    size_t const                n         = 3;
    auto                        src_typed = create_random_tensor<double>("src", n, n, n);
    RuntimeTensor<double> const src(src_typed);

    cg::Graph graph("coset");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &dst, 1.0, src);
    }

    auto const pass = infer(graph);
    CHECK(pass->num_candidates() == 1);
    CHECK(pass->num_tagged() == 0);
    CHECK(hint_on(graph, "dst") == nullptr);

    // Confirm the decline is right rather than merely conservative: the output
    // really is antisymmetric under none of the three transpositions.
    graph.execute();
    CHECK_FALSE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    CHECK_FALSE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(0, 2)));
    CHECK_FALSE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(1, 2)));
}

TEST_CASE("AntisymmetryInference - an accumulating node is declined", "[ComputeGraph][AntisymmetryInference]") {
    // C = c_pf*C + P[G](...) is the old contents plus an antisymmetric part,
    // which is not antisymmetric. The gate is not conservatism; the tag would be
    // false.
    size_t const                n         = 3;
    auto                        src_typed = create_random_tensor<double>("src", n, n, n);
    RuntimeTensor<double> const src(src_typed);

    cg::Graph graph("accumulating");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/j/k) i,j,k", 1.0, &dst, 1.0, src);
    }

    auto const pass = infer(graph);
    CHECK(pass->num_candidates() == 1);
    CHECK(pass->num_tagged() == 0);
}

TEST_CASE("AntisymmetryInference - a graph with no operator is untouched", "[ComputeGraph][AntisymmetryInference]") {
    auto                        A_typed = create_random_tensor<double>("A", 4, 3);
    auto                        B_typed = create_random_tensor<double>("B", 3, 5);
    RuntimeTensor<double> const A(A_typed);
    RuntimeTensor<double> const B(B_typed);
    RuntimeTensor<double>       C("C", {4, 5});

    cg::Graph graph("plain");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }
    auto const pass = infer(graph);
    // The contraction IS examined: R3 asks of every einsum whether its operands
    // carry an antisymmetry to pass on. What matters is that nothing is
    // concluded from a graph with no symmetry anywhere in it.
    CHECK(pass->num_tagged() == 0);
    CHECK(pass->explain().empty());
}

// R3's load-bearing condition. C(..p..q..) = sum A(..p..q..) B(...) negates
// under swapping p and q only because the OTHER operand does not see them. When
// it does, it moves under the swap too and nothing cancels, so a rule that
// skipped this check would tag a tensor with an antisymmetry it does not have.
TEST_CASE("AntisymmetryInference - R3 declines when the other operand sees the letters", "[ComputeGraph][AntisymmetryInference]") {
    size_t const n = 4;

    auto const            r = create_random_tensor<double>("r", n, n, n);
    RuntimeTensor<double> t2("t2", {n, n, n});
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t m = 0; m < n; ++m) {
                t2(std::vector<size_t>{j, k, m}) = r(j, k, m) - r(k, j, m);
            }
        }
    }
    REQUIRE(check_symmetry(t2, SymmetryDescriptor::antisymmetric_pair(0, 1)));

    auto const            b_typed = create_random_tensor<double>("b", n, n, n);
    RuntimeTensor<double> shared(b_typed); // b(m,j,k): SEES j and k

    cg::Graph graph("other_operand_sees");
    auto     &Xc = graph.create_zero_runtime_tensor<double>("Xc", {n, n, n}, true);
    auto     &W  = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        // The second operand carries j and k as well, so the swap moves it too.
        cg::einsum("j,k,m <- j,k,p ; m,j,k", 0.0, &Xc, 1.0, t2, shared);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &W, 1.0, Xc);
    }

    auto            detection = std::make_shared<cg::passes::AntisymmetryDetection>();
    cg::PassManager pre;
    pre.add(detection);
    graph.apply(pre);

    auto const pass = infer(graph);
    CHECK(hint_on(graph, "Xc") == nullptr);

    // And the decline is right rather than merely cautious: the product genuinely
    // is not antisymmetric in those axes.
    graph.execute();
    CHECK_FALSE(check_symmetry(Xc, SymmetryDescriptor::antisymmetric_pair(0, 1)));
}
