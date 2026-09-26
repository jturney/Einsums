//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryDetection.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A rank-3 tensor antisymmetric in axes (1,2) and nothing else.
RuntimeTensor<double> antisymmetric_in_12(size_t n) {
    RuntimeTensor<double> t("asym", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                t(std::vector<size_t>{i, j, k}) = (static_cast<double>(j) - static_cast<double>(k)) * (1.0 + static_cast<double>(i));
            }
        }
    }
    return t;
}

/// A rank-3 tensor invariant under every permutation of its three axes.
RuntimeTensor<double> fully_invariant(size_t n) {
    RuntimeTensor<double> t("inv", {n, n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                t(std::vector<size_t>{i, j, k}) = static_cast<double>(i) + static_cast<double>(j) + static_cast<double>(k);
            }
        }
    }
    return t;
}

std::shared_ptr<cg::passes::AntisymmetryDetection> detect(cg::Graph &graph) {
    auto            pass = std::make_shared<cg::passes::AntisymmetryDetection>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    return pass;
}

SymmetryDescriptor const *hint_on(cg::Graph const &graph, std::string_view name) {
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.name == name) {
            return handle.symmetry_hint.get();
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("AntisymmetryDetection - finds a bound input's within-group antisymmetry", "[ComputeGraph][AntisymmetryDetection]") {
    // P(i/jk) groups {j,k}, so what its operand needs is antisymmetry in the
    // axes carrying j and k. The source has exactly that and nothing else.
    size_t const n   = 4;
    auto         src = antisymmetric_in_12(n);

    cg::Graph graph("within_group");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &dst, 1.0, src);
    }

    auto const pass = detect(graph);
    CHECK(pass->num_probed() > 0);
    REQUIRE(pass->num_tensors() == 1);

    auto const *hint = hint_on(graph, "asym");
    REQUIRE(hint != nullptr);
    CHECK(check_symmetry(src, *hint));
    // The fact found is the one the operator asked about, not some other.
    CHECK(check_symmetry(src, SymmetryDescriptor::antisymmetric_pair(1, 2)));
}

// THE soundness test. A graph intermediate is zeros at optimize time, and a zero
// tensor satisfies every symmetry there is. Tagging one records a fact true of
// the buffer and false of the value the graph will compute into it, which is the
// most direct route to a wrong answer this design has.
TEST_CASE("AntisymmetryDetection - never tags a tensor the graph writes", "[ComputeGraph][AntisymmetryDetection]") {
    size_t const          n     = 3;
    auto                  typed = create_random_tensor<double>("rand", n, n, n);
    RuntimeTensor<double> src(typed);

    cg::Graph graph("no_intermediates");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &dst, 1.0, src);
    }

    // `dst` is all zeros right now and would pass every generator tested.
    REQUIRE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    REQUIRE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(1, 2)));

    detect(graph);
    CHECK(hint_on(graph, "dst") == nullptr);

    // And after execution it genuinely does not have the symmetry a zero buffer
    // appeared to, which is what the gate was protecting against. The source is
    // random here deliberately: P(i/jk) over an operand that DOES carry the
    // within-group antisymmetry produces a fully antisymmetric result, so that
    // source would have made this assertion pass for the wrong reason.
    graph.execute();
    CHECK_FALSE(check_symmetry(dst, SymmetryDescriptor::antisymmetric_pair(0, 1)));
}

// R1's conditional arm, measured rather than argued. A coset operator carries no
// antisymmetry of its own (DESIGN section 1.1), but over an operand that already
// has the within-group antisymmetry it produces a fully antisymmetric result.
// That is the precondition its well-definedness rests on, and it is the rule
// detection exists to feed: find the operand's fact, conclude the output's.
TEST_CASE("AntisymmetryDetection - a coset operator over a qualifying operand IS antisymmetric", "[ComputeGraph][AntisymmetryDetection]") {
    size_t const n = 4;

    // Antisymmetric in (1,2), which is the group P(i/jk) does not permute within.
    auto      qualifying = antisymmetric_in_12(n);
    cg::Graph good("qualifying");
    auto     &good_dst = good.create_zero_runtime_tensor<double>("good_dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(good);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &good_dst, 1.0, qualifying);
    }
    good.execute();
    CHECK(check_symmetry(good_dst, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    CHECK(check_symmetry(good_dst, SymmetryDescriptor::antisymmetric_pair(1, 2)));

    // Without the precondition, nothing.
    auto                  typed = create_random_tensor<double>("rand", n, n, n);
    RuntimeTensor<double> arbitrary(typed);
    cg::Graph             bad("arbitrary");
    auto                 &bad_dst = bad.create_zero_runtime_tensor<double>("bad_dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(bad);
        cg::permute("i,j,k <- P(i/jk) i,j,k", 0.0, &bad_dst, 1.0, arbitrary);
    }
    bad.execute();
    CHECK_FALSE(check_symmetry(bad_dst, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    CHECK_FALSE(check_symmetry(bad_dst, SymmetryDescriptor::antisymmetric_pair(1, 2)));
}

TEST_CASE("AntisymmetryDetection - finds an invariant divisor", "[ComputeGraph][AntisymmetryDetection]") {
    // An energy denominator is invariant under the permutations the residual
    // applies, which is what lets a division by it keep the numerator's
    // antisymmetry. Same shape as the operator's output, so it is addressed.
    size_t const n   = 4;
    auto         src = antisymmetric_in_12(n);
    auto         den = fully_invariant(n);

    cg::Graph graph("invariant_divisor");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/j/k) i,j,k", 0.0, &dst, 1.0, src);
        cg::direct_division(1.0, dst, den, 0.0, &dst);
    }

    auto const  pass = detect(graph);
    auto const *hint = hint_on(graph, "inv");
    REQUIRE(hint != nullptr);
    CHECK(check_symmetry(den, *hint));
    // Every generator recorded must be an invariance, since i+j+k is symmetric
    // and antisymmetric under nothing.
    for (auto const &op : hint->ops) {
        CHECK(op.sign == +1);
    }
    CHECK(pass->num_found() > 0);
}

TEST_CASE("AntisymmetryDetection - a random input gains nothing", "[ComputeGraph][AntisymmetryDetection]") {
    size_t const          n     = 4;
    auto                  typed = create_random_tensor<double>("rand", n, n, n);
    RuntimeTensor<double> src(typed);

    cg::Graph graph("random");
    auto     &dst = graph.create_zero_runtime_tensor<double>("dst", {n, n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/j/k) i,j,k", 0.0, &dst, 1.0, src);
    }

    auto const pass = detect(graph);
    CHECK(pass->num_probed() > 0); // it looked
    CHECK(pass->num_found() == 0); // and found nothing, which is correct
    CHECK(pass->num_tensors() == 0);
    CHECK(pass->explain().empty());
}

TEST_CASE("AntisymmetryDetection - a graph with no operator probes nothing", "[ComputeGraph][AntisymmetryDetection]") {
    size_t const          n     = 4;
    auto                  typed = create_random_tensor<double>("a", n, n);
    RuntimeTensor<double> A(typed);

    cg::Graph graph("plain");
    auto     &C = graph.create_zero_runtime_tensor<double>("C", {n, n}, true);
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, A);
    }
    auto const pass = detect(graph);
    CHECK(pass->num_probed() == 0);
}

TEST_CASE("AntisymmetryDetection - never reads a declared tensor that has no storage yet", "[ComputeGraph][AntisymmetryDetection]") {
    // Detection probes tensors shaped like an operator's output for the symmetry it asks about.
    // Two tensors declared on the graph but not yet allocated have that shape, and a declared shell's
    // impl can carry a sentinel base rather than null: probing its "data" read through the sentinel
    // and crashed. Only the loop body's plain contraction into the second one made it a candidate
    // that no earlier guard excluded.
    auto X = create_random_tensor<double>("X", 2, 3, 3);
    auto Y = create_random_tensor<double>("Y", 3, 8, 2);

    cg::Graph graph("detection_shell");
    auto     &T1 = graph.declare_runtime_tensor<double>("t1", {2, 3, 8, 2}, /*intermediate=*/true);
    auto     &T2 = graph.declare_runtime_tensor<double>("t2", {2, 3, 8, 2}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("a,g,h,c <- P(a/c) a,g,e ; e,h,c", &T1, X, Y);
    }
    auto &body = graph.add_loop("iteration", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("a,g,h,c <- a,g,e ; e,h,c", &T2, X, Y);
    }

    auto manager = cg::PassManager::create_default();
    REQUIRE_NOTHROW(manager.run(graph));
    REQUIRE_NOTHROW(graph.execute());
}
