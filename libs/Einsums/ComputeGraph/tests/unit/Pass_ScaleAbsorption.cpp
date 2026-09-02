//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ScaleAbsorption.cpp
/// @brief Unit tests for the ScaleAbsorption optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// Every in-tree writer of a prefactor keeps the descriptor SNAPSHOT and the shared params block
// in step -- ScaleAbsorption::apply_fold, CSE::fold_reader, ElementWiseFusion and
// Graph::update_prefactors all write both, and each says so. These cases pin what happens when
// that discipline is not followed, which is the reason the ``live_*`` accessors exist: a reader
// that goes through them is correct on the value the EXECUTOR will use, without depending on
// every present and future writer having remembered to mirror it.
//
// A hand-built node is the only way to reach the divergent state, and that is the point: nothing
// captures one, so nothing would notice a writer that stopped mirroring until a fold silently
// keyed on the stale half.
TEST_CASE("PassUtil - the destination predicates read the live prefactor, not the snapshot", "[ComputeGraph][Passes][PassUtil]") {
    SECTION("einsum: live says overwrite, snapshot says accumulate") {
        cg::Node node;
        node.kind = cg::OpKind::Einsum;

        cg::EinsumDescriptor desc;
        desc.c_prefactor  = cg::PrefactorScalar{1.0}; // the at-capture snapshot: accumulating
        desc.params       = std::make_shared<cg::EinsumParams>();
        desc.params->c_pf = cg::PrefactorScalar{0.0}; // what a replay would actually apply
        node.op_data      = std::move(desc);

        REQUIRE(cg::passes::pure_overwrite(node));
        REQUIRE_FALSE(cg::passes::reads_destination(node));
    }

    SECTION("einsum: live says accumulate, snapshot says overwrite") {
        cg::Node node;
        node.kind = cg::OpKind::Einsum;

        cg::EinsumDescriptor desc;
        desc.c_prefactor  = cg::PrefactorScalar{0.0};
        desc.params       = std::make_shared<cg::EinsumParams>();
        desc.params->c_pf = cg::PrefactorScalar{2.0};
        node.op_data      = std::move(desc);

        REQUIRE_FALSE(cg::passes::pure_overwrite(node));
        REQUIRE(cg::passes::reads_destination(node));
    }

    SECTION("permute: live wins over the snapshot in both directions") {
        cg::Node node;
        node.kind = cg::OpKind::Permute;

        cg::PermuteDescriptor desc;
        desc.beta         = std::complex<double>{1.0, 0.0}; // snapshot: accumulating
        desc.params       = std::make_shared<cg::ElementwiseParams>();
        desc.params->beta = cg::PrefactorScalar{0.0}; // live: overwriting
        node.op_data      = desc;

        REQUIRE(cg::passes::pure_overwrite(node));
        REQUIRE_FALSE(cg::passes::reads_destination(node));

        auto *live         = std::get_if<cg::PermuteDescriptor>(&node.op_data);
        live->params->beta = cg::PrefactorScalar{3.0};
        REQUIRE_FALSE(cg::passes::pure_overwrite(node));
        REQUIRE(cg::passes::reads_destination(node));
    }

    SECTION("a node with no params block still reads its snapshot") {
        cg::Node node;
        node.kind = cg::OpKind::Einsum;

        cg::EinsumDescriptor desc;
        desc.c_prefactor = cg::PrefactorScalar{0.0};
        desc.params      = nullptr; // a node some pass assembled by hand
        node.op_data     = std::move(desc);

        REQUIRE(cg::passes::pure_overwrite(node));
        REQUIRE_FALSE(cg::passes::reads_destination(node));
    }
}

TEST_CASE("ScaleAbsorption - absorbs into einsum", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(3.0, &C_ref);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("absorb_einsum");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    REQUIRE(modified);
    REQUIRE(pass.num_absorbed() == 1);
    REQUIRE(graph.num_nodes() == 1);

    // The einsum must be left untouched: the dead scale is deleted, not
    // "absorbed" into c_prefactor (CPU executors read EinsumParams live,
    // GPU dispatch reads the descriptor - editing it would desync them).
    auto &surviving = graph.nodes()[0];
    REQUIRE(surviving.kind == cg::OpKind::Einsum);
    auto *desc = std::get_if<cg::EinsumDescriptor>(&surviving.op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(cg::as<double>(desc->c_prefactor) == 0.0);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - absorbs into permute", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 4, 6);
    auto C = create_random_tensor<double>("C", 6, 4);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(5.0, &C_ref);
    tensor_algebra::permute(0.0, Indices{j, i}, &C_ref, 1.0, Indices{i, j}, A);

    cg::Graph graph("absorb_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(5.0, &C);
        cg::permute("ji <- ij", 0.0, &C, 1.0, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    REQUIRE(modified);
    REQUIRE(pass.num_absorbed() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - no fold into an accumulating permute", "[ComputeGraph][Passes]") {
    // An accumulating consumer folds only when it exposes live shared params.
    // permute bakes its prefactors into the executor closure, so a scale
    // feeding an accumulating permute has nowhere to go and must be kept.
    // (The einsum and axpby forms of the same shape DO fold; see the
    // accumulator tests below.)
    auto A = create_random_tensor<double>("A", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);

    auto C_ref = Tensor<double, 2>(C);
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            C_ref(ii, jj) = 2.0 * C(ii, jj) + A(jj, ii);

    cg::Graph graph("no_absorb_accum_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
        cg::permute("ji <- ij", 1.0, &C, 1.0, A); // accumulates into C
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    REQUIRE_FALSE(modified);

    graph.execute();
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
}

TEST_CASE("ScaleAbsorption - no fusion when different tensors", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_random_tensor<double>("D", 3, 3);

    cg::Graph graph("different_tensors");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &D);                          // scale D
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // write C, not D
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("ScaleAbsorption - folds scale into a sole einsum operand", "[ComputeGraph][Passes]") {
    // scale(3, C) whose only reader (before C is overwritten) is an einsum using
    // C as an operand: fold 3 into that einsum's ab_prefactor (einsum is linear
    // in each operand) and drop the scale. D = 3 * (E · C).
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);
    auto E = create_random_tensor<double>("E", 4, 4);

    // Oracle computed eagerly (C is still its original value here).
    auto D_ref = create_zero_tensor<double>("Dref", 4, 5);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D_ref, 3.0, Indices{i, k}, E, Indices{k, j}, C);

    cg::Graph graph("sa_fold_operand");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, E, C); // sole reader of the scaled C
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // C overwritten (closes C's live range)
    }

    // Applied through a PassManager so the program-order validator runs; the
    // fold must declare its compensated read so the validator does not throw.
    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    REQUIRE(graph.apply(pm));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - a loop body reading the scaled tensor keeps the scale", "[ComputeGraph][Passes][ControlFlow]") {
    // A Loop node's Node::inputs do not list what its body reads (that is what
    // Graph::effective_io reconstructs), so the window scan sees no reader
    // between the scale and the later overwrite and calls the scale dead. The
    // body then reads the UNSCALED tensor.
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto B   = create_random_tensor<double>("B", 4, 4);
    auto C   = create_random_tensor<double>("C", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(3.0, &C_ref); // what the body must observe

    cg::Graph graph("sa_loop_body_reader");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
    }
    auto &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        cg::axpby(1.0, C, 0.0, &out); // reads the scaled C, invisible to the parent scan
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // C overwritten afterwards
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(out(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - two readers with no overwrite keeps the scale", "[ComputeGraph][Passes]") {
    // C is never overwritten, so its scaled value stays observable to the
    // caller after execute; the scale has to stay whatever its readers can do.
    auto C  = create_random_tensor<double>("C", 4, 5);
    auto E1 = create_random_tensor<double>("E1", 4, 4);
    auto E2 = create_random_tensor<double>("E2", 4, 4);
    auto D1 = create_zero_tensor<double>("D1", 4, 5);
    auto D2 = create_zero_tensor<double>("D2", 4, 5);

    cg::Graph graph("sa_two_readers");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D1, 1.0, E1, C);
        cg::einsum("ik;kj->ij", 0.0, &D2, 1.0, E2, C);
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);
}

TEST_CASE("ScaleAbsorption - folds into every reader of the scaled tensor", "[ComputeGraph][Passes]") {
    // Two readers before the overwrite: the factor goes into BOTH, and the
    // scale is then redundant. Folding into only one of them would be wrong,
    // so this is all-or-nothing.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_random_tensor<double>("C", 4, 5);
    auto E1 = create_random_tensor<double>("E1", 4, 4);
    auto E2 = create_random_tensor<double>("E2", 4, 4);
    auto D1 = create_zero_tensor<double>("D1", 4, 5);
    auto D2 = create_zero_tensor<double>("D2", 4, 5);

    auto D1_ref = create_zero_tensor<double>("D1ref", 4, 5);
    auto D2_ref = create_zero_tensor<double>("D2ref", 4, 5);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D1_ref, 3.0, Indices{i, k}, E1, Indices{k, j}, C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D2_ref, 3.0, Indices{i, k}, E2, Indices{k, j}, C);

    cg::Graph graph("sa_fold_all_readers");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D1, 1.0, E1, C);
        cg::einsum("ik;kj->ij", 0.0, &D2, 1.0, E2, C);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // closes C's live range
    }

    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    REQUIRE(graph.apply(pm));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(D1(ii, jj) - D1_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(D2(ii, jj) - D2_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - folds into an axpby source prefactor", "[ComputeGraph][Passes]") {
    // axpby is linear in X, so scaling X equals scaling alpha.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto X = create_random_tensor<double>("X", 4, 5);
    auto Y = create_zero_tensor<double>("Y", 4, 5);

    auto Y_ref = Tensor<double, 2>("Yref", 4, 5);
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            Y_ref(ii, jj) = 2.0 * 3.0 * X(ii, jj);

    cg::Graph graph("sa_fold_axpby_operand");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &X);
        cg::axpby(2.0, X, 0.0, &Y);
        cg::einsum("ik;kj->ij", 0.0, &X, 1.0, A, B); // closes X's live range
    }

    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    REQUIRE(graph.apply(pm));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE(std::abs(Y(ii, jj) - Y_ref(ii, jj)) < 1e-12);
}

TEST_CASE("ScaleAbsorption - folds into an accumulating einsum destination", "[ComputeGraph][Passes]") {
    // scale(a, C) then C = c_pf*C + ab_pf*A*B is exactly C = (c_pf*a)*C + ...,
    // so the factor folds into the ACCUMULATE prefactor. The accumulating
    // einsum is both the sole reader and the writer that ends C's live range.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(3.0, &C_ref);
    tensor_algebra::einsum(1.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("sa_fold_accumulator");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B); // accumulates into C
    }

    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    REQUIRE(graph.apply(pm));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
}

TEST_CASE("ScaleAbsorption - folds into an accumulating axpby destination", "[ComputeGraph][Passes]") {
    // Y = alpha*X + beta*Y with beta != 0: scale(a, Y) folds into beta.
    auto X = create_random_tensor<double>("X", 4, 5);
    auto Y = create_random_tensor<double>("Y", 4, 5);

    auto Y_ref = Tensor<double, 2>("Yref", 4, 5);
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            Y_ref(ii, jj) = 2.0 * X(ii, jj) + 5.0 * (3.0 * Y(ii, jj));

    cg::Graph graph("sa_fold_axpby_accumulator");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &Y);
        cg::axpby(2.0, X, 5.0, &Y);
    }

    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    REQUIRE(graph.apply(pm));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE(std::abs(Y(ii, jj) - Y_ref(ii, jj)) < 1e-12);
}

TEST_CASE("ScaleAbsorption - one reader that cannot take the factor blocks the fold", "[ComputeGraph][Passes]") {
    // permute bakes its prefactors into the executor closure, so it has no live
    // params to fold into. With a permute among the readers the whole scale
    // stays: a partial fold would be wrong, not merely a missed optimization.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);
    auto E = create_random_tensor<double>("E", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 5);
    auto P = create_zero_tensor<double>("P", 5, 4);

    auto C_scaled = Tensor<double, 2>(C);
    linear_algebra::scale(3.0, &C_scaled);
    auto D_ref = create_zero_tensor<double>("Dref", 4, 5);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D_ref, 1.0, Indices{i, k}, E, Indices{k, j}, C_scaled);

    cg::Graph graph("sa_unfoldable_reader");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, E, C); // could take it
        cg::permute("ji <- ij", 0.0, &P, 1.0, C);    // cannot
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // closes C's live range
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(P(jj, ii) - C_scaled(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - does not fold when the scaled value stays live", "[ComputeGraph][Passes]") {
    // C read by an einsum but NOT overwritten afterward: its scaled value is
    // still observable (in-place scale), so the scale must be kept.
    auto C = create_random_tensor<double>("C", 4, 5);
    auto E = create_random_tensor<double>("E", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 5);

    cg::Graph graph("sa_live");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, E, C); // C not overwritten after
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);
}

TEST_CASE("ScaleAbsorption - does not fold when the tensor is both einsum operands", "[ComputeGraph][Passes]") {
    // C appears as both operands, so the scale contributes a**2, not a. Keep it.
    auto C = create_random_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);
    auto F = create_random_tensor<double>("F", 4, 4);

    cg::Graph graph("sa_both_operands");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, C); // C is both operands
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, F, F); // C overwritten
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);
}

TEST_CASE("ScaleAbsorption - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("sa_empty");

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);
    CHECK(pass.num_absorbed() == 0);
}

TEST_CASE("ScaleAbsorption - single node", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("sa_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    CHECK_FALSE(modified);
}

TEST_CASE("ScaleAbsorption in Pipeline loop", "[ComputeGraph][Passes][Pipeline]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    for (int iter = 0; iter < 3; iter++) {
        linear_algebra::scale(0.5, &C_ref);
        tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    }

    cg::Pipeline pipeline("fuse_loop");
    {
        auto                  &loop_body = pipeline.add_loop("iter", 3, [](size_t iter) { return iter < 2; });
        cg::CaptureGuard const guard(loop_body);
        cg::scale(0.5, &C);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    pipeline.apply(pm);

    pipeline.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - rank-3 BatchedGemm", "[ComputeGraph][Passes][HigherRank]") {
    // The rank-3 einsum hits the strided-batched fast path and captures as
    // OpKind::BatchedGemm. With beta == 0 it overwrites C, so the preceding
    // scale is dead and gets removed.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_random_tensor<double>("C", 3, 6, 4);

    auto C_ref = Tensor<double, 3>(C);
    linear_algebra::scale(2.5, &C_ref);
    tensor_algebra::einsum(0.0, Indices{i, j, b}, &C_ref, 1.0, Indices{i, k, b}, A, Indices{k, j, b}, B);

    cg::Graph graph("sa_rank3_batched");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.5, &C);
        cg::einsum("ikb;kjb->ijb", 0.0, &C, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);
    REQUIRE(graph.nodes()[1].kind == cg::OpKind::BatchedGemm);

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    CHECK(modified);
    CHECK(pass.num_absorbed() == 1);
    CHECK(graph.num_nodes() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            for (size_t bb = 0; bb < 4; bb++) {
                REQUIRE(std::abs(C(ii, jj, bb) - C_ref(ii, jj, bb)) < 1e-12);
            }
        }
    }
}

TEST_CASE("ScaleAbsorption - rank-4 scale into permute", "[ComputeGraph][Passes][HigherRank]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5, 6);
    auto C = create_random_tensor<double>("C", 6, 5, 4, 3);

    auto C_ref = Tensor<double, 4>(C);
    linear_algebra::scale(1.5, &C_ref);
    {
        using namespace einsums::index;
        tensor_algebra::permute(0.0, Indices{l, k, j, i}, &C_ref, 1.0, Indices{i, j, k, l}, A);
    }

    cg::Graph graph("sa_rank4_permute");
    {
        using namespace einsums::index;
        cg::CaptureGuard const guard(graph);
        cg::scale(1.5, &C);
        cg::permute("lkji <- ijkl", 0.0, &C, 1.0, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    REQUIRE(modified);
    REQUIRE(pass.num_absorbed() == 1);

    graph.execute();

    for (size_t aa = 0; aa < 6; aa++) {
        for (size_t bb = 0; bb < 5; bb++) {
            for (size_t cc = 0; cc < 4; cc++) {
                for (size_t dd = 0; dd < 3; dd++) {
                    REQUIRE(std::abs(C(aa, bb, cc, dd) - C_ref(aa, bb, cc, dd)) < 1e-12);
                }
            }
        }
    }
}

// The operand fold removes a writer and compensates the reader, declaring the
// exemption via compensated_reads() so PassManager's program-order validator
// does not flag it. That list is per-APPLY state: the validator reads it once,
// after the recursive descent, and observed_writes() only inspects top-level
// nodes. Clearing it inside run() therefore let ANY subgraph -- even an empty
// loop body, since the clear precedes the early return -- discard a top-level
// exemption and make the validator throw on a legitimate fold.
TEST_CASE("ScaleAbsorption - top-level compensation survives subgraph recursion", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);
    auto D = create_random_tensor<double>("D", 4, 3);

    cg::Graph graph("sa_compensation_with_subgraph");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);                          // writer of A, folded away
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B); // sole operand read, compensated
        cg::axpby(1.0, D, 0.0, &A);                  // A overwritten: scaled value dead
    }
    // Any subgraph at all is enough to re-enter run() after the top-level fold.
    graph.add_loop("empty_body", 1, [](size_t) { return false; });

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);

    REQUIRE_NOTHROW(graph.apply(pm));
    CHECK(pass->num_absorbed() == 1);
    CHECK(pass->compensated_reads().size() == 1);
}

// ── view-mediated observers ──────────────────────────────────────────────────
//
// A store cleared whole and then accumulated into through per-slice views is
// DLPNO's residual container, and it is the shape that broke this pass. Node
// I/O lists carry TensorIds; a view of a tensor is a DIFFERENT id whose handle
// aliases the parent, so the raw id scan did not see the slice accumulations at
// all. On the merged DLPNO iteration it saw only the whole-store accumulate at
// the end, called it the sole observer, folded the zeroing factor into its beta
// and deleted the scale -- discarding every slice accumulation in between.
//
// The views are recorded BEFORE the scale in these tests on purpose. A View
// node reads its parent, so a view built inside the scale's window is an
// ordinary reader the raw scan already saw; the shape that hid was the one
// where the views (like DLPNO's, built once for the whole iteration) precede
// the scale and only their ids appear afterwards.

TEST_CASE("ScaleAbsorption - a slice view accumulating into the scaled store keeps the scale", "[ComputeGraph][Passes]") {
    constexpr size_t N = 6;
    constexpr size_t M = 3;

    auto T  = create_random_tensor<double>("T", N, N);
    auto Tn = create_random_tensor<double>("Tn", N, N);
    auto X  = create_random_tensor<double>("X", M, M);

    // The loop of eager ops the replay has to reproduce. The zeroing matters
    // most OUTSIDE the slice: nothing else ever writes those elements, so a
    // dropped scale leaves T's incoming garbage there.
    auto T_ref = Tensor<double, 2>(T);
    linear_algebra::scale(0.0, &T_ref);
    {
        auto slice_ref = T_ref(Range{0, M}, Range{0, M});
        linear_algebra::axpby(1.0, X, 1.0, &slice_ref);
    }
    linear_algebra::axpby(1.0, Tn, 1.0, &T_ref);

    cg::Graph graph("sa_view_accumulator");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &slice = cg::view<double, 2>(T, cg::ViewAxis::range(0, M), cg::ViewAxis::range(0, M));
        cg::scale(0.0, &T);             // clears the whole store
        cg::axpby(1.0, X, 1.0, &slice); // accumulates into part of it
        cg::axpby(1.0, Tn, 1.0, &T);    // the whole-store accumulate that closes the range
    }
    size_t const nodes_before = graph.num_nodes();

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);
    REQUIRE_NOTHROW(graph.apply(pm));

    CHECK(pass->num_absorbed() == 0);
    CHECK(graph.num_nodes() == nodes_before);
    bool found_scale = false;
    for (auto const &node : graph.nodes()) {
        found_scale = found_scale || node.kind == cg::OpKind::Scale;
    }
    CHECK(found_scale);

    graph.execute();
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(T(ii, jj) == T_ref(ii, jj));
        }
    }
}

TEST_CASE("ScaleAbsorption - a slice view READING the scaled store keeps the scale", "[ComputeGraph][Passes]") {
    // The dead-scale route, through the same blind spot. The only reader of the
    // scaled T is an einsum operand that is a slice VIEW of it, so the raw scan
    // found no reader at all, saw the later pure overwrite of T, and called the
    // scale dead -- leaving the einsum reading the UNSCALED slice.
    constexpr size_t N = 6;
    constexpr size_t M = 3;

    auto T = create_random_tensor<double>("T", N, N);
    auto E = create_random_tensor<double>("E", M, M);
    auto D = create_zero_tensor<double>("D", M, M);
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);

    auto T_scaled = Tensor<double, 2>(T);
    linear_algebra::scale(3.0, &T_scaled);
    auto D_ref = create_zero_tensor<double>("Dref", M, M);
    {
        auto slice_ref = T_scaled(Range{0, M}, Range{0, M});
        tensor_algebra::einsum(0.0, Indices{i, j}, &D_ref, 1.0, Indices{i, k}, E, Indices{k, j}, slice_ref);
    }

    cg::Graph graph("sa_view_reader");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &slice = cg::view<double, 2>(T, cg::ViewAxis::range(0, M), cg::ViewAxis::range(0, M));
        cg::scale(3.0, &T);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, E, slice); // reads a slice of the scaled T
        cg::einsum("ik;kj->ij", 0.0, &T, 1.0, A, B);     // closes T's live range
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);
    REQUIRE_NOTHROW(graph.apply(pm));
    CHECK(pass->num_absorbed() == 0);

    graph.execute();
    for (size_t ii = 0; ii < M; ii++) {
        for (size_t jj = 0; jj < M; jj++) {
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - a grouped axpby over slice views keeps the scale", "[ComputeGraph][Passes]") {
    // DLPNO's actual shape: the per-pair accumulations are ONE GroupedAxpby node
    // whose destinations are slice views of the cleared store.
    constexpr size_t N = 6;
    constexpr size_t M = 3;

    auto T  = create_random_tensor<double>("T", N, N);
    auto Tn = create_random_tensor<double>("Tn", N, N);
    auto X0 = create_random_tensor<double>("X0", M, M);
    auto X1 = create_random_tensor<double>("X1", M, M);

    auto T_ref = Tensor<double, 2>(T);
    linear_algebra::scale(0.0, &T_ref);
    {
        auto s0 = T_ref(Range{0, M}, Range{0, M});
        auto s1 = T_ref(Range{M, N}, Range{M, N});
        linear_algebra::axpby(1.0, X0, 1.0, &s0);
        linear_algebra::axpby(2.0, X1, 1.0, &s1);
    }
    linear_algebra::axpby(1.0, Tn, 1.0, &T_ref);

    cg::Graph graph("sa_grouped_view_accumulator");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &s0 = cg::view<double, 2>(T, cg::ViewAxis::range(0, M), cg::ViewAxis::range(0, M));
        auto                  &s1 = cg::view<double, 2>(T, cg::ViewAxis::range(M, N), cg::ViewAxis::range(M, N));
        cg::scale(0.0, &T);
        cg::grouped_axpby<Tensor<double, 2>, TensorView<double, 2>>({1.0, 2.0}, {&X0, &X1}, {1.0, 1.0}, {&s0, &s1});
        cg::axpby(1.0, Tn, 1.0, &T);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);
    REQUIRE_NOTHROW(graph.apply(pm));
    CHECK(pass->num_absorbed() == 0);

    graph.execute();
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(T(ii, jj) == T_ref(ii, jj));
        }
    }
}

TEST_CASE("ScaleAbsorption - a grouped axpby on the scaled tensor itself vetoes the fold", "[ComputeGraph][Passes]") {
    // No view here: the grouped node accumulates into the scaled tensor WHOLE,
    // so the window scan sees it by its own id. Its per-entry prefactors live in
    // a baked executor closure, which is exactly what the all-or-nothing rule
    // calls a non-taker, and fold_site's default arm has to say so for every op
    // kind it does not enumerate rather than crash or fold silently.
    auto X = create_random_tensor<double>("X", 4, 5);
    auto Y = create_random_tensor<double>("Y", 4, 5);
    auto Z = create_random_tensor<double>("Z", 4, 5);

    auto Y_ref = Tensor<double, 2>(Y);
    linear_algebra::scale(3.0, &Y_ref);
    linear_algebra::axpby(2.0, X, 1.0, &Y_ref);
    auto Z_ref = Tensor<double, 2>(Z);
    linear_algebra::axpby(1.0, Y_ref, 0.0, &Z_ref);

    cg::Graph graph("sa_grouped_whole_tensor");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &Y);
        cg::grouped_axpby<Tensor<double, 2>, Tensor<double, 2>>({2.0}, {&X}, {1.0}, {&Y});
        cg::axpby(1.0, Y, 0.0, &Z); // reads the accumulated Y, then Y is overwritten
        cg::axpby(1.0, X, 0.0, &Y); // closes Y's live range
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);
    REQUIRE_NOTHROW(graph.apply(pm));
    CHECK(pass->num_absorbed() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(Z(ii, jj) - Z_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ScaleAbsorption - the documented operand fold still fires beside unrelated views", "[ComputeGraph][Passes]") {
    // The header's own C++ example, in a graph that also contains a view of a
    // DIFFERENT tensor. The alias veto is per-buffer: a graph merely CONTAINING
    // views must not stop folding scales of tensors those views do not touch.
    auto A = create_random_tensor<double>("A", 4, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto x = create_random_tensor<double>("x", 4);
    auto y = create_random_tensor<double>("y", 5);
    auto U = create_random_tensor<double>("U", 4, 5);
    auto V = create_random_tensor<double>("V", 4, 2);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 3.0, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("sa_fold_beside_unrelated_views");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &u_slice = cg::view<double, 2>(U, cg::ViewAxis::full(), cg::ViewAxis::range(0, 2));
        cg::axpby(1.0, V, 0.0, &u_slice); // a live view of an unrelated tensor
        cg::scale(3.0, &A);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B); // C = 3*(A.B), A's sole reader
        cg::einsum("ik <- i ; k", 0.0, &A, 1.0, x, y);   // A overwritten
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::ScaleAbsorption>();
    pm.add(pass);
    REQUIRE_NOTHROW(graph.apply(pm));
    CHECK(pass->num_absorbed() == 1);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

// `Y *= s` followed by `Y += a*X` is the accumulator fold: the scale is not
// dead (the axpy reads Y), so it folds into beta rather than being deleted
// outright. This shape could not match at all while cg::axpy recorded a
// separate opaque node kind - the pass gates on Axpby and had no scalar to
// fold into - so it is a regression guard for the spelling as much as the fold.
TEST_CASE("ScaleAbsorption - absorbs into an axpy accumulation", "[ComputeGraph][Passes]") {
    auto X = create_random_tensor<double>("X", 4, 5);
    auto Y = create_random_tensor<double>("Y", 4, 5);

    auto Y_ref = Tensor<double, 2>(Y);
    linear_algebra::scale(3.0, &Y_ref);
    linear_algebra::axpy(2.0, X, &Y_ref);

    cg::Graph graph("absorb_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &Y);
        cg::axpy(2.0, X, &Y);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();

    REQUIRE(modified);
    REQUIRE(pass.num_absorbed() == 1);
    REQUIRE(graph.num_nodes() == 1);

    // The scale went into the accumulate's beta, through the LIVE params - the
    // executor reads those, so a fold written only to the descriptor snapshot
    // would leave the replay computing the unscaled result.
    auto &surviving = graph.nodes()[0];
    REQUIRE(surviving.kind == cg::OpKind::Axpby);
    auto *desc = std::get_if<cg::AxpbyDescriptor>(&surviving.op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->params != nullptr);
    REQUIRE(cg::as<double>(desc->params->beta) == 3.0);
    REQUIRE(cg::as<double>(desc->params->alpha) == 2.0);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(Y(ii, jj) - Y_ref(ii, jj)) < 1e-12);
        }
    }
}
