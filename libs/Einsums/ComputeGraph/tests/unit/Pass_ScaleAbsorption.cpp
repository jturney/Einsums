//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ScaleAbsorption.cpp
/// @brief Unit tests for the ScaleAbsorption optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

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
