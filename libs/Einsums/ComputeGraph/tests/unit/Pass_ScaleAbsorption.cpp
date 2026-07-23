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

TEST_CASE("ScaleAbsorption - no absorption when beta != 0", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);

    cg::Graph graph("no_absorb");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::ScaleAbsorption>();
    REQUIRE_FALSE(modified);
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

TEST_CASE("ScaleAbsorption - does not fold with two readers of the scaled tensor", "[ComputeGraph][Passes]") {
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
