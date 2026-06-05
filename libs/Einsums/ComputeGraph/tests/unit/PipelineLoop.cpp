//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Pipeline - linear stages", "[ComputeGraph][Pipeline]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Pipeline pipeline("test_linear");

    // Stage 1: B = A (copy via permute), then B *= 2
    {
        auto                  &stage = pipeline.add_stage("init");
        cg::CaptureGuard const guard(stage);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
        cg::scale(2.0, &B);
    }

    // Stage 2: C = B (copy)
    {
        auto                  &stage = pipeline.add_stage("copy");
        cg::CaptureGuard const guard(stage);
        cg::permute("ij <- ij", 0.0, &C, 1.0, B);
    }

    pipeline.execute();

    // C should equal 2 * A
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - 2.0 * A(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Pipeline - loop with early exit", "[ComputeGraph][Pipeline]") {
    // Each iteration: value *= 0.5
    // Converge when value < threshold

    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    double threshold         = 1.0;
    size_t actual_iterations = 0;

    cg::Pipeline pipeline("test_loop");

    // Loop stage: halve the value each iteration
    {
        auto                  &loop_body = pipeline.add_loop("halving_loop",
                                                             /*max_iterations=*/1000,
                                                             /*condition=*/[&](size_t iter) -> bool {
                                                actual_iterations = iter + 1;
                                                return value(0) >= threshold; // continue if above threshold
                                            });
        cg::CaptureGuard const guard(loop_body);
        cg::scale(0.5, &value);
    }

    pipeline.execute();

    // 100 * 0.5^n < 1.0 → n >= 7 (100 * 0.5^7 = 0.78125)
    REQUIRE(actual_iterations == 7);
    REQUIRE(value(0) < threshold);
    REQUIRE(std::abs(value(0) - 100.0 * std::pow(0.5, 7)) < 1e-12);
}

TEST_CASE("Pipeline - loop hits max iterations", "[ComputeGraph][Pipeline]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    cg::Pipeline pipeline("test_max_iter");

    {
        auto                  &loop_body = pipeline.add_loop("never_converge",
                                                             /*max_iterations=*/10,
                                                             /*condition=*/[](size_t) -> bool {
                                                return true; // never converge
                                            });
        cg::CaptureGuard const guard(loop_body);
        cg::scale(0.9, &value);
    }

    pipeline.execute();

    // Should have run exactly 10 iterations
    REQUIRE(std::abs(value(0) - 100.0 * std::pow(0.9, 10)) < 1e-10);
}

TEST_CASE("Pipeline - setup + loop + postprocess", "[ComputeGraph][Pipeline]") {
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto C   = create_zero_tensor<double>("C", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    size_t loop_count = 0;

    cg::Pipeline pipeline("full_pipeline");

    // Stage 1: Setup - compute C = A * B
    {
        auto                  &setup = pipeline.add_stage("setup");
        cg::CaptureGuard const guard(setup);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Stage 2: Loop - accumulate C into acc, 5 times
    {
        auto                  &loop_body = pipeline.add_loop("accumulate",
                                                             /*max_iterations=*/5,
                                                             /*condition=*/[&](size_t iter) -> bool {
                                                loop_count = iter + 1;
                                                return iter < 4; // run exactly 5 iterations (0..4)
                                            });
        cg::CaptureGuard const guard(loop_body);
        cg::axpy(1.0, C, &acc);
    }

    // Stage 3: Post-process - scale result
    {
        auto                  &post = pipeline.add_stage("postprocess");
        cg::CaptureGuard const guard(post);
        cg::scale(0.2, &acc); // average over 5 iterations
    }

    pipeline.execute();

    REQUIRE(loop_count == 5);

    // acc should equal C (5 * C * 0.2 = C)
    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(acc(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}
