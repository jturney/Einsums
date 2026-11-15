//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
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
    REQUIRE(pipeline.last_iteration_count(0) == 7);
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
    reference_einsum("ij <- ik ; kj", &C_ref, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(acc(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

// A loop stage's body was the stage graph itself, so a pipeline-declared tensor first used in a
// loop stage had its lifecycle adopted into the body, and its Initialize ran on every iteration:
// an accumulator declared at zero restarted each iteration and ended holding one iteration's sum.
// The lifecycle belongs before the loop, once per execute, as an in-graph loop places it.
TEST_CASE("Pipeline - a declared zero tensor first used in a loop stage accumulates across iterations", "[ComputeGraph][Pipeline]") {
    constexpr size_t n   = 4;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             B   = create_random_tensor<double>("B", n, n);
    auto             out = create_zero_tensor<double>("out", n, n);

    cg::Pipeline pipeline("declared_accumulator");
    auto        &acc = pipeline.declare_zero_tensor<double, 2>("acc", n, n);
    {
        auto                  &body = pipeline.add_loop("accumulate", /*max_iterations=*/10, [](size_t iter) { return iter < 2; });
        cg::CaptureGuard const guard(body);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &acc, 1.0, A, B);
    }
    {
        auto                  &post = pipeline.add_stage("copy_out");
        cg::CaptureGuard const guard(post);
        cg::permute("ij <- ij", 0.0, &out, 1.0, acc);
    }
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);

    auto product = create_zero_tensor<double>("product", n, n);
    reference_einsum("i,j <- i,k ; k,j", &product, A, B);
    // Twice, so the accumulator is shown to start from zero once per execute rather than once ever.
    for (int run = 0; run < 2; ++run) {
        INFO("execute " << run);
        pipeline.execute();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                CHECK_THAT(out(i, j), Catch::Matchers::WithinAbs(3.0 * product(i, j), 1e-12));
            }
        }
    }
}

// The pipeline's first-use walk passed over Setup bodies, so a stage whose only use of a
// pipeline-declared tensor was inside its Setup body was not taken as the first user. The next stage
// to touch the tensor was, and its Initialize zeroed the setup's result just before that stage read
// it. The lifecycle belongs inside the Setup body, where Materialization places a body's own
// scratch, so it runs once per bind with the setup and the second execute keeps the result.
TEST_CASE("Pipeline - a declared zero tensor written only by a setup body keeps the setup's result", "[ComputeGraph][Pipeline]") {
    constexpr size_t n   = 4;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             B   = create_random_tensor<double>("B", n, n);
    auto             out = create_zero_tensor<double>("out", n, n);

    cg::Pipeline pipeline("setup_only_writer");
    auto        &acc = pipeline.declare_zero_tensor<double, 2>("acc", n, n);
    {
        auto                  &stage = pipeline.add_stage("fit");
        auto                  &setup = stage.add_setup("fit_setup");
        cg::CaptureGuard const guard(setup);
        cg::einsum("i,j <- i,k ; k,j", 1.0, &acc, 1.0, A, B); // reads acc, so it must start at zero
    }
    {
        auto                  &post = pipeline.add_stage("copy_out");
        cg::CaptureGuard const guard(post);
        cg::permute("ij <- ij", 0.0, &out, 1.0, acc);
    }
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);

    auto product = create_zero_tensor<double>("product", n, n);
    reference_einsum("i,j <- i,k ; k,j", &product, A, B);
    // Twice: the setup runs on the first execute only, so the second shows nothing zeroes its result.
    for (int run = 0; run < 2; ++run) {
        INFO("execute " << run);
        pipeline.execute();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                CHECK_THAT(out(i, j), Catch::Matchers::WithinAbs(product(i, j), 1e-12));
            }
        }
    }
}

TEST_CASE("Pipeline - stage references survive later add_stage", "[ComputeGraph][Pipeline]") {
    // add_stage() hands out a reference into the pipeline's stage container.
    // While that container was a std::vector, the next add_stage() could
    // reallocate and leave this reference pointing at a moved-from Graph,
    // whose content mutex is null. The symptom was a "recursive_mutex lock
    // failed" from whatever touched the stale reference, arbitrarily far from
    // the add_stage() that caused it.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);

    cg::Pipeline pipeline("stage_reference_stability");

    auto &first = pipeline.add_stage("first");
    {
        cg::CaptureGuard const guard(first);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
    }
    REQUIRE(first.num_nodes() == 1);

    // Enough stages to force any vector to reallocate more than once.
    for (int s = 0; s < 16; ++s) {
        pipeline.add_stage(fmt::format("filler{}", s));
    }

    // The reference handed out first must still name a live, usable Graph.
    REQUIRE(first.num_nodes() == 1);
    REQUIRE_NOTHROW(first.to_json());
    REQUIRE_NOTHROW(first.execute());
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(B(ii, jj) - A(ii, jj)) < 1e-12);
        }
    }
}
