//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DataflowPipeline.cpp
/// @brief Demonstrates dataflow task dependencies for multi-stage pipelines.
///
/// Shows how to:
///   - Use dataflow() to express task dependencies via futures
///   - Build a pipeline where each stage runs only when its inputs are ready
///   - Simulate a quantum chemistry workflow: screening -> integrals -> assembly

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <tuple>
#include <vector>

namespace tp = einsums::task_pool;

int einsums_main() {
    using namespace einsums;

    auto &pool = tp::TaskPool::get_singleton();

    einsums::println("=== Dataflow Pipeline Demo ===\n");

    // ── Simulate a multi-stage pipeline ──────────────────────────────────────
    // Stage 1: Compute some data (e.g., screening bounds)
    // Stage 2: Use that data to compute batches in parallel
    // Stage 3: Assemble results

    // Stage 1: compute screening bounds
    einsums::println("Submitting stage 1: screening...");
    auto screening = pool.submit("screening", []() {
        // Simulate computation
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return std::vector<double>{0.1, 0.5, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4};
    });

    // Stage 2: depends on screening; use .then() for typed chaining
    einsums::println("Submitting stage 2: batch computation (continuation)...");
    auto batches = screening.then("batch_compute", [&pool](std::vector<double> const &bounds) {
        einsums::println("  Stage 2 running: received {} screening bounds", bounds.size());

        size_t count = 0;
        for (double b : bounds) {
            if (b > 0.3)
                count++;
        }
        einsums::println("  {} significant pairs found", count);

        // Process significant pairs in parallel
        double total = pool.parallel_reduce<double>(
            "process_pairs", 0, bounds.size(), []() { return 0.0; },
            [&bounds](size_t i, double &acc) {
                if (bounds[i] > 0.3) {
                    acc += std::sin(bounds[i]) * std::cos(bounds[i]);
                }
            },
            [](double &g, double const &l) { g += l; });

        return total;
    });

    // Stage 3: depends on batch results; continuation chain
    einsums::println("Submitting stage 3: assembly (continuation)...");
    auto result = batches.then("assemble", [](double batch_result) {
        einsums::println("  Stage 3 running: assembling from batch result = {:.6f}", batch_result);
        return batch_result * 2.0 + 1.0;
    });

    // Wait for final result
    double final_value = result.get();
    einsums::println("\nFinal result: {:.6f}", final_value);

    // ── Demonstrate independent dataflow branches ────────────────────────────
    einsums::println("\n=== Independent Branches ===\n");

    // Two independent computations that can run concurrently
    auto branch_a = pool.submit("branch_a", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return 42.0;
    });

    auto branch_b = pool.submit("branch_b", []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return 58.0;
    });

    // Fan-in with variadic when_all (returns TaskHandle<tuple<double, double>>)
    auto combined = tp::when_all(branch_a, branch_b);
    auto merge    = combined.then("merge", [](std::tuple<double, double> const &vals) { return std::get<0>(vals) + std::get<1>(vals); });

    einsums::println("Combined result: {:.1f} (expected: 100.0)", merge.get());

    // ── Typed dataflow: combine different types ──────────────────────────────
    einsums::println("\n=== Typed Dataflow ===\n");

    auto int_val = pool.submit("int", []() { return 10; });
    auto dbl_val = pool.submit("dbl", []() { return 2.5; });
    auto str_val = pool.submit("str", []() { return std::string("result"); });

    // dataflow with mixed types: fully async, no blocking
    auto typed_result = pool.dataflow(
        "combine_typed",
        [](int x, double y, std::string const &label) { return fmt::format("{}: {:.1f}", label, static_cast<double>(x) * y); }, int_val,
        dbl_val, str_val);

    einsums::println("Typed dataflow: {}", typed_result.get());

    // Chained typed dataflow: A + B -> C -> D
    auto a = pool.submit("a", []() { return 3; });
    auto b = pool.submit("b", []() { return 4.0; });

    auto sum = pool.dataflow(
        "sum", [](int x, double y) { return static_cast<double>(x) + y; }, a, b);

    auto final_val = sum.then("format", [](double v) { return fmt::format("Final: {:.1f}", v); });

    einsums::println("{}", final_val.get());

    einsums::println("\n=== Pipeline Complete ===");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
