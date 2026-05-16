//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BasicTaskPool.cpp
/// @brief Demonstrates core TaskPool features: submit, then, when_all, submit_group.
///
/// Shows how to:
///   - Submit individual tasks and get results via TaskHandle
///   - Chain computations with .then() continuations
///   - Wait for multiple tasks with when_all()
///   - Submit groups of tasks with a collective barrier

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <cmath>
#include <numeric>
#include <vector>

namespace tp = einsums::task_pool;

int einsums_main() {
    using namespace einsums;

    auto &pool = tp::TaskPool::get_singleton();
    einsums::println("TaskPool: {} worker threads", pool.num_workers());

    // ── 1. Basic submit and get ──────────────────────────────────────────────
    einsums::println("\n--- Basic submit ---");

    auto handle = pool.submit("compute_pi", []() {
        // Leibniz formula for pi/4
        double sum = 0.0;
        for (int k = 0; k < 1000000; k++) {
            sum += (k % 2 == 0 ? 1.0 : -1.0) / (2.0 * k + 1.0);
        }
        return 4.0 * sum;
    });

    double pi = handle.get();
    einsums::println("Computed pi = {:.10f}", pi);

    // ── 2. Continuation chains with .then() ──────────────────────────────────
    einsums::println("\n--- Continuations ---");

    auto result = pool.submit("base_value", []() { return 10; })
                      .then("double_it", [](int v) { return v * 2; })
                      .then("add_seven", [](int v) { return v + 7; })
                      .then("to_string", [](int v) { return std::to_string(v) + " (computed)"; });

    einsums::println("Chain result: {}", result.get());

    // ── 3. when_all: fan-in from multiple tasks ──────────────────────────────
    einsums::println("\n--- when_all ---");

    std::vector<tp::TaskHandle<double>> tasks;
    for (int i = 0; i < 8; i++) {
        tasks.push_back(pool.submit("sqrt", [i]() { return std::sqrt(static_cast<double>(i * i + 1)); }));
    }

    auto all    = tp::when_all(std::move(tasks));
    auto values = all.get();

    for (double v : values) {
        einsums::println("  {:.2f}", v);
    }

    // ── 4. Task group with collective barrier ────────────────────────────────
    einsums::println("\n--- Task group ---");

    std::atomic<int>                   counter{0};
    std::vector<std::function<void()>> group_tasks;
    for (int i = 0; i < 20; i++) {
        group_tasks.push_back([&counter, i]() { counter.fetch_add(i); });
    }

    auto group = pool.submit_group("accumulate", std::move(group_tasks));
    group.wait_all();
    einsums::println("Group sum: {} (expected: {})", counter.load(), 20 * 19 / 2);

    // ── 5. Metrics ───────────────────────────────────────────────────────────
    einsums::println("\n--- Metrics ---");
    auto m = pool.snapshot_metrics();
    einsums::println("Total submitted: {}", m.total_submitted);
    einsums::println("Total completed: {}", m.total_completed);
    einsums::println("Total steals:    {}", m.total_steals);
    for (size_t i = 0; i < m.per_worker_executed.size(); i++) {
        einsums::println("  Worker {}: {} executed, {} stolen", i, m.per_worker_executed[i], m.per_worker_stolen[i]);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
