//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkTaskPool.cpp
/// @brief Exhaustive benchmarks for TaskPool: submit latency, parallel_for/reduce
///        scaling, work-stealing efficiency, continuation overhead, dataflow.

#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums::performance;
namespace tp = einsums::task_pool;

// ═══════════════════════════════════════════════════════════════════════════════
// Submit + get latency (round-trip overhead per task)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: submit+get latency (empty task)", "[TaskPool][latency][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "latency");

    auto t = time_us(
        "submit+get empty",
        [&]() {
            auto h = pool.submit("noop", []() {});
            h.wait();
        },
        1000);

    std::printf("[TaskPool submit+get latency] %.2f us avg  (%.2f min, %.2f max)\n", t.avg, t.min, t.max);
    publish_benchmark_result("submit+get empty", "t_latency", 0, t);
}

EINSUMS_TEST_CASE("Bench TaskPool: submit+get latency (returning int)", "[TaskPool][latency][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "latency");

    auto t = time_us(
        "submit+get int",
        [&]() {
            auto h = pool.submit("ret42", []() { return 42; });
            h.get();
        },
        1000);

    std::printf("[TaskPool submit+get int] %.2f us avg\n", t.avg);
    publish_benchmark_result("submit+get int", "t_latency", 0, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Continuation overhead (.then chain)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: .then() chain length 1", "[TaskPool][continuation][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "continuation");
    ProfileAnnotate("chain_length", int64_t(1));

    auto t = time_us(
        "then-1",
        [&]() {
            auto h = pool.submit("base", []() { return 1; }).then([](int v) { return v + 1; });
            h.get();
        },
        500);

    std::printf("[TaskPool .then x1] %.2f us avg\n", t.avg);
    publish_benchmark_result("then-1", "t_latency", 0, t);
}

EINSUMS_TEST_CASE("Bench TaskPool: .then() chain length 5", "[TaskPool][continuation][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "continuation");
    ProfileAnnotate("chain_length", int64_t(5));

    auto t = time_us(
        "then-5",
        [&]() {
            auto h = pool.submit("base", []() { return 1; })
                         .then([](int v) { return v + 1; })
                         .then([](int v) { return v + 1; })
                         .then([](int v) { return v + 1; })
                         .then([](int v) { return v + 1; })
                         .then([](int v) { return v + 1; });
            h.get();
        },
        200);

    std::printf("[TaskPool .then x5] %.2f us avg\n", t.avg);
    publish_benchmark_result("then-5", "t_latency", 0, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Throughput: many independent tasks
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: 100 independent tasks", "[TaskPool][throughput][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "throughput");
    ProfileAnnotate("tasks", int64_t(100));

    auto t = time_us(
        "100-tasks",
        [&]() {
            std::vector<tp::TaskHandle<void>> handles;
            handles.reserve(100);
            std::atomic<int> sum{0};
            for (int i = 0; i < 100; i++) {
                handles.push_back(pool.submit("work", [&sum]() { sum.fetch_add(1); }));
            }
            for (auto &h : handles)
                h.wait();
        },
        50);

    std::printf("[TaskPool 100 tasks] %.2f us avg  (%.0f tasks/sec)\n", t.avg, 100.0 / (t.avg * 1e-6));
    publish_benchmark_result("100-tasks", "t_throughput", 100, t);
}

EINSUMS_TEST_CASE("Bench TaskPool: 1000 independent tasks", "[TaskPool][throughput][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "throughput");
    ProfileAnnotate("tasks", int64_t(1000));

    auto t = time_us(
        "1000-tasks",
        [&]() {
            std::vector<tp::TaskHandle<void>> handles;
            handles.reserve(1000);
            std::atomic<int> sum{0};
            for (int i = 0; i < 1000; i++) {
                handles.push_back(pool.submit("work", [&sum]() { sum.fetch_add(1); }));
            }
            for (auto &h : handles)
                h.wait();
        },
        20);

    std::printf("[TaskPool 1000 tasks] %.2f us avg  (%.0f tasks/sec)\n", t.avg, 1000.0 / (t.avg * 1e-6));
    publish_benchmark_result("1000-tasks", "t_throughput", 1000, t);
}

EINSUMS_TEST_CASE("Bench TaskPool: 10000 independent tasks", "[TaskPool][throughput][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "throughput");
    ProfileAnnotate("tasks", int64_t(10000));

    auto t = time_us(
        "10000-tasks",
        [&]() {
            std::vector<tp::TaskHandle<void>> handles;
            handles.reserve(10000);
            std::atomic<int> sum{0};
            for (int i = 0; i < 10000; i++) {
                handles.push_back(pool.submit("work", [&sum]() { sum.fetch_add(1); }));
            }
            for (auto &h : handles)
                h.wait();
        },
        5);

    std::printf("[TaskPool 10000 tasks] %.2f us avg  (%.0f tasks/sec)\n", t.avg, 10000.0 / (t.avg * 1e-6));
    publish_benchmark_result("10000-tasks", "t_throughput", 10000, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// parallel_for scaling
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: parallel_for N=1000 (light work)", "[TaskPool][parallel_for][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();

    std::vector<double> data(1000);
    ProfileAnnotate("category", "parallel_for");
    ProfileAnnotate("N", int64_t(1000));

    auto t_seq = time_us(
        "seq",
        [&]() {
            for (size_t i = 0; i < 1000; i++)
                data[i] = std::sin(static_cast<double>(i));
        },
        100);

    auto t_par = time_us(
        "par", [&]() { pool.parallel_for("fill", 0, 1000, [&data](size_t i) { data[i] = std::sin(static_cast<double>(i)); }); }, 100);

    double speedup = t_seq.avg / t_par.avg;
    std::printf("[parallel_for N=1K] seq: %.2f us  par: %.2f us  speedup: %.2fx\n", t_seq.avg, t_par.avg, speedup);
    publish_benchmark_result("parallel_for-1K-seq", "t_sequential", 1000, t_seq);
    publish_benchmark_result("parallel_for-1K-par", "t_parallel", 1000, t_par);
}

EINSUMS_TEST_CASE("Bench TaskPool: parallel_for N=100000 (light work)", "[TaskPool][parallel_for][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();

    std::vector<double> data(100000);
    ProfileAnnotate("category", "parallel_for");
    ProfileAnnotate("N", int64_t(100000));

    auto t_seq = time_us(
        "seq",
        [&]() {
            for (size_t i = 0; i < 100000; i++)
                data[i] = std::sin(static_cast<double>(i));
        },
        20);

    auto t_par = time_us(
        "par", [&]() { pool.parallel_for("fill", 0, 100000, [&data](size_t i) { data[i] = std::sin(static_cast<double>(i)); }); }, 20);

    double speedup = t_seq.avg / t_par.avg;
    std::printf("[parallel_for N=100K] seq: %.2f us  par: %.2f us  speedup: %.2fx\n", t_seq.avg, t_par.avg, speedup);
    publish_benchmark_result("parallel_for-100K-seq", "t_sequential", 100000, t_seq);
    publish_benchmark_result("parallel_for-100K-par", "t_parallel", 100000, t_par);
}

EINSUMS_TEST_CASE("Bench TaskPool: parallel_for N=1000000 (light work)", "[TaskPool][parallel_for][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();

    std::vector<double> data(1000000);
    ProfileAnnotate("category", "parallel_for");
    ProfileAnnotate("N", int64_t(1000000));

    auto t_seq = time_us(
        "seq",
        [&]() {
            for (size_t i = 0; i < 1000000; i++)
                data[i] = std::sin(static_cast<double>(i));
        },
        5);

    auto t_par = time_us(
        "par", [&]() { pool.parallel_for("fill", 0, 1000000, [&data](size_t i) { data[i] = std::sin(static_cast<double>(i)); }); }, 5);

    double speedup = t_seq.avg / t_par.avg;
    std::printf("[parallel_for N=1M] seq: %.2f us  par: %.2f us  speedup: %.2fx\n", t_seq.avg, t_par.avg, speedup);
    publish_benchmark_result("parallel_for-1M-seq", "t_sequential", 1000000, t_seq);
    publish_benchmark_result("parallel_for-1M-par", "t_parallel", 1000000, t_par);
}

// ═══════════════════════════════════════════════════════════════════════════════
// parallel_reduce scaling
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: parallel_reduce N=100000 (dot product)", "[TaskPool][parallel_reduce][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();

    std::vector<double> a(100000), b(100000);
    for (size_t i = 0; i < 100000; i++) {
        a[i] = static_cast<double>(i) * 0.001;
        b[i] = 1.0 / (static_cast<double>(i) + 1.0);
    }
    ProfileAnnotate("category", "parallel_reduce");
    ProfileAnnotate("N", int64_t(100000));

    double volatile seq_result = 0.0;
    auto t_seq                 = time_us(
        "seq",
        [&]() {
            double sum = 0.0;
            for (size_t i = 0; i < 100000; i++)
                sum += a[i] * b[i];
            seq_result = sum;
        },
        50);

    double volatile par_result = 0.0;
    auto t_par                 = time_us(
        "par",
        [&]() {
            par_result = pool.parallel_reduce<double>(
                "dot", 0, 100000, []() { return 0.0; }, [&](size_t i, double &acc) { acc += a[i] * b[i]; },
                [](double &g, double const &l) { g += l; });
        },
        50);

    double speedup = t_seq.avg / t_par.avg;
    std::printf("[parallel_reduce N=100K] seq: %.2f us  par: %.2f us  speedup: %.2fx\n", t_seq.avg, t_par.avg, speedup);
    std::printf("  seq_result=%.6f  par_result=%.6f\n", (double)seq_result, (double)par_result);
    publish_benchmark_result("parallel_reduce-100K-seq", "t_sequential", 100000, t_seq);
    publish_benchmark_result("parallel_reduce-100K-par", "t_parallel", 100000, t_par);
}

EINSUMS_TEST_CASE("Bench TaskPool: parallel_reduce N=10000000 (dot product)", "[TaskPool][parallel_reduce][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();

    std::vector<double> a(10000000), b(10000000);
    for (size_t i = 0; i < 10000000; i++) {
        a[i] = static_cast<double>(i) * 0.0001;
        b[i] = 1.0 / (static_cast<double>(i) + 1.0);
    }
    ProfileAnnotate("category", "parallel_reduce");
    ProfileAnnotate("N", int64_t(10000000));

    double volatile seq_result = 0.0;
    auto t_seq                 = time_us(
        "seq",
        [&]() {
            double sum = 0.0;
            for (size_t i = 0; i < 10000000; i++)
                sum += a[i] * b[i];
            seq_result = sum;
        },
        5);

    double volatile par_result = 0.0;
    auto t_par                 = time_us(
        "par",
        [&]() {
            par_result = pool.parallel_reduce<double>(
                "dot", 0, 10000000, []() { return 0.0; }, [&](size_t i, double &acc) { acc += a[i] * b[i]; },
                [](double &g, double const &l) { g += l; });
        },
        5);

    double speedup = t_seq.avg / t_par.avg;
    std::printf("[parallel_reduce N=10M] seq: %.2f us  par: %.2f us  speedup: %.2fx\n", t_seq.avg, t_par.avg, speedup);
    std::printf("  seq_result=%.6f  par_result=%.6f\n", (double)seq_result, (double)par_result);
    publish_benchmark_result("parallel_reduce-10M-seq", "t_sequential", 10000000, t_seq);
    publish_benchmark_result("parallel_reduce-10M-par", "t_parallel", 10000000, t_par);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Work-stealing efficiency: imbalanced workload
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: work-stealing imbalanced 100 tasks", "[TaskPool][steal][benchmark]") {
    LabeledSection0();
    auto &pool   = tp::TaskPool::get_singleton();
    auto  before = pool.snapshot_metrics();
    ProfileAnnotate("category", "work-stealing");
    ProfileAnnotate("tasks", int64_t(100));

    auto t = time_us(
        "imbalanced",
        [&]() {
            std::vector<tp::TaskHandle<void>> handles;
            handles.reserve(100);
            for (int i = 0; i < 100; i++) {
                handles.push_back(pool.submit("imbal", [i]() {
                    volatile double sum   = 0.0;
                    int             iters = (i % 10 == 0) ? 100000 : 1000;
                    for (int j = 0; j < iters; j++)
                        sum += static_cast<double>(j) * 0.001;
                }));
            }
            for (auto &h : handles)
                h.wait();
        },
        5);

    auto   after  = pool.snapshot_metrics();
    size_t steals = after.total_steals - before.total_steals;

    std::printf("[TaskPool imbalanced 100] %.2f us avg  steals: %zu\n", t.avg, steals);
    for (size_t w = 0; w < after.per_worker_executed.size(); w++) {
        size_t delta = after.per_worker_executed[w] - (w < before.per_worker_executed.size() ? before.per_worker_executed[w] : 0);
        std::printf("  Worker %zu: %zu tasks\n", w, delta);
    }
    publish_benchmark_result("imbalanced-100", "t_throughput", 100, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// submit_group overhead
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench TaskPool: submit_group 100 tasks", "[TaskPool][group][benchmark]") {
    LabeledSection0();
    auto &pool = tp::TaskPool::get_singleton();
    ProfileAnnotate("category", "submit_group");
    ProfileAnnotate("tasks", int64_t(100));

    auto t = time_us(
        "group-100",
        [&]() {
            std::atomic<int>                   sum{0};
            std::vector<std::function<void()>> tasks;
            tasks.reserve(100);
            for (int i = 0; i < 100; i++) {
                tasks.push_back([&sum]() { sum.fetch_add(1); });
            }
            auto group = pool.submit_group("bench", std::move(tasks));
            group.wait_all();
        },
        50);

    std::printf("[TaskPool submit_group 100] %.2f us avg\n", t.avg);
    publish_benchmark_result("submit_group-100", "t_throughput", 100, t);
}
