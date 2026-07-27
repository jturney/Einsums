//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/ParallelThreshold.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums {

namespace {

/// Time entering and leaving an empty parallel region. The team is warmed first
/// so this measures steady-state fork/join rather than one-off thread creation,
/// and the best of several trials is taken because anything else running on the
/// machine only ever makes it look slower.
double measure_omp_region_cost_ns() {
#ifdef _OPENMP
    if (omp_get_max_threads() <= 1) {
        return 0.0;
    }
    constexpr int kReps   = 200;
    constexpr int kTrials = 5;
    int volatile sink     = 0;
    auto        once      = [&sink]() {
#    pragma omp parallel
        { sink = omp_get_thread_num(); }
    };
    for (int i = 0; i < kReps; ++i) {
        once();
    }
    double best = 1e30;
    for (int t = 0; t < kTrials; ++t) {
        auto const start = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) {
            once();
        }
        auto const stop = std::chrono::steady_clock::now();
        best            = std::min(best, std::chrono::duration<double, std::nano>(stop - start).count() / kReps);
    }
    return best;
#else
    return 0.0;
#endif
}

} // namespace

double omp_region_cost_ns() {
    static double const cost = measure_omp_region_cost_ns();
    return cost;
}

std::size_t omp_min_parallel_elements() {
    static std::size_t const threshold = []() -> std::size_t {
        // Elementwise kernels are bandwidth-bound. One element per nanosecond is a
        // conservative rate for cache-resident data, so the break-even element
        // count is numerically the region cost in nanoseconds.
        auto value = static_cast<std::size_t>(omp_region_cost_ns());

        // Diagnostic override, so a threshold can be measured against the
        // unthresholded behaviour from ONE binary. Comparing across rebuilds is
        // not viable for this: the benchmarks involved swing by tens of percent
        // with unrelated machine activity, and only a same-binary A/B holds its
        // controls steady. Zero restores "always parallelize".
        if (char const *env = std::getenv("EINSUMS_OMP_MIN_PARALLEL_ELEMENTS"); env != nullptr) {
            errno            = 0;
            char     *end    = nullptr;
            long long parsed = std::strtoll(env, &end, 10);
            if (errno == 0 && end != env && parsed >= 0) {
                value = static_cast<std::size_t>(parsed);
            }
        }
        return value;
    }();
    return threshold;
}

} // namespace einsums
