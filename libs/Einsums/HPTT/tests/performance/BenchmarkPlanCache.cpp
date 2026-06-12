//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Plan cache regression coverage.
//
// HPTT's create_plan() builds a plan tree (loop ordering, parallelism strategy)
// every call. The cache helper in TensorAlgebra/Detail/HpttPlanCache.hpp keeps
// that tree across calls so only A/B/alpha/beta change per execute. The win is
// largest at small N where the build cost dominates the actual transpose.
//
// We measure two metrics per size:
//   t_plan_uncached = create_plan + execute  (raw HPTT path)
//   t_plan_cached   = lookup + setters + execute  (the cache helper)
//
// A regression where the cache stops working would show t_cached approaching
// t_uncached. The diff between them is the plan-construction cost.

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TensorAlgebra/Detail/HpttPlanCache.hpp>

#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace einsums::performance;

namespace {

inline int hptt_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void bench_plan(int N) {
    LabeledSection0();
    std::vector<float> A(static_cast<size_t>(N) * N, 1.0f), B(static_cast<size_t>(N) * N);
    int const          perm[2]{1, 0};
    size_t const       size[2]{static_cast<size_t>(N), static_cast<size_t>(N)};

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("dtype", "float");
    ProfileAnnotate("elements", int64_t(N) * N);

    auto t_uncached = time_us("plan-uncached", [&] {
        auto const plan =
            hptt::create_plan<float>(perm, 2, 1.0f, A.data(), size, nullptr, 0.0f, B.data(), nullptr, hptt::ESTIMATE, hptt_threads());
        plan->execute();
    });
    publish_benchmark_result("plan-uncached", "t_plan", N, t_uncached);

    // Warm the per-thread cache once before the timed loop so we measure
    // the steady-state hit cost, not the first-build cost.
    auto warm = einsums::tensor_algebra::detail::get_or_create_hptt_plan<float>(perm, 2, 1.0f, A.data(), size, 0.0f, B.data(), false);
    warm->execute();

    auto t_cached = time_us("plan-cached", [&] {
        auto const plan =
            einsums::tensor_algebra::detail::get_or_create_hptt_plan<float>(perm, 2, 1.0f, A.data(), size, 0.0f, B.data(), false);
        plan->execute();
    });
    publish_benchmark_result("plan-cached", "t_plan", N, t_cached);
}

} // namespace

EINSUMS_TEST_CASE("Plan cache hit vs miss", "[performance][hptt][plancache]") {
    // Small sizes where plan construction dominates the actual transpose;
    // that's where the cache saves the most.
    for (int N : {32, 64, 128, 256, 512, 1024})
        bench_plan(N);
}
