//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The batched GEMMs' threading gate, and the inline path it sends small batches down.
//
// The gate once tested only for an enclosing team, so a batch of two 2x2 products opened a full
// OpenMP team and took 28 us against 2 us run inline. A region that does not fork cannot be seen
// from outside, so the decision is pinned through the function every batched entry point asks.

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Testing.hpp>

using einsums::blas::int_t;
namespace vendor = einsums::blas::vendor;

TEST_CASE("GemmBatchGate - a batch of tiny products never forks", "[BLASVendor][GemmBatch]") {
    // Two 2x2x2 real products: 32 flops.
    REQUIRE_FALSE(vendor::gemm_batch_wants_threads(32.0));
    REQUIRE_FALSE(vendor::gemm_batch_wants_threads(0.0));
    auto const threshold = static_cast<double>(einsums::hardware::omp_min_parallel_flops());
    REQUIRE_FALSE(vendor::gemm_batch_wants_threads(threshold / 2));
}

TEST_CASE("GemmBatchGate - a batch worth a team forks when there are threads to fork", "[BLASVendor][GemmBatch]") {
#ifdef _OPENMP
    bool const threads = omp_get_max_threads() > 1;
#else
    bool const threads = false;
#endif
    auto const threshold = static_cast<double>(einsums::hardware::omp_min_parallel_flops());
    REQUIRE(vendor::gemm_batch_wants_threads(threshold) == threads);
    REQUIRE(vendor::gemm_batch_wants_threads(1e12) == threads);
}

TEST_CASE("GemmBatchGate - a batch inside an enclosing team never forks", "[BLASVendor][GemmBatch]") {
#ifdef _OPENMP
    bool                 inside = true;
#    pragma omp parallel num_threads(2) reduction(&& : inside)
    {
        inside = !vendor::gemm_batch_wants_threads(1e12);
    }
    REQUIRE(inside);
#else
    SUCCEED("no OpenMP");
#endif
}

TEMPLATE_TEST_CASE("GemmBatchGate - the inline path computes every member", "[BLASVendor][GemmBatch]", float, double, std::complex<float>,
                   std::complex<double>) {
    // Small enough to stay inline whatever the thread count, so this exercises the serial loop.
    using T           = TestType;
    constexpr int_t n = 3, count = 4;
    std::vector<T>  a(n * n * count), b(n * n * count), c(n * n * count, T{-7});
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = T(static_cast<float>(i % 5) - 2.0F);
        b[i] = T(static_cast<float>(i % 3) + 0.5F);
    }
    std::vector<T const *> ap, bp;
    std::vector<T *>       cp;
    for (int_t m = 0; m < count; ++m) {
        ap.push_back(a.data() + m * n * n);
        bp.push_back(b.data() + m * n * n);
        cp.push_back(c.data() + m * n * n);
    }
    REQUIRE_FALSE(vendor::gemm_batch_wants_threads(2.0 * n * n * n * count * (einsums::IsComplexV<T> ? 4 : 1)));
    if constexpr (std::is_same_v<T, float>) {
        vendor::sgemm_batch('n', 'n', n, n, n, 1.0F, ap.data(), n, bp.data(), n, 0.0F, cp.data(), n, count);
    } else if constexpr (std::is_same_v<T, double>) {
        vendor::dgemm_batch('n', 'n', n, n, n, 1.0, ap.data(), n, bp.data(), n, 0.0, cp.data(), n, count);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        vendor::cgemm_batch('n', 'n', n, n, n, T{1}, ap.data(), n, bp.data(), n, T{0}, cp.data(), n, count);
    } else {
        vendor::zgemm_batch('n', 'n', n, n, n, T{1}, ap.data(), n, bp.data(), n, T{0}, cp.data(), n, count);
    }
    for (int_t m = 0; m < count; ++m) {
        for (int_t j = 0; j < n; ++j) {
            for (int_t i = 0; i < n; ++i) {
                T ref{0};
                for (int_t p = 0; p < n; ++p) {
                    ref += a[m * n * n + i + p * n] * b[m * n * n + p + j * n];
                }
                REQUIRE(std::abs(c[m * n * n + i + j * n] - ref) <= 1e-5 * (1 + std::abs(ref)));
            }
        }
    }
}
