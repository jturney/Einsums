//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Hardware/CpuInfo.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(_WIN32)
#    include <windows.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::hardware {

namespace {

/// Detect CPU cache sizes. Returns {L1, L2, L3} in bytes.
/// Falls back to conservative defaults if detection fails.
struct CacheSizes {
    int64_t l1 = 32 * 1024;       // 32 KB
    int64_t l2 = 256 * 1024;      // 256 KB
    int64_t l3 = 8 * 1024 * 1024; // 8 MB
};

CacheSizes detect_cache_sizes() {
    CacheSizes cs;

    // Diagnostic override, so a blocking change can be measured against the old
    // sizes from ONE binary. Comparing across rebuilds is not reliable for these
    // benchmarks. Format: "L1,L2,L3" in bytes; any field <= 0 keeps the detected
    // value.
    auto const apply_override = [&cs]() {
        char const *env = std::getenv("EINSUMS_CACHE_SIZES");
        if (env == nullptr) {
            return;
        }
        long long a = 0, b = 0, c = 0;
        if (std::sscanf(env, "%lld,%lld,%lld", &a, &b, &c) >= 1) {
            if (a > 0) {
                cs.l1 = static_cast<int64_t>(a);
            }
            if (b > 0) {
                cs.l2 = static_cast<int64_t>(b);
            }
            if (c > 0) {
                cs.l3 = static_cast<int64_t>(c);
            }
        }
    };

#if defined(__APPLE__)
    auto sysctl_i64 = [](char const *name, int64_t fallback) -> int64_t {
        int64_t val = 0;
        size_t  len = sizeof(val);
        if (sysctlbyname(name, &val, &len, nullptr, 0) == 0 && val > 0) {
            return val;
        }
        return fallback;
    };
    cs.l1 = sysctl_i64("hw.l1dcachesize", cs.l1);
    // Plain hw.l2cachesize reports the EFFICIENCY cluster on Apple Silicon, which
    // is not where this work runs: on an M-series machine it reads 4 MB against
    // the performance cluster's 16 MB, so blocking derived from it sized the A
    // panel a quarter of what the cores actually have. hw.perflevel0 is the
    // performance cluster. Same rule ComputeGraph's HardwareProfile already used;
    // the two detectors disagreeing is exactly why they should be one.
    cs.l2 = std::max(sysctl_i64("hw.perflevel0.l2cachesize", 0), sysctl_i64("hw.l2cachesize", cs.l2));
    cs.l3 = sysctl_i64("hw.l3cachesize", cs.l3);
    // Apple Silicon may report L3 as 0; fall back to a reasonable default.
    if (cs.l3 <= 0) {
        cs.l3 = 8 * 1024 * 1024;
    }
#elif defined(__linux__)
    // Read from sysfs: /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/
    auto read_cache = [](int index) -> int64_t {
        std::string   base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
        std::ifstream size_file(base + "size");
        if (!size_file.is_open()) {
            return -1;
        }
        std::string size_str;
        std::getline(size_file, size_str);
        if (size_str.empty()) {
            return -1;
        }
        // Parse "32K", "256K", "8192K" etc.
        int64_t val  = std::stoll(size_str);
        char    unit = size_str.back();
        if (unit == 'K' || unit == 'k') {
            val *= 1024;
        } else if (unit == 'M' || unit == 'm') {
            val *= 1024 * 1024;
        }
        return val;
    };
    // index0 = L1d (usually), index2 = L2, index3 = L3
    // Verify via the "level" file to be safe.
    for (int idx = 0; idx <= 4; ++idx) {
        std::string   level_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/level";
        std::ifstream level_file(level_path);
        if (!level_file.is_open()) {
            continue;
        }
        int level = 0;
        level_file >> level;
        // Also check type: we want "Data" or "Unified", not "Instruction"
        std::string   type_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/type";
        std::ifstream type_file(type_path);
        std::string   type_str;
        if (type_file.is_open()) {
            std::getline(type_file, type_str);
        }
        if (type_str == "Instruction") {
            continue;
        }
        int64_t val = read_cache(idx);
        if (val <= 0) {
            continue;
        }
        if (level == 1) {
            cs.l1 = val;
        } else if (level == 2) {
            cs.l2 = val;
        } else if (level == 3) {
            cs.l3 = val;
        }
    }
#endif

    apply_override();
    return cs;
}

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

/// Native SIMD width in doubles, from the compile-time ISA.
int detect_simd_width_f64() {
#if defined(__AVX512F__)
    return 8;
#elif defined(__AVX__) || defined(__AVX2__)
    return 4;
#else
    // SSE2 (x86-64 baseline), NEON (ARM), or unknown: 128-bit / 2 doubles.
    return 2;
#endif
}

} // namespace

CpuInfo const &cpu_info() {
    static CpuInfo const info = []() {
        CpuInfo i;
        i.simd_width_f64     = detect_simd_width_f64();
        auto const cs        = detect_cache_sizes();
        i.cache.l1           = cs.l1;
        i.cache.l2           = cs.l2;
        i.cache.l3           = cs.l3;
        i.omp_region_cost_ns = measure_omp_region_cost_ns();
        return i;
    }();
    return info;
}

double omp_region_cost_ns() {
    return cpu_info().omp_region_cost_ns;
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
            errno                  = 0;
            char           *end    = nullptr;
            long long const parsed = std::strtoll(env, &end, 10);
            if (errno == 0 && end != env && parsed >= 0) {
                value = static_cast<std::size_t>(parsed);
            }
        }
        return value;
    }();
    return threshold;
}

std::int64_t omp_min_parallel_flops() {
    static std::int64_t const threshold = []() -> std::int64_t {
        // The rate is the one SMALL contractions actually achieve, measured at
        // ~1 GFLOP/s, not the peak. Using peak would put the break-even 20x too
        // high and exclude contractions that genuinely benefit from threads.
        constexpr double kNominalFlopsPerNs = 1.0;
        auto             value              = static_cast<std::int64_t>(omp_region_cost_ns() * kNominalFlopsPerNs);

        if (char const *env = std::getenv("EINSUMS_PACKED_MIN_PARALLEL_FLOPS"); env != nullptr) {
            errno                  = 0;
            char           *end    = nullptr;
            long long const parsed = std::strtoll(env, &end, 10);
            if (errno == 0 && end != env && parsed >= 0) {
                value = static_cast<std::int64_t>(parsed);
            }
        }
        return value;
    }();
    return threshold;
}

} // namespace einsums::hardware
