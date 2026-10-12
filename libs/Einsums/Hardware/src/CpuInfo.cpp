//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Hardware/Profile.hpp>

#include <Stripes/RuntimeFeatures.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>

#if !defined(_WIN32)
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(hardware)

namespace detail {

/// Detect CPU cache sizes in bytes, keeping CacheSizes' conservative defaults
/// for any level detection cannot read.
CacheSizes detected_cache_sizes() {
    CacheSizes cs;

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
    // On Apple Silicon hw.l2cachesize is the efficiency cluster's; hw.perflevel0 is the
    // performance cluster, where the work runs.
    cs.l2 = std::max(sysctl_i64("hw.perflevel0.l2cachesize", 0), sysctl_i64("hw.l2cachesize", cs.l2));
    cs.l3 = sysctl_i64("hw.l3cachesize", cs.l3);
    // Apple Silicon may report L3 as 0; fall back to a reasonable default.
    if (cs.l3 <= 0) {
        cs.l3 = CacheSizes{}.l3;
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

    return cs;
}

/// Time an empty parallel region: warm team, best of several trials (interference only adds).
double measure_omp_region_cost_ns() {
#ifdef _OPENMP
    if (omp_get_max_threads() <= 1) {
        return 0.0;
    }
    constexpr int kReps   = 200;
    constexpr int kTrials = 5;
    int volatile sink     = 0;
    auto once             = [&sink]() {
#    pragma omp parallel
        {
            sink = omp_get_thread_num();
        }
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

/// Where measured constants may be kept between runs: option::CacheDir, else the platform cache
/// location, else "" (then nothing is kept).
std::filesystem::path cache_directory() {
    if (auto const dir = config::get(option::CacheDir); !dir.empty()) {
        return std::filesystem::path(dir);
    }
#if defined(_WIN32)
    if (char const *base = std::getenv("LOCALAPPDATA"); base != nullptr && *base != '\0') {
        return std::filesystem::path(base) / "einsums";
    }
#else
    if (char const *base = std::getenv("XDG_CACHE_HOME"); base != nullptr && *base != '\0') {
        return std::filesystem::path(base) / "einsums";
    }
    if (char const *home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "einsums";
    }
#endif
    return {};
}

} // namespace detail

namespace {

/// Detected cache sizes with the diagnostic override applied: a level pinned for a same-binary
/// A/B run replaces the detected one, and a level left at zero keeps it.
CacheSizes detect_cache_sizes() {
    CacheSizes cs = detail::detected_cache_sizes();
    if (auto const l1 = config::get(option::HardwareL1CacheSize); l1 > 0) {
        cs.l1 = l1;
    }
    if (auto const l2 = config::get(option::HardwareL2CacheSize); l2 > 0) {
        cs.l2 = l2;
    }
    if (auto const l3 = config::get(option::HardwareL3CacheSize); l3 > 0) {
        cs.l3 = l3;
    }
    return cs;
}

/// The profile's entry for @p threads, if it has a usable one.
std::optional<double> profiled_region_cost(int threads) {
    auto const section = profile_section("omp_region_cost_ns");
    if (!section || !section->is_object()) {
        return std::nullopt;
    }
    auto const *entry = section->as_object().peek(std::to_string(threads));
    if (entry == nullptr || !entry->is_number()) {
        return std::nullopt;
    }
    double const value = entry->as_double();
    // A negative or absurd number means a hand-edited file. Measuring again is always safe, so
    // never trust one through.
    if (!(value >= 0.0) || value > 1e9) {
        return std::nullopt;
    }
    return value;
}

/// The region cost: pinned, else from the hardware profile, else measured now. A per-process
/// measurement drifts by tens of percent, which is fine for a threshold but not for ranking options
/// against it. The library never keeps its own measurement: one taken on a busy machine looks like
/// a slow machine.
struct ResolvedRegionCost {
    double value{0.0};
    bool   calibrated{false};
};

ResolvedRegionCost resolve_omp_region_cost_ns() {
    // An explicit pin beats everything, so a benchmark can hold the rate fixed
    // across machines without touching any file.
    if (double const pinned = config::get(option::HardwareOmpRegionCostNs); pinned >= 0.0) {
        return {pinned, true};
    }

#ifdef _OPENMP
    int const threads = omp_get_max_threads();
#else
    int const threads = 1;
#endif
    // A single thread enters no region, so there is nothing to calibrate and
    // nothing that could drift: the answer is exactly zero either way.
    if (threads <= 1) {
        return {0.0, true};
    }

    if (auto const profiled = profiled_region_cost(threads)) {
        return {*profiled, true};
    }
    return {detail::measure_omp_region_cost_ns(), false};
}

/// SIMD width in doubles implied by the compile-time ISA of this translation
/// unit - the width the library's own (non-rung) code was vectorized at.
int compiled_simd_width_f64() {
#if defined(__AVX512F__)
    return 8;
#elif defined(__AVX__) || defined(__AVX2__)
    return 4;
#else
    // SSE2 (x86-64 baseline), NEON (ARM), or unknown: 128-bit / 2 doubles.
    return 2;
#endif
}

/// SIMD width in doubles of the rung the kernels dispatch to, so it follows `--einsums:simd:arch`.
int runtime_simd_width_f64() {
    return stripes::vector_bits(stripes::selected_arch()) / 64;
}

} // namespace

CpuInfo const &cpu_info() {
    static CpuInfo const info = []() {
        CpuInfo i;
        i.simd_width_f64          = runtime_simd_width_f64();
        i.simd_width_f32          = 2 * i.simd_width_f64;
        i.compiled_simd_width_f64 = compiled_simd_width_f64();
        auto const cs             = detect_cache_sizes();
        i.cache.l1                = cs.l1;
        i.cache.l2                = cs.l2;
        i.cache.l3                = cs.l3;
        i.omp_region_cost_ns      = resolve_omp_region_cost_ns().value;
        return i;
    }();
    return info;
}

namespace {

/// The resolved cost for the current team size, memoized per size. Not frozen at first use: the
/// team size changes during a process (importing psi4 clamps OpenMP to one thread for a while).
ResolvedRegionCost const &region_cost_for_current_team() {
    static std::mutex                        memo_mutex;
    static std::map<int, ResolvedRegionCost> memo;

#ifdef _OPENMP
    int const threads = omp_get_max_threads();
#else
    int const threads = 1;
#endif

    std::lock_guard<std::mutex> const guard(memo_mutex);
    auto const                        it = memo.find(threads);
    if (it != memo.end()) {
        return it->second;
    }
    return memo.emplace(threads, resolve_omp_region_cost_ns()).first->second;
}

} // namespace

double omp_region_cost_ns() {
    return region_cost_for_current_team().value;
}

bool region_cost_is_calibrated() {
    return region_cost_for_current_team().calibrated;
}

std::size_t omp_min_parallel_elements() {
    // Derived per call, since the team size can change. The override is for same-binary A/B runs;
    // zero means always parallelize.
    if (auto const pinned = config::get(option::HardwareOmpMinParallelElements); pinned >= 0) {
        return static_cast<std::size_t>(pinned);
    }
    // At a conservative one element per nanosecond, the break-even count is the cost in ns.
    return static_cast<std::size_t>(omp_region_cost_ns());
}

std::int64_t omp_min_parallel_flops() {
    // A pin beats the derivation below, so the threshold can be swept from ONE
    // binary: zero always parallelizes, a huge value never does.
    if (auto const pinned = config::get(option::HardwareOmpMinParallelFlops); pinned >= 0) {
        return pinned;
    }

    // The break-even scales with the region cost; this multiplier (flops per ns) is a calibration,
    // not a flop rate. Measured end to end on two tiled CCSD residuals, same binary:
    //
    //   threshold      26 spin-orb     50 spin-orb
    //   26k            3.9-4.1 ms      66.9-68.7 ms
    //   300k           4.0-4.1 ms      33.6-34.3 ms
    //   835k           4.2-4.3 ms      33.5-33.6 ms
    //
    // 12 lands on ~300k. Measured on arm64 (M4 Pro, 10 threads) only, which is why the rate is
    // an option that einsums-tune measures on the machine at hand.
    return static_cast<std::int64_t>(omp_region_cost_ns() * config::get(option::HardwareOmpBreakEvenFlopsPerNs));
}

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void set_num_threads(int nthreads) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif
}

EINSUMS_NAMESPACE_END(hardware)
