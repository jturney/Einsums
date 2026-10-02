//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Hardware/Options.hpp>

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

namespace {

/// Detect CPU cache sizes in bytes, keeping CacheSizes' conservative defaults
/// for any level detection cannot read.
CacheSizes detect_cache_sizes() {
    CacheSizes cs;

    // Diagnostic override for same-binary A/B runs. A level left at zero keeps the detected value.
    auto const apply_override = [&cs]() {
        if (auto const l1 = config::get(option::HardwareL1CacheSize); l1 > 0) {
            cs.l1 = l1;
        }
        if (auto const l2 = config::get(option::HardwareL2CacheSize); l2 > 0) {
            cs.l2 = l2;
        }
        if (auto const l3 = config::get(option::HardwareL3CacheSize); l3 > 0) {
            cs.l3 = l3;
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

    apply_override();
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

/// This machine's name, reduced to something safe in a filename.
std::string host_tag() {
    char raw[256] = {};
#if defined(_WIN32)
    DWORD size = sizeof(raw);
    if (GetComputerNameA(raw, &size) == 0) {
        return "unknown";
    }
#else
    if (gethostname(raw, sizeof(raw) - 1) != 0) {
        return "unknown";
    }
#endif
    std::string tag(raw);
    if (tag.empty()) {
        return "unknown";
    }
    for (char &c : tag) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
            c = '_';
        }
    }
    return tag;
}

/// The calibration file: option::HardwareCalibration, else one under the cache directory keyed by
/// host, so a shared home directory cannot mix machines.
std::filesystem::path calibration_file() {
    if (auto const file = config::get(option::HardwareCalibration); !file.empty()) {
        return std::filesystem::path(file);
    }
    std::filesystem::path const dir = cache_directory();
    if (dir.empty()) {
        return {};
    }
    return dir / ("hardware-calibration-v1-" + host_tag() + ".txt");
}

/// One ``omp_region_cost_ns <threads> <value>`` entry, if the file has one. Plain lines, not JSON:
/// no parser is available at this level, and people read and diff the file.
std::optional<double> read_calibrated_region_cost(std::filesystem::path const &path, int threads) {
    if (path.empty()) {
        return std::nullopt;
    }
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        std::string        key;
        int                entry_threads = 0;
        double             value         = 0.0;
        if (!(fields >> key >> entry_threads >> value)) {
            continue;
        }
        if (key != "omp_region_cost_ns" || entry_threads != threads) {
            continue;
        }
        // A negative, infinite or absurd number means a truncated or hand-edited
        // file. Measuring again is always safe, so never trust one through.
        if (!(value >= 0.0) || value > 1e9) {
            return std::nullopt;
        }
        return value;
    }
    return std::nullopt;
}

/// The region cost: pinned, else calibrated, else measured now. A per-process measurement drifts
/// by tens of percent, which is fine for a threshold but not for ranking options against it. The
/// library never caches its own measurement: one taken on a busy machine looks like a slow machine.
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

    if (auto const calibrated = read_calibrated_region_cost(calibration_file(), threads)) {
        return {*calibrated, true};
    }
    return {measure_omp_region_cost_ns(), false};
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

std::string default_calibration_path() {
    return calibration_file().string();
}

bool write_calibration(std::string const &path, std::string *error) {
    std::filesystem::path const target = path.empty() ? calibration_file() : std::filesystem::path(path);
    if (target.empty()) {
        if (error != nullptr) {
            *error = "no calibration path given and no usable cache directory";
        }
        return false;
    }

#ifdef _OPENMP
    int const max_threads = omp_get_max_threads();
#else
    int const max_threads = 1;
#endif

    // Every team size, since the cost depends on the team. A few milliseconds each.
    std::string body;
    body += "# einsums hardware calibration, format 1\n";
    body += "# host " + host_tag() + "\n";
    body += "# Regenerate with calibrate_hardware. Delete to fall back to measuring\n";
    body += "# per process. Read by einsums::hardware::omp_region_cost_ns().\n";
    for (int t = 1; t <= max_threads; ++t) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        double const value = measure_omp_region_cost_ns();
        body += "omp_region_cost_ns " + std::to_string(t) + " " + std::to_string(value) + "\n";
    }
#ifdef _OPENMP
    omp_set_num_threads(max_threads);
#endif

    std::error_code ec;
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            if (error != nullptr) {
                *error = "could not create " + target.parent_path().string() + ": " + ec.message();
            }
            return false;
        }
    }
    // Temporary then rename, so a concurrent reader never sees half a file.
    std::filesystem::path const tmp = target.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            if (error != nullptr) {
                *error = "could not open " + tmp.string() + " for writing";
            }
            return false;
        }
        out << body;
        if (!out) {
            if (error != nullptr) {
                *error = "could not write " + tmp.string();
            }
            return false;
        }
    }
    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        if (error != nullptr) {
            *error = "could not rename into " + target.string();
        }
        return false;
    }
    return true;
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
    // 12 lands on ~300k. Measured on arm64 (M4 Pro, 10 threads) only; x86 wants the same sweep.
    constexpr double kBreakEvenFlopsPerNs = 12.0;
    return static_cast<std::int64_t>(omp_region_cost_ns() * kBreakEvenFlopsPerNs);
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
