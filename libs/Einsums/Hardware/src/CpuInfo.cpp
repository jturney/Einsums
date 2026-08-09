//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
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
    // performance cluster. Same rule ComputeGraph's CostModel already used;
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

/// Where a measured constant may be remembered between runs, or "" if nowhere.
///
/// ``EINSUMS_CACHE_DIR`` first so a test or a CI job can point this somewhere
/// disposable, then the platform's own cache location. Never a fatal condition:
/// a machine with nowhere to write simply measures every time, which is what
/// happened before there was a cache at all.
std::filesystem::path cache_directory() {
    if (char const *env = std::getenv("EINSUMS_CACHE_DIR"); env != nullptr && *env != '\0') {
        return std::filesystem::path(env);
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

/// The calibration file this machine's measurements are read from.
///
/// ``EINSUMS_HARDWARE_CALIBRATION`` first, mirroring the contract
/// ``EINSUMS_HARDWARE_PROFILE`` already has for the ComputeGraph cost model:
/// an explicit file, missing is fine, never load-bearing. Otherwise a default
/// under the cache directory, keyed by host so a shared home directory cannot
/// hand one machine's fork/join cost to another.
std::filesystem::path calibration_file() {
    if (char const *env = std::getenv("EINSUMS_HARDWARE_CALIBRATION"); env != nullptr && *env != '\0') {
        return std::filesystem::path(env);
    }
    std::filesystem::path const dir = cache_directory();
    if (dir.empty()) {
        return {};
    }
    return dir / ("hardware-calibration-v1-" + host_tag() + ".txt");
}

/// One ``omp_region_cost_ns <threads> <value>`` entry, if the file has one.
///
/// Deliberately a line-oriented key/value format rather than JSON. This module
/// sits below the one that owns a JSON parser, and a calibration file a person
/// is expected to read, diff and delete is better off in a format that needs no
/// parser at all.
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

/// The region cost: pinned, else calibrated, else measured here and now.
///
/// Measured per process this drifts by tens of percent with whatever else the
/// machine is doing at startup. That is harmless for a threshold - a few percent
/// either way does not move "is this loop worth a team" - and not harmless for a
/// chooser that ranks discrete options against the rate, which is what the DLPNO
/// example does when it weighs padded elements against batched calls.
///
/// So a calibrated value wins when there is one, and it gets there because
/// somebody ran ``calibrate_hardware``. Nothing in this library writes that file.
/// That is the whole point: a value the library caches for itself pins the first
/// measurement it happens to take, and a measurement taken while the machine was
/// busy is indistinguishable from a slower machine. A file somebody chose to
/// generate can be regenerated, inspected, diffed and deleted, and its staleness
/// is a question with an obvious answer rather than a silent one.
double resolve_omp_region_cost_ns() {
    // An explicit pin beats everything, so a benchmark can hold the rate fixed
    // across machines without touching any file.
    if (char const *env = std::getenv("EINSUMS_OMP_REGION_COST_NS"); env != nullptr && *env != '\0') {
        errno               = 0;
        char        *end    = nullptr;
        double const pinned = std::strtod(env, &end);
        if (errno == 0 && end != env && pinned >= 0.0) {
            return pinned;
        }
    }

#ifdef _OPENMP
    int const threads = omp_get_max_threads();
#else
    int const threads = 1;
#endif
    // A single thread enters no region, so there is nothing to calibrate.
    if (threads <= 1) {
        return 0.0;
    }

    if (auto const calibrated = read_calibrated_region_cost(calibration_file(), threads)) {
        return *calibrated;
    }
    return measure_omp_region_cost_ns();
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
        i.omp_region_cost_ns = resolve_omp_region_cost_ns();
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

    // Every team size the process could later run at, because the cost is a
    // function of the team and a run at four threads must not be handed the
    // ten-thread number. Each entry is a few milliseconds.
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

double omp_region_cost_ns() {
    // Resolved against the CURRENT team size, not frozen at first use. The cost
    // is a function of the team, and the team is not fixed for the life of a
    // process: importing psi4 clamps process-wide OpenMP to one thread and the
    // caller restores it afterwards, so anything that touched this in between -
    // any elementwise kernel consulting omp_min_parallel_elements will do it -
    // used to freeze the serial answer of zero and hand it to every later
    // caller. The DLPNO bucket chooser then saw no per-call cost on ten threads
    // and picked the finest bucketing available, which is the worst one there.
    //
    // Memoized per team size, so the measurement still happens at most once for
    // each, and a calibrated file makes it happen none.
    static std::mutex            memo_mutex;
    static std::map<int, double> memo;

#ifdef _OPENMP
    int const threads = omp_get_max_threads();
#else
    int const threads     = 1;
#endif

    std::lock_guard<std::mutex> const guard(memo_mutex);
    if (auto const it = memo.find(threads); it != memo.end()) {
        return it->second;
    }
    double const value = resolve_omp_region_cost_ns();
    memo.emplace(threads, value);
    return value;
}

std::size_t omp_min_parallel_elements() {
    // Derived per call for the same reason omp_region_cost_ns is: a threshold
    // frozen while OpenMP was clamped to one thread is zero forever after.
    auto const compute = []() -> std::size_t {
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
    };
    return compute();
}

std::int64_t omp_min_parallel_flops() {
    auto const compute = []() -> std::int64_t {
        // Work is worth a parallel region once it takes longer than entering one,
        // so the break-even scales with the measured region cost. This constant is
        // what that cost is multiplied by, in flops per nanosecond, and it is a
        // CALIBRATION rather than a flop rate - see below.
        //
        // It had been 1.0, on the reasoning that the smallest contractions manage
        // about 1 GFLOP/s. That rate is real but it is the rate of work far below
        // the break-even, which never decides anything; using it dragged the
        // threshold down to a few tens of KFLOP and handed a region to everything.
        // A tiled CCSD residual at 50 spin orbitals expands to 3751 contractions
        // of ~295 KFLOP each, and forking for every one of them cost 71 ms of that
        // replay's 72 ms einsum time, against 32 ms with the regions declined.
        //
        // The rate achieved AT the break-even is what a straight calculation would
        // want, and BenchmarkParallelGate measures it on a `ijab <- ijcd ; cdab`
        // ladder:
        //
        //   flops     8k    295k    524k    2.7M     13M
        //   GFLOP/s  4.2    34.3    23.8    60.2    61.1
        //
        // But ~32 over-excludes. Measured end to end on two tiled CCSD residuals,
        // same binary, threshold forced by the environment override:
        //
        //   threshold      26 spin-orb     50 spin-orb
        //   26k (old)      3.9-4.1 ms      66.9-68.7 ms
        //   300k           4.0-4.1 ms      33.6-34.3 ms
        //   835k (=32x)    4.2-4.3 ms      33.5-33.6 ms
        //
        // 300k keeps the whole win on the large residual and costs the small one
        // nothing, where 835k costs it ~6% for no further gain. The gap is that a
        // region does not cost the full isolated figure when it is entered from a
        // stream that keeps the team hot, so the naive break-even is too
        // conservative. 12 is the multiplier that lands on the measured optimum;
        // it is not a claim about achieved GFLOP/s.
        //
        // Measured on arm64 (Apple M4 Pro, 10 threads) only - x86 wants the same
        // sweep before this is trusted there.
        constexpr double kBreakEvenFlopsPerNs = 12.0;
        auto             value                = static_cast<std::int64_t>(omp_region_cost_ns() * kBreakEvenFlopsPerNs);

        if (char const *env = std::getenv("EINSUMS_PACKED_MIN_PARALLEL_FLOPS"); env != nullptr) {
            errno                  = 0;
            char           *end    = nullptr;
            long long const parsed = std::strtoll(env, &end, 10);
            if (errno == 0 && end != env && parsed >= 0) {
                value = static_cast<std::int64_t>(parsed);
            }
        }
        return value;
    };
    return compute();
}

EINSUMS_NAMESPACE_END(hardware)
