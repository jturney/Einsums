//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/Node.hpp> // Target enum
#include <Einsums/ComputeGraphTypes/EnumNames.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Kernel families the thread axis distinguishes.
 *
 * A family is a group of kernels that scale with thread count the same way, so
 * one measured curve prices all of them. The split is by how the work is
 * parallelized, not by what it computes: @ref GemmLarge threads inside the
 * vendor BLAS call, @ref BatchedGemm threads across independent BLAS calls
 * while each one stays serial, @ref Permute threads inside HPTT, and
 * @ref Elementwise threads an OpenMP loop over elements. @ref GemmSmall is the
 * route that never threads at all, and is a family rather than a size class
 * because the kernel is different code, not the same code on less data.
 */
enum class KernelFamily : std::uint8_t {
    GemmSmall = 0, ///< Small-GEMM route: microkernel, serial by construction
    GemmLarge,     ///< Vendor BLAS GEMM, threaded internally
    BatchedGemm,   ///< Many independent GEMMs, threaded across entries
    Permute,       ///< Tensor transpose (HPTT)
    Elementwise    ///< axpy/scale/copy-shaped loops over elements
};

/// Number of entries in @ref KernelFamily.
inline constexpr std::size_t kKernelFamilyCount = 5;

/**
 * @brief Working-set size classes, cut at the measured cache levels.
 *
 * The boundaries come from @ref DeviceProfile::caches, the same calibration
 * that prices the levels, so a size class means "resident in this level" on
 * the machine that was measured rather than a number somebody picked.
 */
enum class SizeClass : std::uint8_t {
    L1Resident = 0, ///< Working set fits in L1
    L2Resident,     ///< Fits in L2 but not L1
    LLCResident,    ///< Fits in the last-level cache but not L2
    Streaming       ///< Larger than the last-level cache
};

/// Number of entries in @ref SizeClass.
inline constexpr std::size_t kSizeClassCount = 4;

/// The one @ref KernelFamily name table; both directions read it.
inline constexpr EnumNames kKernelFamilyNames{std::array<std::pair<KernelFamily, std::string_view>, kKernelFamilyCount>{{
                                                  {KernelFamily::GemmSmall, "gemm_small"},
                                                  {KernelFamily::GemmLarge, "gemm_large"},
                                                  {KernelFamily::BatchedGemm, "batched_gemm"},
                                                  {KernelFamily::Permute, "permute"},
                                                  {KernelFamily::Elementwise, "elementwise"},
                                              }},
                                              "gemm_large"};

/// The one @ref SizeClass name table; both directions read it.
inline constexpr EnumNames kSizeClassNames{std::array<std::pair<SizeClass, std::string_view>, kSizeClassCount>{{
                                               {SizeClass::L1Resident, "l1"},
                                               {SizeClass::L2Resident, "l2"},
                                               {SizeClass::LLCResident, "llc"},
                                               {SizeClass::Streaming, "streaming"},
                                           }},
                                           "streaming"};

[[nodiscard]] inline std::string_view to_string(KernelFamily family) {
    return kKernelFamilyNames.name(family);
}

[[nodiscard]] inline std::string_view to_string(SizeClass size_class) {
    return kSizeClassNames.name(size_class);
}

/// Parse a @ref KernelFamily name. Returns false and leaves @p out alone when
/// the name is not one of the five; an unknown family in a profile file is
/// dropped rather than silently priced as something else.
[[nodiscard]] inline bool kernel_family_from_string(std::string_view name, KernelFamily &out) {
    auto const family = kKernelFamilyNames.from_name(name);
    if (!family) {
        return false;
    }
    out = *family;
    return true;
}

/// Parse a @ref SizeClass name. Same contract as @ref kernel_family_from_string.
[[nodiscard]] inline bool size_class_from_string(std::string_view name, SizeClass &out) {
    auto const size_class = kSizeClassNames.from_name(name);
    if (!size_class) {
        return false;
    }
    out = *size_class;
    return true;
}

/**
 * @brief Measured speedup against thread count for one kernel family at one size class.
 *
 * `speedup[i]` is `t(1) / t(widths[i])`, so `widths[0] == 1` and
 * `speedup[0] == 1` by construction. The width rungs are powers of two up to
 * `P`, plus `P` itself when it is not a power of two (a 10-core machine gets
 * 1, 2, 4, 8, 10): calibration cost scales with the rung count, and the
 * executor resizes hot teams on every distinct width.
 *
 * The rungs are stored rather than implied because a profile is written on one
 * machine and read on another, and a curve whose rungs were guessed from the
 * reader's core count would be read off at the wrong widths.
 */
struct EfficiencyCurve {
    KernelFamily               family{KernelFamily::GemmLarge};
    SizeClass                  size_class{SizeClass::Streaming};
    std::vector<std::uint32_t> widths;  ///< Ascending, first entry 1
    std::vector<double>        speedup; ///< Parallel to @ref widths

    /// A curve is usable when it has at least one rung and the two vectors agree.
    [[nodiscard]] bool valid() const { return !widths.empty() && widths.size() == speedup.size(); }

    /**
     * @brief Speedup at @p width, interpolated linearly in log2(width).
     *
     * Widths between rungs interpolate; widths outside the measured range snap
     * to the nearest end rung, so asking for more threads than were calibrated
     * predicts the widest measured behaviour rather than extrapolating. A
     * width of 0 is read as 1.
     *
     * Never returns zero or a negative number, whatever the file says, because
     * callers divide by it.
     */
    [[nodiscard]] double speedup_at(unsigned width) const {
        constexpr double kFloor = 0.01;
        if (!valid()) {
            return 1.0;
        }
        auto const w = static_cast<double>(std::max(1u, width));
        if (w <= static_cast<double>(widths.front())) {
            return std::max(kFloor, speedup.front());
        }
        if (w >= static_cast<double>(widths.back())) {
            return std::max(kFloor, speedup.back());
        }
        for (std::size_t idx = 1; idx < widths.size(); idx++) {
            auto const hi = static_cast<double>(widths[idx]);
            if (w > hi) {
                continue;
            }
            auto const lo = static_cast<double>(widths[idx - 1]);
            // Equal rungs would divide by zero; a duplicated rung snaps low.
            double const span = std::log2(hi) - std::log2(lo);
            double const t    = (span > 0.0) ? (std::log2(w) - std::log2(lo)) / span : 0.0;
            return std::max(kFloor, speedup[idx - 1] + t * (speedup[idx] - speedup[idx - 1]));
        }
        return std::max(kFloor, speedup.back());
    }
};

/// Width rungs for a machine with @p max_threads: powers of two up to it, plus
/// it. Always non-empty and always starts at 1.
[[nodiscard]] inline std::vector<unsigned> width_rungs_for(unsigned max_threads) {
    std::vector<unsigned> rungs;
    unsigned const        p = std::max(1u, max_threads);
    for (unsigned w = 1; w <= p; w *= 2) {
        rungs.push_back(w);
    }
    if (rungs.back() != p) {
        rungs.push_back(p);
    }
    return rungs;
}

/// A single measured GEMM efficiency data point.
struct GemmEfficiencyPoint {
    size_t M{0}, N{0}, K{0};
    double gflops{0};
};

/// A single measured permute (tensor transpose) data point.
///
/// `gbps` is the achieved TRAFFIC rate, counting the source read and the
/// destination write, so it is directly comparable with @ref
/// DeviceProfile::mem_bandwidth_gbps from the copy benchmark.
struct PermuteEfficiencyPoint {
    size_t bytes{0}; ///< Size of the permuted tensor (one side, not the traffic)
    size_t rank{0};  ///< Rank of the permuted tensor
    double gbps{0};  ///< Achieved (read + write) bandwidth
};

/// Cache level description.
struct CacheLevel {
    size_t size_bytes{0};
    double bandwidth_gbps{0.0};
    double latency_ns{0.0};
};

/// Device type for profile classification.
enum class DeviceType : std::uint8_t { CPU, GPU };

/**
 * @brief Performance profile for a single device (CPU or GPU).
 *
 * Contains measured or estimated performance characteristics used by
 * the ContractionPlanning pass to estimate execution time.
 */
struct DeviceProfile {
    std::string              name;                         ///< e.g., "Apple M4 Pro", "NVIDIA A100"
    std::string              source{"default"};            ///< "default", "database", "calibrated"
    DeviceType               device_type{DeviceType::CPU}; ///< CPU or GPU
    std::string              brand_family;                 ///< Normalized key, e.g., "apple_m4_pro"
    std::vector<std::string> match_patterns;               ///< Substrings for brand string matching

    double peak_gflops_fp64{50.0};
    double peak_gflops_fp32{100.0};
    double mem_bandwidth_gbps{40.0};
    double kernel_launch_overhead_us{0.5};
    double alloc_overhead_us{2.0};
    double device_bandwidth_gbps{0.0};
    double pcie_bandwidth_gbps{0.0};
    double gpu_launch_latency_us{0.0};

    // ── Network (populated when MPI is active) ─────────────────────────────
    double inter_node_bandwidth_gbps{0.0}; ///< Network bandwidth (e.g., 25 Gbps InfiniBand)
    double inter_node_latency_us{1.0};     ///< Network latency (e.g., 1-5 us for IB)
    double nccl_bandwidth_gbps{0.0};       ///< Measured NCCL throughput for GPU-direct

    std::vector<CacheLevel>             caches;
    std::vector<GemmEfficiencyPoint>    gemm_efficiency;
    std::vector<PermuteEfficiencyPoint> permute_efficiency;

    /// The `P` the efficiency curves were measured at. Zero means the profile
    /// was never swept, and then no width is clamped against it.
    unsigned max_threads{0};

    /// Measured `e(w)` per kernel family and size class. Empty is the normal
    /// state for a database profile; see @ref default_parallel_speedup for
    /// what that costs.
    std::vector<EfficiencyCurve> thread_efficiency;

    /// Estimate GEMM throughput using nearest-neighbor interpolation.
    [[nodiscard]] double estimate_gemm_gflops(size_t M, size_t N, size_t K) const {
        if (gemm_efficiency.empty()) {
            double const volume     = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
            double const efficiency = std::min(0.85, 0.1 + 0.75 * (1.0 - std::exp(-volume / 1e6)));
            return peak_gflops_fp64 * efficiency;
        }
        auto const target_vol  = static_cast<double>(M * N * K);
        double     best_dist   = 1e30;
        double     best_gflops = peak_gflops_fp64 * 0.5;
        for (auto const &pt : gemm_efficiency) {
            auto const   pt_vol = static_cast<double>(pt.M * pt.N * pt.K);
            double const dist   = std::abs(std::log(target_vol + 1.0) - std::log(pt_vol + 1.0));
            if (dist < best_dist) {
                best_dist   = dist;
                best_gflops = pt.gflops;
            }
        }
        return best_gflops;
    }

    /**
     * @brief Estimate GEMM time at @p threads threads.
     *
     * `t(w) = t(1) / s(w) + launch overhead`, with `s(w)` from the family and
     * size class this shape falls in. The launch overhead is not divided: it is
     * paid once whatever the width, and dividing it would make a wide plan look
     * cheaper than it can be. `threads = 1` reproduces the width-1 estimate
     * exactly, so every existing call site is unchanged.
     */
    [[nodiscard]] double estimate_gemm_time_us(size_t M, size_t N, size_t K, unsigned threads = 1) const {
        double const flops  = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double const gflops = estimate_gemm_gflops(M, N, K);
        return flops / (gflops * 1e3 * gemm_parallel_speedup(M, N, K, threads)) + kernel_launch_overhead_us;
    }

    /// Estimate the time to move @p bytes at @p threads threads. Bandwidth
    /// scales with width along the @ref KernelFamily::Elementwise curve, which
    /// is the family whose kernels are pure traffic.
    [[nodiscard]] double estimate_memory_time_us(size_t bytes, unsigned threads = 1) const {
        double const bw = (device_type == DeviceType::GPU && device_bandwidth_gbps > 0) ? device_bandwidth_gbps : mem_bandwidth_gbps;
        return static_cast<double>(bytes) / (bw * 1e3 * parallel_speedup(KernelFamily::Elementwise, bytes, threads));
    }

    [[nodiscard]] double estimate_total_gemm_time_us(size_t M, size_t N, size_t K, size_t element_size, unsigned threads = 1) const {
        double const compute_us  = estimate_gemm_time_us(M, N, K, threads);
        size_t const read_bytes  = (M * K + K * N) * element_size;
        size_t const write_bytes = M * N * element_size;
        double const memory_us   = estimate_memory_time_us(read_bytes + write_bytes, threads);
        return std::max(compute_us, memory_us) + alloc_overhead_us;
    }

    /**
     * @brief Fraction of @ref mem_bandwidth_gbps a permute of this rank is
     *        expected to reach, when no calibration data is available.
     *
     * A transpose moves the same bytes as a copy with one side strided, and the
     * penalty grows with rank because the innermost contiguous run shortens.
     *
     * Calibrate rather than lean on these. An HPTT sweep on one machine (Apple
     * M4, doubles) put the achieved rate between 52% and 104% of the copy
     * figure for working sets up to about 1 MB, and ABOVE it from 8 MB up,
     * where HPTT's threading beats the single-threaded copy benchmark that
     * mem_bandwidth_gbps comes from. The constants below track the small end,
     * which is where a cost model actually has to make a call, and so they
     * over-price a large permute. @c calibrate_hardware replaces them with
     * measured points.
     */
    [[nodiscard]] static double default_permute_efficiency(size_t rank) {
        switch (rank) {
        case 0:
        case 1:
            return 1.0; // no reordering: this is a copy
        case 2:
            return 0.80;
        case 3:
            return 0.75;
        default:
            return 0.60;
        }
    }

    /// Achieved (read + write) bandwidth for permuting a tensor of @p bytes at
    /// @p rank. Calibration points win; nearest neighbour prefers an exact rank
    /// match and then the closest size on a log scale, the same shape of
    /// interpolation @ref estimate_gemm_gflops uses.
    [[nodiscard]] double estimate_permute_gbps(size_t bytes, size_t rank) const {
        if (permute_efficiency.empty()) {
            return mem_bandwidth_gbps * default_permute_efficiency(rank);
        }
        double const target = std::log(static_cast<double>(bytes) + 1.0);
        double       best   = mem_bandwidth_gbps * default_permute_efficiency(rank);
        double       best_d = 1e30;
        for (auto const &pt : permute_efficiency) {
            // A rank mismatch is a bigger error than a size mismatch, so it
            // costs more than the whole plausible range of the log-size term.
            double const d = std::abs(std::log(static_cast<double>(pt.bytes) + 1.0) - target) +
                             100.0 * std::abs(static_cast<double>(pt.rank) - static_cast<double>(rank));
            if (d < best_d) {
                best_d = d;
                best   = pt.gbps;
            }
        }
        return best;
    }

    /**
     * @brief Estimate the time to permute a tensor of @p bytes at @p rank.
     *
     * Bandwidth-bound: the source is read once and the destination written
     * once, at the rate @ref estimate_permute_gbps predicts for that shape.
     * This is what prices a transpose against the contraction that would
     * otherwise absorb it.
     */
    [[nodiscard]] double estimate_permute_time_us(size_t bytes, size_t rank, unsigned threads = 1) const {
        double const gbps = estimate_permute_gbps(bytes, rank);
        if (gbps <= 0.0) {
            return kernel_launch_overhead_us;
        }
        return 2.0 * static_cast<double>(bytes) / (gbps * 1e3 * parallel_speedup(KernelFamily::Permute, bytes, threads)) +
               kernel_launch_overhead_us;
    }

    // ── Thread axis ────────────────────────────────────────────────────────

    /// Size of cache level @p level (0-based), or @p fallback when the profile
    /// does not describe that many levels.
    [[nodiscard]] size_t cache_bytes(size_t level, size_t fallback) const {
        if (level < caches.size() && caches[level].size_bytes > 0) {
            return caches[level].size_bytes;
        }
        // A machine with fewer levels than asked for prices the deeper class at
        // its own last level rather than at a number from nowhere.
        if (!caches.empty() && caches.back().size_bytes > 0 && level >= caches.size()) {
            return std::max(fallback, caches.back().size_bytes);
        }
        return fallback;
    }

    /// Which @ref SizeClass a working set of @p bytes falls in, cut at this
    /// profile's cache levels. Boundaries are forced strictly increasing so a
    /// machine that reports an L2 no bigger than its L1 still yields four
    /// distinct classes.
    [[nodiscard]] SizeClass size_class_for_bytes(size_t bytes) const {
        size_t const l1  = cache_bytes(0, 32UL * 1024);
        size_t const l2  = std::max(l1 + 1, cache_bytes(1, 1024UL * 1024));
        size_t const llc = std::max(l2 + 1, cache_bytes(2, 8UL * 1024 * 1024));
        if (bytes <= l1) {
            return SizeClass::L1Resident;
        }
        if (bytes <= l2) {
            return SizeClass::L2Resident;
        }
        if (bytes <= llc) {
            return SizeClass::LLCResident;
        }
        return SizeClass::Streaming;
    }

    /// Flop count below which a GEMM takes the small-kernel route. Set at a
    /// 64-cubed GEMM, which is where the vendor's own threading gate sits on
    /// the machines we calibrate; the sweep measures both sides of it.
    static constexpr double kGemmSmallFlops = 2.0 * 64.0 * 64.0 * 64.0;

    /// Which GEMM family a shape belongs to.
    [[nodiscard]] static KernelFamily gemm_family(size_t M, size_t N, size_t K) {
        double const flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        return flops < kGemmSmallFlops ? KernelFamily::GemmSmall : KernelFamily::GemmLarge;
    }

    /**
     * @brief Parallel fraction of the default (uncalibrated) model.
     *
     * The default curve is Amdahl's law, `s(w) = 1 / ((1-p) + p/w)`, with `p`
     * the product of a per-family ceiling and a per-size-class factor. Two
     * properties are deliberate and are what a planner leans on:
     *
     * - `s(w)` is non-decreasing in `w` and `e(w) = s(w)/w` is non-increasing,
     *   so widening never looks free and the marginal gain shrinks.
     * - An L1-resident working set has `p = 0`, so `s(w) = 1` and
     *   `e(w) = 1/w` exactly: below the threshold, width buys nothing and a
     *   planner leaves the node narrow without needing a special case. The
     *   small-GEMM family is `p = 0` at every size for the same reason.
     *
     * The family ceilings are ordered by how much of the kernel is loop and
     * how much is fixed cost: a large BLAS GEMM parallelizes nearly perfectly,
     * a batch of them slightly less, HPTT less again, and a bandwidth-bound
     * elementwise loop least because it saturates the bus before it saturates
     * the cores. All of them are conservative placeholders except the batched
     * one, which is fitted to a measured sweep; see the case below. Calibrate
     * rather than lean on them: @ref measure_thread_efficiency replaces them
     * wholesale.
     */
    [[nodiscard]] static double default_parallel_fraction(KernelFamily family, SizeClass size_class) {
        double base = 0.0;
        switch (family) {
        case KernelFamily::GemmSmall:
            base = 0.0;
            break;
        case KernelFamily::GemmLarge:
            base = 0.95;
            break;
        case KernelFamily::BatchedGemm:
            // The one ceiling here that is a FIT rather than a placeholder. The others are ordered
            // guesses; this one is least squares against a measured sweep of the DLPNO-(T)
            // residual's grouped batched GEMMs - 125 nodes over 326 triplets, on a 10-core machine
            // - whose efficiencies came out 97, 88, 70, 52 and 44 percent at widths 1, 2, 4, 8 and
            // 10. Amdahl at p = 0.862 reproduces all five to within a point of efficiency, so the
            // saturating form the model already uses is the right shape and only the constant was
            // wrong: at 0.90 it promised 5.26x at width 10 where the kernel delivers 4.41x, and a
            // planner that over-credits width buys the machine at a discount and then spends it.
            //
            // One family on one machine, so it is a better constant and not a calibration; @ref
            // measure_thread_efficiency still replaces the whole table with measurements when a
            // caller sweeps.
            base = 0.862;
            break;
        case KernelFamily::Permute:
            base = 0.80;
            break;
        case KernelFamily::Elementwise:
            base = 0.70;
            break;
        }
        double scale = 0.0;
        switch (size_class) {
        case SizeClass::L1Resident:
            scale = 0.0;
            break;
        case SizeClass::L2Resident:
            scale = 0.35;
            break;
        case SizeClass::LLCResident:
            scale = 0.75;
            break;
        case SizeClass::Streaming:
            scale = 1.0;
            break;
        }
        return base * scale;
    }

    /// Speedup the default model predicts. See @ref default_parallel_fraction.
    [[nodiscard]] static double default_parallel_speedup(KernelFamily family, SizeClass size_class, unsigned width) {
        auto const   w     = static_cast<double>(std::max(1u, width));
        double const p     = default_parallel_fraction(family, size_class);
        double const denom = (1.0 - p) + p / w;
        return denom > 0.0 ? 1.0 / denom : w;
    }

    /// Clamp a requested width to something this profile can speak about:
    /// never below 1, never above the `P` the curves were measured at.
    [[nodiscard]] unsigned clamp_width(unsigned width) const {
        unsigned const w = std::max(1u, width);
        return max_threads > 0 ? std::min(w, max_threads) : w;
    }

    /// The curve for a cell, or nullptr. An exact cell wins; failing that the
    /// nearest size class within the same family, because a partly swept
    /// profile should still price the cells it did measure.
    [[nodiscard]] EfficiencyCurve const *find_curve(KernelFamily family, SizeClass size_class) const {
        EfficiencyCurve const *best      = nullptr;
        int                    best_dist = 0;
        for (auto const &curve : thread_efficiency) {
            if (curve.family != family || !curve.valid()) {
                continue;
            }
            int const dist = std::abs(static_cast<int>(curve.size_class) - static_cast<int>(size_class));
            if (best == nullptr || dist < best_dist) {
                best      = &curve;
                best_dist = dist;
            }
        }
        return best;
    }

    /// `s(w)`: how much faster this cell runs at @p width than at width 1.
    /// Measured when the cell was swept, the default Amdahl curve otherwise.
    /// Always strictly positive, so callers may divide by it.
    [[nodiscard]] double parallel_speedup(KernelFamily family, SizeClass size_class, unsigned width) const {
        unsigned const w = clamp_width(width);
        if (auto const *curve = find_curve(family, size_class); curve != nullptr) {
            return curve->speedup_at(w);
        }
        return default_parallel_speedup(family, size_class, w);
    }

    /// Same, with the size class derived from a working set of @p bytes.
    [[nodiscard]] double parallel_speedup(KernelFamily family, size_t bytes, unsigned width) const {
        return parallel_speedup(family, size_class_for_bytes(bytes), width);
    }

    /// `e(w) = s(w)/w`: the fraction of the added threads that turns into work.
    /// One at width 1, and collapsing toward `1/w` for work that cannot use
    /// the machine.
    [[nodiscard]] double parallel_efficiency(KernelFamily family, size_t bytes, unsigned width) const {
        unsigned const w = clamp_width(width);
        return parallel_speedup(family, bytes, w) / static_cast<double>(w);
    }

    /// Same, on an explicit size class.
    [[nodiscard]] double parallel_efficiency(KernelFamily family, SizeClass size_class, unsigned width) const {
        unsigned const w = clamp_width(width);
        return parallel_speedup(family, size_class, w) / static_cast<double>(w);
    }

    /// `s(w)` for a GEMM shape: picks the family and the size class for it.
    /// The working set is priced in doubles because the GEMM estimator is
    /// already an fp64 model (@ref peak_gflops_fp64).
    [[nodiscard]] double gemm_parallel_speedup(size_t M, size_t N, size_t K, unsigned width) const {
        size_t const bytes = (M * K + K * N + M * N) * sizeof(double);
        return parallel_speedup(gemm_family(M, N, K), size_class_for_bytes(bytes), width);
    }

    /// The width rungs this profile was calibrated at, or the rungs it would
    /// be calibrated at on a machine of @ref max_threads.
    [[nodiscard]] std::vector<unsigned> width_rungs() const {
        for (auto const &curve : thread_efficiency) {
            if (curve.valid()) {
                std::vector<unsigned> rungs;
                rungs.reserve(curve.widths.size());
                for (auto const w : curve.widths) {
                    rungs.push_back(static_cast<unsigned>(w));
                }
                return rungs;
            }
        }
        return width_rungs_for(max_threads);
    }
};

/**
 * @brief Target-aware cost model built from CPU and GPU device profiles.
 *
 * Every estimate dispatches to the profile for the requested @ref Target, so
 * GPUPlacement and ContractionPlanning compare CPU and GPU alternatives against
 * one consistent model rather than each inventing its own constants.
 *
 * The measured hardware facts this is built from live elsewhere: cache sizes and
 * OpenMP region cost come from @ref einsums::hardware::cpu_info, and the
 * per-device performance table from @ref DeviceProfileDB or a calibration file.
 * What this type adds is the modelling on top - roofline GEMM time, ring
 * allreduce, binomial-tree broadcast.
 */
struct EINSUMS_EXPORT CostModel {
    DeviceProfile cpu;
    DeviceProfile gpu;
    std::string   source{"default"};

    [[nodiscard]] bool has_gpu() const { return !gpu.name.empty(); }

    /// Select the device profile for a given target.
    [[nodiscard]] DeviceProfile const &device(Target target) const { return (target == Target::GPU && has_gpu()) ? gpu : cpu; }

    /// Estimate GEMM time on a specific target, at @p threads threads.
    [[nodiscard]] double estimate_gemm_time_us(size_t M, size_t N, size_t K, Target target, unsigned threads = 1) const {
        return device(target).estimate_gemm_time_us(M, N, K, threads);
    }

    /// Estimate memory transfer time on a specific target.
    [[nodiscard]] double estimate_memory_time_us(size_t bytes, Target target, unsigned threads = 1) const {
        return device(target).estimate_memory_time_us(bytes, threads);
    }

    /// Estimate the time to permute a tensor of @p bytes at @p rank.
    [[nodiscard]] double estimate_permute_time_us(size_t bytes, size_t rank, Target target, unsigned threads = 1) const {
        return device(target).estimate_permute_time_us(bytes, rank, threads);
    }

    /// `s(w)` for a kernel family and working set on a specific target. This is
    /// the primitive a width planner reads: `t(w) = t(1) / s(w)`.
    [[nodiscard]] double parallel_speedup(KernelFamily family, size_t bytes, unsigned width, Target target) const {
        return device(target).parallel_speedup(family, bytes, width);
    }

    /// `e(w) = s(w)/w` for a kernel family and working set on a specific target.
    [[nodiscard]] double parallel_efficiency(KernelFamily family, size_t bytes, unsigned width, Target target) const {
        return device(target).parallel_efficiency(family, bytes, width);
    }

    /// Estimate host-device transfer time (via PCIe or unified memory bus).
    [[nodiscard]] double estimate_transfer_time_us(size_t bytes) const {
        double bw = gpu.pcie_bandwidth_gbps;
        if (bw <= 0)
            bw = cpu.mem_bandwidth_gbps; // Unified memory fallback
        return static_cast<double>(bytes) / (bw * 1e3);
    }

    /// Estimate total GEMM time (roofline) on a specific target.
    [[nodiscard]] double estimate_total_gemm_time_us(size_t M, size_t N, size_t K, size_t element_size, Target target) const {
        return device(target).estimate_total_gemm_time_us(M, N, K, element_size);
    }

    /**
     * @brief What it costs merely to BE a node, before any work happens.
     *
     * The dispatch plus the per-call allocation the executor pays whatever the
     * operation is. It is the figure a lowering weighs when deciding how finely
     * to split work: below it, an operation spends more getting invoked than
     * running.
     */
    [[nodiscard]] double node_overhead_us(Target target) const {
        auto const &d = device(target);
        return d.kernel_launch_overhead_us + d.alloc_overhead_us;
    }

    // ── Distributed communication cost estimation ────────────────────────

    /**
     * @brief Estimate allreduce time using ring algorithm model.
     *
     * Ring allreduce: 2*(p-1)/p * bytes/bandwidth + 2*(p-1) * latency.
     *
     * @param bytes     Total data size in bytes.
     * @param num_ranks Number of MPI ranks participating.
     * @return Estimated time in microseconds.
     */
    [[nodiscard]] double estimate_allreduce_time_us(size_t bytes, int num_ranks) const {
        double bw = cpu.inter_node_bandwidth_gbps;
        if (bw <= 0)
            bw = 1.0; // Conservative fallback
        if (num_ranks <= 1)
            return 0.0;
        double const factor = 2.0 * (static_cast<double>(num_ranks) - 1.0) / static_cast<double>(num_ranks);
        return factor * static_cast<double>(bytes) / (bw * 1e3) + factor * cpu.inter_node_latency_us;
    }

    /// Estimate broadcast time using binomial tree model.
    [[nodiscard]] double estimate_broadcast_time_us(size_t bytes, int num_ranks) const {
        double bw = cpu.inter_node_bandwidth_gbps;
        if (bw <= 0)
            bw = 1.0;
        if (num_ranks <= 1)
            return 0.0;
        return std::log2(static_cast<double>(num_ranks)) * (static_cast<double>(bytes) / (bw * 1e3) + cpu.inter_node_latency_us);
    }

    /// Estimate allgather time using ring algorithm model.
    [[nodiscard]] double estimate_allgather_time_us(size_t bytes_per_rank, int num_ranks) const {
        double bw = cpu.inter_node_bandwidth_gbps;
        if (bw <= 0)
            bw = 1.0;
        if (num_ranks <= 1)
            return 0.0;
        double const factor = (static_cast<double>(num_ranks) - 1.0) / static_cast<double>(num_ranks);
        return factor * static_cast<double>(bytes_per_rank) * static_cast<double>(num_ranks) / (bw * 1e3) +
               factor * cpu.inter_node_latency_us;
    }

    // ── Factory methods ────────────────────────────────────────────────────

    /// Auto-detect hardware and load profile from shipped database.
    [[nodiscard]] static CostModel detect_default();

    /// Load from a JSON file. Returns error if file cannot be read or parsed.
    [[nodiscard]] static expected<CostModel, GraphError> load_json(std::string const &path);

    /// Save to a JSON file. Returns error if file cannot be written.
    [[nodiscard]] expected<void, GraphError> save_json(std::string const &path) const;
};

/**
 * @brief Database of device profiles for known CPUs and GPUs.
 *
 * At runtime, the database matches the detected CPU brand string and GPU
 * device name against stored profiles to select the best match. This avoids
 * requiring every user to run the calibration tool.
 */
class EINSUMS_EXPORT DeviceProfileDB {
  public:
    /// Load the shipped database from a JSON file.
    [[nodiscard]] static expected<DeviceProfileDB, GraphError> load_json(std::string const &path);

    /// Create a database with built-in default entries.
    static DeviceProfileDB load_defaults();

    /// Match the detected CPU brand against database entries.
    [[nodiscard]] DeviceProfile const &match_cpu() const;

    /// Match the detected GPU name against database entries.
    [[nodiscard]] DeviceProfile const &match_gpu() const;

    /// Build a CostModel from the best-matching CPU and GPU entries.
    [[nodiscard]] CostModel build_cost_model() const;

    /// Add or replace a profile entry (matched by brand_family).
    void upsert(DeviceProfile profile);

    /// Save the database to a JSON file.
    [[nodiscard]] expected<void, GraphError> save_json(std::string const &path) const;

    [[nodiscard]] std::vector<DeviceProfile> const &profiles() const { return _profiles; }

    /// Detect the CPU brand string at runtime.
    [[nodiscard]] static std::string detect_cpu_brand();

    /// Detect the GPU device name at runtime.
    [[nodiscard]] static std::string detect_gpu_name();

  private:
    std::vector<DeviceProfile> _profiles;
    DeviceProfile              _fallback_cpu; ///< Generic CPU if no match found
    DeviceProfile              _fallback_gpu; ///< Generic GPU if no match found

    /// Normalize a brand string for matching (lowercase, strip whitespace).
    static std::string normalize(std::string const &s);

    /// Find best-matching profile for a brand string and device type.
    [[nodiscard]] DeviceProfile const *find_best_match(std::string const &brand, DeviceType type) const;
};

/// Knobs for @ref measure_thread_efficiency. The defaults are what
/// `calibrate_hardware` runs; the caps are what keep it to tens of seconds.
struct ThreadSweepOptions {
    size_t   warmup{1};                     ///< Untimed runs before each timed set
    size_t   repeats{3};                    ///< Timed runs per point; the median is kept
    unsigned max_threads{0};                ///< Widest rung; 0 asks the hardware
    size_t   max_bytes{64UL * 1024 * 1024}; ///< Cap on any cell's working set
    size_t   max_gemm_extent{1024};         ///< Cap on a square GEMM's extent
    size_t   batch_entries{64};             ///< Entries in the batched-GEMM cell
};

/**
 * @brief Measure `e(w)` for every kernel family at every size class it is
 *        defined at, on this machine.
 *
 * Size-class working sets come from @p profile's cache levels, so the sweep
 * measures the same boundaries @ref DeviceProfile::size_class_for_bytes reads
 * back. Each cell is timed at every rung of @ref width_rungs_for with the same
 * warmup-and-median discipline the rest of the calibration uses, and the point
 * is recorded as `t(1)/t(w)`.
 *
 * Not every family is measured at every size class: a small-GEMM cell at a
 * streaming working set is not a small GEMM, and a large-GEMM cell that fits in
 * L1 is not a large one. Cells that are not measured resolve through the
 * nearest measured size class in the same family, and then through the default
 * curve.
 *
 * Returns an empty vector when the machine has one thread, since every curve
 * would be the single point `s(1) = 1`.
 */
[[nodiscard]] EINSUMS_EXPORT std::vector<EfficiencyCurve> measure_thread_efficiency(DeviceProfile const      &profile,
                                                                                    ThreadSweepOptions const &options = {});

EINSUMS_NAMESPACE_END(compute_graph)
