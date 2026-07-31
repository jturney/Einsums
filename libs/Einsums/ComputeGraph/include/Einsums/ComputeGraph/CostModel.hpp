//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/Node.hpp> // Target enum

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace einsums::compute_graph {

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

    [[nodiscard]] double estimate_gemm_time_us(size_t M, size_t N, size_t K) const {
        double const flops  = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double const gflops = estimate_gemm_gflops(M, N, K);
        return flops / (gflops * 1e3) + kernel_launch_overhead_us;
    }

    [[nodiscard]] double estimate_memory_time_us(size_t bytes) const {
        double const bw = (device_type == DeviceType::GPU && device_bandwidth_gbps > 0) ? device_bandwidth_gbps : mem_bandwidth_gbps;
        return static_cast<double>(bytes) / (bw * 1e3);
    }

    [[nodiscard]] double estimate_total_gemm_time_us(size_t M, size_t N, size_t K, size_t element_size) const {
        double const compute_us  = estimate_gemm_time_us(M, N, K);
        size_t const read_bytes  = (M * K + K * N) * element_size;
        size_t const write_bytes = M * N * element_size;
        double const memory_us   = estimate_memory_time_us(read_bytes + write_bytes);
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
    [[nodiscard]] double estimate_permute_time_us(size_t bytes, size_t rank) const {
        double const gbps = estimate_permute_gbps(bytes, rank);
        if (gbps <= 0.0) {
            return kernel_launch_overhead_us;
        }
        return 2.0 * static_cast<double>(bytes) / (gbps * 1e3) + kernel_launch_overhead_us;
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

    /// Estimate GEMM time on a specific target.
    [[nodiscard]] double estimate_gemm_time_us(size_t M, size_t N, size_t K, Target target) const {
        return device(target).estimate_gemm_time_us(M, N, K);
    }

    /// Estimate memory transfer time on a specific target.
    [[nodiscard]] double estimate_memory_time_us(size_t bytes, Target target) const {
        return device(target).estimate_memory_time_us(bytes);
    }

    /// Estimate the time to permute a tensor of @p bytes at @p rank.
    [[nodiscard]] double estimate_permute_time_us(size_t bytes, size_t rank, Target target) const {
        return device(target).estimate_permute_time_us(bytes, rank);
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

} // namespace einsums::compute_graph
