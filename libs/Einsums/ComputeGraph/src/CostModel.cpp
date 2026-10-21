//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Hardware/Machine.hpp>
#include <Einsums/Hardware/Profile.hpp>
#include <Einsums/Json/Json.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// ═══════════════════════════════════════════════════════════════════════════════
// DeviceProfileDB: runtime detection
// ═══════════════════════════════════════════════════════════════════════════════

std::string DeviceProfileDB::detect_cpu_brand() {
    return hardware::cpu_brand();
}

std::string DeviceProfileDB::detect_gpu_name() {
    return gpu::device_name();
}

namespace {

/// The CPU data-cache hierarchy, in L1 -> L3 order, taken from the one detector
/// (Einsums_Hardware), so it agrees with PackedGemm. Bandwidth and latency stay
/// unmeasured (zero) - consumers key off size_bytes.
/// The machine the built-in scatter and pack curves were measured on, an Apple M4, by its copy
/// bandwidth in GB/s. Another table's curves are scaled by its own bandwidth against this.
constexpr double kReferenceBandwidthGbps = 120.0;

/// The built-in scatter curve, ns(r) = floor + per_run / r extra nanoseconds per element of C. Set
/// from the CCSD ladder's tile on the M4, one thread: the scatter wrote 0.62 ns per element in runs
/// of 16 and 0.33 ns in runs of 81, of which about 0.1 ns is the write a C in place pays too.
constexpr double kScatterFloorNs  = 0.16;
constexpr double kScatterPerRunNs = 5.8;

/// The built-in pack curve: nothing extra for a long run, about half a nanosecond per element
/// gathered one at a time. A placeholder of the right order; einsums-tune measures the real one.
constexpr double kPackFloorNs  = 0.0;
constexpr double kPackPerRunNs = 0.5;

/// Points of ns(r) = (floor + per_run / r) * scale at run lengths 1, 2, 4, ..., 1024.
std::vector<RunCostPoint> run_cost_curve(double floor, double per_run, double scale) {
    std::vector<RunCostPoint> curve;
    for (size_t run = 1; run <= 1024; run *= 2) {
        curve.push_back({.run = run, .ns_per_element = (floor + per_run / static_cast<double>(run)) * scale});
    }
    return curve;
}

std::vector<CacheLevel> detect_cpu_caches() {
    auto const             &cache = einsums::hardware::cpu_info().cache;
    std::vector<CacheLevel> levels;
    for (std::int64_t const bytes : {cache.l1, cache.l2, cache.l3}) {
        if (bytes > 0) {
            levels.push_back({.size_bytes = static_cast<size_t>(bytes)});
        }
    }
    return levels;
}

} // namespace

std::string DeviceProfileDB::normalize(std::string const &s) {
    std::string result;
    result.reserve(s.size());
    for (char const c : s) {
        if (std::isspace(static_cast<unsigned char>(c)))
            continue;
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

DeviceProfile const *DeviceProfileDB::find_best_match(std::string const &brand, DeviceType type) const {
    std::string const    norm_brand = normalize(brand);
    DeviceProfile const *best       = nullptr;
    size_t               best_len   = 0;

    for (auto const &p : _profiles) {
        if (p.device_type != type)
            continue;
        for (auto const &pattern : p.match_patterns) {
            std::string const norm_pat = normalize(pattern);
            if (norm_brand.find(norm_pat) != std::string::npos && norm_pat.size() > best_len) {
                best     = &p;
                best_len = norm_pat.size();
            }
        }
    }
    return best;
}

DeviceProfile const &DeviceProfileDB::match_cpu() const {
    std::string const brand = detect_cpu_brand();
    auto             *match = find_best_match(brand, DeviceType::CPU);
    return match ? *match : _fallback_cpu;
}

DeviceProfile const &DeviceProfileDB::match_gpu() const {
    std::string const name = detect_gpu_name();
    if (name.empty())
        return _fallback_gpu;
    auto *match = find_best_match(name, DeviceType::GPU);
    return match ? *match : _fallback_gpu;
}

CostModel DeviceProfileDB::build_cost_model() const {
    CostModel p;
    p.cpu    = match_cpu();
    p.gpu    = match_gpu();
    p.source = "database";
    return p;
}

void DeviceProfileDB::upsert(DeviceProfile profile) {
    for (auto &p : _profiles) {
        if (p.brand_family == profile.brand_family && p.device_type == profile.device_type) {
            p = std::move(profile);
            return;
        }
    }
    _profiles.push_back(std::move(profile));
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeviceProfile: contraction rates
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

/// Natural log of an extent, with an empty extent read as one.
double log_extent(size_t extent) {
    return std::log(static_cast<double>(std::max<size_t>(extent, 1)));
}

/// Where @p x falls among the ascending log positions @p axis: the lower neighbour's index and the
/// fraction of the way to the next, clamped to the ends.
std::pair<size_t, double> bracket(std::vector<double> const &axis, double x) {
    if (axis.size() < 2 || x <= axis.front()) {
        return {0, 0.0};
    }
    if (x >= axis.back()) {
        return {axis.size() - 2, 1.0};
    }
    auto const   upper = static_cast<size_t>(std::ranges::upper_bound(axis, x) - axis.begin());
    size_t const lower = upper - 1;
    return {lower, (x - axis[lower]) / (axis[upper] - axis[lower])};
}

/// Linear interpolation of @p ys over the ascending @p xs at @p x, clamped to the ends.
double interpolate(std::vector<double> const &xs, std::vector<double> const &ys, double x) {
    if (xs.size() == 1) {
        return ys.front();
    }
    auto const [i, t] = bracket(xs, x);
    return ys[i] + t * (ys[i + 1] - ys[i]);
}

/// The log rate of @p points read at the cube of the GEMM's volume, interpolated in log size.
/// Every point is a cube. A table of cubes says how a kernel's fixed costs amortize with size and
/// nothing about shape, so a shape is read by its size alone; a thin GEMM's low arithmetic
/// intensity is the memory floor's to price (see DeviceProfile::estimate_contraction_time_us).
double cube_log_rate(std::vector<GemmEfficiencyPoint const *> points, size_t M, size_t N, size_t K) {
    std::ranges::sort(points, {}, &GemmEfficiencyPoint::M);
    std::vector<double> xs, ys;
    for (auto const *pt : points) {
        if (!xs.empty() && log_extent(pt->M) == xs.back()) {
            continue;
        }
        xs.push_back(log_extent(pt->M));
        ys.push_back(std::log(pt->gflops));
    }
    return interpolate(xs, ys, (log_extent(M) + log_extent(N) + log_extent(K)) / 3.0);
}

/// The log rate of @p points read trilinearly in log M, log N and log K over the grid their shapes
/// span. A corner the grid lacks drops out and the rest are reweighted; with none, the nearest
/// point by log distance answers.
double grid_log_rate(std::vector<GemmEfficiencyPoint const *> const &points, size_t M, size_t N, size_t K) {
    auto const axis_of = [&](auto member) {
        std::vector<double> axis;
        axis.reserve(points.size());
        for (auto const *pt : points) {
            axis.push_back(log_extent(pt->*member));
        }
        std::ranges::sort(axis);
        axis.erase(std::ranges::unique(axis).begin(), axis.end());
        return axis;
    };
    std::array<std::vector<double>, 3> const axes{axis_of(&GemmEfficiencyPoint::M), axis_of(&GemmEfficiencyPoint::N),
                                                  axis_of(&GemmEfficiencyPoint::K)};
    std::array<double, 3> const              at{log_extent(M), log_extent(N), log_extent(K)};
    std::array<std::pair<size_t, double>, 3> where{};
    for (size_t d = 0; d < 3; ++d) {
        where[d] = bracket(axes[d], at[d]);
    }

    double weighted = 0.0;
    double total    = 0.0;
    for (unsigned corner = 0; corner < 8; ++corner) {
        double                weight = 1.0;
        std::array<double, 3> position{};
        for (size_t d = 0; d < 3; ++d) {
            bool const upper  = ((corner >> d) & 1U) != 0U;
            auto const [i, t] = where[d];
            if (axes[d].size() < 2) {
                if (upper) {
                    weight = 0.0;
                }
                position[d] = axes[d].front();
                continue;
            }
            weight *= upper ? t : 1.0 - t;
            position[d] = axes[d][upper ? i + 1 : i];
        }
        if (weight == 0.0) {
            continue;
        }
        auto const found = std::ranges::find_if(points, [&](GemmEfficiencyPoint const *pt) {
            return log_extent(pt->M) == position[0] && log_extent(pt->N) == position[1] && log_extent(pt->K) == position[2];
        });
        if (found != points.end()) {
            weighted += weight * std::log((*found)->gflops);
            total += weight;
        }
    }
    if (total > 0.0) {
        return weighted / total;
    }

    GemmEfficiencyPoint const *nearest  = points.front();
    double                     shortest = std::numeric_limits<double>::infinity();
    for (auto const *pt : points) {
        double const dm = log_extent(pt->M) - at[0], dn = log_extent(pt->N) - at[1], dk = log_extent(pt->K) - at[2];
        if (double const d = dm * dm + dn * dn + dk * dk; d < shortest) {
            shortest = d;
            nearest  = pt;
        }
    }
    return std::log(nearest->gflops);
}

} // namespace

double DeviceProfile::estimate_gemm_gflops(size_t M, size_t N, size_t K, GemmEngine engine) const {
    std::vector<GemmEfficiencyPoint const *> points;
    for (GemmEngine const from : {engine, GemmEngine::Routed}) {
        for (auto const &pt : gemm_efficiency) {
            if (pt.engine == from && pt.gflops > 0.0) {
                points.push_back(&pt);
            }
        }
        if (!points.empty()) {
            break;
        }
    }
    if (points.empty()) {
        double const volume     = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double const efficiency = std::min(0.85, 0.1 + 0.75 * (1.0 - std::exp(-volume / 1e6)));
        return peak_gflops_fp64 * efficiency;
    }
    bool const cubes = std::ranges::all_of(points, [](GemmEfficiencyPoint const *pt) { return pt->M == pt->N && pt->N == pt->K; });
    return std::exp(cubes ? cube_log_rate(points, M, N, K) : grid_log_rate(points, M, N, K));
}

double DeviceProfile::run_cost_ns(std::vector<RunCostPoint> const &curve, size_t run) {
    if (curve.empty()) {
        return 0.0;
    }
    std::vector<RunCostPoint> sorted = curve;
    std::ranges::sort(sorted, {}, &RunCostPoint::run);
    std::vector<double> xs, ys;
    for (auto const &pt : sorted) {
        if (!xs.empty() && log_extent(pt.run) == xs.back()) {
            continue;
        }
        xs.push_back(log_extent(pt.run));
        ys.push_back(pt.ns_per_element);
    }
    return std::max(0.0, interpolate(xs, ys, log_extent(run)));
}

double DeviceProfile::estimate_contraction_time_us(packed_gemm::PackedContractionShape const &shape, unsigned threads,
                                                   size_t element_size) const {
    if (!shape.valid) {
        return kernel_launch_overhead_us;
    }
    auto const   M            = static_cast<size_t>(shape.M);
    auto const   N            = static_cast<size_t>(shape.N);
    auto const   K            = static_cast<size_t>(shape.K);
    auto const   batch        = static_cast<double>(std::max<int64_t>(shape.batch, 1));
    auto const   md           = static_cast<double>(M);
    auto const   nd           = static_cast<double>(N);
    auto const   kd           = static_cast<double>(K);
    double const gemm_speedup = gemm_parallel_speedup(M, N, K, threads);

    double const     flops  = 2.0 * md * nd * kd * batch;
    GemmEngine const engine = shape.c_scattered ? GemmEngine::Packed : GemmEngine::Routed;
    // At the rate the GEMM curve gives for this shape, but never faster than its operands and its
    // output can stream: a low-intensity GEMM is bandwidth-bound whatever the curve says.
    double const traffic    = batch * (md * kd + kd * nd + md * nd) * static_cast<double>(element_size);
    double const arithmetic = std::max(flops / (estimate_gemm_gflops(M, N, K, engine) * 1e3 * gemm_speedup),
                                       estimate_memory_time_us(static_cast<size_t>(traffic), threads));

    double const packing = batch *
                           (md * kd * run_cost_ns(pack_cost, static_cast<size_t>(shape.a_run)) +
                            kd * nd * run_cost_ns(pack_cost, static_cast<size_t>(shape.b_run))) /
                           (1e3 * gemm_speedup);

    double scatter = 0.0;
    if (shape.c_scattered) {
        size_t const c_bytes = M * N * element_size;
        scatter              = batch * md * nd * run_cost_ns(scatter_cost, static_cast<size_t>(shape.c_run)) /
                               (1e3 * parallel_speedup(KernelFamily::Elementwise, c_bytes, threads));
    }

    // A cut extent writes each share's whole output into a buffer of its own and adds the buffers
    // into C: every share's output written once and read back once.
    double     shares_traffic = 0.0;
    auto const shares         = static_cast<double>(std::max(shape.k_shares, 1));
    if (shape.k_shares > 1) {
        shares_traffic = estimate_memory_time_us(static_cast<size_t>(2.0 * shares * md * nd * static_cast<double>(element_size)), threads);
    }

    return shares * kernel_launch_overhead_us + arithmetic + packing + scatter + shares_traffic;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in default profiles
// ═══════════════════════════════════════════════════════════════════════════════

DeviceProfileDB DeviceProfileDB::load_defaults() {
    DeviceProfileDB db;

    // ── Fallbacks ──────────────────────────────────────────────────────────
    db._fallback_cpu.name               = "Generic CPU";
    db._fallback_cpu.device_type        = DeviceType::CPU;
    db._fallback_cpu.brand_family       = "generic_cpu";
    db._fallback_cpu.peak_gflops_fp64   = 50.0;
    db._fallback_cpu.peak_gflops_fp32   = 100.0;
    db._fallback_cpu.mem_bandwidth_gbps = 40.0;
    db._fallback_cpu.gemm_efficiency    = {{.M = 16, .N = 16, .K = 16, .gflops = 5.0},
                                           {.M = 64, .N = 64, .K = 64, .gflops = 30.0},
                                           {.M = 256, .N = 256, .K = 256, .gflops = 45.0},
                                           {.M = 1024, .N = 1024, .K = 1024, .gflops = 49.0}};
    db._fallback_cpu.scatter_cost       = run_cost_curve(kScatterFloorNs, kScatterPerRunNs, kReferenceBandwidthGbps / 40.0);
    db._fallback_cpu.pack_cost          = run_cost_curve(kPackFloorNs, kPackPerRunNs, kReferenceBandwidthGbps / 40.0);

    db._fallback_gpu.name         = "";
    db._fallback_gpu.device_type  = DeviceType::GPU;
    db._fallback_gpu.brand_family = "none";

    // Shared CPU DeviceProfile builder: the vendor tables below differ only in
    // the launch/alloc overheads and the gemm-efficiency curve, so everything
    // else lives here once.
    auto cpu_base = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw, double launch_us,
                       double alloc_us, std::vector<GemmEfficiencyPoint> gemm) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::CPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp64 * 2.0;
        p.mem_bandwidth_gbps        = bw;
        p.kernel_launch_overhead_us = launch_us;
        p.alloc_overhead_us         = alloc_us;
        p.gemm_efficiency           = std::move(gemm);
        p.scatter_cost              = run_cost_curve(kScatterFloorNs, kScatterPerRunNs, kReferenceBandwidthGbps / bw);
        p.pack_cost                 = run_cost_curve(kPackFloorNs, kPackPerRunNs, kReferenceBandwidthGbps / bw);
        return p;
    };

    // ── Apple Silicon ──────────────────────────────────────────────────────
    auto apple_base = [&](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        return cpu_base(std::move(name), std::move(family), std::move(patterns), fp64, bw, 0.3, 1.0,
                        {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.08},
                         {.M = 32, .N = 32, .K = 32, .gflops = fp64 * 0.25},
                         {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.55},
                         {.M = 128, .N = 128, .K = 128, .gflops = fp64 * 0.80},
                         {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.92},
                         {.M = 512, .N = 512, .K = 512, .gflops = fp64 * 0.97},
                         {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.99}});
    };

    db._profiles.push_back(apple_base("Apple M1", "apple_m1", {"Apple M1"}, 60.0, 68.0));
    db._profiles.push_back(apple_base("Apple M1 Pro", "apple_m1_pro", {"Apple M1 Pro"}, 75.0, 200.0));
    db._profiles.push_back(apple_base("Apple M1 Max", "apple_m1_max", {"Apple M1 Max"}, 75.0, 400.0));
    db._profiles.push_back(apple_base("Apple M2", "apple_m2", {"Apple M2"}, 80.0, 100.0));
    db._profiles.push_back(apple_base("Apple M2 Pro", "apple_m2_pro", {"Apple M2 Pro"}, 90.0, 200.0));
    db._profiles.push_back(apple_base("Apple M2 Max", "apple_m2_max", {"Apple M2 Max"}, 90.0, 400.0));
    db._profiles.push_back(apple_base("Apple M3", "apple_m3", {"Apple M3"}, 85.0, 100.0));
    db._profiles.push_back(apple_base("Apple M3 Pro", "apple_m3_pro", {"Apple M3 Pro"}, 100.0, 150.0));
    db._profiles.push_back(apple_base("Apple M3 Max", "apple_m3_max", {"Apple M3 Max"}, 100.0, 400.0));
    db._profiles.push_back(apple_base("Apple M4", "apple_m4", {"Apple M4"}, 100.0, 120.0));
    db._profiles.push_back(apple_base("Apple M4 Pro", "apple_m4_pro", {"Apple M4 Pro"}, 120.0, 273.0));
    db._profiles.push_back(apple_base("Apple M4 Max", "apple_m4_max", {"Apple M4 Max"}, 120.0, 546.0));

    // ── Intel / AMD ──────────────────────────────────────────────────────────
    auto intel_base = [&](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        return cpu_base(std::move(name), std::move(family), std::move(patterns), fp64, bw, 0.5, 2.0,
                        {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.10},
                         {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.60},
                         {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.90},
                         {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.98}});
    };

    db._profiles.push_back(intel_base("Intel Skylake", "intel_skylake", {"Skylake", "i7-6", "i9-6", "E5-26"}, 50.0, 40.0));
    db._profiles.push_back(intel_base("Intel Ice Lake", "intel_icelake", {"Ice Lake", "i7-10", "i9-10"}, 80.0, 50.0));
    db._profiles.push_back(intel_base("Intel Sapphire Rapids", "intel_spr", {"Sapphire Rapids", "w9-3", "w7-3"}, 120.0, 80.0));

    // ── AMD ────────────────────────────────────────────────────────────────
    db._profiles.push_back(intel_base("AMD EPYC Rome", "amd_rome", {"Rome", "EPYC 7"}, 60.0, 50.0));
    db._profiles.push_back(intel_base("AMD EPYC Milan", "amd_milan", {"Milan", "EPYC 73"}, 80.0, 80.0));
    db._profiles.push_back(intel_base("AMD EPYC Genoa", "amd_genoa", {"Genoa", "EPYC 9"}, 120.0, 115.0));

    // ── NVIDIA GPUs ────────────────────────────────────────────────────────
    auto nvidia_gpu = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double fp32, double dev_bw,
                         double pcie_bw) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::GPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp32;
        p.device_bandwidth_gbps     = dev_bw;
        p.pcie_bandwidth_gbps       = pcie_bw;
        p.gpu_launch_latency_us     = 5.0;
        p.kernel_launch_overhead_us = 1.0;
        p.alloc_overhead_us         = 5.0;
        p.gemm_efficiency           = {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.01},
                                       {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.20},
                                       {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.70},
                                       {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.96},
                                       {.M = 4096, .N = 4096, .K = 4096, .gflops = fp64 * 0.99}};
        return p;
    };

    db._profiles.push_back(nvidia_gpu("NVIDIA V100", "nvidia_v100", {"V100"}, 7800.0, 15700.0, 900.0, 12.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA A100", "nvidia_a100", {"A100"}, 9700.0, 19500.0, 2039.0, 25.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA H100", "nvidia_h100", {"H100"}, 30000.0, 60000.0, 3350.0, 50.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA RTX 3090", "nvidia_rtx3090", {"RTX 3090", "3090"}, 556.0, 35600.0, 936.0, 12.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA RTX 4090", "nvidia_rtx4090", {"RTX 4090", "4090"}, 1290.0, 82600.0, 1008.0, 12.0));

    // ── AMD GPUs ───────────────────────────────────────────────────────────
    db._profiles.push_back(nvidia_gpu("AMD MI250X", "amd_mi250x", {"MI250", "MI250X"}, 47900.0, 47900.0, 3277.0, 25.0));
    db._profiles.push_back(nvidia_gpu("AMD MI300X", "amd_mi300x", {"MI300", "MI300X"}, 81900.0, 163800.0, 5300.0, 50.0));

    // ── Apple MPS (GPU profile for MPS backend) ────────────────────────────
    {
        DeviceProfile p;
        p.name                  = "Apple MPS (Metal)";
        p.device_type           = DeviceType::GPU;
        p.brand_family          = "apple_mps";
        p.match_patterns        = {"Apple"};
        p.source                = "default";
        p.peak_gflops_fp64      = 0.0; // MPS doesn't support FP64 GEMM
        p.peak_gflops_fp32      = 200.0;
        p.device_bandwidth_gbps = 200.0; // Unified memory
        p.pcie_bandwidth_gbps   = 200.0; // No PCIe, unified
        p.gpu_launch_latency_us = 5.0;
        p.gemm_efficiency       = {{.M = 16, .N = 16, .K = 16, .gflops = 10.0},
                                   {.M = 64, .N = 64, .K = 64, .gflops = 80.0},
                                   {.M = 256, .N = 256, .K = 256, .gflops = 160.0},
                                   {.M = 1024, .N = 1024, .K = 1024, .gflops = 195.0}};
        db._profiles.push_back(p);
    }

    return db;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CostModel factory
// ═══════════════════════════════════════════════════════════════════════════════

CostModel CostModel::built_in() {
    // Cache sizes are pure topology, so runtime detection beats a table entry
    // that omits them.
    CostModel model = DeviceProfileDB::load_defaults().build_cost_model();
    if (model.cpu.caches.empty()) {
        model.cpu.caches = detect_cpu_caches();
    }
    return model;
}

CostModel CostModel::detect_default() {
    // The machine's hardware profile, written by einsums-tune, takes precedence
    // over the built-in table: every profile consumer (ContractionPlanning's
    // chain DP, GPUPlacement, GEMMBatching's profitability gate) then uses the
    // measured efficiency curve and bandwidths instead of generic estimates.
    // The Hardware module finds the profile and decides whether it applies to
    // this machine and build, and says why when it does not; without one the
    // table is used, since the profile shapes optimization choices, never
    // correctness. A profile that carries no cache data takes the detected
    // levels, as the table does.
    if (auto const section = hardware::profile_section("cost_model"); section && section->is_object()) {
        CostModel model = from_json(section->as_object());
        if (model.cpu.caches.empty()) {
            model.cpu.caches = detect_cpu_caches();
        }
        // A profile from before the scatter and pack terms were measured still has its rates and
        // thread curves to give; the table supplies the terms it lacks, and the log says so once,
        // since tiling choices lean on them.
        if (model.cpu.scatter_cost.empty() || model.cpu.pack_cost.empty()) {
            CostModel const table = built_in();
            if (model.cpu.scatter_cost.empty()) {
                model.cpu.scatter_cost = table.cpu.scatter_cost;
            }
            if (model.cpu.pack_cost.empty()) {
                model.cpu.pack_cost = table.cpu.pack_cost;
            }
            EINSUMS_LOG_WARN("The hardware profile was measured before the contraction scatter and pack costs; the built-in "
                             "table supplies them. Re-run einsums-tune to measure them on this machine.");
        }
        return model;
    }
    return built_in();
}

CostModel CostModel::flop_count() {
    CostModel model;
    model.source                        = "flop_count";
    model.cpu.name                      = "Flop count";
    model.cpu.brand_family              = "flop_count";
    model.cpu.source                    = "flop_count";
    model.cpu.peak_gflops_fp64          = 1.0;
    model.cpu.peak_gflops_fp32          = 1.0;
    model.cpu.kernel_launch_overhead_us = 0.0;
    model.cpu.alloc_overhead_us         = 0.0;
    // One efficiency point, so the nearest-neighbour lookup answers 1 GFLOP/s for
    // every shape.
    model.cpu.gemm_efficiency = {{.M = 1, .N = 1, .K = 1, .gflops = 1.0}};
    // Memory traffic priced at nothing. Finite, because a JSON document cannot
    // hold an infinity and this model should survive save_json like any other;
    // at this rate the largest GEMM's traffic stays far below its smallest flop
    // term, so the roofline max always picks the compute time.
    model.cpu.mem_bandwidth_gbps = 1e30;
    return model;
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON I/O
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

/// The number at @p key, or @p fallback when the key is missing or is not a number.
double number_or(json::Object const &obj, std::string_view key, double fallback) {
    auto const *value = obj.peek(key);
    return (value != nullptr && value->is_number()) ? value->as_double() : fallback;
}

/// The string at @p key, or "" when the key is missing or is not a string.
std::string string_or(json::Object const &obj, std::string_view key) {
    auto const *value = obj.peek(key);
    return (value != nullptr && value->is_string()) ? value->as_string() : std::string{};
}

/// Call @p on_object once per object element of the array at @p key.
///
/// A key that is absent, or holds something other than an array, contributes
/// nothing; a non-object element is skipped. That leniency is deliberate and is
/// the one place this file departs from the strict, consumed-key policy the rest
/// of the graph IR follows: a profile is a machine measurement someone may have
/// written by hand or with an older build, and a row this build cannot price is
/// dropped rather than turned into a load failure.
void for_each_object(json::Object const &obj, std::string_view key, auto const &on_object) {
    auto const *value = obj.peek(key);
    if (value == nullptr || !value->is_array()) {
        return;
    }
    for (auto const &item : value->as_array()) {
        if (item.is_object()) {
            on_object(item.as_object());
        }
    }
}

json::Object write_profile(DeviceProfile const &p) {
    json::Object obj;
    obj.set("name", p.name);
    obj.set("device_type", p.device_type == DeviceType::GPU ? "gpu" : "cpu");
    obj.set("brand_family", p.brand_family);

    json::Array patterns;
    for (auto const &pattern : p.match_patterns) {
        patterns.emplace_back(pattern);
    }
    obj.set("match_patterns", json::Value{std::move(patterns)});

    obj.set("peak_gflops_fp64", p.peak_gflops_fp64);
    obj.set("peak_gflops_fp32", p.peak_gflops_fp32);
    obj.set("mem_bandwidth_gbps", p.mem_bandwidth_gbps);
    obj.set("kernel_launch_overhead_us", p.kernel_launch_overhead_us);
    obj.set("alloc_overhead_us", p.alloc_overhead_us);
    obj.set("device_bandwidth_gbps", p.device_bandwidth_gbps);
    obj.set("pcie_bandwidth_gbps", p.pcie_bandwidth_gbps);
    obj.set("gpu_launch_latency_us", p.gpu_launch_latency_us);

    obj.set("inter_node_bandwidth_gbps", p.inter_node_bandwidth_gbps);
    obj.set("inter_node_latency_us", p.inter_node_latency_us);
    obj.set("nccl_bandwidth_gbps", p.nccl_bandwidth_gbps);

    json::Array caches;
    for (auto const &level : p.caches) {
        json::Object entry;
        entry.set("size_bytes", level.size_bytes);
        entry.set("bandwidth_gbps", level.bandwidth_gbps);
        entry.set("latency_ns", level.latency_ns);
        caches.emplace_back(std::move(entry));
    }
    obj.set("caches", json::Value{std::move(caches)});

    json::Array gemm;
    for (auto const &pt : p.gemm_efficiency) {
        json::Object entry;
        entry.set("M", pt.M);
        entry.set("N", pt.N);
        entry.set("K", pt.K);
        entry.set("gflops", pt.gflops);
        entry.set("engine", std::string(to_string(pt.engine)));
        gemm.emplace_back(std::move(entry));
    }
    obj.set("gemm_efficiency", json::Value{std::move(gemm)});

    auto const write_runs = [&obj](std::string_view key, std::vector<RunCostPoint> const &curve) {
        json::Array rows;
        for (auto const &pt : curve) {
            json::Object entry;
            entry.set("run", pt.run);
            entry.set("ns_per_element", pt.ns_per_element);
            rows.emplace_back(std::move(entry));
        }
        obj.set(std::string(key), json::Value{std::move(rows)});
    };
    write_runs("scatter", p.scatter_cost);
    write_runs("pack", p.pack_cost);

    json::Array permute;
    for (auto const &pt : p.permute_efficiency) {
        json::Object entry;
        entry.set("bytes", pt.bytes);
        entry.set("rank", pt.rank);
        entry.set("gbps", pt.gbps);
        permute.emplace_back(std::move(entry));
    }
    obj.set("permute_efficiency", json::Value{std::move(permute)});

    obj.set("max_threads", static_cast<std::int64_t>(p.max_threads));

    // One row per (family, size class, width) rather than one object per curve
    // with a nested widths array. The shape is kept for the sake of profile
    // files already on disk, and it reads no worse: a row names everything it
    // needs, so a reader that does not know a family drops that row alone.
    json::Array curves;
    for (auto const &curve : p.thread_efficiency) {
        if (!curve.valid()) {
            continue;
        }
        for (size_t i = 0; i < curve.widths.size(); i++) {
            json::Object entry;
            entry.set("family", to_string(curve.family));
            entry.set("size_class", to_string(curve.size_class));
            entry.set("width", static_cast<std::int64_t>(curve.widths[i]));
            entry.set("speedup", curve.speedup[i]);
            curves.emplace_back(std::move(entry));
        }
    }
    obj.set("thread_efficiency", json::Value{std::move(curves)});

    return obj;
}

DeviceProfile read_profile(json::Object const &obj) {
    DeviceProfile p;

    p.name                      = string_or(obj, "name");
    p.brand_family              = string_or(obj, "brand_family");
    p.device_type               = (string_or(obj, "device_type") == "gpu") ? DeviceType::GPU : DeviceType::CPU;
    p.peak_gflops_fp64          = number_or(obj, "peak_gflops_fp64", 50.0);
    p.peak_gflops_fp32          = number_or(obj, "peak_gflops_fp32", 100.0);
    p.mem_bandwidth_gbps        = number_or(obj, "mem_bandwidth_gbps", 40.0);
    p.kernel_launch_overhead_us = number_or(obj, "kernel_launch_overhead_us", 0.5);
    p.alloc_overhead_us         = number_or(obj, "alloc_overhead_us", 2.0);
    p.device_bandwidth_gbps     = number_or(obj, "device_bandwidth_gbps", 0.0);
    p.pcie_bandwidth_gbps       = number_or(obj, "pcie_bandwidth_gbps", 0.0);
    p.gpu_launch_latency_us     = number_or(obj, "gpu_launch_latency_us", 0.0);

    // The network figures round-trip too; dropping them would reload a
    // calibrated fabric with the struct defaults.
    p.inter_node_bandwidth_gbps = number_or(obj, "inter_node_bandwidth_gbps", 0.0);
    p.inter_node_latency_us     = number_or(obj, "inter_node_latency_us", 1.0);
    p.nccl_bandwidth_gbps       = number_or(obj, "nccl_bandwidth_gbps", 0.0);

    if (auto const *patterns = obj.peek("match_patterns"); patterns != nullptr && patterns->is_array()) {
        for (auto const &item : patterns->as_array()) {
            if (item.is_string()) {
                p.match_patterns.push_back(item.as_string());
            }
        }
    }

    for_each_object(obj, "gemm_efficiency", [&](json::Object const &entry) {
        GemmEfficiencyPoint pt;
        pt.M      = static_cast<size_t>(number_or(entry, "M", 0.0));
        pt.N      = static_cast<size_t>(number_or(entry, "N", 0.0));
        pt.K      = static_cast<size_t>(number_or(entry, "K", 0.0));
        pt.gflops = number_or(entry, "gflops", 0.0);
        // A point from before engines were told apart was measured on the routed engine. A point
        // naming an engine this build does not know is dropped rather than priced as another.
        if (auto const engine = string_or(entry, "engine"); !engine.empty() && !gemm_engine_from_string(engine, pt.engine)) {
            return;
        }
        if (pt.M > 0 && pt.gflops > 0) {
            p.gemm_efficiency.push_back(pt);
        }
    });

    auto const read_runs = [&obj](std::string_view key, std::vector<RunCostPoint> &curve) {
        for_each_object(obj, key, [&](json::Object const &entry) {
            RunCostPoint pt;
            pt.run            = static_cast<size_t>(number_or(entry, "run", 0.0));
            pt.ns_per_element = number_or(entry, "ns_per_element", -1.0);
            if (pt.run > 0 && pt.ns_per_element >= 0.0) {
                curve.push_back(pt);
            }
        });
    };
    read_runs("scatter", p.scatter_cost);
    read_runs("pack", p.pack_cost);

    for_each_object(obj, "permute_efficiency", [&](json::Object const &entry) {
        PermuteEfficiencyPoint pt;
        pt.bytes = static_cast<size_t>(number_or(entry, "bytes", 0.0));
        pt.rank  = static_cast<size_t>(number_or(entry, "rank", 0.0));
        pt.gbps  = number_or(entry, "gbps", 0.0);
        if (pt.bytes > 0 && pt.gbps > 0) {
            p.permute_efficiency.push_back(pt);
        }
    });

    // The cache array was written but never read back, so a calibrated profile
    // silently lost its cache sizes on reload and fell back to whatever the
    // detector reported for the machine doing the loading.
    for_each_object(obj, "caches", [&](json::Object const &entry) {
        CacheLevel level;
        level.size_bytes     = static_cast<size_t>(number_or(entry, "size_bytes", 0.0));
        level.bandwidth_gbps = number_or(entry, "bandwidth_gbps", 0.0);
        level.latency_ns     = number_or(entry, "latency_ns", 0.0);
        if (level.size_bytes > 0) {
            p.caches.push_back(level);
        }
    });

    p.max_threads = static_cast<unsigned>(number_or(obj, "max_threads", 0.0));

    // Flat (family, size class, width, speedup) rows regrouped into curves.
    // Rows naming a family or size class this build does not know are dropped:
    // pricing them as something else would be worse than pricing them from the
    // default model.
    std::map<std::pair<std::uint8_t, std::uint8_t>, EfficiencyCurve> curves;
    for_each_object(obj, "thread_efficiency", [&](json::Object const &entry) {
        KernelFamily family{};
        SizeClass    size_class{};
        if (!kernel_family_from_string(string_or(entry, "family"), family) ||
            !size_class_from_string(string_or(entry, "size_class"), size_class)) {
            return;
        }
        auto const width = static_cast<std::uint32_t>(number_or(entry, "width", 0.0));
        if (width == 0) {
            return;
        }
        auto &curve      = curves[{static_cast<std::uint8_t>(family), static_cast<std::uint8_t>(size_class)}];
        curve.family     = family;
        curve.size_class = size_class;
        curve.widths.push_back(width);
        curve.speedup.push_back(number_or(entry, "speedup", 0.0));
    });

    for (auto &[key, curve] : curves) {
        // Rows may arrive in any order; the query interpolates and so needs
        // ascending rungs.
        std::vector<size_t> order(curve.widths.size());
        for (size_t idx = 0; idx < order.size(); idx++) {
            order[idx] = idx;
        }
        std::ranges::sort(order, [&](size_t a, size_t b) { return curve.widths[a] < curve.widths[b]; });

        EfficiencyCurve sorted;
        sorted.family     = curve.family;
        sorted.size_class = curve.size_class;
        for (size_t const idx : order) {
            sorted.widths.push_back(curve.widths[idx]);
            sorted.speedup.push_back(curve.speedup[idx]);
        }
        if (sorted.valid()) {
            p.thread_efficiency.push_back(std::move(sorted));
        }
    }

    return p;
}

/// Write @p document to @p path, laid out one key per line.
expected<void, GraphError> write_document(std::string const &path, json::Value const &document, std::string_view what) {
    std::ofstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", what, path)));
    }
    f << json::emit(document, json::EmitOptions{.style = json::EmitStyle::Pretty});
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: could not write '{}'", what, path)));
    }
    return {};
}

/// Read @p path and parse it as a JSON object.
expected<json::Value, GraphError> read_document(std::string const &path, std::string_view what) {
    std::ifstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", what, path)));
    }
    std::string const content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto document = json::parse(content);
    if (!document) {
        return unexpected(GraphError::parse(fmt::format("{}: '{}' is not valid JSON: {}", what, path, document.error().to_string())));
    }
    if (!document->is_object()) {
        return unexpected(GraphError::parse(
            fmt::format("{}: '{}' holds a {} where an object was expected", what, path, json::Value::type_name(document->type()))));
    }
    return std::move(*document);
}

} // namespace

json::Object CostModel::to_json() const {
    json::Object root;
    root.set("source", source);
    root.set("cpu", write_profile(cpu));
    root.set("gpu", write_profile(gpu));
    return root;
}

expected<void, GraphError> CostModel::save_json(std::string const &path) const {
    return write_document(path, json::Value{to_json()}, "CostModel::save_json");
}

expected<CostModel, GraphError> CostModel::load_json(std::string const &path) {
    auto document = read_document(path, "CostModel::load_json");
    if (!document) {
        return unexpected(document.error());
    }
    return from_json(document->as_object());
}

CostModel CostModel::from_json(json::Object const &root) {
    CostModel p;
    // Only when the file names one: an absent key leaves the struct default
    // ("default") rather than blanking it.
    if (auto const *named = root.peek("source"); named != nullptr && named->is_string()) {
        p.source = named->as_string();
    }
    if (auto const *cpu_obj = root.peek("cpu"); cpu_obj != nullptr && cpu_obj->is_object()) {
        p.cpu = read_profile(cpu_obj->as_object());
    }
    if (auto const *gpu_obj = root.peek("gpu"); gpu_obj != nullptr && gpu_obj->is_object()) {
        p.gpu = read_profile(gpu_obj->as_object());
    }
    return p;
}

expected<void, GraphError> DeviceProfileDB::save_json(std::string const &path) const {
    json::Array profiles;
    for (auto const &profile : _profiles) {
        profiles.emplace_back(write_profile(profile));
    }

    json::Object root;
    root.set("version", std::int64_t{1});
    root.set("profiles", json::Value{std::move(profiles)});
    return write_document(path, json::Value{std::move(root)}, "DeviceProfileDB::save_json");
}

expected<DeviceProfileDB, GraphError> DeviceProfileDB::load_json(std::string const &path) {
    auto document = read_document(path, "DeviceProfileDB::load_json");
    if (!document) {
        return unexpected(document.error());
    }

    DeviceProfileDB db = load_defaults(); // Start with defaults, overlay from file

    for_each_object(document->as_object(), "profiles", [&](json::Object const &entry) {
        auto p = read_profile(entry);
        if (!p.name.empty()) {
            p.source = "database";
            db.upsert(std::move(p));
        }
    });

    return db;
}

EINSUMS_NAMESPACE_END(compute_graph)
