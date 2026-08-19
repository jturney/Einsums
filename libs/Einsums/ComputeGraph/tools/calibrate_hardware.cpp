//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file calibrate_hardware.cpp
/// @brief Hardware calibration tool for the ContractionPlanning pass.
///
/// Measures GEMM performance across a range of matrix shapes and memory
/// bandwidth, then saves the results as a JSON CostModel that the
/// ContractionPlanning pass can load for architecture-specific optimization.
///
/// Usage:
///   ./calibrate_hardware --output my_hardware.json
///   ./calibrate_hardware --output gpu_profile.json --dtype float
///   ./calibrate_hardware --min-size 16 --max-size 4096 --num-points 30
///
/// The tool sweeps GEMM shapes logarithmically from min-size to max-size,
/// measuring sustained GFLOPS at each point. It also measures memory
/// bandwidth via a STREAM-like copy benchmark.

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

struct Config {
    std::string output = "hardware_profile.json";
    std::string calibration; // empty: hardware::default_calibration_path()
    std::string dtype          = "double";
    size_t      min_size       = 16;
    size_t      max_size       = 2048;
    size_t      num_points     = 15;
    size_t      warmup         = 3;
    size_t      repeats        = 10;
    bool        thread_sweep   = true;
    size_t      thread_repeats = 3;
    size_t      thread_max_mb  = 64;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string const arg(argv[i]);
        if (arg == "--output" && i + 1 < argc)
            cfg.output = argv[++i];
        else if (arg == "--calibration" && i + 1 < argc)
            cfg.calibration = argv[++i];
        else if (arg == "--dtype" && i + 1 < argc)
            cfg.dtype = argv[++i];
        else if (arg == "--min-size" && i + 1 < argc)
            cfg.min_size = std::stoull(argv[++i]);
        else if (arg == "--max-size" && i + 1 < argc)
            cfg.max_size = std::stoull(argv[++i]);
        else if (arg == "--num-points" && i + 1 < argc)
            cfg.num_points = std::stoull(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)
            cfg.warmup = std::stoull(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc)
            cfg.repeats = std::stoull(argv[++i]);
        else if (arg == "--no-thread-sweep")
            cfg.thread_sweep = false;
        else if (arg == "--thread-repeats" && i + 1 < argc)
            cfg.thread_repeats = std::stoull(argv[++i]);
        else if (arg == "--thread-max-mb" && i + 1 < argc)
            cfg.thread_max_mb = std::stoull(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            println("Usage: calibrate_hardware [options]");
            println("  --output <path>    Output JSON file (default: hardware_profile.json)");
            println("  --calibration <path>  Hardware calibration file (default: platform cache dir)");
            println("  --dtype <type>     Data type: double or float (default: double)");
            println("  --min-size <N>     Minimum matrix dimension (default: 16)");
            println("  --max-size <N>     Maximum matrix dimension (default: 2048)");
            println("  --num-points <N>   Number of measurement points (default: 15)");
            println("  --warmup <N>       Warmup iterations (default: 3)");
            println("  --repeats <N>      Measurement iterations (default: 10)");
            println("  --no-thread-sweep  Skip the per-family efficiency curves");
            println("  --thread-repeats <N>  Timed runs per curve point (default: 3)");
            println("  --thread-max-mb <N>   Cap on a curve cell's working set (default: 64)");
            std::exit(0);
        }
    }
    return cfg;
}

/// Generate logarithmically spaced sizes from min to max.
std::vector<size_t> log_space(size_t min_val, size_t max_val, size_t num_points) {
    std::vector<size_t> sizes;
    double const        log_min = std::log2(static_cast<double>(min_val));
    double const        log_max = std::log2(static_cast<double>(max_val));

    for (size_t idx = 0; idx < num_points; idx++) {
        double const t     = (num_points > 1) ? static_cast<double>(idx) / static_cast<double>(num_points - 1) : 0.0;
        double const log_s = log_min + t * (log_max - log_min);
        auto         s     = static_cast<size_t>(std::round(std::pow(2.0, log_s)));
        // Round to nearest multiple of 4 for alignment
        s = std::max(static_cast<size_t>(4), (s + 3) / 4 * 4);
        if (sizes.empty() || sizes.back() != s)
            sizes.push_back(s);
    }
    return sizes;
}

/// Measure DGEMM GFLOPS for a given square size.
double measure_dgemm_gflops(size_t N, size_t warmup, size_t repeats) {
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);
    auto C = Tensor<double, 2>("C", N, N);

    // Warmup
    for (size_t w = 0; w < warmup; w++) {
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    }

    // Measure
    std::vector<double> times(repeats);
    for (size_t r = 0; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
        auto t1  = std::chrono::steady_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    // Median time
    std::ranges::sort(times);
    double const median_s = times[repeats / 2];

    double const flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    return flops / (median_s * 1e9);
}

/// Measure SGEMM GFLOPS for a given square size.
double measure_sgemm_gflops(size_t N, size_t warmup, size_t repeats) {
    auto A = create_random_tensor<float>("A", N, N);
    auto B = create_random_tensor<float>("B", N, N);
    auto C = Tensor<float, 2>("C", N, N);

    for (size_t w = 0; w < warmup; w++) {
        linear_algebra::gemm<false, false>(1.0f, A, B, 0.0f, &C);
    }

    std::vector<double> times(repeats);
    for (size_t r = 0; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        linear_algebra::gemm<false, false>(1.0f, A, B, 0.0f, &C);
        auto t1  = std::chrono::steady_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    std::ranges::sort(times);
    double const median_s = times[repeats / 2];

    double const flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    return flops / (median_s * 1e9);
}

/// Measure memory bandwidth via a simple copy benchmark.
double measure_bandwidth_gbps() {
    constexpr size_t BUFFER_SIZE = static_cast<long>(64 * 1024) * 1024; // 64 MB
    constexpr size_t NUM_ITERS   = 10;

    auto src = std::vector<double>(BUFFER_SIZE / sizeof(double));
    auto dst = std::vector<double>(BUFFER_SIZE / sizeof(double));

    // Fill src
    for (size_t idx = 0; idx < src.size(); idx++)
        src[idx] = static_cast<double>(idx);

    // Use volatile to prevent optimizer from eliding the copies
    double volatile sink = 0.0;

    // Warmup
    std::memcpy(dst.data(), src.data(), BUFFER_SIZE);
    sink += dst[0];

    // Measure
    auto t0 = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
        std::memcpy(dst.data(), src.data(), BUFFER_SIZE);
        sink += dst[iter % dst.size()]; // Prevent optimization
    }
    auto t1 = std::chrono::steady_clock::now();
    (void)sink;

    double const seconds = std::chrono::duration<double>(t1 - t0).count();
    double const bytes   = static_cast<double>(BUFFER_SIZE) * static_cast<double>(NUM_ITERS);
    return bytes / (seconds * 1e9);
}

/// Median of a timed callable, in seconds.
template <typename F>
double median_seconds(size_t warmup, size_t repeats, F &&run) {
    for (size_t w = 0; w < warmup; w++) {
        run();
    }
    std::vector<double> times(repeats);
    for (size_t r = 0; r < repeats; r++) {
        auto const t0 = std::chrono::steady_clock::now();
        run();
        auto const t1 = std::chrono::steady_clock::now();
        times[r]      = std::chrono::duration<double>(t1 - t0).count();
    }
    std::ranges::sort(times);
    return times[repeats / 2];
}

/// Achieved (read + write) bandwidth of a rank-2 transpose of an N x N tensor.
double measure_permute_rank2_gbps(size_t N, size_t warmup, size_t repeats) {
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", N, N);
    auto C = Tensor<double, 2>("C", N, N);

    double const seconds = median_seconds(warmup, repeats, [&] { tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, A); });

    double const bytes = 2.0 * static_cast<double>(N) * static_cast<double>(N) * sizeof(double);
    return bytes / (seconds * 1e9);
}

/// Same for a rank-3 cyclic permutation of an N x N x N tensor.
double measure_permute_rank3_gbps(size_t N, size_t warmup, size_t repeats) {
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", N, N, N);
    auto C = Tensor<double, 3>("C", N, N, N);

    double const seconds = median_seconds(warmup, repeats, [&] { tensor_algebra::permute(Indices{k, i, j}, &C, Indices{i, j, k}, A); });

    double const bytes = 2.0 * std::pow(static_cast<double>(N), 3.0) * sizeof(double);
    return bytes / (seconds * 1e9);
}

/// Same for a rank-4 permutation that swaps both index pairs, the shape a
/// coupled-cluster residual keeps asking for.
double measure_permute_rank4_gbps(size_t N, size_t warmup, size_t repeats) {
    using namespace einsums::index;
    auto A = create_random_tensor<double>("A", N, N, N, N);
    auto C = Tensor<double, 4>("C", N, N, N, N);

    double const seconds = median_seconds(warmup, repeats, [&] {
        tensor_algebra::permute(Indices{j, i, l, k}, &C, Indices{i, j, k, l}, A);
    });

    double const bytes = 2.0 * std::pow(static_cast<double>(N), 4.0) * sizeof(double);
    return bytes / (seconds * 1e9);
}

/// Read bandwidth over a working set of @p bytes, for pricing a cache level.
///
/// Sums a buffer sized to sit inside the level under test, repeatedly, so the
/// data stays resident after the first pass.
///
/// The eight independent accumulators are the point: a single `acc += buf[i]`
/// chain is a dependent sequence of FP adds, so it measures the adder's
/// latency (it reported the same ~16 GB/s for L1, L2 and L3) rather than how
/// fast the level delivers bytes. Eight chains keep enough loads in flight for
/// the memory side to be the limit.
double measure_cache_bandwidth_gbps(size_t bytes) {
    constexpr size_t kLanes = 8;

    size_t n = std::max<size_t>(4096, bytes / sizeof(double));
    n        = n / kLanes * kLanes;
    std::vector<double> buf(n, 1.0);

    // Enough passes that a small level still takes a measurable time.
    size_t const passes = std::max<size_t>(8, (256UL * 1024 * 1024) / (n * sizeof(double)));

    double volatile sink = 0.0;

    double const seconds = median_seconds(1, 5, [&] {
        std::array<double, kLanes> acc{};
        for (size_t p = 0; p < passes; p++) {
            for (size_t idx = 0; idx < n; idx += kLanes) {
                for (size_t lane = 0; lane < kLanes; lane++) {
                    acc[lane] += buf[idx + lane];
                }
            }
        }
        double total = 0.0;
        for (double const a : acc) {
            total += a;
        }
        sink = total;
    });

    double const moved = static_cast<double>(n) * sizeof(double) * static_cast<double>(passes);
    return (sink == 0.0 ? 0.0 : moved) / (seconds * 1e9);
}

/// Dependent-load latency over a working set of @p bytes, in nanoseconds.
///
/// Classic pointer chase: a random cyclic permutation of the indices, walked
/// one dependent load at a time so the measured time is latency and not
/// throughput. The walked position has to escape through the volatile sink or
/// the whole loop is dead code and the level reports 0 ns.
double measure_cache_latency_ns(size_t bytes) {
    size_t const n = std::max<size_t>(4096, bytes / sizeof(size_t));

    // A random single cycle over 0..n-1: shuffle the visit order, then link
    // each position to the one visited after it.
    std::vector<size_t> order(n);
    for (size_t idx = 0; idx < n; idx++) {
        order[idx] = idx;
    }
    // Fixed seed on purpose: the shuffle must be identical between calibration
    // runs or the measured points move for reasons that are not hardware.
    std::mt19937_64 rng(12345); // NOLINT(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
    for (size_t idx = n - 1; idx > 0; idx--) {
        std::swap(order[idx], order[rng() % (idx + 1)]);
    }
    std::vector<size_t> next(n);
    for (size_t idx = 0; idx < n; idx++) {
        next[order[idx]] = order[(idx + 1) % n];
    }

    size_t const steps   = std::max<size_t>(n, 1UL << 20);
    size_t volatile sink = 0;

    double const seconds = median_seconds(1, 5, [&] {
        size_t p = sink;
        for (size_t s = 0; s < steps; s++) {
            p = next[p];
        }
        sink = p;
    });

    return seconds * 1e9 / static_cast<double>(steps);
}

/// Measure kernel launch overhead by timing very small GEMMs.
/// The fixed cost of issuing one GEMM, before any arithmetic.
///
/// Measured at 2x2x2 rather than 1x1x1. A 1x1 operand is the one shape whose
/// two strides collide, and until that was fixed it missed BLAS and ran the
/// generic path, which opened two OpenMP regions to compute a single
/// multiply-add: this returned about 60 us on a ten-core machine, roughly 600x
/// the real figure, and wrote it into every profile as launch overhead. The
/// cost model adds that term to every GEMM estimate without dividing it by
/// width, so on a graph of a few thousand nodes it swamped the arithmetic
/// entirely and the thread planner declined to widen anything.
///
/// 2x2x2 is eight flops, which is nothing next to the call itself, so the
/// measurement is still essentially pure overhead - it just is not measured on
/// the one shape in the library with a special case behind it. Anything that
/// makes a degenerate extent interesting again belongs on a size the rest of
/// the library treats as ordinary.
double measure_overhead_us() {
    constexpr size_t kDim = 2;
    auto             A    = Tensor<double, 2>("A", kDim, kDim);
    auto             B    = Tensor<double, 2>("B", kDim, kDim);
    auto             C    = Tensor<double, 2>("C", kDim, kDim);
    A.set_all(1.0);
    B.set_all(1.0);
    C.set_all(0.0);

    constexpr size_t NUM_ITERS = 1000;

    // Warmup
    for (size_t w = 0; w < 100; w++)
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    }
    auto t1 = std::chrono::steady_clock::now();

    double const total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return total_us / static_cast<double>(NUM_ITERS);
}

// einsums_main() has no argc/argv; main() stashes them here before
// einsums::start so parse_args can see the tool's own flags (einsums has
// already consumed and removed its --einsums: arguments by this point).
int    g_argc = 0;
char **g_argv = nullptr;

int einsums_main() {
    Config const cfg = parse_args(g_argc, g_argv);

    einsums::println("=== Einsums Hardware Calibration Tool ===\n");
    einsums::println("Output: {}", cfg.output);
    einsums::println("Data type: {}", cfg.dtype);
    einsums::println("Size range: {} to {} ({} points)", cfg.min_size, cfg.max_size, cfg.num_points);
    einsums::println("Warmup: {}, Repeats: {}", cfg.warmup, cfg.repeats);

    // Start from auto-detected cost_model
    cg::CostModel cost_model = cg::CostModel::detect_default();
    cost_model.source        = "calibrated";

    // Detect CPU brand
    std::string cpu_brand = cg::DeviceProfileDB::detect_cpu_brand();
    einsums::println("Detected CPU: {}", cpu_brand);

    // Generate measurement sizes
    auto sizes = log_space(cfg.min_size, cfg.max_size, cfg.num_points);

    // Measure GEMM efficiency
    einsums::println("\n--- GEMM Benchmark ---\n");
    cost_model.cpu.gemm_efficiency.clear();

    double peak_gflops = 0.0;
    for (size_t const N : sizes) {
        double gflops;
        if (cfg.dtype == "float") {
            gflops = measure_sgemm_gflops(N, cfg.warmup, cfg.repeats);
        } else {
            gflops = measure_dgemm_gflops(N, cfg.warmup, cfg.repeats);
        }

        peak_gflops = std::max(peak_gflops, gflops);
        cost_model.cpu.gemm_efficiency.push_back({.M = N, .N = N, .K = N, .gflops = gflops});

        einsums::println("  {} x {} x {}: {:.1f} GFLOPS", N, N, N, gflops);
    }

    if (cfg.dtype == "float") {
        cost_model.cpu.peak_gflops_fp32 = peak_gflops;
    } else {
        cost_model.cpu.peak_gflops_fp64 = peak_gflops;
    }

    // Measure memory bandwidth
    einsums::println("\n--- Memory Bandwidth ---\n");
    double const bw                   = measure_bandwidth_gbps();
    cost_model.cpu.mem_bandwidth_gbps = bw;
    einsums::println("  Sustained copy bandwidth: {:.1f} GB/s", bw);

    // Measure permute (tensor transpose) efficiency. A transpose moves the
    // same bytes as a copy with one side strided, so what the cost model needs
    // is how far short of the copy rate it lands, per rank and size.
    einsums::println("\n--- Permute Benchmark ---\n");
    cost_model.cpu.permute_efficiency.clear();

    auto record_permute = [&](size_t rank, size_t extent, size_t elements, double gbps) {
        size_t const bytes = elements * sizeof(double);
        cost_model.cpu.permute_efficiency.push_back({.bytes = bytes, .rank = rank, .gbps = gbps});
        einsums::println("  rank {} extent {:>5}: {:>8.2f} MB  {:>7.1f} GB/s  ({:.0f}% of copy)", rank, extent,
                         static_cast<double>(bytes) / (1024.0 * 1024.0), gbps, 100.0 * gbps / bw);
    };

    for (size_t const N : {64UL, 256UL, 1024UL, 4096UL}) {
        record_permute(2, N, N * N, measure_permute_rank2_gbps(N, cfg.warmup, cfg.repeats));
    }
    for (size_t const N : {16UL, 48UL, 128UL, 256UL}) {
        record_permute(3, N, N * N * N, measure_permute_rank3_gbps(N, cfg.warmup, cfg.repeats));
    }
    for (size_t const N : {8UL, 16UL, 32UL, 64UL}) {
        record_permute(4, N, N * N * N * N, measure_permute_rank4_gbps(N, cfg.warmup, cfg.repeats));
    }

    // Measure per-cache-level bandwidth and dependent-load latency. Sizes come
    // from the detector; this fills in the two fields it leaves at zero.
    einsums::println("\n--- Cache Levels ---\n");
    for (size_t lvl = 0; lvl < cost_model.cpu.caches.size(); lvl++) {
        auto &level = cost_model.cpu.caches[lvl];
        if (level.size_bytes == 0) {
            continue;
        }
        // Half the level, so the working set stays resident with room for the
        // walk's own metadata.
        size_t const working_set = level.size_bytes / 2;
        level.bandwidth_gbps     = measure_cache_bandwidth_gbps(working_set);
        level.latency_ns         = measure_cache_latency_ns(working_set);
        einsums::println("  L{} ({:>6.1f} KB): {:>7.1f} GB/s  {:>6.2f} ns", lvl + 1, static_cast<double>(level.size_bytes) / 1024.0,
                         level.bandwidth_gbps, level.latency_ns);
    }

    // Measure kernel overhead
    einsums::println("\n--- Kernel Overhead ---\n");
    double const overhead                    = measure_overhead_us();
    cost_model.cpu.kernel_launch_overhead_us = overhead;
    einsums::println("  DGEMM(2x2x2) call overhead: {:.2f} us", overhead);

    // Measure how each kernel family scales with thread count, at each size
    // class the cache levels just defined. This is the axis a width planner
    // reads: t(w) = t(1) / s(w).
    cost_model.cpu.max_threads = static_cast<unsigned>(std::max(1, einsums::hardware::get_max_threads()));
    if (cfg.thread_sweep) {
        einsums::println("\n--- Thread Efficiency ---\n");
        cg::ThreadSweepOptions sweep;
        sweep.warmup    = 1;
        sweep.repeats   = cfg.thread_repeats;
        sweep.max_bytes = cfg.thread_max_mb * 1024 * 1024;

        auto const t_sweep_0             = std::chrono::steady_clock::now();
        cost_model.cpu.thread_efficiency = cg::measure_thread_efficiency(cost_model.cpu, sweep);
        auto const t_sweep_1             = std::chrono::steady_clock::now();

        for (auto const &curve : cost_model.cpu.thread_efficiency) {
            std::string line = fmt::format("  {:>12} / {:<9}:", cg::to_string(curve.family), cg::to_string(curve.size_class));
            for (size_t idx = 0; idx < curve.widths.size(); idx++) {
                line += fmt::format("  w{}={:.2f}x", curve.widths[idx], curve.speedup[idx]);
            }
            einsums::println("{}", line);
        }
        if (cost_model.cpu.thread_efficiency.empty()) {
            einsums::println("  (single-threaded machine: every curve would be the point s(1) = 1)");
        }
        einsums::println("  swept {} cells in {:.1f} s", cost_model.cpu.thread_efficiency.size(),
                         std::chrono::duration<double>(t_sweep_1 - t_sweep_0).count());
    }

    // Update cost_model name
    cost_model.cpu.name   = fmt::format("{} (calibrated)", cpu_brand);
    cost_model.cpu.source = "calibrated";

    // The OpenMP region cost, swept across team sizes and written where the
    // Hardware module reads it. Separate from the profile JSON on purpose:
    // Hardware sits below the module that owns this JSON parser, and the number
    // is consumed by things that never load a CostModel (PackedGemm's parallel
    // gate, the DLPNO bucket chooser through the Python binding).
    einsums::println("\n--- OpenMP Region Cost ---\n");
    std::string       calib_error;
    std::string const calib_path = cfg.calibration.empty() ? einsums::hardware::default_calibration_path() : cfg.calibration;
    if (einsums::hardware::write_calibration(calib_path, &calib_error)) {
        einsums::println("  Calibration written to: {}", calib_path);
    } else {
        einsums::println("  WARNING: could not write calibration ({}); consumers will measure per process", calib_error);
    }

    // Save
    auto save_result = cost_model.save_json(cfg.output);
    if (!save_result) {
        einsums::println("ERROR: {}", save_result.error().message);
        return EXIT_FAILURE;
    }
    einsums::println("\nProfile saved to: {}", cfg.output);
    einsums::println("Peak GFLOPS ({}): {:.1f}", cfg.dtype, peak_gflops);

    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    return einsums::start(einsums_main, argc, argv);
}
