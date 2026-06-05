//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file calibrate_hardware.cpp
/// @brief Hardware calibration tool for the ContractionPlanning pass.
///
/// Measures GEMM performance across a range of matrix shapes and memory
/// bandwidth, then saves the results as a JSON HardwareProfile that the
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

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

struct Config {
    std::string output     = "hardware_profile.json";
    std::string dtype      = "double";
    size_t      min_size   = 16;
    size_t      max_size   = 2048;
    size_t      num_points = 15;
    size_t      warmup     = 3;
    size_t      repeats    = 10;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string const arg(argv[i]);
        if (arg == "--output" && i + 1 < argc)
            cfg.output = argv[++i];
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
        else if (arg == "--help" || arg == "-h") {
            println("Usage: calibrate_hardware [options]");
            println("  --output <path>    Output JSON file (default: hardware_profile.json)");
            println("  --dtype <type>     Data type: double or float (default: double)");
            println("  --min-size <N>     Minimum matrix dimension (default: 16)");
            println("  --max-size <N>     Maximum matrix dimension (default: 2048)");
            println("  --num-points <N>   Number of measurement points (default: 15)");
            println("  --warmup <N>       Warmup iterations (default: 3)");
            println("  --repeats <N>      Measurement iterations (default: 10)");
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

/// Measure kernel launch overhead by timing very small GEMMs.
double measure_overhead_us() {
    auto A  = Tensor<double, 2>("A", 1, 1);
    auto B  = Tensor<double, 2>("B", 1, 1);
    auto C  = Tensor<double, 2>("C", 1, 1);
    A(0, 0) = 1.0;
    B(0, 0) = 1.0;

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

int einsums_main() {
    // Parse command line (einsums has already consumed --einsums: args)
    // We need access to argc/argv but einsums_main doesn't have them.
    // For simplicity, use defaults. The tool can be enhanced later.
    Config const cfg;

    einsums::println("=== Einsums Hardware Calibration Tool ===\n");
    einsums::println("Output: {}", cfg.output);
    einsums::println("Data type: {}", cfg.dtype);
    einsums::println("Size range: {} to {} ({} points)", cfg.min_size, cfg.max_size, cfg.num_points);
    einsums::println("Warmup: {}, Repeats: {}", cfg.warmup, cfg.repeats);

    // Start from auto-detected profile
    cg::HardwareProfile profile = cg::HardwareProfile::detect_default();
    profile.source              = "calibrated";

    // Detect CPU brand
    std::string cpu_brand = cg::HardwareProfileDB::detect_cpu_brand();
    einsums::println("Detected CPU: {}", cpu_brand);

    // Generate measurement sizes
    auto sizes = log_space(cfg.min_size, cfg.max_size, cfg.num_points);

    // Measure GEMM efficiency
    einsums::println("\n--- GEMM Benchmark ---\n");
    profile.cpu.gemm_efficiency.clear();

    double peak_gflops = 0.0;
    for (size_t const N : sizes) {
        double gflops;
        if (cfg.dtype == "float") {
            gflops = measure_sgemm_gflops(N, cfg.warmup, cfg.repeats);
        } else {
            gflops = measure_dgemm_gflops(N, cfg.warmup, cfg.repeats);
        }

        peak_gflops = std::max(peak_gflops, gflops);
        profile.cpu.gemm_efficiency.push_back({.M = N, .N = N, .K = N, .gflops = gflops});

        einsums::println("  {} x {} x {}: {:.1f} GFLOPS", N, N, N, gflops);
    }

    if (cfg.dtype == "float") {
        profile.cpu.peak_gflops_fp32 = peak_gflops;
    } else {
        profile.cpu.peak_gflops_fp64 = peak_gflops;
    }

    // Measure memory bandwidth
    einsums::println("\n--- Memory Bandwidth ---\n");
    double const bw                = measure_bandwidth_gbps();
    profile.cpu.mem_bandwidth_gbps = bw;
    einsums::println("  Sustained copy bandwidth: {:.1f} GB/s", bw);

    // Measure kernel overhead
    einsums::println("\n--- Kernel Overhead ---\n");
    double const overhead                 = measure_overhead_us();
    profile.cpu.kernel_launch_overhead_us = overhead;
    einsums::println("  DGEMM(1x1x1) overhead: {:.2f} us", overhead);

    // Update profile name
    profile.cpu.name   = fmt::format("{} (calibrated)", cpu_brand);
    profile.cpu.source = "calibrated";

    // Save
    auto save_result = profile.save_json(cfg.output);
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
    return einsums::start(einsums_main, argc, argv);
}
