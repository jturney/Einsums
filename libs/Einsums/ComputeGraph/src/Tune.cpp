//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The tune: measure this machine and write its hardware profile, or report on the profile in
// effect. One implementation behind every way of running it: the einsums-tune executable,
// `einsums tune` and `python -m einsums.tune`.

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Tune.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Hardware/Machine.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Hardware/Profile.hpp>
#include <Einsums/Json/Json.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorPermute/Permute.hpp>

#include <fmt/chrono.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#ifdef _OPENMP
#    include <omp.h>
#endif

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

namespace cg = einsums::compute_graph;

/// What the tune does when it is not measuring.
enum class Action : std::uint8_t { Measure, Show, Check, Import, Export };

struct Config {
    Action      action = Action::Measure;
    std::string import_path;
    std::string output; // empty: hardware::default_profile_path()
    std::string dtype          = "double";
    size_t      min_size       = 16;
    size_t      max_size       = 2048;
    size_t      num_points     = 8;
    size_t      warmup         = 3;
    size_t      repeats        = 10;
    bool        thread_sweep   = true;
    size_t      thread_repeats = 3;
    size_t      thread_max_mb  = 64;
    bool        probes         = true;
    size_t      probe_repeats  = 5;
};

void print_usage() {
    fmt::print("Usage: einsums-tune [options]\n");
    fmt::print("\nMeasure this machine and write its hardware profile, or report on the one in effect.\n\n");
    fmt::print("  --show             Print the profile in effect, where it came from, and why one is not used\n");
    fmt::print("  --check            Exit 0 when a current profile applies to this machine, 1 otherwise\n");
    fmt::print("  --import <path>    Adopt a profile measured elsewhere, filed under the machine it describes\n");
    fmt::print("  --export           Print the path of the profile in effect, to copy or share it\n");
    fmt::print("\nMeasuring:\n");
    fmt::print("  --output <path>    Profile to write (default: this machine's, in the cache directory)\n");
    fmt::print("  --dtype <type>     Data type: double or float (default: double)\n");
    fmt::print("  --min-size <N>     Smallest extent on each axis of the GEMM grid (default: 16)\n");
    fmt::print("  --max-size <N>     Largest extent on each axis of the GEMM grid (default: 2048)\n");
    fmt::print("  --num-points <N>   Extents per axis of the GEMM grid (default: 8)\n");
    fmt::print("  --warmup <N>       Warmup iterations (default: 3)\n");
    fmt::print("  --repeats <N>      Measurement iterations (default: 10)\n");
    fmt::print("  --no-thread-sweep  Skip the per-family efficiency curves\n");
    fmt::print("  --thread-repeats <N>  Timed runs per curve point (default: 3)\n");
    fmt::print("  --thread-max-mb <N>   Cap on a curve cell's working set (default: 64)\n");
    fmt::print("  --quick            Measure the cost model and region cost only, without the crossover probes\n");
    fmt::print("  --probe-repeats <N>   Timed runs per side and size in a crossover probe (default: 5)\n");
    fmt::print("\nEvery --einsums:* option also applies.\n");
}

/// The arguments, or why they cannot be read. --einsums:* options are the runtime's and are
/// skipped; anything else unknown is an error, so a misspelt size is not silently the default.
std::optional<Config> parse_args(std::vector<std::string> const &args, std::string &error) {
    Config     cfg;
    auto const size_at = [&](std::size_t &i, std::size_t &out) {
        if (i + 1 >= args.size()) {
            error = fmt::format("{} needs a value", args[i]);
            return false;
        }
        std::string const &text = args[++i];
        auto const [end, ec]    = std::from_chars(text.data(), text.data() + text.size(), out);
        if (ec != std::errc{} || end != text.data() + text.size()) {
            error = fmt::format("{} wants a count, not '{}'", args[i - 1], text);
            return false;
        }
        return true;
    };
    auto const text_at = [&](std::size_t &i, std::string &out) {
        if (i + 1 >= args.size()) {
            error = fmt::format("{} needs a value", args[i]);
            return false;
        }
        out = args[++i];
        return true;
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string const &arg = args[i];
        bool               ok  = true;
        if (arg.starts_with("--einsums:")) {
            continue;
        } else if (arg == "--show") {
            cfg.action = Action::Show;
        } else if (arg == "--check") {
            cfg.action = Action::Check;
        } else if (arg == "--export") {
            cfg.action = Action::Export;
        } else if (arg == "--import") {
            cfg.action = Action::Import;
            ok         = text_at(i, cfg.import_path);
        } else if (arg == "--output") {
            ok = text_at(i, cfg.output);
        } else if (arg == "--dtype") {
            ok = text_at(i, cfg.dtype);
            if (ok && cfg.dtype != "double" && cfg.dtype != "float") {
                error = fmt::format("--dtype is double or float, not '{}'", cfg.dtype);
                ok    = false;
            }
        } else if (arg == "--min-size") {
            ok = size_at(i, cfg.min_size);
        } else if (arg == "--max-size") {
            ok = size_at(i, cfg.max_size);
        } else if (arg == "--num-points") {
            ok = size_at(i, cfg.num_points);
        } else if (arg == "--warmup") {
            ok = size_at(i, cfg.warmup);
        } else if (arg == "--repeats") {
            ok = size_at(i, cfg.repeats);
        } else if (arg == "--no-thread-sweep") {
            cfg.thread_sweep = false;
        } else if (arg == "--thread-repeats") {
            ok = size_at(i, cfg.thread_repeats);
        } else if (arg == "--thread-max-mb") {
            ok = size_at(i, cfg.thread_max_mb);
        } else if (arg == "--quick") {
            cfg.probes = false;
        } else if (arg == "--probe-repeats") {
            ok = size_at(i, cfg.probe_repeats);
        } else {
            error = fmt::format("unknown argument '{}'; see --help", arg);
            ok    = false;
        }
        if (!ok) {
            return std::nullopt;
        }
    }
    if (cfg.repeats == 0 || cfg.probe_repeats == 0 || cfg.num_points == 0 || cfg.min_size == 0 || cfg.min_size > cfg.max_size) {
        error = "--repeats, --probe-repeats and --num-points must be positive, and --min-size at most --max-size";
        return std::nullopt;
    }
    return cfg;
}

/// A tensor of uniform values in [-1, 1), from a fixed seed so every run times the same data.
template <typename T, typename... Dims>
Tensor<T, sizeof...(Dims)> random_tensor(std::string name, Dims... dims) {
    Tensor<T, sizeof...(Dims)>        tensor(std::move(name), static_cast<std::size_t>(dims)...);
    std::mt19937                      engine(20261012U);
    std::uniform_real_distribution<T> uniform(T{-1}, T{1});
    T                                *data = tensor.data();
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        data[i] = uniform(engine);
    }
    return tensor;
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

/// Multiply-adds above which a GEMM grid point is skipped. The cost model reads past the largest
/// measured shape at the edge of the grid, and the largest shapes are most of a sweep's time.
constexpr double kGridMaxMultiplyAdds = 1073741824.0; // 2^30

/// Seconds for `c <- a ; b` on @p C, @p A and @p B through the packed engine, pinned to it as a
/// planned graph pins it, or NaN when the engine declines the contraction.
template <typename T, typename CType, typename AType, typename BType>
double packed_seconds(Config const &cfg, std::vector<std::string> c, std::vector<std::string> a, std::vector<std::string> b, CType &C,
                      AType const &A, BType const &B) {
    packed_gemm::ContractionSpec spec;
    spec.c_indices = std::move(c);
    spec.a_indices = std::move(a);
    spec.b_indices = std::move(b);
    packed_gemm::ContractionSite site;
    site.route           = packed_gemm::KernelRoute::Packed;
    bool         handled = true;
    double const seconds = median_seconds(
        cfg.warmup, cfg.repeats, [&] { handled = packed_gemm::try_packed_gemm(spec, T{0}, &C, T{1}, A, B, true, &site) && handled; });
    return handled ? seconds : std::numeric_limits<double>::quiet_NaN();
}

/// The GEMM rate of both engines over a grid of shapes, every M, N and K in @p sizes up to
/// kGridMaxMultiplyAdds: the routed one through the string einsum a graph's flat contraction
/// takes, the packed one through PackedGemm pinned to its tile kernel. Returns the routed peak.
template <typename T>
double measure_gemm_grid(Config const &cfg, std::vector<size_t> const &sizes, cg::DeviceProfile &profile) {
    double peak     = 0.0;
    size_t measured = 0;
    for (size_t const M : sizes) {
        for (size_t const N : sizes) {
            for (size_t const K : sizes) {
                if (static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) > kGridMaxMultiplyAdds) {
                    continue;
                }
                auto const   A     = random_tensor<T>("A", M, K);
                auto const   B     = random_tensor<T>("B", K, N);
                auto         C     = Tensor<T, 2>("C", M, N);
                double const flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
                double const routed =
                    median_seconds(cfg.warmup, cfg.repeats, [&] { cg::einsum("m,n <- m,k ; k,n", T{0}, &C, T{1}, A, B); });
                double const packed = packed_seconds<T>(cfg, {"m", "n"}, {"m", "k"}, {"k", "n"}, C, A, B);
                profile.gemm_efficiency.push_back(
                    {.M = M, .N = N, .K = K, .gflops = flops / (routed * 1e9), .engine = cg::GemmEngine::Routed});
                if (std::isfinite(packed)) {
                    profile.gemm_efficiency.push_back(
                        {.M = M, .N = N, .K = K, .gflops = flops / (packed * 1e9), .engine = cg::GemmEngine::Packed});
                }
                peak = std::max(peak, flops / (routed * 1e9));
                if (M == N && N == K) {
                    einsums::println("  {:>5} x {:>5} x {:>5}: {:>7.1f} GFLOPS routed, {:>7.1f} packed", M, N, K, flops / (routed * 1e9),
                                     std::isfinite(packed) ? flops / (packed * 1e9) : 0.0);
                }
                ++measured;
            }
        }
    }
    einsums::println("  {} shapes on a grid of {} extents per axis (cubes shown)", measured, sizes.size());
    return peak;
}

/// The scatter's extra cost per element of C by run length: the CCSD ladder's tile a,b,e,f <-
/// Q,a,e ; Q,b,f, whose C is scattered in runs of a x b, against the same contraction writing
/// a,e,b,f in place. M = N = 480 and K = 252 throughout, so only the runs change.
template <typename T>
std::vector<cg::RunCostPoint> measure_scatter_cost(Config const &cfg) {
    constexpr size_t              Q = 252, MN = 480; // off the powers of two, as grid_extents explains
    std::vector<cg::RunCostPoint> curve;
    for (size_t const side : {2UL, 4UL, 8UL, 16UL, 32UL}) {
        size_t const e        = MN / side;
        auto const   Ba       = random_tensor<T>("Ba", Q, side, e);
        auto const   Bb       = random_tensor<T>("Bb", Q, side, e);
        auto         W        = Tensor<T, 4>("W", side, side, e, e);
        auto         in_place = Tensor<T, 4>("W_flat", side, e, side, e);
        double const scatter  = packed_seconds<T>(cfg, {"a", "b", "e", "f"}, {"Q", "a", "e"}, {"Q", "b", "f"}, W, Ba, Bb);
        double const direct   = packed_seconds<T>(cfg, {"a", "e", "b", "f"}, {"Q", "a", "e"}, {"Q", "b", "f"}, in_place, Ba, Bb);
        if (!std::isfinite(scatter) || !std::isfinite(direct)) {
            continue;
        }
        double const ns = std::max(0.0, (scatter - direct) * 1e9 / static_cast<double>(MN * MN));
        curve.push_back({.run = side * side, .ns_per_element = ns});
        einsums::println("  runs of {:>4}: {:>6.3f} ns per element of C", side * side, ns);
    }
    return curve;
}

/// The packers' extra cost per element of an operand by run length: A with its free extent
/// split as x,k,y, contiguous in runs of x along the extent the packer walks, against the same A
/// laid out x,y,k and read in one run. M = N = 480 and K = 252 throughout, and C keeps x and y
/// side by side, so it is written in place either way.
template <typename T>
std::vector<cg::RunCostPoint> measure_pack_cost(Config const &cfg) {
    constexpr size_t              K = 252, MN = 480; // off the powers of two, as grid_extents explains
    std::vector<cg::RunCostPoint> curve;
    auto const                    B = random_tensor<T>("B", K, MN);
    for (size_t run = 2; run <= 32; run *= 2) {
        size_t const y        = MN / run;
        auto const   gathered = random_tensor<T>("A_split", run, K, y);
        auto const   flat     = random_tensor<T>("A_flat", run, y, K);
        auto         C        = Tensor<T, 3>("C", run, y, MN);
        double const split    = packed_seconds<T>(cfg, {"x", "y", "n"}, {"x", "k", "y"}, {"k", "n"}, C, gathered, B);
        double const whole    = packed_seconds<T>(cfg, {"x", "y", "n"}, {"x", "y", "k"}, {"k", "n"}, C, flat, B);
        if (!std::isfinite(split) || !std::isfinite(whole)) {
            continue;
        }
        double const ns = std::max(0.0, (split - whole) * 1e9 / static_cast<double>(MN * K));
        curve.push_back({.run = run, .ns_per_element = ns});
        einsums::println("  runs of {:>4}: {:>6.3f} ns per element of A", run, ns);
    }
    return curve;
}

/// Achieved (read + write) bandwidth of a rank-2 transpose of an N x N tensor.
double measure_permute_rank2_gbps(size_t N, size_t warmup, size_t repeats) {
    auto A = random_tensor<double>("A", N, N);
    auto C = Tensor<double, 2>("C", N, N);

    double const seconds = median_seconds(warmup, repeats, [&] { tensor_permute::permute("ji <- ij", 0.0, &C, 1.0, A); });

    double const bytes = 2.0 * static_cast<double>(N) * static_cast<double>(N) * sizeof(double);
    return bytes / (seconds * 1e9);
}

/// Same for a rank-3 cyclic permutation of an N x N x N tensor.
double measure_permute_rank3_gbps(size_t N, size_t warmup, size_t repeats) {
    auto A = random_tensor<double>("A", N, N, N);
    auto C = Tensor<double, 3>("C", N, N, N);

    double const seconds = median_seconds(warmup, repeats, [&] { tensor_permute::permute("kij <- ijk", 0.0, &C, 1.0, A); });

    double const bytes = 2.0 * std::pow(static_cast<double>(N), 3.0) * sizeof(double);
    return bytes / (seconds * 1e9);
}

/// Same for a rank-4 permutation that swaps both index pairs, the shape a
/// coupled-cluster residual keeps asking for.
double measure_permute_rank4_gbps(size_t N, size_t warmup, size_t repeats) {
    auto A = random_tensor<double>("A", N, N, N, N);
    auto C = Tensor<double, 4>("C", N, N, N, N);

    double const seconds = median_seconds(warmup, repeats, [&] { tensor_permute::permute("jilk <- ijkl", 0.0, &C, 1.0, A); });

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
/// Holds the calling thread to one thread for the duration of a measurement.
///
/// Every rate this tool records - GEMM GFLOPS, memory and cache bandwidth,
/// permute throughput, call overhead - is consumed as the WIDTH-1 figure. The
/// cost model says so in one line: `t(w) = t(1) / s(w)`. Measured at whatever
/// thread count the calibrating shell happened to have, the table is not t(1)
/// at all, and the model then divides it by s(w) a second time. On a ten-core
/// machine that made a serial 2048-cube GEMM look 3.2x faster than it is
/// (179.0 GFLOPS recorded against 55.9 actual), so every node in a captured
/// graph looked far cheaper than it was, fell under the thread planner's fork
/// floor, and the planner declined to widen anything.
///
/// Deliberately NOT applied to the thread-efficiency sweep, which sets its own
/// widths and measures the speedup between them, nor to the OpenMP region-cost
/// calibration, whose whole subject is what a fork costs at the real width.
///
/// For an OpenMP-threaded vendor - conda-forge's openblas, the usual case here
/// - the ICV governs the vendor too, so the omp call alone would do. The blas
/// call is what reaches MKL, which runs its own runtime and does not share our
/// ICVs.
struct [[maybe_unused]] SerialMeasurementScope {
#ifdef _OPENMP
    int const prior{omp_get_max_threads()};

    SerialMeasurementScope() {
        omp_set_num_threads(1);
        blas::set_num_threads_this_thread(1);
    }

    ~SerialMeasurementScope() {
        omp_set_num_threads(prior);
        blas::set_num_threads_this_thread(prior);
    }

    SerialMeasurementScope(SerialMeasurementScope const &)            = delete;
    SerialMeasurementScope &operator=(SerialMeasurementScope const &) = delete;
#endif
};

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

    // Through the string einsum, the call every contraction in a graph makes.
    for (size_t w = 0; w < 100; w++) {
        cg::einsum("m,n <- m,k ; k,n", 0.0, &C, 1.0, A, B);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
        cg::einsum("m,n <- m,k ; k,n", 0.0, &C, 1.0, A, B);
    }
    auto t1 = std::chrono::steady_clock::now();

    double const total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return total_us / static_cast<double>(NUM_ITERS);
}

// einsums_main() has no argc/argv; main() stashes them here before
// einsums::start so parse_args can see the tool's own flags (einsums has
// already consumed and removed its --einsums: arguments by this point).
/// Print the profile in effect, where it came from, and why one is not used.
int show() {
    auto const status = hardware::profile_status();
    einsums::println("Machine:  {} ({})", hardware::machine_fingerprint(), hardware::machine_identity().cpu);
    auto const &software = hardware::software_identity();
    einsums::println("Software: Einsums {}.{}.{}, BLAS {}", software.major, software.minor, software.patch, software.blas);
    einsums::println("Profile:  {}", hardware::to_string(status.state));
    if (!status.path.empty()) {
        einsums::println("Path:     {}", status.path);
    }
    if (!status.reason.empty()) {
        einsums::println("Reason:   {}", status.reason);
    }
    einsums::println("Searched:");
    for (auto const &place : hardware::profile_search_path()) {
        einsums::println("  {}", place);
    }
    if (auto const options = hardware::profile_options(); !options.empty()) {
        einsums::println("Options:");
        for (auto const &[name, value] : options) {
            einsums::println("  --{}={}", name, value);
        }
    }
    return EXIT_SUCCESS;
}

/// Succeed when a current profile applies to this machine, saying which, or say why not.
int check() {
    auto const status = hardware::profile_status();
    if (status.state == hardware::ProfileState::Loaded) {
        einsums::println("{}", status.path);
        return EXIT_SUCCESS;
    }
    if (status.state == hardware::ProfileState::None) {
        einsums::println("no profile for this machine; run einsums-tune to measure one");
    } else {
        einsums::println("{} ({}): {}", hardware::to_string(status.state), status.path, status.reason);
    }
    return EXIT_FAILURE;
}

/// Print the path of the profile in effect.
int export_path() {
    auto const status = hardware::profile_status();
    if (!status.in_use()) {
        einsums::println("no profile is in use{}", status.reason.empty() ? std::string{} : fmt::format(": {}", status.reason));
        return EXIT_FAILURE;
    }
    einsums::println("{}", status.path);
    return EXIT_SUCCESS;
}

int import_file(std::string const &path) {
    std::string message;
    bool const  filed = hardware::import_profile(path, message);
    einsums::println("{}", message);
    return filed ? EXIT_SUCCESS : EXIT_FAILURE;
}

// ── Crossover probes ──────────────────────────────────────────────────────────────────────────
//
// Each threshold that decides between two routes is measured by running the decision's own shape
// through the public einsum with the threshold forced to each extreme, so what is timed is what a
// program runs, and the route each side took is checked to differ: a side that silently takes a
// third route would otherwise time noise and write it into the profile.

/// An option held at a value for one scope, restored on the way out.
template <typename T>
class OptionPin {
  public:
    OptionPin(cl::ConfigOption<T> const &option, T value) : _option(option), _previous(config::get(option)) { config::set(option, value); }
    OptionPin(OptionPin const &)            = delete;
    OptionPin &operator=(OptionPin const &) = delete;
    ~OptionPin() { config::set(_option, _previous); }

  private:
    cl::ConfigOption<T> const &_option;
    T                          _previous;
};

/// How a probe's two sides are forced, and the call it times at one size.
struct ProbeSides {
    std::function<void()> small; ///< Forces the side small sizes get.
    std::function<void()> large; ///< Forces the side large sizes get.
    std::function<void()> reset; ///< Back to the configured threshold.
};

/// A sweep's points, or why the probe could not separate its sides.
struct Sweep {
    std::vector<detail::CrossoverPoint> points;
    std::string                         small_route;
    std::string                         large_route;
    std::string                         failure;
};

/// The route a call took: the string engine's, and the packed engine's path within it.
std::string route_taken() {
    std::string route{dispatch::last_dispatch_route()};
    if (std::string_view const path{packed_gemm::last_packed_decision()}; !path.empty()) {
        route += fmt::format("/{}", path);
    }
    return route;
}

/// Time both sides at every size, interleaved so drift lands on both. Unless @p same_route, the two
/// sides must take different routes at every size; with it, the same one, as when the sides differ
/// in how many threads run a path rather than in the path.
Sweep sweep(std::vector<double> const &sizes, std::function<std::function<void()>(double)> const &make, ProbeSides const &sides,
            size_t repeats, bool same_route = false) {
    Sweep result;
    for (double const size : sizes) {
        auto const          run = make(size);
        std::vector<double> small_times;
        std::vector<double> large_times;
        std::string         small_route;
        std::string         large_route;
        auto const          timed = [&run](std::vector<double> &times) {
            auto const t0 = std::chrono::steady_clock::now();
            run();
            auto const t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(t1 - t0).count());
        };
        // One untimed call of each side first, for the caches and any plan built on first use.
        sides.small();
        run();
        sides.large();
        run();
        for (size_t r = 0; r < repeats; ++r) {
            sides.small();
            timed(small_times);
            small_route = route_taken();
            sides.large();
            timed(large_times);
            large_route = route_taken();
        }
        sides.reset();
        if (!same_route && small_route == large_route) {
            result.failure = fmt::format("both sides took the route '{}' at size {:.0f}", small_route, size);
            return result;
        }
        if (same_route && small_route != large_route) {
            result.failure = fmt::format("the sides took '{}' and '{}' at size {:.0f}", small_route, large_route, size);
            return result;
        }
        result.small_route = small_route;
        result.large_route = large_route;
        std::ranges::sort(small_times);
        std::ranges::sort(large_times);
        result.points.push_back({.size = size, .small_side = small_times[repeats / 2], .large_side = large_times[repeats / 2]});
    }
    return result;
}

/// Sizes from @p lo to @p hi, @p count of them, evenly spaced in the logarithm and rounded.
std::vector<double> log_sizes(double lo, double hi, size_t count) {
    std::vector<double> sizes;
    for (size_t i = 0; i < count; ++i) {
        double const t = count > 1 ? static_cast<double>(i) / static_cast<double>(count - 1) : 0.0;
        double const s = std::round(std::exp(std::log(lo) + t * (std::log(hi) - std::log(lo))));
        if (sizes.empty() || s > sizes.back()) {
            sizes.push_back(s);
        }
    }
    return sizes;
}

/// The value a probe writes, said the way the run log says it.
std::string describe(detail::Crossover const &c) {
    switch (c.bound) {
    case detail::CrossoverBound::AtMost:
        return fmt::format("at most {:.0f}", c.value);
    case detail::CrossoverBound::AtLeast:
        return fmt::format("at least {:.0f}", c.value);
    case detail::CrossoverBound::Exact:
        break;
    }
    return fmt::format("{:.0f}", c.value);
}

/// Report a crossover probe and, when it separated its sides, record its option.
void record(json::Object &options, std::string_view option, std::string_view what, Sweep const &result, double scale = 1.0) {
    if (!result.failure.empty()) {
        einsums::println("  {:<44} not measured: {}", what, result.failure);
        return;
    }
    auto const crossover = detail::find_crossover(result.points);
    einsums::println("  {:<44} {} ({} below, {} above)", what, describe(crossover), result.small_route, result.large_route);
    options.set(std::string(option), static_cast<std::int64_t>(std::llround(crossover.value * scale)));
}

/// A threshold no size reaches: 2^40, the options' upper bound.
constexpr std::int64_t kNever = 1'099'511'627'776;

/// Output elements from which an outer product of rank three takes the packed passes.
void probe_outer_product_floor(json::Object &options, size_t repeats) {
    OptionPin<std::int64_t> const hold(option::PackedGemmOuterProductFloor, config::get(option::PackedGemmOuterProductFloor));
    ProbeSides const              sides{
        .small = [] { config::set(option::PackedGemmOuterProductFloor, kNever); },
        .large = [] { config::set(option::PackedGemmOuterProductFloor, std::int64_t{0}); },
        .reset = [] {},
    };
    // Below 4096 output elements, where the matrix-shaped direct paths leave outer products to it.
    std::vector<double> sizes;
    for (double const m : {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}) {
        sizes.push_back(m * m * m);
    }
    auto const make = [](double elements) -> std::function<void()> {
        auto const m = static_cast<size_t>(std::llround(std::cbrt(elements)));
        auto       A = std::make_shared<Tensor<double, 2>>(random_tensor<double>("A", m, m));
        auto       B = std::make_shared<Tensor<double, 1>>(random_tensor<double>("B", m));
        auto       C = std::make_shared<Tensor<double, 3>>("C", m, m, m);
        return [A, B, C] { einsum("ijk <- ij ; k", 0.0, C.get(), 1.0, *A, *B); };
    };
    record(options, "einsums:packed-gemm:outer-product-floor", "outer-product floor (elements)", sweep(sizes, make, sides, repeats));
}

/// Output elements from which an outer product with a prefactor takes ger over a k=1 GEMM.
void probe_outer_gemm_max(json::Object &options, size_t repeats) {
    OptionPin<std::int64_t> const hold(option::PackedGemmOuterGemmMax, config::get(option::PackedGemmOuterGemmMax));
    ProbeSides const              sides{
        .small = [] { config::set(option::PackedGemmOuterGemmMax, kNever); },
        .large = [] { config::set(option::PackedGemmOuterGemmMax, std::int64_t{0}); },
        .reset = [] {},
    };
    // A rank-2 outer product takes the string engine's own GER route, so the decision is the packed
    // engine's matrix-shaped direct path for a higher rank, from 4096 elements where that path starts.
    std::vector<double> sizes;
    for (double const m : log_sizes(16, 100, 9)) {
        sizes.push_back(m * m * m);
    }
    auto const make = [](double elements) -> std::function<void()> {
        auto const m = static_cast<size_t>(std::llround(std::cbrt(elements)));
        auto       A = std::make_shared<Tensor<double, 2>>(random_tensor<double>("A", m, m));
        auto       B = std::make_shared<Tensor<double, 1>>(random_tensor<double>("B", m));
        auto       C = std::make_shared<Tensor<double, 3>>(random_tensor<double>("C", m, m, m));
        return [A, B, C] { einsum("ijk <- ij ; k", 0.5, C.get(), 1.0, *A, *B); };
    };
    record(options, "einsums:packed-gemm:outer-gemm-max", "outer product, GEMM to ger (elements)", sweep(sizes, make, sides, repeats));
}

/// The exchange shape K(i,j) = A(i,k,j,l) B(k,l) at @p n per axis.
std::function<void()> exchange(size_t n) {
    auto A = std::make_shared<Tensor<double, 4>>(random_tensor<double>("A", n, n, n, n));
    auto B = std::make_shared<Tensor<double, 2>>(random_tensor<double>("B", n, n));
    auto C = std::make_shared<Tensor<double, 2>>("C", n, n);
    return [A, B, C] { einsum("ij <- ikjl ; kl", 0.0, C.get(), 1.0, *A, *B); };
}

/// Supplying-operand elements from which an interleaved GEMV-shaped contraction is streamed.
void probe_stream_min_elems(json::Object &options, size_t repeats) {
    OptionPin<std::int64_t> const hold(option::PackedGemmStreamMinElems, config::get(option::PackedGemmStreamMinElems));
    ProbeSides const              sides{
        .small = [] { config::set(option::PackedGemmStreamMinElems, kNever); },
        .large = [] { config::set(option::PackedGemmStreamMinElems, std::int64_t{0}); },
        .reset = [] {},
    };
    std::vector<double> sizes;
    for (double const n : {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0}) {
        sizes.push_back(n * n * n * n);
    }
    auto const make = [](double elements) { return exchange(static_cast<size_t>(std::llround(std::sqrt(std::sqrt(elements))))); };
    record(options, "einsums:packed-gemm:stream-min-elems", "stream minimum (operand elements)", sweep(sizes, make, sides, repeats));
}

/// Streamed elements per thread: not a crossover between two routes but a team size, so each
/// candidate is timed over a range of exchange sizes and the one closest to the best at every size
/// is kept.
void probe_stream_elems_per_thread(json::Object &options, size_t repeats) {
#ifdef _OPENMP
    if (omp_get_max_threads() <= 1) {
        einsums::println("  {:<44} not measured: one thread, so the stream never forks", "stream elements per thread");
        return;
    }
#else
    einsums::println("  {:<44} not measured: built without OpenMP", "stream elements per thread");
    return;
#endif
    OptionPin<std::int64_t> const    hold(option::PackedGemmStreamElemsPerThread, config::get(option::PackedGemmStreamElemsPerThread));
    std::vector<std::int64_t> const  candidates{8192, 16384, 32768, 65536, 131072, 262144, 524288};
    std::vector<size_t> const        extents{12, 16, 20, 24, 28, 32};
    std::vector<std::vector<double>> times(candidates.size(), std::vector<double>(extents.size()));
    for (size_t e = 0; e < extents.size(); ++e) {
        auto const                       run = exchange(extents[e]);
        std::vector<std::vector<double>> samples(candidates.size());
        run();
        for (size_t r = 0; r < repeats; ++r) {
            for (size_t c = 0; c < candidates.size(); ++c) {
                config::set(option::PackedGemmStreamElemsPerThread, candidates[c]);
                auto const t0 = std::chrono::steady_clock::now();
                run();
                auto const t1 = std::chrono::steady_clock::now();
                samples[c].push_back(std::chrono::duration<double>(t1 - t0).count());
            }
        }
        for (size_t c = 0; c < candidates.size(); ++c) {
            std::ranges::sort(samples[c]);
            times[c][e] = samples[c][repeats / 2];
        }
    }
    // Score each candidate by the geometric mean of its time over the best at each size.
    size_t best       = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < candidates.size(); ++c) {
        double log_sum = 0.0;
        for (size_t e = 0; e < extents.size(); ++e) {
            double fastest = std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < candidates.size(); ++k) {
                fastest = std::min(fastest, times[k][e]);
            }
            log_sum += std::log(times[c][e] / fastest);
        }
        double const score = std::exp(log_sum / static_cast<double>(extents.size()));
        if (score < best_score) {
            best_score = score;
            best       = c;
        }
    }
    einsums::println("  {:<44} {} ({:.2f}x the best at each size, on average)", "stream elements per thread", candidates[best], best_score);
    options.set("einsums:packed-gemm:stream-elems-per-thread", candidates[best]);
}

/// Flops per nanosecond of region cost from which a packed contraction is worth a team.
void probe_parallel_break_even(json::Object &options, size_t repeats) {
    double const region_ns = hardware::omp_region_cost_ns();
    if (region_ns <= 0.0) {
        einsums::println("  {:<44} not measured: one thread, so nothing forks", "parallel break-even (flops/ns)");
        return;
    }
    ProbeSides const sides{
        .small = [] { packed_gemm::set_min_parallel_flops_for_tuning(std::numeric_limits<std::int64_t>::max()); },
        .large = [] { packed_gemm::set_min_parallel_flops_for_tuning(0); },
        .reset = [] { packed_gemm::set_min_parallel_flops_for_tuning(-1); },
    };
    // A contraction whose output interleaves its operands' axes, so it takes the packed engine
    // rather than a vendor GEMM, at a cube of extent m per axis pair: 2 m^3 n^2 flops.
    std::vector<double> sizes;
    for (double const m : log_sizes(4, 48, 10)) {
        sizes.push_back(2.0 * m * m * m * m);
    }
    auto const make = [](double flops) -> std::function<void()> {
        auto const m = static_cast<size_t>(std::llround(std::pow(flops / 2.0, 0.25)));
        auto       A = std::make_shared<Tensor<double, 3>>(random_tensor<double>("A", m, m, m));
        auto       B = std::make_shared<Tensor<double, 2>>(random_tensor<double>("B", m, m));
        auto       C = std::make_shared<Tensor<double, 3>>("C", m, m, m);
        return [A, B, C] { einsum("ibj <- ikj ; kb", 0.0, C.get(), 1.0, *A, *B); };
    };
    auto result = sweep(sizes, make, sides, repeats, /*same_route=*/true);
    sides.reset();
    if (!result.failure.empty()) {
        einsums::println("  {:<44} not measured: {}", "parallel break-even (flops/ns)", result.failure);
        return;
    }
    auto const   crossover = detail::find_crossover(result.points);
    double const rate      = std::clamp(crossover.value / region_ns, 1e-3, 1e6);
    einsums::println("  {:<44} {:.3g} ({} flops over a {:.0f} ns region)", "parallel break-even (flops/ns)", rate, describe(crossover),
                     region_ns);
    options.set("einsums:hardware:omp-break-even-flops-per-ns", rate);
}

int measure(Config cfg) {
    if (cfg.output.empty()) {
        cfg.output = einsums::hardware::default_profile_path();
        if (cfg.output.empty()) {
            einsums::println("ERROR: there is no cache directory to write the profile to; pass --output or --einsums:cache-dir");
            return EXIT_FAILURE;
        }
    }

    einsums::println("=== einsums-tune ===\n");
    einsums::println("Machine: {}", einsums::hardware::machine_fingerprint());
    einsums::println("Output: {}", cfg.output);
    einsums::println("Data type: {}", cfg.dtype);
    einsums::println("GEMM grid: {} to {}, {} extents per axis", cfg.min_size, cfg.max_size, cfg.num_points);
    einsums::println("Warmup: {}, Repeats: {}", cfg.warmup, cfg.repeats);

    // Start from the built-in table, never from an earlier profile, so nothing
    // this run does not measure is inherited from one.
    cg::CostModel cost_model = cg::CostModel::built_in();
    cost_model.source        = "calibrated";

    // Detect CPU brand
    std::string cpu_brand = cg::DeviceProfileDB::detect_cpu_brand();
    einsums::println("Detected CPU: {}", cpu_brand);

    // Generate measurement sizes
    auto sizes = detail::grid_extents(cfg.min_size, cfg.max_size, cfg.num_points);

    // Everything from here to the end of the kernel-overhead block is a
    // width-1 rate by contract, so it is measured at width 1. See
    // SerialMeasurementScope. The scope closes before max_threads is read and
    // before the thread sweep, both of which need the real thread count.
    double peak_gflops = 0.0;
    {
        SerialMeasurementScope const serial;

        // The GEMM rate of both engines over a grid of shapes, and the scatter's and the packers'
        // extra costs by run length, all through the entry points a graph's contractions take.
        einsums::println("\n--- GEMM Grid ---\n");
        cost_model.cpu.gemm_efficiency.clear();
        peak_gflops = cfg.dtype == "float" ? measure_gemm_grid<float>(cfg, sizes, cost_model.cpu)
                                           : measure_gemm_grid<double>(cfg, sizes, cost_model.cpu);
        if (cfg.dtype == "float") {
            cost_model.cpu.peak_gflops_fp32 = peak_gflops;
        } else {
            cost_model.cpu.peak_gflops_fp64 = peak_gflops;
        }

        einsums::println("\n--- C Scatter ---\n");
        cost_model.cpu.scatter_cost = cfg.dtype == "float" ? measure_scatter_cost<float>(cfg) : measure_scatter_cost<double>(cfg);
        einsums::println("\n--- Operand Packing ---\n");
        cost_model.cpu.pack_cost = cfg.dtype == "float" ? measure_pack_cost<float>(cfg) : measure_pack_cost<double>(cfg);

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
        einsums::println("  einsum(2x2x2) call overhead: {:.2f} us", overhead);
    } // end SerialMeasurementScope

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

    // The OpenMP region cost, swept across team sizes. It lives in the same
    // file as the cost model but is read by the Hardware module, since things
    // that never load a cost model consume it too (PackedGemm's parallel gate,
    // the DLPNO bucket chooser through the Python binding).
    einsums::println("\n--- OpenMP Region Cost ---\n");
    json::Object region_cost = einsums::hardware::measure_region_cost_section();
    for (std::size_t idx = 0; idx < region_cost.size(); ++idx) {
        einsums::println("  {:>3} threads: {:.0f} ns", region_cost.key_at(idx), region_cost.value_at(idx).as_double());
    }

    // One file: the header saying what it describes, then each module's section.
    json::Object measured;
    measured.set("date", fmt::format("{:%Y-%m-%d}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now())));
    json::Object profile = einsums::hardware::profile_header();
    profile.set("measured", std::move(measured));
    profile.set("omp_region_cost_ns", std::move(region_cost));
    if (cfg.probes) {
        einsums::println("\n--- Crossovers ---\n");
        json::Object options;
        probe_outer_product_floor(options, cfg.probe_repeats);
        probe_outer_gemm_max(options, cfg.probe_repeats);
        probe_stream_min_elems(options, cfg.probe_repeats);
        probe_stream_elems_per_thread(options, cfg.probe_repeats);
        probe_parallel_break_even(options, cfg.probe_repeats);
        if (!options.empty()) {
            profile.set("options", std::move(options));
        }
    }
    profile.set("cost_model", cost_model.to_json());

    std::string error;
    if (!einsums::hardware::write_profile(cfg.output, json::Value{std::move(profile)}, error)) {
        einsums::println("ERROR: {}", error);
        return EXIT_FAILURE;
    }
    einsums::println("\nProfile saved to: {}", cfg.output);
    einsums::println("Peak GFLOPS ({}): {:.1f}", cfg.dtype, peak_gflops);

    return EXIT_SUCCESS;
}

} // namespace

int tune(std::vector<std::string> const &args) {
    if (std::ranges::any_of(args, [](std::string const &arg) { return arg == "--help" || arg == "-h"; })) {
        print_usage();
        std::fflush(stdout);
        return EXIT_SUCCESS;
    }
    std::string error;
    auto const  cfg = parse_args(args, error);
    if (!cfg) {
        fmt::print(stderr, "einsums-tune: {}\n", error);
        return 2;
    }
    int status = EXIT_SUCCESS;
    switch (cfg->action) {
    case Action::Show:
        status = show();
        break;
    case Action::Check:
        status = check();
        break;
    case Action::Export:
        status = export_path();
        break;
    case Action::Import:
        status = import_file(cfg->import_path);
        break;
    case Action::Measure:
        status = measure(*cfg);
        break;
    }
    // Python captures output through the file, not the stream buffer.
    std::fflush(stdout);
    return status;
}

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

std::vector<size_t> grid_extents(size_t min_val, size_t max_val, size_t num_points) {
    std::vector<size_t> sizes;
    double const        log_min = std::log2(static_cast<double>(min_val));
    double const        log_max = std::log2(static_cast<double>(max_val));

    for (size_t idx = 0; idx < num_points; idx++) {
        double const t     = (num_points > 1) ? static_cast<double>(idx) / static_cast<double>(num_points - 1) : 0.0;
        double const log_s = log_min + t * (log_max - log_min);
        auto         s     = static_cast<size_t>(std::round(std::pow(2.0, log_s)));
        // Round to nearest multiple of 4 for alignment
        s = std::max(static_cast<size_t>(4), (s + 3) / 4 * 4);
        if (s >= 16 && (s & (s - 1)) == 0) {
            s -= 4;
        }
        if (sizes.empty() || sizes.back() != s)
            sizes.push_back(s);
    }
    return sizes;
}

Crossover find_crossover(std::vector<CrossoverPoint> const &points, double tie) {
    // The large side loses at a point when it is slower by more than the tie band.
    auto const loses = [tie](CrossoverPoint const &p) { return p.large_side > p.small_side * (1.0 + tie); };
    size_t     first = points.size();
    while (first > 0 && !loses(points[first - 1])) {
        --first;
    }
    if (first == 0) {
        return {.value = points.front().size, .bound = CrossoverBound::AtMost};
    }
    if (first == points.size()) {
        return {.value = points.back().size, .bound = CrossoverBound::AtLeast};
    }
    return {.value = std::sqrt(points[first - 1].size * points[first].size), .bound = CrossoverBound::Exact};
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
