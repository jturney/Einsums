//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// SIMD kernel benchmarks, on every dispatch rung this machine supports, in one run:
//
//   gather          a table lookup per lane, from an L1-sized and a memory-sized table
//   convert         float/double widening and narrowing, double to the 64- and 32-bit integers
//   interpolation   a six-term Taylor series about the grid point below x, the shape of the Boys
//                   function inside its table: FP32, FP64, FP64 at FP32's lane count with a 64-bit
//                   and with a shared 32-bit index, and the scalar instantiation of the same body
//   exp             e^-T over the range the Boys recursion takes, in the same tiers
//
// Each label names the rung, so one run compares rungs, and the interpolation labels compare the
// precision tiers: the FP64-at-FP32-width variants against plain FP64 show what the second register
// per vector costs.

#include "BenchmarkKernels.hpp"

#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/SIMD/RungLadder.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums::performance;
namespace simd = einsums::simd;

namespace simd_bench {

struct Rung {
    simd::InstructionSet set;
    Kernels const       *kernels;
};

/// Every rung that was built and that this machine can run, lowest first.
std::vector<Rung> runnable_rungs() {
    using Fn                           = Kernels const &(*)() noexcept;
    Fn const                   slots[] = {EINSUMS_SIMD_LADDER(kernels)};
    simd::InstructionSet const sets[]  = {simd::InstructionSet::Baseline, simd::InstructionSet::V2, simd::InstructionSet::V3,
                                          simd::InstructionSet::V4, simd::InstructionSet::Sme};
    std::vector<Rung>          out;
    for (int i = 0; i < 5; ++i) {
        if (slots[i] != nullptr && simd::supports(simd::cpu_features(), sets[i])) {
            out.push_back({sets[i], &slots[i]()});
        }
    }
    return out;
}

} // namespace simd_bench

namespace {

/// Elements per call: a multiple of every lane count, and small enough that the inputs stay in L2.
constexpr std::size_t n = std::size_t{1} << 14;

/// Grid points of the interpolation table, as many as a Boys table holds for one order.
constexpr std::size_t grid_points = 400;

template <typename T>
struct OwnedTable {
    std::vector<T>       c[simd_bench::taylor_terms];
    simd_bench::Table<T> view;

    OwnedTable() {
        T const dx = T(0.125);
        for (int k = 0; k < simd_bench::taylor_terms; ++k) {
            c[k].resize(grid_points);
            for (std::size_t g = 0; g < grid_points; ++g) {
                c[k][g] = static_cast<T>(std::exp(-static_cast<double>(g) * static_cast<double>(dx)) / std::tgamma(k + 1.0));
            }
            view.c[k] = c[k].data();
        }
        view.dx     = dx;
        view.inv_dx = T(1) / dx;
    }
};

void bench(std::string const &label, simd::InstructionSet set, auto &&fn) {
    std::string const full = label + " [" + simd::to_string(set) + "]";
    ProfileAnnotate("rung", simd::to_string(set));
    publish_benchmark_result(full.c_str(), "t_us", static_cast<int>(n), time_us(full.c_str(), fn, 50));
}

} // namespace

EINSUMS_TEST_CASE("SIMD gather, conversion and interpolation on every supported rung", "[performance][simd]") {
    std::mt19937 rng(20261001);

    // Gather indices into a table that fits L1 and one that does not.
    for (std::size_t table_size : {std::size_t{4096}, std::size_t{1} << 22}) {
        std::uniform_int_distribution<int32_t> pick(0, static_cast<int32_t>(table_size - 1));
        std::vector<int32_t>                   idx32(n);
        std::vector<int64_t>                   idx64(n);
        for (std::size_t i = 0; i < n; ++i) {
            idx32[i] = pick(rng);
            idx64[i] = idx32[i];
        }
        std::vector<float>  tf(table_size, 1.5f), of(n);
        std::vector<double> td(table_size, 2.5), od(n);
        std::string const   where = table_size == 4096 ? " (L1 table)" : " (memory table)";
        for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
            bench("gather f32" + where, r.set, [&] { r.kernels->gather_f32(tf.data(), idx32.data(), of.data(), n); });
            bench("gather f64" + where, r.set, [&] { r.kernels->gather_f64(td.data(), idx64.data(), od.data(), n); });
            bench("gather f64 at i32" + where, r.set, [&] { r.kernels->gather_f64_i32(td.data(), idx32.data(), od.data(), n); });
        }
    }

    {
        std::uniform_real_distribution<double> value(-1.0e6, 1.0e6);
        std::vector<float>                     f(n);
        std::vector<double>                    d(n);
        std::vector<int64_t>                   i64(n);
        std::vector<int32_t>                   i32(n);
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = value(rng);
            f[i] = static_cast<float>(d[i]);
        }
        for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
            bench("convert f32 to f64", r.set, [&] { r.kernels->widen_f32_f64(f.data(), d.data(), n); });
            bench("convert f64 to f32", r.set, [&] { r.kernels->narrow_f64_f32(d.data(), f.data(), n); });
            bench("convert f64 to i64", r.set, [&] { r.kernels->f64_to_i64(d.data(), i64.data(), n); });
            bench("convert f64 to i32", r.set, [&] { r.kernels->f64_to_i32(d.data(), i32.data(), n); });
        }
    }

    {
        OwnedTable<float>                      tf;
        OwnedTable<double>                     td;
        std::uniform_real_distribution<double> arg(0.0, 49.0); // inside the 400-point table
        std::vector<float>                     xf(n), of(n);
        std::vector<double>                    xd(n), od(n);
        for (std::size_t i = 0; i < n; ++i) {
            xd[i] = arg(rng);
            xf[i] = static_cast<float>(xd[i]);
        }
        for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
            bench("interpolation f32", r.set, [&] { r.kernels->interp_f32(tf.view, xf.data(), of.data(), n); });
            bench("interpolation f64", r.set, [&] { r.kernels->interp_f64(td.view, xd.data(), od.data(), n); });
            bench("interpolation f64 at f32 width", r.set, [&] { r.kernels->interp_f64_wide(td.view, xd.data(), od.data(), n); });
            bench("interpolation f64 at f32 width, i32 index", r.set,
                  [&] { r.kernels->interp_f64_wide_i32(td.view, xd.data(), od.data(), n); });
            bench("interpolation f64 scalar body", r.set, [&] { r.kernels->interp_f64_scalar(td.view, xd.data(), od.data(), n); });
        }
    }

    {
        std::uniform_real_distribution<double> arg(-50.0, 0.0); // e^-T, as the Boys recursion takes it
        std::vector<float>                     xf(n), of(n);
        std::vector<double>                    xd(n), od(n);
        for (std::size_t i = 0; i < n; ++i) {
            xd[i] = arg(rng);
            xf[i] = static_cast<float>(xd[i]);
        }
        for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
            bench("exp f32", r.set, [&] { r.kernels->exp_f32(xf.data(), of.data(), n); });
            bench("exp f64", r.set, [&] { r.kernels->exp_f64(xd.data(), od.data(), n); });
            bench("exp f64 at f32 width", r.set, [&] { r.kernels->exp_f64_wide(xd.data(), od.data(), n); });
            bench("exp f64 scalar body", r.set, [&] { r.kernels->exp_f64_scalar(xd.data(), od.data(), n); });
        }
    }

    {
        std::uniform_real_distribution<double> arg(-6.0, 6.0);
        std::uniform_real_distribution<double> pos(0.01, 100.0);
        std::vector<float>                     xf(n), of(n);
        std::vector<double>                    xd(n), od(n), pd(n);
        for (std::size_t i = 0; i < n; ++i) {
            xd[i] = arg(rng);
            xf[i] = static_cast<float>(xd[i]);
            pd[i] = pos(rng);
        }
        for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
            bench("erf f64", r.set, [&] { r.kernels->erf_f64(xd.data(), od.data(), n); });
            bench("erfc f64", r.set, [&] { r.kernels->erfc_f64(xd.data(), od.data(), n); });
            bench("erfc f32", r.set, [&] { r.kernels->erfc_f32(xf.data(), of.data(), n); });
            bench("rsqrt f64", r.set, [&] { r.kernels->rsqrt_f64(pd.data(), od.data(), n); });
            bench("rsqrt f64, hardware estimate + Newton", r.set, [&] { r.kernels->rsqrt_f64_estimate(pd.data(), od.data(), n); });
        }
    }

    // Segmented reduction over group-size distributions an integral code meets: fixed small sizes,
    // contraction degrees (K^4 primitive quartets per contracted quartet for K = 2, 3), and screened
    // groups of random size. Each strategy is checked against the scalar sum once; the padding line
    // reports what padding every group to a whole number of vectors would cost the integral kernel
    // in wasted lanes, for every vector width this machine has.
    {
        struct Distribution {
            std::string name;
            int         lo, hi; // group sizes uniform in [lo, hi]
        };
        std::vector<Distribution> const distributions = {{"size 1", 1, 1},
                                                         {"size 3", 3, 3},
                                                         {"size 4", 4, 4},
                                                         {"size 7", 7, 7},
                                                         {"size 16 (K = 2)", 16, 16},
                                                         {"size 81 (K = 3)", 81, 81},
                                                         {"sizes 1-8", 1, 8},
                                                         {"sizes 1-16", 1, 16},
                                                         {"sizes 1-100", 1, 100}};
        for (Distribution const &d : distributions) {
            std::uniform_int_distribution<int> size(d.lo, d.hi);
            std::vector<int32_t>               offset{0};
            while (offset.back() < static_cast<int32_t>(n)) {
                offset.push_back(offset.back() + size(rng));
            }
            offset.back()                                 = static_cast<int32_t>(n); // the last group ends at n
            int const                              groups = static_cast<int>(offset.size()) - 1;
            std::uniform_real_distribution<double> value(-1.0, 1.0);
            std::vector<double>                    v(n);
            for (double &x : v) {
                x = value(rng);
            }
            std::vector<double> reference(static_cast<std::size_t>(groups)), out(static_cast<std::size_t>(groups));

            for (int width : {2, 4, 8}) {
                std::size_t padded = 0;
                for (int g = 0; g < groups; ++g) {
                    padded += static_cast<std::size_t>((offset[g + 1] - offset[g] + width - 1) / width * width);
                }
                std::printf("[segmented %s] padding to %d lanes: %.1f%% of kernel lanes wasted\n", d.name.c_str(), width,
                            100.0 * static_cast<double>(padded - n) / static_cast<double>(padded));
            }

            for (simd_bench::Rung const &r : simd_bench::runnable_rungs()) {
                r.kernels->segsum_scalar(v.data(), offset.data(), groups, reference.data());
                for (auto const &[label, fn] : {std::pair{std::string("horizontal"), r.kernels->segsum_horizontal},
                                                std::pair{std::string("transpose"), r.kernels->segsum_transpose}}) {
                    fn(v.data(), offset.data(), groups, out.data());
                    for (int g = 0; g < groups; ++g) {
                        REQUIRE(std::fabs(out[g] - reference[g]) <= 1e-12 * (1.0 + std::fabs(reference[g])));
                    }
                }
                bench("segmented sum " + d.name + ": scalar", r.set,
                      [&] { r.kernels->segsum_scalar(v.data(), offset.data(), groups, out.data()); });
                bench("segmented sum " + d.name + ": horizontal", r.set,
                      [&] { r.kernels->segsum_horizontal(v.data(), offset.data(), groups, out.data()); });
                bench("segmented sum " + d.name + ": transpose", r.set,
                      [&] { r.kernels->segsum_transpose(v.data(), offset.data(), groups, out.data()); });
                bench("segmented sum " + d.name + ": scalar tail", r.set,
                      [&] { r.kernels->segsum_scalar_tail(v.data(), offset.data(), groups, out.data()); });

                // The interleaved layout, with groups in arrival order and with groups sorted by
                // size first (a producer batching by contraction degree); the zero rows past a
                // group's end are lanes the integral kernel computed for nothing.
                int const L = r.kernels->vector_bits / 64;
                for (bool sorted : {false, true}) {
                    std::vector<int> order(static_cast<std::size_t>(groups));
                    std::iota(order.begin(), order.end(), 0);
                    if (sorted) {
                        std::stable_sort(order.begin(), order.end(),
                                         [&](int a, int b) { return offset[a + 1] - offset[a] < offset[b + 1] - offset[b]; });
                    }
                    int const            batches = (groups + L - 1) / L;
                    std::vector<int32_t> rows(static_cast<std::size_t>(batches));
                    std::vector<double>  tiles;
                    for (int b = 0; b < batches; ++b) {
                        for (int k = 0; k < L && b * L + k < groups; ++k) {
                            int const g = order[static_cast<std::size_t>(b * L + k)];
                            rows[b]     = std::max(rows[b], offset[g + 1] - offset[g]);
                        }
                        std::size_t const base = tiles.size();
                        tiles.resize(base + static_cast<std::size_t>(rows[b]) * L, 0.0);
                        for (int k = 0; k < L && b * L + k < groups; ++k) {
                            int const g = order[static_cast<std::size_t>(b * L + k)];
                            for (int32_t i = offset[g]; i < offset[g + 1]; ++i) {
                                tiles[base + static_cast<std::size_t>(i - offset[g]) * L + k] = v[i];
                            }
                        }
                    }
                    std::vector<double> sums(static_cast<std::size_t>(batches) * L);
                    r.kernels->segsum_interleaved(tiles.data(), rows.data(), batches, sums.data());
                    for (int j = 0; j < groups; ++j) {
                        double const want = reference[order[j]];
                        REQUIRE(std::fabs(sums[j] - want) <= 1e-12 * (1.0 + std::fabs(want)));
                    }
                    std::string const which = sorted ? "interleaved, sorted" : "interleaved";
                    std::printf("[segmented %s] %s at %d lanes: %.1f%% of kernel lanes wasted\n", d.name.c_str(), which.c_str(), L,
                                100.0 * static_cast<double>(tiles.size() - n) / static_cast<double>(tiles.size()));
                    bench("segmented sum " + d.name + ": " + which, r.set,
                          [&] { r.kernels->segsum_interleaved(tiles.data(), rows.data(), batches, sums.data()); });
                }
            }
        }
    }
}
