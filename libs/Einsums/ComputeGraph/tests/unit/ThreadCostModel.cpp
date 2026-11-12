//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ThreadCostModel.cpp
/// @brief Tests for the thread axis of the cost model: efficiency curves, the
///        default curve, and their JSON round trip.

#include <Einsums/ComputeGraph/CostModel.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

namespace cg = einsums::compute_graph;

namespace {

/// A profile with cache levels pinned, so the size classes a test asks for do
/// not move with the machine running it.
cg::DeviceProfile pinned_profile() {
    cg::DeviceProfile p;
    p.name               = "Pinned";
    p.peak_gflops_fp64   = 100.0;
    p.mem_bandwidth_gbps = 50.0;
    p.caches             = {{.size_bytes = 64UL * 1024, .bandwidth_gbps = 100.0, .latency_ns = 1.0},
                            {.size_bytes = 1024UL * 1024, .bandwidth_gbps = 80.0, .latency_ns = 4.0},
                            {.size_bytes = 8UL * 1024 * 1024, .bandwidth_gbps = 60.0, .latency_ns = 20.0}};
    return p;
}

constexpr size_t kL1Bytes        = 1024;               // inside L1
constexpr size_t kStreamingBytes = 64UL * 1024 * 1024; // past the LLC

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Width rungs
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Thread axis - width rungs are powers of two plus P", "[ComputeGraph][CostModel]") {
    CHECK(cg::width_rungs_for(8) == std::vector<unsigned>{1, 2, 4, 8});
    CHECK(cg::width_rungs_for(10) == std::vector<unsigned>{1, 2, 4, 8, 10});
    CHECK(cg::width_rungs_for(1) == std::vector<unsigned>{1});
    // A machine that reports nothing still yields a usable single rung rather
    // than an empty list somebody indexes into.
    CHECK(cg::width_rungs_for(0) == std::vector<unsigned>{1});
}

// ═══════════════════════════════════════════════════════════════════════════════
// Default (uncalibrated) curve
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Thread axis - default curve is monotone", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();
    REQUIRE(p.thread_efficiency.empty());

    for (auto const family :
         {cg::KernelFamily::GemmLarge, cg::KernelFamily::BatchedGemm, cg::KernelFamily::Permute, cg::KernelFamily::Elementwise}) {
        double last_speedup    = 0.0;
        double last_efficiency = 2.0;
        for (unsigned const w : {1u, 2u, 4u, 8u, 16u}) {
            double const s = p.parallel_speedup(family, kStreamingBytes, w);
            double const e = p.parallel_efficiency(family, kStreamingBytes, w);
            CHECK(s >= last_speedup);    // speedup never falls with width
            CHECK(e <= last_efficiency); // efficiency never rises with width
            CHECK(s > 0.0);
            CHECK(s <= static_cast<double>(w) + 1e-9); // never superlinear
            last_speedup    = s;
            last_efficiency = e;
        }
        CHECK(p.parallel_speedup(family, kStreamingBytes, 1) == Catch::Approx(1.0));
        CHECK(p.parallel_efficiency(family, kStreamingBytes, 1) == Catch::Approx(1.0));
    }
}

TEST_CASE("Thread axis - the batched-GEMM ceiling reproduces its measured sweep", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();
    REQUIRE(p.thread_efficiency.empty());

    // The sweep the family's parallel fraction was fitted to: the DLPNO-(T)
    // residual's grouped batched GEMMs over 326 triplets on a 10-core machine.
    // The fit is a fit, not an interpolation, so the tolerance is what a fit
    // earns - one point of efficiency at every width it was taken at.
    struct Point {
        unsigned width;
        double   efficiency;
    };
    for (auto const [width, efficiency] : {Point{1, 1.00}, Point{2, 0.88}, Point{4, 0.70}, Point{8, 0.52}, Point{10, 0.44}}) {
        double const modeled = p.parallel_efficiency(cg::KernelFamily::BatchedGemm, kStreamingBytes, width);
        INFO("width " << width);
        CHECK(std::abs(modeled - efficiency) < 0.015);
    }

    // Still ordered against its neighbours: a batch of GEMMs scales worse than
    // one large GEMM and better than a permute.
    CHECK(p.parallel_speedup(cg::KernelFamily::BatchedGemm, kStreamingBytes, 8) <
          p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 8));
    CHECK(p.parallel_speedup(cg::KernelFamily::BatchedGemm, kStreamingBytes, 8) >
          p.parallel_speedup(cg::KernelFamily::Permute, kStreamingBytes, 8));
}

TEST_CASE("Thread axis - default curve is worse for smaller working sets", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();

    // Same family, four size classes: bigger is never worse, and the
    // sub-threshold class buys nothing at all.
    double const l1  = p.parallel_speedup(cg::KernelFamily::Permute, kL1Bytes, 8);
    double const l2  = p.parallel_speedup(cg::KernelFamily::Permute, 512UL * 1024, 8);
    double const llc = p.parallel_speedup(cg::KernelFamily::Permute, 4UL * 1024 * 1024, 8);
    double const str = p.parallel_speedup(cg::KernelFamily::Permute, kStreamingBytes, 8);

    CHECK(l1 < l2);
    CHECK(l2 < llc);
    CHECK(llc < str);
}

TEST_CASE("Thread axis - below threshold, efficiency collapses to 1/w", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();

    // An L1-resident working set gains nothing from width, whatever the family,
    // so a planner that maximizes efficiency naturally leaves it narrow.
    for (unsigned const w : {1u, 2u, 4u, 8u}) {
        CHECK(p.parallel_speedup(cg::KernelFamily::Elementwise, kL1Bytes, w) == Catch::Approx(1.0));
        CHECK(p.parallel_efficiency(cg::KernelFamily::Elementwise, kL1Bytes, w) == Catch::Approx(1.0 / w));
    }

    // The small-GEMM family never threads, at any size.
    for (unsigned const w : {1u, 2u, 4u, 8u}) {
        CHECK(p.parallel_speedup(cg::KernelFamily::GemmSmall, kStreamingBytes, w) == Catch::Approx(1.0));
    }
}

TEST_CASE("Thread axis - width 0 and width past P are clamped", "[ComputeGraph][CostModel]") {
    auto p        = pinned_profile();
    p.max_threads = 8;

    // Zero threads is not a thing; it reads as one and never divides by zero.
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 0) == Catch::Approx(1.0));
    CHECK(p.parallel_efficiency(cg::KernelFamily::GemmLarge, kStreamingBytes, 0) == Catch::Approx(1.0));

    // Past the calibrated P, the answer stops moving.
    double const at_p = p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 8);
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 16) == Catch::Approx(at_p));
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 1000) == Catch::Approx(at_p));

    // Efficiency past P is computed at the clamped width, so it does not keep
    // shrinking as 1/w for widths the machine cannot supply.
    CHECK(p.parallel_efficiency(cg::KernelFamily::GemmLarge, kStreamingBytes, 16) == Catch::Approx(at_p / 8.0));

    // Without a recorded P, nothing is clamped from above.
    auto unclamped = pinned_profile();
    CHECK(unclamped.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 16) >
          unclamped.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 8));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calibrated curves
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Thread axis - calibrated curve interpolates in log2(width)", "[ComputeGraph][CostModel]") {
    cg::EfficiencyCurve curve;
    curve.family     = cg::KernelFamily::GemmLarge;
    curve.size_class = cg::SizeClass::Streaming;
    curve.widths     = {1, 2, 4, 8};
    curve.speedup    = {1.0, 2.0, 4.0, 6.0};

    CHECK(curve.speedup_at(1) == Catch::Approx(1.0));
    CHECK(curve.speedup_at(4) == Catch::Approx(4.0));

    // Between rungs the curve is linear in log2(width), so a width closer to
    // the upper rung in log space lands closer to its speedup.
    CHECK(curve.speedup_at(6) == Catch::Approx(4.0 + (std::log2(6.0) - 2.0) * 2.0).epsilon(1e-9));
    CHECK(curve.speedup_at(5) < curve.speedup_at(6));
    CHECK(curve.speedup_at(6) < curve.speedup_at(7));
    CHECK(curve.speedup_at(7) < curve.speedup_at(8));

    // Outside the measured range, snap to the end rungs. No extrapolation.
    CHECK(curve.speedup_at(0) == Catch::Approx(1.0));
    CHECK(curve.speedup_at(64) == Catch::Approx(6.0));
}

TEST_CASE("Thread axis - a calibrated cell beats the default", "[ComputeGraph][CostModel]") {
    auto p        = pinned_profile();
    p.max_threads = 8;
    p.thread_efficiency.push_back({.family     = cg::KernelFamily::Permute,
                                   .size_class = cg::SizeClass::Streaming,
                                   .widths     = {1, 2, 4, 8},
                                   .speedup    = {1.0, 1.1, 1.2, 1.25}});

    // The measured curve is used where it exists...
    CHECK(p.parallel_speedup(cg::KernelFamily::Permute, kStreamingBytes, 8) == Catch::Approx(1.25));

    // ...and the nearest measured size class stands in for a cell the sweep
    // skipped, rather than falling straight back to the default.
    CHECK(p.parallel_speedup(cg::KernelFamily::Permute, kL1Bytes, 8) == Catch::Approx(1.25));

    // A family with no curve at all still answers from the default model.
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 8) > 1.25);
}

TEST_CASE("Thread axis - malformed curves never divide by zero", "[ComputeGraph][CostModel]") {
    cg::EfficiencyCurve const empty;
    CHECK_FALSE(empty.valid());
    CHECK(empty.speedup_at(4) == Catch::Approx(1.0));

    // Mismatched vectors are not a curve.
    cg::EfficiencyCurve ragged;
    ragged.widths  = {1, 2};
    ragged.speedup = {1.0};
    CHECK_FALSE(ragged.valid());
    CHECK(ragged.speedup_at(2) == Catch::Approx(1.0));

    // A file claiming a zero or negative speedup must not produce a zero
    // divisor for t(w) = t(1)/s(w).
    cg::EfficiencyCurve poisoned;
    poisoned.widths  = {1, 2, 4};
    poisoned.speedup = {1.0, 0.0, -3.0};
    CHECK(poisoned.speedup_at(2) > 0.0);
    CHECK(poisoned.speedup_at(4) > 0.0);

    // A profile carrying it still prices a GEMM finitely.
    auto p = pinned_profile();
    p.thread_efficiency.push_back(poisoned);
    p.thread_efficiency.back().family     = cg::KernelFamily::GemmLarge;
    p.thread_efficiency.back().size_class = cg::SizeClass::Streaming;
    double const t                        = p.estimate_gemm_time_us(512, 512, 512, 4);
    CHECK(std::isfinite(t));
    CHECK(t > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Thread-aware estimators
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Thread axis - estimators default to width 1", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();

    CHECK(p.estimate_gemm_time_us(512, 512, 512) == Catch::Approx(p.estimate_gemm_time_us(512, 512, 512, 1)));
    CHECK(p.estimate_permute_time_us(1UL << 20, 2) == Catch::Approx(p.estimate_permute_time_us(1UL << 20, 2, 1)));
    CHECK(p.estimate_memory_time_us(1UL << 20) == Catch::Approx(p.estimate_memory_time_us(1UL << 20, 1)));
}

TEST_CASE("Thread axis - width makes big work cheaper and small work not", "[ComputeGraph][CostModel]") {
    auto const p = pinned_profile();

    // A GEMM past the small-kernel flop threshold, with a streaming working
    // set, gets most of the machine.
    double const wide   = p.estimate_gemm_time_us(1024, 1024, 1024, 8);
    double const narrow = p.estimate_gemm_time_us(1024, 1024, 1024, 1);
    CHECK(wide < narrow);
    CHECK(wide > narrow / 8.0); // never better than perfect scaling

    // A GEMM below the threshold is the small-kernel family: width buys
    // nothing, so its estimate is unchanged.
    CHECK(cg::DeviceProfile::gemm_family(16, 16, 16) == cg::KernelFamily::GemmSmall);
    CHECK(p.estimate_gemm_time_us(16, 16, 16, 8) == Catch::Approx(p.estimate_gemm_time_us(16, 16, 16, 1)));

    // Permutes scale too, and the launch overhead is not divided away.
    CHECK(p.estimate_permute_time_us(kStreamingBytes, 2, 8) < p.estimate_permute_time_us(kStreamingBytes, 2, 1));
    CHECK(p.estimate_permute_time_us(kStreamingBytes, 2, 8) > p.kernel_launch_overhead_us);
}

TEST_CASE("Thread axis - CostModel dispatches the thread axis by target", "[ComputeGraph][CostModel]") {
    cg::CostModel model;
    model.cpu = pinned_profile();

    CHECK(model.estimate_gemm_time_us(1024, 1024, 1024, cg::Target::CPU, 8) <
          model.estimate_gemm_time_us(1024, 1024, 1024, cg::Target::CPU, 1));
    CHECK(model.estimate_gemm_time_us(1024, 1024, 1024, cg::Target::CPU) ==
          Catch::Approx(model.estimate_gemm_time_us(1024, 1024, 1024, cg::Target::CPU, 1)));
    CHECK(model.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 4, cg::Target::CPU) > 1.0);
    CHECK(model.parallel_efficiency(cg::KernelFamily::GemmLarge, kStreamingBytes, 4, cg::Target::CPU) <= 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON round trip
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Thread axis - efficiency curves survive the JSON round trip", "[ComputeGraph][CostModel]") {
    // The cache array was once written and never parsed back (ef7875aa2); this
    // is the same test for the same trap, one field family later.
    cg::CostModel original;
    original.source                = "test";
    original.cpu.name              = "Curve CPU";
    original.cpu.max_threads       = 10;
    original.cpu.caches            = {{.size_bytes = 65536, .bandwidth_gbps = 118.0, .latency_ns = 0.95}};
    original.cpu.thread_efficiency = {
        {.family     = cg::KernelFamily::GemmLarge,
         .size_class = cg::SizeClass::Streaming,
         .widths     = {1, 2, 4, 8, 10},
         .speedup    = {1.0, 1.95, 3.7, 6.4, 7.1}},
        {.family = cg::KernelFamily::Permute, .size_class = cg::SizeClass::L2Resident, .widths = {1, 2}, .speedup = {1.0, 1.4}},
        {.family = cg::KernelFamily::Elementwise, .size_class = cg::SizeClass::L1Resident, .widths = {1, 2, 4}, .speedup = {1.0, 0.9, 0.8}},
    };
    original.gpu.name = "Curve GPU";

    std::string const path = "test_thread_curves.json";
    REQUIRE(original.save_json(path).has_value());

    auto loaded_result = cg::CostModel::load_json(path);
    REQUIRE(loaded_result.has_value());
    auto &loaded = loaded_result.value();

    CHECK(loaded.cpu.max_threads == 10);
    REQUIRE(loaded.cpu.thread_efficiency.size() == 3);

    auto const *gemm = loaded.cpu.find_curve(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming);
    REQUIRE(gemm != nullptr);
    CHECK(gemm->size_class == cg::SizeClass::Streaming);
    REQUIRE(gemm->widths == std::vector<std::uint32_t>{1, 2, 4, 8, 10});
    REQUIRE(gemm->speedup.size() == 5);
    CHECK(gemm->speedup[1] == Catch::Approx(1.95));
    CHECK(gemm->speedup[4] == Catch::Approx(7.1));

    auto const *permute = loaded.cpu.find_curve(cg::KernelFamily::Permute, cg::SizeClass::L2Resident);
    REQUIRE(permute != nullptr);
    CHECK(permute->widths == std::vector<std::uint32_t>{1, 2});
    CHECK(permute->speedup[1] == Catch::Approx(1.4));

    // A curve where width hurt must come back as measured, not clamped to 1.
    auto const *elem = loaded.cpu.find_curve(cg::KernelFamily::Elementwise, cg::SizeClass::L1Resident);
    REQUIRE(elem != nullptr);
    CHECK(elem->speedup[2] == Catch::Approx(0.8));

    // And the reloaded profile answers queries the same way the original did.
    CHECK(loaded.cpu.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 8) ==
          Catch::Approx(original.cpu.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 8)));

    std::remove(path.c_str());
}

TEST_CASE("Thread axis - a curve's route survives the JSON round trip", "[ComputeGraph][CostModel]") {
    cg::CostModel original;
    original.cpu.max_threads       = 4;
    original.cpu.thread_efficiency = {
        {.family     = cg::KernelFamily::GemmLarge,
         .size_class = cg::SizeClass::Streaming,
         .route      = cg::CurveRoute::Einsums,
         .widths     = {1, 2, 4},
         .speedup    = {1.0, 1.9, 3.5}},
        {.family     = cg::KernelFamily::GemmLarge,
         .size_class = cg::SizeClass::Streaming,
         .route      = cg::CurveRoute::Vendor,
         .widths     = {1, 2, 4},
         .speedup    = {1.0, 1.0, 1.0}},
    };

    std::string const path = "test_thread_curves_routes.json";
    REQUIRE(original.save_json(path).has_value());
    auto loaded_result = cg::CostModel::load_json(path);
    std::remove(path.c_str());
    REQUIRE(loaded_result.has_value());
    auto const &loaded = loaded_result.value().cpu;

    // Two curves for one (family, size class), told apart by their route, and
    // not merged into one curve of six rows.
    REQUIRE(loaded.thread_efficiency.size() == 2);
    auto const *packed = loaded.find_curve(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, cg::CurveRoute::Einsums);
    auto const *vendor = loaded.find_curve(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, cg::CurveRoute::Vendor);
    REQUIRE(packed != nullptr);
    REQUIRE(vendor != nullptr);
    CHECK(packed->widths.size() == 3);
    CHECK(packed->speedup[2] == Catch::Approx(3.5));
    CHECK(vendor->speedup[2] == Catch::Approx(1.0));
}

// A lookup on one route never answers from the other route's measurement. The vendor curve here is
// what Accelerate produces, flat because it ignores the thread count it is given; reading it for a
// contraction the planner widens on the packed loops would price that contraction as unable to
// use the machine.
TEST_CASE("Thread axis - a lookup never answers from the other route", "[ComputeGraph][CostModel]") {
    auto p                = pinned_profile();
    p.max_threads         = 4;
    p.thread_efficiency   = {{.family     = cg::KernelFamily::GemmLarge,
                              .size_class = cg::SizeClass::Streaming,
                              .route      = cg::CurveRoute::Vendor,
                              .widths     = {1, 2, 4},
                              .speedup    = {1.0, 1.0, 1.0}}};
    double const fallback = cg::DeviceProfile::default_parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 4);

    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 4, cg::CurveRoute::Vendor) == Catch::Approx(1.0));
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 4, cg::CurveRoute::Einsums) == Catch::Approx(fallback));
    // The route a caller does not name is Einsums', which is the kernel every family but the vendor
    // GEMM calls runs once widened.
    CHECK(p.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 4) == Catch::Approx(fallback));
    CHECK(p.find_curve(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming) == nullptr);
}

// A profile written before curves carried a route timed the vendor's GEMM and axpy for the GEMM and
// element-wise families, which is not what a widened node of either runs; on Accelerate those
// curves came out flat at every width. Loading such a profile must drop exactly those curves and
// keep the batched-GEMM and permute curves, which always timed Einsums' own loops.
TEST_CASE("Thread axis - a pre-route profile keeps only curves that timed the right kernel", "[ComputeGraph][CostModel]") {
    cg::CostModel original;
    original.cpu.name              = "Pre-route CPU";
    original.cpu.max_threads       = 4;
    original.cpu.thread_efficiency = {
        {.family = cg::KernelFamily::GemmSmall, .size_class = cg::SizeClass::L1Resident, .widths = {1, 2}, .speedup = {1.0, 1.0}},
        {.family = cg::KernelFamily::GemmLarge, .size_class = cg::SizeClass::Streaming, .widths = {1, 2}, .speedup = {1.0, 1.0}},
        {.family = cg::KernelFamily::Elementwise, .size_class = cg::SizeClass::Streaming, .widths = {1, 2}, .speedup = {1.0, 1.0}},
        {.family = cg::KernelFamily::BatchedGemm, .size_class = cg::SizeClass::L2Resident, .widths = {1, 2}, .speedup = {1.0, 1.5}},
        {.family = cg::KernelFamily::Permute, .size_class = cg::SizeClass::Streaming, .widths = {1, 2}, .speedup = {1.0, 1.7}},
    };

    // Written as this build writes it, then with every route removed: a file from before routes.
    std::string const path = "test_thread_curves_pre_route.json";
    REQUIRE(original.save_json(path).has_value());
    std::string text;
    {
        std::ifstream      in(path);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        text = buffer.str();
    }
    std::string const stripped = std::regex_replace(text, std::regex(R"re("route"\s*:\s*"[a-z]+"\s*,?)re"), "");
    REQUIRE(stripped != text);
    {
        std::ofstream out(path, std::ios::trunc);
        out << stripped;
    }

    auto loaded_result = cg::CostModel::load_json(path);
    std::remove(path.c_str());
    REQUIRE(loaded_result.has_value());
    auto const &loaded = loaded_result.value().cpu;

    REQUIRE(loaded.thread_efficiency.size() == 2);
    for (auto const &curve : loaded.thread_efficiency) {
        CHECK(curve.route == cg::CurveRoute::Einsums);
    }
    auto const *batched = loaded.find_curve(cg::KernelFamily::BatchedGemm, cg::SizeClass::L2Resident);
    auto const *permute = loaded.find_curve(cg::KernelFamily::Permute, cg::SizeClass::Streaming);
    REQUIRE(batched != nullptr);
    REQUIRE(permute != nullptr);
    CHECK(batched->speedup[1] == Catch::Approx(1.5));
    CHECK(permute->speedup[1] == Catch::Approx(1.7));

    for (auto const family : {cg::KernelFamily::GemmSmall, cg::KernelFamily::GemmLarge, cg::KernelFamily::Elementwise}) {
        INFO(cg::to_string(family));
        CHECK(loaded.find_curve(family, cg::SizeClass::Streaming, cg::CurveRoute::Einsums) == nullptr);
        CHECK(loaded.find_curve(family, cg::SizeClass::Streaming, cg::CurveRoute::Vendor) == nullptr);
    }
    // The dropped families answer from the default model, not from the flat curves.
    CHECK(loaded.parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 2) ==
          Catch::Approx(cg::DeviceProfile::default_parallel_speedup(cg::KernelFamily::GemmLarge, cg::SizeClass::Streaming, 2)));
    // Everything else in the profile is kept.
    CHECK(loaded.max_threads == 4);
}

TEST_CASE("Thread axis - a profile with no curves round-trips cleanly", "[ComputeGraph][CostModel]") {
    cg::CostModel original;
    original.cpu.name = "Bare CPU";

    std::string const path = "test_thread_curves_empty.json";
    REQUIRE(original.save_json(path).has_value());
    auto loaded_result = cg::CostModel::load_json(path);
    REQUIRE(loaded_result.has_value());

    CHECK(loaded_result.value().cpu.thread_efficiency.empty());
    CHECK(loaded_result.value().cpu.max_threads == 0);
    // And still answers, from the default model.
    CHECK(loaded_result.value().cpu.parallel_speedup(cg::KernelFamily::GemmLarge, kStreamingBytes, 4) > 1.0);

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calibration sweep mechanics
// ═══════════════════════════════════════════════════════════════════════════════

// A profile swept before revision 2 timed the batched-GEMM cell as a static loop over GEMM calls,
// not the dynamic loop a grouped batched GEMM node runs. Loading such a profile drops exactly the
// batched-GEMM curves and keeps every other one, and a profile written now keeps them all.
TEST_CASE("Thread axis - a profile swept before the batched cell changed drops its batched curves", "[ComputeGraph][CostModel]") {
    cg::CostModel original;
    original.cpu.name              = "Revision 1 CPU";
    original.cpu.max_threads       = 4;
    original.cpu.thread_efficiency = {
        {.family = cg::KernelFamily::BatchedGemm, .size_class = cg::SizeClass::L2Resident, .widths = {1, 2}, .speedup = {1.0, 0.86}},
        {.family = cg::KernelFamily::Permute, .size_class = cg::SizeClass::Streaming, .widths = {1, 2}, .speedup = {1.0, 1.7}},
        {.family = cg::KernelFamily::GroupedPermute, .size_class = cg::SizeClass::L2Resident, .widths = {1, 2}, .speedup = {1.0, 1.9}},
    };

    std::string const path = "test_thread_curves_revision.json";
    REQUIRE(original.save_json(path).has_value());

    SECTION("written now, every curve loads") {
        auto loaded = cg::CostModel::load_json(path);
        REQUIRE(loaded.has_value());
        auto const &cpu = loaded.value().cpu;
        CHECK(cpu.thread_efficiency.size() == 3);
        auto const *grouped = cpu.find_curve(cg::KernelFamily::GroupedPermute, cg::SizeClass::L2Resident);
        REQUIRE(grouped != nullptr); // and the new family's name survives the round trip
        CHECK(grouped->speedup[1] == Catch::Approx(1.9));
    }

    SECTION("with no revision recorded, the batched-GEMM curve is dropped") {
        std::string text;
        {
            std::ifstream      in(path);
            std::ostringstream buffer;
            buffer << in.rdbuf();
            text = buffer.str();
        }
        std::string const stripped = std::regex_replace(text, std::regex(R"re("thread_sweep_revision"\s*:\s*[0-9]+\s*,?)re"), "");
        REQUIRE(stripped != text);
        {
            std::ofstream out(path, std::ios::trunc);
            out << stripped;
        }
        auto loaded = cg::CostModel::load_json(path);
        REQUIRE(loaded.has_value());
        auto const &cpu = loaded.value().cpu;
        CHECK(cpu.find_curve(cg::KernelFamily::BatchedGemm, cg::SizeClass::L2Resident) == nullptr);
        CHECK(cpu.find_curve(cg::KernelFamily::Permute, cg::SizeClass::Streaming) != nullptr);
        CHECK(cpu.find_curve(cg::KernelFamily::GroupedPermute, cg::SizeClass::L2Resident) != nullptr);
    }
    std::remove(path.c_str());
}

// A grouped permute was priced from the single-permute curve, HPTT dividing one tensor, which dips
// from two threads to four; the DLPNO (T) grouped permute measured 2.8x at eight. It has a curve of
// its own now, and a profile swept before it existed prices it from the element-wise curve: the same
// loop over members, each memory-bound. With neither, the model answers.
TEST_CASE("Thread axis - a grouped permute reads its own curve, else the element-wise one", "[ComputeGraph][CostModel]") {
    auto p           = pinned_profile();
    p.max_threads    = 8;
    auto const curve = [](cg::KernelFamily family, std::vector<double> speedup) {
        return cg::EfficiencyCurve{
            .family = family, .size_class = cg::SizeClass::Streaming, .widths = {1, 2, 4, 8}, .speedup = std::move(speedup)};
    };

    SECTION("its own curve") {
        p.thread_efficiency = {curve(cg::KernelFamily::GroupedPermute, {1.0, 1.9, 2.4, 2.8}),
                               curve(cg::KernelFamily::Elementwise, {1.0, 1.5, 1.6, 1.7}),
                               curve(cg::KernelFamily::Permute, {1.0, 1.86, 1.74, 2.05})};
        CHECK(p.parallel_speedup(cg::KernelFamily::GroupedPermute, kStreamingBytes, 8) == Catch::Approx(2.8));
    }
    SECTION("the element-wise curve when it has none") {
        p.thread_efficiency = {curve(cg::KernelFamily::Elementwise, {1.0, 1.5, 1.6, 1.7}),
                               curve(cg::KernelFamily::Permute, {1.0, 1.86, 1.74, 2.05})};
        CHECK(p.parallel_speedup(cg::KernelFamily::GroupedPermute, kStreamingBytes, 8) == Catch::Approx(1.7));
    }
    SECTION("the model when there is neither") {
        p.thread_efficiency = {curve(cg::KernelFamily::Permute, {1.0, 1.86, 1.74, 2.05})};
        CHECK(p.parallel_speedup(cg::KernelFamily::GroupedPermute, kStreamingBytes, 8) ==
              Catch::Approx(cg::DeviceProfile::default_parallel_speedup(cg::KernelFamily::GroupedPermute, cg::SizeClass::Streaming, 8)));
    }
}

TEST_CASE("Thread axis - sweep populates curves", "[ComputeGraph][CostModel]") {
    // A drastically shortened sweep: two rungs, tiny working sets, one timed
    // run. This asserts the mechanics, never the numbers, which belong to
    // whatever else the machine is doing.
    cg::ThreadSweepOptions options;
    options.warmup          = 0;
    options.repeats         = 1;
    options.max_threads     = 2;
    options.max_bytes       = 64UL * 1024;
    options.max_gemm_extent = 96;
    options.batch_entries   = 4;

    auto const curves = cg::measure_thread_efficiency(pinned_profile(), options);
    REQUIRE_FALSE(curves.empty());

    for (auto const &curve : curves) {
        CHECK(curve.valid());
        REQUIRE(curve.widths.size() == 2);
        CHECK(curve.widths[0] == 1);
        CHECK(curve.widths[1] == 2);
        CHECK(curve.speedup[0] == Catch::Approx(1.0));
        CHECK(curve.speedup[1] > 0.0);
        CHECK(std::isfinite(curve.speedup[1]));
    }

    // The GEMM families are measured on both routes, and the packed side's
    // presence says its pin held: a packed cell that ran anything but the
    // packed loops is dropped. Every other family times Einsums' own loop.
    auto const has = [&](cg::KernelFamily family, cg::CurveRoute route) {
        return std::ranges::any_of(curves, [&](cg::EfficiencyCurve const &c) { return c.family == family && c.route == route; });
    };
    for (auto const family : {cg::KernelFamily::GemmSmall, cg::KernelFamily::GemmLarge}) {
        INFO(cg::to_string(family));
        CHECK(has(family, cg::CurveRoute::Einsums));
        CHECK(has(family, cg::CurveRoute::Vendor));
    }
    for (auto const family :
         {cg::KernelFamily::BatchedGemm, cg::KernelFamily::Permute, cg::KernelFamily::Elementwise, cg::KernelFamily::GroupedPermute}) {
        INFO(cg::to_string(family));
        CHECK(has(family, cg::CurveRoute::Einsums));
        CHECK_FALSE(has(family, cg::CurveRoute::Vendor));
    }

    // Dropping the curves into a profile makes it answer from measurement.
    auto p              = pinned_profile();
    p.max_threads       = 2;
    p.thread_efficiency = curves;
    CHECK(p.parallel_speedup(cg::KernelFamily::Permute, kStreamingBytes, 2) > 0.0);
}

// A sample is as many back-to-back runs as fill the minimum sample time. Timed one run at a time,
// the microsecond cells were the clock's jitter: a three-sample median of a small vendor GEMM
// read 1.94x at width 2 on one sweep and 1.00x on the next. The bound below is a floor on wall
// time, which a slow machine can only exceed, so it fails only when samples are not filled.
TEST_CASE("Thread axis - each sweep sample lasts at least the minimum sample time", "[ComputeGraph][CostModel]") {
    cg::ThreadSweepOptions options;
    options.warmup             = 0;
    options.repeats            = 1;
    options.max_threads        = 2;
    options.max_bytes          = 64UL * 1024;
    options.max_gemm_extent    = 96;
    options.batch_entries      = 4;
    options.min_sample_seconds = 0.01;

    auto const t0     = std::chrono::steady_clock::now();
    auto const curves = cg::measure_thread_efficiency(pinned_profile(), options);
    auto const t1     = std::chrono::steady_clock::now();
    REQUIRE_FALSE(curves.empty());

    // Every recorded point spent at least one full batch finding its run count and one timing it.
    size_t points = 0;
    for (auto const &curve : curves) {
        points += curve.widths.size();
    }
    double const elapsed = std::chrono::duration<double>(t1 - t0).count();
    CHECK(elapsed >= 2.0 * static_cast<double>(points) * options.min_sample_seconds);
}

TEST_CASE("Thread axis - sweep on a single-thread machine yields nothing", "[ComputeGraph][CostModel]") {
    cg::ThreadSweepOptions options;
    options.max_threads = 1;
    CHECK(cg::measure_thread_efficiency(pinned_profile(), options).empty());
}
