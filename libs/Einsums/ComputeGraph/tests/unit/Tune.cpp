//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The tune's crossover probes: the rule that reads a threshold off a sweep, and the thresholds
// themselves, each of which must decide its own route and nothing else, or a probe forcing it to
// either extreme would time one route twice. Timing is not tested; the tune is run end to end by
// the Python tests of the command.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Tune.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Options/Parse.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

using cg::detail::CrossoverBound;
using cg::detail::CrossoverPoint;
using cg::detail::find_crossover;

TEST_CASE("Tune - a crossover between two swept sizes is their geometric mean", "[ComputeGraph][Tune]") {
    std::vector<CrossoverPoint> const points{{.size = 100, .small_side = 1.0, .large_side = 2.0},
                                             {.size = 400, .small_side = 1.0, .large_side = 1.5},
                                             {.size = 1600, .small_side = 2.0, .large_side = 1.0},
                                             {.size = 6400, .small_side = 4.0, .large_side = 1.0}};
    auto const                        crossover = find_crossover(points);
    CHECK(crossover.bound == CrossoverBound::Exact);
    CHECK_THAT(crossover.value, Catch::Matchers::WithinRel(800.0, 1e-12));
}

TEST_CASE("Tune - a crossover outside the sweep is a bound at the end it ran into", "[ComputeGraph][Tune]") {
    std::vector<CrossoverPoint> const always{{.size = 10, .small_side = 2.0, .large_side = 1.0},
                                             {.size = 100, .small_side = 3.0, .large_side = 1.0}};
    auto const                        at_most = find_crossover(always);
    CHECK(at_most.bound == CrossoverBound::AtMost);
    CHECK(at_most.value == 10.0);

    std::vector<CrossoverPoint> const never{{.size = 10, .small_side = 1.0, .large_side = 2.0},
                                            {.size = 100, .small_side = 1.0, .large_side = 3.0}};
    auto const                        at_least = find_crossover(never);
    CHECK(at_least.bound == CrossoverBound::AtLeast);
    CHECK(at_least.value == 100.0);
}

TEST_CASE("Tune - a point within the tie band does not move the crossover", "[ComputeGraph][Tune]") {
    // The large side wins from 400 on except for a point at 1600 that is even within 3%.
    std::vector<CrossoverPoint> const points{{.size = 100, .small_side = 1.0, .large_side = 2.0},
                                             {.size = 400, .small_side = 2.0, .large_side = 1.0},
                                             {.size = 1600, .small_side = 1.0, .large_side = 1.02},
                                             {.size = 6400, .small_side = 4.0, .large_side = 1.0}};
    auto const                        crossover = find_crossover(points);
    CHECK(crossover.bound == CrossoverBound::Exact);
    CHECK_THAT(crossover.value, Catch::Matchers::WithinRel(200.0, 1e-12));

    // A clear loss late in the sweep does move it, past the loss.
    std::vector<CrossoverPoint> late = points;
    late[2].large_side               = 1.5;
    CHECK_THAT(find_crossover(late).value, Catch::Matchers::WithinRel(std::sqrt(1600.0 * 6400.0), 1e-12));
}

namespace {

/// The declared default of a registered option, as the registry reports it: what the library runs
/// with whatever the environment says.
cl::RegisteredOption registered(std::string_view name) {
    for (auto const &option : cl::registered_options()) {
        if (option.name == name) {
            return option;
        }
    }
    FAIL("no option " << name);
    return {};
}

/// An option held at a value for one case.
template <typename T>
struct Pin {
    cl::ConfigOption<T> const &option;
    T                          previous;
    Pin(cl::ConfigOption<T> const &o, T value) : option(o), previous(config::get(o)) { config::set(o, value); }
    Pin(Pin const &)            = delete;
    Pin &operator=(Pin const &) = delete;
    ~Pin() { config::set(option, previous); }
};

/// What the string engine and the packed engine said about the last call.
std::string decided() {
    return fmt::format("{}/{}", cg::dispatch::last_dispatch_route(), packed_gemm::last_packed_decision());
}

/// A threshold no size reaches: 2^40, the options' upper bound.
constexpr std::int64_t never = 1'099'511'627'776;

} // namespace

TEST_CASE("Tune - each tuned option defaults to the constant it replaced", "[ComputeGraph][Tune]") {
    // Measured on one machine each; the options exist so a tune can replace them, and an untuned
    // machine must behave exactly as before they did.
    CHECK(registered("einsums:packed-gemm:outer-product-floor").default_int == 768);
    CHECK(registered("einsums:packed-gemm:outer-gemm-max").default_int == 131072);
    CHECK(registered("einsums:packed-gemm:stream-min-elems").default_int == 4096);
    CHECK(registered("einsums:packed-gemm:stream-elems-per-thread").default_int == 65536);
    CHECK(registered("einsums:hardware:omp-break-even-flops-per-ns").default_double == 12.0);
}

TEST_CASE("Tune - the outer-product floor decides between the generic loop and the packed passes", "[ComputeGraph][Tune]") {
    Tensor<double, 2> A("A", 8, 8);
    Tensor<double, 1> B("B", 8);
    Tensor<double, 3> C("C", 8, 8, 8);
    A.zero();
    B.zero();
    std::string below, above;
    {
        Pin<std::int64_t> const pin(option::PackedGemmOuterProductFloor, never);
        cg::einsum("ijk <- ij ; k", 0.0, &C, 1.0, A, B);
        below = decided();
    }
    {
        Pin<std::int64_t> const pin(option::PackedGemmOuterProductFloor, std::int64_t{0});
        cg::einsum("ijk <- ij ; k", 0.0, &C, 1.0, A, B);
        above = decided();
    }
    INFO(below << " against " << above);
    CHECK(below.find("defer_small_outer_to_generic") != std::string::npos);
    CHECK(above.find("defer_") == std::string::npos);
}

TEST_CASE("Tune - the outer GEMM limit decides between a k=1 GEMM and ger", "[ComputeGraph][Tune]") {
    // A rank-2 outer product takes the string engine's own GER route, so the decision is the
    // packed engine's, for a higher rank past the size where its direct paths start.
    Tensor<double, 2> A("A", 16, 16);
    Tensor<double, 1> B("B", 16);
    Tensor<double, 3> C("C", 16, 16, 16);
    A.zero();
    B.zero();
    C.zero();
    std::string below, above;
    {
        Pin<std::int64_t> const pin(option::PackedGemmOuterGemmMax, never);
        cg::einsum("ijk <- ij ; k", 0.5, &C, 1.0, A, B);
        below = decided();
    }
    {
        Pin<std::int64_t> const pin(option::PackedGemmOuterGemmMax, std::int64_t{0});
        cg::einsum("ijk <- ij ; k", 0.5, &C, 1.0, A, B);
        above = decided();
    }
    INFO(below << " against " << above);
    CHECK(below.find("direct_outer_gemm") != std::string::npos);
    CHECK(above.find("direct_ger") != std::string::npos);
}

TEST_CASE("Tune - the stream minimum decides whether an interleaved GEMV is streamed", "[ComputeGraph][Tune]") {
    Tensor<double, 4> A("A", 6, 6, 6, 6);
    Tensor<double, 2> B("B", 6, 6);
    Tensor<double, 2> C("C", 6, 6);
    A.zero();
    B.zero();
    std::string below, above;
    {
        Pin<std::int64_t> const pin(option::PackedGemmStreamMinElems, never);
        cg::einsum("ij <- ikjl ; kl", 0.0, &C, 1.0, A, B);
        below = decided();
    }
    {
        Pin<std::int64_t> const pin(option::PackedGemmStreamMinElems, std::int64_t{0});
        cg::einsum("ij <- ikjl ; kl", 0.0, &C, 1.0, A, B);
        above = decided();
    }
    INFO(below << " against " << above);
    CHECK(below.find("stream") == std::string::npos);
    CHECK(above.find("stream") != std::string::npos);
}

TEST_CASE("Tune - the break-even rate scales the parallel threshold", "[ComputeGraph][Tune]") {
    // The region cost is resolved once per team size, so the threshold is checked against the one
    // in effect.
    double const region = hardware::omp_region_cost_ns();
    for (double const rate : {12.0, 100.0}) {
        Pin<double> const pin(option::HardwareOmpBreakEvenFlopsPerNs, rate);
        CHECK(hardware::omp_min_parallel_flops() == static_cast<std::int64_t>(region * rate));
    }
}
