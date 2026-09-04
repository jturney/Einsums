//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkSmallElementwise.cpp
/// @brief Per-call floor of the elementwise kernels at small operand sizes.
///
/// These kernels are bandwidth-bound and their per-element cost is a fraction
/// of a nanosecond, so at small sizes the whole call SHOULD be well under a
/// microsecond. What can wreck that is opening an OpenMP parallel region
/// regardless of operand size: forking a team costs tens of microseconds when
/// the workers are parked, which for a few hundred elements is several orders
/// of magnitude more than the arithmetic.
///
/// That is not hypothetical - ?dirprod did exactly this, at 23-37 us for a
/// 64-element product against 0.10 us serial. The kernels here are the same
/// family, reached by axpy / scale / division / element_transform / permute,
/// and a tiled or per-tile workload calls them thousands of times per
/// iteration at exactly these sizes.
///
/// Reading this: every row should be a fraction of a microsecond. A row in the
/// tens of microseconds that does NOT scale with the element count is a fork
/// cost, not work. Cross-check by re-running with ``OMP_NUM_THREADS=1``: a
/// serial run that is dramatically FASTER means the region needs a size gate.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <complex>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

namespace {

constexpr int kReps = 200;

void row(std::string const &op, size_t elements, TimingStats const &t) {
    fmt::println("[SmallElementwise] {:<22} elements={:8d}  {:9.3f} us  ({:8.2f} ns/element)", op, elements, t.avg,
                 1000.0 * t.avg / static_cast<double>(elements));
    std::string const key = fmt::format("SmallElementwise {}", op);
    publish_benchmark_result(key.c_str(), "t_call", static_cast<int>(elements), t);
}

} // namespace

EINSUMS_TEST_CASE("Bench SmallElementwise: per-call floor", "[LinearAlgebra][SmallElementwise][benchmark]") {
    LabeledSection0();

    for (size_t n : {64UL, 512UL, 4096UL}) {
        auto X = create_random_tensor<double>(std::string("X"), n);
        auto Y = create_random_tensor<double>(std::string("Y"), n);

        row("axpy", n, time_us("axpy", [&]() { linear_algebra::axpy(1.5, X, &Y); }, kReps));
        row("scale", n, time_us("scale", [&]() { linear_algebra::scale(0.99, &Y); }, kReps));
        row("direct_division", n, time_us("div", [&]() { linear_algebra::direct_division(1.0, X, X, 0.0, &Y); }, kReps));
        row("element_transform", n,
            time_us("et", [&]() { tensor_algebra::element_transform(&Y, [](double v) { return v * 0.5; }); }, kReps));
    }

    // Permute is the other elementwise-family entry point, and it is the one a
    // tiled workload leans on hardest.
    for (size_t n : {8UL, 24UL, 64UL}) {
        auto A = create_random_tensor<double>(std::string("A"), n, n);
        auto B = create_zero_tensor<double>(std::string("B"), n, n);
        row("permute ji<-ij", n * n, time_us("permute", [&]() { tensor_algebra::permute(Indices{j, i}, &B, Indices{i, j}, A); }, kReps));
    }

    // Tensor-level elementwise operators, which reach the TensorImpl kernels
    // directly rather than through a BLAS call.
    for (size_t n : {64UL, 4096UL}) {
        auto P = create_random_tensor<double>(std::string("P"), n);
        auto Q = create_random_tensor<double>(std::string("Q"), n);
        row("tensor A *= B", n, time_us("mult", [&]() { Q *= P; }, kReps));
        row("tensor A /= B", n, time_us("div", [&]() { Q /= P; }, kReps));
    }
}
