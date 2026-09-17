//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Where the generic algorithm's parallel region starts paying for itself.
//
// einsums_generic_target_walk opens a parallel region with no trip-count floor,
// unlike the 27 elementwise regions that gate on omp_min_parallel_elements() and
// unlike ComputeGraph's own walk, which gates on a parallel_floor. Forking a team
// costs tens of microseconds when the workers are parked, so below some element
// count the fork dwarfs the contraction.
//
// Each row prints the route the dispatcher actually chose, so a pattern that stops
// reaching GENERIC shows up as a changed label rather than as a silently different
// measurement. Run once with default threads and once with OMP_NUM_THREADS=1; the
// ratio between them is the fork cost, and where it crosses 1.0 is the floor.

#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <omp.h>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

namespace {

char const *route_name(tensor_algebra::detail::AlgorithmChoice c) {
    using namespace tensor_algebra::detail;
    switch (c) {
    case GENERIC:
        return "GENERIC";
    case DOT:
        return "DOT";
    case DIRECT:
        return "DIRECT";
    case GER:
        return "GER";
    case GEMV:
        return "GEMV";
    case GEMM:
        return "GEMM";
    case PACKED_GEMM:
        return "PACKED_GEMM";
    case SORT_GEMM:
        return "SORT_GEMM";
    default:
        return "INDETERMINATE";
    }
}

} // namespace

// A batched contraction: j is free in all three operands, k is contracted.
// C(i,j) = sum_k A(i,j,k) * B(k,j). No single GEMM covers it, so the dispatcher
// falls through to the generic walk.
EINSUMS_TEST_CASE("Bench GenericSmall: batched contraction", "[TensorAlgebra][benchmark]") {
    LabeledSection0();

    fmt::println("[GenericSmall] threads={} region_cost_ns={:.0f} min_parallel_flops={} min_parallel_elements={}", omp_get_max_threads(),
                 hardware::omp_region_cost_ns(), hardware::omp_min_parallel_flops(), hardware::omp_min_parallel_elements());
    fmt::println("[GenericSmall] {:>10} {:>12} {:>12} {:>12} {:>14}", "elements", "route", "flops", "time(us)", "ns/elem");

    for (int n : {2, 4, 6, 8, 12, 16, 24, 32, 48, 64}) {
        Tensor<double, 3> A("A", n, n, n);
        Tensor<double, 2> B("B", n, n);
        Tensor<double, 2> C("C", n, n);
        fill(A);
        fill(B);
        C.zero();

        tensor_algebra::detail::AlgorithmChoice route = tensor_algebra::detail::INDETERMINATE;

        // Warm the route decision and the pages before timing.
        tensor_algebra::einsum<false, false>(0.0, Indices{i, j}, &C, 1.0, Indices{i, j, k}, A, Indices{k, j}, B, &route);

        auto t = time_us(
            "generic",
            [&] {
                tensor_algebra::detail::AlgorithmChoice r = tensor_algebra::detail::INDETERMINATE;
                tensor_algebra::einsum<false, false>(0.0, Indices{i, j}, &C, 1.0, Indices{i, j, k}, A, Indices{k, j}, B, &r);
            },
            200);

        size_t const elems = static_cast<size_t>(n) * n;
        size_t const flops = 2 * elems * static_cast<size_t>(n);
        fmt::println("[GenericSmall] {:>10d} {:>12} {:>12d} {:>12.3f} {:>14.2f}", elems, route_name(route), flops, t.avg,
                     t.avg * 1000.0 / static_cast<double>(elems));
    }
}
