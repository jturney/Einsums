//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Does the EAGER einsum path take the packed scatter, and should it?
//
// `MicroKernelShape::fast_scatter` is false on x86, so
// `tensor_algebra::einsum` - the compile-time-indices entry point - DECLINES
// every multi-M/N contraction to Sort+GEMM and never reaches the packed loops.
// The string / ComputeGraph path the head-to-head harness drives passes
// allow_scatter = true and is unaffected, which is why the flag has stayed
// hidden.
//
// Neither BenchmarkSortGemm nor BenchmarkMultiMN can settle whether the flag
// should flip: every case in both has C's m indices ADJACENT, so coalesce_plan
// merges them and the contraction is not a scatter at all. The flag only bites
// when C's two index groups INTERLEAVE, which is the rank-6 ccsd_t shape. That
// is what this builds.
//
//   C[a,b,c,d,e,f] += A[g,d,b,c] * B[e,f,g,a]      M = (d,b,c)  N = (e,f,a)
//
// Times three ways and reports the algorithm einsum actually chose:
//   packed    - try_packed_gemm called directly with allow_scatter = true
//   einsum    - the full eager dispatch, with its AlgorithmChoice
//   sortgemm  - permute both operands to canonical order, then einsum -> GEMM

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace einsums;
using namespace einsums::index;
namespace ta = einsums::tensor_algebra;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename F>
static double best_of(int reps, F &&f) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double const t0 = now();
        f();
        best = std::min(best, now() - t0);
    }
    return best;
}

static char const *algo_name(ta::detail::AlgorithmChoice a) {
    switch (a) {
    case ta::detail::GENERIC:
        return "GENERIC";
    case ta::detail::DOT:
        return "DOT";
    case ta::detail::DIRECT:
        return "DIRECT";
    case ta::detail::GER:
        return "GER";
    case ta::detail::GEMV:
        return "GEMV";
    case ta::detail::GEMM:
        return "GEMM";
    case ta::detail::PACKED_GEMM:
        return "PACKED_GEMM";
    case ta::detail::SORT_GEMM:
        return "SORT_GEMM";
    default:
        return "INDETERMINATE";
    }
}

int main(int argc, char **argv) {
    einsums::initialize(argc, argv);

    // Extents kept modest so the probe runs in seconds; the SHAPE is what
    // matters, not the size.
    int const    reps = argc > 1 ? std::atoi(argv[1]) : 5;
    size_t const n    = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 12;
    size_t const a = n + 4, b = n, c = n, d = n, e = n + 4, f = n, g = n + 4;

    Tensor<double, 6> C("C", a, b, c, d, e, f);
    Tensor<double, 4> A("A", g, d, b, c);
    Tensor<double, 4> B("B", e, f, g, a);
    C.zero();
    A.set_all(0.5);
    B.set_all(0.25);

    double const flops = 2.0 * static_cast<double>(a * b * c * d) * static_cast<double>(e * f) * static_cast<double>(g);

    // ---- packed, called directly (allow_scatter defaults true) ----
    double const t_packed = best_of(reps, [&] {
        einsums::packed_gemm::try_packed_gemm<false, false>(0.0, Indices{index::a, index::b, index::c, index::d, index::e, index::f}, &C,
                                                            1.0, Indices{index::g, index::d, index::b, index::c}, A,
                                                            Indices{index::e, index::f, index::g, index::a}, B);
    });

    // ---- the eager dispatch, and what it chose ----
    ta::detail::AlgorithmChoice alg      = ta::detail::INDETERMINATE;
    double const                t_einsum = best_of(reps, [&] {
        ta::einsum(0.0, Indices{index::a, index::b, index::c, index::d, index::e, index::f}, &C, 1.0,
                   Indices{index::g, index::d, index::b, index::c}, A, Indices{index::e, index::f, index::g, index::a}, B, &alg);
    });

    std::printf("shape  C[a,b,c,d,e,f] += A[g,d,b,c] * B[e,f,g,a]   M=(d,b,c)=%zu  N=(e,f,a)=%zu  K=g=%zu\n", d * b * c, e * f * a, g);
    std::printf("  packed (direct)   %8.4f s   %7.2f GF/s\n", t_packed, flops / t_packed * 1e-9);
    std::printf("  einsum (eager)    %8.4f s   %7.2f GF/s   chose %s\n", t_einsum, flops / t_einsum * 1e-9, algo_name(alg));
    std::printf("  packed / einsum   %6.2fx  %s\n", t_einsum / t_packed,
                t_einsum / t_packed > 1.02 ? "<- packed is FASTER; fast_scatter should flip"
                                           : (t_einsum / t_packed < 0.98 ? "<- packed is SLOWER; leave fast_scatter alone" : "<- a wash"));

    einsums::finalize();
    return 0;
}
