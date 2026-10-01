//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The boundary between the SIMD benchmark driver (BenchmarkKernels.cpp, compiled once) and its
// kernels (BenchmarkKernelsImpl.cpp, compiled once per dispatch rung). The kernels take plain
// pointers and sizes and do no timing, logging or allocation, so each rung's copy defines nothing
// another copy shares; the driver times every rung the machine supports through this table.

#pragma once

#include <Einsums/SIMD/RungLadder.hpp>

#include <cstddef>
#include <cstdint>

namespace simd_bench {

/// Number of Taylor terms in the tabulated interpolation.
inline constexpr int taylor_terms = 6;

/// A function tabulated on a uniform grid of spacing dx: c[k][g] is the k-th Taylor coefficient about
/// grid point g.
template <typename T>
struct Table {
    T const *c[taylor_terms];
    T        dx;
    T        inv_dx;
};

/// One rung's kernels. Every array holds n elements, n a multiple of every lane count.
struct Kernels {
    int vector_bits;

    // Index gather: out[i] = table[idx[i]].
    void (*gather_f32)(float const *table, int32_t const *idx, float *out, std::size_t n);
    void (*gather_f64)(double const *table, int64_t const *idx, double *out, std::size_t n);
    /// Doubles at 32-bit indices, as the FP64 tier of a mixed-precision kernel shares the FP32 index.
    void (*gather_f64_i32)(double const *table, int32_t const *idx, double *out, std::size_t n);

    // Conversions.
    void (*widen_f32_f64)(float const *in, double *out, std::size_t n);
    void (*narrow_f64_f32)(double const *in, float *out, std::size_t n);
    void (*f64_to_i64)(double const *in, int64_t *out, std::size_t n);
    void (*f64_to_i32)(double const *in, int32_t *out, std::size_t n);

    // The interpolation, out[i] = sum_k c[k][g] d^k with g = floor(x / dx) and d = x - g dx, as the
    // Boys function is evaluated inside its table.
    void (*interp_f32)(Table<float> const &t, float const *x, float *out, std::size_t n);
    void (*interp_f64)(Table<double> const &t, double const *x, double *out, std::size_t n);
    /// FP64 at FP32's lane count, Vec<double, lanes<float>>, with a 64-bit index.
    void (*interp_f64_wide)(Table<double> const &t, double const *x, double *out, std::size_t n);
    /// The same with a 32-bit index, which is what lets the two tiers share one.
    void (*interp_f64_wide_i32)(Table<double> const &t, double const *x, double *out, std::size_t n);
    /// The scalar instantiation of the same kernel body, compiled at the rung's flags.
    void (*interp_f64_scalar)(Table<double> const &t, double const *x, double *out, std::size_t n);

    // exp, which the Boys function's downward recursion needs once per argument.
    void (*exp_f32)(float const *x, float *out, std::size_t n);
    void (*exp_f64)(double const *x, double *out, std::size_t n);
    void (*exp_f64_wide)(double const *x, double *out, std::size_t n);
    void (*exp_f64_scalar)(double const *x, double *out, std::size_t n);

    // erf and erfc, for a direct F_0(T) and for attenuated Coulomb integrals.
    void (*erf_f64)(double const *x, double *out, std::size_t n);
    void (*erfc_f64)(double const *x, double *out, std::size_t n);
    void (*erfc_f32)(float const *x, float *out, std::size_t n);

    // rsqrt as the library computes it, 1 / sqrt(x) correctly rounded twice, and, for comparison only,
    // the hardware estimate refined by two Newton steps, whose bits differ between CPU vendors.
    void (*rsqrt_f64)(double const *x, double *out, std::size_t n);
    void (*rsqrt_f64_estimate)(double const *x, double *out, std::size_t n);

    // Segmented reduction: out[g] = sum of v[offset[g] .. offset[g + 1]), the primitive-to-contracted
    // sum of integral kernels. Groups are contiguous, of any size, and straddle vector boundaries.
    /// A sequential sum per group.
    void (*segsum_scalar)(double const *v, int32_t const *offset, int groups, double *out);
    /// Per group: vertical adds over its vectors, a masked tail, one horizontal reduction.
    void (*segsum_horizontal)(double const *v, int32_t const *offset, int groups, double *out);
    /// Each group reduced to one vector as above, a register's worth of those transposed, and the
    /// rows added: one transpose instead of a horizontal reduction per group.
    void (*segsum_transpose)(double const *v, int32_t const *offset, int groups, double *out);
    /// As horizontal, but the tail summed in scalar instead of by a masked load.
    void (*segsum_scalar_tail)(double const *v, int32_t const *offset, int groups, double *out);
    /// No segmented reduction at all: the producer writes a register's worth of groups lane-
    /// interleaved, row i holding element i of each, zero past a group's end. Batch b has rows[b]
    /// rows; out receives one vector of sums per batch.
    void (*segsum_interleaved)(double const *v, int32_t const *rows, int batches, double *out);
};

#define EINSUMS_SIMD_BENCH_DECLARE_RUNG(ns)                                                                                                \
    namespace ns {                                                                                                                         \
    Kernels const &kernels() noexcept;                                                                                                     \
    }

EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(EINSUMS_SIMD_BENCH_DECLARE_RUNG)

#undef EINSUMS_SIMD_BENCH_DECLARE_RUNG

} // namespace simd_bench
