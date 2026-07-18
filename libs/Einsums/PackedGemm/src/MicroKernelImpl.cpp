//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Per-rung micro-kernel translation unit. NOT compiled directly: it is
// included by the thin wrappers einsums_add_simd_dispatch_sources() generates
// (one per instruction-set rung), each of which defines EINSUMS_SIMD_ARCH_NS
// and adds the rung's -march flags. The kernel templates therefore compile
// once per rung, in that rung's namespace, at that rung's ISA.
//
// The x86 rungs and the aarch64 native rung use the portable register-block
// bodies from MicroKernelBody.hpp. The aarch64 `sme` rung (compiled with
// +sme2+sme-f64f64) additionally carries an SME outer-product kernel for
// double: each K step issues FMOPA rank-1 updates into ZA64 tile
// accumulators, which is the BLIS micro-kernel expressed in the matrix
// unit's native operation. The rung also widens the double block shape to
// MR = 2*VL, NR = 4*VL (16x32 on Apple M4's 512-bit SVL).

#include <Einsums/PackedGemm/MicroKernel.hpp>

#define EINSUMS_PACKED_GEMM_KERNEL_NS EINSUMS_SIMD_ARCH_NS
#include <Einsums/PackedGemm/MicroKernelBody.hpp>

#include <complex>
#include <cstdint>

#if defined(__ARM_FEATURE_SME2) && defined(__ARM_FEATURE_SME_F64F64)
#    include <arm_sme.h>
#    define EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL 1
#endif

namespace einsums::packed_gemm {
namespace EINSUMS_SIMD_ARCH_NS {

#if defined(EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL)

// Largest streaming vector length the fixed extraction buffer supports:
// SVL 512 bits = 8 doubles per vector, 16x32 tile block. (SVL is 512 on
// Apple M4; a future CPU with a larger SVL falls back to the portable
// kernel via the shape query below.)
inline constexpr int64_t kSmeMaxVl = 8;

/// @brief Accumulate one (2*VL) x (4*VL) double block via SME FMOPA outer
///        products into a contiguous row-major buffer.
///
/// Ap is a column-major MR*kc panel and Bp a row-major kc*NR panel
/// (MR = 2*VL, NR = 4*VL), zero-padded by the packers, so each K step is two
/// A vector loads, four B vector loads, and eight FMOPA rank-1 updates -
/// one per ZA64 tile, giving eight independent accumulator chains, which is
/// what it takes to cover the FMOPA latency (a 2x2-tile variant measures
/// ~187 GFLOPS on M4, this 2x4 arrangement ~234 GFLOPS, near the unit's
/// FP64 ceiling):
///
///   ZA0..ZA3 += a0 (x) b0..b3      ZA4..ZA7 += a1 (x) b0..b3
///
/// Only the accumulation and ZA extraction run in streaming mode; the
/// caller applies the masked, alpha-scaled update into strided C in normal
/// mode where the compiler can use NEON.
__arm_new("za") __arm_locally_streaming static void sme_dgemm_accumulate(int64_t kc, double const *Ap, double const *Bp, double *buf) {
    int64_t const  vl = static_cast<int64_t>(svcntd()); // streaming VL inside the function
    int64_t const  mr = 2 * vl;
    int64_t const  nr = 4 * vl;
    svbool_t const pg = svptrue_b64();

    svzero_za();

    for (int64_t k = 0; k < kc; ++k) {
        double const *a = Ap + k * mr;
        double const *b = Bp + k * nr;

        svfloat64_t const a0 = svld1_f64(pg, a);
        svfloat64_t const a1 = svld1_f64(pg, a + vl);
        svfloat64_t const b0 = svld1_f64(pg, b);
        svfloat64_t const b1 = svld1_f64(pg, b + vl);
        svfloat64_t const b2 = svld1_f64(pg, b + 2 * vl);
        svfloat64_t const b3 = svld1_f64(pg, b + 3 * vl);

        svmopa_za64_f64_m(0, pg, pg, a0, b0);
        svmopa_za64_f64_m(1, pg, pg, a0, b1);
        svmopa_za64_f64_m(2, pg, pg, a0, b2);
        svmopa_za64_f64_m(3, pg, pg, a0, b3);
        svmopa_za64_f64_m(4, pg, pg, a1, b0);
        svmopa_za64_f64_m(5, pg, pg, a1, b1);
        svmopa_za64_f64_m(6, pg, pg, a1, b2);
        svmopa_za64_f64_m(7, pg, pg, a1, b3);
    }

    // Extract the eight ZA64 tiles into the row-major buffer. Tile (i, j)
    // holds C rows [i*VL, (i+1)*VL) x cols [j*VL, (j+1)*VL).
    for (uint32_t r = 0; r < static_cast<uint32_t>(vl); ++r) {
        svst1_hor_za64(0, r, pg, buf + r * nr);
        svst1_hor_za64(1, r, pg, buf + r * nr + vl);
        svst1_hor_za64(2, r, pg, buf + r * nr + 2 * vl);
        svst1_hor_za64(3, r, pg, buf + r * nr + 3 * vl);
        svst1_hor_za64(4, r, pg, buf + (vl + r) * nr);
        svst1_hor_za64(5, r, pg, buf + (vl + r) * nr + vl);
        svst1_hor_za64(6, r, pg, buf + (vl + r) * nr + 2 * vl);
        svst1_hor_za64(7, r, pg, buf + (vl + r) * nr + 3 * vl);
    }
}

#endif // EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL

template <typename T>
void micro_kernel_tile(int mr_block, int nr_block, int64_t kc, T alpha, T const *Ap, T const *Bp, int64_t mr_eff, int64_t nr_eff, T *C,
                       int64_t rs_c, int64_t cs_c) {
#if defined(EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL)
    if constexpr (std::is_same_v<T, double>) {
        int64_t const vl = static_cast<int64_t>(svcntsd()); // streaming VL, queryable from normal mode
        if (vl <= kSmeMaxVl && mr_block == 2 * vl && nr_block == 4 * vl) {
            double        buf[(2 * kSmeMaxVl) * (4 * kSmeMaxVl)];
            int64_t const nr = 4 * vl;
            sme_dgemm_accumulate(kc, Ap, Bp, buf);
            // Normal (non-streaming) mode: masked, alpha-scaled accumulation
            // into strided C, NEON-vectorizable by the compiler.
            for (int64_t i = 0; i < mr_eff; ++i) {
                for (int64_t j = 0; j < nr_eff; ++j) {
                    C[i * rs_c + j * cs_c] += alpha * buf[i * nr + j];
                }
            }
            return;
        }
    }
#endif
    micro_kernel_run<T>(mr_block, nr_block, kc, alpha, Ap, Bp, mr_eff, nr_eff, C, rs_c, cs_c);
}

/// @brief The register-block shape this rung's micro_kernel_tile wants.
template <typename T>
MicroKernelShape micro_kernel_block() {
#if defined(EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL)
    if constexpr (std::is_same_v<T, double>) {
        int64_t const vl = static_cast<int64_t>(svcntsd());
        if (vl <= kSmeMaxVl) {
            return {static_cast<int>(2 * vl), static_cast<int>(4 * vl)};
        }
    }
#endif
    auto const &cfg = cpu_config();
    return {cfg.MR, cfg.NR};
}

template void micro_kernel_tile<float>(int, int, int64_t, float, float const *, float const *, int64_t, int64_t, float *, int64_t, int64_t);
template void micro_kernel_tile<double>(int, int, int64_t, double, double const *, double const *, int64_t, int64_t, double *, int64_t,
                                        int64_t);
template void micro_kernel_tile<std::complex<float>>(int, int, int64_t, std::complex<float>, std::complex<float> const *,
                                                     std::complex<float> const *, int64_t, int64_t, std::complex<float> *, int64_t,
                                                     int64_t);
template void micro_kernel_tile<std::complex<double>>(int, int, int64_t, std::complex<double>, std::complex<double> const *,
                                                      std::complex<double> const *, int64_t, int64_t, std::complex<double> *, int64_t,
                                                      int64_t);

template MicroKernelShape micro_kernel_block<float>();
template MicroKernelShape micro_kernel_block<double>();
template MicroKernelShape micro_kernel_block<std::complex<float>>();
template MicroKernelShape micro_kernel_block<std::complex<double>>();

} // namespace EINSUMS_SIMD_ARCH_NS
} // namespace einsums::packed_gemm
