//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Per-rung micro-kernel translation unit. NOT compiled directly: it is
// included by the thin wrappers stripes_add_dispatch_sources() generates
// (one per instruction-set rung), each of which defines STRIPES_ARCH_NS
// and adds the rung's -march flags. The kernel templates therefore compile
// once per rung, in that rung's namespace, at that rung's ISA.
//
// The x86 rungs and the aarch64 native rung use the portable register-block
// bodies from MicroKernelBody.hpp. The aarch64 `sme` rung (compiled with
// +sme2+sme-f64f64) additionally carries an SME outer-product kernel for
// double: each K step issues FMOPA rank-1 updates into ZA64 tile
// accumulators, which is the BLIS micro-kernel expressed in the matrix
// unit's native operation, and moves C through ZA in streaming mode rather
// than adding a copied-out tile in normal mode. The rung also widens the double
// block shape to MR = 2*VL, NR = 4*VL (16x32 on Apple M4's 512-bit SVL).

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>

#define EINSUMS_PACKED_GEMM_KERNEL_NS STRIPES_ARCH_NS
#include <Einsums/PackedGemm/MicroKernelBody.hpp>

#include <Stripes/Shuffle.hpp>
#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>

// The sme rung TU is compiled with -march=...+sme2+sme-f64f64 (flag-probed
// by the CMake helper), which guarantees the FP64 FMOPA intrinsics. Gate on
// __ARM_FEATURE_SME2 only: clang defines no separate feature macro for the
// f64f64 extension, so testing one would (silently!) compile this TU with
// just the portable body.
#if defined(__ARM_FEATURE_SME2)
#    include <arm_sme.h>
#    define EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL 1
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)
namespace STRIPES_ARCH_NS {

#if defined(EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL)

// Largest streaming vector length the fixed extraction buffer supports:
// SVL 512 bits = 8 doubles per vector, 16x32 tile block. (SVL is 512 on
// Apple M4; a future CPU with a larger SVL falls back to the portable
// kernel via the shape query below.)
inline constexpr int64_t kSmeMaxVl = 8;

// How C reaches the accumulators decides the tile's speed. The ZA tiles hold
// them and normal-mode code cannot read ZA, so a tile that is copied out and
// added into C by a normal-mode loop pays for that loop on every tile: on M4 it
// cost three times the arithmetic of a 93-deep double tile, and a quarter of a
// 2048-deep one. Where C has a unit stride along either axis the whole tile
// therefore runs in streaming mode: ZA is loaded from C one contiguous slice at
// a time, the outer products accumulate alpha * A B onto it with alpha folded
// into the A vectors, and ZA is stored back the same way. Predicates cut the
// slices to the ragged tail. Only a C with no unit stride goes through a buffer
// and a normal-mode update.

/// @brief Add a column-major tile buffer (leading dimension @p ld) into a C
///        with no unit stride, walking C along its smaller stride.
template <typename T>
void add_tile_to_strided_c(T alpha, T const *buf, int64_t ld, int64_t mr_eff, int64_t nr_eff, T *C, int64_t rs_c, int64_t cs_c) {
    if (rs_c <= cs_c) {
        for (int64_t j = 0; j < nr_eff; ++j) {
            for (int64_t i = 0; i < mr_eff; ++i) {
                C[i * rs_c + j * cs_c] += alpha * buf[i + j * ld];
            }
        }
    } else {
        for (int64_t i = 0; i < mr_eff; ++i) {
            for (int64_t j = 0; j < nr_eff; ++j) {
                C[i * rs_c + j * cs_c] += alpha * buf[i + j * ld];
            }
        }
    }
}

// ---- double: MR = 2*VL, NR = 4*VL, eight ZA64 tiles ------------------------
//
// Tile (ti, tj) holds C rows [ti*VL, (ti+1)*VL) x cols [tj*VL, (tj+1)*VL) and is
// ZA tile 4*ti + tj. Column j of the block is therefore vertical slice j % VL of
// tiles j / VL and 4 + j / VL, and row i is horizontal slice i % VL of tiles
// 4*(i / VL) + 0..3.

/// @brief The K loop: two A vector loads, four B vector loads and eight FMOPA
///        rank-1 updates per step, one per ZA64 tile.
///
/// Eight independent accumulator chains cover the FMOPA latency; this reaches
/// about 500 GFLOP/s on one M4 core, the unit's FP64 ceiling:
///
///   ZA0..ZA3 += a0 (x) b0..b3      ZA4..ZA7 += a1 (x) b0..b3
///
/// @p Scale multiplies each A vector by @p va first.
template <bool Scale>
__attribute__((always_inline)) inline void sme_dgemm_kloop(int64_t kc, svfloat64_t va, double const *Ap,
                                                           double const *Bp) __arm_streaming __arm_inout("za") {
    int64_t const  vl = static_cast<int64_t>(svcntd());
    int64_t const  mr = 2 * vl;
    int64_t const  nr = 4 * vl;
    svbool_t const pg = svptrue_b64();

    for (int64_t k = 0; k < kc; ++k) {
        double const *a = Ap + k * mr;
        double const *b = Bp + k * nr;

        svfloat64_t a0 = svld1_f64(pg, a);
        svfloat64_t a1 = svld1_f64(pg, a + vl);
        if constexpr (Scale) {
            a0 = svmul_f64_x(pg, a0, va);
            a1 = svmul_f64_x(pg, a1, va);
        }
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
}

__attribute__((always_inline)) inline void sme_dgemm_kloop_alpha(int64_t kc, double alpha, double const *Ap,
                                                                 double const *Bp) __arm_streaming __arm_inout("za") {
    if (alpha == 1.0) {
        sme_dgemm_kloop<false>(kc, svdup_f64(1.0), Ap, Bp);
    } else {
        sme_dgemm_kloop<true>(kc, svdup_f64(alpha), Ap, Bp);
    }
}

// One column of C (ld = column stride) through vertical slice c of tiles tj and 4 + tj.
#    define EINSUMS_SME_D_COL(op, tj)                                                                                                      \
        if ((tj) * vl + c < nr_eff) {                                                                                                      \
            double *col = C + ((tj) * vl + c) * ldc;                                                                                       \
            op((tj), slice, top, col);                                                                                                     \
            if (two_halves) {                                                                                                              \
                op(4 + (tj), slice, bot, col + vl);                                                                                        \
            }                                                                                                                              \
        }

// One row of C (ld = row stride) through horizontal slice r of tiles 4*ti + 0..3.
#    define EINSUMS_SME_D_ROW(op, ti, tj)                                                                                                  \
        if ((tj) * vl < nr_eff) {                                                                                                          \
            op(4 * (ti) + (tj), slice, cols##tj, row + (tj) * vl);                                                                         \
        }

/// @brief The double tile, C contiguous down its columns (rs_c == 1).
__arm_new("za") __arm_locally_streaming static void sme_dgemm_tile_cols(int64_t kc, double alpha, double const *Ap, double const *Bp,
                                                                        int64_t mr_eff, int64_t nr_eff, double *C, int64_t ldc) {
    int64_t const  vl         = static_cast<int64_t>(svcntd());
    svbool_t const top        = svwhilelt_b64_s64(0, mr_eff);
    svbool_t const bot        = svwhilelt_b64_s64(vl, mr_eff);
    bool const     two_halves = mr_eff > vl;

    svzero_za();
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        EINSUMS_SME_D_COL(svld1_ver_za64, 0)
        EINSUMS_SME_D_COL(svld1_ver_za64, 1)
        EINSUMS_SME_D_COL(svld1_ver_za64, 2)
        EINSUMS_SME_D_COL(svld1_ver_za64, 3)
    }
    sme_dgemm_kloop_alpha(kc, alpha, Ap, Bp);
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        EINSUMS_SME_D_COL(svst1_ver_za64, 0)
        EINSUMS_SME_D_COL(svst1_ver_za64, 1)
        EINSUMS_SME_D_COL(svst1_ver_za64, 2)
        EINSUMS_SME_D_COL(svst1_ver_za64, 3)
    }
}

/// @brief The double tile, C contiguous along its rows (cs_c == 1).
__arm_new("za") __arm_locally_streaming static void sme_dgemm_tile_rows(int64_t kc, double alpha, double const *Ap, double const *Bp,
                                                                        int64_t mr_eff, int64_t nr_eff, double *C, int64_t ldc) {
    int64_t const  vl    = static_cast<int64_t>(svcntd());
    svbool_t const cols0 = svwhilelt_b64_s64(0, nr_eff);
    svbool_t const cols1 = svwhilelt_b64_s64(vl, nr_eff);
    svbool_t const cols2 = svwhilelt_b64_s64(2 * vl, nr_eff);
    svbool_t const cols3 = svwhilelt_b64_s64(3 * vl, nr_eff);

    svzero_za();
    for (int64_t r = 0; r < vl; ++r) {
        uint32_t const slice = static_cast<uint32_t>(r);
        if (r < mr_eff) {
            double *row = C + r * ldc;
            EINSUMS_SME_D_ROW(svld1_hor_za64, 0, 0)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 0, 1)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 0, 2)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 0, 3)
        }
        if (vl + r < mr_eff) {
            double *row = C + (vl + r) * ldc;
            EINSUMS_SME_D_ROW(svld1_hor_za64, 1, 0)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 1, 1)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 1, 2)
            EINSUMS_SME_D_ROW(svld1_hor_za64, 1, 3)
        }
    }
    sme_dgemm_kloop_alpha(kc, alpha, Ap, Bp);
    for (int64_t r = 0; r < vl; ++r) {
        uint32_t const slice = static_cast<uint32_t>(r);
        if (r < mr_eff) {
            double *row = C + r * ldc;
            EINSUMS_SME_D_ROW(svst1_hor_za64, 0, 0)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 0, 1)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 0, 2)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 0, 3)
        }
        if (vl + r < mr_eff) {
            double *row = C + (vl + r) * ldc;
            EINSUMS_SME_D_ROW(svst1_hor_za64, 1, 0)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 1, 1)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 1, 2)
            EINSUMS_SME_D_ROW(svst1_hor_za64, 1, 3)
        }
    }
}

/// @brief The double tile into a column-major buffer (ld = MR), for a C with no unit stride.
__arm_new("za") __arm_locally_streaming static void sme_dgemm_tile_buffer(int64_t kc, double const *Ap, double const *Bp, double *buf) {
    int64_t const  vl = static_cast<int64_t>(svcntd());
    int64_t const  mr = 2 * vl;
    svbool_t const pg = svptrue_b64();

    svzero_za();
    sme_dgemm_kloop<false>(kc, svdup_f64(1.0), Ap, Bp);
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        svst1_ver_za64(0, slice, pg, buf + (0 * vl + c) * mr);
        svst1_ver_za64(1, slice, pg, buf + (1 * vl + c) * mr);
        svst1_ver_za64(2, slice, pg, buf + (2 * vl + c) * mr);
        svst1_ver_za64(3, slice, pg, buf + (3 * vl + c) * mr);
        svst1_ver_za64(4, slice, pg, buf + (0 * vl + c) * mr + vl);
        svst1_ver_za64(5, slice, pg, buf + (1 * vl + c) * mr + vl);
        svst1_ver_za64(6, slice, pg, buf + (2 * vl + c) * mr + vl);
        svst1_ver_za64(7, slice, pg, buf + (3 * vl + c) * mr + vl);
    }
}

#    undef EINSUMS_SME_D_COL
#    undef EINSUMS_SME_D_ROW

// ---- float: MR = NR = 2*VL32, four ZA32 tiles -----------------------------
//
// Tile (ti, tj) is ZA tile 2*ti + tj; f32 FMOPA is four times as dense per
// instruction as f64. Same slice arithmetic as the double tile with a 2 x 2
// grid.

template <bool Scale>
__attribute__((always_inline)) inline void sme_sgemm_kloop(int64_t kc, svfloat32_t va, float const *Ap,
                                                           float const *Bp) __arm_streaming __arm_inout("za") {
    int64_t const  vl = static_cast<int64_t>(svcntw());
    int64_t const  mr = 2 * vl;
    svbool_t const pg = svptrue_b32();

    for (int64_t k = 0; k < kc; ++k) {
        float const *a = Ap + k * mr;
        float const *b = Bp + k * mr;

        svfloat32_t a0 = svld1_f32(pg, a);
        svfloat32_t a1 = svld1_f32(pg, a + vl);
        if constexpr (Scale) {
            a0 = svmul_f32_x(pg, a0, va);
            a1 = svmul_f32_x(pg, a1, va);
        }
        svfloat32_t const b0 = svld1_f32(pg, b);
        svfloat32_t const b1 = svld1_f32(pg, b + vl);

        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);
    }
}

__attribute__((always_inline)) inline void sme_sgemm_kloop_alpha(int64_t kc, float alpha, float const *Ap,
                                                                 float const *Bp) __arm_streaming __arm_inout("za") {
    if (alpha == 1.0F) {
        sme_sgemm_kloop<false>(kc, svdup_f32(1.0F), Ap, Bp);
    } else {
        sme_sgemm_kloop<true>(kc, svdup_f32(alpha), Ap, Bp);
    }
}

#    define EINSUMS_SME_S_COL(op, tj)                                                                                                      \
        if ((tj) * vl + c < nr_eff) {                                                                                                      \
            float *col = C + ((tj) * vl + c) * ldc;                                                                                        \
            op((tj), slice, top, col);                                                                                                     \
            if (two_halves) {                                                                                                              \
                op(2 + (tj), slice, bot, col + vl);                                                                                        \
            }                                                                                                                              \
        }

#    define EINSUMS_SME_S_ROW(op, ti, tj)                                                                                                  \
        if ((tj) * vl < nr_eff) {                                                                                                          \
            op(2 * (ti) + (tj), slice, cols##tj, row + (tj) * vl);                                                                         \
        }

/// @brief The float tile, C contiguous down its columns (rs_c == 1).
__arm_new("za") __arm_locally_streaming static void sme_sgemm_tile_cols(int64_t kc, float alpha, float const *Ap, float const *Bp,
                                                                        int64_t mr_eff, int64_t nr_eff, float *C, int64_t ldc) {
    int64_t const  vl         = static_cast<int64_t>(svcntw());
    svbool_t const top        = svwhilelt_b32_s64(0, mr_eff);
    svbool_t const bot        = svwhilelt_b32_s64(vl, mr_eff);
    bool const     two_halves = mr_eff > vl;

    svzero_za();
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        EINSUMS_SME_S_COL(svld1_ver_za32, 0)
        EINSUMS_SME_S_COL(svld1_ver_za32, 1)
    }
    sme_sgemm_kloop_alpha(kc, alpha, Ap, Bp);
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        EINSUMS_SME_S_COL(svst1_ver_za32, 0)
        EINSUMS_SME_S_COL(svst1_ver_za32, 1)
    }
}

/// @brief The float tile, C contiguous along its rows (cs_c == 1).
__arm_new("za") __arm_locally_streaming static void sme_sgemm_tile_rows(int64_t kc, float alpha, float const *Ap, float const *Bp,
                                                                        int64_t mr_eff, int64_t nr_eff, float *C, int64_t ldc) {
    int64_t const  vl    = static_cast<int64_t>(svcntw());
    svbool_t const cols0 = svwhilelt_b32_s64(0, nr_eff);
    svbool_t const cols1 = svwhilelt_b32_s64(vl, nr_eff);

    svzero_za();
    for (int64_t r = 0; r < vl; ++r) {
        uint32_t const slice = static_cast<uint32_t>(r);
        if (r < mr_eff) {
            float *row = C + r * ldc;
            EINSUMS_SME_S_ROW(svld1_hor_za32, 0, 0)
            EINSUMS_SME_S_ROW(svld1_hor_za32, 0, 1)
        }
        if (vl + r < mr_eff) {
            float *row = C + (vl + r) * ldc;
            EINSUMS_SME_S_ROW(svld1_hor_za32, 1, 0)
            EINSUMS_SME_S_ROW(svld1_hor_za32, 1, 1)
        }
    }
    sme_sgemm_kloop_alpha(kc, alpha, Ap, Bp);
    for (int64_t r = 0; r < vl; ++r) {
        uint32_t const slice = static_cast<uint32_t>(r);
        if (r < mr_eff) {
            float *row = C + r * ldc;
            EINSUMS_SME_S_ROW(svst1_hor_za32, 0, 0)
            EINSUMS_SME_S_ROW(svst1_hor_za32, 0, 1)
        }
        if (vl + r < mr_eff) {
            float *row = C + (vl + r) * ldc;
            EINSUMS_SME_S_ROW(svst1_hor_za32, 1, 0)
            EINSUMS_SME_S_ROW(svst1_hor_za32, 1, 1)
        }
    }
}

/// @brief The float tile into a column-major buffer (ld = MR), for a C with no unit stride.
__arm_new("za") __arm_locally_streaming static void sme_sgemm_tile_buffer(int64_t kc, float const *Ap, float const *Bp, float *buf) {
    int64_t const  vl = static_cast<int64_t>(svcntw());
    int64_t const  mr = 2 * vl;
    svbool_t const pg = svptrue_b32();

    svzero_za();
    sme_sgemm_kloop<false>(kc, svdup_f32(1.0F), Ap, Bp);
    for (int64_t c = 0; c < vl; ++c) {
        uint32_t const slice = static_cast<uint32_t>(c);
        svst1_ver_za32(0, slice, pg, buf + (0 * vl + c) * mr);
        svst1_ver_za32(1, slice, pg, buf + (1 * vl + c) * mr);
        svst1_ver_za32(2, slice, pg, buf + (0 * vl + c) * mr + vl);
        svst1_ver_za32(3, slice, pg, buf + (1 * vl + c) * mr + vl);
    }
}

#    undef EINSUMS_SME_S_COL
#    undef EINSUMS_SME_S_ROW

// ---- Panel transpose through a ZA tile -----------------------------------
//
// pack_transpose_rows' contract (panel[r + k * ld] = rows[r][k]) is a
// transpose of K-contiguous runs, and a ZA tile is a transpose engine: VL rows
// go in as horizontal slices, each read along K, and come out as vertical
// slices, each VL consecutive panel elements of one K column. A VL x VL block
// costs VL vector loads and VL vector stores; the NEON rung's two double lanes
// need a shuffle per pair of elements instead. On M4 the panel of a 93-row,
// cache-resident operand packs at 0.08 ns per element this way against 0.32 for
// the 2 x 2 register transpose, and from DRAM at 0.27 against 0.37. Predicates
// cut the slices to the ragged row count and the K tail.

__arm_new("za") __arm_locally_streaming static void sme_pack_transpose_d(double *panel, double const *const *rows, int64_t nrows,
                                                                         int64_t kc, int64_t ld) {
    int64_t const vl = static_cast<int64_t>(svcntd());
    for (int64_t r0 = 0; r0 < nrows; r0 += vl) {
        int64_t const  h_end   = std::min(vl, nrows - r0);
        svbool_t const rows_ok = svwhilelt_b64_s64(r0, nrows);
        for (int64_t k = 0; k < kc; k += vl) {
            svbool_t const ks = svwhilelt_b64_s64(k, kc);
            for (int64_t h = 0; h < h_end; ++h) {
                svld1_hor_za64(0, static_cast<uint32_t>(h), ks, rows[r0 + h] + k);
            }
            int64_t const c_end = std::min(vl, kc - k);
            for (int64_t c = 0; c < c_end; ++c) {
                svst1_ver_za64(0, static_cast<uint32_t>(c), rows_ok, panel + r0 + (k + c) * ld);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming static void sme_pack_transpose_s(float *panel, float const *const *rows, int64_t nrows, int64_t kc,
                                                                         int64_t ld) {
    int64_t const vl = static_cast<int64_t>(svcntw());
    for (int64_t r0 = 0; r0 < nrows; r0 += vl) {
        int64_t const  h_end   = std::min(vl, nrows - r0);
        svbool_t const rows_ok = svwhilelt_b32_s64(r0, nrows);
        for (int64_t k = 0; k < kc; k += vl) {
            svbool_t const ks = svwhilelt_b32_s64(k, kc);
            for (int64_t h = 0; h < h_end; ++h) {
                svld1_hor_za32(0, static_cast<uint32_t>(h), ks, rows[r0 + h] + k);
            }
            int64_t const c_end = std::min(vl, kc - k);
            for (int64_t c = 0; c < c_end; ++c) {
                svst1_ver_za32(0, static_cast<uint32_t>(c), rows_ok, panel + r0 + (k + c) * ld);
            }
        }
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
            if (rs_c == 1) {
                sme_dgemm_tile_cols(kc, alpha, Ap, Bp, mr_eff, nr_eff, C, cs_c);
            } else if (cs_c == 1) {
                sme_dgemm_tile_rows(kc, alpha, Ap, Bp, mr_eff, nr_eff, C, rs_c);
            } else {
                double buf[(2 * kSmeMaxVl) * (4 * kSmeMaxVl)];
                sme_dgemm_tile_buffer(kc, Ap, Bp, buf);
                add_tile_to_strided_c(alpha, buf, 2 * vl, mr_eff, nr_eff, C, rs_c, cs_c);
            }
            return;
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        int64_t const vl = static_cast<int64_t>(svcntsw()); // f32 streaming VL, queryable from normal mode
        if (vl <= 2 * kSmeMaxVl && mr_block == 2 * vl && nr_block == 2 * vl) {
            if (rs_c == 1) {
                sme_sgemm_tile_cols(kc, alpha, Ap, Bp, mr_eff, nr_eff, C, cs_c);
            } else if (cs_c == 1) {
                sme_sgemm_tile_rows(kc, alpha, Ap, Bp, mr_eff, nr_eff, C, rs_c);
            } else {
                float buf[(4 * kSmeMaxVl) * (4 * kSmeMaxVl)];
                sme_sgemm_tile_buffer(kc, Ap, Bp, buf);
                add_tile_to_strided_c(alpha, buf, 2 * vl, mr_eff, nr_eff, C, rs_c, cs_c);
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
            return {static_cast<int>(2 * vl), static_cast<int>(4 * vl), int64_t{4096}, /*fast_scatter=*/true, /*block_gemm=*/false};
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        int64_t const vl = static_cast<int64_t>(svcntsw());
        if (vl <= 2 * kSmeMaxVl) {
            // Measured on M4: f32 FMOPA scatter runs the ring benchmark in
            // 1.67 ms against Sort+GEMM's 7.12 ms (4.3x).
            return {static_cast<int>(2 * vl), static_cast<int>(2 * vl), int64_t{4096}, /*fast_scatter=*/true, /*block_gemm=*/false};
        }
    }
    // Complex on the SME rung: 1m through the REAL kernel of the underlying
    // type - there is no complex FMOPA, and 1m puts all complex arithmetic on
    // the real ZA tiles (measured 1.74x over Sort+GEMM for complex<double>).
    // The advertised geometry is the real kernel's; blis_contraction doubles
    // the working extents.
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        int64_t const vl = static_cast<int64_t>(svcntsd());
        if (vl <= kSmeMaxVl) {
            return {static_cast<int>(2 * vl), static_cast<int>(4 * vl), int64_t{4096}, /*fast_scatter=*/true, /*block_gemm=*/false,
                    /*use_1m=*/true};
        }
    }
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        int64_t const vl = static_cast<int64_t>(svcntsw());
        if (vl <= 2 * kSmeMaxVl) {
            return {static_cast<int>(2 * vl), static_cast<int>(2 * vl), int64_t{4096}, /*fast_scatter=*/true, /*block_gemm=*/false,
                    /*use_1m=*/true};
        }
    }
#endif
    auto const      &cfg = cpu_config();
    MicroKernelShape shape{cfg.MR, cfg.NR};
    // Real types take this rung's vector tile (see MicroKernelBody.hpp), not cfg.MR, which follows
    // the library's compile width.
    if constexpr (has_vector_kernel<T>) {
        shape.mr = vector_kernel_mr<T>;
        shape.nr = vector_kernel_nr;
    }
#if defined(__x86_64__) || defined(_M_X64)
    // With the tile stated in the rung's own vectors the kernel runs at or
    // above the vendor's full-problem rate on one packed block (Zen+, single
    // core: 61 vs 57 GF/s float, 30 vs 28 double), while a vendor GEMM called
    // per cache block re-packs that block internally on every call and
    // measured 28 GF/s float on the same panels. So on x86 the scatter shapes
    // run the rung's own tile loops rather than block GEMMs. Complex keeps the
    // block strategy: it has no vector tile here.
    if constexpr (has_vector_kernel<T>) {
        shape.block_gemm = false;
        // Take the scatter path even with a TTGT fallback: 18.75 GF/s against Sort+GEMM's 3.23 on the
        // rank-6 ccsd_t pattern with interleaved C groups (devtools/packedgemm-probes/scatter_probe.cpp;
        // the benchmarks' C groups coalesce). Batched shapes still decline.
        shape.fast_scatter = true;
    }
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    // Accelerate's GEMM reaches the AMX/SME matrix unit that the portable
    // tile kernel cannot, so the block-GEMM scatter strategy beats Sort+GEMM
    // here (measured ~11% on the CCSD ring benchmark) while eliminating its
    // operand-sized temporaries. Measured for double; other types keep the
    // conservative default.
    if constexpr (std::is_same_v<T, double>) {
        shape.fast_scatter = true;
    }
    // Complex on Apple without SME: the 3m method runs the block path on
    // three real GEMMs (AMX-backed via Accelerate) instead of one complex
    // GEMM. Measured on the ring benchmark (baseline rung, M4 as an M1-M3
    // proxy): 17.7 ms vs Sort+GEMM's 23.4 ms (1.32x) for complex<double>.
    if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
        shape.use_3m       = true;
        shape.fast_scatter = true;
    }
#endif
    return shape;
}

/// @brief The register-block shape of this rung's 1m route for complex @p T,
///        or micro_kernel_block<T>() where the rung has none.
///
/// The opt-in counterpart of the SME branch above (see
/// option::PackedGemmComplex1m): on an x86 rung from V2 up, complex runs on the
/// REAL vector tile of its underlying type, so the advertised geometry is that
/// kernel's, and blis_contraction doubles the working extents. Only the engine
/// changes. fast_scatter keeps the complex shape's value, so the decision of
/// which contractions reach the packed loops at all is the same with the flag
/// on or off, and kc stays 0: the K block comes from the cache model, which
/// keeps the B micro-panel in L1, where the SME rung's deep ZA-tile K block
/// would not.
template <typename T>
MicroKernelShape micro_kernel_block_1m() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__SSE4_2__) || defined(__AVX__))
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        using RealT = typename T::value_type;
        if constexpr (has_vector_kernel<RealT>) {
            MicroKernelShape shape = micro_kernel_block<RealT>();
            shape.kc               = 0;
            shape.block_gemm       = false;
            shape.fast_scatter     = micro_kernel_block<T>().fast_scatter;
            shape.use_1m           = true;
            shape.use_3m           = false;
            return shape;
        }
    }
#endif
    return micro_kernel_block<T>();
}

/// pack_A's K-contiguous transpose at this rung's register width: a lanes x
/// lanes tile is loaded along K from lanes rows, transposed in registers and
/// stored as lanes whole vectors of panel columns, where a scalar copy stores
/// one element at a time. A panel of fewer rows than lanes whose columns are
/// contiguous (ld == nrows) goes out as one interleaved store per tile. Other
/// tails, in either direction, take the scalar copy.
// The tile loops must unroll completely for the tile to stay in registers; at
// the library's -O2 GCC leaves them rolled and spills the tile to the stack,
// which cost float panels 11 to 17 per cent (abc-dca-bd, abcd-ebad-ce).
#if defined(__GNUC__) || defined(__clang__)
#    define EINSUMS_PACK_UNROLL _Pragma("GCC unroll 32")
#else
#    define EINSUMS_PACK_UNROLL
#endif

/// A whole panel of R < lanes rows whose columns are contiguous (ld == R), as every full pack_B panel
/// of NR = 6 rows is on a rung wider than six lanes: each lanes-wide K block is R rows loaded whole and
/// written as R * lanes contiguous elements by one interleaved store. The K tail is a scalar copy.
template <typename T, int R>
void pack_rows_interleaved(T *panel, T const *const *rows, int64_t kc) {
    constexpr int64_t L = stripes::native_lanes<T>;
    int64_t           k = 0;
    for (; k + L <= kc; k += L) {
        stripes::Vec<T> tile[R];
        EINSUMS_PACK_UNROLL
        for (int i = 0; i < R; ++i) {
            tile[i] = stripes::loadu(rows[i] + k);
        }
        stripes::storeu_interleaved<R>(panel + k * R, tile);
    }
    for (; k < kc; ++k) {
        for (int i = 0; i < R; ++i) {
            panel[i + k * R] = rows[i][k];
        }
    }
}

template <typename T, int... Rm1>
constexpr auto interleaved_packers(std::integer_sequence<int, Rm1...>) {
    return std::array<void (*)(T *, T const *const *, int64_t), sizeof...(Rm1)>{&pack_rows_interleaved<T, Rm1 + 1>...};
}

template <typename T>
void pack_transpose_rows(T *panel, T const *const *rows, int64_t nrows, int64_t kc, int64_t ld) {
#if defined(EINSUMS_PACKED_GEMM_HAVE_SME_KERNEL)
    // A run shorter than one streaming vector does not fill a ZA block; it stays on the NEON path.
    if constexpr (std::is_same_v<T, double>) {
        if (kc >= static_cast<int64_t>(svcntsd())) {
            sme_pack_transpose_d(panel, rows, nrows, kc, ld);
            return;
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        if (kc >= static_cast<int64_t>(svcntsw())) {
            sme_pack_transpose_s(panel, rows, nrows, kc, ld);
            return;
        }
    }
#endif
    constexpr int64_t L = stripes::native_lanes<T>;
    if constexpr (L > 1) {
        if (nrows > 0 && nrows < L && ld == nrows) {
            static constexpr auto packers = interleaved_packers<T>(std::make_integer_sequence<int, static_cast<int>(L) - 1>{});
            packers[static_cast<size_t>(nrows - 1)](panel, rows, kc);
            return;
        }
    }
    int64_t r = 0;
    if constexpr (L > 1) {
        for (; r + L <= nrows; r += L) {
            T const *const *src = rows + r;
            T              *col = panel + r;
            int64_t         k   = 0;
            for (; k + L <= kc; k += L) {
                stripes::Vec<T> tile[L];
                EINSUMS_PACK_UNROLL
                for (int64_t i = 0; i < L; ++i) {
                    tile[i] = stripes::loadu(src[i] + k);
                }
                stripes::transpose_inplace(tile);
                EINSUMS_PACK_UNROLL
                for (int64_t i = 0; i < L; ++i) {
                    stripes::storeu(col + (k + i) * ld, tile[i]);
                }
            }
            for (; k < kc; ++k) {
                for (int64_t i = 0; i < L; ++i) {
                    col[i + k * ld] = src[i][k];
                }
            }
        }
    }
    for (; r < nrows; ++r) {
        for (int64_t k = 0; k < kc; ++k) {
            panel[r + k * ld] = rows[r][k];
        }
    }
}

template void pack_transpose_rows<float>(float *, float const *const *, int64_t, int64_t, int64_t);
template void pack_transpose_rows<double>(double *, double const *const *, int64_t, int64_t, int64_t);

#undef EINSUMS_PACK_UNROLL

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

template MicroKernelShape micro_kernel_block_1m<float>();
template MicroKernelShape micro_kernel_block_1m<double>();
template MicroKernelShape micro_kernel_block_1m<std::complex<float>>();
template MicroKernelShape micro_kernel_block_1m<std::complex<double>>();

} // namespace STRIPES_ARCH_NS
EINSUMS_NAMESPACE_END(packed_gemm)
