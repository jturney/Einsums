//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Is pack_A's transpose instruction-bound or memory-bound?
//
// Replicates the exact shape pack_A faces on abcde-efcad-bf: A = efcad with
// e(stride 1, extent 48) the unit M axis, f(stride 48, extent 36) the K axis,
// and a(stride 62208, extent 48) the faster-than-e flat-M coordinate. The pack
// must produce panel layout Ap[p][k][i % MR] with i = a + e * 48, which is a
// 48 x 48 transpose between A's order and the panel's.
//
// HOT runs the same group repeatedly (source stays in cache) so the number is
// instruction cost. COLD walks a 430 MB buffer once, as the real pack does.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace {
constexpr int     EX       = 48; // x = e, contiguous in A
constexpr int     SX       = 48; // s = a, the faster flat-M coordinate
constexpr int     KC       = 36; // k = f, stride EX in A
constexpr int     MR       = 16;
constexpr int64_t S_STRIDE = 62208; // a's stride in A, floats

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int64_t dst_slot(int64_t s, int64_t x, int64_t k) {
    int64_t const ii = s + x * SX;
    return (ii / MR) * MR * KC + k * MR + (ii % MR);
}

// (1) What ships today, in essence: walk the panel, gather each element.
void scalar_gather(float const *A, float *Ap, int64_t base) {
    for (int64_t k = 0; k < KC; ++k) {
        for (int64_t x = 0; x < EX; ++x) {
            for (int64_t s = 0; s < SX; ++s) {
                Ap[dst_slot(s, x, k)] = A[base + s * S_STRIDE + x + k * EX];
            }
        }
    }
}

// (2) The tiled scalar transpose I tried: stage MR x (EX*kW), then scatter.
void scalar_tiled(float const *A, float *Ap, int64_t base) {
    constexpr int                         kW = 5;
    alignas(64) static thread_local float stage[MR * EX * kW];
    for (int64_t s0 = 0; s0 < SX; s0 += MR) {
        for (int64_t k0 = 0; k0 < KC; k0 += kW) {
            int64_t const kn = (KC - k0 < kW) ? (KC - k0) : kW;
            for (int64_t s = 0; s < MR; ++s) {
                std::memcpy(stage + s * EX * kW, A + base + (s0 + s) * S_STRIDE + k0 * EX, static_cast<size_t>(EX * kn) * sizeof(float));
            }
            for (int64_t x = 0; x < EX; ++x) {
                int64_t const ii  = s0 + x * SX;
                float *const  dst = Ap + (ii / MR) * MR * KC;
                for (int64_t k = 0; k < kn; ++k) {
                    for (int64_t s = 0; s < MR; ++s) {
                        dst[(k0 + k) * MR + s] = stage[s * EX * kW + k * EX + x];
                    }
                }
            }
        }
    }
}

// (3) Register-blocked 8x8 AVX2 transpose - what HPTT does.
inline void t8x8(__m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3, __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
    __m256 const a0 = _mm256_unpacklo_ps(r0, r1), a1 = _mm256_unpackhi_ps(r0, r1);
    __m256 const a2 = _mm256_unpacklo_ps(r2, r3), a3 = _mm256_unpackhi_ps(r2, r3);
    __m256 const a4 = _mm256_unpacklo_ps(r4, r5), a5 = _mm256_unpackhi_ps(r4, r5);
    __m256 const a6 = _mm256_unpacklo_ps(r6, r7), a7 = _mm256_unpackhi_ps(r6, r7);
    __m256 const b0 = _mm256_shuffle_ps(a0, a2, 0x44), b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
    __m256 const b2 = _mm256_shuffle_ps(a1, a3, 0x44), b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
    __m256 const b4 = _mm256_shuffle_ps(a4, a6, 0x44), b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
    __m256 const b6 = _mm256_shuffle_ps(a5, a7, 0x44), b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
    r0 = _mm256_permute2f128_ps(b0, b4, 0x20);
    r1 = _mm256_permute2f128_ps(b1, b5, 0x20);
    r2 = _mm256_permute2f128_ps(b2, b6, 0x20);
    r3 = _mm256_permute2f128_ps(b3, b7, 0x20);
    r4 = _mm256_permute2f128_ps(b0, b4, 0x31);
    r5 = _mm256_permute2f128_ps(b1, b5, 0x31);
    r6 = _mm256_permute2f128_ps(b2, b6, 0x31);
    r7 = _mm256_permute2f128_ps(b3, b7, 0x31);
}

void avx2_8x8(float const *A, float *Ap, int64_t base) {
    for (int64_t k = 0; k < KC; ++k) {
        for (int64_t s0 = 0; s0 < SX; s0 += 8) {
            for (int64_t x0 = 0; x0 < EX; x0 += 8) {
                float const *src = A + base + s0 * S_STRIDE + x0 + k * EX;
                __m256       r0 = _mm256_loadu_ps(src + 0 * S_STRIDE), r1 = _mm256_loadu_ps(src + 1 * S_STRIDE);
                __m256       r2 = _mm256_loadu_ps(src + 2 * S_STRIDE), r3 = _mm256_loadu_ps(src + 3 * S_STRIDE);
                __m256       r4 = _mm256_loadu_ps(src + 4 * S_STRIDE), r5 = _mm256_loadu_ps(src + 5 * S_STRIDE);
                __m256       r6 = _mm256_loadu_ps(src + 6 * S_STRIDE), r7 = _mm256_loadu_ps(src + 7 * S_STRIDE);
                t8x8(r0, r1, r2, r3, r4, r5, r6, r7);
                // Row j of the transposed tile is 8 consecutive s at x = x0 + j.
                __m256 const rr[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
                for (int j = 0; j < 8; ++j) {
                    int64_t const ii = s0 + (x0 + j) * SX;
                    _mm256_storeu_ps(Ap + (ii / MR) * MR * KC + k * MR + (ii % MR), rr[j]);
                }
            }
        }
    }
}

// (4) Bandwidth reference: move the same bytes with no permutation at all.
void memcpy_ref(float const *A, float *Ap, int64_t base) {
    for (int64_t s = 0; s < SX; ++s) {
        std::memcpy(Ap + s * EX * KC, A + base + s * S_STRIDE, static_cast<size_t>(EX * KC) * sizeof(float));
    }
}

constexpr int64_t kElems = static_cast<int64_t>(SX) * EX * KC;

template <typename F>
void run(char const *name, F &&f, std::vector<float> const &A, std::vector<float> &Ap, int64_t groups, double ghz, bool hot) {
    // warm
    f(A.data(), Ap.data(), 0);
    int    reps = 1;
    double t    = 0;
    for (;;) {
        double const t0 = now();
        for (int r = 0; r < reps; ++r) {
            for (int64_t g = 0; g < groups; ++g) {
                f(A.data(), Ap.data(), hot ? 0 : g * (SX * S_STRIDE));
            }
        }
        t = now() - t0;
        if (t > 1.0)
            break;
        reps *= 2;
    }
    double const elems = static_cast<double>(reps) * groups * kElems;
    std::printf("  %-22s %6.2f cycles/element   %6.2f GB/s\n", name, t * ghz * 1e9 / elems, elems * 4 / t / 1e9);
}
} // namespace

int main(int argc, char **argv) {
    double const       ghz         = (argc > 1) ? std::atof(argv[1]) : 3.9;
    int64_t const      groups_cold = 40;
    std::vector<float> A(static_cast<size_t>(groups_cold * SX * S_STRIDE + EX * KC), 1.0F);
    std::vector<float> Ap(static_cast<size_t>(SX) * EX * KC);
    std::printf("transpose %dx%d x K=%d, source stride %lld floats   (A = %.0f MB, panel = %.0f KB)\n", SX, EX, KC, (long long)S_STRIDE,
                A.size() * 4.0 / 1e6, Ap.size() * 4.0 / 1e3);

    std::printf("HOT (one group, source cache-resident - instruction cost):\n");
    run("scalar gather", scalar_gather, A, Ap, 1, ghz, true);
    run("scalar tiled", scalar_tiled, A, Ap, 1, ghz, true);
    run("avx2 8x8", avx2_8x8, A, Ap, 1, ghz, true);
    run("memcpy (no transpose)", memcpy_ref, A, Ap, 1, ghz, true);

    std::printf("COLD (%lld groups over %.0f MB):\n", (long long)groups_cold, A.size() * 4.0 / 1e6);
    run("scalar gather", scalar_gather, A, Ap, groups_cold, ghz, false);
    run("scalar tiled", scalar_tiled, A, Ap, groups_cold, ghz, false);
    run("avx2 8x8", avx2_8x8, A, Ap, groups_cold, ghz, false);
    run("memcpy (no transpose)", memcpy_ref, A, Ap, groups_cold, ghz, false);
}
