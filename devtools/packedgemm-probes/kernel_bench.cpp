//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Micro-kernel ceiling: library tile kernel (its own shape) vs a prototype at the rung's
// true vector width vs OpenBLAS, all on one packed MC x KC x NC cache block, single thread.
#include <Einsums/BLAS.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace simd = einsums::simd;
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Prototype: MR = 2 vectors along m, NR = 6 broadcast columns -> 12 accumulators.
template <typename T>
void proto_kernel(int64_t kc, T alpha, T const *__restrict Ap, T const *__restrict Bp, T *C, int64_t rs_c, int64_t cs_c) {
    constexpr int L  = simd::VecTraits<T>::lanes;
    constexpr int MR = 2 * L, NR = 6;
    simd::Vec<T>  acc[2][NR];
    for (int j = 0; j < NR; ++j) {
        acc[0][j] = simd::broadcast(T{0});
        acc[1][j] = simd::broadcast(T{0});
    }
    for (int64_t k = 0; k < kc; ++k) {
        simd::Vec<T> const a0 = simd::loadu(Ap + k * MR);
        simd::Vec<T> const a1 = simd::loadu(Ap + k * MR + L);
        T const           *b  = Bp + k * NR;
#pragma GCC unroll 6
        for (int j = 0; j < NR; ++j) {
            simd::Vec<T> const bj = simd::broadcast(b[j]);
            acc[0][j]             = simd::fmadd(a0, bj, acc[0][j]);
            acc[1][j]             = simd::fmadd(a1, bj, acc[1][j]);
        }
    }
    simd::Vec<T> const va = simd::broadcast(alpha);
    if (rs_c == 1) {
        for (int j = 0; j < NR; ++j) {
            T *c = C + j * cs_c;
            simd::storeu(c, simd::fmadd(va, acc[0][j], simd::loadu(c)));
            simd::storeu(c + L, simd::fmadd(va, acc[1][j], simd::loadu(c + L)));
        }
    } else {
        alignas(64) T buf[MR * NR];
        for (int j = 0; j < NR; ++j) {
            simd::storeu(buf + j * MR, acc[0][j]);
            simd::storeu(buf + j * MR + L, acc[1][j]);
        }
        for (int j = 0; j < NR; ++j)
            for (int i = 0; i < MR; ++i)
                C[i * rs_c + j * cs_c] += alpha * buf[j * MR + i];
    }
}

template <typename T>
void run(char const *name, int64_t MC, int64_t KC, int64_t NC, int MR, int NR, auto &&kernel, double target_s = 2.0) {
    std::vector<T> Ap(static_cast<size_t>(MC * KC), T{1}), Bp(static_cast<size_t>(KC * NC), T{1}), C(static_cast<size_t>(MC * NC), T{0});
    int64_t const  num_ir = MC / MR, num_jr = NC / NR;
    double         flops = 2.0 * MC * NC * KC;
    auto           block = [&] {
        for (int64_t jr = 0; jr < num_jr; ++jr)
            for (int64_t ir = 0; ir < num_ir; ++ir)
                kernel(KC, T{1}, Ap.data() + ir * MR * KC, Bp.data() + jr * NR * KC, C.data() + ir * MR + jr * NR * MC, int64_t{1}, MC);
    };
    block();
    int    reps = 1;
    double t;
    for (;;) {
        double t0 = now();
        for (int r = 0; r < reps; ++r)
            block();
        t = now() - t0;
        if (t > target_s)
            break;
        reps *= 2;
    }
    std::printf("  %-28s MC=%lld KC=%lld NC=%lld MRxNR=%dx%d : %7.2f GF/s\n", name, (long long)MC, (long long)KC, (long long)NC, MR, NR,
                flops * reps / t * 1e-9);
}

template <typename T>
void run_blas(int64_t MC, int64_t KC, int64_t NC, double target_s = 2.0) {
    using einsums::blas::int_t;
    std::vector<T> A(static_cast<size_t>(MC * KC), T{1}), B(static_cast<size_t>(KC * NC), T{1}), C(static_cast<size_t>(MC * NC), T{0});
    double         flops = 2.0 * MC * NC * KC;
    auto           block = [&] {
        einsums::blas::gemm<T>('N', 'T', (int_t)MC, (int_t)NC, (int_t)KC, T{1}, A.data(), (int_t)MC, B.data(), (int_t)NC, T{1}, C.data(),
                               (int_t)MC);
    };
    block();
    int    reps = 1;
    double t;
    for (;;) {
        double t0 = now();
        for (int r = 0; r < reps; ++r)
            block();
        t = now() - t0;
        if (t > target_s)
            break;
        reps *= 2;
    }
    std::printf("  %-28s MC=%lld KC=%lld NC=%lld              : %7.2f GF/s\n", "openblas block gemm", (long long)MC, (long long)KC,
                (long long)NC, flops * reps / t * 1e-9);
}

template <typename T>
void suite(char const *tname) {
    std::printf("%s (proto lanes=%d)\n", tname, simd::VecTraits<T>::lanes);
    auto lib  = einsums::packed_gemm::micro_kernel_entry<T>();
    auto shp  = einsums::packed_gemm::micro_kernel_shape<T>();
    auto libk = [&](int64_t kc, T alpha, T const *Ap, T const *Bp, T *C, int64_t rs, int64_t cs) {
        lib(shp.mr, shp.nr, kc, alpha, Ap, Bp, shp.mr, shp.nr, C, rs, cs);
    };
    auto protok = [&](int64_t kc, T alpha, T const *Ap, T const *Bp, T *C, int64_t rs, int64_t cs) {
        proto_kernel<T>(kc, alpha, Ap, Bp, C, rs, cs);
    };
    constexpr int L = simd::VecTraits<T>::lanes;
    // Library's own blocking today, and a BLIS-derived blocking for the wide tile.
    int64_t const KCw = 32768 / (2 * L * sizeof(T));
    run<T>("library kernel (own blocking)", 32, sizeof(T) == 4 ? 2048 : 1024, 510, shp.mr, shp.nr, libk);
    run<T>("library kernel (wide blocking)", 128, KCw, 2046, shp.mr, shp.nr, libk);
    run<T>("proto 2vec x 6", 128, KCw, 2046, 2 * L, 6, protok);
    run<T>("proto 2vec x 6, KC=256", 128, 256, 2046, 2 * L, 6, protok);
    run<T>("proto 2vec x 6, MC=256", 256, 256, 2046, 2 * L, 6, protok);
    run_blas<T>(128, KCw, 2046);
    run_blas<T>(256, 256, 2046);
    run_blas<T>(32, sizeof(T) == 4 ? 4096 : 4096, 510);
}

int main(int argc, char **argv) {
    einsums::initialize(argc, argv);
    suite<float>("float");
    suite<double>("double");
    einsums::finalize();
}
