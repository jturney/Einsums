//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <complex>
#include <cstddef>

// ---------------------------------------------------------------------------
// Direct-product kernels (operate on a unit-stride block of 64 elements).
//
// These are defined in C++ using std::complex rather than C99 `_Complex` on
// purpose: `_Complex` multiplication lowers to the compiler-rt libcalls
// __mulsc3/__muldc3 (Annex G full-range semantics), which are not resolvable
// on the Windows clang-cl link line. std::complex::operator* is inlined with
// the naive formula and emits no libcall, so it links on every toolchain.
// std::complex<T> and `_Complex T` share the same x86-64 ABI, so the AVX2
// assembly kernels (Linux only) are called unchanged.
//
// The scalar fallbacks use EINSUMS_OMP_SIMD (a vectorize-only hint), NOT
// EINSUMS_OMP_SIMD_PRAGMA(for): the latter expands to `#pragma omp for`
// (worksharing) on clang/gcc, and these kernels run inside the caller's
// `#pragma omp parallel for` over blocks - nesting worksharing regions
// deadlocks. (The former .c file was built without -fopenmp, so its pragma
// was inert; compiling here as C++ activates it.)
// ---------------------------------------------------------------------------
extern "C" {
#if defined(__AVX2__) && defined(__FMA3__) && !defined(__ICC) && !defined(__INTEL_COMPILER)
extern int sdirprod_kernel_avx2(size_t n, float alpha, float const *x, float const *y, float *z);
extern int ddirprod_kernel_avx2(size_t n, double alpha, double const *x, double const *y, double *z);
extern int cdirprod_kernel_avx2(size_t n, std::complex<float> alpha, std::complex<float> const *x, std::complex<float> const *y,
                                std::complex<float> *z);
extern int zdirprod_kernel_avx2(size_t n, std::complex<double> alpha, std::complex<double> const *x, std::complex<double> const *y,
                                std::complex<double> *z);

void sdirprod_kernel(size_t n, float alpha, float const *__restrict x, float const *__restrict y, float *__restrict z) {
    sdirprod_kernel_avx2(n, alpha, x, y, z);
}

void ddirprod_kernel(size_t n, double alpha, double const *__restrict x, double const *__restrict y, double *__restrict z) {
    ddirprod_kernel_avx2(n, alpha, x, y, z);
}

void cdirprod_kernel(size_t n, std::complex<float> alpha, std::complex<float> const *__restrict x, std::complex<float> const *__restrict y,
                     std::complex<float> *__restrict z) {
    cdirprod_kernel_avx2(n, alpha, x, y, z);
}

void zdirprod_kernel(size_t n, std::complex<double> alpha, std::complex<double> const *__restrict x,
                     std::complex<double> const *__restrict y, std::complex<double> *__restrict z) {
    zdirprod_kernel_avx2(n, alpha, x, y, z);
}
#else
void sdirprod_kernel(size_t n, float alpha, float const *__restrict x, float const *__restrict y, float *__restrict z) {
    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void ddirprod_kernel(size_t n, double alpha, double const *__restrict x, double const *__restrict y, double *__restrict z) {
    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void cdirprod_kernel(size_t n, std::complex<float> alpha, std::complex<float> const *__restrict x, std::complex<float> const *__restrict y,
                     std::complex<float> *__restrict z) {
    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void zdirprod_kernel(size_t n, std::complex<double> alpha, std::complex<double> const *__restrict x,
                     std::complex<double> const *__restrict y, std::complex<double> *__restrict z) {
    EINSUMS_OMP_SIMD
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}
#endif
} // extern "C"

namespace einsums::blas::vendor {

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::sdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::sdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::ddirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::ddirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::cdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::cdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::zdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::zdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

} // namespace einsums::blas::vendor