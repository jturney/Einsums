//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <complex>
#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

namespace {

/// Element count below which a direct product runs serially.
///
/// The shared elementwise threshold (omp_min_parallel_elements); a fork cost ~23 us against 0.1 us serial.
size_t parallel_threshold() {
    return ::einsums::hardware::omp_min_parallel_elements();
}

// ---------------------------------------------------------------------------
// Unit-stride kernels, z = beta z + alpha x y over one block.
//
// One pass over z: scaling z first and accumulating the product second reads and writes all of z
// twice, which measured 1.3 to 1.6 times slower once the vectors leave cache. A zero beta never
// reads z, as BLAS never reads an output scaled by zero, so an output carved from fresh or reused
// memory that holds NaN still comes out finite.
//
// The kernels use EINSUMS_OMP_SIMD, not EINSUMS_OMP_SIMD_PRAGMA(for): they run inside the caller's
// parallel for, and nested worksharing deadlocks.
// ---------------------------------------------------------------------------

template <typename T>
void block(size_t n, T alpha, T const *__restrict x, T const *__restrict y, T beta, T *__restrict z) {
    if (beta == T{0}) {
        EINSUMS_OMP_SIMD
        for (size_t i = 0; i < n; i++) {
            z[i] = alpha * x[i] * y[i];
        }
    } else {
        EINSUMS_OMP_SIMD
        for (size_t i = 0; i < n; i++) {
            z[i] = beta * z[i] + alpha * x[i] * y[i];
        }
    }
}

// Complex, with the multiply written out on the interleaved real and imaginary parts.
// std::complex's operator* follows C Annex G, recovering an infinity from a NaN result through a
// call per element, which keeps the loop from vectorizing: it ran 1.4 to 2.9 times slower. Finite
// operands give the same result either way.
template <typename R>
void block(size_t n, std::complex<R> alpha, std::complex<R> const *__restrict xc, std::complex<R> const *__restrict yc,
           std::complex<R> beta, std::complex<R> *__restrict zc) {
    auto const *x  = reinterpret_cast<R const *>(xc);
    auto const *y  = reinterpret_cast<R const *>(yc);
    auto       *z  = reinterpret_cast<R *>(zc);
    R const     ar = alpha.real(), ai = alpha.imag(), br = beta.real(), bi = beta.imag();
    if (beta == std::complex<R>{0}) {
        EINSUMS_OMP_SIMD
        for (size_t i = 0; i < n; i++) {
            R const pr   = x[2 * i] * y[2 * i] - x[2 * i + 1] * y[2 * i + 1];
            R const pi   = x[2 * i] * y[2 * i + 1] + x[2 * i + 1] * y[2 * i];
            z[2 * i]     = ar * pr - ai * pi;
            z[2 * i + 1] = ar * pi + ai * pr;
        }
    } else {
        EINSUMS_OMP_SIMD
        for (size_t i = 0; i < n; i++) {
            R const pr = x[2 * i] * y[2 * i] - x[2 * i + 1] * y[2 * i + 1];
            R const pi = x[2 * i] * y[2 * i + 1] + x[2 * i + 1] * y[2 * i];
            R const zr = z[2 * i], zi = z[2 * i + 1];
            z[2 * i]     = br * zr - bi * zi + ar * pr - ai * pi;
            z[2 * i + 1] = br * zi + bi * zr + ar * pi + ai * pr;
        }
    }
}

/// a b, with a complex product written out for the reason the complex block gives.
template <typename T>
T mul(T a, T b) {
    return a * b;
}

template <typename R>
std::complex<R> mul(std::complex<R> a, std::complex<R> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

template <typename T>
void dirprod(int_t n, T alpha, T const *x, int_t incx, T const *y, int_t incy, T beta, T *z, int_t incz) {
    WAGGLE_ZONE_FUNC();

    if (n <= 0) {
        return;
    }

    if (incx == 1 && incy == 1 && incz == 1) {
        auto const blocks    = n / 64;
        auto const remaining = n % 64;
        auto const offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR_IF(n >= parallel_threshold())
            for (int_t i = 0; i < blocks; i++) {
                block(64, alpha, x + i * 64, y + i * 64, beta, z + i * 64);
            }
        }

        if (remaining != 0) {
            block(remaining, alpha, x + offset, y + offset, beta, z + offset);
        }
        return;
    }

    // Offsets in ptrdiff_t: int_t can be 32 bits, and n times a stride overflows it long before the
    // vectors fill memory. A negative increment walks its vector backwards from the far end, as in BLAS.
    std::ptrdiff_t const sx = incx, sy = incy, sz = incz, len = n;
    x += sx < 0 ? (1 - len) * sx : 0;
    y += sy < 0 ? (1 - len) * sy : 0;
    z += sz < 0 ? (1 - len) * sz : 0;

    bool const discard = beta == T{0};
    EINSUMS_OMP_PARALLEL_FOR_SIMD_IF(n >= parallel_threshold())
    for (std::ptrdiff_t i = 0; i < len; i++) {
        T const product = mul(alpha, mul(x[i * sx], y[i * sy]));
        z[i * sz]       = discard ? product : mul(beta, z[i * sz]) + product;
    }
}

} // namespace

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float beta, float *z, int_t incz) {
    dirprod(n, alpha, x, incx, y, incy, beta, z, incz);
}

void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double beta, double *z, int_t incz) {
    dirprod(n, alpha, x, incx, y, incy, beta, z, incz);
}

void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> beta, std::complex<float> *z, int_t incz) {
    dirprod(n, alpha, x, incx, y, incy, beta, z, incz);
}

void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> beta, std::complex<double> *z, int_t incz) {
    dirprod(n, alpha, x, incx, y, incy, beta, z, incz);
}

EINSUMS_NAMESPACE_END(blas::vendor)
