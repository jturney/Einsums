
#include <Einsums/Config/CompilerSpecific.hpp>

#include <stddef.h>

#if defined(__AVX2__) && defined(__FMA3__) && !defined(__ICC) && !defined(__INTEL_COMPILER)
extern int sdirprod_kernel_avx2(size_t n, float alpha, float const *x, float const *y, float *z);
extern int ddirprod_kernel_avx2(size_t n, double alpha, double const *x, double const *y, double *z);
extern int cdirprod_kernel_avx2(size_t n, _Complex float alpha, _Complex float const *x, _Complex float const *y,
                                          _Complex float *z);
extern int zdirprod_kernel_avx2(size_t n, _Complex double alpha, _Complex double const *x,
                                          _Complex double const *y, _Complex double *z);

void sdirprod_kernel(size_t n, float alpha, float const *__restrict__ x, float const *__restrict__ y, float *__restrict__ z) {
    sdirprod_kernel_avx2(n, alpha, x, y, z);
}

void ddirprod_kernel(size_t n, double alpha, double const *__restrict__ x, double const *__restrict__ y, double *__restrict__ z) {
    ddirprod_kernel_avx2(n, alpha, x, y, z);
}

void cdirprod_kernel(size_t n, _Complex float alpha, _Complex float const *__restrict__ x, _Complex float const *__restrict__ y,
                     _Complex float *__restrict__ z) {
    cdirprod_kernel_avx2(n, alpha, x, y, z);
}

void zdirprod_kernel(size_t n, _Complex double alpha, _Complex double const *__restrict__ x, _Complex double const *__restrict__ y,
                     _Complex double *__restrict__ z) {
    zdirprod_kernel_avx2(n, alpha, x, y, z);
}
#else
void sdirprod_kernel(size_t n, float alpha, float const *__restrict__ x, float const *__restrict__ y, float *__restrict__ z) {
    EINSUMS_OMP_SIMD_PRAGMA(for)
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void ddirprod_kernel(size_t n, double alpha, double const *__restrict__ x, double const *__restrict__ y, double *__restrict__ z) {
    EINSUMS_OMP_SIMD_PRAGMA(for)
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void cdirprod_kernel(size_t n, _Complex float alpha, _Complex float const *__restrict__ x, _Complex float const *__restrict__ y,
                     _Complex float *__restrict__ z) {
    EINSUMS_OMP_SIMD_PRAGMA(for)
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}

void zdirprod_kernel(size_t n, _Complex double alpha, _Complex double const *__restrict__ x, _Complex double const *__restrict__ y,
                     _Complex double *__restrict__ z) {
    EINSUMS_OMP_SIMD_PRAGMA(for)
    for (size_t i = 0; i < n; i++) {
        z[i] = z[i] + alpha * x[i] * y[i];
    }
}
#endif
