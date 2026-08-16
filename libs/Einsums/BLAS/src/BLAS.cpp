//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(blas::detail)

namespace {
/// Clamp the OpenMP ICV to one thread for a vendor call made under a
/// node-scoped width, and restore it after.
///
/// The width a moldable executor grants a task is for the kernels einsums
/// threads itself. An OpenMP-built OpenBLAS reads the same ICV but does not
/// support concurrent callers that disagree about it (see
/// @ref einsums::blas::set_moldable_width_scope), so a vendor call made from
/// inside such a scope must present a width of one. Threads outside any
/// scope - the main thread, eager callers, an OpenMPExecutor level - are
/// untouched: a single caller with a stable ICV is the vendor's supported
/// regime, and it is where a wide vendor GEMM is still wanted.
///
/// Placed on every wrapper that forwards to the vendor, because a moldable
/// Custom node's lambda may legally call any of them - a large axpy and a
/// syev thread inside OpenBLAS just as a gemm does. The only exemptions are
/// the gemm_batch families: their vendor entry points are einsums' own
/// OpenMP loops over raw serial GEMMs, so clamping at the wrapper would
/// serialize einsums' parallelism while buying nothing - the inner calls
/// already run inside an active region, where the vendor's own
/// omp_in_parallel check takes the serial path without touching its global.
/// That same nested check is why the flag not being visible to threads of a
/// team the KERNEL forks is harmless: those team members are inside an
/// active region too.
struct [[maybe_unused]] VendorWidthFence {
#ifdef _OPENMP
    int prior{0};

    VendorWidthFence() {
        if (moldable_width_scope() && threads_with_openmp() && omp_get_max_threads() > 1) {
            prior = omp_get_max_threads();
            omp_set_num_threads(1);
        }
    }

    ~VendorWidthFence() {
        if (prior > 1) {
            omp_set_num_threads(prior);
        }
    }

    VendorWidthFence(VendorWidthFence const &)            = delete;
    VendorWidthFence &operator=(VendorWidthFence const &) = delete;
#endif
};
} // namespace

void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
           float beta, float *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
           double beta, double *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda, float const **b_array,
                 int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count) {
    vendor::sgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
void dgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                 double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count) {
    vendor::dgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
void cgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const **a_array,
                 int_t lda, std::complex<float> const **b_array, int_t ldb, std::complex<float> beta, std::complex<float> **c_array,
                 int_t ldc, int_t batch_count) {
    vendor::cgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
void zgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const **a_array,
                 int_t lda, std::complex<double> const **b_array, int_t ldb, std::complex<double> beta, std::complex<double> **c_array,
                 int_t ldc, int_t batch_count) {
    vendor::zgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

void sgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, float const *alpha_array, float const **a_array, int_t const *lda_array,
                         float const **b_array, int_t const *ldb_array, float const *beta_array, float **c_array, int_t const *ldc_array,
                         int_t group_count, int_t const *group_size) {
    vendor::sgemm_batch_grouped(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array,
                                beta_array, c_array, ldc_array, group_count, group_size);
}
void dgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, double const *alpha_array, double const **a_array, int_t const *lda_array,
                         double const **b_array, int_t const *ldb_array, double const *beta_array, double **c_array, int_t const *ldc_array,
                         int_t group_count, int_t const *group_size) {
    vendor::dgemm_batch_grouped(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array,
                                beta_array, c_array, ldc_array, group_count, group_size);
}
void cgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, std::complex<float> const *alpha_array, std::complex<float> const **a_array,
                         int_t const *lda_array, std::complex<float> const **b_array, int_t const *ldb_array,
                         std::complex<float> const *beta_array, std::complex<float> **c_array, int_t const *ldc_array, int_t group_count,
                         int_t const *group_size) {
    vendor::cgemm_batch_grouped(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array,
                                beta_array, c_array, ldc_array, group_count, group_size);
}
void zgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, std::complex<double> const *alpha_array, std::complex<double> const **a_array,
                         int_t const *lda_array, std::complex<double> const **b_array, int_t const *ldb_array,
                         std::complex<double> const *beta_array, std::complex<double> **c_array, int_t const *ldc_array, int_t group_count,
                         int_t const *group_size) {
    vendor::zgemm_batch_grouped(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array,
                                beta_array, c_array, ldc_array, group_count, group_size);
}

void sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta, float *y,
           int_t incy) {
    VendorWidthFence const fence;
    vendor::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta, double *y,
           int_t incy) {
    VendorWidthFence const fence;
    vendor::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda, std::complex<float> const *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    VendorWidthFence const fence;
    return vendor::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    VendorWidthFence const fence;
    return vendor::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

auto ssterf(int_t n, float *d, float *e) -> int_t {
    VendorWidthFence const fence;
    return vendor::ssterf(n, d, e);
}

auto dsterf(int_t n, double *d, double *e) -> int_t {
    VendorWidthFence const fence;
    return vendor::dsterf(n, d, e);
}

auto sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr, int_t ldvr)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr, int_t ldvr)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w, std::complex<float> *vl, int_t ldvl,
           std::complex<float> *vr, int_t ldvr) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w, std::complex<double> *vl,
           int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork, float *rwork)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t {
    VendorWidthFence const fence;
    return vendor::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

void sscal(int_t n, float alpha, float *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::sscal(n, alpha, vec, inc);
}

void dscal(int_t n, double alpha, double *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::dscal(n, alpha, vec, inc);
}

void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::cscal(n, alpha, vec, inc);
}

void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::zscal(n, alpha, vec, inc);
}

void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::csscal(n, alpha, vec, inc);
}

void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::zdscal(n, alpha, vec, inc);
}

void srscl(int_t n, float alpha, float *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::srscl(n, alpha, vec, inc);
}

void drscl(int_t n, double alpha, double *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::drscl(n, alpha, vec, inc);
}

void csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::csrscl(n, alpha, vec, inc);
}

void zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    VendorWidthFence const fence;
    vendor::zdrscl(n, alpha, vec, inc);
}

auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    VendorWidthFence const fence;
    return vendor::sdot(n, x, incx, y, incy);
}

auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    VendorWidthFence const fence;
    return vendor::ddot(n, x, incx, y, incy);
}

auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    VendorWidthFence const fence;
    return vendor::cdot(n, x, incx, y, incy);
}

auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    VendorWidthFence const fence;
    return vendor::zdot(n, x, incx, y, incy);
}

auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    VendorWidthFence const fence;
    return vendor::cdotc(n, x, incx, y, incy);
}

auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    VendorWidthFence const fence;
    return vendor::zdotc(n, x, incx, y, incy);
}

void saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void saxpby(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void daxpby(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void caxpby(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> b, std::complex<float> *y,
            int_t inc_y) {
    VendorWidthFence const fence;
    vendor::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void zaxpby(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> b,
            std::complex<double> *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y, int_t inc_y,
          std::complex<float> *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
          int_t inc_y, std::complex<double> *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y,
           int_t inc_y, std::complex<float> *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::cgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
           int_t inc_y, std::complex<double> *a, int_t lda) {
    VendorWidthFence const fence;
    vendor::zgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

auto sgetrf(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgetrf(m, n, a, lda, ipiv);
}

auto dgetrf(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgetrf(m, n, a, lda, ipiv);
}

auto cgetrf(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgetrf(m, n, a, lda, ipiv);
}

auto zgetrf(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgetrf(m, n, a, lda, ipiv);
}

auto sgetri(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgetri(n, a, lda, ipiv);
}

auto dgetri(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgetri(n, a, lda, ipiv);
}

auto cgetri(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgetri(n, a, lda, ipiv);
}

auto zgetri(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgetri(n, a, lda, ipiv);
}

auto slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float {
    VendorWidthFence const fence;
    return vendor::slange(norm_type, m, n, A, lda, work);
}

auto dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double {
    VendorWidthFence const fence;
    return vendor::dlange(norm_type, m, n, A, lda, work);
}

auto clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float {
    VendorWidthFence const fence;
    return vendor::clange(norm_type, m, n, A, lda, work);
}

auto zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double {
    VendorWidthFence const fence;
    return vendor::zlange(norm_type, m, n, A, lda, work);
}

void slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    VendorWidthFence const fence;
    return vendor::slassq(n, x, incx, scale, sumsq);
}

void dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    VendorWidthFence const fence;
    return vendor::dlassq(n, x, incx, scale, sumsq);
}

void classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    VendorWidthFence const fence;
    return vendor::classq(n, x, incx, scale, sumsq);
}

void zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    VendorWidthFence const fence;
    return vendor::zlassq(n, x, incx, scale, sumsq);
}

float snrm2(int_t n, float const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::snrm2(n, x, incx);
}

double dnrm2(int_t n, double const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::dnrm2(n, x, incx);
}

float scnrm2(int_t n, std::complex<float> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::scnrm2(n, x, incx);
}

double dznrm2(int_t n, std::complex<double> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::dznrm2(n, x, incx);
}

auto sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto sgesvd(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
            float *superb) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt,
            double *superb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto cgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt, std::complex<float> *superb) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto zgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt, std::complex<double> *superb) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto cgees(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w, std::complex<float> *vs, int_t ldvs)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::cgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

auto zgees(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w, std::complex<double> *vs,
           int_t ldvs) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

auto strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb, float *c, int_t ldc,
            float *scale) -> int_t {
    VendorWidthFence const fence;
    return vendor::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb, double *c,
            int_t ldc, double *scale) -> int_t {
    VendorWidthFence const fence;
    return vendor::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> const *b,
            int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
    VendorWidthFence const fence;
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    (void)trana;
    (void)tranb;
    (void)isgn;
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)c;
    (void)ldc;
    (void)scale;
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "ctrsyl not implemented.");
#endif
}

auto ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> const *b,
            int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t {
    VendorWidthFence const fence;
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    (void)trana;
    (void)tranb;
    (void)isgn;
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)c;
    (void)ldc;
    (void)scale;
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "ztrsyl not implemented.");
#endif
}

auto sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgeqrf(m, n, a, lda, tau);
}

auto dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgeqrf(m, n, a, lda, tau);
}

auto cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgeqrf(m, n, a, lda, tau);
}

auto zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgeqrf(m, n, a, lda, tau);
}

auto sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::sorgqr(m, n, k, a, lda, tau);
}

auto dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::dorgqr(m, n, k, a, lda, tau);
}

auto cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::cungqr(m, n, k, a, lda, tau);
}

auto zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::zungqr(m, n, k, a, lda, tau);
}

auto sormqr(char side, char trans, int_t m, int_t n, int_t k, float const *a, int_t lda, float const *tau, float *c, int_t ldc) -> int_t {
    VendorWidthFence const fence;
    return vendor::sormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

auto dormqr(char side, char trans, int_t m, int_t n, int_t k, double const *a, int_t lda, double const *tau, double *c, int_t ldc)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::dormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

auto cunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<float> const *a, int_t lda, std::complex<float> const *tau,
            std::complex<float> *c, int_t ldc) -> int_t {
    VendorWidthFence const fence;
    return vendor::cunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

auto zunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<double> const *a, int_t lda, std::complex<double> const *tau,
            std::complex<double> *c, int_t ldc) -> int_t {
    VendorWidthFence const fence;
    return vendor::zunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

auto sgelqf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgelqf(m, n, a, lda, tau);
}

auto dgelqf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgelqf(m, n, a, lda, tau);
}

auto cgelqf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgelqf(m, n, a, lda, tau);
}

auto zgelqf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgelqf(m, n, a, lda, tau);
}

auto sorglq(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::sorglq(m, n, k, a, lda, tau);
}

auto dorglq(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::dorglq(m, n, k, a, lda, tau);
}

auto cunglq(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::cunglq(m, n, k, a, lda, tau);
}

auto zunglq(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t {
    VendorWidthFence const fence;
    return vendor::zunglq(m, n, k, a, lda, tau);
}

void scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::scopy(n, x, inc_x, y, inc_y);
}

void dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::dcopy(n, x, inc_x, y, inc_y);
}

void ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::ccopy(n, x, inc_x, y, inc_y);
}

void zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    VendorWidthFence const fence;
    vendor::zcopy(n, x, inc_x, y, inc_y);
}

int_t slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda) {
    VendorWidthFence const fence;
    return vendor::slascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}
int_t dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda) {
    VendorWidthFence const fence;
    return vendor::dlascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    VendorWidthFence const fence;
    vendor::sdirprod(n, alpha, x, incx, y, incy, z, incz);
}

void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    VendorWidthFence const fence;
    vendor::ddirprod(n, alpha, x, incx, y, incy, z, incz);
}
void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> *z, int_t incz) {
    VendorWidthFence const fence;
    vendor::cdirprod(n, alpha, x, incx, y, incy, z, incz);
}
void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> *z, int_t incz) {
    VendorWidthFence const fence;
    vendor::zdirprod(n, alpha, x, incx, y, incy, z, incz);
}

float sasum(int_t n, float const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::sasum(n, x, incx);
}

double dasum(int_t n, double const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::dasum(n, x, incx);
}

float scasum(int_t n, std::complex<float> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::scasum(n, x, incx);
}

double dzasum(int_t n, std::complex<double> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::dzasum(n, x, incx);
}

float scsum1(int_t n, std::complex<float> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::scsum1(n, x, incx);
}

double dzsum1(int_t n, std::complex<double> const *x, int_t incx) {
    VendorWidthFence const fence;
    return vendor::dzsum1(n, x, incx);
}

void clacgv(int_t n, std::complex<float> *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::clacgv(n, x, incx);
}

void zlacgv(int_t n, std::complex<double> *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::zlacgv(n, x, incx);
}

// --- trsm ---
void strsm(char side, char uplo, char transa, char diag, int_t m, int_t n, float alpha, float const *a, int_t lda, float *b, int_t ldb) {
    VendorWidthFence const fence;
    vendor::strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void dtrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, double alpha, double const *a, int_t lda, double *b, int_t ldb) {
    VendorWidthFence const fence;
    vendor::dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ctrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a,
           int_t lda, std::complex<float> *b, int_t ldb) {
    VendorWidthFence const fence;
    vendor::ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ztrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a,
           int_t lda, std::complex<double> *b, int_t ldb) {
    VendorWidthFence const fence;
    vendor::ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

// --- potrf ---
auto spotrf(char uplo, int_t n, float *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::spotrf(uplo, n, a, lda);
}

auto dpotrf(char uplo, int_t n, double *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::dpotrf(uplo, n, a, lda);
}

auto cpotrf(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::cpotrf(uplo, n, a, lda);
}

auto zpotrf(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::zpotrf(uplo, n, a, lda);
}

// --- potrs ---
auto spotrs(char uplo, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::spotrs(uplo, n, nrhs, a, lda, b, ldb);
}

auto dpotrs(char uplo, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

auto cpotrs(char uplo, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::cpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

auto zpotrs(char uplo, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::zpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

// --- potri ---
auto spotri(char uplo, int_t n, float *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::spotri(uplo, n, a, lda);
}

auto dpotri(char uplo, int_t n, double *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::dpotri(uplo, n, a, lda);
}

auto cpotri(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::cpotri(uplo, n, a, lda);
}

auto zpotri(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::zpotri(uplo, n, a, lda);
}

// --- syrk ---
void ssyrk(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float beta, float *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void dsyrk(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double beta, double *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void csyrk(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zsyrk(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

// --- herk ---
void cherk(char uplo, char trans, int_t n, int_t k, float alpha, std::complex<float> const *a, int_t lda, float beta,
           std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zherk(char uplo, char trans, int_t n, int_t k, double alpha, std::complex<double> const *a, int_t lda, double beta,
           std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

// --- symm ---
void ssymm(char side, char uplo, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *b, int_t ldb, float beta, float *c,
           int_t ldc) {
    VendorWidthFence const fence;
    vendor::ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsymm(char side, char uplo, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *b, int_t ldb, double beta,
           double *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csymm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::csymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsymm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

// --- hemm ---
void chemm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zhemm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

// --- sygv ---
auto ssygv(int_t itype, char jobz, char uplo, int_t n, float *a, int_t lda, float *b, int_t ldb, float *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::ssygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

auto dsygv(int_t itype, char jobz, char uplo, int_t n, double *a, int_t lda, double *b, int_t ldb, double *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

// --- hegv ---
auto chegv(int_t itype, char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb, float *w)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::chegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

auto zhegv(int_t itype, char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb, double *w)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::zhegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

// --- syevd ---
auto ssyevd(char jobz, char uplo, int_t n, float *a, int_t lda, float *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::ssyevd(jobz, uplo, n, a, lda, w);
}

auto dsyevd(char jobz, char uplo, int_t n, double *a, int_t lda, double *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::dsyevd(jobz, uplo, n, a, lda, w);
}

// --- heevd ---
auto cheevd(char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::cheevd(jobz, uplo, n, a, lda, w);
}

auto zheevd(char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w) -> int_t {
    VendorWidthFence const fence;
    return vendor::zheevd(jobz, uplo, n, a, lda, w);
}

// --- getrs ---
auto sgetrs(char trans, int_t n, int_t nrhs, float const *a, int_t lda, int_t const *ipiv, float *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgetrs(char trans, int_t n, int_t nrhs, double const *a, int_t lda, int_t const *ipiv, double *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgetrs(char trans, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, int_t const *ipiv, std::complex<float> *b, int_t ldb)
    -> int_t {
    VendorWidthFence const fence;
    return vendor::cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgetrs(char trans, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, int_t const *ipiv, std::complex<double> *b,
            int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

// --- gels ---
auto sgels(char trans, int_t m, int_t n, int_t nrhs, float *a, int_t lda, float *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::sgels(trans, m, n, nrhs, a, lda, b, ldb);
}

auto dgels(char trans, int_t m, int_t n, int_t nrhs, double *a, int_t lda, double *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dgels(trans, m, n, nrhs, a, lda, b, ldb);
}

auto cgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::cgels(trans, m, n, nrhs, a, lda, b, ldb);
}

auto zgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::zgels(trans, m, n, nrhs, a, lda, b, ldb);
}

// --- swap ---
void sswap(int_t n, float *x, int_t incx, float *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::sswap(n, x, incx, y, incy);
}

void dswap(int_t n, double *x, int_t incx, double *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::dswap(n, x, incx, y, incy);
}

void cswap(int_t n, std::complex<float> *x, int_t incx, std::complex<float> *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::cswap(n, x, incx, y, incy);
}

void zswap(int_t n, std::complex<double> *x, int_t incx, std::complex<double> *y, int_t incy) {
    VendorWidthFence const fence;
    vendor::zswap(n, x, incx, y, incy);
}

// --- iamax ---
auto isamax(int_t n, float const *x, int_t incx) -> int_t {
    VendorWidthFence const fence;
    return vendor::isamax(n, x, incx);
}

auto idamax(int_t n, double const *x, int_t incx) -> int_t {
    VendorWidthFence const fence;
    return vendor::idamax(n, x, incx);
}

auto icamax(int_t n, std::complex<float> const *x, int_t incx) -> int_t {
    VendorWidthFence const fence;
    return vendor::icamax(n, x, incx);
}

auto izamax(int_t n, std::complex<double> const *x, int_t incx) -> int_t {
    VendorWidthFence const fence;
    return vendor::izamax(n, x, incx);
}

// --- trsv ---
void strsv(char uplo, char trans, char diag, int_t n, float const *a, int_t lda, float *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::strsv(uplo, trans, diag, n, a, lda, x, incx);
}

void dtrsv(char uplo, char trans, char diag, int_t n, double const *a, int_t lda, double *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::dtrsv(uplo, trans, diag, n, a, lda, x, incx);
}

void ctrsv(char uplo, char trans, char diag, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::ctrsv(uplo, trans, diag, n, a, lda, x, incx);
}

void ztrsv(char uplo, char trans, char diag, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> *x, int_t incx) {
    VendorWidthFence const fence;
    vendor::ztrsv(uplo, trans, diag, n, a, lda, x, incx);
}

// --- syr2k ---
void ssyr2k(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb, float beta,
            float *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsyr2k(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb, double beta,
            double *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csyr2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
            std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsyr2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
            std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// --- her2k ---
void cher2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
            std::complex<float> const *b, int_t ldb, float beta, std::complex<float> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zher2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
            std::complex<double> const *b, int_t ldb, double beta, std::complex<double> *c, int_t ldc) {
    VendorWidthFence const fence;
    vendor::zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// --- trtrs ---
auto strtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

auto dtrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

auto ctrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b,
            int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

auto ztrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b,
            int_t ldb) -> int_t {
    VendorWidthFence const fence;
    return vendor::ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

// --- trtri ---
auto strtri(char uplo, char diag, int_t n, float *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::strtri(uplo, diag, n, a, lda);
}

auto dtrtri(char uplo, char diag, int_t n, double *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::dtrtri(uplo, diag, n, a, lda);
}

auto ctrtri(char uplo, char diag, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::ctrtri(uplo, diag, n, a, lda);
}

auto ztrtri(char uplo, char diag, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    VendorWidthFence const fence;
    return vendor::ztrtri(uplo, diag, n, a, lda);
}

EINSUMS_NAMESPACE_END(blas::detail)
