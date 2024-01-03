//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "mkl.hpp"

#include "einsums/Section.hpp"
#include "einsums/_Common.hpp"

#include <fmt/format.h>
#include <mkl_blas.h>
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include <stdexcept>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::mkl)

namespace {
constexpr auto mkl_interface() {
    return EINSUMS_STRINGIFY(MKL_INTERFACE);
}

auto transpose_to_cblas(char transpose) -> CBLAS_TRANSPOSE {
    switch (transpose) {
    case 'N':
    case 'n':
        return CblasNoTrans;
    case 'T':
    case 't':
        return CblasTrans;
    case 'C':
    case 'c':
        return CblasConjTrans;
    }
    println_warn("Unknown transpose code {}, defaulting to CblasNoTrans.", transpose);
    return CblasNoTrans;
}

} // namespace

void initialize() {
}
void finalize() {
}

void sgemm(const char transa, const char transb, eint m, eint n, eint k, float alpha, const float *a, eint lda, const float *b, eint ldb,
           float beta, float *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::sgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, eint m, eint n, eint k, double alpha, const double *a, eint lda, const double *b, eint ldb,
           double beta, double *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, eint m, eint n, eint k, const std::complex<float> alpha, const std::complex<float> *a, eint lda,
           const std::complex<float> *b, eint ldb, const std::complex<float> beta, std::complex<float> *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::cgemm(&transb, &transa, &n, &m, &k, reinterpret_cast<const MKL_Complex8 *>(&alpha), reinterpret_cast<const MKL_Complex8 *>(b), &ldb,
            reinterpret_cast<const MKL_Complex8 *>(a), &lda, reinterpret_cast<const MKL_Complex8 *>(&beta),
            reinterpret_cast<MKL_Complex8 *>(c), &ldc);
}

void zgemm(char transa, char transb, eint m, eint n, eint k, const std::complex<double> alpha, const std::complex<double> *a, eint lda,
           const std::complex<double> *b, eint ldb, const std::complex<double> beta, std::complex<double> *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::zgemm(&transb, &transa, &n, &m, &k, reinterpret_cast<const MKL_Complex16 *>(&alpha), reinterpret_cast<const MKL_Complex16 *>(b), &ldb,
            reinterpret_cast<const MKL_Complex16 *>(a), &lda, reinterpret_cast<const MKL_Complex16 *>(&beta),
            reinterpret_cast<MKL_Complex16 *>(c), &ldc);
}

#define impl_gemm_batch_strided(x, type)                                                                                                   \
    mkl_def_gemm_batch_strided(x, type) {                                                                                                  \
        LabeledSection1(mkl_interface());                                                                                                  \
        if (m == 0 || n == 0 || k == 0)                                                                                                    \
            return;                                                                                                                        \
        cblas_##x##gemm_batch_strided(CblasRowMajor, transpose_to_cblas(transa), transpose_to_cblas(transb), m, n, k, alpha, a, lda,       \
                                      stridea, b, ldb, strideb, beta, c, ldc, stridec, batch_size);                                        \
    }

impl_gemm_batch_strided(s, float);
impl_gemm_batch_strided(d, double);

#define impl_gemm_batch_strided_complex(x, type)                                                                                           \
    mkl_def_gemm_batch_strided(x, type) {                                                                                                  \
        LabeledSection1(mkl_interface());                                                                                                  \
        if (m == 0 || n == 0 || k == 0)                                                                                                    \
            return;                                                                                                                        \
        cblas_##x##gemm_batch_strided(CblasRowMajor, transpose_to_cblas(transa), transpose_to_cblas(transb), m, n, k, &alpha, a, lda,      \
                                      stridea, b, ldb, strideb, &beta, c, ldc, stridec, batch_size);                                       \
    }

impl_gemm_batch_strided_complex(c, std::complex<float>);
impl_gemm_batch_strided_complex(z, std::complex<double>);

void sgemv(const char transa, const eint m, const eint n, const float alpha, const float *a, const eint lda, const float *x,
           const eint incx, const float beta, float *y, const eint incy) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0)
        return;
    char ta = 'N';
    if (transa == 'N' || transa == 'n')
        ta = 'T';
    else if (transa == 'T' || transa == 't')
        ta = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    ::sgemv(&ta, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void dgemv(const char transa, const eint m, const eint n, const double alpha, const double *a, const eint lda, const double *x,
           const eint incx, double beta, double *y, const eint incy) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0)
        return;

    char ta = 'N';
    if (transa == 'N' || transa == 'n')
        ta = 'T';
    else if (transa == 'T' || transa == 't')
        ta = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    ::dgemv(&ta, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void cgemv(const char transa, const eint m, const eint n, const std::complex<float> alpha, const std::complex<float> *a, const eint lda,
           const std::complex<float> *x, const eint incx, const std::complex<float> beta, std::complex<float> *y, const eint incy) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0)
        return;
    char ta = 'N';
    if (transa == 'N' || transa == 'n')
        ta = 'T';
    else if (transa == 'T' || transa == 't')
        ta = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    ::cgemv(&ta, &n, &m, reinterpret_cast<const MKL_Complex8 *>(&alpha), reinterpret_cast<const MKL_Complex8 *>(a), &lda,
            reinterpret_cast<const MKL_Complex8 *>(x), &incx, reinterpret_cast<const MKL_Complex8 *>(&beta),
            reinterpret_cast<MKL_Complex8 *>(y), &incy);
}

void zgemv(const char transa, const eint m, const eint n, const std::complex<double> alpha, const std::complex<double> *a, const eint lda,
           const std::complex<double> *x, const eint incx, const std::complex<double> beta, std::complex<double> *y, const eint incy) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0)
        return;
    char ta = 'N';
    if (transa == 'N' || transa == 'n')
        ta = 'T';
    else if (transa == 'T' || transa == 't')
        ta = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    ::zgemv(&ta, &n, &m, reinterpret_cast<const MKL_Complex16 *>(&alpha), reinterpret_cast<const MKL_Complex16 *>(a), &lda,
            reinterpret_cast<const MKL_Complex16 *>(x), &incx, reinterpret_cast<const MKL_Complex16 *>(&beta),
            reinterpret_cast<MKL_Complex16 *>(y), &incy);
}

auto ssyev(const char job, const char uplo, const eint n, float *a, const eint lda, float *w, float *work, const eint lwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::ssyev(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(const char job, const char uplo, const eint n, double *a, const eint lda, double *w, double *work, const eint lwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::dsyev(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto cheev(const char job, const char uplo, const eint n, std::complex<float> *a, const eint lda, float *w, std::complex<float> *work,
           const eint lwork, float *rwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::cheev(&job, &uplo, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, w, reinterpret_cast<MKL_Complex8 *>(work), &lwork, rwork, &info);
    return info;
}

auto zheev(const char job, const char uplo, const eint n, std::complex<double> *a, const eint lda, double *w, std::complex<double> *work,
           const eint lwork, double *rwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::zheev(&job, &uplo, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, w, reinterpret_cast<MKL_Complex16 *>(work), &lwork, rwork, &info);
    return info;
}

auto sgesv(const eint n, const eint nrhs, float *a, const eint lda, eint *ipiv, float *b, const eint ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::sgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(const eint n, const eint nrhs, double *a, const eint lda, eint *ipiv, double *b, const eint ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::dgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(const eint n, const eint nrhs, std::complex<float> *a, const eint lda, eint *ipiv, std::complex<float> *b, const eint ldb)
    -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::cgesv(&n, &nrhs, reinterpret_cast<MKL_Complex8 *>(a), &lda, ipiv, reinterpret_cast<MKL_Complex8 *>(b), &ldb, &info);
    return info;
}

auto zgesv(const eint n, const eint nrhs, std::complex<double> *a, const eint lda, eint *ipiv, std::complex<double> *b, const eint ldb)
    -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::zgesv(&n, &nrhs, reinterpret_cast<MKL_Complex16 *>(a), &lda, ipiv, reinterpret_cast<MKL_Complex16 *>(b), &ldb, &info);
    return info;
}

void sscal(const eint n, const float alpha, float *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::sscal(&n, &alpha, vec, &inc);
}

void dscal(const eint n, const double alpha, double *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::dscal(&n, &alpha, vec, &inc);
}

void cscal(const eint n, const std::complex<float> alpha, std::complex<float> *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::cscal(&n, reinterpret_cast<const MKL_Complex8 *>(&alpha), reinterpret_cast<MKL_Complex8 *>(vec), &inc);
}

void zscal(const eint n, const std::complex<double> alpha, std::complex<double> *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::zscal(&n, reinterpret_cast<const MKL_Complex16 *>(&alpha), reinterpret_cast<MKL_Complex16 *>(vec), &inc);
}

void csscal(const eint n, const float alpha, std::complex<float> *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::csscal(&n, &alpha, reinterpret_cast<MKL_Complex8 *>(vec), &inc);
}

void zdscal(const eint n, const double alpha, std::complex<double> *vec, const eint inc) {
    LabeledSection1(mkl_interface());

    ::zdscal(&n, &alpha, reinterpret_cast<MKL_Complex16 *>(vec), &inc);
}

auto sdot(const eint n, const float *x, const eint incx, const float *y, const eint incy) -> float {
    LabeledSection1(mkl_interface());

    return ::sdot(&n, x, &incx, y, &incy);
}

auto ddot(const eint n, const double *x, const eint incx, const double *y, const eint incy) -> double {
    LabeledSection1(mkl_interface());

    return ::ddot(&n, x, &incx, y, &incy);
}

auto cdot(const eint n, const std::complex<float> *x, const eint incx, const std::complex<float> *y, const eint incy)
    -> std::complex<float> {
    LabeledSection1(mkl_interface());

    std::complex<float> pres{0., 0.};
    ::cdotu(reinterpret_cast<MKL_Complex8 *>(&pres), &n, reinterpret_cast<const MKL_Complex8 *>(x), &incx,
            reinterpret_cast<const MKL_Complex8 *>(y), &incy);
    return pres;
}

auto zdot(const eint n, const std::complex<double> *x, const eint incx, const std::complex<double> *y, const eint incy)
    -> std::complex<double> {
    LabeledSection1(mkl_interface());

    std::complex<double> pres{0., 0.};
    ::zdotu(reinterpret_cast<MKL_Complex16 *>(&pres), &n, reinterpret_cast<const MKL_Complex16 *>(x), &incx,
            reinterpret_cast<const MKL_Complex16 *>(y), &incy);
    return pres;
}

void saxpy(const eint n, const float alpha_x, const float *x, const eint inc_x, float *y, const eint inc_y) {
    LabeledSection1(mkl_interface());
    ::saxpy(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(const eint n, const double alpha_x, const double *x, const eint inc_x, double *y, const eint inc_y) {
    LabeledSection1(mkl_interface());
    ::daxpy(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(const eint n, const std::complex<float> alpha_x, const std::complex<float> *x, const eint inc_x, std::complex<float> *y,
           const eint inc_y) {
    LabeledSection1(mkl_interface());
    ::caxpy(&n, reinterpret_cast<const MKL_Complex8 *>(&alpha_x), reinterpret_cast<const MKL_Complex8 *>(x), &inc_x,
            reinterpret_cast<MKL_Complex8 *>(y), &inc_y);
}

void zaxpy(const eint n, const std::complex<double> alpha_x, const std::complex<double> *x, const eint inc_x, std::complex<double> *y,
           const eint inc_y) {
    LabeledSection1(mkl_interface());
    ::zaxpy(&n, reinterpret_cast<const MKL_Complex16 *>(&alpha_x), reinterpret_cast<const MKL_Complex16 *>(x), &inc_x,
            reinterpret_cast<MKL_Complex16 *>(y), &inc_y);
}

void saxpby(const eint n, const float a, const float *x, const eint incx, const float b, float *y, const eint incy) {
    LabeledSection1(mkl_interface());
    ::saxpby(&n, &a, x, &incx, &b, y, &incy);
}

void daxpby(const eint n, const double a, const double *x, const eint incx, const double b, double *y, const eint incy) {
    LabeledSection1(mkl_interface());
    ::daxpby(&n, &a, x, &incx, &b, y, &incy);
}

void caxpby(const eint n, const std::complex<float> a, const std::complex<float> *x, const eint incx, const std::complex<float> b,
            std::complex<float> *y, const eint incy) {
    LabeledSection1(mkl_interface());
    ::caxpby(&n, reinterpret_cast<const MKL_Complex8 *>(&a), reinterpret_cast<const MKL_Complex8 *>(x), &incx,
             reinterpret_cast<const MKL_Complex8 *>(&b), reinterpret_cast<MKL_Complex8 *>(y), &incy);
}

void zaxpby(const eint n, const std::complex<double> a, const std::complex<double> *x, const eint incx, const std::complex<double> b,
            std::complex<double> *y, const eint incy) {
    LabeledSection1(mkl_interface());
    ::zaxpby(&n, reinterpret_cast<const MKL_Complex16 *>(&a), reinterpret_cast<const MKL_Complex16 *>(x), &incx,
             reinterpret_cast<const MKL_Complex16 *>(&b), reinterpret_cast<MKL_Complex16 *>(y), &incy);
}

namespace {
void ger_parameter_check(eint m, eint n, eint inc_x, eint inc_y, eint lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::mkl::ger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::mkl::ger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::mkl::ger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::mkl::ger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(eint(1), n)) {
        throw std::runtime_error(fmt::format("einsums::backend::mkl::ger: lda ({}) is less than max(1, n ({})).", lda, n));
    }
}
} // namespace

void sger(const eint m, const eint n, const float alpha, const float *x, const eint inc_x, const float *y, const eint inc_y, float *a,
          const eint lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::sger(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void dger(const eint m, const eint n, const double alpha, const double *x, const eint inc_x, const double *y, const eint inc_y, double *a,
          const eint lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::dger(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void cger(const eint m, const eint n, const std::complex<float> alpha, const std::complex<float> *x, const eint inc_x,
          const std::complex<float> *y, const eint inc_y, std::complex<float> *a, const eint lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::cgeru(&n, &m, reinterpret_cast<const MKL_Complex8 *>(&alpha), reinterpret_cast<const MKL_Complex8 *>(y), &inc_y,
            reinterpret_cast<const MKL_Complex8 *>(x), &inc_x, reinterpret_cast<MKL_Complex8 *>(a), &lda);
}

void zger(const eint m, const eint n, const std::complex<double> alpha, const std::complex<double> *x, const eint inc_x,
          const std::complex<double> *y, const eint inc_y, std::complex<double> *a, const eint lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::zgeru(&n, &m, reinterpret_cast<const MKL_Complex16 *>(&alpha), reinterpret_cast<const MKL_Complex16 *>(y), &inc_y,
            reinterpret_cast<const MKL_Complex16 *>(x), &inc_x, reinterpret_cast<MKL_Complex16 *>(a), &lda);
}

auto sgetrf(const eint m, const eint n, float *a, const eint lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::sgetrf(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(const eint m, const eint n, double *a, const eint lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::dgetrf(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(const eint m, const eint n, std::complex<float> *a, const eint lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::cgetrf(&m, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, ipiv, &info);
    return info;
}

auto zgetrf(const eint m, const eint n, std::complex<double> *a, const eint lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::zgetrf(&m, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, ipiv, &info);
    return info;
}

auto sgetri(const eint n, float *a, const eint lda, const eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint               info{0};
    eint               lwork = n * 64;
    std::vector<float> work(lwork);
    ::sgetri(&n, a, &lda, (eint *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(const eint n, double *a, const eint lda, const eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                info{0};
    eint                lwork = n * 64;
    std::vector<double> work(lwork);
    ::dgetri(&n, a, &lda, (eint *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(const eint n, std::complex<float> *a, const eint lda, const eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                             info{0};
    eint                             lwork = n * 64;
    std::vector<std::complex<float>> work(lwork);
    ::cgetri(&n, reinterpret_cast<MKL_Complex8 *>(a), &lda, (eint *)ipiv, reinterpret_cast<MKL_Complex8 *>(work.data()), &lwork, &info);
    return info;
}

auto zgetri(const eint n, std::complex<double> *a, const eint lda, const eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                              info{0};
    eint                              lwork = n * 64;
    std::vector<std::complex<double>> work(lwork);
    ::zgetri(&n, reinterpret_cast<MKL_Complex16 *>(a), &lda, (eint *)ipiv, reinterpret_cast<MKL_Complex16 *>(work.data()), &lwork, &info);
    return info;
}

auto slange(const char norm_type, const eint m, const eint n, const float *A, const eint lda, float *work) -> float {
    LabeledSection1(mkl_interface());

    return ::slange(&norm_type, &m, &n, A, &lda, work);
}

auto dlange(const char norm_type, const eint m, const eint n, const double *A, const eint lda, double *work) -> double {
    LabeledSection1(mkl_interface());

    return ::dlange(&norm_type, &m, &n, A, &lda, work);
}

auto clange(const char norm_type, const eint m, const eint n, const std::complex<float> *A, const eint lda, float *work) -> float {
    LabeledSection1(mkl_interface());

    return ::clange(&norm_type, &m, &n, reinterpret_cast<const MKL_Complex8 *>(A), &lda, work);
}

auto zlange(const char norm_type, const eint m, const eint n, const std::complex<double> *A, const eint lda, double *work) -> double {
    LabeledSection1(mkl_interface());

    return ::zlange(&norm_type, &m, &n, reinterpret_cast<const MKL_Complex16 *>(A), &lda, work);
}

void slassq(const eint n, const float *x, const eint incx, float *scale, float *sumsq) {
    LabeledSection1(mkl_interface());
    ::slassq(&n, x, &incx, scale, sumsq);
}

void dlassq(const eint n, const double *x, const eint incx, double *scale, double *sumsq) {
    LabeledSection1(mkl_interface());
    ::dlassq(&n, x, &incx, scale, sumsq);
}

void classq(const eint n, const std::complex<float> *x, const eint incx, float *scale, float *sumsq) {
    LabeledSection1(mkl_interface());
    ::classq(&n, reinterpret_cast<const MKL_Complex8 *>(x), &incx, scale, sumsq);
}

void zlassq(const eint n, const std::complex<double> *x, const eint incx, double *scale, double *sumsq) {
    LabeledSection1(mkl_interface());
    ::zlassq(&n, reinterpret_cast<const MKL_Complex16 *>(x), &incx, scale, sumsq);
}

auto sgesdd(char jobz, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, eint m, eint n, std::complex<float> *a, eint lda, float *s, std::complex<float> *u, eint ldu,
            std::complex<float> *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_cgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_float *>(a), lda, s,
                          reinterpret_cast<lapack_complex_float *>(u), ldu, reinterpret_cast<lapack_complex_float *>(vt), ldvt);
}

auto zgesdd(char jobz, eint m, eint n, std::complex<double> *a, eint lda, double *s, std::complex<double> *u, eint ldu,
            std::complex<double> *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_zgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_double *>(a), lda, s,
                          reinterpret_cast<lapack_complex_double *>(u), ldu, reinterpret_cast<lapack_complex_double *>(vt), ldvt);
}

auto sgesvd(char jobu, char jobvt, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt, eint ldvt, float *superb)
    -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgesvd(char jobu, char jobvt, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt, eint ldvt,
            double *superb) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto sgees(char jobvs, eint n, float *a, eint lda, eint *sdim, float *wr, float *wi, float *vs, eint ldvs) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgees(LAPACK_ROW_MAJOR, jobvs, 'N', nullptr, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto dgees(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs, eint ldvs) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dgees(LAPACK_ROW_MAJOR, jobvs, 'N', nullptr, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto strsyl(char trana, char tranb, eint isgn, eint m, eint n, const float *a, eint lda, const float *b, eint ldb, float *c, eint ldc,
            float *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_strsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, const double *a, eint lda, const double *b, eint ldb, double *c, eint ldc,
            double *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, const std::complex<float> *a, eint lda, const std::complex<float> *b,
            eint ldb, std::complex<float> *c, eint ldc, float *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_ctrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, reinterpret_cast<const lapack_complex_float *>(a), lda,
                          reinterpret_cast<const lapack_complex_float *>(b), ldb, reinterpret_cast<lapack_complex_float *>(c), ldc, scale);
}

auto ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, const std::complex<double> *a, eint lda, const std::complex<double> *b,
            eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_ztrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, reinterpret_cast<const lapack_complex_double *>(a), lda,
                          reinterpret_cast<const lapack_complex_double *>(b), ldb, reinterpret_cast<lapack_complex_double *>(c), ldc,
                          scale);
}

auto sgeqrf(eint m, eint n, float *a, eint lda, float *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
}

auto dgeqrf(eint m, eint n, double *a, eint lda, double *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
}

auto cgeqrf(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float *>(tau));
}

auto zgeqrf(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double *>(tau));
}

auto sorgqr(eint m, eint n, eint k, float *a, eint lda, const float *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto dorgqr(eint m, eint n, eint k, double *a, eint lda, const double *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, const std::complex<float> *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<const lapack_complex_float *>(tau));
}

auto zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, const std::complex<double> *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<const lapack_complex_double *>(tau));
}

END_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::mkl)