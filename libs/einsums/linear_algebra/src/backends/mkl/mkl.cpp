//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "mkl.hpp"

#include <einsums/preprocessor/stringize.hpp>

#include "einsums/Print.hpp"
#include "einsums/Section.hpp"

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
    return EINSUMS_PP_STRINGIZE(MKL_INTERFACE);
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

void sgemm(char const transa, char const transb, eint m, eint n, eint k, float alpha, float const *a, eint lda,
           float const *b, eint ldb, float beta, float *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::sgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, eint m, eint n, eint k, double alpha, double const *a, eint lda, double const *b,
           eint ldb, double beta, double *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::dgemm(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, eint m, eint n, eint k, std::complex<float> const alpha,
           std::complex<float> const *a, eint lda, std::complex<float> const *b, eint ldb,
           std::complex<float> const beta, std::complex<float> *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::cgemm(&transb, &transa, &n, &m, &k, reinterpret_cast<MKL_Complex8 const *>(&alpha),
            reinterpret_cast<MKL_Complex8 const *>(b), &ldb, reinterpret_cast<MKL_Complex8 const *>(a), &lda,
            reinterpret_cast<MKL_Complex8 const *>(&beta), reinterpret_cast<MKL_Complex8 *>(c), &ldc);
}

void zgemm(char transa, char transb, eint m, eint n, eint k, std::complex<double> const alpha,
           std::complex<double> const *a, eint lda, std::complex<double> const *b, eint ldb,
           std::complex<double> const beta, std::complex<double> *c, eint ldc) {
    LabeledSection1(mkl_interface());

    if (m == 0 || n == 0 || k == 0)
        return;
    ::zgemm(&transb, &transa, &n, &m, &k, reinterpret_cast<MKL_Complex16 const *>(&alpha),
            reinterpret_cast<MKL_Complex16 const *>(b), &ldb, reinterpret_cast<MKL_Complex16 const *>(a), &lda,
            reinterpret_cast<MKL_Complex16 const *>(&beta), reinterpret_cast<MKL_Complex16 *>(c), &ldc);
}

#define impl_gemm_batch_strided(x, type)                                                                               \
    mkl_def_gemm_batch_strided(x, type) {                                                                              \
        LabeledSection1(mkl_interface());                                                                              \
        if (m == 0 || n == 0 || k == 0)                                                                                \
            return;                                                                                                    \
        cblas_##x##gemm_batch_strided(CblasRowMajor, transpose_to_cblas(transa), transpose_to_cblas(transb), m, n, k,  \
                                      alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec, batch_size);     \
    }

impl_gemm_batch_strided(s, float);
impl_gemm_batch_strided(d, double);

#define impl_gemm_batch_strided_complex(x, type)                                                                       \
    mkl_def_gemm_batch_strided(x, type) {                                                                              \
        LabeledSection1(mkl_interface());                                                                              \
        if (m == 0 || n == 0 || k == 0)                                                                                \
            return;                                                                                                    \
        cblas_##x##gemm_batch_strided(CblasRowMajor, transpose_to_cblas(transa), transpose_to_cblas(transb), m, n, k,  \
                                      &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, batch_size);   \
    }

impl_gemm_batch_strided_complex(c, std::complex<float>);
impl_gemm_batch_strided_complex(z, std::complex<double>);

void sgemv(char const transa, eint const m, eint const n, float const alpha, float const *a, eint const lda,
           float const *x, eint const incx, float const beta, float *y, eint const incy) {
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

void dgemv(char const transa, eint const m, eint const n, double const alpha, double const *a, eint const lda,
           double const *x, eint const incx, double beta, double *y, eint const incy) {
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

void cgemv(char const transa, eint const m, eint const n, std::complex<float> const alpha, std::complex<float> const *a,
           eint const lda, std::complex<float> const *x, eint const incx, std::complex<float> const beta,
           std::complex<float> *y, eint const incy) {
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

    ::cgemv(&ta, &n, &m, reinterpret_cast<MKL_Complex8 const *>(&alpha), reinterpret_cast<MKL_Complex8 const *>(a),
            &lda, reinterpret_cast<MKL_Complex8 const *>(x), &incx, reinterpret_cast<MKL_Complex8 const *>(&beta),
            reinterpret_cast<MKL_Complex8 *>(y), &incy);
}

void zgemv(char const transa, eint const m, eint const n, std::complex<double> const alpha,
           std::complex<double> const *a, eint const lda, std::complex<double> const *x, eint const incx,
           std::complex<double> const beta, std::complex<double> *y, eint const incy) {
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

    ::zgemv(&ta, &n, &m, reinterpret_cast<MKL_Complex16 const *>(&alpha), reinterpret_cast<MKL_Complex16 const *>(a),
            &lda, reinterpret_cast<MKL_Complex16 const *>(x), &incx, reinterpret_cast<MKL_Complex16 const *>(&beta),
            reinterpret_cast<MKL_Complex16 *>(y), &incy);
}

auto ssyev(char const job, char const uplo, eint const n, float *a, eint const lda, float *w, float *work,
           eint const lwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::ssyev(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char const job, char const uplo, eint const n, double *a, eint const lda, double *w, double *work,
           eint const lwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::dsyev(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto cheev(char const job, char const uplo, eint const n, std::complex<float> *a, eint const lda, float *w,
           std::complex<float> *work, eint const lwork, float *rwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::cheev(&job, &uplo, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, w, reinterpret_cast<MKL_Complex8 *>(work),
            &lwork, rwork, &info);
    return info;
}

auto zheev(char const job, char const uplo, eint const n, std::complex<double> *a, eint const lda, double *w,
           std::complex<double> *work, eint const lwork, double *rwork) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::zheev(&job, &uplo, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, w, reinterpret_cast<MKL_Complex16 *>(work),
            &lwork, rwork, &info);
    return info;
}

auto sgesv(eint const n, eint const nrhs, float *a, eint const lda, eint *ipiv, float *b, eint const ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::sgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(eint const n, eint const nrhs, double *a, eint const lda, eint *ipiv, double *b, eint const ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::dgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(eint const n, eint const nrhs, std::complex<float> *a, eint const lda, eint *ipiv, std::complex<float> *b,
           eint const ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::cgesv(&n, &nrhs, reinterpret_cast<MKL_Complex8 *>(a), &lda, ipiv, reinterpret_cast<MKL_Complex8 *>(b), &ldb,
            &info);
    return info;
}

auto zgesv(eint const n, eint const nrhs, std::complex<double> *a, eint const lda, eint *ipiv, std::complex<double> *b,
           eint const ldb) -> eint {
    LabeledSection1(mkl_interface());

    eint info{0};
    ::zgesv(&n, &nrhs, reinterpret_cast<MKL_Complex16 *>(a), &lda, ipiv, reinterpret_cast<MKL_Complex16 *>(b), &ldb,
            &info);
    return info;
}

void sscal(eint const n, float const alpha, float *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::sscal(&n, &alpha, vec, &inc);
}

void dscal(eint const n, double const alpha, double *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::dscal(&n, &alpha, vec, &inc);
}

void cscal(eint const n, std::complex<float> const alpha, std::complex<float> *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::cscal(&n, reinterpret_cast<MKL_Complex8 const *>(&alpha), reinterpret_cast<MKL_Complex8 *>(vec), &inc);
}

void zscal(eint const n, std::complex<double> const alpha, std::complex<double> *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::zscal(&n, reinterpret_cast<MKL_Complex16 const *>(&alpha), reinterpret_cast<MKL_Complex16 *>(vec), &inc);
}

void csscal(eint const n, float const alpha, std::complex<float> *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::csscal(&n, &alpha, reinterpret_cast<MKL_Complex8 *>(vec), &inc);
}

void zdscal(eint const n, double const alpha, std::complex<double> *vec, eint const inc) {
    LabeledSection1(mkl_interface());

    ::zdscal(&n, &alpha, reinterpret_cast<MKL_Complex16 *>(vec), &inc);
}

auto sdot(eint const n, float const *x, eint const incx, float const *y, eint const incy) -> float {
    LabeledSection1(mkl_interface());

    return ::sdot(&n, x, &incx, y, &incy);
}

auto ddot(eint const n, double const *x, eint const incx, double const *y, eint const incy) -> double {
    LabeledSection1(mkl_interface());

    return ::ddot(&n, x, &incx, y, &incy);
}

auto cdot(eint const n, std::complex<float> const *x, eint const incx, std::complex<float> const *y, eint const incy)
    -> std::complex<float> {
    LabeledSection1(mkl_interface());

    std::complex<float> pres{0., 0.};
    ::cdotu(reinterpret_cast<MKL_Complex8 *>(&pres), &n, reinterpret_cast<MKL_Complex8 const *>(x), &incx,
            reinterpret_cast<MKL_Complex8 const *>(y), &incy);
    return pres;
}

auto zdot(eint const n, std::complex<double> const *x, eint const incx, std::complex<double> const *y, eint const incy)
    -> std::complex<double> {
    LabeledSection1(mkl_interface());

    std::complex<double> pres{0., 0.};
    ::zdotu(reinterpret_cast<MKL_Complex16 *>(&pres), &n, reinterpret_cast<MKL_Complex16 const *>(x), &incx,
            reinterpret_cast<MKL_Complex16 const *>(y), &incy);
    return pres;
}

void saxpy(eint const n, float const alpha_x, float const *x, eint const inc_x, float *y, eint const inc_y) {
    LabeledSection1(mkl_interface());
    ::saxpy(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(eint const n, double const alpha_x, double const *x, eint const inc_x, double *y, eint const inc_y) {
    LabeledSection1(mkl_interface());
    ::daxpy(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(eint const n, std::complex<float> const alpha_x, std::complex<float> const *x, eint const inc_x,
           std::complex<float> *y, eint const inc_y) {
    LabeledSection1(mkl_interface());
    ::caxpy(&n, reinterpret_cast<MKL_Complex8 const *>(&alpha_x), reinterpret_cast<MKL_Complex8 const *>(x), &inc_x,
            reinterpret_cast<MKL_Complex8 *>(y), &inc_y);
}

void zaxpy(eint const n, std::complex<double> const alpha_x, std::complex<double> const *x, eint const inc_x,
           std::complex<double> *y, eint const inc_y) {
    LabeledSection1(mkl_interface());
    ::zaxpy(&n, reinterpret_cast<MKL_Complex16 const *>(&alpha_x), reinterpret_cast<MKL_Complex16 const *>(x), &inc_x,
            reinterpret_cast<MKL_Complex16 *>(y), &inc_y);
}

void saxpby(eint const n, float const a, float const *x, eint const incx, float const b, float *y, eint const incy) {
    LabeledSection1(mkl_interface());
    ::saxpby(&n, &a, x, &incx, &b, y, &incy);
}

void daxpby(eint const n, double const a, double const *x, eint const incx, double const b, double *y,
            eint const incy) {
    LabeledSection1(mkl_interface());
    ::daxpby(&n, &a, x, &incx, &b, y, &incy);
}

void caxpby(eint const n, std::complex<float> const a, std::complex<float> const *x, eint const incx,
            std::complex<float> const b, std::complex<float> *y, eint const incy) {
    LabeledSection1(mkl_interface());
    ::caxpby(&n, reinterpret_cast<MKL_Complex8 const *>(&a), reinterpret_cast<MKL_Complex8 const *>(x), &incx,
             reinterpret_cast<MKL_Complex8 const *>(&b), reinterpret_cast<MKL_Complex8 *>(y), &incy);
}

void zaxpby(eint const n, std::complex<double> const a, std::complex<double> const *x, eint const incx,
            std::complex<double> const b, std::complex<double> *y, eint const incy) {
    LabeledSection1(mkl_interface());
    ::zaxpby(&n, reinterpret_cast<MKL_Complex16 const *>(&a), reinterpret_cast<MKL_Complex16 const *>(x), &incx,
             reinterpret_cast<MKL_Complex16 const *>(&b), reinterpret_cast<MKL_Complex16 *>(y), &incy);
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
        throw std::runtime_error(
            fmt::format("einsums::backend::mkl::ger: lda ({}) is less than max(1, n ({})).", lda, n));
    }
}
} // namespace

void sger(eint const m, eint const n, float const alpha, float const *x, eint const inc_x, float const *y,
          eint const inc_y, float *a, eint const lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::sger(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void dger(eint const m, eint const n, double const alpha, double const *x, eint const inc_x, double const *y,
          eint const inc_y, double *a, eint const lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::dger(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void cger(eint const m, eint const n, std::complex<float> const alpha, std::complex<float> const *x, eint const inc_x,
          std::complex<float> const *y, eint const inc_y, std::complex<float> *a, eint const lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::cgeru(&n, &m, reinterpret_cast<MKL_Complex8 const *>(&alpha), reinterpret_cast<MKL_Complex8 const *>(y), &inc_y,
            reinterpret_cast<MKL_Complex8 const *>(x), &inc_x, reinterpret_cast<MKL_Complex8 *>(a), &lda);
}

void zger(eint const m, eint const n, std::complex<double> const alpha, std::complex<double> const *x, eint const inc_x,
          std::complex<double> const *y, eint const inc_y, std::complex<double> *a, eint const lda) {
    LabeledSection1(mkl_interface());
    ger_parameter_check(m, n, inc_x, inc_y, lda);
    ::zgeru(&n, &m, reinterpret_cast<MKL_Complex16 const *>(&alpha), reinterpret_cast<MKL_Complex16 const *>(y), &inc_y,
            reinterpret_cast<MKL_Complex16 const *>(x), &inc_x, reinterpret_cast<MKL_Complex16 *>(a), &lda);
}

auto sgetrf(eint const m, eint const n, float *a, eint const lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::sgetrf(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(eint const m, eint const n, double *a, eint const lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::dgetrf(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(eint const m, eint const n, std::complex<float> *a, eint const lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::cgetrf(&m, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, ipiv, &info);
    return info;
}

auto zgetrf(eint const m, eint const n, std::complex<double> *a, eint const lda, eint *ipiv) -> eint {
    LabeledSection1(mkl_interface());
    eint info{0};
    ::zgetrf(&m, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, ipiv, &info);
    return info;
}

auto sgetri(eint const n, float *a, eint const lda, eint const *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint               info{0};
    eint               lwork = n * 64;
    std::vector<float> work(lwork);
    ::sgetri(&n, a, &lda, (eint *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(eint const n, double *a, eint const lda, eint const *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                info{0};
    eint                lwork = n * 64;
    std::vector<double> work(lwork);
    ::dgetri(&n, a, &lda, (eint *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(eint const n, std::complex<float> *a, eint const lda, eint const *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                             info{0};
    eint                             lwork = n * 64;
    std::vector<std::complex<float>> work(lwork);
    ::cgetri(&n, reinterpret_cast<MKL_Complex8 *>(a), &lda, (eint *)ipiv, reinterpret_cast<MKL_Complex8 *>(work.data()),
             &lwork, &info);
    return info;
}

auto zgetri(eint const n, std::complex<double> *a, eint const lda, eint const *ipiv) -> eint {
    LabeledSection1(mkl_interface());

    eint                              info{0};
    eint                              lwork = n * 64;
    std::vector<std::complex<double>> work(lwork);
    ::zgetri(&n, reinterpret_cast<MKL_Complex16 *>(a), &lda, (eint *)ipiv,
             reinterpret_cast<MKL_Complex16 *>(work.data()), &lwork, &info);
    return info;
}

auto slange(char const norm_type, eint const m, eint const n, float const *A, eint const lda, float *work) -> float {
    LabeledSection1(mkl_interface());

    return ::slange(&norm_type, &m, &n, A, &lda, work);
}

auto dlange(char const norm_type, eint const m, eint const n, double const *A, eint const lda, double *work) -> double {
    LabeledSection1(mkl_interface());

    return ::dlange(&norm_type, &m, &n, A, &lda, work);
}

auto clange(char const norm_type, eint const m, eint const n, std::complex<float> const *A, eint const lda, float *work)
    -> float {
    LabeledSection1(mkl_interface());

    return ::clange(&norm_type, &m, &n, reinterpret_cast<MKL_Complex8 const *>(A), &lda, work);
}

auto zlange(char const norm_type, eint const m, eint const n, std::complex<double> const *A, eint const lda,
            double *work) -> double {
    LabeledSection1(mkl_interface());

    return ::zlange(&norm_type, &m, &n, reinterpret_cast<MKL_Complex16 const *>(A), &lda, work);
}

void slassq(eint const n, float const *x, eint const incx, float *scale, float *sumsq) {
    LabeledSection1(mkl_interface());
    ::slassq(&n, x, &incx, scale, sumsq);
}

void dlassq(eint const n, double const *x, eint const incx, double *scale, double *sumsq) {
    LabeledSection1(mkl_interface());
    ::dlassq(&n, x, &incx, scale, sumsq);
}

void classq(eint const n, std::complex<float> const *x, eint const incx, float *scale, float *sumsq) {
    LabeledSection1(mkl_interface());
    ::classq(&n, reinterpret_cast<MKL_Complex8 const *>(x), &incx, scale, sumsq);
}

void zlassq(eint const n, std::complex<double> const *x, eint const incx, double *scale, double *sumsq) {
    LabeledSection1(mkl_interface());
    ::zlassq(&n, reinterpret_cast<MKL_Complex16 const *>(x), &incx, scale, sumsq);
}

auto sgesdd(char jobz, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt, eint ldvt)
    -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, eint m, eint n, std::complex<float> *a, eint lda, float *s, std::complex<float> *u, eint ldu,
            std::complex<float> *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_cgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_float *>(a), lda, s,
                          reinterpret_cast<lapack_complex_float *>(u), ldu,
                          reinterpret_cast<lapack_complex_float *>(vt), ldvt);
}

auto zgesdd(char jobz, eint m, eint n, std::complex<double> *a, eint lda, double *s, std::complex<double> *u, eint ldu,
            std::complex<double> *vt, eint ldvt) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_zgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_double *>(a), lda, s,
                          reinterpret_cast<lapack_complex_double *>(u), ldu,
                          reinterpret_cast<lapack_complex_double *>(vt), ldvt);
}

auto sgesvd(char jobu, char jobvt, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt,
            eint ldvt, float *superb) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgesvd(char jobu, char jobvt, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt,
            eint ldvt, double *superb) -> eint {
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

auto strsyl(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b, eint ldb,
            float *c, eint ldc, float *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_strsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda, double const *b, eint ldb,
            double *c, eint ldc, double *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a, eint lda,
            std::complex<float> const *b, eint ldb, std::complex<float> *c, eint ldc, float *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_ctrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, reinterpret_cast<lapack_complex_float const *>(a),
                          lda, reinterpret_cast<lapack_complex_float const *>(b), ldb,
                          reinterpret_cast<lapack_complex_float *>(c), ldc, scale);
}

auto ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<double> const *a, eint lda,
            std::complex<double> const *b, eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_ztrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n,
                          reinterpret_cast<lapack_complex_double const *>(a), lda,
                          reinterpret_cast<lapack_complex_double const *>(b), ldb,
                          reinterpret_cast<lapack_complex_double *>(c), ldc, scale);
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

auto sorgqr(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto dorgqr(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, std::complex<float> const *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float const *>(tau));
}

auto zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, std::complex<double> const *tau) -> eint {
    LabeledSection1(mkl_interface());
    return LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double const *>(tau));
}

END_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::mkl)