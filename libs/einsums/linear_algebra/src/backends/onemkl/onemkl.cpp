//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "onemkl.hpp"

#include "einsums/Print.hpp"
#include "fmt/format.h"
#include "oneapi/mkl/lapack.hpp"

#include <CL/sycl.hpp>
#include <exception>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/types.hpp>
#include <stdexcept>

namespace einsums::backend::onemkl {

namespace {

// List of valid SYCL devices
std::vector<sycl::device> g_Devices;

// List of valid SYCL queues; 1-to-1 correpondence to g_Devices
std::vector<sycl::queue> g_Queues;

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

auto transpose_to_oneapi(char transpose) -> oneapi::mkl::transpose {
    switch (transpose) {
    case 'N':
    case 'n':
        return oneapi::mkl::transpose::nontrans;
    case 'T':
    case 't':
        return oneapi::mkl::transpose::trans;
    case 'C':
    case 'c':
        return oneapi::mkl::transpose::conjtrans;
    }
    println_warn("Unknown transpose code {}, defaulting to CblasNoTrans.", transpose);
    return oneapi::mkl::transpose::nontrans;
}

auto g_ExceptionHandler = [](const sycl::exception_list &exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            println("Caught asynchronous SYCL exception during GEMM:\n{}\n{}", e.what(), e.code().value());
        }
    }
};

} // namespace

void initialize() {
    g_Devices.emplace_back(sycl::host_selector());
    g_Queues.emplace_back(sycl::queue(g_Devices[0], g_ExceptionHandler));
}

void finalize() {
    g_Queues.clear();
    g_Devices.clear();
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, // NOLINT
           double beta, double *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    // Call gemm. This call is asynchronous.
    auto event = oneapi::mkl::blas::row_major::gemm(g_Queues[0], transpose_to_oneapi(transa), transpose_to_oneapi(transb), m, n, k, alpha,
                                                    a, lda, b, ldb, beta, c, ldc);
    // The call to gemm returns immediately. Wait for the event to be completed.
    event.wait();
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, // NOLINT
           int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_oneapi(transa);
    if (TransA == oneapi::mkl::transpose::conjtrans)
        throw std::invalid_argument("einsums::backend::onemkl::dgemv transa argument is invalid.");

    // cblas_dgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
    auto event = oneapi::mkl::blas::row_major::gemv(g_Queues[0], TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
    event.wait();
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *scratchpad, int scratchpad_size) -> int {
    // return LAPACKE_dsyev(LAPACK_ROW_MAJOR, job, uplo, n, a, lda, w);
    oneapi::mkl::job jobz;
    switch (job) {
    case 'n':
    case 'N':
        jobz = oneapi::mkl::job::novec;
        break;
    case 'v':
    case 'V':
        jobz = oneapi::mkl::job::vec;
        break;
    default:
        throw std::invalid_argument(fmt::format("einsums::backend::onemkl::dsyev job argument is invalid: {}", job));
    }
    oneapi::mkl::uplo uploz;
    switch (uplo) {
    case 'u':
    case 'U':
        uploz = oneapi::mkl::uplo::upper;
        break;
    case 'l':
    case 'L':
        uploz = oneapi::mkl::uplo::lower;
        break;
    default:
        throw std::invalid_argument(fmt::format("einsums::backend::onemkl::dsyev uplo argument is invalid: {}", uplo));
    }

    auto event = oneapi::mkl::lapack::syevd(g_Queues[0], jobz, uploz, n, a, lda, w, scratchpad, scratchpad_size);
    event.wait();
    return 0;
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    // return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
    auto scratchpad_size =
        oneapi::mkl::lapack::getrs_scratchpad_size<double>(g_Queues[0], oneapi::mkl::transpose::nontrans, n, nrhs, lda, ldb);
    std::vector<double> scratchpad(scratchpad_size);
    std::vector<int64_t> scratchpad_ipiv(n);

    auto event = oneapi::mkl::lapack::getrs(g_Queues[0], oneapi::mkl::transpose::nontrans, n, nrhs, a, lda, scratchpad_ipiv.data(), b, ldb,
                                            scratchpad.data(), scratchpad_size);
    event.wait();

    for (int i = 0; i < n; i++) {
        ipiv[i] = static_cast<int>(scratchpad_ipiv[i]);
    }
    return 0;
}

void dscal(int n, double alpha, double *vec, int inc) {
    // cblas_dscal(n, alpha, vec, inc);
    auto event = oneapi::mkl::blas::row_major::scal(g_Queues[0], n, alpha, vec, inc);
    event.wait();
}

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double {
    // return cblas_ddot(n, x, incx, y, incy);
    double result{0};
    auto event = oneapi::mkl::blas::row_major::dot(g_Queues[0], n, x, incx, y, incy, &result);
    event.wait();
    return result;
}

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    // cblas_daxpy(n, alpha_x, x, inc_x, y, inc_y);
    auto event = oneapi::mkl::blas::row_major::axpy(g_Queues[0], n, alpha_x, x, inc_x, y, inc_y);
    event.wait();
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::onemkl::dger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::onemkl::dger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::onemkl::dger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::onemkl::dger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(1, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::onemkl::dger: lda ({}) is less than max(1, n ({})).", lda, n));
    }

    // cblas_dger(CblasRowMajor, m, n, alpha, y, inc_y, x, inc_x, a, lda);

    auto event = oneapi::mkl::blas::row_major::ger(g_Queues[0], m, n, alpha, x, inc_x, y, inc_y, a, lda);
    event.wait();
}

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    // return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
    auto scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(g_Queues[0], m, n, lda);
    std::vector<double> scratchpad(scratchpad_size);
    std::vector<int64_t> scratchpad_ipiv(std::min(m, n));

    auto event = oneapi::mkl::lapack::getrf(g_Queues[0], m, n, a, lda, scratchpad_ipiv.data(), scratchpad.data(), scratchpad_size);
    event.wait();

    for (int i = 0; i < std::min(m, n); i++) {
        ipiv[i] = scratchpad_ipiv[i];
    }

    return 0;
}

auto dgetri(int n, double *a, int lda, const int *ipiv, double *scratchpad, int scratchpad_size) -> int {
    // return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, lda, (int *)ipiv);
    // auto scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<double>(g_Queues[0], n, lda);
    // std::vector<double> scratchpad(scratchpad_size);
    std::vector<int64_t> scratchpad_ipiv(n);
    for (int i = 0; i < n; i++) {
        scratchpad_ipiv[i] = ipiv[i];
    }

    auto event = oneapi::mkl::lapack::getri(g_Queues[0], n, a, lda, scratchpad_ipiv.data(), scratchpad, scratchpad_size);
    event.wait();

    return 0;
}

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *) -> double {
    println_warn("calling non-onemkl dlange function");
    return LAPACKE_dlange(LAPACK_ROW_MAJOR, norm_type, m, n, A, lda);
}

auto dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *, int, int *) -> int {
    // TODO: Wrap gesvd here.
    // return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);

    if (jobz != 'A' || jobz != 'a') {
        throw std::runtime_error("dgess: only jobz == 'A' was expected");
    }

    uint64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(g_Queues[0], oneapi::mkl::jobsvd::vectors,
                                                                                  oneapi::mkl::jobsvd::vectors, m, n, lda, ldu, ldvt);
    std::vector<double> scratchpad(scratchpad_size);

    auto event = oneapi::mkl::lapack::gesvd(g_Queues[0], oneapi::mkl::jobsvd::vectors, oneapi::mkl::jobsvd::vectors, m, n, a, lda, s, u,
                                            ldu, vt, ldvt, scratchpad.data(), scratchpad_size);
    event.wait();

    return 0;
}

} // namespace einsums::backend::onemkl
