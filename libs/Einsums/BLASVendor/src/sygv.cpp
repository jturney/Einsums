//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(ssygv, SSYGV)(int_t *, char *, char *, int_t *, float *, int_t *, float *, int_t *, float *, float *, int_t *,
                                    int_t *);
extern void FC_GLOBAL(dsygv, DSYGV)(int_t *, char *, char *, int_t *, double *, int_t *, double *, int_t *, double *, double *, int_t *,
                                    int_t *);
extern void FC_GLOBAL(chegv, CHEGV)(int_t *, char *, char *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *,
                                    float *, std::complex<float> *, int_t *, float *, int_t *);
extern void FC_GLOBAL(zhegv, ZHEGV)(int_t *, char *, char *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                    double *, std::complex<double> *, int_t *, double *, int_t *);
}

auto ssygv(int_t itype, char jobz, char uplo, int_t n, float *a, int_t lda, float *b, int_t ldb, float *w) -> int_t {
    LabeledSection0();

    int_t info{0};
    int_t lwork{-1};
    float work_query;

    FC_GLOBAL(ssygv, SSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)work_query;
    BufferVector<float> work(lwork);

    FC_GLOBAL(ssygv, SSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, &info);

    return info;
}

auto dsygv(int_t itype, char jobz, char uplo, int_t n, double *a, int_t lda, double *b, int_t ldb, double *w) -> int_t {
    LabeledSection0();

    int_t  info{0};
    int_t  lwork{-1};
    double work_query;

    FC_GLOBAL(dsygv, DSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)work_query;
    BufferVector<double> work(lwork);

    FC_GLOBAL(dsygv, DSYGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, &info);

    return info;
}

auto chegv(int_t itype, char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb, float *w)
    -> int_t {
    LabeledSection0();

    int_t               info{0};
    int_t               lwork{-1};
    std::complex<float> work_query;
    int_t               lrwork = std::max((int_t)1, 3 * n - 2);
    BufferVector<float> rwork(lrwork);

    FC_GLOBAL(chegv, CHEGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork, rwork.data(), &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)(work_query.real());
    BufferVector<std::complex<float>> work(lwork);

    FC_GLOBAL(chegv, CHEGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, rwork.data(), &info);

    return info;
}

auto zhegv(int_t itype, char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb, double *w)
    -> int_t {
    LabeledSection0();

    int_t                info{0};
    int_t                lwork{-1};
    std::complex<double> work_query;
    int_t                lrwork = std::max((int_t)1, 3 * n - 2);
    BufferVector<double> rwork(lrwork);

    FC_GLOBAL(zhegv, ZHEGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork, rwork.data(), &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)(work_query.real());
    BufferVector<std::complex<double>> work(lwork);

    FC_GLOBAL(zhegv, ZHEGV)(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, rwork.data(), &info);

    return info;
}

} // namespace einsums::blas::vendor
