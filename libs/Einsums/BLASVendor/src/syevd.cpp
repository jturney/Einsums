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
extern void FC_GLOBAL(ssyevd, SSYEVD)(char *, char *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(dsyevd, DSYEVD)(char *, char *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(cheevd, CHEEVD)(char *, char *, int_t *, std::complex<float> *, int_t *, float *, std::complex<float> *, int_t *,
                                      float *, int_t *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zheevd, ZHEEVD)(char *, char *, int_t *, std::complex<double> *, int_t *, double *, std::complex<double> *, int_t *,
                                      double *, int_t *, int_t *, int_t *, int_t *);
}

auto ssyevd(char jobz, char uplo, int_t n, float *a, int_t lda, float *w) -> int_t {
    LabeledSection0();

    int_t info{0};
    int_t lwork{-1};
    int_t liwork{-1};
    float work_query;
    int_t iwork_query;

    FC_GLOBAL(ssyevd, SSYEVD)(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &iwork_query, &liwork, &info);

    if (info != 0) {
        return info;
    }

    lwork  = (int_t)work_query;
    liwork = iwork_query;
    BufferVector<float> work(lwork);
    BufferVector<int_t> iwork(liwork);

    FC_GLOBAL(ssyevd, SSYEVD)(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);

    return info;
}

auto dsyevd(char jobz, char uplo, int_t n, double *a, int_t lda, double *w) -> int_t {
    LabeledSection0();

    int_t  info{0};
    int_t  lwork{-1};
    int_t  liwork{-1};
    double work_query;
    int_t  iwork_query;

    FC_GLOBAL(dsyevd, DSYEVD)(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &iwork_query, &liwork, &info);

    if (info != 0) {
        return info;
    }

    lwork  = (int_t)work_query;
    liwork = iwork_query;
    BufferVector<double> work(lwork);
    BufferVector<int_t>  iwork(liwork);

    FC_GLOBAL(dsyevd, DSYEVD)(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);

    return info;
}

auto cheevd(char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w) -> int_t {
    LabeledSection0();

    int_t               info{0};
    int_t               lwork{-1};
    int_t               lrwork{-1};
    int_t               liwork{-1};
    std::complex<float> work_query;
    float               rwork_query;
    int_t               iwork_query;

    FC_GLOBAL(cheevd, CHEEVD)
    (&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info);

    if (info != 0) {
        return info;
    }

    lwork  = (int_t)(work_query.real());
    lrwork = (int_t)rwork_query;
    liwork = iwork_query;
    BufferVector<std::complex<float>> work(lwork);
    BufferVector<float>               rwork(lrwork);
    BufferVector<int_t>               iwork(liwork);

    FC_GLOBAL(cheevd, CHEEVD)
    (&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);

    return info;
}

auto zheevd(char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w) -> int_t {
    LabeledSection0();

    int_t                info{0};
    int_t                lwork{-1};
    int_t                lrwork{-1};
    int_t                liwork{-1};
    std::complex<double> work_query;
    double               rwork_query;
    int_t                iwork_query;

    FC_GLOBAL(zheevd, ZHEEVD)
    (&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info);

    if (info != 0) {
        return info;
    }

    lwork  = (int_t)(work_query.real());
    lrwork = (int_t)rwork_query;
    liwork = iwork_query;
    BufferVector<std::complex<double>> work(lwork);
    BufferVector<double>               rwork(lrwork);
    BufferVector<int_t>                iwork(liwork);

    FC_GLOBAL(zheevd, ZHEEVD)
    (&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);

    return info;
}

} // namespace einsums::blas::vendor
