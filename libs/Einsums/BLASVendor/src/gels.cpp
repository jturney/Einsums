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
extern void FC_GLOBAL(sgels, SGELS)(char *, int_t *, int_t *, int_t *, float *, int_t *, float *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgels, DGELS)(char *, int_t *, int_t *, int_t *, double *, int_t *, double *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgels, CGELS)(char *, int_t *, int_t *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *,
                                    std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zgels, ZGELS)(char *, int_t *, int_t *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                    std::complex<double> *, int_t *, int_t *);
}

auto sgels(char trans, int_t m, int_t n, int_t nrhs, float *a, int_t lda, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    int_t lwork{-1};
    float work_query;

    FC_GLOBAL(sgels, SGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)work_query;
    BufferVector<float> work(lwork);

    FC_GLOBAL(sgels, SGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work.data(), &lwork, &info);

    return info;
}

auto dgels(char trans, int_t m, int_t n, int_t nrhs, double *a, int_t lda, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t  info{0};
    int_t  lwork{-1};
    double work_query;

    FC_GLOBAL(dgels, DGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)work_query;
    BufferVector<double> work(lwork);

    FC_GLOBAL(dgels, DGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work.data(), &lwork, &info);

    return info;
}

auto cgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t               info{0};
    int_t               lwork{-1};
    std::complex<float> work_query;

    FC_GLOBAL(cgels, CGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)(work_query.real());
    BufferVector<std::complex<float>> work(lwork);

    FC_GLOBAL(cgels, CGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work.data(), &lwork, &info);

    return info;
}

auto zgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t                info{0};
    int_t                lwork{-1};
    std::complex<double> work_query;

    FC_GLOBAL(zgels, ZGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &work_query, &lwork, &info);

    if (info != 0) {
        return info;
    }

    lwork = (int_t)(work_query.real());
    BufferVector<std::complex<double>> work(lwork);

    FC_GLOBAL(zgels, ZGELS)(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work.data(), &lwork, &info);

    return info;
}

} // namespace einsums::blas::vendor
