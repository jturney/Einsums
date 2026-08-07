//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(cheev, CHEEV)(char *job, char *uplo, int_t *n, std::complex<float> *a, int_t *lda, float *w,
                                    std::complex<float> *work, int_t *lwork, float *rwork, int_t *info);
extern void FC_GLOBAL(zheev, ZHEEV)(char *job, char *uplo, int_t *n, std::complex<double> *a, int_t *lda, double *w,
                                    std::complex<double> *work, int_t *lwork, double *rwork, int_t *info);
}

auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork, float *rwork)
    -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cheev, CHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}
auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zheev, ZHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)