//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <stdexcept>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(ssyev, SSYEV)(char *, char *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *);
extern void FC_GLOBAL(ssterf, SSTERF)(int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(dsterf, DSTERF)(int_t *, double *, double *, int_t *);
}

auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ssyev, SSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto ssterf(int_t n, float *d, float *e) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ssterf, SSTERF)(&n, d, e, &info);

    return info;
}

auto dsterf(int_t n, double *d, double *e) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dsterf, DSTERF)(&n, d, e, &info);

    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)