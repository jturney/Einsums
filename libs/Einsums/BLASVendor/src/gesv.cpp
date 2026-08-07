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
extern void FC_GLOBAL(sgesv, SGESV)(int_t *, int_t *, float *, int_t *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgesv, DGESV)(int_t *, int_t *, double *, int_t *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgesv, CGESV)(int_t *, int_t *, std::complex<float> *, int_t *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zgesv, ZGESV)(int_t *, int_t *, std::complex<double> *, int_t *, int_t *, std::complex<double> *, int_t *, int_t *);
}

auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(sgesv, SGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dgesv, DGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cgesv, CGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zgesv, ZGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)