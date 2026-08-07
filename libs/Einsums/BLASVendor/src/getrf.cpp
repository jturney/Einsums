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
extern void FC_GLOBAL(sgetrf, SGETRF)(int_t *, int_t *, float *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(dgetrf, DGETRF)(int_t *, int_t *, double *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(cgetrf, CGETRF)(int_t *, int_t *, std::complex<float> *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zgetrf, ZGETRF)(int_t *, int_t *, std::complex<double> *, int_t *, int_t *, int_t *);
}

auto sgetrf(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(sgetrf, SGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dgetrf, DGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cgetrf, CGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto zgetrf(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zgetrf, ZGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)