//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(sgetrs, SGETRS)(char *, int_t *, int_t *, float const *, int_t *, int_t const *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgetrs, DGETRS)(char *, int_t *, int_t *, double const *, int_t *, int_t const *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgetrs, CGETRS)(char *, int_t *, int_t *, std::complex<float> const *, int_t *, int_t const *, std::complex<float> *,
                                      int_t *, int_t *);
extern void FC_GLOBAL(zgetrs, ZGETRS)(char *, int_t *, int_t *, std::complex<double> const *, int_t *, int_t const *,
                                      std::complex<double> *, int_t *, int_t *);
}

auto sgetrs(char trans, int_t n, int_t nrhs, float const *a, int_t lda, int_t const *ipiv, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(sgetrs, SGETRS)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgetrs(char trans, int_t n, int_t nrhs, double const *a, int_t lda, int_t const *ipiv, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dgetrs, DGETRS)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgetrs(char trans, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, int_t const *ipiv, std::complex<float> *b, int_t ldb)
    -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cgetrs, CGETRS)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto zgetrs(char trans, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, int_t const *ipiv, std::complex<double> *b,
            int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zgetrs, ZGETRS)(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

} // namespace einsums::blas::vendor
