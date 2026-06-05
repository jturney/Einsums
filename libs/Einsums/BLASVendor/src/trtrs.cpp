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
extern void FC_GLOBAL(strtrs, STRTRS)(char *, char *, char *, int_t *, int_t *, float const *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dtrtrs, DTRTRS)(char *, char *, char *, int_t *, int_t *, double const *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(ctrtrs, CTRTRS)(char *, char *, char *, int_t *, int_t *, std::complex<float> const *, int_t *, std::complex<float> *,
                                      int_t *, int_t *);
extern void FC_GLOBAL(ztrtrs, ZTRTRS)(char *, char *, char *, int_t *, int_t *, std::complex<double> const *, int_t *,
                                      std::complex<double> *, int_t *, int_t *);
}

auto strtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(strtrs, STRTRS)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto dtrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dtrtrs, DTRTRS)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto ctrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b,
            int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ctrtrs, CTRTRS)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto ztrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b,
            int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ztrtrs, ZTRTRS)(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

} // namespace einsums::blas::vendor
