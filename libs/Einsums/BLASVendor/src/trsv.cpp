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
extern void FC_GLOBAL(strsv, STRSV)(char *, char *, char *, int_t *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dtrsv, DTRSV)(char *, char *, char *, int_t *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(ctrsv, CTRSV)(char *, char *, char *, int_t *, std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(ztrsv, ZTRSV)(char *, char *, char *, int_t *, std::complex<double> const *, int_t *, std::complex<double> *,
                                    int_t *);
}

void strsv(char uplo, char trans, char diag, int_t n, float const *a, int_t lda, float *x, int_t incx) {
    LabeledSection0();

    FC_GLOBAL(strsv, STRSV)(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

void dtrsv(char uplo, char trans, char diag, int_t n, double const *a, int_t lda, double *x, int_t incx) {
    LabeledSection0();

    FC_GLOBAL(dtrsv, DTRSV)(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

void ctrsv(char uplo, char trans, char diag, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> *x, int_t incx) {
    LabeledSection0();

    FC_GLOBAL(ctrsv, CTRSV)(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

void ztrsv(char uplo, char trans, char diag, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> *x, int_t incx) {
    LabeledSection0();

    FC_GLOBAL(ztrsv, ZTRSV)(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

} // namespace einsums::blas::vendor
