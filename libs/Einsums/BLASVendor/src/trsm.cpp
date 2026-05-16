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
extern void FC_GLOBAL(strsm, STRSM)(char *, char *, char *, char *, int_t *, int_t *, float *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dtrsm, DTRSM)(char *, char *, char *, char *, int_t *, int_t *, double *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(ctrsm, CTRSM)(char *, char *, char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *,
                                    int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(ztrsm, ZTRSM)(char *, char *, char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *,
                                    int_t *, std::complex<double> *, int_t *);
}

void strsm(char side, char uplo, char transa, char diag, int_t m, int_t n, float alpha, float const *a, int_t lda, float *b, int_t ldb) {
    LabeledSection0();

    FC_GLOBAL(strsm, STRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void dtrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, double alpha, double const *a, int_t lda, double *b, int_t ldb) {
    LabeledSection0();

    FC_GLOBAL(dtrsm, DTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void ctrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a,
           int_t lda, std::complex<float> *b, int_t ldb) {
    LabeledSection0();

    FC_GLOBAL(ctrsm, CTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

void ztrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a,
           int_t lda, std::complex<double> *b, int_t ldb) {
    LabeledSection0();

    FC_GLOBAL(ztrsm, ZTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

} // namespace einsums::blas::vendor
