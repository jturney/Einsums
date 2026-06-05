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
extern void FC_GLOBAL(ssymm, SSYMM)(char *, char *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *, float *,
                                    float *, int_t *);
extern void FC_GLOBAL(dsymm, DSYMM)(char *, char *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *, double *,
                                    double *, int_t *);
extern void FC_GLOBAL(csymm, CSYMM)(char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zsymm, ZSYMM)(char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(chemm, CHEMM)(char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zhemm, ZHEMM)(char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *, int_t *);
}

void ssymm(char side, char uplo, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *b, int_t ldb, float beta, float *c,
           int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(ssymm, SSYMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void dsymm(char side, char uplo, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *b, int_t ldb, double beta,
           double *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(dsymm, DSYMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void csymm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(csymm, CSYMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void zsymm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zsymm, ZSYMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void chemm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(chemm, CHEMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void zhemm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zhemm, ZHEMM)(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

} // namespace einsums::blas::vendor
