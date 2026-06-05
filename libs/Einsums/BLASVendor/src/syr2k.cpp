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
extern void FC_GLOBAL(ssyr2k, SSYR2K)(char *, char *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *, float *,
                                      float *, int_t *);
extern void FC_GLOBAL(dsyr2k, DSYR2K)(char *, char *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *,
                                      double *, double *, int_t *);
extern void FC_GLOBAL(csyr2k, CSYR2K)(char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                      std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zsyr2k, ZSYR2K)(char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                      std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(cher2k, CHER2K)(char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                      std::complex<float> const *, int_t *, float *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zher2k, ZHER2K)(char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                      std::complex<double> const *, int_t *, double *, std::complex<double> *, int_t *);
}

void ssyr2k(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb, float beta,
            float *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(ssyr2k, SSYR2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void dsyr2k(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb, double beta,
            double *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(dsyr2k, DSYR2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void csyr2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
            std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(csyr2k, CSYR2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void zsyr2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
            std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zsyr2k, ZSYR2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void cher2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
            std::complex<float> const *b, int_t ldb, float beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(cher2k, CHER2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void zher2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
            std::complex<double> const *b, int_t ldb, double beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zher2k, ZHER2K)(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

} // namespace einsums::blas::vendor
