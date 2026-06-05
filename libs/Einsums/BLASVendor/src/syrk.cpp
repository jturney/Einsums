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
extern void FC_GLOBAL(ssyrk, SSYRK)(char *, char *, int_t *, int_t *, float *, float const *, int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(dsyrk, DSYRK)(char *, char *, int_t *, int_t *, double *, double const *, int_t *, double *, double *, int_t *);
extern void FC_GLOBAL(csyrk, CSYRK)(char *, char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zsyrk, ZSYRK)(char *, char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(cherk, CHERK)(char *, char *, int_t *, int_t *, float *, std::complex<float> const *, int_t *, float *,
                                    std::complex<float> *, int_t *);
extern void FC_GLOBAL(zherk, ZHERK)(char *, char *, int_t *, int_t *, double *, std::complex<double> const *, int_t *, double *,
                                    std::complex<double> *, int_t *);
}

void ssyrk(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float beta, float *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(ssyrk, SSYRK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void dsyrk(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double beta, double *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(dsyrk, DSYRK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void csyrk(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(csyrk, CSYRK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void zsyrk(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zsyrk, ZSYRK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void cherk(char uplo, char trans, int_t n, int_t k, float alpha, std::complex<float> const *a, int_t lda, float beta,
           std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(cherk, CHERK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void zherk(char uplo, char trans, int_t n, int_t k, double alpha, std::complex<double> const *a, int_t lda, double beta,
           std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    FC_GLOBAL(zherk, ZHERK)(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

} // namespace einsums::blas::vendor
