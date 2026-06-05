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
extern void FC_GLOBAL(spotrf, SPOTRF)(char *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dpotrf, DPOTRF)(char *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cpotrf, CPOTRF)(char *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zpotrf, ZPOTRF)(char *, int_t *, std::complex<double> *, int_t *, int_t *);

extern void FC_GLOBAL(spotrs, SPOTRS)(char *, int_t *, int_t *, float const *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dpotrs, DPOTRS)(char *, int_t *, int_t *, double const *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cpotrs, CPOTRS)(char *, int_t *, int_t *, std::complex<float> const *, int_t *, std::complex<float> *, int_t *,
                                      int_t *);
extern void FC_GLOBAL(zpotrs, ZPOTRS)(char *, int_t *, int_t *, std::complex<double> const *, int_t *, std::complex<double> *, int_t *,
                                      int_t *);

extern void FC_GLOBAL(spotri, SPOTRI)(char *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dpotri, DPOTRI)(char *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cpotri, CPOTRI)(char *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zpotri, ZPOTRI)(char *, int_t *, std::complex<double> *, int_t *, int_t *);
}

auto spotrf(char uplo, int_t n, float *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(spotrf, SPOTRF)(&uplo, &n, a, &lda, &info);
    return info;
}

auto dpotrf(char uplo, int_t n, double *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dpotrf, DPOTRF)(&uplo, &n, a, &lda, &info);
    return info;
}

auto cpotrf(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cpotrf, CPOTRF)(&uplo, &n, a, &lda, &info);
    return info;
}

auto zpotrf(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zpotrf, ZPOTRF)(&uplo, &n, a, &lda, &info);
    return info;
}

auto spotrs(char uplo, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(spotrs, SPOTRS)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto dpotrs(char uplo, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dpotrs, DPOTRS)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto cpotrs(char uplo, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cpotrs, CPOTRS)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto zpotrs(char uplo, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zpotrs, ZPOTRS)(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    return info;
}

auto spotri(char uplo, int_t n, float *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(spotri, SPOTRI)(&uplo, &n, a, &lda, &info);
    return info;
}

auto dpotri(char uplo, int_t n, double *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dpotri, DPOTRI)(&uplo, &n, a, &lda, &info);
    return info;
}

auto cpotri(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cpotri, CPOTRI)(&uplo, &n, a, &lda, &info);
    return info;
}

auto zpotri(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zpotri, ZPOTRI)(&uplo, &n, a, &lda, &info);
    return info;
}

} // namespace einsums::blas::vendor
