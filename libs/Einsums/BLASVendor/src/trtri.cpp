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
extern void FC_GLOBAL(strtri, STRTRI)(char *, char *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dtrtri, DTRTRI)(char *, char *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(ctrtri, CTRTRI)(char *, char *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(ztrtri, ZTRTRI)(char *, char *, int_t *, std::complex<double> *, int_t *, int_t *);
}

auto strtri(char uplo, char diag, int_t n, float *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(strtri, STRTRI)(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

auto dtrtri(char uplo, char diag, int_t n, double *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dtrtri, DTRTRI)(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

auto ctrtri(char uplo, char diag, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ctrtri, CTRTRI)(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

auto ztrtri(char uplo, char diag, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ztrtri, ZTRTRI)(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

} // namespace einsums::blas::vendor
