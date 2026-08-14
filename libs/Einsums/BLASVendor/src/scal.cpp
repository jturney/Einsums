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
extern void FC_GLOBAL(sscal, SSCAL)(int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(dscal, DSCAL)(int_t *, double *, double *, int_t *);
extern void FC_GLOBAL(cscal, CSCAL)(int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zscal, ZSCAL)(int_t *, std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(csscal, CSSCAL)(int_t *, float *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zdscal, ZDSCAL)(int_t *, double *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(srscl, SRSCL)(int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(drscl, DRSCL)(int_t *, double *, double *, int_t *);
extern void FC_GLOBAL(csrscl, CSRSCL)(int_t *, float *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zdrscl, ZDRSCL)(int_t *, double *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(slascl, SLASCL)(char *, int_t *, int_t *, float *, float *, int_t *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dlascl, DLASCL)(char *, int_t *, int_t *, double *, double *, int_t *, int_t *, double *, int_t *, int_t *);
}

// Each routine below quick-returns on an empty vector rather than handing it to
// the vendor. Reference BLAS returns on n < 1 before validating anything, but
// Accelerate validates the increment first and ABORTS THE PROCESS on a zero
// one -- and an empty tensor legitimately carries a zero increment, because
// `get_incx` reports the smallest stride over the extent > 1 axes and a
// column-major (0, k) tensor has stride(1) == dim(0) == 0. `axpby` on such a
// tensor scales before it adds, so a grouped run holding one zero-extent entry
// was enough to kill the process on macOS/Accelerate while passing everywhere
// else.
//
// Same reasoning as the m/n guards in gemm.cpp and gemv.cpp, and the n guard in
// copy.cpp.

void sscal(int_t n, float alpha, float *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(sscal, SSCAL)(&n, &alpha, vec, &inc);
}

void dscal(int_t n, double alpha, double *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc);
}

void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(cscal, CSCAL)(&n, &alpha, vec, &inc);
}

void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(zscal, ZSCAL)(&n, &alpha, vec, &inc);
}

void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(csscal, CSSCAL)(&n, &alpha, vec, &inc);
}

void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(zdscal, ZDSCAL)(&n, &alpha, vec, &inc);
}

void srscl(int_t n, float alpha, float *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(srscl, SRSCL)(&n, &alpha, vec, &inc);
}

void drscl(int_t n, double alpha, double *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(drscl, DRSCL)(&n, &alpha, vec, &inc);
}

void csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(csrscl, CSRSCL)(&n, &alpha, vec, &inc);
}

void zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(zdrscl, ZDRSCL)(&n, &alpha, vec, &inc);
}

int_t slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda) {
    LabeledSection0();

    int_t info = 0;
    FC_GLOBAL(slascl, SLASCL)(&type, &kl, &ku, &cfrom, &cto, &m, &n, vec, &lda, &info);
    return info;
}

int_t dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda) {
    LabeledSection0();

    int_t info = 0;
    FC_GLOBAL(dlascl, DLASCL)(&type, &kl, &ku, &cfrom, &cto, &m, &n, vec, &lda, &info);
    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)