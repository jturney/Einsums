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
extern void FC_GLOBAL(slassq, SLASSQ)(int_t *n, float const *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(dlassq, DLASSQ)(int_t *n, double const *x, int_t *incx, double *scale, double *sumsq);
extern void FC_GLOBAL(classq, CLASSQ)(int_t *n, std::complex<float> const *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(zlassq, ZLASSQ)(int_t *n, std::complex<double> const *x, int_t *incx, double *scale, double *sumsq);

extern float  FC_GLOBAL(snrm2, SNRM2)(int_t *n, float const *x, int_t *incx);
extern double FC_GLOBAL(dnrm2, DNRM2)(int_t *n, double const *x, int_t *incx);
extern float  FC_GLOBAL(scnrm2, SCNRM2)(int_t *n, std::complex<float> const *x, int_t *incx);
extern double FC_GLOBAL(dznrm2, DZNRM2)(int_t *n, std::complex<double> const *x, int_t *incx);
}

void slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection("slassq");
    FC_GLOBAL(slassq, SLASSQ)(&n, x, &incx, scale, sumsq);
}

void dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection("dlassq");
    FC_GLOBAL(dlassq, DLASSQ)(&n, x, &incx, scale, sumsq);
}

void classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection("classq");
    FC_GLOBAL(classq, CLASSQ)(&n, x, &incx, scale, sumsq);
}

void zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(zlassq, ZLASSQ)(&n, x, &incx, scale, sumsq);
}

float snrm2(int_t n, float const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(snrm2, SNRM2)(&n, x, &incx);
}

double dnrm2(int_t n, double const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(dnrm2, DNRM2)(&n, x, &incx);
}

float scnrm2(int_t n, std::complex<float> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(scnrm2, SCNRM2)(&n, x, &incx);
}

double dznrm2(int_t n, std::complex<double> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(dznrm2, DZNRM2)(&n, x, &incx);
}

EINSUMS_NAMESPACE_END(blas::vendor)