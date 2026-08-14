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
extern void FC_GLOBAL(saxpy, SAXPY)(int_t *, float *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(daxpy, DAXPY)(int_t *, double *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(caxpy, CAXPY)(int_t *, std::complex<float> *, std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zaxpy, ZAXPY)(int_t *, std::complex<double> *, std::complex<double> const *, int_t *, std::complex<double> *,
                                    int_t *);
}

// The n guards below are the same defence as copy.cpp's and scal.cpp's:
// Accelerate validates the increments before the length and ABORTS THE PROCESS
// on a zero one, while an empty tensor legitimately carries a zero increment.
// `axpby` scales then adds, so the same zero-extent operand reaches both.

void saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(saxpy, SAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(daxpy, DAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(caxpy, CAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(zaxpy, ZAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void saxpby(int_t const n, float const a, float const *x, int_t const incx, float const b, float *y, int_t const incy) {
    LabeledSection0();
    // If x aliases y, the scal-then-axpy split below would scale y (and hence x)
    // before axpy reads it -- corrupting the result (e.g. b==0 zeroes x first).
    // For that case y = a*x + b*y = (a + b)*y, a single scaling.
    if (x == y && incx == incy) {
        if (incy == 0) {
            *y *= (a + b);
        } else {
            sscal(n, a + b, y, incy);
        }
        return;
    }
    if (incy == 0) {
        *y *= b;
    } else {
        sscal(n, b, y, incy);
    }
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(int_t const n, double const a, double const *x, int_t const incx, double const b, double *y, int_t const incy) {
    LabeledSection0();
    // See saxpby: alias-safe path when x == y.
    if (x == y && incx == incy) {
        if (incy == 0) {
            *y *= (a + b);
        } else {
            dscal(n, a + b, y, incy);
        }
        return;
    }
    if (incy == 0) {
        *y *= b;
    } else {
        dscal(n, b, y, incy);
    }
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(int_t const n, std::complex<float> const a, std::complex<float> const *x, int_t const incx, std::complex<float> const b,
            std::complex<float> *y, int_t const incy) {
    LabeledSection0();
    // See saxpby: alias-safe path when x == y.
    if (x == y && incx == incy) {
        if (incy == 0) {
            *y *= (a + b);
        } else {
            cscal(n, a + b, y, incy);
        }
        return;
    }
    if (incy == 0) {
        *y *= b;
    } else {
        cscal(n, b, y, incy);
    }
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(int_t const n, std::complex<double> const a, std::complex<double> const *x, int_t const incx, std::complex<double> const b,
            std::complex<double> *y, int_t const incy) {
    LabeledSection0();
    // See saxpby: alias-safe path when x == y.
    if (x == y && incx == incy) {
        if (incy == 0) {
            *y *= (a + b);
        } else {
            zscal(n, a + b, y, incy);
        }
        return;
    }
    if (incy == 0) {
        *y *= b;
    } else {
        zscal(n, b, y, incy);
    }
    zaxpy(n, a, x, incx, y, incy);
}

EINSUMS_NAMESPACE_END(blas::vendor)