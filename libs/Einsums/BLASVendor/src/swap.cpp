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
extern void FC_GLOBAL(sswap, SSWAP)(int_t *, float *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dswap, DSWAP)(int_t *, double *, int_t *, double *, int_t *);
extern void FC_GLOBAL(cswap, CSWAP)(int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zswap, ZSWAP)(int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *);
}

void sswap(int_t n, float *x, int_t incx, float *y, int_t incy) {
    LabeledSection0();

    FC_GLOBAL(sswap, SSWAP)(&n, x, &incx, y, &incy);
}

void dswap(int_t n, double *x, int_t incx, double *y, int_t incy) {
    LabeledSection0();

    FC_GLOBAL(dswap, DSWAP)(&n, x, &incx, y, &incy);
}

void cswap(int_t n, std::complex<float> *x, int_t incx, std::complex<float> *y, int_t incy) {
    LabeledSection0();

    FC_GLOBAL(cswap, CSWAP)(&n, x, &incx, y, &incy);
}

void zswap(int_t n, std::complex<double> *x, int_t incx, std::complex<double> *y, int_t incy) {
    LabeledSection0();

    FC_GLOBAL(zswap, ZSWAP)(&n, x, &incx, y, &incy);
}

} // namespace einsums::blas::vendor
