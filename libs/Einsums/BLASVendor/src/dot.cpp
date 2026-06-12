//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

// Real dot products use Fortran BLAS (no calling convention issues)
EINSUMS_DISABLE_WARNING_PUSH
EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
extern "C" {
extern float  FC_GLOBAL(sdot, SDOT)(int_t *, float const *, int_t *, float const *, int_t *);
extern double FC_GLOBAL(ddot, DDOT)(int_t *, double const *, int_t *, double const *, int_t *);

// Complex dot products use CBLAS interface to avoid the Fortran complex return
// convention issue (by-value vs hidden-first-pointer varies across vendors).
// The CBLAS _sub variants always write the result to a pointer, which is unambiguous.
extern void cblas_cdotu_sub(int_t n, void const *x, int_t incx, void const *y, int_t incy, void *result);
extern void cblas_zdotu_sub(int_t n, void const *x, int_t incx, void const *y, int_t incy, void *result);
extern void cblas_cdotc_sub(int_t n, void const *x, int_t incx, void const *y, int_t incy, void *result);
extern void cblas_zdotc_sub(int_t n, void const *x, int_t incx, void const *y, int_t incy, void *result);
}
EINSUMS_DISABLE_WARNING_POP

auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    LabeledSection0();

    return FC_GLOBAL(sdot, SDOT)(&n, x, &incx, y, &incy);
}

auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    LabeledSection0();

    return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy);
}

auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;
    cblas_cdotu_sub(n, x, incx, y, incy, &result);
    return result;
}

auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;
    cblas_zdotu_sub(n, x, incx, y, incy, &result);
    return result;
}

auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;
    cblas_cdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;
    cblas_zdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

} // namespace einsums::blas::vendor