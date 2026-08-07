//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern float  FC_GLOBAL(sasum, SASUM)(int_t *n, float const *x, int_t *incx);
extern double FC_GLOBAL(dasum, DASUM)(int_t *n, double const *x, int_t *incx);
extern float  FC_GLOBAL(scasum, SCASUM)(int_t *n, std::complex<float> const *x, int_t *incx);
extern double FC_GLOBAL(dzasum, DZASUM)(int_t *n, std::complex<double> const *x, int_t *incx);
extern float  FC_GLOBAL(scsum1, SCSUM1)(int_t *n, std::complex<float> const *x, int_t *incx);
extern double FC_GLOBAL(dzsum1, DZSUM1)(int_t *n, std::complex<double> const *x, int_t *incx);
}

float sasum(int_t n, float const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(sasum, SASUM)(&n, x, &incx);
}

double dasum(int_t n, double const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(dasum, DASUM)(&n, x, &incx);
}

float scasum(int_t n, std::complex<float> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(scasum, SCASUM)(&n, x, &incx);
}

double dzasum(int_t n, std::complex<double> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(dzasum, DZASUM)(&n, x, &incx);
}

float scsum1(int_t n, std::complex<float> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(scsum1, SCSUM1)(&n, x, &incx);
}

double dzsum1(int_t n, std::complex<double> const *x, int_t incx) {
    LabeledSection0();

    return FC_GLOBAL(dzsum1, DZSUM1)(&n, x, &incx);
}

EINSUMS_NAMESPACE_END(blas::vendor)