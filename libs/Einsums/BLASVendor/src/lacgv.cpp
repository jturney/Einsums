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
extern void FC_GLOBAL(clacgv, CLACGV)(int_t *n, std::complex<float> *x, int_t *incx);
extern void FC_GLOBAL(zlacgv, ZLACGV)(int_t *n, std::complex<double> *x, int_t *incx);
}

// Same n guard as copy.cpp, scal.cpp and axpy.cpp: Accelerate validates the
// increment before the length and ABORTS THE PROCESS on a zero one, and an
// empty tensor legitimately carries a zero increment. Conjugating an empty
// complex tensor reaches here.

void clacgv(int_t n, std::complex<float> *x, int_t incx) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(clacgv, CLACGV)(&n, x, &incx);
}

void zlacgv(int_t n, std::complex<double> *x, int_t incx) {
    LabeledSection0();

    if (n <= 0)
        return;

    FC_GLOBAL(zlacgv, ZLACGV)(&n, x, &incx);
}

EINSUMS_NAMESPACE_END(blas::vendor)