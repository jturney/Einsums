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
extern void FC_GLOBAL(scopy, SCOPY)(int_t *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dcopy, DCOPY)(int_t *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(ccopy, CCOPY)(int_t *, std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zcopy, ZCOPY)(int_t *, std::complex<double> const *, int_t *, std::complex<double> *, int_t *);
}

// Each routine below quick-returns on an empty copy rather than handing it to
// the vendor. Reference BLAS returns on n < 1 before validating anything, but
// Accelerate validates the increments first and ABORTS THE PROCESS on a zero
// one -- and an empty copy legitimately carries zero increments, because
// `impl_scalar_copy` broadcasts a scalar with inc_x == 0 and an empty tensor's
// own increment is 0 as well. Creating a (0, 5) tensor was enough to kill the
// interpreter on macOS/Accelerate, while passing everywhere else.
//
// Same reasoning as the m/n guards in gemm.cpp and gemv.cpp.

void scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection("scopy");
    if (n <= 0)
        return;
    FC_GLOBAL(scopy, SCOPY)(&n, x, &inc_x, y, &inc_y);
}

void dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection("dcopy");
    if (n <= 0)
        return;
    FC_GLOBAL(dcopy, DCOPY)(&n, x, &inc_x, y, &inc_y);
}

void ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection("ccopy");
    if (n <= 0)
        return;
    FC_GLOBAL(ccopy, CCOPY)(&n, x, &inc_x, y, &inc_y);
}

void zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection("zcopy");
    if (n <= 0)
        return;
    FC_GLOBAL(zcopy, ZCOPY)(&n, x, &inc_x, y, &inc_y);
}

EINSUMS_NAMESPACE_END(blas::vendor)