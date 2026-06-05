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
extern int_t FC_GLOBAL(isamax, ISAMAX)(int_t *, float const *, int_t *);
extern int_t FC_GLOBAL(idamax, IDAMAX)(int_t *, double const *, int_t *);
extern int_t FC_GLOBAL(icamax, ICAMAX)(int_t *, std::complex<float> const *, int_t *);
extern int_t FC_GLOBAL(izamax, IZAMAX)(int_t *, std::complex<double> const *, int_t *);
}

auto isamax(int_t n, float const *x, int_t incx) -> int_t {
    LabeledSection0();

    return FC_GLOBAL(isamax, ISAMAX)(&n, x, &incx) - 1;
}

auto idamax(int_t n, double const *x, int_t incx) -> int_t {
    LabeledSection0();

    return FC_GLOBAL(idamax, IDAMAX)(&n, x, &incx) - 1;
}

auto icamax(int_t n, std::complex<float> const *x, int_t incx) -> int_t {
    LabeledSection0();

    return FC_GLOBAL(icamax, ICAMAX)(&n, x, &incx) - 1;
}

auto izamax(int_t n, std::complex<double> const *x, int_t incx) -> int_t {
    LabeledSection0();

    return FC_GLOBAL(izamax, IZAMAX)(&n, x, &incx) - 1;
}

} // namespace einsums::blas::vendor
