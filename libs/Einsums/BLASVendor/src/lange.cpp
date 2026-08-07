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
extern float  FC_GLOBAL(slange, SLANGE)(char const *, int_t *, int_t *, float const *, int_t *, float *);
extern double FC_GLOBAL(dlange, DLANGE)(char const *, int_t *, int_t *, double const *, int_t *, double *);
extern float  FC_GLOBAL(clange, CLANGE)(char const *, int_t *, int_t *, std::complex<float> const *, int_t *, float *);
extern double FC_GLOBAL(zlange, ZLANGE)(char const *, int_t *, int_t *, std::complex<double> const *, int_t *, double *);
}

auto slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(slange, SLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(dlange, DLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(clange, CLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(zlange, ZLANGE)(&norm_type, &m, &n, A, &lda, work);
}

EINSUMS_NAMESPACE_END(blas::vendor)