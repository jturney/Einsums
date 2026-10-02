//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(sgetri, SGETRI)(int_t *, float *, int_t *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgetri, DGETRI)(int_t *, double *, int_t *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgetri, CGETRI)(int_t *, std::complex<float> *, int_t *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zgetri, ZGETRI)(int_t *, std::complex<double> *, int_t *, int_t *, std::complex<double> *, int_t *, int_t *);
}

auto sgetri(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t {
    WAGGLE_ZONE_FUNC();

    // A 0x0 inversion is a no-op; OpenBLAS's getri hangs on it.
    if (n == 0) {
        return 0;
    }

    int_t               info{0};
    int_t               lwork = n * 64;
    BufferVector<float> work(lwork);
    FC_GLOBAL(sgetri, SGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t {
    WAGGLE_ZONE_FUNC();

    // A 0x0 inversion is a no-op; OpenBLAS's getri hangs on it.
    if (n == 0) {
        return 0;
    }

    int_t                info{0};
    int_t                lwork = n * 64;
    BufferVector<double> work(lwork);
    FC_GLOBAL(dgetri, DGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t {
    WAGGLE_ZONE_FUNC();

    // A 0x0 inversion is a no-op; OpenBLAS's getri hangs on it.
    if (n == 0) {
        return 0;
    }

    int_t                             info{0};
    int_t                             lwork = n * 64;
    BufferVector<std::complex<float>> work(lwork);
    FC_GLOBAL(cgetri, CGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto zgetri(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t {
    WAGGLE_ZONE_FUNC();

    // A 0x0 inversion is a no-op; OpenBLAS's getri hangs on it.
    if (n == 0) {
        return 0;
    }

    int_t                              info{0};
    int_t                              lwork = n * 64;
    BufferVector<std::complex<double>> work(lwork);
    FC_GLOBAL(zgetri, ZGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

EINSUMS_NAMESPACE_END(blas::vendor)