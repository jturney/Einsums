//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(sorgqr, SORGQR)(int_t *, int_t *, int_t *, float *, int_t *, float const *, float const *, int_t *, int_t *);
extern void FC_GLOBAL(dorgqr, DORGQR)(int_t *, int_t *, int_t *, double *, int_t *, double const *, double const *, int_t *, int_t *);
extern void FC_GLOBAL(cungqr, CUNGQR)(int_t *, int_t *, int_t *, std::complex<float> *, int_t *, std::complex<float> const *,
                                      std::complex<float> const *, int_t *, int_t *);
extern void FC_GLOBAL(zungqr, ZUNGQR)(int_t *, int_t *, int_t *, std::complex<double> *, int_t *, std::complex<double> const *,
                                      std::complex<double> const *, int_t *, int_t *);
}

#define ORGQR(Type, lc, uc)                                                                                                                \
    auto lc##orgqr(int_t m, int_t n, int_t k, Type *a, int_t lda, const Type *tau)->int_t {                                                \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query{};                                                                                                                \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("orgqr warning: lda < n, lda = {}, n = {}", lda, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a, &lda, tau, &work_query, &lwork, &info);                                             \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);                                             \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

ORGQR(double, d, D);
ORGQR(float, s, S);

#define UNGQR(Type, lc, uc)                                                                                                                \
    auto lc##ungqr(int_t m, int_t n, int_t k, Type *a, int_t lda, const Type *tau)->int_t {                                                \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query{};                                                                                                                \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("ungqr warning: lda < n, lda = {}, n = {}", lda, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a, &lda, tau, &work_query, &lwork, &info);                                             \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);                                             \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNGQR(std::complex<float>, c, C);
UNGQR(std::complex<double>, z, Z);

} // namespace einsums::blas::vendor