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
extern void FC_GLOBAL(sorglq, SORGLQ)(int_t *, int_t *, int_t *, float *, int_t *, float const *, float const *, int_t *, int_t *);
extern void FC_GLOBAL(dorglq, DORGLQ)(int_t *, int_t *, int_t *, double *, int_t *, double const *, double const *, int_t *, int_t *);
extern void FC_GLOBAL(cunglq, CUNGLQ)(int_t *, int_t *, int_t *, std::complex<float> *, int_t *, std::complex<float> const *,
                                      std::complex<float> const *, int_t *, int_t *);
extern void FC_GLOBAL(zunglq, ZUNGLQ)(int_t *, int_t *, int_t *, std::complex<double> *, int_t *, std::complex<double> const *,
                                      std::complex<double> const *, int_t *, int_t *);
}

#define ORGLQ(Type, lc, uc)                                                                                                                \
    auto lc##orglq(int_t m, int_t n, int_t k, Type *a, int_t lda, const Type *tau)->int_t {                                                \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query{};                                                                                                                \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("orglq warning: lda < m, lda = {}, n = {}", lda, m);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##orglq, UC##ORGLQ)(&m, &n, &k, a, &lda, tau, &work_query, &lwork, &info);                                             \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##orglq, UC##ORGLQ)(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);                                             \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

ORGLQ(double, d, D);
ORGLQ(float, s, S);

#define UNGLQ(Type, lc, uc)                                                                                                                \
    auto lc##unglq(int_t m, int_t n, int_t k, Type *a, int_t lda, const Type *tau)->int_t {                                                \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query{};                                                                                                                \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("unglq warning: lda < m, lda = {}, n = {}", lda, m);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##unglq, UC##UNGLQ)(&m, &n, &k, a, &lda, tau, &work_query, &lwork, &info);                                             \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##unglq, UC##UNGLQ)(&m, &n, &k, a, &lda, tau, work.data(), &lwork, &info);                                             \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNGLQ(std::complex<float>, c, C);
UNGLQ(std::complex<double>, z, Z);

} // namespace einsums::blas::vendor