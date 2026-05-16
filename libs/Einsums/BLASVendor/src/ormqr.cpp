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
extern void FC_GLOBAL(sormqr, SORMQR)(char *, char *, int_t *, int_t *, int_t *, float const *, int_t *, float const *, float *, int_t *,
                                      float *, int_t *, int_t *);
extern void FC_GLOBAL(dormqr, DORMQR)(char *, char *, int_t *, int_t *, int_t *, double const *, int_t *, double const *, double *, int_t *,
                                      double *, int_t *, int_t *);
extern void FC_GLOBAL(cunmqr, CUNMQR)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> const *, int_t *,
                                      std::complex<float> const *, std::complex<float> *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zunmqr, ZUNMQR)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> const *, int_t *,
                                      std::complex<double> const *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                      int_t *);
}

#define ORMQR(Type, lc, uc)                                                                                                                \
    auto lc##ormqr(char side, char trans, int_t m, int_t n, int_t k, Type const *a, int_t lda, Type const *tau, Type *c, int_t ldc)        \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##ormqr, UC##ORMQR)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &work_query, &lwork, &info);                     \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function */                                                                                                         \
        FC_GLOBAL(lc##ormqr, UC##ORMQR)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);                     \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

ORMQR(float, s, S);
ORMQR(double, d, D);

#define UNMQR(Type, lc, uc)                                                                                                                \
    auto lc##unmqr(char side, char trans, int_t m, int_t n, int_t k, Type const *a, int_t lda, Type const *tau, Type *c, int_t ldc)        \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##unmqr, UC##UNMQR)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, &work_query, &lwork, &info);                     \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function */                                                                                                         \
        FC_GLOBAL(lc##unmqr, UC##UNMQR)(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work.data(), &lwork, &info);                     \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNMQR(std::complex<float>, c, C);
UNMQR(std::complex<double>, z, Z);

} // namespace einsums::blas::vendor
