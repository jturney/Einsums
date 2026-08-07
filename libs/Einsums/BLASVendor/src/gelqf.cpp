//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(sgelqf, SGELQF)(int_t *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgelqf, DGELQF)(int_t *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgelqf, CGELQF)(int_t *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *,
                                      int_t *, int_t *);
extern void FC_GLOBAL(zgelqf, ZGELQF)(int_t *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, std::complex<double> *,
                                      int_t *, int_t *);
}

#define GELQF(Type, lc, uc)                                                                                                                \
    auto lc##gelqf(int_t m, int_t n, Type *a, int_t lda, Type *tau)->int_t {                                                               \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##gelqf, UC##GELQF)(&m, &n, a, &lda, tau, &work_query, &lwork, &info);                                                 \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gelqf, UC##GELQF)(&m, &n, a, &lda, tau, work.data(), &lwork, &info);                                                 \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

#define GELQF_complex(Type, lc, uc)                                                                                                        \
    auto lc##gelqf(int_t m, int_t n, Type *a, int_t lda, Type *tau)->int_t {                                                               \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##gelqf, UC##GELQF)(&m, &n, a, &lda, tau, &work_query, &lwork, &info);                                                 \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gelqf, UC##GELQF)(&m, &n, a, &lda, tau, work.data(), &lwork, &info);                                                 \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

GELQF(double, d, D);
GELQF(float, s, S);
GELQF_complex(std::complex<double>, z, Z);
GELQF_complex(std::complex<float>, c, C);

EINSUMS_NAMESPACE_END(blas::vendor)