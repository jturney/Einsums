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
extern void FC_GLOBAL(sgeev, SGEEV)(char *, char *, int_t *, float *, int_t *, float *, float *, float *, int_t *, float *, int_t *,
                                    float *, int_t *, int_t *);
extern void FC_GLOBAL(dgeev, DGEEV)(char *, char *, int_t *, double *, int_t *, double *, double *, double *, int_t *, double *, int_t *,
                                    double *, int_t *, int_t *);
extern void FC_GLOBAL(cgeev, CGEEV)(char *, char *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *,
                                    int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *);
extern void FC_GLOBAL(zgeev, ZGEEV)(char *, char *, int_t *, std::complex<double> *, int_t *, std::complex<double> *,
                                    std::complex<double> *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                    double *, int_t *);
}

#define GEEV_complex(Type, lc, UC)                                                                                                         \
    auto lc##geev(char jobvl, char jobvr, int_t n, std::complex<Type> *a, int_t lda, std::complex<Type> *w, std::complex<Type> *vl,        \
                  int_t ldvl, std::complex<Type> *vr, int_t ldvr)                                                                          \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t                            info  = 0;                                                                                        \
        int_t                            lwork = -1;                                                                                       \
        BufferVector<Type>               rwork;                                                                                            \
        BufferVector<std::complex<Type>> work;                                                                                             \
        std::complex<Type>               work_query;                                                                                       \
                                                                                                                                           \
        /* Allocate memory for working array(s) */                                                                                         \
        rwork.resize(std::max(int_t{1}, 2 * n));                                                                                           \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                           \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            EINSUMS_LOG_WARN("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                          \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            EINSUMS_LOG_WARN("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, &work_query, &lwork, rwork.data(), &info);                                  \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (int_t)work_query.real();                                                                                                  \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work.data(), &lwork, rwork.data(), &info);                                  \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEEV_complex(float, c, C);
GEEV_complex(double, z, Z);

#define GEEV(Type, lc, uc)                                                                                                                 \
    auto lc##geev(char jobvl, char jobvr, int_t n, Type *a, int_t lda, std::complex<Type> *w, Type *vl, int_t ldvl, Type *vr, int_t ldvr)  \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t              info  = 0;                                                                                                      \
        int_t              lwork = -1;                                                                                                     \
        BufferVector<Type> work;                                                                                                           \
        Type               work_query;                                                                                                     \
                                                                                                                                           \
        BufferVector<Type> wr(n), wi(n);                                                                                                   \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                           \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            EINSUMS_LOG_WARN("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                          \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            EINSUMS_LOG_WARN("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda, wr.data(), wi.data(), vl, &ldvl, vr, &ldvr, &work_query, &lwork, &info);                             \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (int_t)work_query;                                                                                                         \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda, wr.data(), wi.data(), vl, &ldvl, vr, &ldvr, work.data(), &lwork, &info);                             \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Pack wr and wi into w */                                                                                                        \
        for (int_t i = 0; i < n; i++) {                                                                                                    \
            w[i] = std::complex<Type>(wr[i], wi[i]);                                                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    }                                                                                                                                      \
    /**/

GEEV(float, s, S);
GEEV(double, d, D);

EINSUMS_NAMESPACE_END(blas::vendor)