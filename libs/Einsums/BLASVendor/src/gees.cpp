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
extern void FC_GLOBAL(dgees, DGEES)(char *, char *, int_t (*)(double *, double *), int_t *, double *, int_t *, int_t *, double *, double *,
                                    double *, int_t *, double *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgees, SGEES)(char *, char *, int_t (*)(float *, float *), int_t *, float *, int_t *, int_t *, float *, float *,
                                    float *, int_t *, float *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(cgees, CGEES)(char *, char *, int_t (*)(double *, double *), int_t *, std::complex<float> *, int_t *, int_t *,
                                    std::complex<float> *, std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *,
                                    int_t *);
extern void FC_GLOBAL(zgees, ZGEES)(char *, char *, int_t (*)(double *, double *), int_t *, std::complex<double> *, int_t *, int_t *,
                                    std::complex<double> *, std::complex<double> *, int_t *, std::complex<double> *, int_t *, double *,
                                    int_t *, int_t *);
}

#define GEES(Type, lc, UC)                                                                                                                 \
    auto lc##gees(char jobvs, int_t n, Type *a, int_t lda, int_t *sdim, Type *wr, Type *wi, Type *vs, int_t ldvs)->int_t {                 \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t  info  = 0;                                                                                                                  \
        int_t  lwork = -1;                                                                                                                 \
        int_t *bwork = nullptr;                                                                                                            \
                                                                                                                                           \
        Type work_query;                                                                                                                   \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("gees warning: lda < n, lda = {}, n = {}", lda, n);                                                           \
        }                                                                                                                                  \
        if (ldvs < n) {                                                                                                                    \
            EINSUMS_LOG_WARN("gees warning: ldvs < n, ldvs = {}, n = {}", ldvs, n);                                                        \
        }                                                                                                                                  \
                                                                                                                                           \
        char sort = 'N';                                                                                                                   \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a, &lda, sdim, wr, wi, vs, &ldvs, &work_query, &lwork, bwork, &info);                                 \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        /* Allocate memory for work array */                                                                                               \
        BufferVector<Type> work(lwork);                                                                                                    \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gees, UC##GEES)(&jobvs, &sort, nullptr, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work.data(), &lwork, bwork, &info);    \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

GEES(double, d, D);
GEES(float, s, S);

#undef GEES
#define GEES(Type, lc, UC)                                                                                                                 \
    auto lc##gees(char jobvs, int_t n, std::complex<Type> *a, int_t lda, int_t *sdim, std::complex<Type> *w, std::complex<Type> *vs,       \
                  int_t ldvs)                                                                                                              \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t  info  = 0;                                                                                                                  \
        int_t  lwork = -1;                                                                                                                 \
        int_t *bwork = nullptr;                                                                                                            \
                                                                                                                                           \
        std::complex<Type> work_query;                                                                                                     \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("gees warning: lda < n, lda = {}, n = {}", lda, n);                                                           \
        }                                                                                                                                  \
        if (ldvs < n) {                                                                                                                    \
            EINSUMS_LOG_WARN("gees warning: ldvs < n, ldvs = {}, n = {}", ldvs, n);                                                        \
        }                                                                                                                                  \
                                                                                                                                           \
        BufferVector<Type> rwork(n); /* real work */                                                                                       \
        char               sort = 'N';                                                                                                     \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a, &lda, sdim, w, vs, &ldvs, &work_query, &lwork, rwork.data(), bwork, &info);                        \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query.real(); /* Allocate memory for work array */                                                             \
        BufferVector<std::complex<Type>> work(lwork);                                                                                      \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a, &lda, sdim, w, vs, &ldvs, work.data(), &lwork, rwork.data(), bwork, &info);                        \
                                                                                                                                           \
        return info;                                                                                                                       \
    }

GEES(double, z, Z);
GEES(float, c, C);

} // namespace einsums::blas::vendor