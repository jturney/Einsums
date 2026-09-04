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
extern void FC_GLOBAL(dgesvd, DGESVD)(char *, char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *,
                                      double *, int_t *, int_t *);
extern void FC_GLOBAL(sgesvd, SGESVD)(char *, char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *,
                                      float *, int_t *, int_t *);
extern void FC_GLOBAL(zgesvd, ZGESVD)(char *jobu, char *jobvt, int_t *m, int_t *n, std::complex<double> *A, int_t *lda, double *S,
                                      std::complex<double> *U, int_t *ldu, std::complex<double> *Vt, int_t *ldvt,
                                      std::complex<double> *work, int_t *lwork, double *rwork, int_t *info);
extern void FC_GLOBAL(cgesvd, CGESVD)(char *jobu, char *jobvt, int_t *m, int_t *n, std::complex<float> *A, int_t *lda, float *S,
                                      std::complex<float> *U, int_t *ldu, std::complex<float> *Vt, int_t *ldvt, std::complex<float> *work,
                                      int_t *lwork, float *rwork, int_t *info);
}

#define GESVD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesvd(char jobu, char jobvt, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt,   \
                         Type *superb) -> int_t {                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info  = 0;                                                                                                                   \
        int_t lwork = -1;                                                                                                                  \
                                                                                                                                           \
        Type  work_query;                                                                                                                  \
        int_t i;                                                                                                                           \
                                                                                                                                           \
        int_t ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                   \
        int_t ncols_vt = (lsame(jobvt, 'a')) ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("gesvd warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            EINSUMS_LOG_WARN("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                        \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            EINSUMS_LOG_WARN("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) size */                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &work_query, &lwork, &info);                                               \
        if (info != 0)                                                                                                                     \
            return info;                                                                                                                   \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(), &lwork, &info);                                               \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Backup significant data from working arrays into superb */                                                                      \
        for (i = 0; i < std::min(m, n) - 1; i++) {                                                                                         \
            superb[i] = work[i + 1];                                                                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESVD(double, d, D);
GESVD(float, s, S);

#define GESVD_complex(Type, lcletter, UCLETTER)                                                                                            \
    auto lcletter##gesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<Type> *a, int_t lda, Type *s, std::complex<Type> *u,        \
                         int_t ldu, std::complex<Type> *vt, int_t ldvt, std::complex<Type> *superb) -> int_t {                             \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info  = 0;                                                                                                                   \
        int_t lwork = -1;                                                                                                                  \
                                                                                                                                           \
        std::complex<Type> work_query;                                                                                                     \
        int_t              i;                                                                                                              \
                                                                                                                                           \
        int_t ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                   \
        int_t ncols_vt = (lsame(jobvt, 'a')) ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("gesvd warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            EINSUMS_LOG_WARN("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                        \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            EINSUMS_LOG_WARN("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        BufferVector<Type> rwork(5 * std::min(m, n));                                                                                      \
        /* Query optimal working array(s) size */                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &work_query, &lwork, rwork.data(), &info);                                 \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int_t)work_query.real();                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        BufferVector<std::complex<Type>> work(lwork);                                                                                      \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(), &lwork, rwork.data(), &info);                                 \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Backup significant data from working arrays into superb */                                                                      \
        for (i = 0; i < std::min(m, n) - 1; i++) {                                                                                         \
            superb[i] = work[i + 1];                                                                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESVD_complex(double, z, Z);
GESVD_complex(float, c, C);

EINSUMS_NAMESPACE_END(blas::vendor)