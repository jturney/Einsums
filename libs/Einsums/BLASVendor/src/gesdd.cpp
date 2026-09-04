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
extern void FC_GLOBAL(dgesdd, DGESDD)(char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *, double *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgesdd, SGESDD)(char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *, float *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zgesdd, ZGESDD)(char *, int_t *, int_t *, std::complex<double> *, int_t *, double *, std::complex<double> *, int_t *,
                                      std::complex<double> *, int_t *, std::complex<double> *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgesdd, CGESDD)(char *, int_t *, int_t *, std::complex<float> *, int_t *, float *, std::complex<float> *, int_t *,
                                      std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *, int_t *);
}

#define GESDD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesdd(char jobz, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt) -> int_t {    \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        /* Query optimal working array(s) */                                                                                               \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &work_query, &lwork, nullptr, &info);                                              \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("The {} parameter to gesdd was invalid! 1: (jobz) {}, 2: (m) {}, 3: (n) {}, 5: (lda) {}, 8: (ldu) {}, 10: "   \
                             "(ldvt) {}, 12: (lwork) {}.",                                                                                 \
                             print::ordinal(-info), jobz, m, n, lda, ldu, ldvt, lwork);                                                    \
        } else if (info > 0) {                                                                                                             \
            EINSUMS_LOG_WARN("gesdd lapack routine failed. info {}", info);                                                                \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int)work_query;                                                                                                           \
                                                                                                                                           \
        /* Allocate work array */                                                                                                          \
        BufferVector<Type> work(lwork);                                                                                                    \
                                                                                                                                           \
        /* Allocate iwork array */                                                                                                         \
        BufferVector<int_t> iwork(8 * std::min(m, n));                                                                                     \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(), &lwork, iwork.data(), &info);                                         \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("The {} parameter to gesdd was invalid! 1: (jobz) {}, 2: (m) {}, 3: (n) {}, 5: (lda) {}, 8: (ldu) {}, 10: "   \
                             "(ldvt) {}, 12: (lwork) {}.",                                                                                 \
                             print::ordinal(-info), jobz, m, n, lda, ldu, ldvt, lwork);                                                    \
        } else if (info > 0) {                                                                                                             \
            EINSUMS_LOG_WARN("gesdd lapack routine failed. info {}", info);                                                                \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

#define GESDD_complex(Type, lc, UC)                                                                                                        \
    auto lc##gesdd(char jobz, int_t m, int_t n, std::complex<Type> *a, int_t lda, Type *s, std::complex<Type> *u, int_t ldu,               \
                   std::complex<Type> *vt, int_t ldvt) -> int_t {                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t                            info{0};                                                                                          \
        int_t                            lwork{-1};                                                                                        \
        size_t                           lrwork;                                                                                           \
        std::complex<Type>               work_query;                                                                                       \
        BufferVector<Type>               rwork;                                                                                            \
        BufferVector<std::complex<Type>> work;                                                                                             \
        BufferVector<int_t>              iwork;                                                                                            \
                                                                                                                                           \
        if (lsame(jobz, 'n')) {                                                                                                            \
            lrwork = std::max(int_t{1}, 7 * std::min(m, n));                                                                               \
        } else {                                                                                                                           \
            lrwork = (size_t)std::max(int_t{1},                                                                                            \
                                      std::min(m, n) * std::max(5 * std::min(m, n) + 7, 2 * std::max(m, n) + 2 * std::min(m, n) + 1));     \
        }                                                                                                                                  \
                                                                                                                                           \
        iwork.resize(std::max(int_t{1}, 8 * std::min(m, n)));                                                                              \
        rwork.resize(lrwork);                                                                                                              \
                                                                                                                                           \
        /* Query optimal working array(s) */                                                                                               \
        FC_GLOBAL(lc##gesdd, UC##GESDD)                                                                                                    \
        (&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &work_query, &lwork, rwork.data(), iwork.data(), &info);                           \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("The {} parameter to gesdd was invalid! 1: (jobz) {}, 2: (m) {}, 3: (n) {}, 5: (lda) {}, 8: (ldu) {}, 10: "   \
                             "(ldvt) {}, 12: (lwork) {}.",                                                                                 \
                             print::ordinal(-info), jobz, m, n, lda, ldu, ldvt, lwork);                                                    \
        } else if (info > 0) {                                                                                                             \
            EINSUMS_LOG_WARN("gesdd lapack routine failed. info {}", info);                                                                \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        lwork = (int)(work_query.real());                                                                                                  \
                                                                                                                                           \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lc##gesdd, UC##GESDD)                                                                                                    \
        (&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work.data(), &lwork, rwork.data(), iwork.data(), &info);                           \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("The {} parameter to gesdd was invalid! 1: (jobz) {}, 2: (m) {}, 3: (n) {}, 5: (lda) {}, 8: (ldu) {}, 10: "   \
                             "(ldvt) {}, 12: (lwork) {}.",                                                                                 \
                             print::ordinal(-info), jobz, m, n, lda, ldu, ldvt, lwork);                                                    \
        } else if (info > 0) {                                                                                                             \
            EINSUMS_LOG_WARN("gesdd lapack routine failed. info {}", info);                                                                \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESDD(double, d, D);
GESDD(float, s, S);
GESDD_complex(float, c, C);
GESDD_complex(double, z, Z);

EINSUMS_NAMESPACE_END(blas::vendor)