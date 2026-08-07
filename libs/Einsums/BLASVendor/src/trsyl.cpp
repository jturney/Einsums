//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(dtrsyl, DTRSYL)(char *, char *, int_t *, int_t *, int_t *, double const *, int_t *, double const *, int_t *, double *,
                                      int_t *, double *, int_t *);
extern void FC_GLOBAL(strsyl, STRSYL)(char *, char *, int_t *, int_t *, int_t *, float const *, int_t *, float const *, int_t *, float *,
                                      int_t *, float *, int_t *);
extern void FC_GLOBAL(ztrsyl, ZTRSYL)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> const *, int_t *,
                                      std::complex<double> const *, int_t *, std::complex<double> *, int_t *, double *, int_t *);
extern void FC_GLOBAL(ctrsyl, CTRSYL)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> const *, int_t *,
                                      std::complex<float> const *, int_t *, std::complex<float> *, int_t *, float *, int_t *);
}

#define TRSYL(Type, lc, uc)                                                                                                                \
    auto lc##trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const Type *a, int_t lda, const Type *b, int_t ldb, Type *c,      \
                   int_t ldc, Type *scale)                                                                                                 \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
        int_t info = 0;                                                                                                                    \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldb < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: ldb < n, ldb = {}, n = {}", ldb, n);                                                          \
        }                                                                                                                                  \
        if (ldc < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: ldc < n, ldc = {}, n = {}", ldc, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##trsyl, UC##TRSYL)                                                                                                    \
        (&trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, scale, &info);                                                          \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("trsyl warning: the {} argument had an illegal value.", print::ordinal(-info));                               \
        }                                                                                                                                  \
        if (info == 1) {                                                                                                                   \
            EINSUMS_LOG_INFO("trsyl warning: The input matrices have common or very close eigenvalues. Pertubed values were used to "      \
                             "solve the equation, but the matrices are unchanged.");                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

#define TRSYL_complex(Type, lc, uc)                                                                                                        \
    auto lc##trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<Type> *a, int_t lda,                           \
                   const std::complex<Type> *b, int_t ldb, std::complex<Type> *c, int_t ldc, Type *scale)                                  \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
        int_t info = 0;                                                                                                                    \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldb < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: ldb < n, ldb = {}, n = {}", ldb, n);                                                          \
        }                                                                                                                                  \
        if (ldc < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("trsyl warning: ldc < n, ldc = {}, n = {}", ldc, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##trsyl, UC##TRSYL)                                                                                                    \
        (&trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, scale, &info);                                                          \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            EINSUMS_LOG_WARN("trsyl warning: the {} argument had an illegal value.", print::ordinal(-info));                               \
        }                                                                                                                                  \
        if (info == 1) {                                                                                                                   \
            EINSUMS_LOG_INFO("trsyl warning: The input matrices have common or very close eigenvalues. Pertubed values were used to "      \
                             "solve the equation, but the matrices are unchanged.");                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

TRSYL(double, d, D);
TRSYL(float, s, S);
TRSYL_complex(double, z, Z);
TRSYL_complex(float, c, C);

EINSUMS_NAMESPACE_END(blas::vendor)