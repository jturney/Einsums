//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/blas/vendor.hpp>
#include <einsums/print.hpp>
#include <einsums/profile/section.hpp>

#if defined(EINSUMS_HAVE_MKL)
typedef void (*XerblaEntry)(const char *Name, const int *Num, const intLen);
extern "C" {
XerblaEntry mkl_set_xerbla(XerblaEntry xerbla);
}
#endif

#if !defined(FC_SYMBOL)
#    define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) NAME##_
#endif

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::blas::vendor)

EINSUMS_DISABLE_WARNING_PUSH
// EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
extern "C" {
extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int_t *, int_t *, int_t *, float *, const float *, int_t *, const float *, int_t *,
                                    float *, float *, int_t *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int_t *, int_t *, int_t *, double *, const double *, int_t *, const double *, int_t *,
                                    double *, double *, int_t *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> *, const std::complex<float> *, int_t *,
                                    const std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> *, const std::complex<double> *,
                                    int_t *, const std::complex<double> *, int_t *, std::complex<double> *, std::complex<double> *,
                                    int_t *);

extern void FC_GLOBAL(sgemv, SGEMV)(char *, int_t *, int_t *, float *, const float *, int_t *, const float *, int_t *, float *, float *,
                                    int_t *);
extern void FC_GLOBAL(dgemv, DGEMV)(char *, int_t *, int_t *, double *, const double *, int_t *, const double *, int_t *, double *,
                                    double *, int_t *);
extern void FC_GLOBAL(cgemv, CGEMV)(char *, int_t *, int_t *, std::complex<float> *, const std::complex<float> *, int_t *,
                                    const std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemv, ZGEMV)(char *, int_t *, int_t *, std::complex<double> *, const std::complex<double> *, int_t *,
                                    const std::complex<double> *, int_t *, std::complex<double> *, std::complex<double> *, int_t *);

extern void FC_GLOBAL(cheev, CHEEV)(char *job, char *uplo, int_t *n, std::complex<float> *a, int_t *lda, float *w,
                                    std::complex<float> *work, int_t *lwork, float *rwork, int_t *info);
extern void FC_GLOBAL(zheev, ZHEEV)(char *job, char *uplo, int_t *n, std::complex<double> *a, int_t *lda, double *w,
                                    std::complex<double> *work, int_t *lwork, double *rwork, int_t *info);

extern void FC_GLOBAL(ssyev, SSYEV)(char *, char *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *);

extern void FC_GLOBAL(sgeev, SGEEV)(char *, char *, int_t *, float *, int_t *, float *, float *, float *, int_t *, float *, int_t *,
                                    float *, int_t *, int_t *);
extern void FC_GLOBAL(dgeev, DGEEV)(char *, char *, int_t *, double *, int_t *, double *, double *, double *, int_t *, double *, int_t *,
                                    double *, int_t *, int_t *);
extern void FC_GLOBAL(cgeev, CGEEV)(char *, char *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *,
                                    int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *);
extern void FC_GLOBAL(zgeev, ZGEEV)(char *, char *, int_t *, std::complex<double> *, int_t *, std::complex<double> *,
                                    std::complex<double> *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                    double *, int_t *);

extern void FC_GLOBAL(sgesv, SGESV)(int_t *, int_t *, float *, int_t *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgesv, DGESV)(int_t *, int_t *, double *, int_t *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgesv, CGESV)(int_t *, int_t *, std::complex<float> *, int_t *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zgesv, ZGESV)(int_t *, int_t *, std::complex<double> *, int_t *, int_t *, std::complex<double> *, int_t *, int_t *);

extern void FC_GLOBAL(sscal, SSCAL)(int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(dscal, DSCAL)(int_t *, double *, double *, int_t *);
extern void FC_GLOBAL(cscal, CSCAL)(int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zscal, ZSCAL)(int_t *, std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(csscal, CSSCAL)(int_t *, float *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zdscal, ZDSCAL)(int_t *, double *, std::complex<double> *, int_t *);

extern float                FC_GLOBAL(sdot, SDOT)(int_t *, const float *, int_t *, const float *, int_t *);
extern double               FC_GLOBAL(ddot, DDOT)(int_t *, const double *, int_t *, const double *, int_t *);
extern std::complex<float>  FC_GLOBAL(cdotc, CDOTC)(int_t *, const std::complex<float> *, int_t *, const std::complex<float> *, int_t *);
extern std::complex<double> FC_GLOBAL(zdotc, ZDOTC)(int_t *, const std::complex<double> *, int_t *, const std::complex<double> *, int_t *);
// MKL seems to have a different function signature than openblas.
// extern std::complex<float>  FC_GLOBAL(cdotu, CDOTU)(int_t *, const std::complex<float> *, int_t *, const std::complex<float> *,
//                                                    int_t *);
// extern std::complex<double> FC_GLOBAL(zdotu, ZDOTU)(int_t *, const std::complex<double> *, int_t *, const std::complex<double> *,
//                                                     int_t *);

extern void FC_GLOBAL(saxpy, SAXPY)(int_t *, float *, const float *, int_t *, float *, int_t *);
extern void FC_GLOBAL(daxpy, DAXPY)(int_t *, double *, const double *, int_t *, double *, int_t *);
extern void FC_GLOBAL(caxpy, CAXPY)(int_t *, std::complex<float> *, const std::complex<float> *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zaxpy, ZAXPY)(int_t *, std::complex<double> *, const std::complex<double> *, int_t *, std::complex<double> *,
                                    int_t *);

extern void FC_GLOBAL(sger, DGER)(int_t *, int_t *, float *, const float *, int_t *, const float *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dger, DGER)(int_t *, int_t *, double *, const double *, int_t *, const double *, int_t *, double *, int_t *);
extern void FC_GLOBAL(cgeru, CGERU)(int_t *, int_t *, std::complex<float> *, const std::complex<float> *, int_t *,
                                    const std::complex<float> *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgeru, ZGERU)(int_t *, int_t *, std::complex<double> *, const std::complex<double> *, int_t *,
                                    const std::complex<double> *, int_t *, std::complex<double> *, int_t *);

extern void FC_GLOBAL(sgetrf, SGETRF)(int_t *, int_t *, float *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(dgetrf, DGETRF)(int_t *, int_t *, double *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(cgetrf, CGETRF)(int_t *, int_t *, std::complex<float> *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zgetrf, ZGETRF)(int_t *, int_t *, std::complex<double> *, int_t *, int_t *, int_t *);

extern void FC_GLOBAL(sgetri, SGETRI)(int_t *, float *, int_t *, int_t *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgetri, DGETRI)(int_t *, double *, int_t *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgetri, CGETRI)(int_t *, std::complex<float> *, int_t *, int_t *, std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zgetri, ZGETRI)(int_t *, std::complex<double> *, int_t *, int_t *, std::complex<double> *, int_t *, int_t *);

// According to my Xcode 15.3 macOS 14.4 SDK:
// The Accelerate clapack.h header does use double for Xlange's. However, that interface is deprecated according to the headers.
// The "new lapack" Accelerate header does use the following return types. For now, we're going to leave it as is.
// If it becomes an issue then we'll need to do something about it.
extern float  FC_GLOBAL(slange, SLANGE)(const char *, int_t *, int_t *, const float *, int_t *, float *);
extern double FC_GLOBAL(dlange, DLANGE)(const char *, int_t *, int_t *, const double *, int_t *, double *);
extern float  FC_GLOBAL(clange, CLANGE)(const char *, int_t *, int_t *, const std::complex<float> *, int_t *, float *);
extern double FC_GLOBAL(zlange, ZLANGE)(const char *, int_t *, int_t *, const std::complex<double> *, int_t *, double *);

extern void FC_GLOBAL(slassq, SLASSQ)(int_t *n, const float *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(dlassq, DLASSQ)(int_t *n, const double *x, int_t *incx, double *scale, double *sumsq);
extern void FC_GLOBAL(classq, CLASSQ)(int_t *n, const std::complex<float> *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(zlassq, ZLASSQ)(int_t *n, const std::complex<double> *x, int_t *incx, double *scale, double *sumsq);

extern void FC_GLOBAL(dgesvd, DGESVD)(char *, char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *,
                                      double *, int_t *, int_t *);
extern void FC_GLOBAL(sgesvd, SGESVD)(char *, char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *,
                                      float *, int_t *, int_t *);

extern void FC_GLOBAL(dgesdd, DGESDD)(char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *, double *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgesdd, SGESDD)(char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *, float *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zgesdd, ZGESDD)(char *, int_t *, int_t *, std::complex<double> *, int_t *, double *, std::complex<double> *, int_t *,
                                      std::complex<double> *, int_t *, std::complex<double> *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgesdd, CGESDD)(char *, int_t *, int_t *, std::complex<float> *, int_t *, float *, std::complex<float> *, int_t *,
                                      std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *, int_t *);

extern void FC_GLOBAL(dgees, DGEES)(char *, char *, int_t (*)(double *, double *), int_t *, double *, int_t *, int_t *, double *, double *,
                                    double *, int_t *, double *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgees, SGEES)(char *, char *, int_t (*)(float *, float *), int_t *, float *, int_t *, int_t *, float *, float *,
                                    float *, int_t *, float *, int_t *, int_t *, int_t *);

extern void FC_GLOBAL(dtrsyl, DTRSYL)(char *, char *, int_t *, int_t *, int_t *, const double *, int_t *, const double *, int_t *, double *,
                                      int_t *, double *, int_t *);
extern void FC_GLOBAL(strsyl, STRSYL)(char *, char *, int_t *, int_t *, int_t *, const float *, int_t *, const float *, int_t *, float *,
                                      int_t *, float *, int_t *);

extern void FC_GLOBAL(sorgqr, SORGQR)(int_t *, int_t *, int_t *, float *, int_t *, const float *, const float *, int_t *, int_t *);
extern void FC_GLOBAL(dorgqr, DORGQR)(int_t *, int_t *, int_t *, double *, int_t *, const double *, const double *, int_t *, int_t *);
extern void FC_GLOBAL(cungqr, CUNGQR)(int_t *, int_t *, int_t *, std::complex<float> *, int_t *, const std::complex<float> *,
                                      const std::complex<float> *, int_t *, int_t *);
extern void FC_GLOBAL(zungqr, ZUNGQR)(int_t *, int_t *, int_t *, std::complex<double> *, int_t *, const std::complex<double> *,
                                      const std::complex<double> *, int_t *, int_t *);

extern void FC_GLOBAL(sgeqrf, SGEQRF)(int_t *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dgeqrf, DGEQRF)(int_t *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgeqrf, CGEQRF)(int_t *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *,
                                      int_t *, int_t *);
extern void FC_GLOBAL(zgeqrf, ZGEQRF)(int_t *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, std::complex<double> *,
                                      int_t *, int_t *);
}

namespace {
extern "C" void xerbla(const char *srname, const int *info, const int /*len*/) {
    if (*info == 1001) {
        println_abort("BLAS/LAPACK: Incompatible optional parameters on entry to {}", srname);
    } else if (*info == 1000 || *info == 1089) {
        println_abort("BLAS/LAPACK: Insufficient workspace available in function {}.", srname);
    } else if (*info < 0) {
        println_abort("BLAS/LAPACK: Condition {} detected in function {}.", -(*info), srname);
    } else {
        println_abort("BLAS/LAPACK: The value of parameter {} is invalid in function call to {}.", *info, srname);
    }
}

} // namespace

void initialize() {
#if defined(EINSUMS_HAVE_MKL)
    mkl_set_xerbla(&xerbla);
#endif
}

void finalize() {
}

inline bool lsame(char ca, char cb) {
    return std::tolower(ca) == std::tolower(cb);
}

enum OrderMajor { Column, Row };

// OrderMajor indicates the order of the input matrix. C is Row, Fortran is Column
template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const T *in, Integer ldin, T *out, Integer ldout) {
    Integer i, j, x, y;

    if (in == nullptr || out == nullptr) {
        return;
    }

    if constexpr (Order == OrderMajor::Column) {
        x = n;
        y = m;
    } else if constexpr (Order == OrderMajor::Row) {
        x = m;
        y = n;
    } else {
        static_assert(Order == OrderMajor::Column || Order == OrderMajor::Row, "Invalid OrderMajor");
    }

    // Look into replacing this with hptt or librett
    for (i = 0; i < std::min(y, ldin); i++) {
        for (j = 0; j < std::min(x, ldout); j++) {
            out[(size_t)i * ldout + j] = in[(size_t)j * ldin + i];
        }
    }
}

template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const std::vector<T> &in, Integer ldin, T *out, Integer ldout) {
    transpose<Order>(m, n, in.data(), ldin, out, ldout);
}

template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const T *in, Integer ldin, std::vector<T> &out, Integer ldout) {
    transpose<Order>(m, n, in, ldin, out.data(), ldout);
}

void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, const float *a, int_t lda, const float *b, int_t ldb,
           float beta, float *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(sgemm, SGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, const double *a, int_t lda, const double *b, int_t ldb,
           double beta, double *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(dgemm, DGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, const std::complex<float> *a, int_t lda,
           const std::complex<float> *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(cgemm, CGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(zgemm, ZGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void sgemv(char transa, int_t m, int_t n, float alpha, const float *a, int_t lda, const float *x, int_t incx, float beta, float *y,
           int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(sgemv, SGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void dgemv(char transa, int_t m, int_t n, double alpha, const double *a, int_t lda, const double *x, int_t incx, double beta, double *y,
           int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(dgemv, DGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *a, int_t lda, const std::complex<float> *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(cgemv, CGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(zgemv, ZGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(ssyev, SSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
           float *rwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cheev, CHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}
auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zheev, ZHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}

auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(sgesv, SGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dgesv, DGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cgesv, CGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zgesv, ZGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

void sscal(int_t n, float alpha, float *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(sscal, SSCAL)(&n, &alpha, vec, &inc);
}

void dscal(int_t n, double alpha, double *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc);
}

void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(cscal, CSCAL)(&n, &alpha, vec, &inc);
}

void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(zscal, ZSCAL)(&n, &alpha, vec, &inc);
}

void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(csscal, CSSCAL)(&n, &alpha, vec, &inc);
}

void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    LabeledSection0();

    FC_GLOBAL(zdscal, ZDSCAL)(&n, &alpha, vec, &inc);
}

auto sdot(int_t n, const float *x, int_t incx, const float *y, int_t incy) -> float {
    LabeledSection0();

    return FC_GLOBAL(sdot, SDOT)(&n, x, &incx, y, &incy);
}

auto ddot(int_t n, const double *x, int_t incx, const double *y, int_t incy) -> double {
    LabeledSection0();

    return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy);
}

// We implement the cdotu as the default for cdot.
auto cdot(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float> {
    LabeledSection0();

    // Since MKL does not conform to the netlib standard, we need to use the following code.
    std::complex<float> result{0.0F, 0.0F};
    for (int_t i = 0; i < n; ++i) {
        result += x[static_cast<ptrdiff_t>(i * incx)] * y[static_cast<ptrdiff_t>(i * incy)];
    }
    return result;
}

// We implement the zdotu as the default for cdot.
auto zdot(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double> {
    LabeledSection0();

    // Since MKL does not conform to the netlib standard, we need to use the following code.
    std::complex<double> result{0.0, 0.0};
    for (int_t i = 0; i < n; ++i) {
        result += x[static_cast<ptrdiff_t>(i * incx)] * y[static_cast<ptrdiff_t>(i * incy)];
    }
    return result;
}

auto cdotc(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float> {
    LabeledSection0();

    return FC_GLOBAL(cdotc, CDOTC)(&n, x, &incx, y, &incy);
}

auto zdotc(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double> {
    LabeledSection0();

    return FC_GLOBAL(zdotc, ZDOTC)(&n, x, &incx, y, &incy);
}

void saxpy(int_t n, float alpha_x, const float *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection0();

    FC_GLOBAL(saxpy, SAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(int_t n, double alpha_x, const double *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection0();

    FC_GLOBAL(daxpy, DAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection0();

    FC_GLOBAL(caxpy, CAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void zaxpy(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection0();

    FC_GLOBAL(zaxpy, ZAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void saxpby(const int_t n, const float a, const float *x, const int_t incx, const float b, float *y, const int_t incy) {
    LabeledSection0();
    sscal(n, b, y, incy);
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(const int_t n, const double a, const double *x, const int_t incx, const double b, double *y, const int_t incy) {
    LabeledSection0();
    dscal(n, b, y, incy);
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(const int_t n, const std::complex<float> a, const std::complex<float> *x, const int_t incx, const std::complex<float> b,
            std::complex<float> *y, const int_t incy) {
    LabeledSection0();
    cscal(n, b, y, incy);
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(const int_t n, const std::complex<double> a, const std::complex<double> *x, const int_t incx, const std::complex<double> b,
            std::complex<double> *y, const int_t incy) {
    LabeledSection0();
    zscal(n, b, y, incy);
    zaxpy(n, a, x, incx, y, incy);
}

namespace {
void ger_parameter_check(int_t m, int_t n, int_t inc_x, int_t inc_y, int_t lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: m ({}) is less than zero.", m));
    }
    if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: n ({}) is less than zero.", n));
    }
    if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_x ({}) is zero.", inc_x));
    }
    if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_y ({}) is zero.", inc_y));
    }
    if (lda < std::max(int_t{1}, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: lda ({}) is less than max(1, n ({})).", lda, n));
    }
}
} // namespace

void sger(int_t m, int_t n, float alpha, const float *x, int_t inc_x, const float *y, int_t inc_y, float *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(sger, SGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void dger(int_t m, int_t n, double alpha, const double *x, int_t inc_x, const double *y, int_t inc_y, double *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(dger, DGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void cger(int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *x, int_t inc_x, const std::complex<float> *y, int_t inc_y,
          std::complex<float> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(cgeru, CGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void zger(int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *x, int_t inc_x, const std::complex<double> *y,
          int_t inc_y, std::complex<double> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(zgeru, ZGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

auto sgetrf(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(sgetrf, SGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(dgetrf, DGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(cgetrf, CGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto zgetrf(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t info{0};
    FC_GLOBAL(zgetrf, ZGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto sgetri(int_t n, float *a, int_t lda, const int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t              info{0};
    int_t              lwork = n * 64;
    std::vector<float> work(lwork);
    FC_GLOBAL(sgetri, SGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(int_t n, double *a, int_t lda, const int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t               info{0};
    int_t               lwork = n * 64;
    std::vector<double> work(lwork);
    FC_GLOBAL(dgetri, DGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(int_t n, std::complex<float> *a, int_t lda, const int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t                            info{0};
    int_t                            lwork = n * 64;
    std::vector<std::complex<float>> work(lwork);
    FC_GLOBAL(cgetri, CGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto zgetri(int_t n, std::complex<double> *a, int_t lda, const int_t *ipiv) -> int_t {
    LabeledSection0();

    int_t                             info{0};
    int_t                             lwork = n * 64;
    std::vector<std::complex<double>> work(lwork);
    FC_GLOBAL(zgetri, ZGETRI)(&n, a, &lda, (int_t *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto slange(char norm_type, int_t m, int_t n, const float *A, int_t lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(slange, SLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto dlange(char norm_type, int_t m, int_t n, const double *A, int_t lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(dlange, DLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto clange(char norm_type, int_t m, int_t n, const std::complex<float> *A, int_t lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(clange, CLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto zlange(char norm_type, int_t m, int_t n, const std::complex<double> *A, int_t lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(zlange, ZLANGE)(&norm_type, &m, &n, A, &lda, work);
}

void slassq(int_t n, const float *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(slassq, SLASSQ)(&n, x, &incx, scale, sumsq);
}

void dlassq(int_t n, const double *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(dlassq, DLASSQ)(&n, x, &incx, scale, sumsq);
}

void classq(int_t n, const std::complex<float> *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(classq, CLASSQ)(&n, x, &incx, scale, sumsq);
}

void zlassq(int_t n, const std::complex<double> *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(zlassq, ZLASSQ)(&n, x, &incx, scale, sumsq);
}

#define GESDD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesdd(char jobz, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt)->int_t {      \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                    \
        int_t ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);                  \
        int_t nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);                 \
                                                                                                                                           \
        int_t             lda_t  = std::max(int_t{1}, m);                                                                                  \
        int_t             ldu_t  = std::max(int_t{1}, nrows_u);                                                                            \
        int_t             ldvt_t = std::max(int_t{1}, nrows_vt);                                                                           \
        std::vector<Type> a_t, u_t, vt_t;                                                                                                  \
                                                                                                                                           \
        /* Check leading dimensions(s) */                                                                                                  \
        if (lda < n) {                                                                                                                     \
            println_warn("gesdd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesdd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < n) {                                                                                                                    \
            println_warn("gesdd warning: ldvt < n, ldvt = {}, n = {}", ldvt, n);                                                           \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) */                                                                                               \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, nullptr, &info);                                        \
        lwork = (int)work_query;                                                                                                           \
                                                                                                                                           \
        /* Allocate memory for temporary arrays(s) */                                                                                      \
        a_t.resize(lda_t *std::max(int_t{1}, n));                                                                                          \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t *std::max(int_t{1}, ncols_u));                                                                                \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate work array */                                                                                                          \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate iwork array */                                                                                                         \
        std::vector<int_t> iwork(8 * std::min(m, n));                                                                                      \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, iwork.data(), &info);        \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

#define GESDD_complex(Type, lc, UC)                                                                                                        \
    auto lc##gesdd(char jobz, int_t m, int_t n, std::complex<Type> *a, int_t lda, Type *s, std::complex<Type> *u, int_t ldu,               \
                   std::complex<Type> *vt, int_t ldvt)                                                                                     \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                    \
        int_t ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);                  \
        int_t nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);                 \
                                                                                                                                           \
        int_t                           lda_t  = std::max(int_t{1}, m);                                                                    \
        int_t                           ldu_t  = std::max(int_t{1}, nrows_u);                                                              \
        int_t                           ldvt_t = std::max(int_t{1}, nrows_vt);                                                             \
        int_t                           info{0};                                                                                           \
        int_t                           lwork{-1};                                                                                         \
        size_t                          lrwork;                                                                                            \
        std::complex<Type>              work_query;                                                                                        \
        std::vector<std::complex<Type>> a_t, u_t, vt_t;                                                                                    \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::vector<int_t>              iwork;                                                                                             \
                                                                                                                                           \
        /* Check leading dimensions(s) */                                                                                                  \
        if (lda < n) {                                                                                                                     \
            println_warn("gesdd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesdd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < n) {                                                                                                                    \
            println_warn("gesdd warning: ldvt < n, ldvt = {}, n = {}", ldvt, n);                                                           \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
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
        (&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, rwork.data(), iwork.data(), &info);                     \
        lwork = (int)(work_query.real());                                                                                                  \
                                                                                                                                           \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        /* Allocate memory for temporary arrays(s) */                                                                                      \
        a_t.resize(lda_t *std::max(int_t{1}, n));                                                                                          \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t *std::max(int_t{1}, ncols_u));                                                                                \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lc##gesdd, UC##GESDD)                                                                                                    \
        (&jobz, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, rwork.data(), iwork.data(),  \
         &info);                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            println_warn("gesdd lapack routine failed. info {}", info);                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESDD(double, d, D);
GESDD(float, s, S);
GESDD_complex(float, c, C);
GESDD_complex(double, z, Z);

#define GESVD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesvd(char jobu, char jobvt, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt,   \
                         Type *superb)                                                                                                     \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info  = 0;                                                                                                                   \
        int_t lwork = -1;                                                                                                                  \
                                                                                                                                           \
        Type  work_query;                                                                                                                  \
        int_t i;                                                                                                                           \
                                                                                                                                           \
        int_t nrows_u  = (lsame(jobu, 'a') || lsame(jobu, 's')) ? m : 1;                                                                   \
        int_t ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                   \
        int_t nrows_vt = lsame(jobvt, 'a') ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                                 \
        int_t ncols_vt = (lsame(jobvt, 'a') || lsame(jobvt, 's')) ? n : 1;                                                                 \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, m);                                                                                              \
        int_t ldu_t  = std::max(int_t{1}, nrows_u);                                                                                        \
        int_t ldvt_t = std::max(int_t{1}, nrows_vt);                                                                                       \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("gesvd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -6;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            println_warn("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                      \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) size */                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, &info);                                         \
        if (info != 0)                                                                                                                     \
            println_abort("gesvd work array size query failed. info {}", info);                                                            \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> u_t, vt_t;                                                                                                       \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            u_t.resize(ldu_t *std::max(int_t{1}, ncols_u));                                                                                \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            vt_t.resize(ldvt_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, &info);              \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            println_abort("gesvd lapack routine failed. info {}", info);                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
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
        int_t lda_t  = std::max(int_t{1}, n);                                                                                              \
        int_t ldvs_t = std::max(int_t{1}, n);                                                                                              \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("gees warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvs < n) {                                                                                                                    \
            println_warn("gees warning: ldvs < n, ldvs = {}, n = {}", ldvs, n);                                                            \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        char sort = 'N';                                                                                                                   \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a, &lda_t, sdim, wr, wi, vs, &ldvs_t, &work_query, &lwork, bwork, &info);                             \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> vs_t;                                                                                                            \
        if (lsame(jobvs, 'v')) {                                                                                                           \
            vs_t.resize(ldvs_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a_t.data(), &lda_t, sdim, wr, wi, vs_t.data(), &ldvs_t, work.data(), &lwork, bwork, &info);           \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvs, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vs_t, ldvs_t, vs, ldvs);                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEES(double, d, D);
GEES(float, s, S);

#define TRSYL(Type, lc, uc)                                                                                                                \
    auto lc##trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const Type *a, int_t lda, const Type *b, int_t ldb, Type *c,      \
                   int_t ldc, Type *scale)                                                                                                 \
        ->int_t {                                                                                                                          \
        int_t info  = 0;                                                                                                                   \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
        int_t ldb_t = std::max(int_t{1}, n);                                                                                               \
        int_t ldc_t = std::max(int_t{1}, m);                                                                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            println_warn("trsyl warning: lda < m, lda = {}, m = {}", lda, m);                                                              \
            return -7;                                                                                                                     \
        }                                                                                                                                  \
        if (ldb < n) {                                                                                                                     \
            println_warn("trsyl warning: ldb < n, ldb = {}, n = {}", ldb, n);                                                              \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldc < n) {                                                                                                                     \
            println_warn("trsyl warning: ldc < n, ldc = {}, n = {}", ldc, n);                                                              \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, m));                                                                               \
        std::vector<Type> b_t(ldb_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> c_t(ldc_t *std::max(int_t{1}, n));                                                                               \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, m, a, lda, a_t, lda_t);                                                                              \
        transpose<OrderMajor::Row>(n, n, b, ldb, b_t, ldb_t);                                                                              \
        transpose<OrderMajor::Row>(m, n, c, ldc, c_t, ldc_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##trsyl, UC##TRSYL)                                                                                                    \
        (&trana, &tranb, &isgn, &m, &n, a_t.data(), &lda_t, b_t.data(), &ldb_t, c_t.data(), &ldc_t, scale, &info);                         \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, c_t, ldc_t, c, ldc);                                                                           \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

TRSYL(double, d, D);
TRSYL(float, s, S);

#define ORGQR(Type, lc, uc)                                                                                                                \
    auto lc##orgqr(int_t m, int_t n, int_t k, Type *a, int_t lda, const Type *tau)->int_t {                                                \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("orgqr warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a, &lda_t, tau, &work_query, &lwork, &info);                                           \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                  \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
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
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("ungqr warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a, &lda_t, tau, &work_query, &lwork, &info);                                           \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                  \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNGQR(std::complex<float>, c, C);
UNGQR(std::complex<double>, z, Z);

#define GEQRF(Type, lc, uc)                                                                                                                \
    auto lc##geqrf(int_t m, int_t n, Type *a, int_t lda, Type *tau)->int_t {                                                               \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geqrf warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a, &lda_t, tau, &work_query, &lwork, &info);                                               \
                                                                                                                                           \
        lwork = (int_t)work_query;                                                                                                         \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                      \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

#define GEQRF_complex(Type, lc, uc)                                                                                                        \
    auto lc##geqrf(int_t m, int_t n, Type *a, int_t lda, Type *tau)->int_t {                                                               \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
                                                                                                                                           \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geqrf warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a, &lda_t, tau, &work_query, &lwork, &info);                                               \
                                                                                                                                           \
        lwork = (int_t)(work_query.real());                                                                                                \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                      \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEQRF(double, d, D);
GEQRF(float, s, S);
GEQRF_complex(std::complex<double>, z, Z);
GEQRF_complex(std::complex<float>, c, C);

#define GEEV_complex(Type, lc, UC)                                                                                                         \
    auto lc##geev(char jobvl, char jobvr, int_t n, std::complex<Type> *a, int_t lda, std::complex<Type> *w, std::complex<Type> *vl,        \
                  int_t ldvl, std::complex<Type> *vr, int_t ldvr)                                                                          \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t                           info  = 0;                                                                                         \
        int_t                           lwork = -1;                                                                                        \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::complex<Type>              work_query;                                                                                        \
                                                                                                                                           \
        /* Allocate memory for working array(s) */                                                                                         \
        rwork.resize(std::max(int_t{1}, 2 * n));                                                                                           \
                                                                                                                                           \
        int_t                           lda_t  = std::max(int_t{1}, n);                                                                    \
        int_t                           ldvl_t = std::max(int_t{1}, n);                                                                    \
        int_t                           ldvr_t = std::max(int_t{1}, n);                                                                    \
        std::vector<std::complex<Type>> a_t;                                                                                               \
        std::vector<std::complex<Type>> vl_t;                                                                                              \
        std::vector<std::complex<Type>> vr_t;                                                                                              \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            println_warn("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                              \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            println_warn("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                              \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda_t, w, vl, &ldvl_t, vr, &ldvr_t, &work_query, &lwork, rwork.data(), &info);                            \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (int_t)work_query.real();                                                                                                  \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t *std::max(int_t{1}, n));                                                                                          \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a_t.data(), &lda_t, w, vl_t.data(), &ldvl_t, vr_t.data(), &ldvr_t, work.data(), &lwork, rwork.data(), &info); \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vl_t, ldvl_t, vl, ldvl);                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vr_t, ldvr_t, vr, ldvr);                                                                   \
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
        int_t             info  = 0;                                                                                                       \
        int_t             lwork = -1;                                                                                                      \
        std::vector<Type> work;                                                                                                            \
        Type              work_query;                                                                                                      \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, n);                                                                                              \
        int_t ldvl_t = std::max(int_t{1}, n);                                                                                              \
        int_t ldvr_t = std::max(int_t{1}, n);                                                                                              \
                                                                                                                                           \
        std::vector<Type> a_t;                                                                                                             \
        std::vector<Type> vl_t;                                                                                                            \
        std::vector<Type> vr_t;                                                                                                            \
        std::vector<Type> wr(n), wi(n);                                                                                                    \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            println_warn("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                              \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            println_warn("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                              \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda_t, wr.data(), wi.data(), vl, &ldvl_t, vr, &ldvr_t, &work_query, &lwork, &info);                       \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (int_t)work_query;                                                                                                         \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t *std::max(int_t{1}, n));                                                                                          \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a_t.data(), &lda_t, wr.data(), wi.data(), vl_t.data(), &ldvl_t, vr_t.data(), &ldvr_t, work.data(), &lwork,    \
         &info);                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vl_t, ldvl_t, vl, ldvl);                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vr_t, ldvr_t, vr, ldvr);                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Pack wr and wi into w */                                                                                                        \
        for (int_t i = 0; i < n; i++) {                                                                                                    \
            w[i] = std::complex<float>(wr[i], wi[i]);                                                                                      \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    }                                                                                                                                      \
    /**/

GEEV(float, s, S);
GEEV(double, d, D);

END_EINSUMS_NAMESPACE_CPP(einsums::blas::vendor)
