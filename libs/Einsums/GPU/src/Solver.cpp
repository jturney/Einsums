//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/GPU/Solver.hpp>
#include <Einsums/GPU/Stream.hpp>

#if defined(EINSUMS_HAVE_CUDA)
#    include <cusolverDn.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hipsolver/hipsolver.h>
#else
// Mock backend: delegate to CPU BLAS/LAPACK
#    include <Einsums/BLAS.hpp>
#    include <Einsums/BLAS/Types.hpp>
#    include <Einsums/BLASVendor/Vendor.hpp>
#endif

#if !defined(EINSUMS_HAVE_CUDA) && !defined(EINSUMS_HAVE_HIP)
namespace {
using int_t = ::einsums::blas::int_t;
}
#endif

#include <algorithm>
#include <complex>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(gpu::solver)

#if defined(EINSUMS_HAVE_CUDA)
namespace {

cusolverEigMode_t to_eig_mode(char jobz) {
    return (jobz == 'V' || jobz == 'v') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
}

cublasFillMode_t to_fill_mode(char uplo) {
    return (uplo == 'U' || uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

/// Device scratch with a destructor, so a throwing gpu_solver_catch in the
/// middle of a factorization does not leak the workspace.
struct DeviceScratch {
    void *ptr{nullptr};

    explicit DeviceScratch(std::size_t bytes) {
        if (bytes == 0) {
            return;
        }
        auto result = gpu::device_malloc(bytes);
        if (!result) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "cuSOLVER workspace allocation failed: {}", result.error().message);
        }
        ptr = result.value();
    }
    ~DeviceScratch() { gpu::device_free(ptr); }

    DeviceScratch(DeviceScratch const &)            = delete;
    DeviceScratch &operator=(DeviceScratch const &) = delete;

    // Movable so a helper can build one and hand it back; the moved-from
    // object must not free the buffer it no longer owns.
    DeviceScratch(DeviceScratch &&other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceScratch &operator=(DeviceScratch &&other) noexcept {
        if (this != &other) {
            gpu::device_free(ptr);
            ptr       = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    template <typename T>
    T *as() const {
        return static_cast<T *>(ptr);
    }
};

/// cuSOLVER reports LAPACK's `info` through a device int. Read it back, since
/// every entry point here returns it to a host caller.
int read_dev_info(DeviceScratch const &info) {
    int host_info = 0;
    gpu::memcpy_device_to_host(&host_info, info.ptr, sizeof(int));
    return host_info;
}

/// Copy a device pivot array (cuSOLVER uses int) into the caller's int64_t
/// buffer.
///
/// Note the asymmetry in this API: the matrices are device pointers but `ipiv`
/// is host memory - see the tests in libs/Einsums/GPU/tests/unit/Runtime.cpp,
/// which pass a stack array. So the pivots have to come back across the bus.
void pivots_to_host(DeviceScratch const &dev_ipiv, int64_t *ipiv, int64_t count) {
    if (count <= 0) {
        return;
    }
    std::vector<int> host32(static_cast<std::size_t>(count));
    gpu::memcpy_device_to_host(host32.data(), dev_ipiv.ptr, static_cast<std::size_t>(count) * sizeof(int));
    for (int64_t i = 0; i < count; ++i) {
        ipiv[i] = host32[static_cast<std::size_t>(i)];
    }
}

/// Upload a host int64_t pivot array as the int array cuSOLVER expects.
DeviceScratch pivots_to_device(int64_t const *ipiv, int64_t count) {
    DeviceScratch    dev(static_cast<std::size_t>(count) * sizeof(int));
    std::vector<int> host32(static_cast<std::size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        host32[static_cast<std::size_t>(i)] = static_cast<int>(ipiv[i]);
    }
    gpu::memcpy_host_to_device(dev.ptr, host32.data(), static_cast<std::size_t>(count) * sizeof(int));
    return dev;
}

} // namespace
#endif // EINSUMS_HAVE_CUDA

// ===========================================================================
// syev: Symmetric eigenvalue decomposition
// ===========================================================================

template <>
EINSUMS_EXPORT int syev<float>(char jobz, char uplo, int64_t n, float *A, int64_t lda, float *W) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto       handle = get_solver_handle();
    auto const mode   = to_eig_mode(jobz);
    auto const fill   = to_fill_mode(uplo);
    int        lwork  = 0;
    gpu_solver_catch(cusolverDnSsyevd_bufferSize(handle, mode, fill, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(float));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(
        cusolverDnSsyevd(handle, mode, fill, static_cast<int>(n), A, static_cast<int>(lda), W, work.as<float>(), lwork, info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::syev<float>");
#else
    return static_cast<int>(::einsums::blas::vendor::ssyevd(jobz, uplo, static_cast<int_t>(n), A, static_cast<int_t>(lda), W));
#endif
}

template <>
EINSUMS_EXPORT int syev<double>(char jobz, char uplo, int64_t n, double *A, int64_t lda, double *W) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto       handle = get_solver_handle();
    auto const mode   = to_eig_mode(jobz);
    auto const fill   = to_fill_mode(uplo);
    int        lwork  = 0;
    gpu_solver_catch(cusolverDnDsyevd_bufferSize(handle, mode, fill, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(double));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(
        cusolverDnDsyevd(handle, mode, fill, static_cast<int>(n), A, static_cast<int>(lda), W, work.as<double>(), lwork, info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::syev<double>");
#else
    return static_cast<int>(::einsums::blas::vendor::dsyevd(jobz, uplo, static_cast<int_t>(n), A, static_cast<int_t>(lda), W));
#endif
}

// ===========================================================================
// heev: Hermitian eigenvalue decomposition (complex)
// ===========================================================================

template <>
EINSUMS_EXPORT int heev<float>(char jobz, char uplo, int64_t n, std::complex<float> *A, int64_t lda, float *W) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto       handle = get_solver_handle();
    auto const mode   = to_eig_mode(jobz);
    auto const fill   = to_fill_mode(uplo);
    auto      *a_dev  = reinterpret_cast<cuComplex *>(A);
    int        lwork  = 0;
    gpu_solver_catch(cusolverDnCheevd_bufferSize(handle, mode, fill, static_cast<int>(n), a_dev, static_cast<int>(lda), W, &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(cuComplex));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnCheevd(handle, mode, fill, static_cast<int>(n), a_dev, static_cast<int>(lda), W, work.as<cuComplex>(), lwork,
                                      info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::heev<float>");
#else
    std::complex<float> lwork_query;
    std::vector<float>  rwork(std::max(int64_t(1), 3 * n - 2));
    int info = ::einsums::blas::vendor::cheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork_query, -1, rwork.data());
    int lwork = static_cast<int>(lwork_query.real());
    std::vector<std::complex<float>> work(lwork);
    info = ::einsums::blas::vendor::cheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, work.data(), lwork, rwork.data());
    return info;
#endif
}

template <>
EINSUMS_EXPORT int heev<double>(char jobz, char uplo, int64_t n, std::complex<double> *A, int64_t lda, double *W) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto       handle = get_solver_handle();
    auto const mode   = to_eig_mode(jobz);
    auto const fill   = to_fill_mode(uplo);
    auto      *a_dev  = reinterpret_cast<cuDoubleComplex *>(A);
    int        lwork  = 0;
    gpu_solver_catch(cusolverDnZheevd_bufferSize(handle, mode, fill, static_cast<int>(n), a_dev, static_cast<int>(lda), W, &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(cuDoubleComplex));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnZheevd(handle, mode, fill, static_cast<int>(n), a_dev, static_cast<int>(lda), W, work.as<cuDoubleComplex>(),
                                      lwork, info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::heev<double>");
#else
    std::complex<double> lwork_query;
    std::vector<double>  rwork(std::max(int64_t(1), 3 * n - 2));
    int info = ::einsums::blas::vendor::zheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, &lwork_query, -1, rwork.data());
    int lwork = static_cast<int>(lwork_query.real());
    std::vector<std::complex<double>> work(lwork);
    info = ::einsums::blas::vendor::zheev(jobz, uplo, static_cast<int>(n), A, static_cast<int>(lda), W, work.data(), lwork, rwork.data());
    return info;
#endif
}

// ===========================================================================
// gesv: Linear solve AX = B
// ===========================================================================

template <>
EINSUMS_EXPORT int gesv<float>(int64_t n, int64_t nrhs, float *A, int64_t lda, int64_t *ipiv, float *B, int64_t ldb) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER has no single gesv for this shape; factor then solve. The pivots
    // stay on the device between the two calls and are handed back to the
    // caller's host buffer at the end.
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnSgetrf_bufferSize(handle, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(float));
    DeviceScratch const dev_ipiv(static_cast<std::size_t>(n) * sizeof(int));
    DeviceScratch const info(sizeof(int));

    gpu_solver_catch(cusolverDnSgetrf(handle, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda), work.as<float>(),
                                      dev_ipiv.as<int>(), info.as<int>()));
    int const factor_info = read_dev_info(info);
    if (factor_info != 0) {
        pivots_to_host(dev_ipiv, ipiv, n);
        return factor_info;
    }

    gpu_solver_catch(cusolverDnSgetrs(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda),
                                      dev_ipiv.as<int>(), B, static_cast<int>(ldb), info.as<int>()));
    pivots_to_host(dev_ipiv, ipiv, n);
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::gesv<float>");
#else
    // CPU gesv uses int* for ipiv, need to convert
    std::vector<int_t> ipiv32(n);
    int info = ::einsums::blas::vendor::sgesv(static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda), ipiv32.data(), B,
                                              static_cast<int>(ldb));
    for (int64_t i = 0; i < n; ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

template <>
EINSUMS_EXPORT int gesv<double>(int64_t n, int64_t nrhs, double *A, int64_t lda, int64_t *ipiv, double *B, int64_t ldb) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER has no single gesv for this shape; factor then solve. The pivots
    // stay on the device between the two calls and are handed back to the
    // caller's host buffer at the end.
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnDgetrf_bufferSize(handle, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(double));
    DeviceScratch const dev_ipiv(static_cast<std::size_t>(n) * sizeof(int));
    DeviceScratch const info(sizeof(int));

    gpu_solver_catch(cusolverDnDgetrf(handle, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda), work.as<double>(),
                                      dev_ipiv.as<int>(), info.as<int>()));
    int const factor_info = read_dev_info(info);
    if (factor_info != 0) {
        pivots_to_host(dev_ipiv, ipiv, n);
        return factor_info;
    }

    gpu_solver_catch(cusolverDnDgetrs(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda),
                                      dev_ipiv.as<int>(), B, static_cast<int>(ldb), info.as<int>()));
    pivots_to_host(dev_ipiv, ipiv, n);
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::gesv<double>");
#else
    std::vector<int_t> ipiv32(n);
    int info = ::einsums::blas::vendor::dgesv(static_cast<int>(n), static_cast<int>(nrhs), A, static_cast<int>(lda), ipiv32.data(), B,
                                              static_cast<int>(ldb));
    for (int64_t i = 0; i < n; ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

// ===========================================================================
// getrf: LU factorization
// ===========================================================================

template <>
EINSUMS_EXPORT int getrf<float>(int64_t m, int64_t n, float *A, int64_t lda, int64_t *ipiv) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnSgetrf_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(float));
    int64_t const       npiv = std::min(m, n);
    DeviceScratch const dev_ipiv(static_cast<std::size_t>(npiv) * sizeof(int));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnSgetrf(handle, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), work.as<float>(),
                                      dev_ipiv.as<int>(), info.as<int>()));
    pivots_to_host(dev_ipiv, ipiv, npiv);
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::getrf<float>");
#else
    std::vector<int_t> ipiv32(std::min(m, n));
    int info = ::einsums::blas::vendor::sgetrf(static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    for (int64_t i = 0; i < static_cast<int64_t>(ipiv32.size()); ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

template <>
EINSUMS_EXPORT int getrf<double>(int64_t m, int64_t n, double *A, int64_t lda, int64_t *ipiv) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnDgetrf_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(double));
    int64_t const       npiv = std::min(m, n);
    DeviceScratch const dev_ipiv(static_cast<std::size_t>(npiv) * sizeof(int));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnDgetrf(handle, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), work.as<double>(),
                                      dev_ipiv.as<int>(), info.as<int>()));
    pivots_to_host(dev_ipiv, ipiv, npiv);
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::getrf<double>");
#else
    std::vector<int_t> ipiv32(std::min(m, n));
    int info = ::einsums::blas::vendor::dgetrf(static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    for (int64_t i = 0; i < static_cast<int64_t>(ipiv32.size()); ++i)
        ipiv[i] = ipiv32[i];
    return info;
#endif
}

// ===========================================================================
// getri: Inverse from LU factorization
// ===========================================================================

template <>
EINSUMS_EXPORT int getri<float>(int64_t n, float *A, int64_t lda, int64_t const *ipiv) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER has no getri. The inverse is the solution of A X = I against the
    // LU factors the caller already computed, which is exactly what getrs does.
    auto                handle   = get_solver_handle();
    DeviceScratch const dev_ipiv = pivots_to_device(ipiv, n);
    DeviceScratch const info(sizeof(int));

    auto const          nn = static_cast<std::size_t>(n);
    DeviceScratch const rhs(nn * nn * sizeof(float));
    std::vector<float>  identity(nn * nn, float(0));
    for (std::size_t i = 0; i < nn; ++i) {
        identity[i + i * nn] = float(1);
    }
    gpu::memcpy_host_to_device(rhs.ptr, identity.data(), nn * nn * sizeof(float));

    gpu_solver_catch(cusolverDnSgetrs(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda),
                                      dev_ipiv.as<int>(), rhs.as<float>(), static_cast<int>(n), info.as<int>()));
    int const solve_info = read_dev_info(info);
    if (solve_info != 0) {
        return solve_info;
    }

    // Write the inverse back over A. Column by column, because the result is
    // packed with leading dimension n while A's is lda.
    for (int64_t col = 0; col < n; ++col) {
        gpu::memcpy_device_to_device(A + col * lda, rhs.as<float>() + col * n, static_cast<std::size_t>(n) * sizeof(float));
    }
    return 0;
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::getri<float>");
#else
    std::vector<int_t> ipiv32(n);
    for (int64_t i = 0; i < n; ++i)
        ipiv32[i] = static_cast<int>(ipiv[i]);
    int info = ::einsums::blas::vendor::sgetri(static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    return info;
#endif
}

template <>
EINSUMS_EXPORT int getri<double>(int64_t n, double *A, int64_t lda, int64_t const *ipiv) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER has no getri. The inverse is the solution of A X = I against the
    // LU factors the caller already computed, which is exactly what getrs does.
    auto                handle   = get_solver_handle();
    DeviceScratch const dev_ipiv = pivots_to_device(ipiv, n);
    DeviceScratch const info(sizeof(int));

    auto const          nn = static_cast<std::size_t>(n);
    DeviceScratch const rhs(nn * nn * sizeof(double));
    std::vector<double> identity(nn * nn, double(0));
    for (std::size_t i = 0; i < nn; ++i) {
        identity[i + i * nn] = double(1);
    }
    gpu::memcpy_host_to_device(rhs.ptr, identity.data(), nn * nn * sizeof(double));

    gpu_solver_catch(cusolverDnDgetrs(handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n), A, static_cast<int>(lda),
                                      dev_ipiv.as<int>(), rhs.as<double>(), static_cast<int>(n), info.as<int>()));
    int const solve_info = read_dev_info(info);
    if (solve_info != 0) {
        return solve_info;
    }

    // Write the inverse back over A. Column by column, because the result is
    // packed with leading dimension n while A's is lda.
    for (int64_t col = 0; col < n; ++col) {
        gpu::memcpy_device_to_device(A + col * lda, rhs.as<double>() + col * n, static_cast<std::size_t>(n) * sizeof(double));
    }
    return 0;
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::getri<double>");
#else
    std::vector<int_t> ipiv32(n);
    for (int64_t i = 0; i < n; ++i)
        ipiv32[i] = static_cast<int>(ipiv[i]);
    int info = ::einsums::blas::vendor::dgetri(static_cast<int>(n), A, static_cast<int>(lda), ipiv32.data());
    return info;
#endif
}

// ===========================================================================
// gesvd: Singular value decomposition
// ===========================================================================

template <>
EINSUMS_EXPORT int gesvd<float>(char jobu, char jobvt, int64_t m, int64_t n, float *A, int64_t lda, float *S, float *U, int64_t ldu,
                                float *VT, int64_t ldvt) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER's gesvd requires m >= n and offers no transposed form, so the
    // wide case has no device path. Raising beats returning a wrong answer.
    if (m < n) {
        not_implemented("gpu::solver::gesvd<float> with m < n (cuSOLVER requires m >= n)");
    }
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnSgesvd_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(float));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnSgesvd(handle, jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                      static_cast<int>(ldu), VT, static_cast<int>(ldvt), work.as<float>(), lwork, nullptr, info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::gesvd<float>");
#else
    std::vector<float> superb(std::min(m, n));
    int info = ::einsums::blas::vendor::sgesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                               static_cast<int>(ldu), VT, static_cast<int>(ldvt), superb.data());
    return info;
#endif
}

template <>
EINSUMS_EXPORT int gesvd<double>(char jobu, char jobvt, int64_t m, int64_t n, double *A, int64_t lda, double *S, double *U, int64_t ldu,
                                 double *VT, int64_t ldvt) {
    EINSUMS_GPU_MOCK_KERNEL_SCOPE;
#if defined(EINSUMS_HAVE_CUDA)
    // cuSOLVER's gesvd requires m >= n and offers no transposed form, so the
    // wide case has no device path. Raising beats returning a wrong answer.
    if (m < n) {
        not_implemented("gpu::solver::gesvd<double> with m < n (cuSOLVER requires m >= n)");
    }
    auto handle = get_solver_handle();
    int  lwork  = 0;
    gpu_solver_catch(cusolverDnDgesvd_bufferSize(handle, static_cast<int>(m), static_cast<int>(n), &lwork));
    DeviceScratch const work(static_cast<std::size_t>(lwork) * sizeof(double));
    DeviceScratch const info(sizeof(int));
    gpu_solver_catch(cusolverDnDgesvd(handle, jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                      static_cast<int>(ldu), VT, static_cast<int>(ldvt), work.as<double>(), lwork, nullptr,
                                      info.as<int>()));
    return read_dev_info(info);
#elif defined(EINSUMS_HAVE_HIP)
    not_implemented("gpu::solver::gesvd<double>");
#else
    std::vector<double> superb(std::min(m, n));
    int info = ::einsums::blas::vendor::dgesvd(jobu, jobvt, static_cast<int>(m), static_cast<int>(n), A, static_cast<int>(lda), S, U,
                                               static_cast<int>(ldu), VT, static_cast<int>(ldvt), superb.data());
    return info;
#endif
}

EINSUMS_NAMESPACE_END(gpu::solver)
