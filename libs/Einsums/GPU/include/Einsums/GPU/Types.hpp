//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/GPU/Platform.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

// Include vendor type headers
#if defined(EINSUMS_HAVE_CUDA)
#    include <cuComplex.h>
#    include <cuda_fp16.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_complex.h>
#endif

namespace einsums::gpu {

// ===========================================================================
// Complex type mapping: std::complex<T> → vendor GPU complex type.
// BLAS/Solver calls on GPU require vendor-specific complex types.
// ===========================================================================

// NOLINTBEGIN
template <typename T>
struct DeviceComplexType {
    using type = T;
};

#if defined(EINSUMS_HAVE_CUDA)
template <>
struct DeviceComplexType<std::complex<float>> {
    using type = cuFloatComplex;
};
template <>
struct DeviceComplexType<std::complex<double>> {
    using type = cuDoubleComplex;
};
#elif defined(EINSUMS_HAVE_HIP)
template <>
struct DeviceComplexType<std::complex<float>> {
    using type = hipFloatComplex;
};
template <>
struct DeviceComplexType<std::complex<double>> {
    using type = hipDoubleComplex;
};
#endif
// NOLINTEND

/// Alias: device_complex_t<std::complex<float>> → cuFloatComplex (CUDA) or hipFloatComplex (HIP)
template <typename T>
using device_complex_t = typename DeviceComplexType<T>::type; // NOLINT(readability-identifier-naming)

// ===========================================================================
// BLAS operation enum: vendor-independent.
// ===========================================================================

enum class Operation : std::uint8_t { None, Transpose, ConjTranspose };

// ===========================================================================
// Reduced-precision type aliases for Ozaki pass.
// ===========================================================================

// NOLINTBEGIN
#if defined(EINSUMS_HAVE_CUDA)
using half_t     = __half;   ///< CUDA FP16
using bfloat16_t = uint16_t; ///< CUDA BFloat16 storage (interpret via cublasLt)
using fp8_t      = uint8_t;  ///< FP8 E4M3 storage (interpret via cublasLt)
#elif defined(EINSUMS_HAVE_HIP)
using half_t     = uint16_t; ///< HIP FP16 (storage, cast to _Float16 for ops)
using bfloat16_t = uint16_t; ///< HIP BFloat16 storage
using fp8_t      = uint8_t;  ///< FP8 storage (vendor-specific interpretation)
#elif defined(EINSUMS_HAVE_MPS)
using half_t     = __fp16;  ///< MPS FP16 (ARM native __fp16)
using bfloat16_t = __bf16;  ///< MPS BFloat16 (ARM native __bf16, macOS 14+)
using fp8_t      = uint8_t; ///< FP8 not supported on MPS
#else // Mock
using half_t     = uint16_t; ///< Mock: raw 16-bit storage
using bfloat16_t = uint16_t; ///< Mock: raw 16-bit storage
using fp8_t      = uint8_t;  ///< Mock: raw 8-bit storage
#endif
// NOLINTEND

// ===========================================================================
// Helper: convert Operation enum to vendor-specific type.
// ===========================================================================

#if defined(EINSUMS_HAVE_CUDA)
inline cublasOperation_t to_vendor_op(Operation op) {
    switch (op) {
    case Operation::None:
        return CUBLAS_OP_N;
    case Operation::Transpose:
        return CUBLAS_OP_T;
    case Operation::ConjTranspose:
        return CUBLAS_OP_C;
    }
    return CUBLAS_OP_N;
}
#elif defined(EINSUMS_HAVE_HIP)
inline hipblasOperation_t to_vendor_op(Operation op) {
    switch (op) {
    case Operation::None:
        return HIPBLAS_OP_N;
    case Operation::Transpose:
        return HIPBLAS_OP_T;
    case Operation::ConjTranspose:
        return HIPBLAS_OP_C;
    }
    return HIPBLAS_OP_N;
}
#else
inline int to_vendor_op(Operation) {
    return 0;
}
#endif

/// Convert a char ('n'/'t'/'c') to Operation enum.
inline Operation char_to_op(char c) {
    switch (c) {
    case 'n':
    case 'N':
        return Operation::None;
    case 't':
    case 'T':
        return Operation::Transpose;
    case 'c':
    case 'C':
        return Operation::ConjTranspose;
    default:
        return Operation::None;
    }
}

} // namespace einsums::gpu
