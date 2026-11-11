//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ScalarTypes.hpp
/// @brief What a graph asks of an element type enum: whether it is complex, real or double
///        precision, whether the executors can dispatch on it, and how many bytes one element takes.
///
/// Kept apart from ScalarDispatch.hpp so that a pass asking one of these questions does not pull in
/// the dispatch machinery.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <complex>
#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// Whether @p t is a complex type.
[[nodiscard]] constexpr bool is_complex(packed_gemm::ScalarType t) noexcept {
    return t == packed_gemm::ScalarType::Complex64 || t == packed_gemm::ScalarType::Complex128;
}

/// Whether @p t is a real type. Unknown is neither real nor complex.
[[nodiscard]] constexpr bool is_real(packed_gemm::ScalarType t) noexcept {
    return t == packed_gemm::ScalarType::Float32 || t == packed_gemm::ScalarType::Float64;
}

/// Whether @p t is double precision, real or complex.
[[nodiscard]] constexpr bool is_double_precision(packed_gemm::ScalarType t) noexcept {
    return t == packed_gemm::ScalarType::Float64 || t == packed_gemm::ScalarType::Complex128;
}

/// Whether @ref dispatch_scalar_type can dispatch on @p t, which is every type but Unknown.
[[nodiscard]] constexpr bool is_dispatchable(packed_gemm::ScalarType t) noexcept {
    return is_real(t) || is_complex(t);
}

/// Bytes per element of @p t, or 0 for Unknown.
[[nodiscard]] constexpr std::size_t element_bytes(packed_gemm::ScalarType t) noexcept {
    switch (t) {
    case packed_gemm::ScalarType::Float32:
        return sizeof(float);
    case packed_gemm::ScalarType::Float64:
        return sizeof(double);
    case packed_gemm::ScalarType::Complex64:
        return sizeof(std::complex<float>);
    case packed_gemm::ScalarType::Complex128:
        return sizeof(std::complex<double>);
    case packed_gemm::ScalarType::Unknown:
        break;
    }
    return 0;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
