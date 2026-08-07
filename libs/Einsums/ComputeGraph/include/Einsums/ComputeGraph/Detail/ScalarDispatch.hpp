//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Invoke @p f with a value-initialized tag of the C++ element type named
 *        by @p dtype and return its result.
 *
 * Collapses the four-arm `switch (dtype) { Float32 -> f(float{}); ... }` that the
 * type-erased graph executors and several passes each open-coded. @p f is called
 * as `f(float{})`, `f(double{})`, `f(std::complex<float>{})`, or
 * `f(std::complex<double>{})`; use `decltype(tag)` inside a generic lambda to
 * recover the type. Throws `std::invalid_argument` on ScalarType::Unknown (the
 * same behavior as the `default:` arms this replaces).
 */
template <typename F>
decltype(auto) dispatch_scalar_type(packed_gemm::ScalarType dtype, F &&f) {
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return f(float{});
    case packed_gemm::ScalarType::Float64:
        return f(double{});
    case packed_gemm::ScalarType::Complex64:
        return f(std::complex<float>{});
    case packed_gemm::ScalarType::Complex128:
        return f(std::complex<double>{});
    case packed_gemm::ScalarType::Unknown:
        break;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_scalar_type: unknown ScalarType");
}

/**
 * @brief Invoke @p f with a `std::integral_constant<size_t, Rank>` tag for the
 *        runtime @p rank (1..4) and return its result.
 *
 * Collapses the compile-time rank `switch (rank) { case 1: ...Tensor<T,1>...; }`
 * that the binary and unary type-erased dispatchers each open-coded. Recover the
 * rank inside a generic lambda with `decltype(tag)::value`. Throws
 * `std::invalid_argument` for ranks outside 1..4.
 */
template <typename F>
decltype(auto) dispatch_by_rank(std::size_t rank, F &&f) {
    switch (rank) {
    case 1:
        return f(std::integral_constant<std::size_t, 1>{});
    case 2:
        return f(std::integral_constant<std::size_t, 2>{});
    case 3:
        return f(std::integral_constant<std::size_t, 3>{});
    case 4:
        return f(std::integral_constant<std::size_t, 4>{});
    default:
        break;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_by_rank: unsupported rank {}", rank);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
