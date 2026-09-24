//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file MixedPrecision.hpp
/// @brief The arithmetic rules of a mixed-precision einsum, once at compile time and once at run time.
///
/// An einsum whose operands have different element types forms its products and its sum in one
/// accumulator type R, promoted from A and B: complex if either is complex, at the wider of the two
/// precisions. Both operands are converted to R before they are multiplied, so pairs with no
/// operator* between them (``complex<float> * double``) still contract. The result is then stored
/// in C, narrowing its precision if C is narrower.
///
/// The compile-time form picks the types the generic loop is instantiated with; the run-time form
/// validates a node built from dtype enums, at capture and on load. Their agreement is tested.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <complex>
#include <stdexcept>
#include <string_view>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
struct RealOf {
    using type = T;
};
template <typename T>
struct RealOf<std::complex<T>> {
    using type = T;
};

template <typename T>
inline constexpr bool is_complex_v = false;
template <typename T>
inline constexpr bool is_complex_v<std::complex<T>> = true;

/// The element types an einsum may use.
template <typename T>
concept EinsumElement = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> ||
                        std::is_same_v<T, std::complex<double>>;

/// The type a contraction of A with B accumulates in: complex if either is complex, at the wider of
/// the two real precisions. Unlike BiggestTypeT, which compares sizes and so ranks complex<float>
/// and double as equals, this keeps both the imaginary part and the precision.
template <EinsumElement TA, EinsumElement TB>
struct Promote {
  private:
    using RealA = typename RealOf<TA>::type;
    using RealB = typename RealOf<TB>::type;
    using Real  = std::conditional_t<(sizeof(RealA) >= sizeof(RealB)), RealA, RealB>;

  public:
    using type = std::conditional_t<is_complex_v<TA> || is_complex_v<TB>, std::complex<Real>, Real>;
};

template <EinsumElement TA, EinsumElement TB>
using PromoteT = typename Promote<TA, TB>::type;

/// Whether a value of type @p From can be stored in @p To without losing its imaginary part.
/// Precision may narrow; a complex value may not become real.
template <EinsumElement From, EinsumElement To>
inline constexpr bool storable_v = !is_complex_v<From> || is_complex_v<To>;

constexpr bool is_complex(packed_gemm::ScalarType t) {
    return t == packed_gemm::ScalarType::Complex64 || t == packed_gemm::ScalarType::Complex128;
}

constexpr bool is_double_precision(packed_gemm::ScalarType t) {
    return t == packed_gemm::ScalarType::Float64 || t == packed_gemm::ScalarType::Complex128;
}

/// Run-time mirror of PromoteT. Unknown if either operand's type is unknown.
constexpr packed_gemm::ScalarType promote(packed_gemm::ScalarType a, packed_gemm::ScalarType b) {
    using packed_gemm::ScalarType;
    if (a == ScalarType::Unknown || b == ScalarType::Unknown) {
        return ScalarType::Unknown;
    }
    bool const complex = is_complex(a) || is_complex(b);
    bool const wide    = is_double_precision(a) || is_double_precision(b);
    if (complex) {
        return wide ? ScalarType::Complex128 : ScalarType::Complex64;
    }
    return wide ? ScalarType::Float64 : ScalarType::Float32;
}

/// Throw if an einsum over these element types cannot store its result in C.
///
/// @param c,a,b The element types of the operands.
/// @param complex_prefactor Whether either prefactor has a nonzero imaginary part.
/// @param where Names the caller in the message.
/// @throws std::invalid_argument when a type is unknown, when a complex product would be stored in a
///         real C, or when a complex prefactor would scale a real C. Each is rejected when the
///         einsum is built, not when it first runs.
inline void check_mixed_einsum(packed_gemm::ScalarType c, packed_gemm::ScalarType a, packed_gemm::ScalarType b, bool complex_prefactor,
                               std::string_view where) {
    using packed_gemm::ScalarType;
    if (c == ScalarType::Unknown || a == ScalarType::Unknown || b == ScalarType::Unknown) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: einsum operands must be float, double, complex<float> or complex<double>",
                                where);
    }
    if (is_complex(promote(a, b)) && !is_complex(c)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: a complex operand makes the contraction complex, and a real output cannot hold it; make the output "
                                "complex",
                                where);
    }
    if (complex_prefactor && !is_complex(c)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: a complex prefactor cannot scale a real output; make the output complex",
                                where);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
