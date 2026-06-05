//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace einsums::packed_gemm {

/// Scalar element type enumeration for type selection.
enum class ScalarType : std::uint8_t {
    Float32,
    Float64,
    Complex64,  ///< complex<float>
    Complex128, ///< complex<double>
    Unknown,
};

/// @brief Extract the ScalarType from a C++ value type at compile time.
/// Primary template returns Unknown; specializations below map known types.
template <typename T>
constexpr ScalarType get_scalar_type() {
    return ScalarType::Unknown;
}

template <>
constexpr ScalarType get_scalar_type<float>() {
    return ScalarType::Float32;
}
template <>
constexpr ScalarType get_scalar_type<double>() {
    return ScalarType::Float64;
}
template <>
constexpr ScalarType get_scalar_type<std::complex<float>>() {
    return ScalarType::Complex64;
}
template <>
constexpr ScalarType get_scalar_type<std::complex<double>>() {
    return ScalarType::Complex128;
}

/// @brief Describes the topology of a tensor contraction.
///
/// Indices may repeat (e.g., a_indices = {"i","i"} for a Hadamard A[i,i]).
/// all_indices is the unique loop space: target_indices ++ link_indices.
struct ContractionSpec {
    std::vector<std::string> c_indices;      ///< Raw C index list (may repeat)
    std::vector<std::string> a_indices;      ///< Raw A index list (may repeat)
    std::vector<std::string> b_indices;      ///< Raw B index list (may repeat)
    std::vector<std::string> all_indices;    ///< Unique loop space: target ++ link
    std::vector<std::string> link_indices;   ///< Unique link (reduction) dimensions
    std::vector<std::string> target_indices; ///< Unique target (parallel) dimensions
    ScalarType               scalar_type{ScalarType::Unknown};
    bool                     conj_a{false};
    bool                     conj_b{false};
    bool                     scalar_output{false}; ///< true when sizeof...(CIndices) == 0

    bool operator==(ContractionSpec const &o) const {
        return c_indices == o.c_indices && a_indices == o.a_indices && b_indices == o.b_indices && all_indices == o.all_indices &&
               link_indices == o.link_indices && target_indices == o.target_indices && scalar_type == o.scalar_type && conj_a == o.conj_a &&
               conj_b == o.conj_b && scalar_output == o.scalar_output;
    }
};

/// @brief Per-tensor metadata stored in the contraction key for cache lookup.
struct TensorDescriptor {
    size_t     rank{0};
    ScalarType dtype{ScalarType::Unknown};

    bool operator==(TensorDescriptor const &o) const { return rank == o.rank && dtype == o.dtype; }
};

/// @brief Full cache key uniquely identifying a contraction topology.
///
/// Combines the contraction topology with tensor ranks, element types, and
/// runtime dimension sizes.
struct ContractionKey {
    ContractionSpec      spec;
    TensorDescriptor     a_desc, b_desc, c_desc;
    std::vector<int64_t> target_dims; ///< Runtime sizes of target dimensions
    std::vector<int64_t> link_dims;   ///< Runtime sizes of link dimensions

    bool operator==(ContractionKey const &o) const {
        return spec == o.spec && a_desc == o.a_desc && b_desc == o.b_desc && c_desc == o.c_desc && target_dims == o.target_dims &&
               link_dims == o.link_dims;
    }
};

} // namespace einsums::packed_gemm

// ---------------------------------------------------------------------------
// std::hash specialisation for ContractionKey
// ---------------------------------------------------------------------------
namespace std {
template <>
struct EINSUMS_EXPORT hash<einsums::packed_gemm::ContractionKey> {
    size_t operator()(einsums::packed_gemm::ContractionKey const &key) const noexcept;
};
} // namespace std
