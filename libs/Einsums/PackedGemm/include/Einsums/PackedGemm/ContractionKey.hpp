//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

struct PackingPlan; // defined in Packing.hpp, which depends on this header

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

    /// Element strides, in index order.
    ///
    /// Present so a cached plan can be stored fully prepared rather than as a
    /// bare topology. fill_strides, sort_k_dims_for_packing and coalesce_plan
    /// read nothing but the strides, so once the key pins them down the result
    /// of all three is a property of the key and can be cached with it. Without
    /// this field, two contractions with the same indices and dims but different
    /// layouts - a dense tensor and a strided view - would collide, and the
    /// stored plan would be wrong for one of them.
    std::vector<int64_t> strides;

    bool operator==(TensorDescriptor const &o) const { return rank == o.rank && dtype == o.dtype && strides == o.strides; }
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

/// @brief Per-call-site memo for a contraction that repeats.
///
/// The plan cache makes PREPARING a packing plan free on a repeat; this makes
/// FINDING it free. Getting to the cache means assembling a ContractionSpec,
/// copying it again into a ContractionKey along with three stride vectors,
/// hashing every index string in it, and comparing them all again under the
/// cache's lock - about twenty allocations that a hit throws away. A
/// ComputeGraph node replaying its contraction, or a tiled expansion driving
/// thousands of same-shape contractions through one node, resolves to the same
/// entry every single time.
///
/// A site is owned by whatever repeats: one per graph node. It holds the key
/// it resolved, which is re-checked against the caller's spec and the
/// operands' current layout on every call, so it is exactly as sound as the
/// plan cache it front-ends - equal key, same plan. Nothing here is
/// thread-safe: a site belongs to one caller.
struct ContractionSite {
    ContractionKey     key;                 ///< what @c plan was resolved for
    PackingPlan const *plan{nullptr};       ///< cache-owned and stable, or null for "declined"
    bool               resolved{false};     ///< @c key and @c plan are filled
    bool               allow_scatter{true}; ///< the policy the resolution was made under
};

/// @brief Whether @p spec already describes this contraction's topology.
///
/// A ContractionSpec is a pure function of the index lists and the conjugation
/// flags, so a caller holding one built for these lists - a @ref
/// ContractionSite's key, say - can reuse it instead of assembling six
/// vector<string> per call. Index lists are rank-bounded and their elements
/// are single letters, so this compares in nanoseconds and allocates nothing.
inline bool spec_matches_indices(ContractionSpec const &spec, std::vector<std::string> const &c_indices,
                                 std::vector<std::string> const &a_indices, std::vector<std::string> const &b_indices,
                                 std::vector<std::string> const &link_indices, bool conj_a, bool conj_b) {
    return spec.conj_a == conj_a && spec.conj_b == conj_b && spec.c_indices == c_indices && spec.a_indices == a_indices &&
           spec.b_indices == b_indices && spec.link_indices == link_indices;
}

EINSUMS_NAMESPACE_END(packed_gemm)

// ---------------------------------------------------------------------------
// std::hash specialisation for ContractionKey
// ---------------------------------------------------------------------------
namespace std {
template <>
struct EINSUMS_EXPORT hash<einsums::packed_gemm::ContractionKey> {
    size_t operator()(einsums::packed_gemm::ContractionKey const &key) const noexcept;
};
} // namespace std
