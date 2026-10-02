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

    bool operator==(ContractionSpec const &) const = default;
};

/// @brief Per-tensor metadata stored in the contraction key for cache lookup.
struct TensorDescriptor {
    size_t     rank{0};
    ScalarType dtype{ScalarType::Unknown};

    /// Element strides, in index order.
    ///
    /// Part of the key so a cached plan can be stored fully prepared (stride-filled, sorted,
    /// coalesced), and a strided view never shares a dense tensor's plan.
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

/// @brief Which kernel a contraction is to be run through.
///
/// One vendor GEMM or the packed tile loops. They differ in the last bit on some shapes, and the
/// adaptive choice follows the thread regime (@ref einsums::blas::vendor_call_is_fenced), so a caller
/// that re-plans widths pins the route to keep results bit-stable.
enum class KernelRoute : std::uint8_t {
    /// Choose per call from the thread regime. Eager callers and unplanned
    /// graphs, whose widths never move, keep exactly this.
    Adaptive,
    /// Always pack, at every width including 1. The width-stable side: the
    /// packed loops give the same bits whatever they are forked at.
    Packed,
    /// Always hand the shape to the vendor. Only sound for a caller whose width
    /// is ALSO pinned - a threaded vendor GEMM is not bit-stable across its own
    /// thread counts on every shape.
    Vendor,
};

/// @brief Per-call-site memo for a contraction that repeats.
///
/// Skips building and hashing a key on a repeat (about twenty allocations). One per graph node; its
/// key is re-checked against the spec and operand layouts every call. Not thread-safe.
struct ContractionSite {
    ContractionKey     key;                 ///< what @c plan was resolved for
    PackingPlan const *plan{nullptr};       ///< cache-owned and stable, or null for "declined"
    bool               resolved{false};     ///< @c key and @c plan are filled
    bool               allow_scatter{true}; ///< the policy the resolution was made under

    /// The route this site's contraction is pinned to, or Adaptive for "read
    /// the thread regime per call". Written once by whoever plans the site's
    /// owner and read on every call; never written by the engine.
    KernelRoute route{KernelRoute::Adaptive};

    /// The effective route the memo above was recorded under.
    ///
    /// A decline depends on the route (a plan does not), so a memo is re-derived when the route moves.
    bool declined_packed{false};
};

/// @brief Whether @p spec already describes this contraction's topology.
///
/// Lets a caller reuse a spec (a @ref ContractionSite's) instead of rebuilding it. Allocation-free.
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
