//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file PackedGemmIndices.hpp
/// @brief The compile-time-index entry into PackedGemm, kept with the templated engine that uses it.
///
/// PackedGemm's own entry point takes a ContractionSpec, which is all ComputeGraph and the string
/// engine need. This overload builds that spec from index-type packs for ``tensor_algebra::einsum``.
/// It lives here rather than in PackedGemm so that the headers ComputeGraph includes carry no code
/// that serves only the templated engine.

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Profile.hpp>

#include <string>
#include <tuple>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_algebra::detail)

/// Extract the static letter string from each index type in a tuple.
template <typename... Indices>
std::vector<std::string> index_letters_from_tuple(std::tuple<Indices...> const & /*unused*/) {
    return {std::string(Indices::letter)...};
}

/// @brief Attempt to execute the einsum contraction via the packed GEMM backend.
///
/// The compile-time-indices form, used by `tensor_algebra::einsum<CIndices...,
/// AIndices..., BIndices...>`. It builds a ContractionSpec from the index packs
/// and forwards to `packed_gemm::try_packed_gemm`, PackedGemm's own entry point.
///
/// Returns `true` if the contraction was handled; `false` if the backend
/// should fall back to `einsum_generic_algorithm`.
template <bool ConjA, bool ConjB, einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(einsums::BasicTensorConcept<CType> || (einsums::ScalarConcept<CType> && sizeof...(CIndices) == 0))
bool try_packed_gemm_indices(einsums::ValueTypeT<CType> C_prefactor, std::tuple<CIndices...> const & /*C_indices_tup*/, CType *C,
                             einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor,
                             std::tuple<AIndices...> const & /*A_indices_tup*/, AType const &A,
                             std::tuple<BIndices...> const & /*B_indices_tup*/, BType const &B, bool allow_scatter = true) {
    LabeledSection0();

    // Scalar-output (CType is `T`, not a tensor) is not yet routed through the
    // runtime entry point, so keep the original handling for that one shape.
    if constexpr (!einsums::TensorConcept<CType>) {
        return false; // The original implementation built a degenerate key here;
                      // current packed-GEMM kernels require a tensor C anyway.
    } else {
        packed_gemm::ContractionSpec spec;
        spec.c_indices = index_letters_from_tuple(std::tuple<CIndices...>{});
        spec.a_indices = index_letters_from_tuple(std::tuple<AIndices...>{});
        spec.b_indices = index_letters_from_tuple(std::tuple<BIndices...>{});
        spec.conj_a    = ConjA;
        spec.conj_b    = ConjB;
        // Derived fields (target/link/all_indices, scalar_type) are filled in by
        // the runtime entry point.
        return packed_gemm::try_packed_gemm<AType, BType, CType>(spec, C_prefactor, C, AB_prefactor, A, B, allow_scatter);
    }
}

EINSUMS_NAMESPACE_END(tensor_algebra::detail)
