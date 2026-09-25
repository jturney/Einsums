//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ErasedEinsum.hpp
/// @brief ``cg::einsum``'s entries into the library, eager and captured: declared here, compiled once in src/.
///
/// ``cg::einsum`` hands its operands' TensorImpls (and, when capturing, their slots) to these
/// functions instead of instantiating the string engine and the capture path in the caller's
/// translation unit. They are declared without a body, so a caller cannot instantiate them and links
/// against the library's explicit instantiations: the route cascade, PackedGemm's driver, the generic
/// loops and the node recording are compiled once, in the library, rather than once per file and per
/// operand-type combination that calls ``cg::einsum``.

#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

/// Name of the kernel route the most recent string_einsum call on this
/// thread selected ("packed_gemm", "gemv_mat_vec", "generic_loop",
/// "generic_loop_repeated_indices", "empty_input_scale_only", ...).
///
/// Test introspection ONLY: lets dispatch-coverage tests assert the intended
/// fast path fired instead of a silent generic-loop fallback, mirroring the
/// eager API's AlgorithmChoice out-parameter. Thread-local; not an API for
/// steering execution.
///
/// Defined OUT OF LINE, and exported, so the whole process shares one slot. An
/// inline function's thread-local gets a copy per shared object under hidden
/// visibility, and a graph whose executors were built inside the library (which
/// is every einsum node since @ref build_executor took over the lowering) would
/// then write a slot no test executable can read.
[[nodiscard]] EINSUMS_EXPORT char const *&last_dispatch_route();

/// The character-index permute behind string specs. Defined in StringDispatch.hpp, which
/// string_einsum needs it for, and explicitly instantiated in the library for the four element
/// types, so a caller of cg::permute compiles none of it.
template <typename T>
EINSUMS_EXPORT void string_permute_impl(ParsedPermuteSpec const &parsed, T beta, einsums::detail::TensorImpl<T> *C, T alpha,
                                        einsums::detail::TensorImpl<T> const &A);

/**
 * @brief Tensor-object overload of @ref string_permute_impl.
 *
 * Forwards both operands' ``impl()`` to the one definition above, so every
 * caller that holds tensor objects (capture's eager path, the tiled lowering,
 * the reassociation passes) and the data-built executors run identical code.
 *
 * @tparam AType,CType Source and destination tensor types; same element type.
 * @param[in] parsed The index lists, and the raw spec for diagnostics.
 * @param[in] beta Prefactor on the destination; 0 overwrites it.
 * @param[in,out] C Destination tensor.
 * @param[in] alpha Prefactor on the source.
 * @param[in] A Source tensor.
 */
template <BasicTensorConcept AType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
void string_permute(ParsedPermuteSpec const &parsed, typename AType::ValueType beta, CType *C, typename AType::ValueType alpha,
                    AType const &A) {
    string_permute_impl<typename AType::ValueType>(parsed, beta, &C->impl(), alpha, A.impl());
}

/**
 * @brief ``C = c_pf * C + ab_pf * contract(A, B)`` for operands of one element type, on their
 *        TensorImpls.
 *
 * Runs ``string_einsum`` exactly as the graph's replay executor does. Explicitly instantiated in the
 * library for float, double, complex<float> and complex<double>.
 */
template <typename T>
EINSUMS_EXPORT void erased_string_einsum(ParsedEinsumSpec const &parsed, T c_pf, einsums::detail::TensorImpl<T> &C, T ab_pf,
                                         einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B, bool conj_a,
                                         bool conj_b);

/**
 * @brief The mixed-precision counterpart of @ref erased_string_einsum, on operands whose element
 *        types differ.
 *
 * Runs ``mixed_string_einsum``. Explicitly instantiated in the library for every triple of the four
 * element types; a triple that is uniform, or that would store a complex product in a real C,
 * throws ``std::logic_error``, since ``cg::einsum`` rejects both before calling it.
 */
template <typename TC, typename TA, typename TB>
EINSUMS_EXPORT void erased_mixed_string_einsum(ParsedEinsumSpec const &parsed, TC c_pf, einsums::detail::TensorImpl<TC> &C,
                                               detail::PromoteT<TA, TB> ab_pf, einsums::detail::TensorImpl<TA> const &A,
                                               einsums::detail::TensorImpl<TB> const &B, bool conj_a, bool conj_b);

EINSUMS_NAMESPACE_END(compute_graph::dispatch)

EINSUMS_NAMESPACE_BEGIN(compute_graph)
class CaptureContext;
struct TensorSlot;
EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// One operand of a capture, as the graph knows it once the caller has registered it.
template <typename T>
struct CapturedOperand {
    TensorId                              id;
    TensorSlot                           *slot;
    einsums::detail::TensorImpl<T> const *impl;
};

/**
 * @brief Record ``C = c_pf * C + ab_pf * contract(A, B)`` into the capturing graph.
 *
 * The capture half of ``cg::einsum``: the caller registers each operand with the context, which is
 * what depends on the tensor types, and this builds the descriptor, binds the index spaces, derives
 * the GEMM hint, and records an einsum node whose executor comes from @ref build_executor (which
 * picks the strided batched-GEMM route when the operands allow it). Explicitly instantiated in the
 * library for the four element types.
 */
template <typename T>
EINSUMS_EXPORT void capture_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, T c_pf, T ab_pf, bool conj_a, bool conj_b,
                                          CapturedOperand<T> const &a, CapturedOperand<T> const &b, CapturedOperand<T> const &c);

/**
 * @brief The mixed-precision counterpart of @ref capture_string_einsum. Explicitly instantiated for
 *        every triple of the element types; a uniform or unstorable triple throws, as for
 *        @ref dispatch::erased_mixed_string_einsum.
 */
template <typename TC, typename TA, typename TB>
EINSUMS_EXPORT void capture_mixed_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, TC c_pf, PromoteT<TA, TB> ab_pf,
                                                bool conj_a, bool conj_b, CapturedOperand<TA> const &a, CapturedOperand<TB> const &b,
                                                CapturedOperand<TC> const &c);

EINSUMS_NAMESPACE_END(compute_graph::detail)
