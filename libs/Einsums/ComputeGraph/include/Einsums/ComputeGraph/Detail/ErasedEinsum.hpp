//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ErasedEinsum.hpp
/// @brief The eager einsum's entry into the library: declared here, compiled once in src/.
///
/// Eager ``cg::einsum`` hands its operands' TensorImpls to these functions instead of instantiating
/// the string engine in the caller's translation unit. They are declared without a body, so a caller
/// cannot instantiate them and links against the library's explicit instantiations: the route
/// cascade, PackedGemm's driver and the generic loops are compiled once, in the library, rather than
/// once per file and per operand-type combination that calls ``cg::einsum``.

#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

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
