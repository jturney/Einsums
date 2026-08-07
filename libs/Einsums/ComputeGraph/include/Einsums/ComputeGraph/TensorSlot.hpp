//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file TensorSlot.hpp
 * @brief Rebindable tensor reference for graph-captured operations.
 *
 * A TensorSlot is a type-erased indirection layer between a graph node's
 * executor lambda and the actual tensor. Instead of capturing a direct tensor
 * reference (which cannot be changed after capture), the lambda captures a
 * TensorSlot pointer. The slot's internal pointer can be updated via
 * Graph::rebind() without re-capturing the graph.
 *
 * @code
 * // Slot created during capture (internal to CaptureContext):
 * TensorSlot slot;
 * slot.ptr = &A;  // Points to tensor A
 *
 * // Lambda captures &slot (stable address):
 * auto executor = [&slot]() {
 *     auto &tensor = *static_cast<Tensor<double,2>*>(slot.ptr);
 *     // ... use tensor ...
 * };
 *
 * // Later, rebind to a different tensor:
 * slot.ptr = &A_new;  // Lambda now uses A_new
 * @endcode
 */

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief A rebindable tensor reference.
 *
 * Holds a void pointer to the current tensor object. The pointer can be
 * updated via Graph::rebind() without re-capturing the graph.
 */
struct TensorSlot {
    void               *ptr{nullptr};    ///< Current pointer to the Tensor object
    TensorId            tensor_id{0};    ///< Associated TensorId in the graph
    std::string         name;            ///< Tensor name for error messages
    size_t              rank{0};         ///< Expected rank (for validation on rebind)
    size_t              element_size{0}; ///< Expected element size (for validation)
    std::vector<size_t> dims;            ///< Expected dimensions (for validation on rebind)

    /// Keeps whatever @ref ptr addresses alive for as long as the slot exists.
    ///
    /// Set to the graph's stand-in for a captured operand (see
    /// ``Graph::adopt_operand``), which is what makes it safe for the caller's
    /// own wrapper to be destroyed between capture and ``execute()``. Empty
    /// when the graph did not adopt the operand -- a tensor the graph already
    /// owns, or a type with no storage block to share -- in which case @ref ptr
    /// keeps the older "must outlive the graph" contract.
    std::shared_ptr<void> owner;
};

/**
 * @brief Mutable scalar parameters for einsum operations.
 *
 * Stored in a shared_ptr so the executor lambda captures it by shared
 * ownership. Updating the values changes the computation on next execute().
 *
 * Both prefactors are @ref PrefactorScalar variants, they hold one of
 * float/double/complex<float>/complex<double>. The executor extracts the
 * concrete scalar via ``as<T>`` when dispatching to typed BLAS.
 */
struct EinsumParams {
    PrefactorScalar c_pf{double{0}};  ///< C prefactor
    PrefactorScalar ab_pf{double{1}}; ///< AB prefactor
    bool            conj_a{false};    ///< Conjugate A (complex; no-op for real)
    bool            conj_b{false};    ///< Conjugate B (complex; no-op for real)
};

/**
 * @brief Mutable index specification for einsum operations.
 *
 * Same story as EinsumParams but for the contraction's index strings
 * (a/b/c indices, link indices). Stored in a shared_ptr so the executor
 * captures it by shared ownership; optimization passes like
 * PermuteFusion can rewrite the indices in place and the updated
 * contraction takes effect on the next execute().
 *
 * Holds @ref ParsedEinsumSpec plus the precomputed link indices so
 * dispatch::string_einsum doesn't have to recompute them per call.
 *
 * The spec is stored WHOLE rather than as its three index vectors so the
 * executors can hand it to the dispatch by reference. Assembling one per call
 * copies three ``vector<string>`` -- about fifteen allocations for a rank-4
 * contraction -- and a tiled einsum expanded per tile turns that into tens of
 * thousands per replay: measured at 2.1 us a node, 8 ms of a 33 ms CCSD replay,
 * more than the tile bookkeeping the graph exists to avoid. A pass that rewrites
 * indices writes into this spec and the next execute() sees it, which is the
 * property the per-call rebuild was there to provide.
 */
struct EinsumIndices {
    ParsedEinsumSpec         spec;         ///< Live index lists, passed straight to the dispatch
    std::vector<std::string> link_indices; ///< Contracted (shared A/B, not in C) indices
};

EINSUMS_NAMESPACE_END(compute_graph)
