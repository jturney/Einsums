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
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Recovers a tensor's ``TensorImpl`` from the erased object @ref TensorSlot::ptr addresses.
 *
 * Named rather than spelled inline at each use because a bare
 * ``void *(*)(void *)`` is a declarator the field, the factory's return type
 * and the documentation extractor each have to spell the same way.
 *
 * @versionadded{2.0.0}
 */
using SlotImplAccessor = void *(*)(void *);

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

    /// @brief Rank-erased geometry accessor for whatever @ref ptr currently
    ///        addresses.
    ///
    /// Given @ref ptr, returns that tensor object's
    /// ``einsums::detail::TensorImpl<T> *``, type-erased to ``void *``. Cast it
    /// to ``TensorImpl<T> *`` once the element type is known (from
    /// @ref TensorHandle::dtype).
    ///
    /// This is how a data-built executor (@ref build_executor) reaches the LIVE
    /// data pointer, dims and strides of an operand. Going through the slot
    /// rather than through ``TensorHandle::impl_fn`` is what makes such an
    /// executor follow ``Graph::rebind`` and ``Graph::redirect_slot``, which is
    /// the same guarantee a capture-baked lambda gets from casting @ref ptr to
    /// its capture-time static type.
    ///
    /// A plain function pointer rather than a ``std::function``: it is
    /// stateless, since the object arrives as the argument, so it costs the
    /// slot one word and costs a replay one indirect call - no allocation, no
    /// lookup.
    ///
    /// Null for tile-wise sparse tensors, which have no single impl, and for
    /// slots created before the accessor existed. Gate on it.
    SlotImplAccessor impl_of{nullptr};

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
 * @brief The @ref TensorSlot::impl_of accessor for one static tensor type.
 * @tparam TensorType The type the slot's @ref TensorSlot::ptr addresses.
 * @return A stateless function pointer, or null for a type with no single
 *         ``impl()`` (tile-wise sparse tensors).
 *
 * Kept here, beside the field it fills, so ``Graph::get_or_create_slot`` and
 * ``Graph::rebind`` derive it the same way rather than each open-coding the
 * ``if constexpr``.
 */
template <typename TensorType>
[[nodiscard]] constexpr auto slot_impl_accessor() -> SlotImplAccessor {
    using Clean = std::remove_cvref_t<TensorType>;
    if constexpr (requires(Clean &t) { t.impl(); }) {
        return [](void *object) -> void * { return static_cast<void *>(&static_cast<Clean *>(object)->impl()); };
    } else {
        return nullptr;
    }
}

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
