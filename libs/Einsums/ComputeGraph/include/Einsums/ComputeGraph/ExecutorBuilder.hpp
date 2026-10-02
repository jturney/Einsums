//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ExecutorBuilder.hpp
 * @brief The single point where a @ref Node::execute callable is derived from data.
 *
 * @par Why this exists
 * @ref build_executor derives the callable from
 * ``(kind, dtype, rank, descriptor, operand ids)`` and nothing else. The
 * descriptor is then the only record of the operation, so a pass that rewrites
 * it changes what replay computes, and a node can be reconstructed from a file.
 * Capture, passes and the IR loader all go through it.
 *
 * @par Operand-passing convention
 * Operands come from the node's own @ref Node::inputs and @ref Node::outputs
 * lists, positionally, in the order capture records them. Descriptors do NOT
 * carry a second copy of the operand ids.
 *
 * The dataflow lists are what every pass already rewrites, so a second copy
 * would drift. ``ViewDescriptor::parent_id`` is the exception: a view's parent
 * is a structural relation (it sets ``TensorHandle::aliases``), not an operand.
 *
 * Positions, per kind (an accumulating op repeats its destination as the LAST
 * input, which is why the leading positions are stable):
 *
 * - Scale: ``A = outputs[0]``.
 * - Permute: ``A = inputs[0]``, ``C = outputs[0]``.
 * - Transpose: ``A = inputs[0]``, ``C = outputs[0]``.
 * - Axpby: ``X = inputs[0]``, ``Y = outputs[0]``.
 * - DirectProduct and DirectDivision: ``A = inputs[0]``, ``B = inputs[1]``, ``C = outputs[0]``.
 * - Einsum: ``A = inputs[0]``, ``B = inputs[1]``, ``C = outputs[0]``. An
 *   accumulating einsum (nonzero C prefactor) also lists C as ``inputs[2]``,
 *   which is the RMW convention and carries no operand of its own.
 * - Dot: ``A = inputs[0]``, ``B = inputs[1]``, ``result = outputs[0]``.
 * - Trace: ``A = inputs[0]``, ``result = outputs[0]``.
 * - Gemm: ``A = inputs[0]``, ``B = inputs[1]``, ``C = outputs[0]``. An
 *   accumulating gemm (nonzero beta) repeats C as ``inputs[2]``, the same RMW
 *   convention the einsum uses.
 * - WriteParam: ``source = inputs[0]``, no outputs, and no operands at all on
 *   the expression arm. Its whole effect is a write into the @ref ParamTable.
 * - ElementTransform: ``C = outputs[0]``. The operation is a read-modify-write,
 *   so capture lists the same tensor as ``inputs[0]`` too; the builder reads
 *   only the output, as ``Scale`` does.
 * - Conditional and Loop: NO operands. A control-flow node's content is its
 *   subgraphs and its predicate, and the operands its body touches belong to
 *   that body's own nodes. ``dtype`` and ``rank`` are meaningless for these two
 *   and are neither dispatched on nor validated.
 *
 * @par How a built executor reaches its operands
 * Once, at build time, through @ref resolve_operand; thereafter through the
 * resolved @ref TensorSlot on every call. A slot's address is stable for the
 * graph's lifetime and is what ``Graph::rebind`` and ``Graph::redirect_slot``
 * repoint, so a built executor follows both. Live geometry comes from
 * @ref TensorSlot::impl_of: one indirect call per operand, no allocation. An
 * operand with no slot falls back to @ref TensorHandle::impl_fn.
 *
 * @par How live-mutable scalars reach a built executor
 * Through the descriptor's shared params block (@ref ElementwiseParams,
 * @ref AxpbyParams), which the executor holds by ``shared_ptr`` and reads on
 * every call. A pointer into the node's ``op_data`` would dangle, since passes
 * insert, erase and reorder the node vector.
 *
 * So a pass that rewrites a PREFACTOR writes the live params and needs no
 * rebuild; one that rewrites a STRUCTURAL field (index lists, operands)
 * rewrites the descriptor and calls @ref build_executor again.
 *
 * @par Dispatch routes
 * Every kind takes the RANK-ERASED route through
 * ``einsums::detail::TensorImpl<T>``, which carries dims and strides at run
 * time, so one dtype dispatch covers every rank. @p rank is validated, not
 * dispatched on.
 *
 * ``Gemm`` reaches ``linear_algebra::detail::gemm(char, char, alpha, impl,
 * impl, beta, impl*)``, not the tensor-object overload with the
 * ``symm``/``hemm`` fast path, because that is the entry ``cg::gemm`` capture
 * has always used.
 *
 * ``Einsum`` wraps each operand's live impl in a ``RuntimeTensorView<T>``: one
 * dtype dispatch instead of one instantiation per rank triple. The views are
 * built once and re-seated on each call, reusing their vector capacity. See
 * ``build_einsum``.
 *
 * @par Scalar destinations
 * ``Dot`` and ``Trace`` write ONE number, and the tensor that number lands in
 * may be a rank-0 handle registered with
 * ``CaptureContext::get_or_register_scalar``: a plain ``T *`` with no slot.
 * @ref ScalarAccessor handles both shapes: a tensor destination (as
 * ``dot_python`` uses) goes through its slot to element 0, a raw scalar
 * through @ref TensorHandle::tensor_ptr, read on every call so repointing the
 * handle moves the write.
 *
 * @see GraphIR.hpp, whose loader rebuilds every node through this same point
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief One operand's live geometry, resolved once when an executor is built.
 *
 * Resolution order is slot first, handle second, and the order is the point.
 * A @ref TensorSlot is what ``Graph::rebind`` and ``Graph::redirect_slot``
 * repoint, so reading through it is what makes a built executor honor both
 * without the pass author having to think about it. ``TensorHandle::impl_fn``
 * is the fallback for a tensor that reached the graph without ever being
 * captured through a slot, which is only ever a node some pass assembled.
 *
 * Neither path allocates or looks anything up at call time: the slot address
 * is stable for the graph's lifetime, and @ref TensorSlot::impl_of is a plain
 * function pointer.
 *
 * @see resolve_operand
 */
class OperandAccessor {
  public:
    OperandAccessor() = default;

    /// Read through @p slot, which must carry a non-null @ref TensorSlot::impl_of. @p dtype is the
    /// tensor's element type, when known.
    explicit OperandAccessor(TensorSlot *slot, packed_gemm::ScalarType dtype = packed_gemm::ScalarType::Unknown)
        : _slot(slot), _dtype(dtype) {}

    /// Read through a handle's own rank-erased accessor.
    explicit OperandAccessor(std::function<void *()> handle_impl, packed_gemm::ScalarType dtype = packed_gemm::ScalarType::Unknown)
        : _handle_impl(std::move(handle_impl)), _dtype(dtype) {}

    /// Whether this accessor was bound to anything.
    [[nodiscard]] bool valid() const noexcept { return _slot != nullptr || static_cast<bool>(_handle_impl); }

    /// The operand's element type, recorded from its tensor when the accessor was resolved; Unknown
    /// when nothing recorded one. An einsum's executor reads its three operands' types from here.
    [[nodiscard]] packed_gemm::ScalarType dtype() const noexcept { return _dtype; }

    /// The operand's CURRENT rank-erased geometry, type-erased.
    [[nodiscard]] void *raw() const { return _slot != nullptr ? _slot->impl_of(_slot->ptr) : _handle_impl(); }

    /// The operand's CURRENT rank-erased geometry, as element type @p T.
    ///
    /// @throws std::logic_error when the operand's recorded type is not @p T. The cast below would
    /// otherwise reinterpret the tensor's memory as another type and compute garbage, silently: an
    /// executor built with the wrong type, by a pass or anything else, fails here instead.
    template <typename T>
    [[nodiscard]] ::einsums::detail::TensorImpl<T> *impl() const {
        constexpr auto asked = packed_gemm::get_scalar_type<T>();
        if (_dtype != packed_gemm::ScalarType::Unknown && asked != packed_gemm::ScalarType::Unknown && _dtype != asked) {
            throw_operand_type_mismatch(_dtype, asked);
        }
        return static_cast<::einsums::detail::TensorImpl<T> *>(raw());
    }

  private:
    [[noreturn]] EINSUMS_EXPORT static void throw_operand_type_mismatch(packed_gemm::ScalarType held, packed_gemm::ScalarType asked);

    TensorSlot             *_slot{nullptr};
    std::function<void *()> _handle_impl;
    packed_gemm::ScalarType _dtype{packed_gemm::ScalarType::Unknown};
};

/**
 * @brief Bind an @ref OperandAccessor to @p id, or return an unbound one.
 *
 * @param[in,out] graph Graph the id belongs to.
 * @param[in]     id    The operand.
 * @return An accessor, or an invalid one when @p id exposes no rank-erased
 *         geometry (a tile-wise sparse tensor, which has no single buffer).
 */
[[nodiscard]] EINSUMS_EXPORT OperandAccessor try_resolve_operand(Graph &graph, TensorId id);

/**
 * @brief Bind an @ref OperandAccessor to @p id, or explain why it cannot be bound.
 *
 * @param[in,out] graph   Graph the id belongs to.
 * @param[in]     id      The operand.
 * @param[in]     context Caller name for the diagnostic, e.g. ``"build_executor(Einsum)"``.
 * @param[in]     role    The operand's role in that caller, e.g. ``"A"``.
 * @return A bound accessor.
 * @throws std::invalid_argument When @p id exposes no rank-erased geometry.
 */
[[nodiscard]] EINSUMS_EXPORT OperandAccessor resolve_operand(Graph &graph, TensorId id, std::string_view context, char const *role);

/**
 * @brief One SCALAR operand's live address, resolved once when an executor is built.
 *
 * The scalar counterpart of @ref OperandAccessor, and it exists because the two
 * things a graph calls a scalar are not the same object. ``cg::dot(&e, A, B)``
 * registers a bare ``double *``: a rank-0 @ref TensorHandle with no
 * ``TensorImpl``, no dims and no slot. ``cg::dot_python`` instead hands the
 * graph a rank-1 tensor, because a Python caller needs a scalar it can then
 * scale and add like any other operand, and that one has a slot like any other
 * tensor. Both name one address; neither shape can be read through the other's
 * accessor.
 *
 * Resolution prefers the tensor route, so a destination that HAS a slot follows
 * ``Graph::rebind`` and ``Graph::redirect_slot``. The raw-scalar route reads
 * @ref TensorHandle::tensor_ptr on every call, so repointing the handle moves
 * the write.
 *
 * Neither path allocates: a handle's address is stable for the graph's lifetime
 * (the tensor table is a node-based map), and the tensor route is one
 * @ref OperandAccessor read.
 *
 * @see resolve_scalar_operand
 */
class ScalarAccessor {
  public:
    ScalarAccessor() = default;

    /// Read element 0 of a tensor destination, through its slot when it has one.
    explicit ScalarAccessor(OperandAccessor tensor) : _tensor(std::move(tensor)) {}

    /// Read a registered raw scalar through @p handle, which must outlive the executor.
    explicit ScalarAccessor(TensorHandle const *handle) : _handle(handle) {}

    /// Whether this accessor was bound to anything.
    [[nodiscard]] bool valid() const noexcept { return _tensor.valid() || _handle != nullptr; }

    /// The destination's CURRENT address.
    template <typename T>
    [[nodiscard]] T *address() const {
        if (_handle != nullptr) {
            // A scalar handle is never adopted (Graph::adopt_operand takes
            // tensor objects), so tensor_ptr is the whole story here.
            return static_cast<T *>(_handle->tensor_ptr);
        }
        return _tensor.impl<T>()->data();
    }

  private:
    OperandAccessor     _tensor;
    TensorHandle const *_handle{nullptr};
};

/**
 * @brief Bind a @ref ScalarAccessor to @p id, or explain why it cannot be bound.
 *
 * @param[in,out] graph   Graph the id belongs to.
 * @param[in]     id      The scalar destination.
 * @param[in]     context Caller name for the diagnostic, e.g. ``"build_executor(Dot)"``.
 * @param[in]     role    The operand's role in that caller, e.g. ``"result"``.
 * @return A bound accessor.
 * @throws std::invalid_argument When @p id names no tensor, or names one with
 *         neither rank-erased geometry nor a registered address.
 */
[[nodiscard]] EINSUMS_EXPORT ScalarAccessor resolve_scalar_operand(Graph &graph, TensorId id, std::string_view context, char const *role);

/**
 * @brief Derive a node's @ref GemmHint, or decide it has none.
 *
 * The one derivation, shared by capture (``cg::einsum``) and
 * @ref Graph::make_einsum_node. A hint describing a product the einsum does not
 * perform stays invisible until GEMMBatching batches it and ``gemm_batch``
 * miscomputes.
 *
 * @param[in]     dtype Element type shared by the three operands.
 * @param[in]     spec  The contraction's index lists and link set.
 * @param[in,out] graph Graph the operand ids belong to.
 * @param[in]     a_id  Left input.
 * @param[in]     b_id  Right input.
 * @param[in]     c_id  Destination.
 * @return The hint, or null when the contraction does not qualify.
 *
 * @par The gate
 * Three rank-2 operands, two index letters apiece, exactly one link index, and
 * strides that are monotone in each operand's own declared storage order. That
 * last clause is not redundant with the layout flag: a permute_view keeps its
 * parent's flag while presenting reordered strides, so the flag alone does not
 * prove the canonical layout ``gemm_batch`` assumes.
 *
 * @par The roles clause
 * The batched form is ``C = op(A) * op(B)``, so C's FIRST index must be the one
 * A contributes and its SECOND the one B contributes. ``"ia <- ma ; mi"`` has
 * them swapped - ``i`` comes from B - and m/n/k would then describe a different
 * matrix product. The generic kernel contracts correctly either way and never
 * consults the hint, so a wrong hint stays invisible until a batch forms. Emit
 * no hint rather than a wrong one.
 *
 * Never throws: an operand with no resolvable geometry simply yields no hint,
 * because a hint is an optimization and its absence is always legal.
 */
[[nodiscard]] EINSUMS_EXPORT std::shared_ptr<GemmHint> derive_gemm_hint(packed_gemm::ScalarType             dtype,
                                                                        packed_gemm::ContractionSpec const &spec, Graph &graph,
                                                                        TensorId a_id, TensorId b_id, TensorId c_id);

/**
 * @brief Whether nodes of @p kind can have their executor rebuilt from data alone.
 *
 * True exactly when @ref build_executor has an entry for the kind, which is
 * also exactly when a node of that kind can be written to a file and read back
 * (the design's *reconstructible* bit). The two must agree, and
 * ``Tests.Unit.Modules.ComputeGraph.ExecutorBuilder`` asserts that they do.
 *
 * @param[in] kind The operation kind.
 * @return True when a builder entry exists.
 *
 * @warning **This set only ever grows.** A kind may be added when its
 *          descriptor is complete and its builder entry lands; removing one
 *          means some node that could be saved no longer can, which is a
 *          regression rather than a refactor. The monotonicity test pins the
 *          current membership so a removal fails the build rather than
 *          quietly shrinking what a graph can persist.
 *
 * @note The bit is per KIND, which is coarser than per node: ``DirectDivision``
 *       is reconstructible, yet a TILED direct division records under that kind
 *       with a @ref TiledElementwiseDescriptor and is not. Ask
 *       @ref reconstruction_blocker for the per-node answer. ``Syev`` splits the
 *       same way, with the tiled arm recording no descriptor at all.
 *
 * @note ``Syev`` is the first entry whose builder refuses a DTYPE rather than a
 *       descriptor. A symmetric eigendecomposition is real; the complex operand
 *       is a Hermitian one and records as ``Heev``, which has no entry yet. No
 *       capture can produce a complex ``Syev`` node, so the refusal is aimed at
 *       a file that describes one.
 *
 * @note For four kinds the true bit covers only PART of what the kind records:
 *       the arm that blocks is a closure held in an otherwise data-shaped
 *       descriptor:
 *
 *       - ``WriteParam``: a @ref BoundExpr source, whose ``Callback`` arm is a
 *         ``std::function`` a file cannot hold.
 *       - ``Conditional`` and ``Loop``: a @ref PredExpr predicate, same story.
 *         Holding a SUBGRAPH does not block them; whether
 *         the subgraph can be written is a question for the graph.
 *       - ``ElementTransform``: a named kernel rebuilds; the lambda-taking
 *         overloads record no descriptor at all and cannot.
 *
 *       So the kind is reconstructible and a callback-arm node is not, and only
 *       @ref reconstruction_blocker can tell them apart. @ref build_executor
 *       builds both, since a closure in a descriptor still lowers in this
 *       process.
 *
 * @note The GROUPED kinds are deliberately false although @ref build_executor
 *       handles them: a group table is a function of one problem's extents,
 *       and saving it would freeze that problem into the file. Saving one
 *       means saving the algebraic form and re-grouping on load.
 */
[[nodiscard]] constexpr bool is_reconstructible(OpKind kind) noexcept {
    switch (kind) {
    case OpKind::Scale:
    case OpKind::Permute:
    case OpKind::Transpose:
    case OpKind::Axpby:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Einsum:
    case OpKind::Dot:
    case OpKind::Trace:
    case OpKind::WriteParam:
    case OpKind::Gemm:
    case OpKind::ElementTransform:
    case OpKind::Conditional:
    case OpKind::Loop:
    case OpKind::Setup:
    case OpKind::Syev:
    case OpKind::LaplaceQuadrature:
        return true;
    default:
        return false;
    }
}

/**
 * @brief One node that blocks a graph from being saved, and what blocks it.
 *
 * @see Graph::serializability_report
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") SerializabilityBlocker {
    APIARY_EXPOSE APIARY_READONLY NodeId node_id{0}; ///< The offending node's id.

    /// Its human-readable label, so the report names something a user recognises.
    APIARY_EXPOSE APIARY_READONLY std::string label;

    /// @ref op_kind_name of its kind.
    APIARY_EXPOSE APIARY_READONLY std::string kind_name;

    /// What blocks it, e.g. "kind not yet reconstructible".
    APIARY_EXPOSE APIARY_READONLY std::string reason;

    /// Where the node lives relative to the graph the report was asked of.
    ///
    /// Empty for a node of that graph itself. Otherwise a ``/``-separated path of the
    /// control-flow nodes descended through, each named by its label and, for a
    /// ``Conditional``, by which branch was taken: ``loop(scf_iter)/then(converged)``.
    ///
    /// A NodeId is graph-local, so ``node_id`` alone does not locate a node inside a body.
    /// This is what makes the pair addressable.
    ///
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_READONLY std::string subgraph_path;
};

/**
 * @brief Why @p node cannot be rebuilt from data alone.
 *
 * @param[in] node The node to inspect.
 * @return An empty string when the node IS reconstructible; otherwise a short
 *         phrase naming the field or property that blocks it.
 *
 * Checks more than @ref is_reconstructible: a kind with a builder entry can
 * still carry a descriptor alternative that entry does not handle (the tiled
 * variants), or no descriptor at all where one is required.
 */
[[nodiscard]] EINSUMS_EXPORT std::string reconstruction_blocker(Node const &node);

/**
 * @brief Build a @ref Node::execute callable from data alone.
 *
 * @param[in]     kind    Which operation to build.
 * @param[in]     dtype   Element type shared by every operand.
 * @param[in]     rank    Operand rank. Validated; see the dispatch-routes note
 *                        in this file's header for why it is not dispatched on.
 * @param[in]     desc    The node's @ref OpData. Must hold the alternative the
 *                        kind expects; a live params handle on it is shared
 *                        with the returned callable, so a later rewrite of
 *                        those scalars is honored on the next execute.
 * @param[in,out] graph   Graph the operand ids belong to. Slots are resolved
 *                        from it once, here, not on every call.
 * @param[in]     inputs  The node's input ids, in capture order.
 * @param[in]     outputs The node's output ids, in capture order.
 * @return A callable that performs the operation.
 *
 * @throws std::invalid_argument When @p kind has no builder entry (the message
 *         names the kind), when @p desc holds the wrong alternative, when the
 *         operand lists are too short, or when an operand has no resolvable
 *         geometry (a tile-wise sparse tensor, which has no single buffer).
 *
 * A descriptor whose params handle is null is not an error: the builder
 * synthesizes a private one from the descriptor's snapshot scalars, so a node
 * a pass assembled by hand still runs. Such a node does not get the
 * live-rewrite guarantee, because there is nothing shared to rewrite - callers
 * that want it populate ``params`` before calling, as capture does.
 */
[[nodiscard]] EINSUMS_EXPORT std::function<void()> build_executor(OpKind kind, packed_gemm::ScalarType dtype, std::size_t rank,
                                                                  OpData const &desc, Graph &graph, std::span<TensorId const> inputs,
                                                                  std::span<TensorId const> outputs);

EINSUMS_NAMESPACE_END(compute_graph)
