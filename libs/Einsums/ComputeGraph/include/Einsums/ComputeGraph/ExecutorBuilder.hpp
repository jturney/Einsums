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
 * A captured node used to carry its operation inside a lambda that capture
 * baked at the call site: the tensor types, the ranks, and the prefactors were
 * all closed over. Two consequences followed, and both are bugs rather than
 * inconveniences.
 *
 * The first is the desync class. A node's descriptor and its executor were two
 * independent records of one operation, so a pass that rewrote the descriptor
 * changed what every analysis believed and nothing about what the next
 * ``graph.execute()`` computed. ``PermuteDescriptor`` was written at capture
 * and read by five passes while its executor ignored it entirely.
 *
 * The second is that a closure cannot be written to a file, which is what
 * blocks graph serialization: there is no way to reconstruct a node from what
 * a file can hold.
 *
 * @ref build_executor answers both. It derives the callable from
 * ``(kind, dtype, rank, descriptor, operand ids)`` and nothing else, so
 * capture, a pass that rewrites a node, and a future loader all reach the same
 * code, and a rewritten descriptor is honored because it is the only place the
 * operation is recorded.
 *
 * @par Operand-passing convention
 * Operands come from the node's own @ref Node::inputs and @ref Node::outputs
 * lists, positionally, in the order capture records them. Descriptors do NOT
 * carry a second copy of the operand ids.
 *
 * The reason is that the dataflow lists are what every pass already rewrites -
 * CSE's redirect, DeadNodeElimination, Reorder's hazard scan, InputSlicing -
 * so a descriptor copy of the same ids would be a second place to keep in step
 * and would desync exactly the way the baked prefactors did.
 * ``ViewDescriptor::parent_id`` is the exception that proves the rule: a view's
 * parent is a STRUCTURAL relation (it sets ``TensorHandle::aliases``) that
 * outlives the node's dataflow, not an operand slot.
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
 *   which is the RMW convention of bug-1009 and carries no operand of its own.
 * - Dot: ``A = inputs[0]``, ``B = inputs[1]``, ``result = outputs[0]``.
 * - Trace: ``A = inputs[0]``, ``result = outputs[0]``.
 * - Gemm: ``A = inputs[0]``, ``B = inputs[1]``, ``C = outputs[0]``. An
 *   accumulating gemm (nonzero beta) repeats C as ``inputs[2]``, the same RMW
 *   convention the einsum uses.
 * - WriteParam: ``source = inputs[0]``, no outputs, and no operands at all on
 *   the expression arm. Its whole effect is a write into the @ref ParamTable,
 *   which no TensorId names.
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
 * graph's lifetime, and the slot is what ``Graph::rebind`` and
 * ``Graph::redirect_slot`` repoint, so a built executor follows both for free -
 * the property ``Optimizer.hpp`` warns pass authors to preserve by hand. The
 * live geometry comes from @ref TensorSlot::impl_of, a function pointer, so a
 * replay pays one indirect call per operand and allocates nothing. An operand
 * with no slot falls back to @ref TensorHandle::impl_fn, resolved at build time
 * as well.
 *
 * @par How live-mutable scalars reach a built executor
 * Through the descriptor's shared params block (@ref ElementwiseParams,
 * @ref AxpbyParams), which the executor holds by ``shared_ptr`` and reads on
 * every call. Pointing at the node's ``op_data`` directly is not an option:
 * nodes live in a ``std::vector`` that passes insert into, erase from and
 * reorder, so a pointer into one is dangling by construction. The shared block
 * is the pattern @ref EinsumDescriptor and @ref AxpbyDescriptor already
 * established and it survives every node move.
 *
 * The division of labour that follows is worth stating once: a pass that
 * rewrites a PREFACTOR writes the live params and needs no rebuild; a pass that
 * rewrites a STRUCTURAL field (index lists, operands) rewrites the descriptor
 * and calls @ref build_executor again.
 *
 * @par Dispatch routes
 * Every kind converted so far takes the RANK-ERASED route: the kernel is
 * reached through ``einsums::detail::TensorImpl<T>``, which carries data, dims
 * and strides as runtime values, so one dtype dispatch covers every rank and
 * no static-rank cast is needed. @p rank is therefore validated rather than
 * dispatched on. It stays in the key because the design's lowering point is
 * keyed on it.
 *
 * ``Gemm`` is rank-erased too, and it is worth saying which entry it reaches,
 * because ``linear_algebra`` has two that differ. The tensor-object overload
 * taking transpose CHARS carries a symmetry-aware fast path that dispatches a
 * declared-symmetric operand to ``symm``/``hemm``; the ``TensorImpl`` overload
 * beneath it does not. Nothing is lost by taking the lower entry, because none
 * of the three ``cg::gemm`` capture overloads ever reached the higher one:
 * the two bool overloads go straight to the impl-level kernel, and the
 * char overload's runtime-rank arm does the same. The builder therefore lands
 * on the same ``linear_algebra::detail::gemm(char, char, alpha, impl, impl,
 * beta, impl*)`` all three already used, which is what makes the conversion
 * bitwise identical rather than merely close. No @ref dispatch_by_rank is
 * needed anywhere: a gemm is rank-2 by definition and the impl entry checks it.
 *
 * ``Einsum`` takes the same route, by wrapping each operand's live impl in a
 * ``RuntimeTensorView<T>``. That is one dtype dispatch over the whole
 * dispatch cascade instead of one instantiation per (dtype, rank_a, rank_b,
 * rank_c) triple, and it is what pass-rebuilt einsum nodes have always done.
 * The three views are built ONCE, at build time, and re-seated from the live
 * impls on each call, so a replay pays three ``TensorImpl`` assignments -
 * which reuse the ``std::vector`` capacity they already hold - rather than
 * three constructions. See ``build_einsum``.
 *
 * @par Scalar destinations
 * ``Dot`` and ``Trace`` write ONE number, and the tensor that number lands in
 * is a rank-0 handle the capture site registered with
 * ``CaptureContext::get_or_register_scalar`` -- a plain ``T *``, with no
 * ``TensorImpl`` and therefore no slot. Their executors used to close over that
 * pointer directly, which is the baked-lambda class again: the graph knew the
 * destination and the executor did not consult it. @ref ScalarAccessor is the
 * answer, and it resolves the same two ways round: a real tensor destination
 * (``dot_python`` hands the graph a rank-1 tensor so Python has a graph-native
 * scalar handle) goes through its slot and gets element 0, while a registered
 * raw scalar goes through @ref TensorHandle::tensor_ptr, read on every call so
 * that repointing the handle moves the write.
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

    /// Read through @p slot, which must carry a non-null @ref TensorSlot::impl_of.
    explicit OperandAccessor(TensorSlot *slot) : _slot(slot) {}

    /// Read through a handle's own rank-erased accessor.
    explicit OperandAccessor(std::function<void *()> handle_impl) : _handle_impl(std::move(handle_impl)) {}

    /// Whether this accessor was bound to anything.
    [[nodiscard]] bool valid() const noexcept { return _slot != nullptr || static_cast<bool>(_handle_impl); }

    /// The operand's CURRENT rank-erased geometry, type-erased.
    [[nodiscard]] void *raw() const { return _slot != nullptr ? _slot->impl_of(_slot->ptr) : _handle_impl(); }

    /// The operand's CURRENT rank-erased geometry.
    template <typename T>
    [[nodiscard]] ::einsums::detail::TensorImpl<T> *impl() const {
        return static_cast<::einsums::detail::TensorImpl<T> *>(raw());
    }

  private:
    TensorSlot             *_slot{nullptr};
    std::function<void *()> _handle_impl;
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
 * ``Graph::rebind`` and ``Graph::redirect_slot`` for free. The raw-scalar route
 * reads @ref TensorHandle::tensor_ptr on every call rather than copying it at
 * build time, so repointing the handle moves the write -- which is exactly what
 * the raw pointer these executors used to close over could not do.
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
 * The ONE derivation. Capture (``cg::einsum``) and @ref Graph::make_einsum_node
 * both call this, because the two used to carry independent copies of the same
 * gate and the same m/n/k arithmetic, and a hint that describes a matrix
 * product the einsum does not perform is invisible until GEMMBatching forms a
 * batch from it and ``gemm_batch`` silently miscomputes.
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
 *       @ref reconstruction_blocker for the per-node answer.
 *
 * @note ``WriteParam`` was the first kind where the true bit covers only PART of
 *       what the kind records, and the split is worth stating because it is not
 *       the tiled-variant story. A tiled node is a different operation wearing a
 *       shared kind; ``write_param``'s arms are one operation reading its value
 *       several ways. Four kinds follow that precedent now, and in every one of
 *       them the arm that blocks is a closure held in an otherwise data-shaped
 *       descriptor:
 *
 *       - ``WriteParam``: a @ref BoundExpr source, whose ``Callback`` arm is a
 *         ``std::function`` a file cannot hold.
 *       - ``Conditional`` and ``Loop``: a @ref PredExpr predicate, same story.
 *         Note what does NOT block them: holding a SUBGRAPH is not a blocker
 *         here. Whether that subgraph can itself be written is a separate
 *         question, and the milestone that answers it (the design's C) answers
 *         it for the graph, not for the node that names it.
 *       - ``ElementTransform``: a named kernel rebuilds; the lambda-taking
 *         overloads record no descriptor at all and cannot.
 *
 *       So the kind is reconstructible and a callback-arm node is not, and only
 *       @ref reconstruction_blocker can tell them apart. @ref build_executor
 *       builds BOTH: a closure sitting in a descriptor is still content the
 *       builder can lower in this process, and refusing it there would only
 *       force the capture site to hand-bake a lambda again, which is the thing
 *       this file exists to stop.
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
