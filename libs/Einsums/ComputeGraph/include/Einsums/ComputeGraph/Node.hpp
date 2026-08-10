//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph; // Forward declaration for ConditionalDescriptor/LoopDescriptor

// NodeId, OpKind, Target, and simple descriptor types are now defined in
// <Einsums/ComputeGraphTypes/Enums.hpp>, <Einsums/ComputeGraphTypes/Ids.hpp>,
// and <Einsums/ComputeGraphTypes/Descriptors.hpp>.

/**
 * @brief BLAS-level batching hint for 2D×2D→2D einsums.
 *
 * Populated at capture time when a contraction matches the GEMM
 * pattern (two rank-2 inputs, one rank-2 output, one link index). The
 * GEMMBatching pass reads this to decide which einsums can be
 * collapsed into a single `blas::gemm_batch` call. Non-GEMM
 * contractions leave `gemm_hint == nullptr`.
 *
 * The `extract_*` callbacks resolve the tensor's live pointer + leading
 * dimension at execute time (handles `graph.rebind()` correctly). Type
 * erasure: they return `void*` + `int`; the batched executor casts to
 * the concrete type based on the @ref BlasScalar tag.
 */
struct GemmHint {
    BlasScalar                                    scalar;       ///< Element type for gemm_batch dispatch.
    int                                           m{0};         ///< Rows of C (and A if trans_a=='N').
    int                                           n{0};         ///< Cols of C (and B if trans_b=='N').
    int                                           k{0};         ///< Link dimension.
    char                                          trans_a{'N'}; ///< Transpose flag for A derived from its index order.
    char                                          trans_b{'N'}; ///< Transpose flag for B.
    std::function<std::pair<void const *, int>()> extract_a;    ///< Returns (data_ptr, lda) at call time.
    std::function<std::pair<void const *, int>()> extract_b;    ///< Returns (data_ptr, ldb) at call time.
    std::function<std::pair<void *, int>()>       extract_c;    ///< Returns (data_ptr, ldc) at call time.
};

/**
 * @brief Metadata for Einsum nodes, enabling optimization passes.
 *
 * Stores the contraction pattern (which indices belong to A, B, C, which are
 * link/target indices) and the scalar prefactors. This metadata is used by:
 * - ScaleAbsorption to spot dead scales (einsum with c_prefactor == 0 overwrites them)
 * - CSE to detect duplicate computations
 * - ContractionPlanning to detect and restructure GEMM chains
 *
 * @see packed_gemm::ContractionSpec for the contraction topology format
 */
struct EinsumDescriptor {
    packed_gemm::ContractionSpec spec;                    ///< Contraction topology: index lists, link/target classification
    PrefactorScalar              c_prefactor{double{0}};  ///< C prefactor (snapshot of EinsumParams::c_pf at capture time)
    PrefactorScalar              ab_prefactor{double{1}}; ///< AB prefactor (snapshot of EinsumParams::ab_pf at capture time)
    bool                         conj_a{false};           ///< Whether to conjugate A (for complex types)
    bool                         conj_b{false};           ///< Whether to conjugate B (for complex types)

    /// Live-mutable index state shared with the executor lambda.
    ///
    /// Optimization passes (PermuteFusion, future index rewriters) mutate
    /// this in place; the executor dereferences it on every call, so
    /// rewrites take effect on the next `graph.execute()`. The
    /// `spec` field above is the at-capture snapshot. Analysis passes
    /// can read either, but rewriters must update both to keep
    /// downstream analysis consistent.
    std::shared_ptr<EinsumIndices> indices;

    /// Live-mutable scalar state shared with the executor lambda, same
    /// pattern as `indices`.
    ///
    /// CPU executors read prefactors from here on
    /// every call; `c_prefactor`/`ab_prefactor` above are the at-capture
    /// snapshots (still read by GPU dispatch). Graph::update_prefactors
    /// writes both through this handle so the node stays self-contained
    /// under passes that reorder or remove nodes.
    std::shared_ptr<EinsumParams> params;

    /// BLAS-level batching hint; non-null only for 2D×2D→2D contractions
    /// with one link index. Read by the GEMMBatching pass.
    std::shared_ptr<GemmHint> gemm_hint;
};

/// Live-mutable scalar state for axpby (Y = alpha*X + beta*Y), shared with the
/// executor lambda - the same snapshot + shared-params pattern as EinsumParams.
///
/// The executor reads alpha/beta from here on every call, so a pass that folds a
/// scale into an axpby writes beta through this handle and the change takes
/// effect on the next `graph.execute()`. A snapshot-only descriptor would leave
/// the executor reading a baked value.
struct AxpbyParams {
    PrefactorScalar alpha{double{1}};
    PrefactorScalar beta{double{0}};
};

/// Metadata for Axpby nodes (Y = alpha*X + beta*Y). Prefactors are type-erased
/// (PrefactorScalar) so complex axpby folds exactly, matching EinsumDescriptor.
struct AxpbyDescriptor {
    PrefactorScalar              alpha{double{1}}; ///< alpha snapshot (at-capture value)
    PrefactorScalar              beta{double{0}};  ///< beta snapshot (at-capture value)
    std::shared_ptr<AxpbyParams> params;           ///< live values the executor reads each call
};

/**
 * @brief Metadata for a TILED einsum node.
 *
 * A tiled contraction records as ``OpKind::Custom`` and executes per tile through
 * ``detail::tiled_runtime_einsum``. This carries the live state that executor
 * reads, so a pass can inspect and rewrite the operation instead of treating the
 * node as an opaque closure.
 *
 * Deliberately a DISTINCT type from @ref EinsumDescriptor rather than a reuse of
 * it. A whole-tiled contraction is not a dense one -- its operands have no single
 * contiguous buffer -- and several passes probe ``get_if<EinsumDescriptor>``
 * without first checking the node kind, so handing the tiled node an
 * EinsumDescriptor would expose it to passes that assume one buffer per tensor.
 * With its own type it stays invisible to all of them, and only a pass that
 * explicitly asks for a tiled einsum finds it.
 *
 * The operand and output TensorIds are on the node itself; the element type comes
 * from the tensor handles.
 */
struct TiledEinsumDescriptor {
    /// Live index lists, shared with the executor: a pass that rewrites these
    /// changes what the next ``graph.execute()`` contracts.
    std::shared_ptr<EinsumIndices> indices;
    /// Live prefactors, shared with the executor. Same contract as
    /// @ref EinsumDescriptor::params.
    std::shared_ptr<EinsumParams> params;
};

/**
 * @brief Metadata for a TILED permute node (``OpKind::Custom``).
 *
 * A tiled ``C = beta*C + alpha*P(A)`` records as Custom and executes through
 * ``detail::tiled_permute``; this descriptor is what lets TiledExpansion lower
 * it into per-tile dense Permute nodes instead of treating the node as an
 * opaque closure (which strands every tensor it touches out of expansion).
 *
 * A distinct type from @ref PermuteDescriptor for the same reason the tiled
 * einsum has its own: passes probe ``get_if<PermuteDescriptor>`` without
 * checking the node kind and then reason about a single dense buffer, which a
 * tiled operand does not have. The scalars are snapshots matching the baked
 * executor, exactly as the dense permute capture stores them.
 */
struct TiledPermuteDescriptor {
    std::vector<std::string> c_indices;        ///< Output index names
    std::vector<std::string> a_indices;        ///< Input index names
    PrefactorScalar          alpha{double{1}}; ///< Source prefactor
    PrefactorScalar          beta{double{0}};  ///< Destination prefactor (0 = overwrite)
};

/**
 * @brief Metadata for a TILED dot / dotc node (``OpKind::Dot``).
 *
 * A tiled scalar reduction sums per-tile dense dots over the operands' shared
 * tiles into a dense 1-element result. Without a descriptor the node is an
 * opaque closure whose whole-tensor tiled reads strand every tensor it
 * touches out of TiledExpansion; with it, the pass can lower the node into a
 * single reduction over PER-TILE ids, freeing the whole-tensor ids entirely.
 */
struct TiledDotDescriptor {
    bool conjugated{false}; ///< true for dotc (sum conj(A)*B), false for dot
};

/// Which elementwise operation a @ref TiledElementwiseDescriptor describes.
enum class TiledElementwiseOp : std::uint8_t {
    Scale,  ///< ``A = alpha * A``
    Axpy,   ///< ``Y = Y + alpha * X``
    Divide, ///< ``C = alpha * (A / B) + beta * C``
};

/// Live scalars for a tiled elementwise node, shared with its executor.
struct TiledElementwiseParams {
    PrefactorScalar alpha{double{1}};
    PrefactorScalar beta{double{0}}; ///< Divide only; unused by Scale and Axpy.
};

/**
 * @brief Metadata for a TILED elementwise node (scale or axpy).
 *
 * Like @ref TiledEinsumDescriptor, these record as ``OpKind::Custom`` and run
 * per tile, so without a descriptor the operation and its scalar are sealed
 * inside a closure. Carrying them lets TiledExpansion lower the node into one
 * dense op per populated tile.
 *
 * A distinct type from @ref ScaleDescriptor and @ref AxpbyDescriptor for the
 * same reason the tiled einsum has its own: passes probe ``get_if<...>`` for
 * those without checking the node kind, and a tiled operand has no single
 * buffer for them to work on.
 *
 * The operand TensorIds are on the node itself. A Scale reads and writes one
 * tensor, listed in both @ref Node::inputs and @ref Node::outputs; an Axpy
 * reads X and Y and writes Y, so Y appears in both lists too. A Divide reads A
 * and B, writes C, and additionally reads C when ``beta != 0``.
 */
struct TiledElementwiseDescriptor {
    TiledElementwiseOp op{TiledElementwiseOp::Scale};
    /// Live scalar, shared with the executor: a pass that rewrites this changes
    /// what the next ``graph.execute()`` computes.
    std::shared_ptr<TiledElementwiseParams> params;
};

/**
 * @brief Metadata for conditional (if-then-else) nodes.
 *
 * Contains a predicate function and two subgraphs. The predicate is evaluated
 * at execution time; if true, the then_branch executes, otherwise else_branch.
 * The predicate can inspect tensor values and external state.
 *
 * @code
 * auto [then_graph, else_graph] = graph.add_conditional([&]() {
 *     return energy_diff < threshold;
 * });
 * @endcode
 */
struct ConditionalDescriptor {
    std::function<bool()>  predicate;   ///< Evaluated at runtime to select branch
    std::shared_ptr<Graph> then_branch; ///< Executed if predicate() returns true
    std::shared_ptr<Graph> else_branch; ///< Executed if predicate() returns false (may be empty)
};

/**
 * @brief Metadata for loop nodes.
 *
 * Contains a body subgraph executed repeatedly until the condition returns false
 * or max_iterations is reached. The condition is evaluated AFTER each iteration
 * and can inspect tensor values for convergence checking.
 *
 * @code
 * auto &body = graph.add_loop(100, [&](size_t iter) {
 *     return std::abs(energy - energy_old) > 1e-8;
 * });
 * @endcode
 */
struct LoopDescriptor {
    std::shared_ptr<Graph>      body;                    ///< Subgraph to execute each iteration
    size_t                      max_iterations{1000};    ///< Safety limit
    std::function<bool(size_t)> condition;               ///< After each iter: true=continue, false=stop
    size_t                      last_iteration_count{0}; ///< Set after execution
};

/**
 * @brief Per-axis specification for a @c View op.
 *
 * Each axis of the parent tensor maps to one of:
 * - Full: keep the entire axis (the result preserves this dimension).
 * - Range: half-open ``[lo, hi)`` slice; the result keeps this dimension
 *               with extent ``hi - lo``.
 * - Drop: pick a single index; the result loses this dimension
 *               (rank-reducing). Honored by the runtime-rank ``view_runtime``;
 *               the typed ``cg::view`` throws on it.
 *
 * Bounds are @ref BoundExpr values, so they may be compile-time constants
 * (``0``, ``5``), references to a Pipeline parameter (``"n_occ"``), or
 * arbitrary callbacks (interpreted-mode only).
 */
struct ViewAxis {
    enum class Kind : std::uint8_t { Full, Range, Drop };
    Kind      kind{Kind::Full};
    BoundExpr lo; ///< Range/Drop start (Drop reads only this).
    BoundExpr hi; ///< Range exclusive end. Unused for Full / Drop.

    static ViewAxis full() { return ViewAxis{.kind = Kind::Full}; }
    static ViewAxis range(BoundExpr lo, BoundExpr hi) { return ViewAxis{.kind = Kind::Range, .lo = std::move(lo), .hi = std::move(hi)}; }
    static ViewAxis drop(BoundExpr i) { return ViewAxis{.kind = Kind::Drop, .lo = std::move(i)}; }
};

/**
 * @brief Metadata for @c View nodes, non-owning slice/alias of another tensor.
 *
 * The output tensor's ``TensorHandle::aliases`` is set to ``parent_id``.
 * Each iteration the executor resolves the per-axis @ref BoundExpr values
 * against the active Pipeline's @ref ParamTable and rebuilds the underlying
 * Einsums TensorView, rebinding the output handle's data/strides/dims.
 */
struct ViewDescriptor {
    TensorId              parent_id{0};   ///< The tensor being sliced.
    std::vector<ViewAxis> axes;           ///< One entry per parent-tensor axis.
    size_t                result_rank{0}; ///< parent.rank - count(Drop). Cached for passes.
    /// Axis permutation: result axis ``i`` reads parent axis ``permutation[i]``
    /// (and ``axes[i]`` slices that parent axis). Empty == identity (no
    /// transpose). Used to express ``.T`` / transpose-via-view as a
    /// graph-registered, parent-aliasing view.
    std::vector<size_t> permutation;
};

/**
 * @brief Metadata for @c WriteParam nodes, explicit dataflow write into a Pipeline parameter.
 *
 * Reads a scalar tensor's value (or evaluates a callback) and stores the
 * result into ``params[name]``. Makes the parameter dependency visible to
 * the scheduler so subsequent @c View nodes that reference the same
 * parameter are correctly ordered.
 */
struct WriteParamDescriptor {
    std::string                   name;         ///< Parameter name to write.
    TensorId                      source_id{0}; ///< Scalar tensor to read (0 if using @ref source_fn).
    std::function<std::int64_t()> source_fn;    ///< Optional: compute the value directly.
};

namespace detail {

/// Fill an EinsumDescriptor's snapshot fields from a parsed spec. Does NOT set
/// the live @c params / @c indices handles or the gemm hint; callers that need a
/// self-contained node should use @ref Graph::make_einsum_node instead of
/// assembling those by hand.
inline EinsumDescriptor build_einsum_descriptor(ParsedEinsumSpec const &parsed, PrefactorScalar c_pf, PrefactorScalar ab_pf,
                                                bool conj_a = false, bool conj_b = false) {
    EinsumDescriptor desc;
    desc.c_prefactor         = c_pf;
    desc.ab_prefactor        = ab_pf;
    desc.conj_a              = conj_a;
    desc.conj_b              = conj_b;
    desc.spec.c_indices      = parsed.c_indices;
    desc.spec.a_indices      = parsed.a_indices;
    desc.spec.b_indices      = parsed.b_indices;
    desc.spec.link_indices   = parsed.link_indices();
    desc.spec.target_indices = parsed.target_indices();
    desc.spec.all_indices    = desc.spec.target_indices;
    desc.spec.all_indices.insert(desc.spec.all_indices.end(), desc.spec.link_indices.begin(), desc.spec.link_indices.end());
    return desc;
}

} // namespace detail

/**
 * @brief Type-erased operation metadata variant.
 *
 * Each Node stores an OpData that may contain operation-specific metadata
 * for use by optimization passes. Nodes with no special metadata use
 * std::monostate.
 */
using OpData = std::variant<std::monostate, EinsumDescriptor, ScaleDescriptor, PermuteDescriptor, ConditionalDescriptor, LoopDescriptor,
                            AllocDescriptor, TransferDescriptor, DiskIODescriptor, CommDescriptor, InitializeDescriptor,
                            BatchedGemmDescriptor, GroupedBatchedGemmDescriptor, ViewDescriptor, WriteParamDescriptor, AxpbyDescriptor,
                            TiledEinsumDescriptor, TiledElementwiseDescriptor, TiledPermuteDescriptor, TiledDotDescriptor>;

/**
 * @brief A single operation node in the computation graph.
 *
 * Each node represents one captured operation (einsum, scale, gemm, etc.).
 * It contains:
 * - A type-erased executor lambda that performs the actual computation
 * - Input/output tensor IDs expressing data dependencies
 * - Operation metadata for optimization passes
 *
 * Nodes are created automatically by the graph-aware operation wrappers
 * in the einsums::compute_graph namespace during capture.
 *
 * @see Graph::add_node()
 * @see CaptureContext::record()
 */
struct Node {
    NodeId      id{0};                ///< Unique identifier assigned by Graph::add_node()
    OpKind      kind{OpKind::Custom}; ///< Operation type for pattern matching
    Target      target{Target::CPU};  ///< Execution target (set by GPUPlacement pass)
    std::string label;                ///< Human-readable label for profiling and debugging

    /**
     * @brief Type-erased executor that performs the captured operation.
     *
     * This lambda captures the fully-resolved template call at capture time.
     * All template parameters (types, ranks, indices) are baked into the lambda.
     * On execution, it simply calls the captured function, no re-dispatch needed.
     *
     * @warning The lambda captures tensor references. All referenced tensors must
     *          outlive the graph. Use Graph::create_tensor() for intermediates.
     */
    std::function<void()> execute;

    /**
     * @brief CPU fallback executor for GPU nodes.
     *
     * When a GPU node's execute() throws, the runtime can fall back to this
     * lambda which performs the same operation on the CPU. Set automatically
     * by GPUPlacement when it promotes a node: the original execute is saved
     * as cpu_fallback before the executor is replaced with a GPU version.
     *
     * Empty for CPU nodes and transfer nodes.
     */
    std::function<void()> cpu_fallback;

    /**
     * @brief Asynchronous start phase for I/O nodes.
     *
     * When set, the DataflowExecutor calls async_start to begin an
     * asynchronous operation (e.g., initiate a disk read) as soon as
     * predecessors complete. Independent compute nodes can then overlap
     * with the I/O. The async_finish lambda is called before any consumer
     * of this node runs.
     *
     * Empty for non-async nodes. SequentialExecutor and OpenMPExecutor
     * ignore this field and call the synchronous execute lambda instead.
     */
    std::function<void()> async_start;

    /**
     * @brief Asynchronous finish/synchronize phase for I/O nodes.
     *
     * When set, waits for the operation started by async_start to complete.
     * Called by the DataflowExecutor before any consumer of this node runs.
     *
     * Empty for non-async nodes.
     */
    std::function<void()> async_finish;

    std::vector<TensorId> inputs;  ///< TensorIds of tensors read by this operation
    std::vector<TensorId> outputs; ///< TensorIds of tensors written by this operation

    /**
     * @brief Operation-specific metadata for optimization passes.
     *
     * Contains EinsumDescriptor, ScaleDescriptor, PermuteDescriptor, or
     * std::monostate for operations without special metadata.
     */
    OpData op_data;

    size_t estimated_flops{0}; ///< Estimated floating-point operations (for cost modeling)
    size_t estimated_bytes{0}; ///< Estimated memory traffic in bytes (for cost modeling)

    /// Stream assignment for async execution (set by StreamAssignment pass).
    /// 0 = default/compute stream, 1 = transfer stream.
    int stream_id{0};
};

/// @brief Names of @ref ParamTable entries this node WRITES - the parameter
///        analogue of @ref Node::outputs.
///
/// A @c WriteParam node's entire effect is this write. It travels through the
/// parameter table rather than through any tensor, so a TensorId-keyed
/// dependency scan cannot see it, and in the callback form the node carries no
/// tensor operands at all - which reads to an unguarded pass as "no inputs,
/// therefore unconditionally movable". Schedulers pair this with
/// @ref param_reads to order parameter writes against their consumers.
[[nodiscard]] inline std::vector<std::string> param_writes(Node const &node) {
    if (auto const *wd = std::get_if<WriteParamDescriptor>(&node.op_data)) {
        return {wd->name};
    }
    return {};
}

/// @brief Names of @ref ParamTable entries this node READS - the analogue of
///        @ref Node::inputs.
///
/// A @c View whose slice bounds name a parameter re-resolves them every time it
/// executes, so it must stay ordered after whatever writes them. Callback-valued
/// bounds are deliberately NOT reported here: they name nothing, so no edge can
/// be derived. Use @ref has_runtime_view_bounds to ask the weaker question "does
/// this slice move at all", which is what hoisting and folding need.
[[nodiscard]] inline std::vector<std::string> param_reads(Node const &node) {
    std::vector<std::string> names;
    auto const              *vd = std::get_if<ViewDescriptor>(&node.op_data);
    if (vd == nullptr) {
        return names;
    }
    auto const add = [&names](BoundExpr const &bound) {
        if (bound.is_param()) {
            names.push_back(bound.param_name());
        }
    };
    for (auto const &ax : vd->axes) {
        add(ax.lo);
        if (ax.kind == ViewAxis::Kind::Range) {
            add(ax.hi);
        }
    }
    return names;
}

/// @brief True when a @c View's slice is resolved from runtime state (a named
///        parameter or a callback) rather than from literals.
///
/// Such a view aliases the same parent every iteration but describes a
/// different slice each time. Passes that would evaluate a node once and reuse
/// the result - constant folding, loop-invariant hoisting - must refuse it: the
/// parent input they inspect is genuinely invariant, and the part that moves is
/// not expressed as dataflow at all.
[[nodiscard]] inline bool has_runtime_view_bounds(Node const &node) {
    if (node.kind != OpKind::View) {
        return false;
    }
    auto const *vd = std::get_if<ViewDescriptor>(&node.op_data);
    if (vd == nullptr) {
        return true; // no descriptor to inspect: assume the slice moves
    }
    return std::ranges::any_of(
        vd->axes, [](ViewAxis const &ax) { return !ax.lo.is_const() || (ax.kind == ViewAxis::Kind::Range && !ax.hi.is_const()); });
}

EINSUMS_NAMESPACE_END(compute_graph)
