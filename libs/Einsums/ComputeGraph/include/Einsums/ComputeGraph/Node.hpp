//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace einsums::compute_graph {

class Graph; // Forward declaration for ConditionalDescriptor/LoopDescriptor

// NodeId, OpKind, Target, and simple descriptor types are now defined in
// <Einsums/ComputeGraphTypes/Enums.hpp>, <Einsums/ComputeGraphTypes/Ids.hpp>,
// and <Einsums/ComputeGraphTypes/Descriptors.hpp>.

/**
 * @brief Metadata for Einsum nodes, enabling optimization passes.
 *
 * Stores the contraction pattern (which indices belong to A, B, C, which are
 * link/target indices) and the scalar prefactors. This metadata is used by:
 * - ScaleAbsorption to absorb scale operations into the einsum's C prefactor
 * - CSE to detect duplicate computations
 * - ChainParenthesization to detect GEMM chains
 *
 * @see packed_gemm::ContractionSpec for the contraction topology format
 */
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

struct EinsumDescriptor {
    packed_gemm::ContractionSpec spec;                    ///< Contraction topology: index lists, link/target classification
    PrefactorScalar              c_prefactor{double{0}};  ///< C prefactor (snapshot of EinsumParams::c_pf at capture time)
    PrefactorScalar              ab_prefactor{double{1}}; ///< AB prefactor (snapshot of EinsumParams::ab_pf at capture time)
    bool                         conj_a{false};           ///< Whether to conjugate A (for complex types)
    bool                         conj_b{false};           ///< Whether to conjugate B (for complex types)

    /// Live-mutable index state shared with the executor lambda.
    /// Optimization passes (PermuteFusion, future index rewriters) mutate
    /// this in place; the executor dereferences it on every call, so
    /// rewrites take effect on the next `graph.execute()`. The
    /// `spec` field above is the at-capture snapshot. Analysis passes
    /// can read either, but rewriters must update both to keep
    /// downstream analysis consistent.
    std::shared_ptr<EinsumIndices> indices;

    /// BLAS-level batching hint; non-null only for 2D×2D→2D contractions
    /// with one link index. Read by the GEMMBatching pass.
    std::shared_ptr<GemmHint> gemm_hint;
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
 * - Drop: pick a single index; the result loses this dimension.
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

/**
 * @brief Type-erased operation metadata variant.
 *
 * Each Node stores an OpData that may contain operation-specific metadata
 * for use by optimization passes. Nodes with no special metadata use
 * std::monostate.
 */
using OpData = std::variant<std::monostate, EinsumDescriptor, ScaleDescriptor, PermuteDescriptor, ConditionalDescriptor, LoopDescriptor,
                            AllocDescriptor, TransferDescriptor, DiskIODescriptor, CommDescriptor, InitializeDescriptor,
                            BatchedGemmDescriptor, ViewDescriptor, WriteParamDescriptor>;

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

} // namespace einsums::compute_graph
