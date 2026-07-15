//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/DeviceShadowMap.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/Executor.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <functional>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph {

class PassManager; // Forward declaration
class OptimizerPass;
enum class OptLevel : std::uint8_t; // Optimizer.hpp
struct ParsedEinsumSpec;

/**
 * @brief A directed acyclic graph (DAG) of tensor operations.
 *
 * The Graph class is the central container for the computation graph. It stores
 * a sequence of operation nodes and the tensor handles they reference. Key features:
 *
 * - **Capture**: Operations are added during a capture phase (via CaptureGuard).
 * - **Topological sorting**: Nodes are sorted based on data dependencies before execution.
 * - **Execution**: All nodes execute in dependency order. Calling execute() multiple
 *   times replays the same operations (useful for iterative algorithms).
 * - **Optimization**: Passes can be applied to fuse, eliminate, or reorder nodes.
 * - **Tensor ownership**: Intermediate tensors can be created via create_tensor()
 *   so the graph manages their lifetimes.
 * - **Validation**: Before execution, tensor pointers are checked for use-after-free.
 * - **Profiling**: execute() wraps each node in profiler regions with annotations.
 * - **Visualization**: print_dot() exports GraphViz format; print_summary() prints text.
 *
 * @code
 *
 * namespace cg = einsums::compute_graph;
 *
 * auto A = create_random_tensor<double>("A", 10, 5);
 * auto B = create_random_tensor<double>("B", 5, 8);
 *
 * cg::Graph graph("my_graph");
 * auto &C = graph.create_zero_tensor<double, 2>("C", 10, 8);
 *
 * {
 *     cg::CaptureGuard guard(graph);
 *     cg::einsum("ik;kj->ij", &C, A, B);
 * }
 *
 * graph.execute();   // C = A * B
 *
 * graph.execute();   // Replay: C = A * B again
 *
 * @endcode
 *
 * @see CaptureGuard for the RAII capture mechanism
 * @see Pipeline for multi-stage workflows with loops
 * @see OptimizerPass for graph optimization
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT Graph {
  public:
    /**
     * @brief Construct an empty graph.
     * @param[in] name Human-readable name for profiling and debugging output.
     */
    APIARY_EXPOSE explicit Graph(std::string name = "graph");
    ~Graph();

    Graph(Graph &&other) noexcept;
    Graph &operator=(Graph &&other) noexcept;
    Graph(Graph const &)            = delete;
    Graph &operator=(Graph const &) = delete;

    /**
     * @brief Add an operation node to the graph.
     *
     * Assigns a unique NodeId and appends the node. Marks the graph as unsorted.
     * Typically called internally by CaptureContext::record(), not by users directly.
     *
     * @param[in] node The node to add (moved into the graph).
     * @return The assigned NodeId.
     */
    NodeId add_node(Node node);

    /**
     * @brief Reserve a fresh unique NodeId without adding a node.
     *
     * For passes that build replacement nodes and splice them into the node
     * list directly (at a chosen position, not appended): every node in a
     * graph must carry a unique id - the default Node::id of 0 collides with
     * the first captured node and corrupts anything keyed by id (dependency
     * bookkeeping, the pass program-order validator, profile strings).
     */
    NodeId reserve_node_id() { return _next_node_id++; }

    /**
     * @brief Register a tensor handle with the graph.
     *
     * Assigns a unique TensorId and stores the handle's metadata. The tensor itself
     * is NOT copied, only a pointer and metadata are stored.
     *
     * @param[in] handle The tensor handle to register (moved into the graph).
     * @return The assigned TensorId.
     */
    TensorId register_tensor(TensorHandle handle);

    /**
     * @brief Look up a tensor handle by its TensorId.
     * @param[in] id The tensor identifier.
     * @return Reference to the TensorHandle.
     * @throws std::out_of_range If no tensor with the given ID exists.
     */
    [[nodiscard]] TensorHandle       &tensor(TensorId id);
    [[nodiscard]] TensorHandle const &tensor(TensorId id) const; ///< @overload

    /**
     * @brief Execute all nodes in topological order.
     *
     * Performs topological sorting (if not already sorted), validates that all
     * tensor pointers are still alive, then executes each node's lambda in order.
     *
     * Can be called multiple times to replay the same computation sequence.
     * All tensors must have the same dimensions as at capture time.
     *
     * @throws std::runtime_error If a tensor appears to have been destroyed (use-after-free detected).
     * @throws std::runtime_error If a cycle is detected during topological sort.
     */
    APIARY_EXPOSE APIARY_RELEASE_GIL void execute();

    /**
     * @brief Execute using a custom executor.
     *
     * Performs topological sorting and validation, then delegates
     * node execution to the provided executor.
     *
     * @param[in] executor The execution backend (e.g., OpenMPExecutor).
     *
     * @note The GIL is released for the duration of execution. The parallel
     *       executors run nodes on worker threads; any node that invokes a
     *       Python callback (e.g. element_transform) re-acquires the GIL from
     *       its worker. Holding the GIL here would deadlock that re-acquire
     *       against the waiting main thread.
     */
    APIARY_EXPOSE APIARY_RELEASE_GIL void execute(Executor &executor);

    // Note: execute() always instruments with the profiler (no separate execute_profiled variant).

    /**
     * @brief Apply a PassManager (ordered sequence of passes) to the graph.
     *
     * Runs all passes in the PassManager in order.
     *
     * @param[in,out] pm The pass manager to run.
     * @return True if any pass modified the graph.
     *
     * @code
     * auto pm = cg::PassManager::create_default();
     * graph.apply(pm);
     * @endcode
     */
    APIARY_EXPOSE bool apply(PassManager &pm);

    /**
     * @brief Optimize with the full default pipeline (OptLevel::O2).
     *
     * Compiler-style front door over PassManager: builds the pipeline, runs
     * it, and records a human-readable summary retrievable via explain().
     *
     * @code
     * graph.optimize();
     * std::cout << graph.explain();
     * @endcode
     */
    APIARY_EXPOSE bool optimize();

    /// @overload Optimize at a specific level (O0 none, O1 cleanup, O2 full).
    APIARY_EXPOSE bool optimize(OptLevel level);

    /**
     * @brief What the last optimize() did, as a human-readable report.
     *
     * Node counts before/after plus the per-pass highlights (chains
     * restructured with estimated speedups, buffers merged in place, Frees
     * inserted, arena size vs the buffers it hosts, batches formed and
     * profitability-gate skips). Empty until optimize() has run.
     */
    APIARY_EXPOSE APIARY_GETTER("explain") [[nodiscard]] std::string const &explain() const { return _last_optimize_report; }

    /**
     * @brief Apply a single pass by type (convenience).
     *
     * Creates the pass, runs it, and returns a pair of (modified, pass).
     * Useful for single-pass application and retrieving analysis results.
     *
     * @tparam PassType The pass class.
     * @return Pair of (was_modified, pass_instance).
     *
     * @code
     * auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
     * mem.print_report(std::cout);
     * @endcode
     */
    template <typename PassType>
    std::pair<bool, PassType> apply() {
        PassType pass;
        bool     modified = pass.run(*this);
        return {modified, std::move(pass)};
    }

    // Note: use apply(PassManager&) or apply<PassType>() for optimization passes.

    /**
     * @brief Validate that tensor dimensions are compatible between connected nodes.
     *
     * For each node, checks that input tensor ranks match the expected number
     * of indices (for Einsum nodes). Called automatically at end of capture.
     *
     * @throws std::runtime_error If a shape mismatch is detected.
     */
    void validate_shapes_at_capture() const;

    /**
     * @brief Per-node timing entry.
     */
    struct NodeTiming {
        NodeId      id;
        std::string label;
        OpKind      kind;
        double      duration_ms{0.0}; ///< Wall-clock time in milliseconds
    };

    /**
     * @brief Print a timing report from the last execute() call.
     *
     * Shows per-node wall-clock times sorted by duration (longest first).
     * Only populated after calling execute().
     *
     * @param[out] os Output stream.
     */
    void print_timing_report(std::ostream &os) const;

    /// Access the timing data from the last execute() call.
    [[nodiscard]] std::vector<NodeTiming> const &timing_report() const { return _timing_report; }

    /**
     * @brief Record timing for a single node (used by custom Executors).
     *
     * Custom executors should call this after executing each node so that
     * print_timing_report() works correctly.
     */
    void record_node_timing(NodeId id, std::string const &label, OpKind kind, double duration_ms) {
        _timing_report.push_back({.id = id, .label = label, .kind = kind, .duration_ms = duration_ms});
    }

    /// Clear timing data (called at the start of execute()).
    void clear_timing_report() { _timing_report.clear(); }

    /**
     * @brief Sort nodes in topological order based on data dependencies.
     *
     * Uses Kahn's algorithm. Edges are inferred from tensor IDs:
     * if node A writes tensor T and node B reads tensor T (with A before B
     * in the original order), then A must execute before B.
     *
     * @throws std::runtime_error If a dependency cycle is detected.
     */
    void topological_sort();

    /**
     * @brief Check that all registered tensors still have their capture-time dimensions.
     * @return True if all shapes match. (Currently a placeholder, always returns true.)
     */
    [[nodiscard]] bool validate_shapes() const;

    /**
     * @brief Export the graph in GraphViz DOT format.
     *
     * Tensor nodes are drawn as rectangles, operation nodes as ellipses.
     * Edges show data flow from tensors to operations and back.
     *
     * @param[out] os Output stream to write DOT content to.
     *
     * @code
     * std::ofstream f("graph.dot");
     * graph.print_dot(f);
     * // Then: dot -Tpng graph.dot -o graph.png
     * @endcode
     */
    void print_dot(std::ostream &os) const;

    /**
     * @brief Print a human-readable summary of the graph.
     *
     * Lists all nodes with their operation kind, label, and input/output tensor names.
     *
     * @param[out] os Output stream.
     */
    void print_summary(std::ostream &os) const;

    /**
     * @brief Serialize the graph structure as a JSON string.
     *
     * Produces a JSON object with:
     * - "name": graph name
     * - "tensors": array of {id, name, rank, dims, element_size, dtype, is_intermediate}
     * - "nodes": array of {id, kind, label, inputs, outputs, timing_ms}
     * - "edges": array of {from_node, to_node, tensor_id} (data dependency edges)
     *
     * Used by the profile viewer to render an interactive node graph.
     * Can be called at any point, edges are computed on-the-fly from node
     * inputs/outputs, so this works both before and after execute().
     *
     * @return JSON string representation of the graph structure.
     */
    APIARY_EXPOSE [[nodiscard]] std::string to_json() const;

    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string const &name() const { return _name; } ///< Graph name.

    // Parent context for profiler hierarchy (Workspace > Pipeline > Graph)
    void set_pipeline_name(std::string const &n) { _pipeline_name = n; }
    void set_workspace_name(std::string const &n) { _workspace_name = n; }
    void set_stage_name(std::string const &n) { _stage_name = n; }
    void set_stage_type(std::string const &t) { _stage_type = t; } ///< "graph" or "loop"
    void set_stage_index(int idx) { _stage_index = idx; }

    /// Hand a custom cleanup function to the graph. Used by capture-time
    /// helpers (e.g., ``cg::view``) that allocate auxiliary objects on the
    /// heap whose lifetime must match the graph's. The deleter runs when
    /// the graph is destroyed.
    void adopt(std::function<void()> deleter);

    /// Set/read the @ref ParamTable used by the View executor and
    /// ``BoundExpr::Param`` resolution. Pipeline plumbs its own table
    /// down to each stage Graph at construction. Standalone graphs get
    /// a default empty table; callers can replace it via
    /// ``set_params_ptr`` if they want to share with another scope.
    void                                             set_params_ptr(std::shared_ptr<ParamTable> params) { _params = std::move(params); }
    [[nodiscard]] std::shared_ptr<ParamTable> const &params_ptr() const { return _params; }
    [[nodiscard]] int                                stage_index() const { return _stage_index; }
    [[nodiscard]] std::string const                 &pipeline_name() const { return _pipeline_name; }
    [[nodiscard]] std::string const                 &workspace_name() const { return _workspace_name; }
    [[nodiscard]] std::string const                 &stage_name() const { return _stage_name; }
    [[nodiscard]] std::string const                 &stage_type() const { return _stage_type; }
    [[nodiscard]] std::vector<Node> const           &nodes() const { return _nodes; } ///< Read-only access to nodes.
    [[nodiscard]] std::vector<Node>                 &nodes() { return _nodes; }       ///< Mutable access (for optimization passes).
    APIARY_EXPOSE [[nodiscard]] size_t               num_nodes() const { return _nodes.size(); }     ///< Number of operation nodes.
    APIARY_EXPOSE [[nodiscard]] size_t               num_tensors() const { return _tensors.size(); } ///< Number of registered tensors.

    /// Read-only access to the tensor registry (TensorId → TensorHandle map).
    [[nodiscard]] std::unordered_map<TensorId, TensorHandle> const &tensors_map() const { return _tensors; }
    /// Mutable access to the tensor registry (for testing / optimization passes).
    [[nodiscard]] std::unordered_map<TensorId, TensorHandle> &tensors_map() { return _tensors; }

    /**
     * @brief Invoke @p visitor on each immediate child sub-graph.
     *
     * Walks this graph's nodes and, for every control-flow node, hands the
     * visitor a mutable reference to each owned sub-graph:
     *   - ``OpKind::Loop``         → ``LoopDescriptor::body``
     *   - ``OpKind::Conditional``  → ``ConditionalDescriptor::then_branch`` and
     *                                ``else_branch`` (when non-null)
     *
     * Does not recurse into nested control flow, visitor must do its own
     * descent if it wants the whole sub-tree. The order of visits is the node
     * order in the parent graph (with then-branch visited before else-branch
     * for conditional nodes).
     *
     * Used by optimization passes that need to look inside or run on
     * sub-graphs. PassManager::run() calls this when a pass overrides
     * ``OptimizerPass::recurse_into_subgraphs()`` to true.
     */
    void for_each_subgraph(std::function<void(Graph &)> const &visitor);

    /// Const overload of @ref for_each_subgraph. Visitor sees ``Graph const &``.
    void for_each_subgraph(std::function<void(Graph const &)> const &visitor) const;

    /**
     * @brief Collect the underlying tensor pointers referenced anywhere in
     *        this graph's descendant sub-graphs.
     *
     * Walks every loop body / conditional branch (recursively) and inserts
     * each referenced tensor's ``TensorHandle::tensor_ptr`` into @p out.
     * Does not include this graph's own node references, only its
     * descendants'.
     *
     * Why pointers, not TensorIds: each Graph assigns its own TensorIds, so
     * the same underlying tensor used in a parent and in a nested body has
     * *different* ids in each map. The ``tensor_ptr`` is the stable identity
     * across graphs.
     *
     * Used by passes that must treat a tensor consumed only by a nested
     * control-flow body as live, e.g. DeadNodeElimination, which would
     * otherwise eliminate the producer of a tensor that only a nested loop
     * reads (a Loop node does not list its body's tensor reads as inputs).
     */
    void collect_subtree_referenced_ptrs(std::unordered_set<void const *> &out) const;

    /**
     * @brief Effective (scheduling) inputs/outputs of a node.
     *
     * For ordinary nodes this is just ``{node.inputs, node.outputs}``. For a
     * Loop or Conditional node, whose declared input/output lists are empty
     * because the body/branches are captured after the node is created, this
     * augments them with the tensors the *subtree* reads (→ inputs) and writes
     * (→ outputs), mapped from the subtree's buffer pointers back to this
     * graph's TensorIds.
     *
     * Schedulers (``topological_sort`` and the Reorder pass) must use this
     * instead of the raw node lists, or a control-flow node has no dependency
     * edges and can be floated past a producer/consumer of a tensor its body
     * touches: silently reordering it relative to surrounding ops. The node's
     * own lists are left untouched so structural passes (e.g. DeadNodeElimination)
     * still see control-flow nodes as having no SSA outputs.
     *
     * Not const: a buffer used *only* inside sub-graphs has no TensorId in this
     * (parent) graph, so it would be dropped from the mapping and two
     * control-flow nodes touching the same such buffer would get no edge between
     * them. To give those buffers a stable shared id this registers a handle for
     * them in the parent on first sight (idempotent; the orphan handle is
     * harmless: no parent node references it).
     */
    std::pair<std::vector<TensorId>, std::vector<TensorId>> effective_io(Node const &node);

    /// Memoization store for effective_io_cached(). Keyed by NodeId so
    /// entries stay valid across in-sort node moves. Scoped to a single
    /// topological_sort() call: control-flow subtree I/O changes when passes
    /// move nodes across loop-body boundaries, so it must not persist.
    using EffectiveIoCache = std::unordered_map<NodeId, std::pair<std::vector<TensorId>, std::vector<TensorId>>>;

    /**
     * @brief Resolve a TensorId through its alias chain to the owning buffer.
     *
     * A view's ``TensorHandle::aliases`` points at its parent; this follows that
     * chain (bounded) and returns the underlying owner's id (or @p id unchanged
     * if it isn't a view). Any analysis that reasons about *which buffer* a node
     * reads/writes, scheduling (topological_sort, Reorder), liveness
     * (DeadNodeElimination), must resolve through this, or a write through a
     * view looks unrelated to a read of its parent.
     */
    [[nodiscard]] TensorId resolve_alias(TensorId id) const {
        for (int hops = 0; hops < 32; ++hops) {
            auto it = _tensors.find(id);
            if (it == _tensors.end() || it->second.aliases == 0) {
                return id;
            }
            id = it->second.aliases;
        }
        return id;
    }

    /// Access dependency info (populated by topological_sort()).
    [[nodiscard]] DependencyInfo const &dependencies() const { return _deps; }

    /**
     * @brief Mark the graph as topologically sorted.
     *
     * Called by optimization passes that produce a valid topological ordering
     * (e.g., Reorder) to prevent execute() from re-sorting.
     */
    void mark_sorted() {
        _sorted   = true;
        _executed = false;
        // The caller vouches for the node ORDER, but node positions changed,
        // so the position-keyed _deps lists must be rebuilt on next demand.
        _deps_valid = false;
        // Passes also rewrite labels/descriptors; refresh cached profiler
        // payloads on next execute.
        _profile_strings_valid = false;
    }

    /**
     * @brief Add a conditional (if-then-else) node to the graph.
     *
     * Returns references to the then-branch and else-branch subgraphs.
     * Use CaptureGuard on each to capture operations into the branches.
     * The predicate is evaluated at execution time to select the branch.
     *
     * @param[in] label Human-readable label for profiling.
     * @param[in] predicate Function returning true for then-branch, false for else-branch.
     *                      Can inspect tensor values and external state.
     * @return Tuple of references: (then_branch_graph, else_branch_graph).
     *         A tuple (not a pair) so the method can be bound to Python,
     *         pybind11 can't cast a pair/tuple of *references* to a
     *         non-copyable type cleanly, but a reference-tuple return casts
     *         each branch with ``reference_internal``. Structured bindings
     *         (``auto [t, e] = ...``) work identically for C++ callers.
     *
     * @code
     * auto [then_g, else_g] = graph.add_conditional("converge_check",
     *     [&]() { return delta < threshold; });
     * { CaptureGuard g(then_g); cg::scale(1.0, &result); }
     * { CaptureGuard g(else_g); cg::einsum(...); }
     * @endcode
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) std::tuple<Graph &, Graph &> add_conditional(std::string           label,
                                                                                              std::function<bool()> predicate);

    /**
     * @brief Add a conditional node with lambda-captured branches.
     *
     * The then_fn and else_fn lambdas are called during graph construction
     * to capture operations into the respective branches. This is the
     * preferred API, more concise than the Graph-returning variant.
     *
     * @param[in] label Human-readable label.
     * @param[in] predicate Runtime predicate: true → then, false → else.
     * @param[in] then_fn Lambda capturing then-branch operations.
     * @param[in] else_fn Lambda capturing else-branch operations (optional).
     *
     * @code
     * graph.add_conditional("check", [&]() { return value(0) > 5.0; },
     *     [&]() { cg::scale(0.5, &value); },    // then
     *     [&]() { cg::scale(2.0, &value); }      // else
     * );
     * @endcode
     */
    template <typename ThenFn, typename ElseFn = std::nullptr_t>
    void add_conditional(std::string label, std::function<bool()> predicate, ThenFn &&then_fn,
                         ElseFn &&else_fn = nullptr); // Defined in CaptureContext.hpp

    /**
     * @brief Add a loop node to the graph (Graph-returning variant).
     *
     * Returns a reference to the loop body subgraph. Use CaptureGuard to
     * capture operations into the body.
     *
     * @param[in] label Human-readable label for profiling.
     * @param[in] max_iterations Maximum number of iterations (safety limit).
     * @param[in] condition Called after each iteration with the iteration number.
     * @return Reference to the loop body Graph.
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) Graph &add_loop(std::string label, size_t max_iterations,
                                                                 std::function<bool(size_t)> condition);

    /**
     * @brief Add a loop node with lambda-captured body.
     *
     * The body_fn lambda is called during graph construction to capture
     * operations into the loop body.
     *
     * @param[in] label Human-readable label.
     * @param[in] max_iterations Maximum iterations (safety limit).
     * @param[in] condition After each iteration: true → continue, false → stop.
     * @param[in] body_fn Lambda capturing loop body operations.
     *
     * @code
     * graph.add_loop("converge", 100,
     *     [&](size_t iter) { return value(0) >= 1.0; },
     *     [&]() { cg::scale(0.5, &value); }
     * );
     * @endcode
     */
    template <typename BodyFn>
    void add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition,
                  BodyFn &&body_fn); // Defined in CaptureContext.hpp

    /**
     * @brief Add a loop node, std::function-typed body for Python-friendly binding.
     *
     * Identical semantics to the template-bodied @ref add_loop above; the
     * concrete std::function signature is what pybind11 can bind directly
     * without rvalue-reference deduction issues. C++ callers can keep
     * using either form.
     */
    APIARY_EXPOSE void add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition,
                                std::function<void()> body_fn);

    /**
     * @brief Mark a tensor's lifetime end with a Free node.
     *
     * Inserts a Free node into the graph marking where the tensor is no longer needed.
     * The tensor is NOT actually deallocated (the graph still owns it), this is a
     * marker for the MemoryPlanning pass to identify buffer reuse opportunities.
     *
     * @param[in] id The TensorId (from create_tensor's Alloc node).
     * @param[in] name Tensor name for debugging.
     * @param[in] size_bytes Size of the tensor in bytes.
     */
    void free_tensor(TensorId id, std::string name = "", size_t size_bytes = 0) {
        AllocDescriptor desc;
        desc.tensor_id   = id;
        desc.size_bytes  = size_bytes;
        desc.tensor_name = std::move(name);

        Node node;
        ProfileMemFree(size_bytes);

        node.kind    = OpKind::Free;
        node.label   = fmt::format("free({})", desc.tensor_name);
        node.execute = []() {}; // No-op: graph still owns the memory
        node.inputs  = {id};
        node.op_data = std::move(desc);

        add_node(std::move(node));
    }

    /**
     * @brief Create a tensor owned by the graph.
     *
     * The tensor is heap-allocated and stored in the graph's ``owned_tensors_`` list.
     * It is destroyed when the graph is destroyed. This ensures the tensor outlives
     * all captured lambdas, preventing dangling reference bugs.
     *
     * Prefer scratch() for intermediates: it defers allocation to execution
     * and hands the buffer to the memory passes (FreeInsertion,
     * InplaceOptimization, MemoryPlanning's arena). create_tensor allocates
     * NOW and stays allocated unless those passes intervene.
     *
     * An Alloc node is inserted into the graph marking the tensor's lifetime start.
     * Pair with free_tensor() to mark the lifetime end. The MemoryPlanning pass
     * uses these markers to identify buffer reuse opportunities.
     *
     * @tparam T Element type (e.g., double, float, std::complex<double>).
     * @tparam Rank Number of dimensions.
     * @tparam Dims Dimension size types (must be integral).
     * @param[in] name Human-readable name for the tensor.
     * @param[in] dims Size of each dimension.
     * @return Reference to the newly created tensor.
     *
     * @code
     * cg::Graph graph("example");
     * auto &tmp = graph.create_tensor<double, 2>("tmp", 100, 100);
     * // tmp is valid for the lifetime of graph
     * @endcode
     */
    template <typename T, size_t Rank, typename... Dims>
    Tensor<T, Rank> &create_tensor(std::string name, Dims... dims) {
        auto *ptr = new Tensor<T, Rank>(name, static_cast<size_t>(dims)...);
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<Tensor<T, Rank> *>(p); });

        auto handle            = make_handle(*ptr, 0);
        handle.is_intermediate = true;
        auto id                = register_tensor(std::move(handle));

        AllocDescriptor desc;
        desc.tensor_id   = id;
        desc.size_bytes  = ptr->size() * sizeof(T);
        desc.tensor_name = name;

        Node node;
        ProfileMemAlloc(desc.size_bytes);

        node.kind    = OpKind::Alloc;
        node.label   = fmt::format("alloc({})", name);
        node.execute = []() {};
        node.outputs = {id};
        node.op_data = std::move(desc);

        add_node(std::move(node));
        return *ptr;
    }

    /**
     * @brief Create a zero-initialized tensor owned by the graph.
     *
     * Same as create_tensor() but fills the tensor with zeros after creation.
     *
     * @tparam T Element type.
     * @tparam Rank Number of dimensions.
     * @tparam Dims Dimension size types.
     * @param[in] name Human-readable name.
     * @param[in] dims Size of each dimension.
     * @return Reference to the zero-initialized tensor.
     */
    template <typename T, size_t Rank, typename... Dims>
    Tensor<T, Rank> &create_zero_tensor(std::string name, Dims... dims) {
        auto &t = create_tensor<T, Rank>(std::move(name), dims...);
        t.zero();
        return t;
    }

    /**
     * @brief Create a graph-owned runtime-rank tensor.
     *
     * Runtime-rank analog of create_tensor(). Allocates a RuntimeTensor<T, Alloc>
     * with rank determined by `dims.size()`, no rank-4 cap, no compile-time
     * rank in the type. Adds an Alloc node and marks the handle as intermediate.
     * Lives for the lifetime of the graph.
     *
     * @code
     * auto &tmp = graph.create_runtime_tensor<double>("tmp", {100, 100});
     * auto &big = graph.create_runtime_tensor<double>("big", {2, 3, 4, 5, 6});  // rank-5, fine
     * @endcode
     */
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("create_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("create_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("create_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("create_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &create_runtime_tensor(std::string name, std::vector<size_t> dims,
                                                                          bool intermediate = true) {
        using TensorType = GeneralRuntimeTensor<T, Alloc>;
        auto *ptr        = new TensorType(name, std::move(dims));
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        // ``intermediate`` controls DeadNodeElimination: a graph-owned
        // intermediate with no in-graph consumer is prunable, but a
        // user-visible result (one a caller holds a Python handle to and reads
        // after execute, e.g. the numpy-ergonomics operators' outputs) must
        // be kept even when nothing downstream in the graph reads it.
        auto handle            = make_handle(*ptr, 0);
        handle.is_intermediate = intermediate;
        auto id                = register_tensor(std::move(handle));

        AllocDescriptor desc;
        desc.tensor_id   = id;
        desc.size_bytes  = ptr->size() * sizeof(T);
        desc.tensor_name = name;

        Node node;
        ProfileMemAlloc(desc.size_bytes);

        node.kind    = OpKind::Alloc;
        node.label   = fmt::format("alloc({})", name);
        node.execute = []() {};
        node.outputs = {id};
        node.op_data = std::move(desc);

        add_node(std::move(node));
        return *ptr;
    }

    /// Runtime-rank analog of create_zero_tensor().
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("create_zero_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("create_zero_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("create_zero_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("create_zero_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &create_zero_runtime_tensor(std::string name, std::vector<size_t> dims,
                                                                               bool intermediate = true) {
        auto &t = create_runtime_tensor<T, Alloc>(std::move(name), std::move(dims), intermediate);
        t.zero();
        return t;
    }

    /**
     * @brief Declare a graph-owned runtime-rank tensor with DEFERRED allocation.
     *
     * The runtime-rank, pybind-exposed analog of declare_tensor(): a shell tensor
     * (valid metadata, no data) registered in THIS graph's tensor map with
     * AllocState::Deferred. No Alloc node is inserted, the MaterializationPass
     * inserts Materialize+Initialize at the right position, and MemoryPlanning /
     * FreeInsertion / InplaceOptimization can then plan, reuse, and free its
     * buffer (unlike create_*_tensor, which is allocated eagerly and so is opaque
     * to those passes). Use this for graph/loop-body scratch you want the memory
     * passes to manage.
     */
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &declare_runtime_tensor(std::string name, std::vector<size_t> dims,
                                                                           bool intermediate = false) {
        using TensorType = GeneralRuntimeTensor<T, Alloc>;
        auto *ptr        = new TensorType(typename TensorType::DeferredAlloc{}, name, std::move(dims));
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        auto handle            = make_handle(*ptr, 0);
        handle.is_intermediate = intermediate;
        handle.alloc_state     = AllocState::Deferred;
        handle.materialize_fn  = [ptr]() { ptr->materialize(); };
        handle.release_fn      = [ptr]() { ptr->release(); };
        handle.zero_fn         = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        handle.random_fn = [ptr]() {
            ptr->materialize();
            auto *data = ptr->data();
            for (size_t idx = 0; idx < ptr->size(); idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<T>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
        register_tensor(std::move(handle));
        // No Alloc node, MaterializationPass inserts Materialize + Initialize.
        return *ptr;
    }

    /// Runtime-rank analog of declare_zero_tensor() (graph-owned, deferred, zeroed
    /// at materialize time).
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tensor", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &declare_zero_runtime_tensor(std::string name, std::vector<size_t> dims,
                                                                                bool intermediate = false) {
        auto &t = declare_runtime_tensor<T, Alloc>(std::move(name), std::move(dims), intermediate);
        for (auto &[tid, handle] : _tensors) {
            if (handle.tensor_ptr == &t) {
                handle.init_kind = InitKind::Zero;
                break;
            }
        }
        t.set_pending_init(PendingInit::Zero);
        return t;
    }

    // ── Deferred tensor declaration ─────────────────────────────────────────

    /**
     * @brief Declare a graph-scoped tensor with deferred allocation.
     *
     * Creates a shell tensor (valid metadata, no data) owned by the graph.
     * Data is allocated by the MaterializationPass during apply().
     * Marked as is_intermediate = true.
     *
     * @tparam T     Element type.
     * @tparam Rank  Number of dimensions.
     * @param  name  Human-readable tensor name.
     * @param  dims  Dimensions of each rank.
     * @return Reference to the shell tensor.
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_tensor(std::string name, Dims... dims) {
        using TensorType = Tensor<T, Rank>;
        auto *ptr        = new TensorType(typename TensorType::DeferredAlloc{}, std::move(name), dims...);
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        auto handle             = make_handle(*ptr, 0);
        handle.is_intermediate  = false; // User-visible by default. Use create_tensor for intermediates.
        handle.alloc_state      = AllocState::Deferred;
        handle.materialize_fn   = [ptr]() { ptr->materialize(); };
        handle.release_fn       = [ptr]() { ptr->release(); };
        handle.allreduce_sum_fn = [ptr]() {
            auto span = std::span<T>(ptr->data(), ptr->size());
            (void)comm::allreduce_inplace<T>(span, comm::ReduceOp::Sum);
        };
        handle.resize_deferred_fn = [ptr](std::vector<size_t> const &new_dims) {
            Dim<Rank> d;
            for (size_t i = 0; i < Rank && i < new_dims.size(); i++)
                d[i] = new_dims[i];
            ptr->resize_deferred(d);
        };
        handle.set_distribution_fn = [ptr](std::vector<size_t> const &global_dims, std::vector<size_t> const &offsets) {
            std::array<size_t, Rank> gd{}, off{};
            for (size_t i = 0; i < Rank && i < global_dims.size(); i++)
                gd[i] = global_dims[i];
            for (size_t i = 0; i < Rank && i < offsets.size(); i++)
                off[i] = offsets[i];
            ptr->set_distribution(gd, off);
        };
        handle.zero_fn = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        handle.random_fn = [ptr]() {
            ptr->materialize();
            auto *data = ptr->data();
            for (size_t idx = 0; idx < ptr->size(); idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<T>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
        register_tensor(std::move(handle));

        // No Alloc node inserted, MaterializationPass will insert
        // Materialize + Initialize nodes at the right position.

        return *ptr;
    }

    /// Declare a graph-scoped tensor initialized to zero after materialization.
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_zero_tensor(std::string name, Dims... dims) {
        auto &t = declare_tensor<T, Rank>(std::move(name), dims...);
        // Set init_kind on the handle (just registered, so it's the last one)
        for (auto &[tid, handle] : _tensors) {
            if (handle.tensor_ptr == &t) {
                handle.init_kind = InitKind::Zero;
                break;
            }
        }
        return t;
    }

    /**
     * @brief Create graph-managed scratch: THE way to make an intermediate.
     *
     * One call replaces the create_tensor / declare_tensor / intermediate-flag
     * decision tree. A scratch tensor is:
     * - **deferred**: no allocation until execution reaches it (the
     *   Materialization pass, or the graph's own lifecycle nodes, allocate it
     *   at the right position - possibly resized to a local partition by
     *   DistributionPlanning first);
     * - **intermediate**: FreeInsertion reclaims it after its last consumer,
     *   InplaceOptimization may merge its storage into a dying input, and
     *   MemoryPlanning's arena may host it at a planned offset.
     *
     * Prefer this over create_tensor() (eager; allocated for the graph's whole
     * lifetime unless the memory passes intervene) for any tensor that only
     * exists to carry a value between nodes.
     *
     * @code
     * auto &tmp = graph.scratch<double, 2>("tmp", nocc, nvir);
     * auto &acc = graph.scratch_zero<double, 2>("acc", nocc, nvir);
     * @endcode
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &scratch(std::string name, Dims... dims) {
        auto &t = declare_tensor<T, Rank>(std::move(name), dims...);
        for (auto &[tid, handle] : _tensors) {
            if (handle.tensor_ptr == &t) {
                handle.is_intermediate = true;
                break;
            }
        }
        return t;
    }

    /// Zero-initialized scratch (zeroed at materialization, like
    /// declare_zero_tensor, and managed like scratch()).
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &scratch_zero(std::string name, Dims... dims) {
        auto &t = declare_zero_tensor<T, Rank>(std::move(name), dims...);
        for (auto &[tid, handle] : _tensors) {
            if (handle.tensor_ptr == &t) {
                handle.is_intermediate = true;
                break;
            }
        }
        return t;
    }

    /// Runtime-rank scratch (see scratch(); pybind-facing analog).
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("scratch", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("scratch", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("scratch", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("scratch", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &scratch_runtime(std::string name, std::vector<size_t> dims) {
        return declare_runtime_tensor<T, Alloc>(std::move(name), std::move(dims), /*intermediate=*/true);
    }

    /// Runtime-rank zero-initialized scratch.
    template <typename T, typename Alloc = std::allocator<T>>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("scratch_zero", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("scratch_zero", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("scratch_zero", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("scratch_zero", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>)
                    GeneralRuntimeTensor<T, Alloc> &scratch_zero_runtime(std::string name, std::vector<size_t> dims) {
        return declare_zero_runtime_tensor<T, Alloc>(std::move(name), std::move(dims), /*intermediate=*/true);
    }

    /**
     * @brief Declare a tensor with a user-provided fill function.
     *
     * The fill lambda is called after materialization (and after distribution
     * metadata is set). The tensor supports `T.range(dim)` for the local
     * global-index range and `T.global(indices...)` for global-indexed access.
     *
     * Works identically for distributed and non-distributed tensors:
     * @code
     * auto &eri = graph.declare_tensor_filled<double, 4>("ERI", nao, nao, nao, nao,
     *     [&](auto& T) {
     *         auto [p0, p1] = T.range(0);
     *         auto [q0, q1] = T.range(1);
     *         for (size_t p = p0; p < p1; p++)
     *             for (size_t q = q0; q < q1; q++)
     *                 // ... compute and fill using T.global(p, q, r, s)
     *     });
     * @endcode
     *
     * @param name  Tensor name.
     * @param dims  Global dimensions.
     * @param fill  Lambda called with a reference to the materialized tensor.
     */
    template <typename T, size_t Rank, typename FillFn>
    Tensor<T, Rank> &declare_tensor_filled(std::string name, Dim<Rank> dims, FillFn &&fill) {
        auto &t = [&]<size_t... Is>(std::index_sequence<Is...>) -> Tensor<T, Rank> & {
            return declare_tensor<T, Rank>(std::move(name), dims[Is]...);
        }(std::make_index_sequence<Rank>{});
        auto *ptr     = &t;
        auto  fill_fn = std::forward<FillFn>(fill);
        for (auto &[tid, handle] : _tensors) {
            if (handle.tensor_ptr == ptr) {
                handle.init_kind = InitKind::Zero; // Triggers an init node
                handle.zero_fn   = [ptr, fill_fn]() {
                    ptr->materialize();
                    fill_fn(*ptr);
                };
                break;
            }
        }
        return t;
    }

    /**
     * @brief Create a zero-initialized tensor owned by the graph, using runtime type info.
     *
     * Unlike the templated create_tensor/create_zero_tensor, this accepts ScalarType
     * and a vector of dimensions at runtime. Used by optimization passes that need
     * to create intermediate tensors without compile-time type information.
     *
     * @param[in] name Human-readable name.
     * @param[in] dtype Element type (Float32, Float64, Complex64, Complex128).
     * @param[in] dims Size of each dimension.
     * @return TensorId of the newly created tensor, and a void* to the tensor object.
     *         Returns error if dtype is Unknown or dims is empty.
     */
    [[nodiscard]] expected<std::pair<TensorId, void *>, GraphError> create_tensor_dynamic(std::string name, packed_gemm::ScalarType dtype,
                                                                                          std::vector<size_t> const &dims);

    /**
     * @brief Create an executor lambda that performs axpy: dst += alpha * src.
     *
     * Uses runtime type dispatch based on the tensor handles' ScalarType and rank.
     * Used by optimization passes (e.g. DistributiveFactoring) to build
     * executor lambdas for dynamically created nodes.
     *
     * @param[in] alpha Scalar multiplier.
     * @param[in] src_id TensorId of the source tensor.
     * @param[in] dst_id TensorId of the destination tensor.
     * @return A callable that performs the axpy operation.
     */
    std::function<void()> make_axpy_executor(double alpha, TensorId src_id, TensorId dst_id);

    /**
     * @brief Create an executor lambda that copies src into dst: dst = src.
     *
     * @param[in] src_id TensorId of the source tensor.
     * @param[in] dst_id TensorId of the destination tensor.
     * @return A callable that performs the copy.
     */
    std::function<void()> make_copy_executor(TensorId src_id, TensorId dst_id);

    /**
     * @brief Create a zero-initialized **runtime** tensor (GeneralRuntimeTensor)
     *        of a dtype/shape known only at run time, returning its id + pointer.
     *
     * The runtime analog of @ref create_tensor_dynamic: where that produces a
     * statically-ranked Tensor<T,K> (consumed via static_cast in some passes),
     * this produces a GeneralRuntimeTensor<T>, matching the tensors that the
     * Python/capture surface uses, so passes that combine such operands can cast
     * uniformly to GeneralRuntimeTensor<T>. Supports any rank.
     *
     * @param[in] name  Human-readable name.
     * @param[in] dtype Element type (Float32, Float64, Complex64, Complex128).
     * @param[in] dims  Size of each dimension.
     * @return TensorId and void* of the new runtime tensor; error if dtype is Unknown.
     */
    [[nodiscard]] expected<std::pair<TensorId, void *>, GraphError>
    create_zero_runtime_tensor_dynamic(std::string name, packed_gemm::ScalarType dtype, std::vector<size_t> const &dims);

    /**
     * @brief Create an executor lambda that performs C = alpha * A * B + beta * C.
     *
     * Uses runtime type dispatch. Only supports rank-2 tensors (matrices).
     * Used by ContractionPlanning to build GEMM nodes for restructured chains.
     *
     * @param[in] a_id   TensorId of left operand (M x K).
     * @param[in] b_id   TensorId of right operand (K x N).
     * @param[in] c_id   TensorId of output tensor (M x N).
     * @param[in] alpha  Scalar prefactor for the product.
     * @param[in] beta   Scalar prefactor for the accumulator (0 = overwrite).
     * @return A callable that performs the GEMM.
     */
    std::function<void()> make_gemm_executor(TensorId a_id, TensorId b_id, TensorId c_id, double alpha = 1.0, double beta = 0.0);

    /**
     * @brief Create an executor for an arbitrary-rank einsum from a ParsedEinsumSpec.
     *
     * Unlike make_gemm_executor (rank-2 only), this handles any rank via
     * runtime dispatch through StringDispatch. Falls through to BLAS for
     * rank-2 GEMM and uses the generic loop for higher ranks.
     *
     * Used by ContractionPlanning to restructure higher-rank chains.
     */
    std::function<void()> make_einsum_executor(TensorId a_id, TensorId b_id, TensorId c_id, ParsedEinsumSpec const &spec,
                                               double alpha = 1.0, double beta = 0.0);

    /**
     * @brief Create an executor lambda that zeros a tensor.
     *
     * @param[in] tensor_id TensorId of the tensor to zero.
     * @return A callable that zeros the tensor.
     */
    std::function<void()> make_zero_executor(TensorId tensor_id);

    /**
     * @brief Validate that all registered tensors are still alive.
     *
     * Calls each tensor's validator function (set by make_handle() at registration time).
     * If any validator returns false, throws a descriptive error message suggesting
     * to use create_tensor() for intermediates.
     *
     * Called automatically at the start of execute().
     *
     * @return Error if any tensor appears to have been destroyed.
     */
    [[nodiscard]] expected<void, GraphError> validate_tensors() const;

    /**
     * @brief Find a TensorSlot by TensorId.
     *
     * Returns nullptr if no slot exists for this TensorId.
     * Used by optimization passes to redirect captured lambdas to new tensors.
     */
    TensorSlot *find_slot(TensorId id) {
        auto it = _slot_map.find(id);
        return it != _slot_map.end() ? it->second.get() : nullptr;
    }

    /**
     * @brief Redirect a tensor's executor slot to another tensor's buffer.
     *
     * Repoints the slot for @p from so that any executor lambda which captured
     * it resolves to @p to's tensor at the next execute(). This is the
     * execution-side companion to the TensorId metadata rewrite that
     * redirect-based passes (e.g. CSE) perform on Node::inputs.
     *
     * The distinction matters because executor lambdas resolve their operands
     * through the captured TensorSlot pointer, *not* through Node::inputs.
     * Rewriting Node::inputs alone keeps liveness analysis correct for
     * downstream passes (MemoryPlanning, FreeInsertion) but is invisible to a
     * lambda baked at capture time; without this slot redirect a consumer of
     * an eliminated duplicate would keep reading the duplicate's (now
     * never-written) buffer. See CSE.
     *
     * No-op if either slot is absent, a tensor never captured through a slot
     * has no baked lambda to fix. The caller guarantees @p from and @p to have
     * identical element type (CSE only merges nodes with equal op_data and
     * output shapes; PermuteFusion fixes up rank/dims itself), so no
     * validation is performed.
     *
     * The redirect is durable, not a one-time pointer copy: it is recorded in
     * @ref _slot_redirects and re-applied whenever the target slot is
     * repointed later (see rebind()). Without that, a rebind of the surviving
     * tensor after CSE would leave consumers of the merged-away duplicate
     * reading the survivor's old buffer.
     *
     * @param[in] from TensorId whose slot should be repointed.
     * @param[in] to   TensorId whose buffer @p from should resolve to, now and
     *                 after future rebinds of @p to.
     */
    void redirect_slot(TensorId from, TensorId to) {
        // Collapse chains so every recorded redirect points at a terminal id.
        for (auto it = _slot_redirects.find(to); it != _slot_redirects.end(); it = _slot_redirects.find(to)) {
            to = it->second;
        }
        if (from == to) {
            return;
        }
        TensorSlot const *to_slot   = find_slot(to);
        TensorSlot       *from_slot = find_slot(from);
        if (to_slot == nullptr || from_slot == nullptr) {
            return;
        }
        from_slot->ptr        = to_slot->ptr;
        _slot_redirects[from] = to;
        // Anything already redirected to `from` now follows the same terminal.
        for (auto &[f, t] : _slot_redirects) {
            if (t == from) {
                t = to;
                if (auto *fs = find_slot(f)) {
                    fs->ptr = to_slot->ptr;
                }
            }
        }
    }

    // ── Rebind support ──────────────────────────────────────────────────────

    /**
     * @brief Create a TensorSlot for a tensor.
     *
     * Slots are used internally by operation wrappers during capture.
     * Returns a pointer to a stable TensorSlot owned by the graph.
     *
     * @tparam TensorType The tensor type.
     * @param[in] tensor The tensor to create a slot for.
     * @param[in] tensor_id The TensorId of this tensor.
     * @return Pointer to the slot (stable for the lifetime of the graph).
     */
    template <GraphCapturableTensor TensorType>
    TensorSlot *get_or_create_slot(TensorType const &tensor, TensorId tensor_id) {
        auto it = _slot_map.find(tensor_id);
        if (it != _slot_map.end()) {
            return it->second.get();
        }
        auto slot          = std::make_unique<TensorSlot>();
        slot->ptr          = const_cast<void *>(static_cast<void const *>(&tensor));
        slot->tensor_id    = tensor_id;
        slot->name         = tensor.name();
        slot->rank         = detail::tensor_rank(tensor);
        slot->element_size = sizeof(typename std::remove_cvref_t<TensorType>::ValueType);
        slot->dims.resize(slot->rank);
        for (size_t d = 0; d < slot->rank; d++) {
            slot->dims[d] = tensor.dim(d);
        }
        auto *raw            = slot.get();
        _slot_map[tensor_id] = std::move(slot);
        return raw;
    }

    /**
     * @brief Rebind a tensor slot to point to a different tensor.
     *
     * The new tensor must have the same rank, element type, and dimensions.
     * After rebinding, subsequent execute() calls will use the new tensor.
     *
     * @tparam TensorType The tensor type (must match the original).
     * @param[in] id The TensorId to rebind.
     * @param[in] new_tensor The new tensor to bind to.
     * @throws std::invalid_argument If rank or dimensions don't match.
     * @throws std::out_of_range If no slot exists for this TensorId.
     */
    template <GraphCapturableTensor TensorType>
    void rebind(TensorId id, TensorType &new_tensor) {
        auto it = _slot_map.find(id);
        if (it == _slot_map.end()) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no slot for tensor id {}", _name, id);
        }
        auto *slot = it->second.get();

        // Validate rank, read from the type when it carries ::Rank,
        // otherwise from the live runtime-rank tensor.
        std::size_t const new_rank = detail::tensor_rank(new_tensor);
        if (new_rank != slot->rank) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': rebind tensor '{}': rank mismatch ({} vs {})", _name, slot->name,
                                    new_rank, slot->rank);
        }

        // Validate dimensions
        for (size_t d = 0; d < slot->rank; d++) {
            if (new_tensor.dim(d) != slot->dims[d]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': rebind tensor '{}': dim {} mismatch ({} vs {})", _name,
                                        slot->name, d, new_tensor.dim(d), slot->dims[d]);
            }
        }

        slot->ptr  = const_cast<void *>(static_cast<void const *>(&new_tensor));
        slot->name = new_tensor.name();

        // Tensor names feed the cached profiler annotations.
        _profile_strings_valid = false;

        // An explicit rebind of a merged-away tensor overrides its pass
        // redirect; otherwise a later rebind of the survivor would stomp it.
        _slot_redirects.erase(id);

        // Slots that a pass redirected to this tensor (CSE duplicates,
        // fused permute outputs) must follow the new buffer.
        for (auto const &[f, t] : _slot_redirects) {
            if (t == id) {
                if (auto *fs = find_slot(f)) {
                    fs->ptr = slot->ptr;
                }
            }
        }

        // Update the TensorHandle too
        auto th_it = _tensors.find(id);
        if (th_it != _tensors.end()) {
            th_it->second.tensor_ptr = slot->ptr;
            th_it->second.name       = new_tensor.name();
            th_it->second.name_hash  = std::hash<std::string>{}(new_tensor.name());
            th_it->second.validator  = [&new_tensor, hash = th_it->second.name_hash]() -> bool {
                try {
                    return std::hash<std::string>{}(new_tensor.name()) == hash;
                } catch (...) {
                    return false;
                }
            };
        }
    }

    /**
     * @brief Rebind a tensor by matching the old tensor's pointer.
     *
     * Finds the slot currently pointing to ``old_tensor`` and redirects it
     * to ``new_tensor``. The new tensor must have the same rank and dimensions.
     *
     * @tparam TensorType The tensor type (must match for both old and new).
     * @param[in] old_tensor The tensor currently bound in the graph.
     * @param[in] new_tensor The new tensor to bind.
     * @throws std::out_of_range If no slot points to old_tensor.
     * @throws std::invalid_argument If rank or dimensions don't match.
     *
     * @code
     * graph.rebind(A1, A2);  // Swap A1 for A2, one line
     * @endcode
     */
    template <GraphCapturableTensor TensorType>
    void rebind(TensorType const &old_tensor, TensorType &new_tensor) {
        void *old_ptr = const_cast<void *>(static_cast<void const *>(&old_tensor));

        // Find the slot and TensorId for old_tensor. After a pass redirect
        // (CSE, PermuteFusion) several slots share one pointer; prefer the
        // surviving tensor's id so redirected followers propagate, rather
        // than whichever duplicate the map yields first.
        bool     have_fallback = false;
        TensorId fallback{};
        for (auto &[id, slot] : _slot_map) {
            if (slot->ptr == old_ptr) {
                if (!_slot_redirects.contains(id)) {
                    rebind(id, new_tensor);
                    return;
                }
                if (!have_fallback) {
                    fallback      = id;
                    have_fallback = true;
                }
            }
        }
        if (have_fallback) {
            rebind(fallback, new_tensor);
            return;
        }

        // Also check tensors_ in case no slot exists yet (tensor registered but never captured via slot)
        for (auto &[id, handle] : _tensors) {
            if (handle.tensor_ptr == old_ptr) {
                // Create a slot for this tensor so rebind(TensorId, ...) works
                get_or_create_slot(old_tensor, id);
                rebind(id, new_tensor);
                return;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor matching '{}' found for rebind", _name, old_tensor.name());
    }

    /**
     * @brief Create mutable einsum parameters owned by the graph.
     *
     * Returns a shared_ptr to EinsumParams that can be captured by executor
     * lambdas. Updating the params changes the computation on next execute().
     *
     * @param[in] c_pf Initial C prefactor.
     * @param[in] ab_pf Initial AB prefactor.
     * @return Shared pointer to the params (stable for graph lifetime).
     */
    template <typename T>
    std::shared_ptr<EinsumParams> create_params(T c_pf, T ab_pf) {
        auto params   = std::make_shared<EinsumParams>();
        params->c_pf  = c_pf;
        params->ab_pf = ab_pf;
        _params_store.push_back(params);
        return params;
    }

    /**
     * @brief Create mutable index state for an einsum operation.
     *
     * Returns a shared_ptr to EinsumIndices that executor lambdas
     * capture by shared ownership. Optimization passes (PermuteFusion,
     * and future rewriters) mutate the indices in place and the
     * updated contraction takes effect on the next execute().
     *
     * @param[in] a Input-A index list.
     * @param[in] b Input-B index list.
     * @param[in] c Output (C) index list.
     * @param[in] link Precomputed link (contracted) indices.
     * @return Shared pointer to the indices (stable for graph lifetime).
     */
    std::shared_ptr<EinsumIndices> create_indices(std::vector<std::string> a, std::vector<std::string> b, std::vector<std::string> c,
                                                  std::vector<std::string> link) {
        auto idx          = std::make_shared<EinsumIndices>();
        idx->a_indices    = std::move(a);
        idx->b_indices    = std::move(b);
        idx->c_indices    = std::move(c);
        idx->link_indices = std::move(link);
        _indices_store.push_back(idx);
        return idx;
    }

    /**
     * @brief Update scalar prefactors for an einsum node.
     *
     * Finds the node by NodeId and updates its EinsumParams if it was
     * captured with mutable parameters. Also updates the EinsumDescriptor.
     *
     * @param[in] node_id The NodeId of the einsum node.
     * @param[in] c_pf New C prefactor.
     * @param[in] ab_pf New AB prefactor.
     */
    void update_prefactors(NodeId node_id, PrefactorScalar c_pf, PrefactorScalar ab_pf);

  private:
    std::string                                _name;
    std::string                                _pipeline_name;   ///< Parent pipeline name (empty if standalone)
    std::string                                _workspace_name;  ///< Parent workspace name (empty if none)
    std::string                                _stage_name;      ///< Stage name within pipeline
    std::string                                _stage_type;      ///< "graph" or "loop"
    int                                        _stage_index{-1}; ///< Order within pipeline
    std::vector<Node>                          _nodes;
    std::unordered_map<TensorId, TensorHandle> _tensors;
    NodeId                                     _next_node_id{0};
    // Starts at 1: id 0 is reserved as the "no tensor" / "no alias" sentinel
    // (TensorHandle::aliases defaults to 0 and the codebase tests `aliases == 0`
    // for "not a view"). If a real tensor could be id 0, a view of it would have
    // aliases == 0 and silently fail to resolve to its parent in the scheduler.
    TensorId _next_tensor_id{1};
    bool     _sorted{false};

    /// Whether _deps matches the current node order. Distinct from _sorted:
    /// mark_sorted() vouches for the order without rebuilding _deps, so
    /// topological_sort() can skip the Kahn pass but must refresh the lists.
    bool _deps_valid{false};

    /// Pre-formatted profiler payloads for one node: the zone name plus every
    /// annotation whose value is invariant across replays. Built once per
    /// graph mutation instead of fmt::format-ing per node per execute() -
    /// for an SCF/CC loop that replays the graph hundreds of times, the
    /// formatting dominated the serial replay overhead.
    struct NodeProfileStrings {
        std::string                                      zone;    ///< "graph:<name>/<label>"
        std::vector<std::pair<std::string, std::string>> texts;   ///< invariant string annotations
        std::vector<std::pair<std::string, int64_t>>     numbers; ///< invariant integer annotations
        std::vector<std::pair<std::string, double>>      reals;   ///< invariant floating-point annotations
    };

    /// Keyed by NodeId (stable across reorders). Invalidated wherever the
    /// node list or annotated metadata changes: add_node, mark_sorted (the
    /// declared-mutation contract - passes rewrite labels/descriptors),
    /// rebind (tensor names), and update_prefactors.
    std::unordered_map<NodeId, NodeProfileStrings> _profile_strings;
    bool                                           _profile_strings_valid{false};
    std::string                                    _exec_zone_name;

    /// Rebuild _profile_strings for the current node list.
    void rebuild_profile_strings();

    /// See EffectiveIoCache. Span-returning fast path used by the hazard
    /// scans: ordinary nodes view their own I/O lists (no copies),
    /// control-flow nodes memoize the subtree walk in @p cache.
    std::pair<std::span<TensorId const>, std::span<TensorId const>> effective_io_cached(Node const &node, EffectiveIoCache &cache);

    /// Rebuild the position-keyed _deps lists for the current node order.
    void           rebuild_deps(EffectiveIoCache &cache);
    bool           _executed{false}; ///< True after first successful execute (caching)
    DependencyInfo _deps;            ///< Populated by topological_sort()

    /// Type-erased storage for graph-owned tensors (from create_tensor()).
    /// Each entry uses a typed deleter captured at creation time.
    std::vector<std::unique_ptr<void, void (*)(void *)>> _owned_tensors;

    /// Captured cleanup callbacks from ``adopt()``, invoked in
    /// reverse-insertion order at graph destruction. Used by capture-time
    /// helpers (``cg::view``) that allocate auxiliary state on the heap.
    std::vector<std::function<void()>> _adopted_cleanups;

    /// Runtime parameter table. ``View`` executors and the @c WriteParam
    /// node read/write through this. Pipeline plumbs its own table down
    /// at stage construction; standalone graphs get a default empty table.
    std::shared_ptr<ParamTable> _params{std::make_shared<ParamTable>()};

    /// Tensor slots for rebindable tensor references (TensorId → TensorSlot).
    std::unordered_map<TensorId, std::unique_ptr<TensorSlot>> _slot_map;

    /// Summary of the last optimize() run (see explain()).
    std::string _last_optimize_report;

    /// Durable slot redirects recorded by redirect_slot(): key resolves to
    /// value's buffer. Chains are collapsed at insert, so values are always
    /// terminal ids. rebind() re-applies these so redirected slots follow.
    std::unordered_map<TensorId, TensorId> _slot_redirects;

    /// Device shadow allocations for GPU execution.
    /// Persists across execute() calls so shadows can be reused.
    DeviceShadowMap _device_shadows;

    /// Per-node timing from last execute() call.
    std::vector<NodeTiming> _timing_report;

    /// Mutable einsum parameters (kept alive by shared_ptr in lambdas + this list).
    std::vector<std::shared_ptr<EinsumParams>>  _params_store;
    std::vector<std::shared_ptr<EinsumIndices>> _indices_store;
};

// ── Global graph registry for profiler integration ─────────────────────────

/**
 * @brief Register a graph for profiler visibility.
 *
 * Called automatically by Graph::execute() on the first execution of each
 * graph. The profiler viewer can then request and display the graph's
 * structure via the "get_compute_graphs" server handler.
 *
 * Graphs are stored by name; re-registering with the same name replaces
 * the old entry. Normally not needed by user code, graphs auto-register.
 *
 * @param[in] graph Pointer to the graph. Must remain valid until unregistered.
 */
void register_graph(Graph *graph);

/**
 * @brief Unregister a graph from the profiler.
 *
 * Called automatically by ~Graph() and by move operations. Normally not
 * needed by user code.
 *
 * @param[in] graph Pointer previously passed to register_graph().
 */
void unregister_graph(Graph *graph);

/**
 * @brief Get JSON describing all registered compute graphs.
 *
 * Returns a JSON object with key "graphs" containing an array of graph JSON objects.
 * Used by the profiler server's "get_compute_graphs" handler.
 */
std::string registered_graphs_json();

} // namespace einsums::compute_graph
