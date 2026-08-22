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
#include <Einsums/ComputeGraph/GateFlags.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/ComputeGraph/UsageAnalysis.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>

#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

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
     * @brief Link a freshly registered handle to any registered tensor whose
     *        storage contains it (or that it contains).
     *
     * ``cg::view()`` sets ``TensorHandle::aliases`` itself, but a view sliced
     * OUTSIDE a capture never goes through it: Python's capture-aware
     * ``__getitem__`` falls through to the eager slice, and the resulting view
     * reaches the graph as an ordinary operand on first use. Registered as-is
     * it has ``aliases == 0``, so the scheduler sees no relationship to its
     * parent and is free to order a read of the parent before a write through
     * the view. That produced silently wrong results with no error, and the
     * pattern is not exotic: a captured view must outlive the graph, which
     * pushes callers to build views up front.
     *
     * So containment is detected here from the registration-time data pointer
     * and strides, which closes the hole whatever the view's provenance.
     * Runs once over all registered tensors rather than per registration:
     * a containment test is O(n) against what is already there, so doing it on
     * every registration is quadratic, and a DLPNO-MP2 capture registers ~13k
     * tensors. Idempotent and cheap to call; register_tensor just marks it
     * stale.
     */
    void link_alias_storage();

    /**
     * @brief Reuse or mint a TensorId for @p handle's buffer in this graph.
     *
     * Scans this graph's tensors for one whose ``tensor_ptr`` matches @p
     * handle's and returns its id; otherwise registers @p handle. Shares the
     * orphan-parent-handle convention with effective_io: a buffer used only
     * inside sub-graphs gets one stable parent id here, so every parent node
     * touching it (hoisted lifecycle nodes and the control-flow node's
     * effective I/O) resolves to the same id and a dependency edge forms.
     *
     * @param[in] handle Handle whose tensor_ptr keys the lookup.
     * @return The existing or newly assigned TensorId.
     */
    TensorId find_or_register_tensor_ptr(TensorHandle const &handle);

    /// The id registered for @p ptr, or 0 if there is none.
    [[nodiscard]] TensorId find_tensor_id_by_ptr(void const *ptr) const noexcept {
        auto const it = _ptr_index.find(ptr);
        return it == _ptr_index.end() ? TensorId{0} : it->second;
    }

    /**
     * @brief Look up a tensor handle by its TensorId.
     * @param[in] id The tensor identifier.
     * @return Reference to the TensorHandle.
     * @throws std::out_of_range If no tensor with the given ID exists.
     */
    [[nodiscard]] TensorHandle       &tensor(TensorId id);
    [[nodiscard]] TensorHandle const &tensor(TensorId id) const; ///< @overload

    /**
     * @brief Look up a tensor handle by its TensorId, without throwing.
     * @param[in] id The tensor identifier.
     * @return Pointer to the TensorHandle, or nullptr if no tensor with the
     *         given ID exists. The non-throwing counterpart of tensor() for
     *         the optimizer passes' "handle or skip" lookups.
     */
    [[nodiscard]] TensorHandle       *find_tensor(TensorId id) noexcept;
    [[nodiscard]] TensorHandle const *find_tensor(TensorId id) const noexcept; ///< @overload

    /**
     * @brief The tensor object a node may legally dereference at EXECUTE time.
     *
     * Use this, never ``tensor(id).tensor_ptr``, in any lambda a pass bakes
     * into ``Node::execute``.
     *
     * ``TensorHandle::tensor_ptr`` names the CALLER's wrapper. That is the
     * handle's identity and it is deliberately not its lifetime: capture adopts
     * the operand's storage into a stand-in the graph keeps alive
     * (``TensorHandle::owner``), precisely so the caller's wrapper may be
     * destroyed before ``execute()``. Every built-in node reaches its operands
     * through the stand-in via ``TensorSlot::ptr``; a pass that rewrites a node
     * into an ``OpKind::Custom`` closure over ``tensor_ptr`` opts back out of
     * that guarantee and reads freed memory, silently, with a zeroed
     * intermediate as the usual symptom.
     *
     * @param[in] id The tensor identifier.
     * @return The stand-in when capture adopted one, otherwise the registered
     *         pointer. Null if no tensor with that id exists.
     */
    [[nodiscard]] void *live_tensor_ptr(TensorId id) const noexcept {
        auto const *handle = find_tensor(id);
        if (handle == nullptr) {
            return nullptr;
        }
        return handle->owner ? handle->owner.get() : handle->tensor_ptr;
    }

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

    /**
     * @brief Choose the executor plain execute() uses for THIS graph.
     *
     * The stored executor applies wherever this graph is executed without an
     * explicit executor argument - most importantly loop bodies, which are
     * replayed by the loop node via the argument-less execute() and had no
     * other way to run on a parallel backend:
     *
     * @code{.py}
     * body = g.add_loop("iter", 100, cont)
     * body.set_executor(cg.DataflowExecutor())   # replays overlap independent nodes
     * @endcode
     *
     * Pass nullptr/None to restore the built-in sequential path. The custom
     * executors dispatch every node's CPU lambda directly, so graphs with
     * GPU-placed nodes should keep the default (it performs the device
     * shadow bookkeeping; the executors do not).
     *
     * @param[in] executor Shared ownership of the executor, or nullptr to reset.
     */
    APIARY_EXPOSE void set_executor(std::shared_ptr<Executor> executor) { _executor = std::move(executor); }

    /// The executor plain execute() will use, or nullptr for the built-in sequential path.
    [[nodiscard]] std::shared_ptr<Executor> const &executor() const { return _executor; }

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
    /// Constructor arguments are forwarded, mirroring PassManager::add, so a
    /// cost-model pass can be priced against an explicit profile.
    template <typename PassType, typename... Args>
    std::pair<bool, PassType> apply(Args &&...args) {
        std::scoped_lock const lock(*_content_mutex);
        PassType               pass{std::forward<Args>(args)...};
        bool                   modified = pass.run(*this);
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
        unsigned    width{0};         ///< Width it ran at; 0 if unplaced. @see NodeTimingSample::width
    };

    /**
     * @brief One raw timing sample, as a replay records it.
     *
     * Deliberately label-free: a replay writes one of these per node, and the
     * label is recoverable from the node id, so copying a label string per
     * node per replay was pure waste in an SCF/CC loop that replays the same
     * graph hundreds of times. @ref timing_report() attaches the labels once,
     * only if anyone asks for the report.
     */
    struct NodeTimingSample {
        NodeId id;
        OpKind kind;
        double duration_ms{0.0}; ///< Wall-clock time in milliseconds

        /// Thread width the node ACTUALLY ran at, or 0 when the executor did
        /// not place it at a chosen width.
        ///
        /// A duration is only meaningful beside the width that produced it.
        /// ThreadPlanning consumes a measurement as the node's SERIAL time, so
        /// without this a node the previous plan widened reports t(w) as t(1),
        /// looks cheaper than it is by its own speedup, drops under the fork
        /// floor and gets narrowed - the planner punishing exactly the nodes
        /// its last plan widened, and the harder the wider they ran.
        ///
        /// Recorded rather than read back off the node because the node's
        /// thread_width says what was PLANNED, not what happened: a stale plan
        /// reaches the dataflow executor with widths_active false and every
        /// node runs unwrapped with its thread_width still set. Inferring from
        /// the node would then correct a measurement that needed no correcting.
        unsigned width{0};
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

    /// Access the timing data from the last execute() call. Labels are
    /// resolved here (cached until the next recorded sample), not on the
    /// replay path.
    [[nodiscard]] std::vector<NodeTiming> const &timing_report() const;

    /**
     * @brief Record timing for a single node (used by custom Executors).
     *
     * Custom executors should call this after executing each node so that
     * print_timing_report() works correctly. Executors that time a whole run
     * before merging should prefer @ref record_node_timings(), which takes the
     * content mutex once for the batch instead of once per node.
     */
    void record_node_timing(NodeId id, OpKind kind, double duration_ms, unsigned width = 0) {
        std::scoped_lock const lock(*_content_mutex);
        _timing_samples.push_back({.id = id, .kind = kind, .duration_ms = duration_ms, .width = width});
        _timing_report_valid = false;
    }

    /// Label-carrying form kept for executors written against the older
    /// signature. The label is ignored: @ref timing_report() resolves it from
    /// the node list.
    void record_node_timing(NodeId id, std::string const & /*label*/, OpKind kind, double duration_ms) {
        record_node_timing(id, kind, duration_ms);
    }

    /// Append a whole run's samples under a single lock acquisition.
    void record_node_timings(std::vector<NodeTimingSample> &&samples) {
        std::scoped_lock const lock(*_content_mutex);
        if (_timing_samples.empty()) {
            _timing_samples = std::move(samples);
        } else {
            _timing_samples.insert(_timing_samples.end(), samples.begin(), samples.end());
        }
        _timing_report_valid = false;
    }

    /// Clear timing data (called at the start of execute()).
    void clear_timing_report() {
        _timing_samples.clear();
        _timing_report.clear();
        _timing_report_valid = true;
    }

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
     * @brief Throw if any execution level holds two nodes that touch overlapping storage.
     *
     * The invariant a level-scheduling executor runs on: everything in one
     * level is launched together, so nothing in one level may conflict. A
     * violation is not a slow schedule, it is a data race that changes the
     * answer from run to run while every serial replay stays correct.
     *
     * **Deliberately does not share the hazard scan's alias resolution.** This
     * exists because a scan whose owner lookup silently truncated put two
     * accesses to one buffer under different owners, found no conflict, and
     * let the executor run them together; a checker that resolved owners the
     * same way would have agreed with it. So storage is grouped here by
     * recomputed byte-span overlap, and element-level disjointness is derived
     * fresh from each handle's pointer and strides rather than read off the
     * `aliases` link.
     *
     * Conservative where it cannot prove disjointness, so it can report a
     * conflict that is not one; it never misses a real one. That is the right
     * direction for a debug check, and the reason it is not on in release
     * builds.
     *
     * One exception to "everything an output names is written": a `View` node
     * writes its slice handle's dims, strides and data pointer, not the
     * parent's elements, so it is modeled as a READER of the region it
     * describes plus a writer of that handle. Two views of one buffer are
     * therefore independent however their slices overlap, while a node that
     * uses the slice, or a second node binding the same handle, still
     * conflicts with the bind.
     *
     * Scope: it checks LEVELS, so it covers the level schedulers exactly.
     * Two nodes the dataflow executor may still overlap can sit in different
     * levels, so a clean run here is not a proof for that executor.
     *
     * @throws std::runtime_error naming both nodes and the storage they share.
     *
     * @versionadded{2.0.0}
     */
    void verify_level_independence() const;

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

    /**
     * @brief Drop nodes flagged for removal, preserving the survivors' order.
     *
     * @p remove is indexed by node position: position @c i is erased when
     * @c i < remove.size() && remove[i]. Indices at or past @c remove.size()
     * are always kept, so a caller may size the mask to a prefix (e.g. only the
     * pre-existing nodes before it appended new ones) and keep the tail.
     *
     * Encapsulates the "rebuild a filtered node vector" idiom shared by the
     * mutating passes. It does NOT resort or touch the sortedness flags: erasing
     * while preserving order keeps an existing valid topological order valid, so
     * a pass that only removed nodes should follow with mark_sorted(), while a
     * pass that also appended nodes should follow with topological_sort().
     *
     * @return The number of nodes removed.
     */
    size_t erase_nodes(std::vector<bool> const &remove);

    /**
     * @brief Splice groups of nodes into the list. Each entry inserts its nodes
     *        immediately BEFORE index @c first in the current numbering.
     *
     * The counterpart to erase_nodes for the passes that build new nodes and
     * insert them at chosen positions. Applied in descending-position order
     * internally so earlier indices stay valid while later ones are spliced, so
     * callers pass positions in the ORIGINAL numbering and need not track shifts.
     * An "insert after node k" is expressed as position ``k+1``. Empty groups are
     * skipped. Marks the graph sorted (the caller vouches for the chosen order).
     */
    void insert_node_groups(std::vector<std::pair<std::size_t, std::vector<Node>>> groups);

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

    /// See EffectiveIoCache. Span-returning fast path used by the hazard
    /// scans and UsageAnalysis: ordinary nodes view their own I/O lists (no
    /// copies), control-flow nodes memoize the subtree walk in @p cache.
    std::pair<std::span<TensorId const>, std::span<TensorId const>> effective_io_cached(Node const &node, EffectiveIoCache &cache);

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
    /// The owning tensor of @p id, following alias links to their root.
    ///
    /// **The bound is a cycle detector, not a depth budget, and overrunning it
    /// throws rather than returning.** That distinction is the whole of a bug
    /// this cost a day of: the previous version gave up after a fixed 32 hops
    /// and returned whatever mid-chain id it had reached, which is not a
    /// conservative failure but a silent wrong answer. Two accesses to one
    /// buffer resolve to different owners, the hazard scan finds no conflict
    /// between them, and a threading executor runs them concurrently.
    /// DLPNO-(T0) hit it with one scratch buffer shared by 40 triplets: the
    /// schedule's widest level came out at exactly (triplets - 32), and the
    /// energy came out somewhere different every run.
    ///
    /// Every writer of ``aliases`` imposes a strict order - a view names its
    /// parent, and ``link_alias_storage`` links only to a strictly larger (or
    /// equal-size, lower-id) container - so a cycle is not constructible today
    /// and this throw is unreachable. It is here because the failure it
    /// replaces was invisible, and a chain longer than the tensor count means
    /// one of those writers has stopped being ordered.
    ///
    /// Chains stay short in practice: ``link_alias_storage`` path-compresses
    /// to the root, so only genuinely nested views (a view of a view of a
    /// view) walk more than one hop.
    [[nodiscard]] TensorId resolve_alias(TensorId id) const {
        for (size_t hops = 0; hops <= _tensors.size(); ++hops) {
            auto it = _tensors.find(id);
            if (it == _tensors.end() || it->second.aliases == 0) {
                return id;
            }
            id = it->second.aliases;
        }
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Graph '{}': alias chain from tensor {} exceeds the tensor count ({}), which means a "
                                "cycle in the alias links; the hazard scan cannot order accesses to it",
                                _name, id, _tensors.size());
    }

    /// Access dependency info (populated by topological_sort()).
    [[nodiscard]] DependencyInfo const &dependencies() const { return _deps; }

    /**
     * @brief How many hazard edges the schedule holds, and how wide its levels are.
     *
     * The two structural numbers a change to the dependence machinery moves:
     * an access that cannot be described precisely widens to its whole buffer
     * and picks up edges against every other access to it, and those edges are
     * what cost a level scheduler its width. Both are deterministic, so they
     * are what a test about ordering should assert - and what a measurement of
     * ordering should report - rather than a wall clock or an energy.
     *
     * Both build the schedule if it is stale, exactly as ``execute()`` would,
     * and so may reorder nodes into a topological order.
     *
     * @return @ref schedule_edge_count the number of dependency edges;
     *         @ref schedule_level_sizes the size of each level in order.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE [[nodiscard]] size_t schedule_edge_count();

    /// @copydoc schedule_edge_count
    APIARY_EXPOSE [[nodiscard]] std::vector<size_t> schedule_level_sizes();

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
        // ... and slot pointers (arena slices, CSE redirects).
        _slots_validated = false;
        // Position-keyed analyses (UsageAnalysis) are stale too.
        _analysis_version++;
    }

    /**
     * @brief Cached reader/writer/liveness index for the current node order.
     *
     * Rebuilt lazily when a mutation-declaration point (add_node,
     * mark_sorted, topological_sort, rebind) has been hit since the last
     * build, with a node-count defense for undeclared mutations (same
     * contract as DependencyInfo). Do NOT hold the reference across graph
     * mutations.
     */
    [[nodiscard]] UsageAnalysis const &usage();

    /// Mutation counter, bumped at every mutation-declaration point (add_node,
    /// mark_sorted, topological_sort, rebind). Anything caching a derived
    /// property of the node list - UsageAnalysis, an executor's per-node
    /// scratch - can stamp itself with this and rebuild when it goes stale.
    [[nodiscard]] std::uint64_t analysis_version() const { return _analysis_version; }

    /**
     * @brief Threads the per-node widths on this graph were planned against.
     *
     * A width is chosen for one machine at one moment: the planner divides a
     * known number of threads between the nodes, so the same widths on a
     * different thread count are not a worse plan, they are a wrong one.
     * Recording the count the plan assumed lets @ref DataflowExecutor notice
     * and fall back to running every node at width 1.
     *
     * 0 means nothing recorded, which is what a hand-set width or an unplanned
     * graph looks like; those are honored as they are found. Deliberately not
     * serialized with the graph, for the same reason the widths are not.
     */
    APIARY_EXPOSE APIARY_GETTER("planned_thread_count") [[nodiscard]] unsigned planned_thread_count() const {
        return _planned_thread_count;
    }

    /// Record the thread count the current widths were planned for.
    /// @see planned_thread_count
    void set_planned_thread_count(unsigned threads) { _planned_thread_count = static_cast<std::uint16_t>(threads); }

    /**
     * @brief Choose a thread width for every node, for this machine, now.
     *
     * Runs the @ref passes::ThreadPlanning heuristic standalone - no pipeline
     * needed, because widths touch no structure - and records the thread count
     * the plan assumed so @ref DataflowExecutor can notice a machine change and
     * fall back to width 1. Loop bodies and conditional branches are planned
     * too, each against the same thread count; the one process-wide width
     * budget composes them at run time.
     *
     * Only @ref DataflowExecutor acts on the result. Under the other executors
     * the graph behaves exactly as an unplanned one.
     *
     * @par Re-plan policy
     * A plan needs per-node serial times, and the honest ones are measured -
     * but a measurement is taken under whatever plan and whatever concurrency
     * produced it, so a plan computed FROM those timings and the plan they
     * were measured UNDER are never comparable on paper: the re-plan's own
     * estimate is polluted by the incumbent's schedule. (Concretely: co-run
     * contention inflates every sample, the inflated area bound swallows the
     * critical path, and a timings re-plan can then only ever narrow - it
     * un-widened measurably profitable plans in every DLPNO trial.) The one
     * comparator that settles it is the wall clock, so the re-plan is a
     * measured trial rather than a decree:
     * - A graph that has already replayed is planned from its recorded
     *   timings, and that plan is final.
     * - A graph that has never replayed is planned from the cost model.
     *   Replay 1 warms up and records timings, and a re-plan is computed from
     *   them; if it chooses the same widths the trial ends there (keeping the
     *   re-plan's measured admission priorities). Otherwise replay 2 runs the
     *   re-planned widths, replay 3 runs the cold widths again - both
     *   steady-state - and the graph keeps whichever replay was faster, the
     *   incumbent unless the candidate clearly beat it. A trial the workload
     *   never finishes (fewer than three replays) leaves the cold plan in
     *   place.
     *
     * The consequence is deliberate and is the determinism contract: widths
     * move only inside the trial window, so from the fourth replay onward -
     * or from the first, with @p freeze - repeated replays at a fixed thread
     * count are bit-identical.
     *
     * @param freeze Plan now and never re-plan, even from the cold-start model.
     *               For workflows that want every replay bit-identical to the
     *               first, at the cost of model-quality widths.
     * @return True when at least one node was given a width above 1.
     *
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE bool plan_threads(bool freeze = false);

    /// Whether the thread plan is still inside its measured trial window: a
    /// re-plan is waiting for the next completed replay, or the trial replays
    /// are still being timed. False once the widths are final. @see plan_threads
    APIARY_EXPOSE APIARY_GETTER("thread_replan_armed") [[nodiscard]] bool thread_replan_armed() const {
        return _plan_trial != ThreadPlanTrial::None;
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
     * @brief Add a conditional whose predicate is one flag of a @ref GateFlags array.
     *
     * The same node @ref add_conditional builds, with the predicate replaced by a load from a
     * shared buffer. Use it when the answers are known before the replay starts and there are many
     * of them: a @c std::function bound to a Python callable takes the GIL every time it is
     * evaluated, and a graph whose conditionals run on several threads serializes on that.
     *
     * The array is shared, not copied, so the caller may keep writing to it between replays; a
     * copy of the @c shared_ptr is baked into the node, so the flags outlive the handle. An index
     * past the end of the array reads false.
     *
     * @param[in] label Human-readable label for profiling.
     * @param[in] flags The array to read.
     * @param[in] index Which flag in it selects this node's branch.
     * @return Tuple of references: (then_branch_graph, else_branch_graph).
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) std::tuple<Graph &, Graph &> add_conditional_flag(std::string      label,
                                                                                                   GateFlags const &flags, size_t index);

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
        _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));

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
        _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));

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
        _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));

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

    /// Tiled analog of declare_zero_runtime_tensor(): a graph-owned
    /// TiledRuntimeTensor shell over @p tile_sizes with DEFERRED lifecycle.
    ///
    /// The shell starts with no populated tiles - ops create tiles on demand
    /// (infer-and-create, zeroed) - so Materialize and Initialize are cheap,
    /// but registering the handle as Deferred puts the tensor under the same
    /// lifecycle machinery as dense scratch: Materialization hoists the pair
    /// to the loop's parent, and FreeInsertion can release the tile storage
    /// after the last consumer (release() keeps the sparsity pattern, so a
    /// later replay re-materializes into the same structure). MemoryPlanning
    /// leaves tiled scratch alone by construction: the arena requires
    /// materialize_into, and a tile-wise tensor has no single buffer to
    /// place. Scratch REUSE per replay is safe for the ops with the
    /// leftover-scale rule (einsum's c_pf, direct_division's and permute's
    /// beta apply to every stored tile, so stale tiles from a previous
    /// iteration are zeroed, not kept).
    template <typename T>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tiled_tensor", T = float)
        APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tiled_tensor", T = double)
            APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tiled_tensor", T = std::complex<float>)
                APIARY_INSTANTIATE_MEMBER_AS("declare_zero_tiled_tensor", T = std::complex<double>)
                    TiledRuntimeTensor<T> &declare_zero_tiled_tensor(std::string name, std::vector<std::vector<int>> tile_sizes,
                                                                     bool intermediate = false) {
        using TensorType = TiledRuntimeTensor<T>;
        auto *ptr        = new TensorType(std::move(name), std::move(tile_sizes));
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });
        _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));

        auto handle            = make_handle(*ptr, 0);
        handle.is_intermediate = intermediate;
        handle.alloc_state     = AllocState::Deferred; // empty shell reads as vacuously materialized; force the lifecycle
        handle.init_kind       = InitKind::Zero;
        handle.zero_fn         = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        register_tensor(std::move(handle));
        // No Alloc node, MaterializationPass inserts Materialize + Initialize.
        return *ptr;
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
        _owned_tensor_ptrs.insert(static_cast<void const *>(ptr));

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
     * @brief Executor for ``dst = alpha*src + beta*dst`` reading LIVE scalars.
     *
     * Prefer this over @ref make_axpy_executor for any pass-built node that also
     * carries an AxpbyDescriptor: the descriptor must share this @p params
     * object, so a later pass that rewrites alpha/beta changes what replay
     * actually computes. @ref make_axpy_executor bakes its alpha into the
     * lambda, which makes a descriptor beside it a snapshot the executor can
     * silently disagree with.
     *
     * @param[in] params Scalars shared with the node's AxpbyDescriptor.
     * @param[in] src_id TensorId of the source tensor.
     * @param[in] dst_id TensorId of the destination tensor.
     * @return A callable that performs the axpby operation.
     */
    std::function<void()> make_axpby_executor(std::shared_ptr<AxpbyParams> params, TensorId src_id, TensorId dst_id);

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
     * @brief Build a FIRST-CLASS einsum node for a pass that synthesizes a contraction.
     *
     * Unlike @ref make_einsum_executor (which hands back a bare closure with the
     * dims, spec, and scalars all baked at pass time, paired with a descriptor-less
     * ``OpKind::Gemm`` node), this returns a complete ``OpKind::Einsum`` node that
     * behaves like a captured one:
     *
     * - a full @ref EinsumDescriptor, so every pass that reads the descriptor or
     *   the contraction spec (CSE, DeadNodeElimination, ScaleAbsorption,
     *   PermuteFusion, StreamContractionFusion, Reorder, the distribution and GPU
     *   passes) can see the node instead of stepping over an opaque blob;
     * - freshly allocated *shared* @ref EinsumParams and @ref EinsumIndices, with an
     *   executor that reads THROUGH those handles on every call. A pass that folds a
     *   scale into ``ab_prefactor`` or rewrites the index lists therefore takes
     *   effect on the next execute, rather than being silently ignored the way a
     *   baked closure would ignore it (the desync class of bug-1002);
     * - operands resolved by @ref TensorId at call time, so ``rebind()`` and
     *   Materialization are honored, and the RMW input convention applied for you
     *   (the output is declared as an input when @p c_pf is nonzero, the omission
     *   that was bug-1009).
     *
     * @par Limits
     * One dtype across the three operands. Rank and static tensor type are NOT
     * restricted: each operand is re-viewed through its rank-erased
     * ``TensorHandle::impl_fn``, which carries data, dims and strides as runtime
     * values, so a statically typed ``Tensor<T, Rank>`` works alongside a runtime
     * tensor and a single dtype dispatch covers every rank -- no static-rank cast,
     * so none of the type confusion of bug-1015. Tile-wise sparse tensors have no
     * single impl and are rejected; so is a dtype mismatch. Both throw, since a
     * pass reaching here without gating has a bug.
     *
     * A @ref GemmHint IS built when the shapes qualify (three rank-2 operands,
     * exactly one link index, strides agreeing with each declared layout), so
     * GEMMBatching can batch these nodes. Its extractors resolve by @ref TensorId
     * at call time, which follows ``rebind()`` and survives MemoryPlanning
     * repointing storage via ``materialize_into``.
     *
     * @param a_id  First operand.
     * @param b_id  Second operand. Must match @p spec.b_indices.
     * @param c_id  Output.
     * @param spec  Index lists for the contraction.
     * @param c_pf  Output prefactor. Nonzero means accumulate (read-modify-write).
     * @param ab_pf Product prefactor.
     * @param conj_a Conjugate the first operand.
     * @param conj_b Conjugate the second operand.
     * @param label Node label; a default is generated from the spec when empty.
     * @return A node ready to splice in, with a reserved id and inputs/outputs set.
     */
    Node make_einsum_node(TensorId a_id, TensorId b_id, TensorId c_id, ParsedEinsumSpec const &spec, PrefactorScalar c_pf,
                          PrefactorScalar ab_pf, bool conj_a = false, bool conj_b = false, std::string label = {});

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
        _slots_validated      = false;
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

    // ── Operand ownership ───────────────────────────────────────────────────

    /**
     * @brief The graph-owned stand-in for a captured operand.
     *
     * Returns a wrapper over @p tensor's storage that this graph keeps alive
     * for its own lifetime. Registration, slots and executor lambdas all bind
     * to the returned object rather than to the caller's, which is what lets a
     * captured operand's own wrapper go out of scope before ``execute()``:
     *
     * @code
     * cg::Graph g("x");
     * {
     *     cg::CaptureGuard guard(g);
     *     auto tmp = create_zero_tensor<double>("tmp", m, n);   // dies here
     *     cg::gemm(1.0, A, B, 0.0, &tmp);
     *     cg::gemm(1.0, tmp, D, 0.0, &E);
     * }
     * g.execute();                                              // still correct
     * @endcode
     *
     * The stand-in shares storage, so writes through it land in the caller's
     * buffer and the caller reads its results as before. One stand-in per
     * caller wrapper, so repeat captures of the same tensor stay one operand.
     *
     * Three kinds of operand are returned unchanged:
     *
     *  - tensors this graph already owns (``create_tensor`` / ``declare_tensor``),
     *    because their wrapper is graph-owned already;
     *  - **deferred** tensors, which have no storage yet. These are the ones
     *    the graph relocates: Materialization allocates them and MemoryPlanning
     *    re-seats them on an arena slice, both during execute, and a stand-in
     *    taken at capture time would still hold the pre-materialization
     *    pointer. They need no stand-in either, since whatever declared them
     *    (a Workspace, a Pipeline, this Graph) outlives the capture by
     *    construction;
     *  - types with no ``shallow_alias()`` (tile-wise sparse tensors have no
     *    single storage block to share), which keep the older "operands must
     *    outlive the graph" contract.
     *
     * What is left is exactly the operands the graph reads and writes but never
     * moves, which is what makes one stand-in safe for a whole replay.
     *
     * @tparam TensorType The operand's type.
     * @param[in] tensor The caller's tensor.
     * @return An owning reference to the stand-in, or empty when the graph
     *         adopted nothing and the caller's own tensor is used directly.
     *         The caller stores it in the operand's TensorHandle and TensorSlot;
     *         the graph keeps no separate index, so adoption costs one
     *         allocation and nothing per node afterwards.
     */
    template <GraphCapturableTensor TensorType>
    std::shared_ptr<void> adopt_operand(TensorType const &tensor) {
        using Clean = std::remove_cvref_t<TensorType>;

        if constexpr (!requires(Clean const &t) { t.shallow_alias(); }) {
            return {};
        } else {
            if (_owned_tensor_ptrs.contains(static_cast<void const *>(&tensor))) {
                return {};
            }
            if constexpr (requires(Clean const &t) { t.is_materialized(); }) {
                if (!tensor.is_materialized()) {
                    return {};
                }
            }
            // make_shared, not shared_ptr(new ...): one allocation for the
            // object and its control block rather than two. Capture allocates
            // one of these per distinct operand, so the difference shows up in
            // BenchmarkGraphOverhead.
            //
            // Built through the tagged constructor, never from the prvalue
            // shallow_alias() returns: these types have no move constructor, so
            // make_shared would bind that prvalue to the COPY constructor and
            // hand back a deep copy that shares nothing. See
            // einsums::detail::SharedStorageTag. Views own no storage, so their
            // plain copy already aliases.
            if constexpr (requires { Clean(::einsums::detail::SharedStorageTag{}, tensor); }) {
                return std::make_shared<Clean>(::einsums::detail::SharedStorageTag{}, tensor);
            } else {
                return std::make_shared<Clean>(tensor);
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
        // If capture adopted a stand-in for this operand, the handle owns it and
        // the slot must point at it and share that ownership: the slot outlives
        // the caller's wrapper, and pointing at a wrapper that may be destroyed
        // is exactly what operand adoption exists to avoid.
        if (auto const *handle = find_tensor(tensor_id); handle != nullptr && handle->owner) {
            slot->ptr   = handle->owner.get();
            slot->owner = handle->owner;
        }

        auto *raw            = slot.get();
        _slot_map[tensor_id] = std::move(slot);
        _slots_validated     = false;
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
        // A fresh pointer has not been through the slot check yet.
        _slots_validated = false;

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
        auto idx            = std::make_shared<EinsumIndices>();
        idx->spec.a_indices = std::move(a);
        idx->spec.b_indices = std::move(b);
        idx->spec.c_indices = std::move(c);
        idx->link_indices   = std::move(link);
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
    /// Move every member from @p other into `this`. Shared by the move
    /// constructor and move assignment so a newly added member cannot be
    /// forgotten in one of the two. Does not touch the global graph registry;
    /// callers handle unregister/register around it.
    void move_members_from(Graph &&other) noexcept;

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

    /// Registration-time byte spans, for the containment search in
    /// link_alias_storage. ``TensorHandle::data_ptr`` is a registration-time
    /// snapshot that nothing refreshes, so these stay valid for the life of the
    /// handle and the search need not recompute an extent per candidate.
    /// False once a tensor has been registered since the last link pass.
    bool _aliases_linked{true};

    /// ``TensorHandle::tensor_ptr`` to id. Capture asks "is this object already
    /// registered?" for every operand, which was a linear scan of the tensor
    /// table and so quadratic over a capture; a DLPNO-MP2 graph registers ~13k
    /// tensors. First registration wins, matching the scan it replaces.
    std::unordered_map<void const *, TensorId> _ptr_index;

    bool _sorted{false};

    /// Whether _deps matches the current node order. Distinct from _sorted:
    /// mark_sorted() vouches for the order without rebuilding _deps, so
    /// topological_sort() can skip the Kahn pass but must refresh the lists.
    bool _deps_valid{false};

    /// Pre-interned profiler payloads for one node: the zone name plus every
    /// annotation whose value is invariant across replays. Built once per
    /// graph mutation instead of fmt::format-ing per node per execute() -
    /// for an SCF/CC loop that replays the graph hundreds of times, the
    /// formatting dominated the serial replay overhead.
    ///
    /// The ids are string-table ids, not strings: interning a key and a value
    /// per annotation per node per replay took the table's lock several times
    /// per node, which is what made a profiled replay several times the cost
    /// of an unprofiled one. Ids are stable for the life of the process (the
    /// table only grows), so caching them here is safe across enable/disable.
    /// @c zone is kept because the Tracy backend wants the characters.
    struct NodeProfileStrings {
        NodeId                                     node_id{0}; ///< owner, checked against the node at replay
        std::string                                zone;       ///< "graph:<name>/<label>"
        uint32_t                                   zone_id{0}; ///< interned @c zone
        std::vector<std::pair<uint32_t, uint32_t>> texts;      ///< invariant string annotations (key id, value id)
        std::vector<std::pair<uint32_t, int64_t>>  numbers;    ///< invariant integer annotations
        std::vector<std::pair<uint32_t, double>>   reals;      ///< invariant floating-point annotations
    };

    /// Parallel to _nodes (position i describes node i), which is what lets
    /// the replay loop index straight in instead of hashing a NodeId per node.
    /// @c node_id is checked against the node anyway, so an UNdeclared
    /// mutation degrades to a bare label rather than mislabelling a zone.
    ///
    /// Invalidated wherever the node list or annotated metadata changes:
    /// add_node, mark_sorted (the declared-mutation contract - passes rewrite
    /// labels/descriptors), rebind (tensor names), and update_prefactors.
    /// Only built when something is recording; a run with the profiler off
    /// never formats or interns any of it.
    std::vector<NodeProfileStrings> _profile_strings;
    bool                            _profile_strings_valid{false};
    std::string                     _exec_zone_name;
    uint32_t                        _exec_zone_id{0};

    /// Threads the node widths were planned for; 0 = never recorded.
    /// @see planned_thread_count
    std::uint16_t _planned_thread_count{0};

    /// Where a cold thread plan is in its measured trial. @see plan_threads
    enum class ThreadPlanTrial : std::uint8_t {
        None,      ///< No decision pending: the widths on the nodes are final.
        Armed,     ///< Cold model plan is live; the re-plan fires after the next replay.
        Candidate, ///< The re-planned widths are live and this replay is timing them.
        Incumbent, ///< The cold widths are back and this replay is timing them.
    };

    /// Every planned width and admission priority in the graph tree, in walk
    /// order: this graph's nodes, then each container body's, recursively.
    using ThreadPlanSnapshot = std::vector<std::pair<std::uint16_t, std::int64_t>>;

    ThreadPlanTrial    _plan_trial{ThreadPlanTrial::None};
    ThreadPlanSnapshot _plan_incumbent;         ///< The cold model plan, held during the trial.
    ThreadPlanSnapshot _plan_candidate;         ///< The timings re-plan, held during the trial.
    double             _plan_candidate_ms{0.0}; ///< Wall clock of the candidate's replay.

    /// Run the width planner for @p threads threads. Shared by
    /// @ref plan_threads and the trial re-plan, neither of which may arm.
    bool run_thread_planner(unsigned threads);

    /// Advance the thread-plan trial, if one is open. Called at the end of a
    /// completed replay with that replay's wall-clock time. @see plan_threads
    void finish_replay_thread_plan(double replay_ms);

    /// Mutation counter for cached analyses. Bumped at every
    /// mutation-declaration point; UsageAnalysis caches against it.
    std::uint64_t _analysis_version{0};
    /// Version _usage was built at (UINT64_MAX = never built).
    std::uint64_t _usage_version{std::numeric_limits<std::uint64_t>::max()};
    UsageAnalysis _usage;

    /// Rebuild _profile_strings for the current node list.
    void rebuild_profile_strings();

    /// Rebuild the position-keyed _deps lists for the current node order.
    void rebuild_deps(EffectiveIoCache &cache);

    /// Partition the current _deps.successors/predecessors into levels.
    /// Split out of rebuild_deps so topological_sort can reach it without
    /// repeating the hazard scan when the sort leaves the node order alone.
    void rebuild_levels();

    /// Walk the node list once and invoke @p emit(producer_pos, consumer_pos)
    /// for every RAW/WAW/WAR hazard edge, keyed by owner TensorId (alias-resolved)
    /// and subtree-aware (effective I/O). The single source of truth for the
    /// data-dependency scan shared by topological_sort (Kahn adjacency) and
    /// rebuild_deps (successor/predecessor lists); defined in Graph.cpp because
    /// both instantiations live there.
    template <typename F>
    void           for_each_hazard_edge(EffectiveIoCache &cache, F &&emit);
    bool           _executed{false}; ///< True after first successful execute (caching)
    DependencyInfo _deps;            ///< Populated by topological_sort()

    /// Serializes structural reads/writes of _nodes / _tensors / _timing_report
    /// so the profiler server thread's to_json() cannot observe a torn or
    /// half-moved node vector while the owning thread mutates the graph. Locked
    /// by to_json (reader) and by the mutating entry points -- add_node,
    /// register_tensor, topological_sort, erase_nodes, insert_node_groups,
    /// record_node_timing, and the pass runners apply(). RECURSIVE because a
    /// locked pass runner calls the also-locked primitives. A unique_ptr because
    /// Graph must stay movable (a mutex is not) and each Graph keeps its OWN
    /// mutex across moves -- move_members_from never transfers it.
    mutable std::unique_ptr<std::recursive_mutex> _content_mutex = std::make_unique<std::recursive_mutex>();

    /// Type-erased storage for graph-owned tensors (from create_tensor()).
    /// Each entry uses a typed deleter captured at creation time.
    std::vector<std::unique_ptr<void, void (*)(void *)>> _owned_tensors;

    /// Addresses of the wrappers in @ref _owned_tensors, for the "do I already
    /// own this one?" test in @ref adopt_operand.
    std::unordered_set<void const *> _owned_tensor_ptrs;

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

    /// Whether every slot pointer has been checked since the last change to
    /// the slot table. The check catches cross-pipeline tensor misuse before
    /// it segfaults, but nothing can invalidate a pointer between two replays
    /// of an unchanged graph, so walking the whole map per replay only taxed
    /// iterative workloads. Cleared wherever a slot is created, rebound,
    /// redirected, or handed to a pass.
    bool _slots_validated{false};

    /// Summary of the last optimize() run (see explain()).
    std::string _last_optimize_report;

    /// Durable slot redirects recorded by redirect_slot(): key resolves to
    /// value's buffer. Chains are collapsed at insert, so values are always
    /// terminal ids. rebind() re-applies these so redirected slots follow.
    std::unordered_map<TensorId, TensorId> _slot_redirects;

    /// Device shadow allocations for GPU execution.
    /// Persists across execute() calls so shadows can be reused.
    DeviceShadowMap _device_shadows;

    /// Per-node timing from last execute() call, as recorded (no labels).
    std::vector<NodeTimingSample> _timing_samples;

    /// Labelled view of _timing_samples, materialized on demand by
    /// timing_report(). Mutable so the const accessor can fill it.
    mutable std::vector<NodeTiming> _timing_report;
    mutable bool                    _timing_report_valid{true};

    /// Executor plain execute() delegates to (see set_executor); nullptr means
    /// the built-in sequential path. Loop bodies replay through this.
    std::shared_ptr<Executor> _executor;

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
EINSUMS_EXPORT std::string registered_graphs_json();

EINSUMS_NAMESPACE_END(compute_graph)
