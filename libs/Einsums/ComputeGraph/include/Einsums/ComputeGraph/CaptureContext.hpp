//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/TensorSlot.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Thread-local capture context for recording operations into a graph.
 *
 * The CaptureContext manages the state machine for graph capture. When capturing
 * is active, the graph-aware operation wrappers (einsums::compute_graph::einsum,
 * etc.) call record() to add nodes to the graph instead of executing operations
 * directly.
 *
 * Each thread has its own CaptureContext (thread-local singleton), so capture
 * is thread-safe. Nested captures are not supported.
 *
 * Users typically don't interact with CaptureContext directly — use CaptureGuard
 * instead for RAII-based capture.
 *
 * @see CaptureGuard for the RAII wrapper
 * @see Graph for the container that receives captured nodes
 */
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph") EINSUMS_PYBIND_NOCOPY EINSUMS_PYBIND_NOMOVE CaptureContext {

  public:
    /**
     * @brief Access the thread-local CaptureContext singleton.
     * @return Reference to this thread's CaptureContext.
     */
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RVP(reference) static CaptureContext &current();

    /**
     * @brief Start capturing operations into the given graph.
     *
     * All subsequent calls to graph-aware operation wrappers on this thread
     * will record nodes into the graph instead of executing.
     *
     * @param[in,out] graph The graph to capture into.
     * @throws std::logic_error If already capturing (nested captures not supported).
     */
    EINSUMS_PYBIND_EXPOSE void begin_capture(Graph &graph);

    /**
     * @brief Stop capturing and finalize the graph.
     *
     * Triggers topological sorting of the captured nodes and clears the
     * internal pointer-to-TensorId mapping.
     *
     * @throws std::logic_error If not currently capturing.
     */
    EINSUMS_PYBIND_EXPOSE void end_capture();

    /**
     * @brief Check if this thread is currently in capture mode.
     * @return True if between begin_capture() and end_capture().
     */
    EINSUMS_PYBIND_EXPOSE [[nodiscard]] bool is_capturing() const { return _capturing; }

    /// Access the graph being captured into.
    [[nodiscard]] Graph *graph() const { return _graph; }

    /**
     * @brief Record an operation node during capture.
     *
     * Creates a Node with the given parameters and adds it to the current graph.
     * Called internally by the graph-aware operation wrappers.
     *
     * @tparam F Callable type for the executor (typically a lambda).
     * @param[in] kind The operation kind (for optimization pass pattern matching).
     * @param[in] label Human-readable label for profiling output.
     * @param[in] inputs TensorIds of tensors read by this operation.
     * @param[in] outputs TensorIds of tensors written by this operation.
     * @param[in] executor Type-erased callable that performs the operation.
     * @param[in] op_data Optional operation-specific metadata (EinsumDescriptor, etc.).
     * @throws std::logic_error If called outside of a capture context.
     */
    template <typename F>
    void record(OpKind kind, std::string label, std::vector<TensorId> inputs, std::vector<TensorId> outputs, F &&executor,
                OpData op_data = std::monostate{}) {
        if (!_capturing || !_graph) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::record called outside of capture");
        }

        Node node;
        node.kind    = kind;
        node.label   = std::move(label);
        node.execute = std::forward<F>(executor);
        node.inputs  = std::move(inputs);
        node.outputs = std::move(outputs);
        node.op_data = std::move(op_data);

        _graph->add_node(std::move(node));
    }

    /**
     * @brief Record a node with asynchronous start/finish phases.
     *
     * Used by read_async() / write_async() to enable I/O-compute overlap
     * in the DataflowExecutor.
     *
     * @param kind          The operation kind (e.g., DiskRead, DiskWrite).
     * @param label         Human-readable label.
     * @param inputs        TensorIds read by this operation.
     * @param outputs       TensorIds written by this operation.
     * @param executor      Synchronous fallback (for Sequential/OpenMP executors).
     * @param async_start   Lambda that begins the async operation (non-blocking).
     * @param async_finish  Lambda that waits for the async operation to complete.
     * @param op_data       Operation-specific metadata.
     */
    template <typename F, typename StartFn, typename FinishFn>
    void record_async(OpKind kind, std::string label, std::vector<TensorId> inputs, std::vector<TensorId> outputs, F &&executor,
                      StartFn &&async_start_fn, FinishFn &&async_finish_fn, OpData op_data = std::monostate{}) {
        if (!_capturing || !_graph) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::record_async called outside of capture");
        }

        Node node;
        node.kind         = kind;
        node.label        = std::move(label);
        node.execute      = std::forward<F>(executor);
        node.async_start  = std::forward<StartFn>(async_start_fn);
        node.async_finish = std::forward<FinishFn>(async_finish_fn);
        node.inputs       = std::move(inputs);
        node.outputs      = std::move(outputs);
        node.op_data      = std::move(op_data);

        _graph->add_node(std::move(node));
    }

    /**
     * @brief Look up or create a TensorId for a given tensor.
     *
     * If the tensor was already registered (same memory address), returns its
     * existing TensorId. Otherwise, creates a new TensorHandle via make_handle()
     * and registers it with the graph.
     *
     * @tparam TensorType Any type satisfying CoreBasicTensorConcept.
     * @param[in] tensor The tensor to look up or register.
     * @return The tensor's TensorId in the current graph.
     */
    template <CoreBasicTensorConcept TensorType>
    TensorId get_or_register(TensorType const &tensor) {
        void *ptr = const_cast<void *>(static_cast<void const *>(&tensor));

        // Check capture-local cache first
        auto it = _ptr_to_id.find(ptr);
        if (it != _ptr_to_id.end()) {
            return it->second;
        }

        // Check if the graph already has this tensor registered (e.g., from create_tensor())
        for (auto const &[tid, handle] : _graph->tensors_map()) {
            if (handle.tensor_ptr == ptr) {
                _ptr_to_id[ptr] = tid;
                return tid;
            }
        }

        // New tensor — register it
        TensorId id     = _graph->register_tensor(make_handle(tensor, 0));
        _ptr_to_id[ptr] = id;
        return id;
    }

    /**
     * @brief Look up or create a TensorId for a scalar value.
     *
     * @tparam T An arithmetic or complex-arithmetic type.
     * @param[in] scalar Pointer to the scalar.
     * @param[in] name Human-readable name.
     * @return The scalar's TensorId in the current graph.
     */
    template <typename T>
        requires(std::is_arithmetic_v<T> || IsComplexV<T>)
    TensorId get_or_register_scalar(T *scalar, std::string name = "scalar") {
        void *ptr = static_cast<void *>(scalar);
        auto  it  = _ptr_to_id.find(ptr);
        if (it != _ptr_to_id.end()) {
            return it->second;
        }

        TensorId id     = _graph->register_tensor(make_scalar_handle(scalar, 0, std::move(name)));
        auto    &handle = _graph->tensor(id);
        _ptr_to_id[ptr] = handle.id;
        return handle.id;
    }

    /**
     * @brief Get or create a TensorSlot for a tensor (for rebindable capture).
     *
     * Registers the tensor if needed, creates a TensorSlot in the graph,
     * and returns both the TensorId and a stable TensorSlot pointer.
     * The slot pointer is captured by executor lambdas for rebind support.
     *
     * @tparam TensorType Tensor type.
     * @param[in] tensor The tensor.
     * @return Pair of (TensorId, TensorSlot pointer).
     */
    template <CoreBasicTensorConcept TensorType>
    std::pair<TensorId, TensorSlot *> get_slot(TensorType const &tensor) {
        TensorId id = get_or_register(tensor);
        return {id, _graph->get_or_create_slot(tensor, id)};
    }

    /// Get-or-create a TensorSlot for an already-registered tensor.
    /// Used by capture-time helpers (e.g. ``cg::view``) that need to
    /// create a slot for a tensor object they constructed themselves
    /// — the @ref get_slot path requires the object's address to match
    /// a re-registration cycle, but explicit-registration callers can
    /// bypass that.
    template <CoreBasicTensorConcept TensorType>
    TensorSlot *create_slot_for(TensorType const &tensor, TensorId id) {
        return _graph->get_or_create_slot(tensor, id);
    }

    /// Active @ref ParamTable shared_ptr (from the active graph). Used by
    /// helpers that bake parameter access into executor lambdas.
    [[nodiscard]] std::shared_ptr<ParamTable> const &params_ptr() const {
        if (!_graph)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::params_ptr called outside of capture");
        return _graph->params_ptr();
    }

  private:
    Graph                               *_graph{nullptr};
    bool                                 _capturing{false};
    std::unordered_map<void *, TensorId> _ptr_to_id; ///< Maps tensor address → TensorId for deduplication
};

/**
 * @brief RAII guard for graph capture.
 *
 * Creates a capture scope: operations called between construction and destruction
 * are recorded into the graph instead of executing immediately.
 *
 * @par Example
 * @code
 * cg::Graph graph("example");
 * {
 *     cg::CaptureGuard guard(graph);
 *     cg::einsum(...);  // Recorded, not executed
 *     cg::scale(...);   // Recorded, not executed
 * }
 * // guard destroyed → capture ends, graph is topologically sorted
 * graph.execute();  // NOW the operations run
 * @endcode
 *
 * @note Not copyable or movable. Nested captures are not supported.
 * @see CaptureContext for the underlying state machine
 */
struct CaptureGuard {
    /**
     * @brief Begin capturing operations into the given graph.
     * @param[in,out] g The graph to capture into.
     */
    explicit CaptureGuard(Graph &g) { CaptureContext::current().begin_capture(g); }

    /// End capture and finalize the graph.
    ~CaptureGuard() { CaptureContext::current().end_capture(); }

    CaptureGuard(CaptureGuard const &)            = delete;
    CaptureGuard &operator=(CaptureGuard const &) = delete;
    CaptureGuard(CaptureGuard &&)                 = delete;
    CaptureGuard &operator=(CaptureGuard &&)      = delete;
};

// ── Deferred template definitions (require CaptureGuard) ────────────────────

template <typename ThenFn, typename ElseFn>
void Graph::add_conditional(std::string label, std::function<bool()> predicate, ThenFn &&then_fn, ElseFn &&else_fn) {
    auto [then_g, else_g] = add_conditional(std::move(label), std::move(predicate));
    {
        CaptureGuard const g(then_g);
        then_fn();
    }
    if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<ElseFn>>) {
        CaptureGuard const g(else_g);
        else_fn();
    }
}

template <typename BodyFn>
void Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition, BodyFn &&body_fn) {
    auto              &body = add_loop(std::move(label), max_iterations, std::move(condition));
    CaptureGuard const g(body);
    body_fn();
}

} // namespace einsums::compute_graph
