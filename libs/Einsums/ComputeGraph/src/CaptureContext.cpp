//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

CaptureContext &CaptureContext::current() {
    thread_local CaptureContext instance;
    return instance;
}

void CaptureContext::begin_capture(Graph &graph) {
    if (_capturing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext: already capturing. Nested captures are not supported.");
    }
    profile::Profiler::instance().push(fmt::format("ComputeGraph::capture({})", graph.name()));
    _graph     = &graph;
    _capturing = true;
    _ptr_to_id.clear();
}

void CaptureContext::end_capture() {
    if (!_capturing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext: not currently capturing.");
    }

    // Reset capture state and balance the profiler push unconditionally,
    // before running validation/finalization that can throw. If validation
    // fails the caller still sees the exception, but the next `with
    // cg.capture(...)` is no longer locked out by a stale `_capturing` flag.
    Graph *g   = _graph;
    _capturing = false;
    _graph     = nullptr;
    _ptr_to_id.clear();
    profile::Profiler::instance().pop();

    g->topological_sort();
    g->validate_shapes_at_capture();
    profile::annotate("num_nodes", static_cast<int64_t>(g->num_nodes()));
    profile::annotate("num_tensors", static_cast<int64_t>(g->num_tensors()));
    register_graph(g);
}

template <OpDataOrAlternative D>
void CaptureContext::record(OpKind kind, std::string label, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                            std::function<void()> executor, D op_data) {
    if (!_capturing || !_graph) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::record called outside of capture");
    }

    Node node;
    node.kind    = kind;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.inputs  = std::move(inputs);
    node.outputs = std::move(outputs);
    node.op_data = std::move(op_data);

    _graph->add_node(std::move(node));
}

template <OpDataOrAlternative D>
void CaptureContext::record_built(OpKind kind, std::string label, packed_gemm::ScalarType dtype, std::size_t rank, D op_data,
                                  std::span<TensorId const> build_inputs, std::span<TensorId const> build_outputs,
                                  std::vector<TensorId> inputs, std::vector<TensorId> outputs) {
    if (!_capturing || !_graph) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::record_built called outside of capture");
    }

    OpData data(std::move(op_data));
    auto   executor = build_executor(kind, dtype, rank, data, *_graph, build_inputs, build_outputs);
    record(kind, std::move(label), std::move(inputs), std::move(outputs), std::move(executor), std::move(data));
}

void CaptureContext::record_async(OpKind kind, std::string label, std::vector<TensorId> inputs, std::vector<TensorId> outputs,
                                  std::function<void()> executor, std::function<void()> async_start, std::function<void()> async_finish,
                                  DiskIODescriptor op_data) {
    if (!_capturing || !_graph) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext::record_async called outside of capture");
    }

    Node node;
    node.kind         = kind;
    node.label        = std::move(label);
    node.execute      = std::move(executor);
    node.async_start  = std::move(async_start);
    node.async_finish = std::move(async_finish);
    node.inputs       = std::move(inputs);
    node.outputs      = std::move(outputs);
    node.op_data      = std::move(op_data);

    _graph->add_node(std::move(node));
}

// Every alternative of OpData, in its order. The count check below fails the
// build when an alternative is added to OpData and not here.
#define EINSUMS_OP_DATA_ALTERNATIVES(X)                                                                                                    \
    X(std::monostate)                                                                                                                      \
    X(EinsumDescriptor)                                                                                                                    \
    X(ScaleDescriptor)                                                                                                                     \
    X(PermuteDescriptor)                                                                                                                   \
    X(ConditionalDescriptor)                                                                                                               \
    X(LoopDescriptor)                                                                                                                      \
    X(AllocDescriptor)                                                                                                                     \
    X(TransferDescriptor)                                                                                                                  \
    X(DiskIODescriptor)                                                                                                                    \
    X(CommDescriptor)                                                                                                                      \
    X(InitializeDescriptor)                                                                                                                \
    X(BatchedGemmDescriptor)                                                                                                               \
    X(GroupedBatchedGemmDescriptor)                                                                                                        \
    X(ViewDescriptor)                                                                                                                      \
    X(WriteParamDescriptor)                                                                                                                \
    X(AxpbyDescriptor)                                                                                                                     \
    X(GroupedDotDescriptor)                                                                                                                \
    X(GroupedAxpbyDescriptor)                                                                                                              \
    X(GroupedElementwiseDescriptor)                                                                                                        \
    X(GroupedSandwichDescriptor)                                                                                                           \
    X(GroupedGatherRotateDescriptor)                                                                                                       \
    X(TiledEinsumDescriptor)                                                                                                               \
    X(TiledElementwiseDescriptor)                                                                                                          \
    X(TiledPermuteDescriptor)                                                                                                              \
    X(TiledDotDescriptor)                                                                                                                  \
    X(ElementwiseBinaryDescriptor)                                                                                                         \
    X(DotDescriptor)                                                                                                                       \
    X(TraceDescriptor)                                                                                                                     \
    X(GemmDescriptor)                                                                                                                      \
    X(ElementTransformDescriptor)                                                                                                          \
    X(SetupDescriptor)                                                                                                                     \
    X(SyevDescriptor)                                                                                                                      \
    X(LaplaceQuadratureDescriptor)                                                                                                         \
    X(OuterSumDescriptor)

#define EINSUMS_COUNT_ONE(D) +1
static_assert(0 EINSUMS_OP_DATA_ALTERNATIVES(EINSUMS_COUNT_ONE) == std::variant_size_v<OpData>,
              "EINSUMS_OP_DATA_ALTERNATIVES must list every OpData alternative");
#undef EINSUMS_COUNT_ONE

#define EINSUMS_RECORD(D)                                                                                                                  \
    template EINSUMS_EXPORT void CaptureContext::record<D>(OpKind, std::string, std::vector<TensorId>, std::vector<TensorId>,              \
                                                           std::function<void()>, D);                                                      \
    template EINSUMS_EXPORT void CaptureContext::record_built<D>(OpKind, std::string, packed_gemm::ScalarType, std::size_t, D,             \
                                                                 std::span<TensorId const>, std::span<TensorId const>,                     \
                                                                 std::vector<TensorId>, std::vector<TensorId>);
EINSUMS_OP_DATA_ALTERNATIVES(EINSUMS_RECORD)
EINSUMS_RECORD(OpData)
#undef EINSUMS_RECORD
#undef EINSUMS_OP_DATA_ALTERNATIVES

#define EINSUMS_CAPTURE_SLOT(...)                                                                                                          \
    template EINSUMS_EXPORT TensorId CaptureContext::get_or_register<__VA_ARGS__>(__VA_ARGS__ const &);                                    \
    template EINSUMS_EXPORT std::pair<TensorId, TensorSlot *> CaptureContext::get_slot<__VA_ARGS__>(__VA_ARGS__ const &);
EINSUMS_CG_COMMON_TENSOR_TYPES(EINSUMS_CAPTURE_SLOT)
#undef EINSUMS_CAPTURE_SLOT

EINSUMS_NAMESPACE_END(compute_graph)
