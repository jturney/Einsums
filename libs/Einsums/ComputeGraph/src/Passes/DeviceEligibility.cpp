//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarTypes.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/DeviceEligibility.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>

#include "../StridedBatchedEinsum.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Can the executor actually dispatch THIS einsum to the GPU?
///
/// OpKind::Einsum covers every contraction the library can express, but
/// try_gpu_blas_dispatch in GpuDispatch.cpp only recognizes two shapes: a plain
/// matrix GEMM (two target indices, one link index, all operands rank 2) and a
/// GEMV (one target index, one link index, rank-2 matrix into a rank-1 vector).
/// A rank-4 tensor contraction matches neither and falls straight back to the
/// CPU.
///
/// Placing those anyway is not free: TransferInsertion wraps them in host-to-
/// device and device-to-host nodes, so the graph pays a full round trip for
/// work that never leaves the host. It was worse than wasteful until the
/// executor learned to skip a device-to-host copy whose GPU node fell back -
/// before that, the stale upload was copied back over the CPU result.

/// Whether @p node is an einsum the device runs as one strided-batched GEMM: the plan the CPU
/// executor and try_gpu_strided_einsum both make, over the operands as they are now.
bool einsum_is_strided_batch(EinsumDescriptor const &desc, Node const &node, Graph const &graph) {
    auto const lists = einsum_index_lists(desc);
    if (desc.conj_a || desc.conj_b || !lists.operators.empty() || node.inputs.size() < 2 || node.outputs.empty()) {
        return false;
    }
    auto const &a = graph.tensor(node.inputs[0]);
    auto const &b = graph.tensor(node.inputs[1]);
    auto const &c = graph.tensor(node.outputs[0]);
    if (!a.has_impl() || !b.has_impl() || !c.has_impl() || a.dtype != c.dtype || b.dtype != c.dtype) {
        return false;
    }
    bool batched = false;
    ::einsums::compute_graph::detail::dispatch_scalar_type(c.dtype, [&]<typename T>(T /*tag*/) {
        using Impl     = ::einsums::detail::TensorImpl<T>;
        auto const *ai = static_cast<Impl const *>(a.impl());
        auto const *bi = static_cast<Impl const *>(b.impl());
        auto const *ci = static_cast<Impl const *>(c.impl());
        batched = ai != nullptr && bi != nullptr && ci != nullptr &&
                  ::einsums::compute_graph::detail::plan_strided_batch(lists.c, lists.a, lists.b, lists.link, *ai, *bi, *ci).has_value();
    });
    return batched;
}

/// Keep this in step with try_gpu_gemm / try_gpu_gemv / try_gpu_strided_einsum.
bool einsum_is_dispatchable(Node const &node, Graph const &graph) {
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr) {
        return false;
    }
    if (einsum_is_strided_batch(*desc, node, graph)) {
        return true;
    }

    size_t const n_target = desc->spec.target_indices.size();
    size_t const n_link   = einsum_index_lists(*desc).link.size();
    if (n_link != 1 || (n_target != 1 && n_target != 2)) {
        return false;
    }

    // Every operand must be small-rank in the way the dispatchers require:
    // rank 2 throughout for GEMM, and rank 2 / rank 1 for GEMV.
    for (auto tid : node.inputs) {
        if (graph.tensor(tid).rank > 2) {
            return false;
        }
    }
    for (auto tid : node.outputs) {
        if (graph.tensor(tid).rank > 2) {
            return false;
        }
    }
    return true;
}

/// Check if the backend supports GPU BLAS for the given element type.
///
/// MPS only supports float32. For CUDA/mock this reports the dtypes the
/// executor's dispatch helpers actually handle: of those, only the
/// strided-batched GEMM path takes complex: try_gpu_gemm, try_gpu_gemv,
/// try_gpu_scale and try_gpu_axpy all return false for anything but
/// Float32/Float64.
bool backend_supports_dtype(packed_gemm::ScalarType dtype, bool strided_batch) {
    if constexpr (gpu::has_mps) {
        return dtype == packed_gemm::ScalarType::Float32;
    }

    if (compute_graph::detail::is_real(dtype)) {
        return true;
    }
    // Complex reaches the device only through gemm_strided_batched.
    return compute_graph::detail::is_complex(dtype) && strided_batch;
}

/// Can the executor actually dispatch this node to the GPU?
///
/// try_gpu_blas_dispatch in GpuDispatch.cpp keys on the DESCRIPTOR carried by the
/// node, not on its OpKind, so this predicate does too. OpKind::Gemm for
/// instance carries a GemmDescriptor, which no dispatcher looks at, so placing
/// it would buy a round trip for work that runs on the CPU anyway.
///
/// Keep it in step with try_gpu_blas_dispatch.
bool node_is_dispatchable(Node const &node, Graph const &graph) {
    // Einsum -> try_gpu_gemm / try_gpu_gemv
    if (node.op_data.holds<EinsumDescriptor>()) {
        return einsum_is_dispatchable(node, graph);
    }
    // A BatchedGemm node is the pointer-array form GEMMBatching and cg::batched_gemm emit, which
    // is CPU-only; a strided batch is an einsum and is judged above.
    // Scale -> try_gpu_scale, Axpby -> try_gpu_axpy. Both are real-scalar only,
    // which the dtype gate also enforces.
    if (node.kind == OpKind::Scale && node.op_data.holds<ScaleDescriptor>()) {
        return true;
    }
    if (node.kind == OpKind::Axpby && node.op_data.holds<AxpbyDescriptor>()) {
        return true;
    }
    return false;
}

/// Check if all tensors involved in a node are supported by the GPU backend.
bool node_dtypes_supported(Node const &node, Graph const &graph) {
    auto const *einsum        = node.op_data.get_if<EinsumDescriptor>();
    bool const  strided_batch = einsum != nullptr && einsum_is_strided_batch(*einsum, node, graph);
    auto const  supported     = [&](TensorId tid) { return backend_supports_dtype(graph.tensor(tid).dtype, strided_batch); };
    return std::ranges::all_of(node.inputs, supported) && std::ranges::all_of(node.outputs, supported);
}

/// True when any tensor the node touches is tile-wise sparse.
///
/// A tiled tensor has no single contiguous buffer -- its ``data_ptr`` is null and
/// its storage is one dense tensor per populated tile -- so the H2D/D2H transfer
/// nodes that placement inserts have nothing to move. Placing such a node makes
/// TransferInsertion emit a copy from a null pointer.
///
/// This was not hypothetical: while every ``dot`` was a placement candidate, a
/// float32 tiled ``dot`` over ``min_bytes`` was placed, because
/// ``TensorHandle::total_bytes`` reports the honest GLOBAL size regardless of tile
/// sparsity. It went unnoticed on Apple Silicon only because
/// ``gpu::has_unified_memory`` makes the transfers no-ops there; on a discrete
/// CUDA build the same graph memcpys from nullptr. @ref node_is_dispatchable
/// no longer admits ``dot``, and the guard stays for every kind it does admit.
bool node_touches_tiled(Node const &node, Graph const &graph) {
    for (auto tid : node.inputs) {
        if (graph.tensor(tid).is_tiled)
            return true;
    }
    for (auto tid : node.outputs) {
        if (graph.tensor(tid).is_tiled)
            return true;
    }
    return false;
}

/// The set one node gets: device 0 when the executor can dispatch it there.
EligiblePlaces eligibility_of(Node const &node, Graph const &graph) {
    EligiblePlaces places;
    if (node_is_dispatchable(node, graph) && node_dtypes_supported(node, graph) && !node_touches_tiled(node, graph)) {
        places.add_device(0);
    }
    return places;
}

/// Set every node of one graph, without descending. Returns how many sets changed.
std::size_t annotate(Graph &graph, std::size_t &eligible) {
    std::size_t changed = 0;
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        EligiblePlaces const places = eligibility_of(graph.nodes()[i], graph);
        if (places.any_device()) {
            ++eligible;
        }
        if (places != graph.nodes()[i].eligible) {
            graph.set_eligible(i, places);
            ++changed;
        }
    }
    return changed;
}

std::size_t annotate_tree(Graph &graph, std::size_t &eligible) {
    std::size_t changed = annotate(graph, eligible);
    graph.for_each_subgraph([&](Graph &sub) { changed += annotate(sub, eligible); });
    return changed;
}

} // namespace

bool node_has_device_kernel(Node const &node, Graph const &graph) {
    return node_is_dispatchable(node, graph);
}

bool compute_device_eligibility(Graph &graph) {
    std::size_t eligible = 0;
    return annotate_tree(graph, eligible) != 0;
}

bool DeviceEligibility::transform(Graph &graph) {
    _stats = {};

    if (!gpu::gpu_available()) {
        note_skip("no usable GPU device", "every node stays host only");
        return false;
    }
    if (config::get(option::GpuDisable)) {
        note_skip("disabled via --einsums:gpu:disable", "every node stays host only");
        return false;
    }

    _stats.num_changed = annotate_tree(graph, _stats.num_eligible);
    if (_stats.num_changed != 0) {
        report(1, fmt::format("{} node(s) can run on a device", _stats.num_eligible));
    }
    return _stats.num_changed != 0;
}

std::vector<std::string> DeviceEligibility::explain() const {
    std::vector<std::string> lines;
    if (_stats.num_eligible != 0) {
        lines.push_back(fmt::format("{} node(s) can run on a device", _stats.num_eligible));
    }
    return lines;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
