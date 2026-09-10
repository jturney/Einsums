//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <functional>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

GPUPlacement::GPUPlacement(size_t min_bytes) : _min_bytes(min_bytes) {
}

GPUPlacement::GPUPlacement(CostModel const &cost_model, size_t min_bytes) : _min_bytes(min_bytes) {
    cpu_throughput_gflops = cost_model.cpu.peak_gflops_fp64;
    if (cost_model.has_gpu()) {
        gpu_throughput_gflops  = cost_model.gpu.peak_gflops_fp64;
        pcie_bandwidth_gbs     = cost_model.gpu.pcie_bandwidth_gbps;
        gpu_launch_overhead_us = cost_model.gpu.gpu_launch_latency_us;
    }
}

namespace {

/// Check if an OpKind is one the executor can actually dispatch to the GPU.
///
/// This list must match try_gpu_blas_dispatch in Graph.cpp, and nothing wider.
/// It used to advertise the whole BLAS/LAPACK surface - Syev, Heev, Gesv, Getrf,
/// Getrs, Getri, Invert, SVD, SVD_DD, QR, Geev, Ger, Dot, DirectProduct,
/// SymmGemm - none of which has a GPU execution path. Placing them marked every
/// one Target::GPU and then dropped it into the CPU fallback, which on a
/// discrete device means running a host kernel while the operand pointers are
/// swapped to device shadows. The fallback now restores host pointers first, so
/// that is no longer fatal, but the transfers and the placement bookkeeping were
/// pure overhead for work that was never going to run on the device.
///
/// Add a kind back here only together with its dispatch path.
bool is_gpu_capable_op(OpKind kind) {
    switch (kind) {
    case OpKind::Gemm:        // try_gpu_gemm
    case OpKind::BatchedGemm: // try_gpu_batched_gemm (strided only)
    case OpKind::Gemv:        // try_gpu_gemv
    case OpKind::Axpby:       // try_gpu_axpy
    case OpKind::Scale:       // try_gpu_scale
        return true;
    default:
        return false;
    }
}

/// Check if the backend supports GPU BLAS for the given element type.
///
/// MPS only supports float32.
///
/// For CUDA/HIP/mock this reports the dtypes the executor's dispatch helpers
/// actually handle. Of those, only the strided-batched GEMM path takes complex;
/// try_gpu_gemm, try_gpu_gemv, try_gpu_scale and try_gpu_axpy all return false
/// for anything but Float32/Float64. Claiming complex here (as this did) placed
/// complex einsums on the GPU that the executor then handed straight back to the
/// CPU. The node's OpKind is threaded in so the complex case can be answered
/// per-op rather than per-backend.
bool backend_supports_dtype(packed_gemm::ScalarType dtype, OpKind kind) {
    if constexpr (gpu::has_mps) {
        return dtype == packed_gemm::ScalarType::Float32;
    }

    if (dtype == packed_gemm::ScalarType::Float32 || dtype == packed_gemm::ScalarType::Float64) {
        return true;
    }
    // Complex reaches the device only through gemm_strided_batched.
    if (dtype == packed_gemm::ScalarType::Complex64 || dtype == packed_gemm::ScalarType::Complex128) {
        return kind == OpKind::BatchedGemm;
    }
    return false;
}

/// Check if all tensors involved in a node are supported by the GPU backend.
bool node_dtypes_supported(Node const &node, Graph const &graph) {
    for (auto tid : node.inputs) {
        if (!backend_supports_dtype(graph.tensor(tid).dtype, node.kind))
            return false;
    }
    for (auto tid : node.outputs) {
        if (!backend_supports_dtype(graph.tensor(tid).dtype, node.kind))
            return false;
    }
    return true;
}

/// True when any tensor the node touches is tile-wise sparse.
///
/// A tiled tensor has no single contiguous buffer -- its ``data_ptr`` is null and
/// its storage is one dense tensor per populated tile -- so the H2D/D2H transfer
/// nodes that placement inserts have nothing to move. Placing such a node makes
/// TransferInsertion emit a copy from a null pointer.
///
/// This is NOT hypothetical: a float32 tiled ``dot`` over ``min_bytes`` is
/// otherwise a valid candidate, since ``dot`` is in @ref is_gpu_capable_op and
/// ``TensorHandle::total_bytes`` reports the honest GLOBAL size regardless of tile
/// sparsity. It goes unnoticed on Apple Silicon only because
/// ``gpu::has_unified_memory`` makes the transfers no-ops there; on a discrete
/// CUDA or HIP build the same graph memcpys from nullptr.
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

/// Compute estimated bytes from a node's input/output tensor handles.
size_t compute_bytes_from_tensors(Node const &node, Graph const &graph) {
    size_t bytes = 0;
    for (auto tid : node.inputs) {
        bytes += graph.tensor(tid).total_bytes();
    }
    for (auto tid : node.outputs) {
        bytes += graph.tensor(tid).total_bytes();
    }
    return bytes;
}

} // namespace

void GPUPlacement::reset_stats() {
    _num_placed = 0;
}

bool GPUPlacement::run(Graph &graph) {
    PassCounter const placed{_num_placed};

    // Is there a usable device right now? gpu::has_gpu cannot answer that - it is
    // a build flag, and `!has_gpu && !is_mock` was a tautologically false guard
    // (is_mock is defined as !has_gpu), so this never returned early at all. A
    // CUDA-enabled binary on a node with no driver therefore still placed work on
    // a "GPU": device_malloc failed, the shadow map cached nullptr, and the
    // executor swapped a tensor's data pointer to it.
    if (!gpu::gpu_available()) {
        EINSUMS_LOG_INFO("GPUPlacement: no usable GPU device, keeping every node on the host");
        return false;
    }

    // Check --einsums:gpu:disable runtime flag. A read is a lock-free load of the
    // descriptor's slot, safe before registration (it yields the declared default)
    // and incapable of failing, so there is nothing here to guard against.
    if (config::get(option::GpuDisable)) {
        EINSUMS_LOG_INFO("GPUPlacement: disabled via --einsums:gpu:disable");
        return false;
    }

    // Phase 1: Identify candidates across the whole graph tree (loop
    // bodies, conditional branches, nesting). A hot GEMM inside an SCF
    // loop is the common case, so body candidates must be considered.
    // recurse_into_subgraphs() stays false, we walk here so that Phase 2
    // can place everything within a *single shared* device-memory budget
    // rather than letting the parent and each body each consume the full
    // budget (which would over-subscribe device memory and risk OOM).
    struct Candidate {
        Graph *owner;
        size_t node_idx;
        size_t eff_bytes;
    };
    std::vector<Candidate> candidates;

    std::function<void(Graph &)> collect = [&](Graph &g) {
        auto &gnodes = g.nodes();
        for (size_t idx = 0; idx < gnodes.size(); ++idx) {
            auto &node = gnodes[idx];

            if (node.target == Target::GPU)
                continue;

            bool const is_candidate = is_gpu_capable_op(node.kind) || node.kind == OpKind::Einsum;
            if (!is_candidate)
                continue;

            if (!node_dtypes_supported(node, g)) {
                EINSUMS_LOG_DEBUG("GPUPlacement: skipping node {} — unsupported dtype for GPU backend", node.id);
                continue;
            }

            if (node_touches_tiled(node, g)) {
                EINSUMS_LOG_DEBUG("GPUPlacement: skipping node {} — tile-wise sparse operand has no single buffer to transfer", node.id);
                continue;
            }

            size_t eff_bytes = node.estimated_bytes;
            if (eff_bytes == 0) {
                eff_bytes = compute_bytes_from_tensors(node, g);
            }

            // Decide whether this node benefits from GPU execution.
            if (node.estimated_flops > 0) {
                auto         flops    = static_cast<double>(node.estimated_flops);
                auto         bytes    = static_cast<double>(eff_bytes);
                double const cpu_time = flops / (cpu_throughput_gflops * 1e9);
                double const gpu_time = flops / (gpu_throughput_gflops * 1e9) + bytes / (pcie_bandwidth_gbs * 1e9) //
                                        + gpu_launch_overhead_us * 1e-6;
                if (gpu_time >= cpu_time) {
                    EINSUMS_LOG_DEBUG("GPUPlacement: cost model rejects node {} (cpu={:.3f}us, gpu={:.3f}us)", node.id, cpu_time * 1e6,
                                      gpu_time * 1e6);
                    continue;
                }
            } else {
                if (eff_bytes < _min_bytes)
                    continue;
            }

            candidates.push_back({.owner = &g, .node_idx = idx, .eff_bytes = eff_bytes});
        }
        g.for_each_subgraph([&](Graph &sub) { collect(sub); });
    };
    collect(graph);

    if (candidates.empty())
        return false;

    // Phase 2: Budget-aware greedy placement across the whole tree.
    // Sort candidates by bytes descending, prioritize the largest operations.
    std::ranges::sort(candidates, [](Candidate const &a, Candidate const &b) { return a.eff_bytes > b.eff_bytes; });

    size_t const budget = gpu::available_device_memory();
    size_t       used   = 0;

    for (auto const &cand : candidates) {
        auto &placed_node = cand.owner->nodes()[cand.node_idx];

        if (used + cand.eff_bytes > budget) {
            EINSUMS_LOG_INFO("GPUPlacement: skipping node {} (needs {} bytes, budget has {} remaining)", placed_node.id, cand.eff_bytes,
                             budget - used);
            continue;
        }

        placed_node.target       = Target::GPU;
        placed_node.cpu_fallback = placed_node.execute; // Save original CPU executor for fallback.
        used += cand.eff_bytes;
        _num_placed++;

        EINSUMS_LOG_INFO("GPUPlacement: placed {} node {} ({}) on GPU (bytes={}, budget_used={}/{})", op_kind_name(placed_node.kind),
                         placed_node.id, placed_node.label, cand.eff_bytes, used, budget);
        report(2, fmt::format("place {} node {} ({}) on GPU (cost model favored it; {} bytes)", op_kind_name(placed_node.kind),
                              placed_node.id, placed_node.label, cand.eff_bytes));
    }

    if (placed.moved()) {
        EINSUMS_LOG_INFO("GPUPlacement: placed {} nodes on GPU ({} / {} bytes used)", _num_placed, used, budget);
        report(1, fmt::format("placed {} node(s) on GPU ({} / {} bytes used)", _num_placed, used, budget));
    }

    return placed.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
