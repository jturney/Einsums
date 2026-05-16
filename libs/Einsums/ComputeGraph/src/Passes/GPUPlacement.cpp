//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/Config/Types.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <vector>

namespace einsums::compute_graph::passes {

GPUPlacement::GPUPlacement(size_t min_flops, size_t min_bytes) : _min_flops(min_flops), _min_bytes(min_bytes) {
}

GPUPlacement::GPUPlacement(HardwareProfile const &profile, size_t min_flops, size_t min_bytes)
    : _min_flops(min_flops), _min_bytes(min_bytes) {
    cpu_throughput_gflops = profile.cpu.peak_gflops_fp64;
    if (profile.has_gpu()) {
        gpu_throughput_gflops  = profile.gpu.peak_gflops_fp64;
        pcie_bandwidth_gbs     = profile.gpu.pcie_bandwidth_gbps;
        gpu_launch_overhead_us = profile.gpu.gpu_launch_latency_us;
    }
}

namespace {

/// Check if an OpKind is a BLAS/LAPACK operation that has a GPU implementation.
bool is_gpu_capable_op(OpKind kind) {
    switch (kind) {
    // BLAS Level 2/3
    case OpKind::Gemm:
    case OpKind::BatchedGemm:
    case OpKind::Gemv:
    case OpKind::Ger:
    case OpKind::Dot:
    case OpKind::Axpy:
    case OpKind::Axpby:
    case OpKind::Scale:
    case OpKind::DirectProduct:
    case OpKind::SymmGemm:
    // LAPACK
    case OpKind::Syev:
    case OpKind::Heev:
    case OpKind::Gesv:
    case OpKind::Getrf:
    case OpKind::Getri:
    case OpKind::Invert:
    case OpKind::SVD:
    case OpKind::SVD_DD:
    case OpKind::QR:
    case OpKind::Geev:
        return true;
    default:
        return false;
    }
}

/// Check if the backend supports GPU BLAS for the given element type.
/// MPS only supports float32. CUDA/HIP support float32, float64, and complex.
bool backend_supports_dtype(packed_gemm::ScalarType dtype) {
    if constexpr (gpu::has_mps) {
        return dtype == packed_gemm::ScalarType::Float32;
    }
    // CUDA/HIP/mock: support all standard types.
    return dtype == packed_gemm::ScalarType::Float32 || dtype == packed_gemm::ScalarType::Float64 ||
           dtype == packed_gemm::ScalarType::Complex64 || dtype == packed_gemm::ScalarType::Complex128;
}

/// Check if all tensors involved in a node are supported by the GPU backend.
bool node_dtypes_supported(Node const &node, Graph const &graph) {
    for (auto tid : node.inputs) {
        if (!backend_supports_dtype(graph.tensor(tid).dtype))
            return false;
    }
    for (auto tid : node.outputs) {
        if (!backend_supports_dtype(graph.tensor(tid).dtype))
            return false;
    }
    return true;
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

bool GPUPlacement::run(Graph &graph) {
    // No GPU backend available — nothing to do.
    if constexpr (!gpu::has_gpu && !gpu::is_mock) {
        _num_placed = 0;
        return false;
    }

    // Check --einsums:disable-gpu runtime flag.
    try {
        auto &gc = GlobalConfigMap::get_singleton();
        if (gc.get_bool("disable-gpu", false)) {
            EINSUMS_LOG_INFO("GPUPlacement: disabled via --einsums:disable-gpu");
            _num_placed = 0;
            return false;
        }
    } catch (...) { // NOLINT
        // Config not available (e.g., in unit tests) — proceed normally.
    }

    _num_placed = 0;
    auto &nodes = graph.nodes();

    // Phase 1: Identify candidates and compute effective bytes.
    struct Candidate {
        size_t node_idx;
        size_t eff_bytes;
    };
    std::vector<Candidate> candidates;

    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        auto &node = nodes[idx];

        if (node.target == Target::GPU)
            continue;

        bool const is_candidate = is_gpu_capable_op(node.kind) || node.kind == OpKind::Einsum;
        if (!is_candidate)
            continue;

        // Check if the backend supports the tensor element types.
        if (!node_dtypes_supported(node, graph)) {
            EINSUMS_LOG_DEBUG("GPUPlacement: skipping node {} — unsupported dtype for GPU backend", node.id);
            continue;
        }

        size_t eff_bytes = node.estimated_bytes;
        if (eff_bytes == 0) {
            eff_bytes = compute_bytes_from_tensors(node, graph);
        }

        // Decide whether this node benefits from GPU execution.
        if (node.estimated_flops > 0) {
            // Cost model: compare estimated CPU time vs GPU time + transfer overhead.
            auto         flops    = static_cast<double>(node.estimated_flops);
            auto         bytes    = static_cast<double>(eff_bytes);
            double const cpu_time = flops / (cpu_throughput_gflops * 1e9); // seconds
            double const gpu_time = flops / (gpu_throughput_gflops * 1e9)  // compute
                                    + bytes / (pcie_bandwidth_gbs * 1e9)   // transfer
                                    + gpu_launch_overhead_us * 1e-6;       // launch
            if (gpu_time >= cpu_time) {
                EINSUMS_LOG_DEBUG("GPUPlacement: cost model rejects node {} (cpu={:.3f}us, gpu={:.3f}us)", node.id, cpu_time * 1e6,
                                  gpu_time * 1e6);
                continue;
            }
        } else {
            // No flops estimate — fall back to size thresholds.
            if (eff_bytes < _min_bytes)
                continue;
        }

        candidates.push_back({.node_idx = idx, .eff_bytes = eff_bytes});
    }

    if (candidates.empty())
        return false;

    // Phase 2: Budget-aware greedy placement.
    // Sort candidates by bytes descending — prioritize the largest operations.
    std::ranges::sort(candidates, [](Candidate const &a, Candidate const &b) { return a.eff_bytes > b.eff_bytes; });

    size_t budget = gpu::available_device_memory();
    size_t used   = 0;

    for (auto const &cand : candidates) {
        if (used + cand.eff_bytes > budget) {
            EINSUMS_LOG_INFO("GPUPlacement: skipping node {} (needs {} bytes, budget has {} remaining)", nodes[cand.node_idx].id,
                             cand.eff_bytes, budget - used);
            continue;
        }

        auto &placed_node        = nodes[cand.node_idx];
        placed_node.target       = Target::GPU;
        placed_node.cpu_fallback = placed_node.execute; // Save original CPU executor for fallback.
        used += cand.eff_bytes;
        _num_placed++;

        EINSUMS_LOG_INFO("GPUPlacement: placed {} node {} ({}) on GPU (bytes={}, budget_used={}/{})",
                         op_kind_name(nodes[cand.node_idx].kind), nodes[cand.node_idx].id, nodes[cand.node_idx].label, cand.eff_bytes, used,
                         budget);
    }

    if (_num_placed > 0) {
        EINSUMS_LOG_INFO("GPUPlacement: placed {} nodes on GPU ({} / {} bytes used)", _num_placed, used, budget);
    }

    return _num_placed > 0;
}

} // namespace einsums::compute_graph::passes
