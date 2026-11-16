//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/DeviceEligibility.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

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
        pcie_bandwidth_gbs     = cost_model.transfer_bandwidth_gbps();
        gpu_launch_overhead_us = cost_model.gpu.gpu_launch_latency_us;
    }
}

namespace {

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

bool GPUPlacement::transform(Graph &graph) {
    PassCounter const placed{_stats.num_placed};

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

    // Which nodes the device CAN run is DeviceEligibility's answer; compute it here too so the
    // pass works built alone. Everything below is the question of whether it SHOULD.
    compute_device_eligibility(graph);

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

            // The shape test comes first and apart from the rest so that a node with no device
            // kernel never reaches admits(), as before.
            if (!node_has_device_kernel(node, g))
                continue;

            if (!admits(g, node)) {
                continue;
            }

            if (!node.eligible.any_device()) {
                EINSUMS_LOG_DEBUG("GPUPlacement: skipping node {}: not eligible for a device (unsupported dtype for the GPU backend, "
                                  "or a tile-wise sparse operand with no single buffer to transfer)",
                                  node.id);
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
        auto const &placed_node = cand.owner->nodes()[cand.node_idx];

        if (used + cand.eff_bytes > budget) {
            EINSUMS_LOG_INFO("GPUPlacement: skipping node {} (needs {} bytes, budget has {} remaining)", placed_node.id, cand.eff_bytes,
                             budget - used);
            continue;
        }

        cand.owner->edit(cand.node_idx, [](NodeEdit &e) { e.place_on(Target::GPU); });
        used += cand.eff_bytes;
        _stats.num_placed++;

        EINSUMS_LOG_INFO("GPUPlacement: placed {} node {} ({}) on GPU (bytes={}, budget_used={}/{})", placed_node.kind, placed_node.id,
                         placed_node.label, cand.eff_bytes, used, budget);
        report(2, fmt::format("place {} node {} ({}) on GPU (cost model favored it; {} bytes)", placed_node.kind, placed_node.id,
                              placed_node.label, cand.eff_bytes));
    }

    if (placed.moved()) {
        EINSUMS_LOG_INFO("GPUPlacement: placed {} nodes on GPU ({} / {} bytes used)", _stats.num_placed, used, budget);
        report(1, fmt::format("placed {} node(s) on GPU ({} / {} bytes used)", _stats.num_placed, used, budget));
    }

    return placed.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
