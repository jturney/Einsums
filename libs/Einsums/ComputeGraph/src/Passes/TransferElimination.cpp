//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Passes/TransferElimination.hpp>
#include <Einsums/ComputeGraph/Passes/TransferNode.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// For each tensor, compute the next position (node index) where it is used as
/// a GPU node input, starting from each position in the node list.
/// next_gpu_use[tid] = sorted list of node indices where tid is a GPU input.
using NextUseMap = std::unordered_map<TensorId, std::vector<size_t>>;

NextUseMap build_next_use_map(std::vector<Node> const &nodes) {
    NextUseMap m;
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        if (nodes[idx].target == Target::GPU) {
            for (auto tid : nodes[idx].inputs) {
                m[tid].push_back(idx);
            }
        }
    }
    return m;
}

/// Given a tensor and the current position, find the next GPU use.
/// Returns SIZE_MAX if no future GPU use exists.
size_t next_gpu_use_after(NextUseMap const &map, TensorId tid, size_t after) {
    auto it = map.find(tid);
    if (it == map.end())
        return std::numeric_limits<size_t>::max();
    // Binary search for first use > after.
    auto &uses = it->second;
    auto  pos  = std::ranges::upper_bound(uses, after);
    if (pos == uses.end())
        return std::numeric_limits<size_t>::max();
    return *pos;
}

} // namespace

bool TransferElimination::transform(Graph &graph) {
    PassCounter const eliminated{_stats.num_eliminated};
    graph.topological_sort();

    auto const &nodes = graph.nodes();
    if (nodes.empty()) {
        return false;
    }

    // Build next-use map for Belady eviction decisions.
    auto next_use = build_next_use_map(nodes);

    // Identify pinned tensors (referenced by a sub-graph body, don't evict). A body numbers its
    // tensors independently of this graph, so the question is asked of the storage, not the id.
    auto const                   guard = EscapeAnalysis::over(graph);
    std::unordered_set<TensorId> pinned;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (guard.touched_by_subtree(tid)) {
            pinned.insert(tid);
        }
    }

    // GPU memory tracking.
    size_t const                            budget   = gpu::available_device_memory();
    size_t                                  gpu_used = 0;
    std::unordered_map<TensorId, size_t>    device_resident; // tid → bytes on device
    std::unordered_map<TensorId, Residency> residency;

    for (auto const &[tid, handle] : graph.tensors_map()) {
        residency[tid] = Residency::Host;
    }

    std::vector<bool> remove(nodes.size(), false);

    // We may need to insert eviction D2H nodes. Collect them as (insert_before_idx, Node).
    struct Insertion {
        size_t before_idx;
        Node   node;
    };
    std::vector<Insertion> insertions;

    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        auto const &node = nodes[idx];

        if (node.kind == OpKind::HostToDevice) {
            auto const *desc = node.op_data.get_if<TransferDescriptor>();
            if (!desc)
                continue;

            auto tid = desc->tensor_id;
            auto res = residency[tid];

            if (res == Residency::Device || res == Residency::Both) {
                // A transfer this pass does not understand stays and leaves both copies current,
                // as any kept upload does; the tensor already holds its device bytes, so the
                // budget has nothing new to count.
                if (!admits(graph, node)) {
                    residency[tid] = Residency::Both;
                    continue;
                }
                // Already on device, redundant.
                remove[idx] = true;
                _stats.num_eliminated++;
                EINSUMS_LOG_INFO("TransferElimination: removed redundant H2D for tensor id={}", tid);
                continue;
            }

            // Check if bringing this tensor to device exceeds budget.
            size_t const tensor_bytes = desc->size_bytes;
            while (gpu_used + tensor_bytes > budget && !device_resident.empty()) {
                // Belady: evict the tensor whose next GPU use is furthest away.
                TensorId evict_tid       = 0;
                size_t   farthest_use    = 0;
                size_t   evict_bytes     = 0;
                bool     found_evictable = false;

                for (auto const &[res_tid, res_bytes] : device_resident) {
                    if (pinned.count(res_tid))
                        continue; // Don't evict pinned tensors.
                    size_t const nu = next_gpu_use_after(next_use, res_tid, idx);
                    if (!found_evictable || nu > farthest_use) {
                        evict_tid       = res_tid;
                        farthest_use    = nu;
                        evict_bytes     = res_bytes;
                        found_evictable = true;
                    }
                }

                if (!found_evictable)
                    break; // All resident tensors are pinned, can't free space.

                // Insert a D2H eviction node.
                auto &evict_handle = graph.tensor(evict_tid);

                Node evict_node = make_transfer_node(OpKind::DeviceToHost, evict_handle, "D2H_evict", evict_bytes);

                insertions.push_back({.before_idx = idx, .node = std::move(evict_node)});

                gpu_used -= evict_bytes;
                device_resident.erase(evict_tid);
                residency[evict_tid] = Residency::Host;

                EINSUMS_LOG_INFO("TransferElimination: evicting tensor '{}' (id={}, {} bytes) — next GPU use at {}", evict_handle.name,
                                 evict_tid, evict_bytes,
                                 farthest_use == std::numeric_limits<size_t>::max() ? -1L : static_cast<long>(farthest_use));
            }

            // Keep this H2D transfer.
            residency[tid]       = Residency::Both;
            device_resident[tid] = tensor_bytes;
            gpu_used += tensor_bytes;

        } else if (node.kind == OpKind::DeviceToHost) {
            auto const *desc = node.op_data.get_if<TransferDescriptor>();
            if (!desc)
                continue;

            auto tid = desc->tensor_id;
            auto res = residency[tid];

            if (res == Residency::Host || res == Residency::Both) {
                // A transfer this pass does not understand stays and leaves both copies current,
                // as any kept download does.
                if (!admits(graph, node)) {
                    residency[tid] = Residency::Both;
                    continue;
                }
                // Already on host, redundant.
                remove[idx] = true;
                _stats.num_eliminated++;
                EINSUMS_LOG_INFO("TransferElimination: removed redundant D2H for tensor id={}", tid);
            } else {
                residency[tid] = Residency::Both;
            }
        } else if (node.target == Target::GPU) {
            for (auto tid : node.outputs) {
                residency[tid] = Residency::Device;
                // GPU output is now on device.
                if (!device_resident.contains(tid)) {
                    size_t const bytes   = graph.tensor(tid).total_bytes();
                    device_resident[tid] = bytes;
                    gpu_used += bytes;
                }
            }
        } else {
            // CPU node outputs go to host.
            for (auto tid : node.outputs) {
                residency[tid] = Residency::Host;
                // If tensor was on device, it's no longer valid there.
                auto it = device_resident.find(tid);
                if (it != device_resident.end()) {
                    gpu_used -= it->second;
                    device_resident.erase(it);
                }
            }
        }
    }

    // Build the final node list: apply removals and insertions.
    bool const modified = (eliminated.moved() || !insertions.empty());

    if (!modified)
        return false;

    // Sort insertions by position (descending) so we can insert without shifting indices.
    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.before_idx > b.before_idx; });

    // Each eviction goes in ahead of the node at its position, whether that node stays or goes.
    std::unordered_map<size_t, std::vector<size_t>> insert_before; // position → insertion indices
    for (size_t i = 0; i < insertions.size(); ++i) {
        insert_before[insertions[i].before_idx].push_back(i);
    }
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;
    inserts.reserve(insert_before.size());
    for (auto &[at, which] : insert_before) {
        std::vector<Node> group;
        group.reserve(which.size());
        for (size_t const i : which) {
            group.push_back(std::move(insertions[i].node));
        }
        inserts.emplace_back(at, std::move(group));
    }
    graph.replace_nodes(remove, std::move(inserts));

    // Update TensorHandle residency.
    for (auto const &[tid, res] : residency) {
        graph.tensor(tid).residency = res;
    }

    if (eliminated.moved() || !insertions.empty()) {
        EINSUMS_LOG_INFO("TransferElimination: eliminated {} redundant transfers, inserted {} evictions", _stats.num_eliminated,
                         insertions.size());
        report(1, fmt::format("eliminated {} redundant transfer(s), inserted {} eviction(s)", _stats.num_eliminated, insertions.size()));
    }

    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
