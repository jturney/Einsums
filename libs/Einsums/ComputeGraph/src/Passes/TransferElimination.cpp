//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/TransferElimination.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

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

/// Check if a tensor is used inside a Loop node body (should be pinned).
bool is_loop_tensor(std::vector<Node> const &nodes, TensorId tid) {
    for (auto const &node : nodes) {
        if (node.kind == OpKind::Loop) {
            auto const *desc = std::get_if<LoopDescriptor>(&node.op_data);
            if (desc && desc->body) {
                for (auto const &inner : desc->body->nodes()) {
                    for (auto inner_tid : inner.inputs) {
                        if (inner_tid == tid)
                            return true;
                    }
                    for (auto inner_tid : inner.outputs) {
                        if (inner_tid == tid)
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

} // namespace

bool TransferElimination::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        _num_eliminated = 0;
        return false;
    }

    _num_eliminated = 0;

    // Build next-use map for Belady eviction decisions.
    auto next_use = build_next_use_map(nodes);

    // Identify pinned tensors (used in loop bodies — don't evict).
    std::unordered_set<TensorId> pinned;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (is_loop_tensor(nodes, tid)) {
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
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (!desc)
                continue;

            auto tid = desc->tensor_id;
            auto res = residency[tid];

            if (res == Residency::Device || res == Residency::Both) {
                // Already on device — redundant.
                remove[idx] = true;
                _num_eliminated++;
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
                    break; // All resident tensors are pinned — can't free space.

                // Insert a D2H eviction node.
                auto &evict_handle = graph.tensor(evict_tid);

                Node evict_node;
                evict_node.kind   = OpKind::DeviceToHost;
                evict_node.target = Target::CPU;
                evict_node.label  = fmt::format("D2H_evict({})", evict_handle.name);

                TransferDescriptor evict_desc;
                evict_desc.tensor_id  = evict_tid;
                evict_desc.size_bytes = evict_bytes;
                evict_node.op_data    = evict_desc;

                evict_node.inputs          = {evict_tid};
                evict_node.outputs         = {evict_tid};
                evict_node.estimated_bytes = evict_bytes;
                evict_node.execute         = []() { gpu::device_synchronize(); };

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
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (!desc)
                continue;

            auto tid = desc->tensor_id;
            auto res = residency[tid];

            if (res == Residency::Host || res == Residency::Both) {
                // Already on host — redundant.
                remove[idx] = true;
                _num_eliminated++;
                EINSUMS_LOG_INFO("TransferElimination: removed redundant D2H for tensor id={}", tid);
            } else {
                residency[tid] = Residency::Both;
            }
        } else if (node.target == Target::GPU) {
            for (auto tid : node.outputs) {
                residency[tid] = Residency::Device;
                // GPU output is now on device.
                if (device_resident.find(tid) == device_resident.end()) {
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
    bool const modified = (_num_eliminated > 0 || !insertions.empty());

    if (!modified)
        return false;

    // Sort insertions by position (descending) so we can insert without shifting indices.
    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.before_idx > b.before_idx; });

    // First, mark removals and build filtered list.
    std::vector<Node> result;
    result.reserve(nodes.size() + insertions.size());

    // Build insertion map, then walk once.
    std::unordered_map<size_t, std::vector<size_t>> insert_before; // position → insertion indices
    for (size_t i = 0; i < insertions.size(); ++i) {
        insert_before[insertions[i].before_idx].push_back(i);
    }

    result.clear();
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        // Insert eviction nodes before this position.
        auto it = insert_before.find(idx);
        if (it != insert_before.end()) {
            for (auto ins_i : it->second) {
                result.push_back(std::move(insertions[ins_i].node));
            }
        }

        if (!remove[idx]) {
            result.push_back(std::move(nodes[idx]));
        }
    }

    nodes = std::move(result);

    // Update TensorHandle residency.
    for (auto const &[tid, res] : residency) {
        graph.tensor(tid).residency = res;
    }

    graph.mark_sorted();

    if (_num_eliminated > 0 || !insertions.empty()) {
        EINSUMS_LOG_INFO("TransferElimination: eliminated {} redundant transfers, inserted {} evictions", _num_eliminated,
                         insertions.size());
    }

    return true;
}

} // namespace einsums::compute_graph::passes
