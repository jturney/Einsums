//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/TransferInsertion.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Check if a tensor's input value is dead, it appears in both inputs and outputs
/// of the node, but the operation semantics overwrite it completely.
/// Currently recognized: Einsum with c_prefactor == 0.0, Scale with factor == 0.0.
bool is_dead_input(Node const &node, TensorId tid) {
    // Must be in both inputs and outputs.
    bool in_outputs = false;
    for (auto out_tid : node.outputs) {
        if (out_tid == tid) {
            in_outputs = true;
            break;
        }
    }
    if (!in_outputs)
        return false;

    // Check operation semantics.
    if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
        // C = c_pf * C + ab_pf * A * B. When c_pf == 0, old C is not read.
        return is_zero(desc->c_prefactor);
    }
    if (auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data)) {
        // D = beta * D + alpha * permute(S). When beta == 0, old D is not read.
        return desc->beta == 0.0;
    }

    return false;
}

/// Check if any node after position `start` reads tensor_id and targets CPU.
bool cpu_node_reads_later(std::vector<Node> const &nodes, size_t start, TensorId tensor_id) {
    for (size_t i = start; i < nodes.size(); ++i) {
        if (nodes[i].target != Target::CPU)
            continue;
        auto const &ins = nodes[i].inputs;
        if (std::ranges::find(ins, tensor_id) != ins.end())
            return true;
    }
    return false;
}

/// Create a HostToDevice transfer node for a tensor.
Node make_h2d_node(TensorHandle const &handle) {
    Node n;
    n.kind   = OpKind::HostToDevice;
    n.target = Target::CPU; // transfer itself is a host-initiated operation
    n.label  = fmt::format("H2D({})", handle.name);

    TransferDescriptor desc;
    desc.tensor_id  = handle.id;
    desc.size_bytes = handle.total_bytes();
    n.op_data       = desc;

    n.inputs  = {handle.id};
    n.outputs = {handle.id};

    n.estimated_bytes = desc.size_bytes;

    // Executor: synchronize and transfer.
    // On mock backend: no separate device memory exists, so this is a synchronization
    // point only. On real GPU: the graph executor will replace this lambda with one
    // that calls gpu::memcpy_host_to_device(device_shadow.data(), host_tensor.data(), bytes)
    // once device shadow allocations are implemented.
    n.execute = []() { gpu::device_synchronize(); };

    return n;
}

/// Create a DeviceToHost transfer node for a tensor.
Node make_d2h_node(TensorHandle const &handle) {
    Node n;
    n.kind   = OpKind::DeviceToHost;
    n.target = Target::CPU;
    n.label  = fmt::format("D2H({})", handle.name);

    TransferDescriptor desc;
    desc.tensor_id  = handle.id;
    desc.size_bytes = handle.total_bytes();
    n.op_data       = desc;

    n.inputs  = {handle.id};
    n.outputs = {handle.id};

    n.estimated_bytes = desc.size_bytes;

    // See make_h2d_node comment, same applies for D2H direction.
    n.execute = []() { gpu::device_synchronize(); };

    return n;
}

} // namespace

bool TransferInsertion::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes    = graph.nodes();
    _num_transfers = 0;

    // Track residency per tensor (start from TensorHandle state).
    std::unordered_map<TensorId, Residency> residency;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        residency[tid] = handle.residency;
    }

    // We'll build a new node list with transfers inserted.
    std::vector<Node> new_nodes;
    new_nodes.reserve(nodes.size() * 2); // generous estimate

    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        auto &node = nodes[idx];

        if (node.target == Target::GPU) {
            // Insert H2D for each input not yet on device.
            for (auto tid : node.inputs) {
                // Skip H2D for dead inputs, the operation will overwrite this tensor
                // completely (e.g., einsum with c_prefactor=0), so the initial value
                // doesn't need to be on device.
                if (is_dead_input(node, tid))
                    continue;

                auto res = residency[tid];
                if (res != Residency::Device && res != Residency::Both) {
                    auto &handle = graph.tensor(tid);
                    new_nodes.push_back(make_h2d_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: H2D for tensor '{}' (id={}) before {} node {}", handle.name, tid,
                                     op_kind_name(node.kind), node.id);
                }
            }

            // The GPU node itself.
            new_nodes.push_back(std::move(node));

            // GPU outputs are now on device.
            for (auto tid : new_nodes.back().outputs) {
                residency[tid] = Residency::Device;
            }

            // Insert D2H for each output that a later CPU node reads.
            for (auto tid : new_nodes.back().outputs) {
                if (cpu_node_reads_later(nodes, idx + 1, tid)) {
                    auto &handle = graph.tensor(tid);
                    new_nodes.push_back(make_d2h_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: D2H for tensor '{}' (id={}) after {} node", handle.name, tid,
                                     op_kind_name(new_nodes[new_nodes.size() - 2].kind));
                }
            }
        } else {
            // CPU node, just pass through.
            new_nodes.push_back(std::move(node));
        }
    }

    // Final D2H: for user-visible tensors that ended up on device with no
    // later CPU consumer, insert a D2H at the end so the user can read the
    // result after execute(). Only needed when GPU nodes actually exist.
    if (_num_transfers > 0) {
        for (auto const &[tid, res] : residency) {
            if (res == Residency::Device) {
                auto &handle = graph.tensor(tid);
                if (!handle.is_intermediate) {
                    new_nodes.push_back(make_d2h_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: final D2H for user-visible tensor '{}' (id={})", handle.name, tid);
                }
            }
        }
    }

    // Always assign new_nodes back, we moved all nodes into it during the loop.
    nodes = std::move(new_nodes);

    if (_num_transfers == 0)
        return false;

    // Update TensorHandle residency to match final state.
    for (auto const &[tid, res] : residency) {
        graph.tensor(tid).residency = res;
    }

    // The new node list is in topological order (we only inserted between existing nodes).
    graph.mark_sorted();

    EINSUMS_LOG_INFO("TransferInsertion: inserted {} transfer nodes", _num_transfers);
    if (_num_transfers > 0) {
        report(1, fmt::format("inserted {} host/device transfer node(s)", _num_transfers));
    }
    return true;
}

} // namespace einsums::compute_graph::passes
