//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/TransferInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/TransferNode.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// True when @p tid's incoming value is dead at @p node: the node names it as a
/// destination and overwrites it without reading it, so there is nothing to
/// transfer in.
///
/// The overwrite question is @ref passes::pure_overwrite, which reads the LIVE
/// prefactor a pass may have folded rather than the at-capture snapshot, and
/// which knows the kinds beyond Einsum and Permute that carry one. The extra
/// kinds it admits (Axpby, BatchedGemm, TiledEinsum at a zero destination
/// prefactor) change nothing here: a node that does not list @p tid among its
/// outputs is rejected before the question is asked.
bool is_dead_input(Node const &node, TensorId tid) {
    return std::ranges::find(node.outputs, tid) != node.outputs.end() && pure_overwrite(node);
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
    return make_transfer_node(OpKind::HostToDevice, handle, "H2D", handle.total_bytes());
}

/// Create a DeviceToHost transfer node for a tensor.
Node make_d2h_node(TensorHandle const &handle) {
    return make_transfer_node(OpKind::DeviceToHost, handle, "D2H", handle.total_bytes());
}

} // namespace

void TransferInsertion::reset_stats() {
    _num_transfers = 0;
}

bool TransferInsertion::transform(Graph &graph) {
    PassCounter const transfers{_num_transfers};
    graph.topological_sort();

    auto const &nodes = graph.nodes();

    // Track residency per tensor (start from TensorHandle state).
    std::unordered_map<TensorId, Residency> residency;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        residency[tid] = handle.residency;
    }

    // The transfers each position gains, in the order they run there: the downloads after the node
    // before it, then the uploads before the node at it.
    std::map<std::size_t, std::vector<Node>> inserted;

    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        auto const &node = nodes[idx];

        if (node.target == Target::GPU) {
            // A device node this pass does not understand has no sound way to be declined: left
            // unbracketed it would run on whatever the device holds. It can only come from a
            // placement pass that understood a feature this one does not, so say so.
            if (!understands(graph, node)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error,
                                        "TransferInsertion: device node '{}' carries a feature this pass does not understand ({})",
                                        node.label, describe_features(features_of(graph, node)));
            }

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
                    inserted[idx].push_back(make_h2d_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: H2D for tensor '{}' (id={}) before {} node {}", handle.name, tid, node.kind,
                                     node.id);
                }
            }

            // GPU outputs are now on device.
            for (auto tid : node.outputs) {
                residency[tid] = Residency::Device;
            }

            // Insert D2H for each output that a later CPU node reads.
            for (auto tid : node.outputs) {
                if (cpu_node_reads_later(nodes, idx + 1, tid)) {
                    auto &handle = graph.tensor(tid);
                    inserted[idx + 1].push_back(make_d2h_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: D2H for tensor '{}' (id={}) after {} node", handle.name, tid, node.kind);
                }
            }
        } else if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            // A transfer an earlier run inserted copies one side onto the other, so both copies
            // are valid afterwards. It is not a host write, and reading it as one would make a
            // second run upload what the first already uploaded.
            for (TensorId const tid : node.outputs) {
                residency[tid] = Residency::Both;
            }
            for (TensorId const tid : node.inputs) {
                residency[tid] = Residency::Both;
            }
        } else {
            // A CPU node that writes a tensor leaves the device copy stale, so it is on the host
            // alone again: a later GPU reader needs an upload, and the final download must not copy
            // the stale device value over what the host wrote. By buffer, since a write through a
            // view or a redirected slot lands in the same storage, and through a control-flow
            // node's body, whose writes its own output list does not name.
            for (TensorId const out : graph.effective_io(node).second) {
                TensorId const buffer = graph.buffer_of(out);
                for (auto &[tid, res] : residency) {
                    if (graph.buffer_of(tid) == buffer) {
                        res = Residency::Host;
                    }
                }
            }
        }
    }

    // Final D2H: for user-visible tensors that ended up on device with no
    // later CPU consumer, insert a D2H at the end so the user can read the
    // result after execute(). Only needed when GPU nodes actually exist.
    if (transfers.moved()) {
        for (auto const &[tid, res] : residency) {
            if (res == Residency::Device) {
                auto &handle = graph.tensor(tid);
                if (!handle.is_intermediate) {
                    inserted[nodes.size()].push_back(make_d2h_node(handle));
                    _num_transfers++;
                    residency[tid] = Residency::Both;

                    EINSUMS_LOG_INFO("TransferInsertion: final D2H for user-visible tensor '{}' (id={})", handle.name, tid);
                }
            }
        }
    }

    if (!transfers.moved())
        return false;

    // Only inserted between existing nodes, so the order stays topological.
    std::vector<std::pair<std::size_t, std::vector<Node>>> groups;
    for (auto &[at, transfers_here] : inserted) {
        groups.emplace_back(at, std::move(transfers_here));
    }
    graph.insert_node_groups(std::move(groups));

    // Update TensorHandle residency to match final state.
    for (auto const &[tid, res] : residency) {
        graph.tensor(tid).residency = res;
    }

    EINSUMS_LOG_INFO("TransferInsertion: inserted {} transfer nodes", _num_transfers);
    if (transfers.moved()) {
        report(1, fmt::format("inserted {} host/device transfer node(s)", _num_transfers));
    }
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
