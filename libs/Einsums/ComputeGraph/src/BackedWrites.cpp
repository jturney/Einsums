//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/BackedWrites.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {
bool moves_storage(OpKind kind) {
    return kind == OpKind::DiskRead || kind == OpKind::DiskWrite;
}
} // namespace

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
bool written_in_memory(Graph const &graph, void const *object, Graph const *skip_subtree, Node const *skip_node) {
    if (&graph == skip_subtree) {
        return false;
    }
    for (auto const &node : graph.nodes()) {
        if (&node != skip_node && !is_lifecycle(node.kind) && node.kind != OpKind::View && !moves_storage(node.kind)) {
            for (TensorId const out : node.outputs) {
                TensorHandle const *handle = graph.find_tensor(graph.buffer_of(out));
                if (handle != nullptr && handle->tensor_ptr == object) {
                    return true;
                }
            }
        }
        bool found = false;
        for_each_child_graph(node, [&](Graph const &child) { found = found || written_in_memory(child, object, skip_subtree, skip_node); });
        if (found) {
            return true;
        }
    }
    return false;
}

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
bool written_to_storage(Graph const &graph, TensorBacking const &backing) {
    std::string const path = backing.describe();
    for (auto const &node : graph.nodes()) {
        if (node.kind == OpKind::DiskWrite) {
            if (auto const *io = node.op_data.get_if<DiskIODescriptor>(); io != nullptr && io->file_path == path) {
                return true;
            }
        }
        bool found = false;
        for_each_child_graph(node, [&](Graph const &child) { found = found || written_to_storage(child, backing); });
        if (found) {
            return true;
        }
    }
    return false;
}

DiskIODescriptor backing_io(TensorBacking const &backing, TensorId tensor, std::size_t bytes) {
    DiskIODescriptor io;
    io.file_path  = backing.describe();
    io.tensor_id  = tensor;
    io.size_bytes = bytes;
    return io;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
