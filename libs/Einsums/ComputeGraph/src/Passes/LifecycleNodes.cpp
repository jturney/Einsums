//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "LifecycleNodes.hpp"

#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>

#include <fmt/format.h>

#include <algorithm>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::lifecycle)

DeclaredTensor declared_tensor(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid) {
    if (&owner == &graph && handle.id == emit_tid) {
        return {};
    }
    return DeclaredTensor{.graph = owner.anchor(), .id = handle.id};
}

namespace {

AllocDescriptor describe(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid) {
    return AllocDescriptor{.tensor_id   = emit_tid,
                           .size_bytes  = handle.total_bytes(),
                           .tensor_name = handle.name,
                           .declared    = declared_tensor(graph, owner, handle, emit_tid)};
}

} // namespace

Node make_materialize_node(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid) {
    Node node;
    node.kind            = OpKind::Materialize;
    node.label           = fmt::format("materialize({})", handle.name);
    node.outputs         = {emit_tid}; // WAW edge orders it before the first writer
    node.op_data         = describe(graph, owner, handle, emit_tid);
    node.estimated_bytes = handle.total_bytes();
    node.execute         = build_executor(node.kind, handle.dtype, handle.rank, node.op_data, graph, node.inputs, node.outputs);
    return node;
}

Node make_free_node(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid) {
    Node node;
    node.kind            = OpKind::Free;
    node.label           = fmt::format("free({})", handle.name);
    node.inputs          = {emit_tid};
    node.outputs         = {emit_tid};
    node.op_data         = describe(graph, owner, handle, emit_tid);
    node.estimated_bytes = handle.total_bytes();
    node.execute         = build_executor(node.kind, handle.dtype, handle.rank, node.op_data, graph, node.inputs, node.outputs);
    return node;
}

std::string_view lifecycle_tensor_name(Node const &node) noexcept {
    if (node.kind != OpKind::Alloc && node.kind != OpKind::Materialize && node.kind != OpKind::Free) {
        return {};
    }
    auto const *desc = node.op_data.get_if<AllocDescriptor>();
    return desc != nullptr ? std::string_view{desc->tensor_name} : std::string_view{};
}

bool has_lifecycle_node(std::span<Node const> nodes, OpKind kind, std::string_view name) noexcept {
    return std::ranges::any_of(nodes, [kind, name](Node const &node) { return node.kind == kind && lifecycle_tensor_name(node) == name; });
}

EINSUMS_NAMESPACE_END(compute_graph::passes::lifecycle)
