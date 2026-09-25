//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

std::map<TensorId, std::vector<std::size_t>> value_writes_by_buffer(Graph const &graph) {
    std::map<TensorId, std::vector<std::size_t>> writes;
    auto const                                  &nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (is_lifecycle(nodes[i].kind)) {
            continue;
        }
        for (auto const out : nodes[i].outputs) {
            writes[graph.resolve_alias(out)].push_back(i);
        }
    }
    return writes;
}

bool einsum_is_uniform(Graph const &graph, Node const &node) {
    if (node.kind != OpKind::Einsum || node.inputs.size() < 2 || node.outputs.empty()) {
        return true;
    }
    auto const dtype_of = [&graph](TensorId id) {
        auto const *handle = graph.find_tensor(id);
        return handle != nullptr ? handle->dtype : packed_gemm::ScalarType::Unknown;
    };
    auto const a     = dtype_of(node.inputs[0]);
    auto const b     = dtype_of(node.inputs[1]);
    auto const c     = dtype_of(node.outputs[0]);
    auto const known = [](packed_gemm::ScalarType t) { return t != packed_gemm::ScalarType::Unknown; };
    return !(known(a) && known(b) && known(c)) || (a == b && a == c);
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
