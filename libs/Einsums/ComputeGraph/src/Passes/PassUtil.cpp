//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>

#include <algorithm>
#include <span>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

std::map<TensorId, std::vector<std::size_t>> value_writes_by_buffer(Graph const &graph) {
    std::map<TensorId, std::vector<std::size_t>> writes;
    auto const                                  &nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (is_lifecycle(nodes[i].kind)) {
            continue;
        }
        for (auto const out : nodes[i].outputs) {
            writes[graph.buffer_of(out)].push_back(i);
        }
    }
    return writes;
}

namespace {

/// Call @p visit with (buffer, is_write) for every effective access of @p node, reads first.
///
/// The one walk behind @ref buffer_io and @ref span_interferes. An ordinary node's own lists ARE
/// its effective lists, so only a control-flow node pays for the sub-graph walk and the copy;
/// the interference gate runs over every node of a span and must not allocate per node.
/// @p visit returns true to stop the walk, and the walk returns whether it was stopped.
template <typename F>
bool for_each_buffer_access(Graph &graph, Node const &node, F &&visit) {
    auto const walk = [&](std::span<TensorId const> ins, std::span<TensorId const> outs) {
        for (TensorId const tid : ins) {
            if (visit(graph.buffer_of(tid), false)) {
                return true;
            }
        }
        for (TensorId const tid : outs) {
            if (visit(graph.buffer_of(tid), true)) {
                return true;
            }
        }
        return false;
    };
    if (!is_control_flow(node.kind)) {
        return walk(node.inputs, node.outputs);
    }
    auto const [ins, outs] = graph.effective_io(node);
    return walk(ins, outs);
}

} // namespace

BufferIo buffer_io(Graph &graph, Node const &node) {
    BufferIo io;
    (void)for_each_buffer_access(graph, node, [&io](TensorId buffer, bool is_write) {
        (is_write ? io.writes : io.reads).insert(buffer);
        return false;
    });
    return io;
}

std::vector<BufferIo> buffer_io(Graph &graph) {
    std::vector<BufferIo> out;
    out.reserve(graph.nodes().size());
    for (Node const &node : graph.nodes()) {
        out.push_back(buffer_io(graph, node));
    }
    return out;
}

bool conflicts(BufferIo const &lhs, BufferIo const &rhs) {
    return std::ranges::any_of(rhs.reads, [&lhs](TensorId buffer) { return lhs.orders(buffer, false); }) ||
           std::ranges::any_of(rhs.writes, [&lhs](TensorId buffer) { return lhs.orders(buffer, true); });
}

bool span_interferes(Graph &graph, std::size_t first, std::size_t last, std::vector<bool> const &is_member,
                     std::unordered_set<TensorId> const &writes, std::unordered_set<TensorId> const &reads, bool reject_control_flow) {
    BufferIo group;
    for (TensorId const id : writes) {
        group.writes.insert(graph.buffer_of(id));
    }
    for (TensorId const id : reads) {
        group.reads.insert(graph.buffer_of(id));
    }
    for (std::size_t n = first + 1; n < last; ++n) {
        if (is_member[n]) {
            continue;
        }
        Node const &other = graph.nodes()[n];
        if (reject_control_flow && is_control_flow(other.kind)) {
            return true;
        }
        if (for_each_buffer_access(graph, other, [&group](TensorId buffer, bool is_write) { return group.orders(buffer, is_write); })) {
            return true;
        }
    }
    return false;
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
