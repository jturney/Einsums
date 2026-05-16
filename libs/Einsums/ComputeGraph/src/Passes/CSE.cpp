//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Check if two EinsumDescriptors are equivalent.
bool einsum_desc_equal(EinsumDescriptor const &a, EinsumDescriptor const &b) {
    return a.spec == b.spec && a.c_prefactor == b.c_prefactor && a.ab_prefactor == b.ab_prefactor && a.conj_a == b.conj_a &&
           a.conj_b == b.conj_b;
}

/// Check if two ScaleDescriptors are equivalent.
bool scale_desc_equal(ScaleDescriptor const &a, ScaleDescriptor const &b) {
    return a.factor == b.factor;
}

/// Check if two PermuteDescriptors are equivalent.
bool permute_desc_equal(PermuteDescriptor const &a, PermuteDescriptor const &b) {
    return a.alpha == b.alpha && a.beta == b.beta;
}

/// Check if two BatchedGemmDescriptors are equivalent. All fields must
/// match — including the strided flag and its per-operand batch strides,
/// since pointer-array and strided batches have different executor
/// semantics even when the BLAS key looks identical.
bool batched_gemm_desc_equal(BatchedGemmDescriptor const &a, BatchedGemmDescriptor const &b) {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc && a.trans_a == b.trans_a &&
           a.trans_b == b.trans_b && a.alpha == b.alpha && a.beta == b.beta && a.batch_count == b.batch_count && a.scalar == b.scalar &&
           a.strided == b.strided && a.batch_stride_a == b.batch_stride_a && a.batch_stride_b == b.batch_stride_b &&
           a.batch_stride_c == b.batch_stride_c;
}

/// Check if two OpData variants are equivalent.
bool op_data_equal(OpData const &a, OpData const &b) {
    if (a.index() != b.index())
        return false;

    if (auto *ea = std::get_if<EinsumDescriptor>(&a)) {
        return einsum_desc_equal(*ea, std::get<EinsumDescriptor>(b));
    }
    if (auto *sa = std::get_if<ScaleDescriptor>(&a)) {
        return scale_desc_equal(*sa, std::get<ScaleDescriptor>(b));
    }
    if (auto *pa = std::get_if<PermuteDescriptor>(&a)) {
        return permute_desc_equal(*pa, std::get<PermuteDescriptor>(b));
    }
    if (auto *ba = std::get_if<BatchedGemmDescriptor>(&a)) {
        return batched_gemm_desc_equal(*ba, std::get<BatchedGemmDescriptor>(b));
    }
    // monostate or unknown — only equal if both monostate
    return std::holds_alternative<std::monostate>(a) && std::holds_alternative<std::monostate>(b);
}

/// Check if two nodes compute the same thing.
bool nodes_equivalent(Node const &a, Node const &b) {
    if (a.kind != b.kind)
        return false;
    if (a.inputs != b.inputs)
        return false;
    if (a.outputs.size() != b.outputs.size())
        return false;
    return op_data_equal(a.op_data, b.op_data);
}

} // namespace

bool CSE::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    bool              modified = false;
    std::vector<bool> remove(nodes.size(), false);

    // For each pair, check if they compute the same expression.
    // Build a remapping of tensor IDs for eliminated nodes.
    std::unordered_map<TensorId, TensorId> tensor_redirect;

    for (size_t i = 0; i < nodes.size(); i++) {
        if (remove[i])
            continue;

        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (remove[j])
                continue;

            // Apply any existing redirections to the candidate's inputs
            // before comparing.
            auto redirected_inputs_j = nodes[j].inputs;
            for (auto &tid : redirected_inputs_j) {
                auto it = tensor_redirect.find(tid);
                if (it != tensor_redirect.end()) {
                    tid = it->second;
                }
            }

            // Check equivalence with redirected inputs
            if (nodes[i].kind != nodes[j].kind)
                continue;
            if (nodes[i].inputs != redirected_inputs_j)
                continue;
            if (nodes[i].outputs.size() != nodes[j].outputs.size())
                continue;
            if (!op_data_equal(nodes[i].op_data, nodes[j].op_data))
                continue;

            // Equivalent! Redirect j's outputs to i's outputs.
            for (size_t k = 0; k < nodes[i].outputs.size(); k++) {
                tensor_redirect[nodes[j].outputs[k]] = nodes[i].outputs[k];
            }

            remove[j] = true;
            modified  = true;

            EINSUMS_LOG_INFO("CSE: eliminated node {} (duplicate of node {})", nodes[j].id, nodes[i].id);
        }
    }

    if (!modified)
        return false;

    // Apply redirections to all remaining nodes' inputs
    for (size_t i = 0; i < nodes.size(); i++) {
        if (remove[i])
            continue;
        for (auto &tid : nodes[i].inputs) {
            auto it = tensor_redirect.find(tid);
            if (it != tensor_redirect.end()) {
                tid = it->second;
            }
        }
    }

    // Remove eliminated nodes
    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!remove[i]) {
            filtered.push_back(std::move(nodes[i]));
        }
    }
    nodes = std::move(filtered);
    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
