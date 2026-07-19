//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

bool is_lifecycle(OpKind kind) {
    return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
}

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
/// match: including the strided flag and its per-operand batch strides,
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
    // monostate or unknown, only equal if both monostate
    return std::holds_alternative<std::monostate>(a) && std::holds_alternative<std::monostate>(b);
}

/// Whether a node may participate in CSE at all.
///
/// CSE eliminates a duplicate node by redirecting readers of its output onto
/// another node's output. That is only valid for a *pure overwrite* producer:
/// the output must be a function of the inputs alone, never read back. Ops that
/// accumulate into or otherwise read their destination, axpby/axpy/scale/
/// element-transform (monostate op_data here), or an einsum/permute/batched
/// gemm with a nonzero destination prefactor, are excluded. Two such nodes
/// writing different buffers are not the same computation (they read different
/// destinations), and their scalar coefficients may not even be represented in
/// op_data, so op_data_equal cannot tell them apart.
bool cse_eligible(Node const &nd) {
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return is_zero(e->c_prefactor);
    }
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->beta == 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta == 0.0;
    }
    return false;
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

    // Resolve a TensorId in this graph to its underlying buffer pointer
    // (stable identity for a tensor; null when unresolved).
    auto ptr_of = [&](TensorId tid) -> void const * {
        auto it = graph.tensors_map().find(tid);
        return (it != graph.tensors_map().end()) ? it->second.tensor_ptr : nullptr;
    };

    // Count real (non-lifecycle) writers of each buffer across the graph.
    // CSE eliminates node j by redirecting readers of its output to node i's
    // output. That is only sound when the surviving buffer holds a *stable*
    // value: if anything writes node i's output again (e.g. a later in-place
    // scale), the redirected readers would observe the mutated value instead
    // of the common subexpression. So every output buffer involved in a merge
    // must have exactly one writer, the producing node itself.
    std::unordered_map<void const *, int> writer_count;
    for (auto const &nd : nodes) {
        if (is_lifecycle(nd.kind))
            continue;
        for (auto out : nd.outputs) {
            if (auto const *p = ptr_of(out))
                writer_count[p]++;
        }
    }

    for (size_t i = 0; i < nodes.size(); i++) {
        if (remove[i])
            continue;

        // Only pure-overwrite producers may be a CSE survivor/candidate. (Since
        // a matched pair must have equal op_data, checking node i covers j.)
        if (!cse_eligible(nodes[i]))
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

            // Guard C: the duplicate's outputs must be graph-owned
            // intermediates. A user-visible output is a contract - the user
            // reads that tensor directly, not through an executor slot, so
            // eliding its producer leaves it unwritten no matter how graph
            // consumers are redirected. (Folding such duplicates behind an
            // inserted copy node is possible future work.)
            bool duplicate_user_visible = false;
            for (auto out : nodes[j].outputs) {
                auto it = graph.tensors_map().find(out);
                if (it == graph.tensors_map().end() || !it->second.is_intermediate || it->second.aliases != 0) {
                    duplicate_user_visible = true;
                    break;
                }
            }
            if (duplicate_user_visible)
                continue;

            // Guard B: both producers' output buffers must be written exactly
            // once (by themselves). Otherwise redirecting readers onto a buffer
            // that gets mutated again would hand them the wrong value.
            bool single_writer = true;
            for (auto out : nodes[i].outputs) {
                auto const *p = ptr_of(out);
                if (p == nullptr || writer_count[p] != 1) {
                    single_writer = false;
                    break;
                }
            }
            if (single_writer) {
                for (auto out : nodes[j].outputs) {
                    auto const *p = ptr_of(out);
                    if (p == nullptr || writer_count[p] != 1) {
                        single_writer = false;
                        break;
                    }
                }
            }
            if (!single_writer)
                continue;

            // Guard A: the shared inputs must not be overwritten between i and
            // j. If some intervening node writes one of node i's inputs, then
            // node i (at its position) and node j (at its position) actually
            // read different values, so they are not the *same* computation at
            // runtime and the merge would reuse a stale result.
            std::unordered_set<void const *> input_ptrs;
            for (auto in : nodes[i].inputs) {
                if (auto const *p = ptr_of(in))
                    input_ptrs.insert(p);
            }
            bool inputs_stable = true;
            for (size_t k = i + 1; k < j && inputs_stable; k++) {
                if (remove[k] || is_lifecycle(nodes[k].kind))
                    continue;
                for (auto out : nodes[k].outputs) {
                    auto const *p = ptr_of(out);
                    if (p != nullptr && input_ptrs.count(p) > 0) {
                        inputs_stable = false;
                        break;
                    }
                }
            }
            if (!inputs_stable)
                continue;

            // Equivalent! Redirect j's outputs to i's outputs.
            //
            // Two redirects are needed, for two different consumers:
            //   1. tensor_redirect + the Node::inputs rewrite below, keeps the
            //      TensorId metadata correct so liveness-based passes
            //      (MemoryPlanning, FreeInsertion) see node i's output as the
            //      live buffer and node j's as dead.
            //   2. Graph::redirect_slot: repoints node j's output slot at node
            //      i's buffer so any *already-baked* executor lambda that
            //      captured j's slot reads i's result at execute time. Without
            //      this the metadata rewrite is invisible at runtime and a
            //      surviving consumer of node j's output reads a never-written
            //      buffer (silent wrong results). Node i is a survivor (never
            //      itself removed in this pass), so its slot ptr is stable and
            //      copying it here is durable across re-execution.
            for (size_t k = 0; k < nodes[i].outputs.size(); k++) {
                tensor_redirect[nodes[j].outputs[k]] = nodes[i].outputs[k];
                graph.redirect_slot(nodes[j].outputs[k], nodes[i].outputs[k]);
            }

            remove[j] = true;
            modified  = true;

            EINSUMS_LOG_INFO("CSE: eliminated node {} (duplicate of node {})", nodes[j].id, nodes[i].id);
            report(2, fmt::format("eliminate node {} '{}' — duplicate of node {}", nodes[j].id, nodes[j].label, nodes[i].id));
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
