//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Logging.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace einsums::compute_graph::passes {

namespace {

/// Soundness context for an inference run on one graph. A symmetry tag is
/// only ever a *guarantee* about the tensor's final contents, so we infer
/// it only when nothing can invalidate it after the producing op:
///   - the tensor has exactly one writer in this graph (no later overwrite
///     could destroy the inferred structure), and
///   - the tensor isn't referenced by a child sub-graph (a nested loop /
///     conditional body could write it without this graph's node list
///     showing it — a Loop node doesn't list its body's writes).
/// Without these guards a stale tag could claim symmetry the data no longer
/// has. They make the pass strictly conservative and therefore safe to
/// recurse into loop bodies.
struct InferGuard {
    std::unordered_map<TensorId, int> writer_count;
    std::unordered_set<void const *>  subtree_ptrs;

    [[nodiscard]] bool safe(Graph const &graph, TensorId tid) const {
        if (auto it = writer_count.find(tid); it == writer_count.end() || it->second != 1) {
            return false;
        }
        auto it = graph.tensors_map().find(tid);
        if (it == graph.tensors_map().end()) {
            return false;
        }
        return it->second.tensor_ptr == nullptr || subtree_ptrs.count(it->second.tensor_ptr) == 0;
    }
};

/// Try to push an inferred descriptor to a graph-owned tensor handle.
/// Returns true if the descriptor was applied (either taking on a new value
/// or overwriting ``nullptr`` on the backing tensor).
bool apply_inferred(Graph &graph, TensorId out_tid, SymmetryDescriptor desc, InferGuard const &guard) {
    auto &handle = graph.tensor(out_tid);
    if (!handle.is_intermediate)
        return false; // Never mutate user-owned tensor state.
    if (!guard.safe(graph, out_tid))
        return false; // Could be overwritten later / by a child body — don't tag.
    // Skip if the tensor already has an equivalent descriptor — avoids
    // redundant work and spurious "new inference" reports on re-runs.
    if (handle.symmetry_hint && *handle.symmetry_hint == desc)
        return false;

    handle.symmetry_hint = std::make_shared<SymmetryDescriptor>(desc);
    if (handle.set_symmetry_fn)
        handle.set_symmetry_fn(std::move(desc));
    return true;
}

/// Rule: ``C = α·A`` — Scale preserves its input's symmetry. Detects
/// ``OpKind::Scale`` nodes and copies the descriptor from input to output.
bool propagate_scale(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Scale)
        return false;
    if (node.inputs.size() != 1 || node.outputs.size() != 1)
        return false;

    auto const &in = graph.tensor(node.inputs[0]);
    if (!in.symmetry_hint)
        return false;
    return apply_inferred(graph, node.outputs[0], *in.symmetry_hint, guard);
}

/// Rule: ``C = α·A + β·B`` (or simple sum) — if A and B carry identical
/// descriptors, C inherits them. Applies to OpKind::Axpy and
/// OpKind::Axpby when the descriptors match exactly.
bool propagate_linear_combination(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Axpy && node.kind != OpKind::Axpby)
        return false;
    if (node.inputs.size() < 2 || node.outputs.size() != 1)
        return false;

    auto const &a = graph.tensor(node.inputs[0]);
    auto const &b = graph.tensor(node.inputs[1]);
    if (!a.symmetry_hint || !b.symmetry_hint)
        return false;
    if (!(*a.symmetry_hint == *b.symmetry_hint))
        return false;

    return apply_inferred(graph, node.outputs[0], *a.symmetry_hint, guard);
}

/// Rule: ``C = AᵀA`` / ``AAᵀ`` / ``AᴴA`` via einsum — when the same tensor
/// feeds both operand slots of a rank-2 contraction with exactly one link
/// index, the output has structure that depends on the element type and
/// conjugation flags:
///
/// - Real + no conj / both conj: output is **symmetric**.
/// - Complex + exactly one operand conjugated (``AᴴA`` or ``AAᴴ``): output
///   is **Hermitian**.
/// - Complex + no conjugation (bare ``A·A``): output is **symmetric** (not
///   Hermitian in general, but symmetric because (A·A)ᵀ = Aᵀ·Aᵀ = A·A when
///   self-contracting).
///
/// We only need the scalar type from the EinsumDescriptor to pick between
/// the symmetric and Hermitian output tag; that information is on the
/// underlying tensor's descriptor, which we read from the handle.
bool propagate_self_contraction(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Einsum)
        return false;
    if (node.inputs.size() != 2 || node.outputs.size() != 1)
        return false;
    if (node.inputs[0] != node.inputs[1])
        return false;

    auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
    if (!desc)
        return false;

    // Classic rank-2 × rank-2 → rank-2 with exactly one link index.
    if (desc->spec.a_indices.size() != 2 || desc->spec.b_indices.size() != 2 || desc->spec.c_indices.size() != 2)
        return false;
    if (desc->spec.link_indices.size() != 1)
        return false;

    auto const &in = graph.tensor(node.inputs[0]);

    bool const is_complex = in.dtype == packed_gemm::ScalarType::Complex64 || in.dtype == packed_gemm::ScalarType::Complex128;
    bool const one_conj   = desc->conj_a ^ desc->conj_b; // XOR — exactly one side conjugated

    if (is_complex && one_conj)
        return apply_inferred(graph, node.outputs[0], SymmetryDescriptor::hermitian_pair(0, 1), guard);
    return apply_inferred(graph, node.outputs[0], SymmetryDescriptor::symmetric_pair(0, 1), guard);
}

/// Rule: permute of a rank-2 symmetric/Hermitian tensor stays
/// symmetric/Hermitian. Covers both the identity permute (trivially
/// preserving) and the swap permute (``{j,i}`` — for a symmetric tensor the
/// output equals the input). Beta must be zero (pure overwrite); otherwise
/// we can't reason about what C was before the add.
bool propagate_permute(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Permute && node.kind != OpKind::Transpose)
        return false;
    if (node.inputs.size() != 1 || node.outputs.size() != 1)
        return false;

    auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
    if (!desc)
        return false;
    if (desc->beta != 0.0)
        return false;
    // Rank-2 only (higher-rank symmetry descriptors can describe cross-
    // axis invariants that don't trivially carry through every permute).
    if (desc->a_indices.size() != 2 || desc->c_indices.size() != 2)
        return false;

    auto const &in = graph.tensor(node.inputs[0]);
    if (!in.symmetry_hint)
        return false;
    // Only propagate rank-2 symmetric / Hermitian pair descriptors.
    auto const &ops = in.symmetry_hint->ops;
    if (ops.size() != 1)
        return false;
    auto const &op = ops[0];
    if (op.permutation[0] != 1 || op.permutation[1] != 0 || op.sign != +1)
        return false;

    return apply_inferred(graph, node.outputs[0], *in.symmetry_hint, guard);
}

} // namespace

bool SymmetryPropagation::run(Graph &graph) {
    graph.topological_sort();

    _num_inferred     = 0;
    auto const &nodes = graph.nodes();

    // Build the soundness guard for this graph: writer counts + the set of
    // tensor pointers referenced by child sub-graphs. Only single-writer,
    // not-used-by-a-child tensors get tagged.
    InferGuard guard;
    for (auto const &node : nodes) {
        // Lifecycle nodes (Alloc/Free/Materialize/Initialize) list the tensor
        // as an output but don't write a *value* that could invalidate an
        // inferred symmetry — a freshly created/zeroed tensor is then filled
        // by exactly one real op. Count only value-producing nodes.
        if (node.kind == OpKind::Alloc || node.kind == OpKind::Free || node.kind == OpKind::Materialize ||
            node.kind == OpKind::Initialize) {
            continue;
        }
        for (auto tid : node.outputs) {
            guard.writer_count[tid]++;
        }
    }
    graph.collect_subtree_referenced_ptrs(guard.subtree_ptrs);

    for (auto const &node : nodes) {
        if (propagate_scale(graph, node, guard))
            ++_num_inferred;
        if (propagate_linear_combination(graph, node, guard))
            ++_num_inferred;
        if (propagate_self_contraction(graph, node, guard))
            ++_num_inferred;
        if (propagate_permute(graph, node, guard))
            ++_num_inferred;
    }

    if (_num_inferred > 0)
        EINSUMS_LOG_INFO("SymmetryPropagation: inferred symmetry on {} tensor(s)", _num_inferred);

    // Analysis pass — never changes the node list.
    return false;
}

} // namespace einsums::compute_graph::passes
