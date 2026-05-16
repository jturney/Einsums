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
#include <variant>

namespace einsums::compute_graph::passes {

namespace {

/// Try to push an inferred descriptor to a graph-owned tensor handle.
/// Returns true if the descriptor was applied (either taking on a new value
/// or overwriting ``nullptr`` on the backing tensor).
bool apply_inferred(TensorHandle &handle, SymmetryDescriptor desc) {
    if (!handle.is_intermediate)
        return false; // Never mutate user-owned tensor state.
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
bool propagate_scale(Graph &graph, Node const &node) {
    if (node.kind != OpKind::Scale)
        return false;
    if (node.inputs.size() != 1 || node.outputs.size() != 1)
        return false;

    auto const &in  = graph.tensor(node.inputs[0]);
    auto       &out = graph.tensor(node.outputs[0]);
    if (!in.symmetry_hint)
        return false;
    return apply_inferred(out, *in.symmetry_hint);
}

/// Rule: ``C = α·A + β·B`` (or simple sum) — if A and B carry identical
/// descriptors, C inherits them. Applies to OpKind::Axpy and
/// OpKind::Axpby when the descriptors match exactly.
bool propagate_linear_combination(Graph &graph, Node const &node) {
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

    auto &out = graph.tensor(node.outputs[0]);
    return apply_inferred(out, *a.symmetry_hint);
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
bool propagate_self_contraction(Graph &graph, Node const &node) {
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

    auto       &out = graph.tensor(node.outputs[0]);
    auto const &in  = graph.tensor(node.inputs[0]);

    bool const is_complex = in.dtype == packed_gemm::ScalarType::Complex64 || in.dtype == packed_gemm::ScalarType::Complex128;
    bool const one_conj   = desc->conj_a ^ desc->conj_b; // XOR — exactly one side conjugated

    if (is_complex && one_conj)
        return apply_inferred(out, SymmetryDescriptor::hermitian_pair(0, 1));
    return apply_inferred(out, SymmetryDescriptor::symmetric_pair(0, 1));
}

/// Rule: permute of a rank-2 symmetric/Hermitian tensor stays
/// symmetric/Hermitian. Covers both the identity permute (trivially
/// preserving) and the swap permute (``{j,i}`` — for a symmetric tensor the
/// output equals the input). Beta must be zero (pure overwrite); otherwise
/// we can't reason about what C was before the add.
bool propagate_permute(Graph &graph, Node const &node) {
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

    auto &out = graph.tensor(node.outputs[0]);
    return apply_inferred(out, *in.symmetry_hint);
}

} // namespace

bool SymmetryPropagation::run(Graph &graph) {
    graph.topological_sort();

    _num_inferred     = 0;
    auto const &nodes = graph.nodes();

    for (auto const &node : nodes) {
        if (propagate_scale(graph, node))
            ++_num_inferred;
        if (propagate_linear_combination(graph, node))
            ++_num_inferred;
        if (propagate_self_contraction(graph, node))
            ++_num_inferred;
        if (propagate_permute(graph, node))
            ++_num_inferred;
    }

    if (_num_inferred > 0)
        EINSUMS_LOG_INFO("SymmetryPropagation: inferred symmetry on {} tensor(s)", _num_inferred);

    // Analysis pass — never changes the node list.
    return false;
}

} // namespace einsums::compute_graph::passes
