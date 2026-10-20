//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/SymmetryRules.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <optional>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Rule: ``C = α·A``, Scale preserves its input's symmetry.
std::optional<SymmetryDescriptor> scale_symmetry(Graph const &graph, Node const &node) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1)
        return std::nullopt;

    auto const &in = graph.tensor(node.inputs[0]);
    if (!in.symmetry_hint)
        return std::nullopt;
    return *in.symmetry_hint;
}

/// Rule: ``C = α·A + β·B`` (or simple sum), if A and B carry identical
/// descriptors, C inherits them.
std::optional<SymmetryDescriptor> linear_combination_symmetry(Graph const &graph, Node const &node) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1)
        return std::nullopt;

    auto const &a = graph.tensor(node.inputs[0]);
    auto const &b = graph.tensor(node.inputs[1]);
    if (!a.symmetry_hint || !b.symmetry_hint)
        return std::nullopt;
    if (!(*a.symmetry_hint == *b.symmetry_hint))
        return std::nullopt;

    return *a.symmetry_hint;
}

/// Rule: ``C = AᵀA`` / ``AAᵀ`` / ``AᴴA`` via einsum, when the same tensor
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
std::optional<SymmetryDescriptor> self_contraction_symmetry(Graph const &graph, Node const &node) {
    if (operand_inputs(node).size() != 2 || node.outputs.size() != 1)
        return std::nullopt;
    if (node.inputs[0] != node.inputs[1])
        return std::nullopt;

    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (!desc)
        return std::nullopt;

    // Classic rank-2 × rank-2 → rank-2 with exactly one link index.
    if (desc->spec.a_indices.size() != 2 || desc->spec.b_indices.size() != 2 || desc->spec.c_indices.size() != 2)
        return std::nullopt;
    if (desc->spec.link_indices.size() != 1)
        return std::nullopt;

    auto const &in = graph.tensor(node.inputs[0]);

    bool const is_complex = in.dtype == packed_gemm::ScalarType::Complex64 || in.dtype == packed_gemm::ScalarType::Complex128;
    bool const one_conj   = live_conj_a(*desc) ^ live_conj_b(*desc); // XOR: exactly one side conjugated

    if (is_complex && one_conj)
        return SymmetryDescriptor::hermitian_pair(0, 1);
    return SymmetryDescriptor::symmetric_pair(0, 1);
}

/// Rule: permute of a rank-2 symmetric/Hermitian tensor stays
/// symmetric/Hermitian. Covers both the identity permute (trivially
/// preserving) and the swap permute (``{j,i}``, for a symmetric tensor the
/// output equals the input). Beta must be zero (pure overwrite); otherwise
/// we can't reason about what C was before the add.
std::optional<SymmetryDescriptor> permute_symmetry(Graph const &graph, Node const &node) {
    if (node.inputs.size() != 1 || node.outputs.size() != 1)
        return std::nullopt;

    auto const *desc = node.op_data.get_if<PermuteDescriptor>();
    if (!desc)
        return std::nullopt;
    if (desc->beta != 0.0)
        return std::nullopt;
    // Rank-2 only (higher-rank symmetry descriptors can describe cross-
    // axis invariants that don't trivially carry through every permute).
    if (desc->a_indices.size() != 2 || desc->c_indices.size() != 2)
        return std::nullopt;

    auto const &in = graph.tensor(node.inputs[0]);
    if (!in.symmetry_hint)
        return std::nullopt;
    // Only propagate rank-2 symmetric / Hermitian pair descriptors.
    auto const &ops = in.symmetry_hint->ops;
    if (ops.size() != 1)
        return std::nullopt;
    auto const &op = ops[0];
    if (op.permutation[0] != 1 || op.permutation[1] != 0 || op.sign != +1)
        return std::nullopt;

    return *in.symmetry_hint;
}

} // namespace

std::optional<SymmetryDescriptor> output_symmetry(Graph const &graph, Node const &node) {
    switch (node.kind) {
    case OpKind::Scale:
        return scale_symmetry(graph, node);
    case OpKind::Axpby:
        return linear_combination_symmetry(graph, node);
    case OpKind::Einsum:
        return self_contraction_symmetry(graph, node);
    case OpKind::Permute:
    case OpKind::Transpose:
        return permute_symmetry(graph, node);
    default:
        return std::nullopt;
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
