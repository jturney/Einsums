//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/SymmetryRules.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>

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

/// The generators @p a and @p b share, compared generator by generator. Two descriptors of the
/// same group written with different generators share none, which declines: conservative, never
/// wrong.
std::optional<SymmetryDescriptor> common_generators(std::optional<SymmetryDescriptor> const &a,
                                                    std::optional<SymmetryDescriptor> const &b) {
    if (!a || !b)
        return std::nullopt;
    SymmetryDescriptor out;
    out.tolerance = std::max(a->tolerance, b->tolerance);
    for (auto const &op : a->ops) {
        if (std::ranges::find(b->ops, op) != b->ops.end())
            out.ops.push_back(op);
    }
    if (out.empty())
        return std::nullopt;
    return out;
}

/// What scaling by @p factor leaves of @p desc. Any scalar keeps a plain or antisymmetric
/// generator, but a conjugating one, C(pi) = conj(C), survives only a real factor, since
/// conj(alpha C) is conj(alpha) conj(C).
std::optional<SymmetryDescriptor> scaled_by(std::optional<SymmetryDescriptor> desc, PrefactorScalar const &factor) {
    if (!desc || is_real_valued(factor))
        return desc;
    std::erase_if(desc->ops, [](SymmetryOp const &op) { return op.conjugate; });
    if (desc->empty())
        return std::nullopt;
    return desc;
}

/// Does @p graph's tensor @p id share its buffer with @p root?
bool same_buffer(Graph const &graph, TensorId id, TensorId root) {
    return graph.resolve_alias(id) == root;
}

/// Most writes the chain rule follows through a tensor's sources before giving up. A cycle cannot
/// form (a source is written before the write that reads it), so this bounds work, not recursion
/// that would otherwise not end.
constexpr int kMaxChainDepth = 8;

std::optional<SymmetryDescriptor> symmetry_at_depth(EscapeAnalysis const &escape, TensorId tensor, NodeId reader, int depth) {
    if (depth > kMaxChainDepth)
        return std::nullopt;
    auto const &graph = escape.graph();
    if (escape.touched_by_subtree(tensor))
        return std::nullopt; // A loop body or branch writes it where this node list cannot see.

    auto const  root  = graph.resolve_alias(tensor);
    auto const &nodes = graph.nodes();
    auto const  stop  = std::ranges::find(nodes, reader, &Node::id);
    if (stop == nodes.end())
        return std::nullopt;

    bool                              written = false;
    std::optional<SymmetryDescriptor> state;
    for (auto it = nodes.begin(); it != stop; ++it) {
        auto const &node = *it;
        if (is_lifecycle(node.kind) || node.kind == OpKind::View)
            continue;
        if (std::ranges::none_of(node.outputs, [&](TensorId out) { return same_buffer(graph, out, root); }))
            continue;
        // A write through another view of the buffer changes elements this tensor may or may not
        // hold; nothing below can say which.
        if (node.outputs.size() != 1 || node.outputs[0] != tensor)
            return std::nullopt;

        std::optional<SymmetryDescriptor> term;
        bool                              overwrite = false;
        switch (node.kind) {
        case OpKind::Einsum: {
            auto const *desc = node.op_data.get_if<EinsumDescriptor>();
            if (!desc)
                return std::nullopt;
            if (std::ranges::any_of(operand_inputs(node), [&](TensorId in) { return same_buffer(graph, in, root); }))
                return std::nullopt; // The term reads what it writes.
            term      = self_product_symmetry(graph, node);
            overwrite = is_zero(live_c_prefactor(*desc));
            if (!overwrite)
                state = scaled_by(state, live_c_prefactor(*desc));
            break;
        }
        case OpKind::Axpby: {
            auto const *desc = node.op_data.get_if<AxpbyDescriptor>();
            if (!desc || node.inputs.empty() || same_buffer(graph, node.inputs[0], root))
                return std::nullopt;
            term      = scaled_by(symmetry_at_depth(escape, node.inputs[0], node.id, depth + 1), live_alpha(*desc));
            overwrite = is_zero(live_beta(*desc));
            if (!overwrite)
                state = scaled_by(state, live_beta(*desc));
            break;
        }
        case OpKind::Scale: {
            auto const *desc = node.op_data.get_if<ScaleDescriptor>();
            if (!desc)
                return std::nullopt;
            state   = scaled_by(state, live_factor(*desc));
            written = true;
            continue;
        }
        default:
            return std::nullopt;
        }

        // An accumulation keeps what both have. Onto contents nothing before it set, the state is
        // empty and stays so until a later write overwrites the tensor.
        state   = overwrite ? term : common_generators(state, term);
        written = true;
    }

    if (!written) {
        auto const &hint = graph.tensor(tensor).symmetry_hint;
        if (!hint)
            return std::nullopt;
        return *hint;
    }
    return state;
}

} // namespace

std::optional<SymmetryDescriptor> self_product_symmetry(Graph const &graph, Node const &node) {
    if (node.kind != OpKind::Einsum || node.outputs.size() != 1)
        return std::nullopt;
    auto const operands = operand_inputs(node);
    // The same tensor id, not the same buffer: two views of one buffer hold different elements.
    if (operands.size() != 2 || operands[0] != operands[1])
        return std::nullopt;
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (!desc)
        return std::nullopt;

    auto const lists = live_index_lists(*desc);
    if (!lists.operators.empty() || lists.a.size() != lists.b.size())
        return std::nullopt;
    auto const rank = lists.c.size();
    if (rank < 2 || rank > static_cast<std::size_t>(kMaxSymmetryRank))
        return std::nullopt;

    // The relabeling pi with pi(a[k]) = b[k] and pi(b[k]) = a[k]. Swapping the two operands is that
    // relabeling, so the term is unchanged by it whenever it is a well-defined map.
    std::unordered_map<std::string, std::string> pi;
    auto const                                   bind = [&pi](std::string const &from, std::string const &to) {
        auto const [it, inserted] = pi.emplace(from, to);
        return inserted || it->second == to;
    };
    for (std::size_t k = 0; k < lists.a.size(); ++k) {
        if (!bind(lists.a[k], lists.b[k]) || !bind(lists.b[k], lists.a[k]))
            return std::nullopt;
    }

    // Read pi on C's positions. Every C letter must land on a C letter, and each letter appears in
    // C once, so the positions form a permutation.
    auto op = SymmetryOp::identity();
    for (std::size_t p = 0; p < rank; ++p) {
        auto const  found  = pi.find(lists.c[p]);
        auto const &target = found == pi.end() ? lists.c[p] : found->second;
        auto const  at     = std::ranges::find(lists.c, target);
        if (at == lists.c.end() || std::ranges::count(lists.c, lists.c[p]) != 1)
            return std::nullopt;
        op.permutation[p] = static_cast<int8_t>(at - lists.c.begin());
    }
    if (op == SymmetryOp::identity())
        return std::nullopt; // Nothing moves, so there is nothing to know.

    auto const &in         = graph.tensor(operands[0]);
    bool const  is_complex = in.dtype == packed_gemm::ScalarType::Complex64 || in.dtype == packed_gemm::ScalarType::Complex128;
    op.conjugate           = is_complex && (live_conj_a(*desc) != live_conj_b(*desc));

    return scaled_by(SymmetryDescriptor{{op}}, live_ab_prefactor(*desc));
}

std::optional<SymmetryDescriptor> symmetry_at(EscapeAnalysis const &escape, TensorId tensor, NodeId reader) {
    return symmetry_at_depth(escape, tensor, reader, 0);
}

std::optional<SymmetryDescriptor> output_symmetry(Graph const &graph, Node const &node) {
    switch (node.kind) {
    case OpKind::Scale:
        return scale_symmetry(graph, node);
    case OpKind::Axpby:
        return linear_combination_symmetry(graph, node);
    case OpKind::Einsum: {
        if (auto rank2 = self_contraction_symmetry(graph, node))
            return rank2;
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr || !is_zero(live_c_prefactor(*desc)))
            return std::nullopt;
        return self_product_symmetry(graph, node);
    }
    case OpKind::Permute:
    case OpKind::Transpose:
        return permute_symmetry(graph, node);
    default:
        return std::nullopt;
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
