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

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

// NOLINTNEXTLINE(misc-no-recursion): an axpby's source is itself a chain; kMaxChainDepth bounds it.
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

// ── contribution_symmetry ─────────────────────────────────────────────────────

/// One factor of an inlined product: a tensor, the letters it is read with, and the node reading it.
struct Factor {
    TensorId                 id;
    std::vector<std::string> letters;
    NodeId                   reader;
};

/// A contribution flattened to factors over output letters and summed letters.
struct Product {
    std::vector<Factor>             factors;
    std::unordered_set<std::string> summed;
    int                             fresh{0};
};

/// Most einsums inlined into one contribution. Bounds work on a pathological chain; the ladder needs two.
constexpr int kMaxInlineDepth = 6;

/// Most elements a factor's symmetry group may have before the search gives up. The 8-fold ERI
/// group is the largest a chemistry tensor carries.
constexpr std::size_t kMaxGroupOrder = 64;

/// The einsum that writes @p id, when inlining it into @p reader's product is sound: it is the only
/// value-writer of the buffer, no sub-graph writes it, it comes before the reader, nothing reads the
/// buffer through another view, and it overwrites rather than accumulates.
Node const *inlinable_writer(EscapeAnalysis const &escape, TensorId id, NodeId reader) {
    auto const &graph = escape.graph();
    if (escape.writer_count(id) != 1 || escape.touched_by_subtree(id) || escape.aliases_of(id).size() != 1)
        return nullptr;
    for (auto const &node : graph.nodes()) {
        if (node.id == reader)
            return nullptr; // Reached the reader first: the write is not before it.
        if (is_lifecycle(node.kind) || std::ranges::find(node.outputs, id) == node.outputs.end())
            continue;
        if (node.kind != OpKind::Einsum || node.outputs.size() != 1)
            return nullptr;
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr || !is_zero(live_c_prefactor(*desc)))
            return nullptr;
        return &node;
    }
    return nullptr;
}

/// Append @p node's product to @p out, its output letters renamed by @p rename and every other
/// letter renamed to a fresh summed one. False when the node cannot be expressed as a plain product.
// NOLINTNEXTLINE(misc-no-recursion): an inlined source may have sources of its own; kMaxInlineDepth bounds it.
bool expand(EscapeAnalysis const &escape, Node const &node, std::unordered_map<std::string, std::string> const &rename, Product &out,
            int depth) {
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr || live_conj_a(*desc) || live_conj_b(*desc))
        return false;
    auto const lists = live_index_lists(*desc);
    if (!lists.operators.empty())
        return false;
    auto const operands = operand_inputs(node);
    if (operands.size() != 2)
        return false;

    std::unordered_map<std::string, std::string> local = rename;
    auto const                                   name  = [&](std::string const &letter) -> std::string const                                    &{
        auto const [it, inserted] = local.try_emplace(letter);
        if (inserted) {
            it->second = fmt::format("#{}", out.fresh++);
            out.summed.insert(it->second);
        }
        return it->second;
    };

    for (int slot = 0; slot < 2; ++slot) {
        auto const &raw = slot == 0 ? lists.a : lists.b;
        auto const  id  = operands[slot];

        std::vector<std::string> letters;
        letters.reserve(raw.size());
        for (auto const &letter : raw)
            letters.push_back(name(letter));

        if (Node const *source = depth < kMaxInlineDepth ? inlinable_writer(escape, id, node.id) : nullptr) {
            // The source's output letters become the letters this node reads it with.
            auto const source_lists = live_index_lists(*source->op_data.get_if<EinsumDescriptor>());
            if (source_lists.c.size() != letters.size())
                return false;
            std::unordered_map<std::string, std::string> source_rename;
            for (std::size_t p = 0; p < letters.size(); ++p) {
                if (!source_rename.emplace(source_lists.c[p], letters[p]).second)
                    return false; // A diagonal output: its letters do not map one to one.
            }
            if (!expand(escape, *source, source_rename, out, depth + 1))
                return false;
        } else {
            out.factors.push_back(Factor{.id = id, .letters = std::move(letters), .reader = node.id});
        }
    }
    return true;
}

/// Every element of the group @p desc generates on a rank-@p rank tensor, or nothing when it is
/// conjugating, malformed, or too large to search.
std::optional<std::vector<SymmetryOp>> group_elements(std::optional<SymmetryDescriptor> const &desc, std::size_t rank) {
    std::vector<SymmetryOp> elements{SymmetryOp::identity()};
    if (!desc)
        return elements;
    for (auto const &gen : desc->ops) {
        if (gen.conjugate)
            return std::nullopt;
        // A permutation of the tensor's own positions, every other position fixed. A declared
        // descriptor is user data, so this is checked rather than assumed.
        std::array<bool, kMaxSymmetryRank> hit{};
        for (std::size_t p = 0; p < static_cast<std::size_t>(kMaxSymmetryRank); ++p) {
            auto const target = static_cast<std::size_t>(static_cast<unsigned char>(gen.permutation[p]));
            if (p < rank ? (target >= rank || hit[target]) : target != p)
                return std::nullopt;
            hit[target] = true;
        }
    }
    // Each element e is an identity T[L] = sign * T[apply(e, L)], with apply(e, L)[p] = L[e.perm[p]].
    // Following e by g gives T[L] = e.sign * g.sign * T[apply(c, L)] with c[p] = e.perm[g.perm[p]].
    for (std::size_t next = 0; next < elements.size(); ++next) {
        for (auto const &gen : desc->ops) {
            auto composed = SymmetryOp::identity();
            for (int p = 0; p < kMaxSymmetryRank; ++p)
                composed.permutation[p] =
                    elements[next].permutation[static_cast<std::size_t>(static_cast<unsigned char>(gen.permutation[p]))];
            composed.sign = static_cast<int8_t>(elements[next].sign * gen.sign);
            if (std::ranges::find(elements, composed) != elements.end())
                continue;
            // The same permutation with the other sign means T = -T: every element is zero.
            auto flipped = composed;
            flipped.sign = static_cast<int8_t>(-flipped.sign);
            if (std::ranges::find(elements, flipped) != elements.end())
                return std::nullopt;
            elements.push_back(composed);
            if (elements.size() > kMaxGroupOrder)
                return std::nullopt;
        }
    }
    return elements;
}

/// The matching search: carry factor @p k onto an unused factor of the same tensor through an
/// element of its group, binding the summed-letter relabeling @p pi as it goes.
struct Matcher {
    std::vector<Factor> const                          &factors;
    std::vector<std::vector<SymmetryOp>> const         &groups;
    std::unordered_map<std::string, std::string> const &sigma; ///< On output letters.
    std::unordered_set<std::string> const              &summed;

    std::vector<bool>                            used;
    std::unordered_map<std::string, std::string> pi;
    std::unordered_set<std::string>              pi_image;

    // NOLINTNEXTLINE(misc-no-recursion): one level per factor.
    std::optional<int> search(std::size_t k, int sign) {
        if (k == factors.size())
            return sign;
        auto const &from = factors[k];
        for (std::size_t j = 0; j < factors.size(); ++j) {
            if (used[j] || factors[j].id != from.id)
                continue;
            for (auto const &g : groups[k]) {
                std::vector<std::string> bound;
                if (!bind(from.letters, factors[j].letters, g, bound)) {
                    unbind(bound);
                    continue;
                }
                used[j] = true;
                if (auto found = search(k + 1, sign * g.sign))
                    return found;
                used[j] = false;
                unbind(bound);
            }
        }
        return std::nullopt;
    }

    /// Does apply(g, image of @p from) equal @p to? Binds pi where it is still free.
    bool bind(std::vector<std::string> const &from, std::vector<std::string> const &to, SymmetryOp const &g,
              std::vector<std::string> &bound) {
        for (std::size_t p = 0; p < to.size(); ++p) {
            auto const &src = from[static_cast<std::size_t>(static_cast<unsigned char>(g.permutation[p]))];
            auto const &dst = to[p];
            if (auto const out = sigma.find(src); out != sigma.end()) {
                if (out->second != dst)
                    return false;
                continue;
            }
            if (!summed.contains(dst))
                return false;
            if (auto const known = pi.find(src); known != pi.end()) {
                if (known->second != dst)
                    return false;
                continue;
            }
            // Injectivity is implied once every factor is matched (each target position is covered
            // once, so pi is onto the summed letters); checking it here only prunes the search early.
            if (pi_image.contains(dst))
                return false;
            pi.emplace(src, dst);
            pi_image.insert(dst);
            bound.push_back(src);
        }
        return true;
    }

    void unbind(std::vector<std::string> const &bound) {
        for (auto const &src : bound) {
            pi_image.erase(pi.at(src));
            pi.erase(src);
        }
    }
};

/// Does any node strictly between two readers of @p id write its buffer? Two factors of one tensor
/// can trade places only if both read the same values.
bool written_between(EscapeAnalysis const &escape, TensorId id, std::vector<NodeId> const &readers) {
    auto const &graph = escape.graph();
    auto const  root  = graph.resolve_alias(id);
    bool        open  = false;
    std::size_t seen  = 0;
    for (auto const &node : graph.nodes()) {
        if (std::ranges::find(readers, node.id) != readers.end()) {
            open = ++seen < readers.size();
            continue;
        }
        if (!open || is_lifecycle(node.kind) || node.kind == OpKind::View)
            continue;
        if (std::ranges::any_of(node.outputs, [&](TensorId out) { return same_buffer(graph, out, root); }))
            return true;
    }
    return false;
}

} // namespace

std::optional<SymmetryOp> contribution_symmetry(EscapeAnalysis const &escape, NodeId writer, SymmetryOp const &sigma) {
    auto const &graph = escape.graph();
    auto const &nodes = graph.nodes();
    auto const  at    = std::ranges::find(nodes, writer, &Node::id);
    if (at == nodes.end() || at->kind != OpKind::Einsum || at->outputs.size() != 1)
        return std::nullopt;
    auto const *desc = at->op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr)
        return std::nullopt;
    auto const &c    = live_index_lists(*desc).c;
    auto const  rank = c.size();
    if (rank == 0 || rank > static_cast<std::size_t>(kMaxSymmetryRank))
        return std::nullopt;

    // Output letters keep their names; sigma maps each onto the letter at its target position.
    std::unordered_map<std::string, std::string> identity;
    std::unordered_map<std::string, std::string> sigma_letters;
    for (std::size_t p = 0; p < rank; ++p) {
        auto const target = static_cast<std::size_t>(static_cast<unsigned char>(sigma.permutation[p]));
        if (target >= rank || !identity.emplace(c[p], c[p]).second)
            return std::nullopt; // Out of range, or a diagonal output.
        sigma_letters.emplace(c[p], c[target]);
    }

    Product product;
    if (!expand(escape, *at, identity, product, 0))
        return std::nullopt;

    // A factor that reads the output makes the contribution depend on what the output held.
    auto const                                        out_root = graph.resolve_alias(at->outputs[0]);
    std::unordered_map<TensorId, std::vector<NodeId>> readers;
    std::vector<std::vector<SymmetryOp>>              groups;
    for (auto const &factor : product.factors) {
        if (same_buffer(graph, factor.id, out_root))
            return std::nullopt;
        auto elements = group_elements(symmetry_at(escape, factor.id, factor.reader), factor.letters.size());
        if (!elements)
            return std::nullopt;
        groups.push_back(std::move(*elements));
        auto &seen = readers[factor.id];
        if (std::ranges::find(seen, factor.reader) == seen.end())
            seen.push_back(factor.reader);
    }
    for (auto const &[id, at_readers] : readers) {
        if (at_readers.size() > 1 && written_between(escape, id, at_readers))
            return std::nullopt;
    }

    Matcher matcher{.factors = product.factors, .groups = groups, .sigma = sigma_letters, .summed = product.summed};
    matcher.used.assign(product.factors.size(), false);
    auto const sign = matcher.search(0, +1);
    if (!sign)
        return std::nullopt;

    auto result      = sigma;
    result.sign      = static_cast<int8_t>(*sign);
    result.conjugate = false;
    return result;
}

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
