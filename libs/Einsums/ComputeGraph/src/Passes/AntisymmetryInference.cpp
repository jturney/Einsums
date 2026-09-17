//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryInference.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The operators and output index list a node carries, or nothing.
///
/// Reads the LIVE lists where the node has them, for the reason the IR writer
/// gives: the executor reads ``indices->spec``, and the descriptor's own copy
/// beside it is the at-capture snapshot a rewriting pass may have left behind.
struct OperatorSite {
    std::vector<PermutationOperator> operators;
    std::vector<std::string>         c_indices;
    bool                             overwrites{false};  ///< destination prefactor is zero
    bool                             has_operand{false}; ///< the operator is applied to a named tensor
    TensorId                         operand{0};         ///< that tensor, for the conditional arm
};

std::optional<OperatorSite> read_operator_site(Node const &node) {
    OperatorSite site;

    if (node.kind == OpKind::Einsum) {
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return std::nullopt;
        }
        bool const live = desc->indices != nullptr;
        site.operators  = live ? desc->indices->spec.operators : desc->operators;
        site.c_indices  = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
        site.overwrites = is_zero(live_c_prefactor(*desc));
    } else if (node.kind == OpKind::Permute) {
        auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return std::nullopt;
        }
        site.operators  = desc->operators;
        site.c_indices  = desc->c_indices;
        site.overwrites = desc->params != nullptr ? is_zero(desc->params->beta) : desc->beta == std::complex<double>{0.0, 0.0};
        // A permute applies its operator to a NAMED tensor, so the conditional
        // arm has something whose antisymmetry it can ask about. An einsum's
        // operator wraps a contraction, whose result is never a tensor until it
        // is written, so that arm cannot reach one.
        if (!node.inputs.empty()) {
            site.has_operand = true;
            site.operand     = node.inputs[0];
        }
    } else {
        return std::nullopt;
    }

    if (site.operators.empty()) {
        return std::nullopt;
    }
    return site;
}

/// Whether every group of every operator names exactly one letter.
///
/// This is the whole of the rule's applicability. A group of two or more makes
/// the expansion a coset sum rather than the full signed sum over a symmetric
/// group, and a coset sum carries no antisymmetry of its own.
bool all_groups_singleton(std::vector<PermutationOperator> const &operators) {
    for (auto const &op : operators) {
        for (auto const &group : op.groups) {
            if (group.size() != 1) {
                return false;
            }
        }
    }
    return true;
}

/// Generators stating that the axes an operator names are fully antisymmetric.
///
/// Adjacent transpositions of those axes, in axis order, which generate the
/// symmetric group on them. Returns nothing when a letter does not name exactly
/// one axis, which the spec parser already rejects and which would otherwise
/// leave the permutation ambiguous.
std::optional<SymmetryDescriptor> antisymmetry_of(OperatorSite const &site) {
    SymmetryDescriptor desc;

    for (auto const &op : site.operators) {
        std::vector<int> axes;
        for (auto const &letter : op.letters()) {
            auto const first = std::ranges::find(site.c_indices, letter);
            if (first == site.c_indices.end()) {
                return std::nullopt;
            }
            auto const position = static_cast<int>(first - site.c_indices.begin());
            if (std::count(site.c_indices.begin(), site.c_indices.end(), letter) != 1) {
                return std::nullopt;
            }
            axes.push_back(position);
        }
        std::ranges::sort(axes);
        if (axes.size() < 2 || axes.back() >= kMaxSymmetryRank) {
            return std::nullopt;
        }
        for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
            desc.add(SymmetryOp::swap(axes[k], axes[k + 1], -1));
        }
    }

    if (desc.empty()) {
        return std::nullopt;
    }
    return desc;
}

/// The antisymmetry a coset operator's operand must ALREADY carry before the
/// operator's output carries any.
///
/// Empty for a singleton partition, which asks nothing of its operand: that is
/// the unconditional arm, and an empty requirement is trivially met.
std::optional<SymmetryDescriptor> within_group_requirement(OperatorSite const &site) {
    SymmetryDescriptor desc;
    for (auto const &op : site.operators) {
        for (auto const &group : op.groups) {
            std::vector<int> axes;
            for (auto const &letter : group) {
                auto const first = std::ranges::find(site.c_indices, letter);
                if (first == site.c_indices.end() || std::count(site.c_indices.begin(), site.c_indices.end(), letter) != 1) {
                    return std::nullopt;
                }
                auto const position = static_cast<int>(first - site.c_indices.begin());
                if (position >= kMaxSymmetryRank) {
                    return std::nullopt;
                }
                axes.push_back(position);
            }
            std::ranges::sort(axes);
            for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
                desc.add(SymmetryOp::swap(axes[k], axes[k + 1], -1));
            }
        }
    }
    return desc;
}

bool contains_all(SymmetryDescriptor const &have, SymmetryDescriptor const &need) {
    return std::ranges::all_of(need.ops, [&](SymmetryOp const &op) { return std::ranges::find(have.ops, op) != have.ops.end(); });
}

/// The same permutation as @p op, asserted as an INVARIANCE rather than whatever
/// @p op asserts. R2 asks of a divisor exactly what the numerator's generator
/// permutes, with the opposite kind of claim.
SymmetryOp as_invariance(SymmetryOp op) {
    op.sign      = +1;
    op.conjugate = false;
    return op;
}

} // namespace

std::vector<std::string> AntisymmetryInference::explain() const {
    if (_num_tagged == 0) {
        return {};
    }
    return {fmt::format("AntisymmetryInference: tagged {} of {} operator output(s) antisymmetric", _num_tagged, _num_candidates)};
}

void AntisymmetryInference::reset_stats() {
    _num_candidates = 0;
    _num_tagged     = 0;
}

bool AntisymmetryInference::run(Graph &graph) {
    // Topological order so a fact established on one node's output is available
    // to the node that consumes it. The rules chain: a detected leaf fact reaches
    // a foldable premise only by being carried forward.
    graph.topological_sort();

    // The soundness guard SymmetryPropagation uses, shared rather than counted a
    // second time: exactly one value-writer in this graph, and no descendant
    // sub-graph touching the buffer. Without it a later overwrite could destroy
    // the structure this pass just promised.
    auto const guard = EscapeAnalysis::over(graph);

    auto const tag = [&](Node const &node, TensorId raw, SymmetryDescriptor const &desc) {
        TensorId const out    = graph.resolve_alias(raw);
        auto          *handle = graph.find_tensor(out);
        if (handle == nullptr || !handle->is_intermediate) {
            note_skip("the destination is not a graph-owned intermediate", fmt::format("node #{}", node.id));
            return;
        }
        if (!guard.stable(out)) {
            note_skip("the destination is written more than once, or by a child sub-graph", fmt::format("node #{}", node.id));
            return;
        }
        if (handle->symmetry_hint != nullptr && contains_all(*handle->symmetry_hint, desc)) {
            return; // already known, and re-running must not count it twice
        }
        SymmetryDescriptor merged = handle->symmetry_hint != nullptr ? *handle->symmetry_hint : SymmetryDescriptor{};
        for (auto const &op : desc.ops) {
            if (std::ranges::find(merged.ops, op) == merged.ops.end()) {
                merged.add(op);
            }
        }
        handle->symmetry_hint = std::make_shared<SymmetryDescriptor>(std::move(merged));
        ++_num_tagged;
    };

    auto const hint_of = [&](TensorId id) -> SymmetryDescriptor const * {
        auto const *handle = graph.find_tensor(graph.resolve_alias(id));
        return handle != nullptr ? handle->symmetry_hint.get() : nullptr;
    };

    for (auto const &node : graph.nodes()) {
        // ── R1: the output of a permutation operator ────────────────────────
        if (auto const site = read_operator_site(node); site.has_value() && node.outputs.size() == 1) {
            ++_num_candidates;

            if (!site->overwrites) {
                note_skip("the node accumulates, so its output is the previous contents plus an antisymmetric part",
                          fmt::format("node #{}", node.id));
            } else if (auto const desc = antisymmetry_of(*site); !desc.has_value()) {
                note_skip("the operator's letters do not each name exactly one addressable output axis", fmt::format("node #{}", node.id));
            } else {
                auto const requirement = within_group_requirement(*site);
                if (!requirement.has_value()) {
                    note_skip("the operator's groups do not map onto addressable axes", fmt::format("node #{}", node.id));
                } else if (requirement->empty()) {
                    // Unconditional arm: every group is a singleton, so the
                    // expansion is the full signed sum over a symmetric group and
                    // the output is antisymmetric whatever the operand was.
                    tag(node, node.outputs[0], *desc);
                } else if (!site->has_operand) {
                    note_skip("a coset operator over a contraction, whose operand is not a tensor this pass can ask about",
                              fmt::format("node #{}", node.id));
                } else if (auto const *operand = hint_of(site->operand); operand == nullptr || !contains_all(*operand, *requirement)) {
                    note_skip("a coset operator whose operand is not known antisymmetric within each group",
                              fmt::format("node #{}", node.id));
                } else {
                    // Conditional arm. A coset sum carries no antisymmetry of its
                    // own, but over an operand that already has it within each
                    // group the output is fully antisymmetric on the operator's
                    // letters, which is the precondition its well-definedness
                    // rests on. Pinned against real data in
                    // AntisymmetryDetection.cpp.
                    tag(node, node.outputs[0], *desc);
                }
            }
        }

        // ── R2: division by an invariant ────────────────────────────────────
        if (node.kind != OpKind::DirectDivision || node.inputs.size() < 2 || node.outputs.size() != 1) {
            continue;
        }
        auto const *edesc = std::get_if<ElementwiseBinaryDescriptor>(&node.op_data);
        if (edesc == nullptr) {
            continue;
        }
        ++_num_candidates;
        if (!is_zero(live_beta(*edesc))) {
            note_skip("the division accumulates, so its output is the previous contents plus a quotient", fmt::format("node #{}", node.id));
            continue;
        }

        auto const *numerator = hint_of(node.inputs[0]);
        auto const *divisor   = hint_of(node.inputs[1]);
        if (numerator == nullptr || divisor == nullptr) {
            note_skip("the numerator's antisymmetry or the divisor's invariance is not established", fmt::format("node #{}", node.id));
            continue;
        }

        // A quotient keeps exactly those of the numerator's antisymmetries whose
        // permutation leaves the divisor alone. Scaling by alpha preserves them;
        // dividing by something that MOVES under the permutation does not.
        SymmetryDescriptor kept;
        for (auto const &op : numerator->ops) {
            if (op.sign >= 0) {
                continue; // only antisymmetry is carried through
            }
            if (std::ranges::find(divisor->ops, as_invariance(op)) != divisor->ops.end()) {
                kept.add(op);
            }
        }
        if (kept.empty()) {
            note_skip("the divisor is not invariant under any of the numerator's antisymmetries", fmt::format("node #{}", node.id));
            continue;
        }
        tag(node, node.outputs[0], kept);
    }

    // Annotation only; the node list is untouched.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
