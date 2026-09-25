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
#include <map>
#include <memory>
#include <span>
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
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr) {
            return std::nullopt;
        }
        auto const lists = live_index_lists(*desc);
        site.operators   = lists.operators;
        site.c_indices   = lists.c;
        site.overwrites  = is_zero(live_c_prefactor(*desc));
    } else if (node.kind == OpKind::Permute) {
        auto const *desc = node.op_data.get_if<PermuteDescriptor>();
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

/// Where a letter sits in a list, when it sits there exactly once.
///
/// "Exactly once" is not fussiness. A letter appearing twice in an operand is a
/// DIAGONAL access, and swapping two output axes then does not correspond to
/// swapping two of that operand's slots, so the rule below would be reasoning
/// about a permutation the contraction does not perform.
std::optional<int> sole_position(std::vector<std::string> const &list, std::string const &letter) {
    if (std::count(list.begin(), list.end(), letter) != 1) {
        return std::nullopt;
    }
    auto const first = std::ranges::find(list, letter);
    auto const at    = static_cast<int>(first - list.begin());
    return at < kMaxSymmetryRank ? std::optional<int>{at} : std::nullopt;
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

namespace {

/// What ONE write contributes to its destination.
///
/// A tensor's antisymmetry is a property of its final contents, and a tensor
/// built by accumulation has no single node that settles them. So the rules
/// answer a smaller question - what does THIS write add - and the caller
/// combines the answers across every write the tensor receives.
struct Contribution {
    SymmetryDescriptor generators;        ///< what this write's addend is antisymmetric under
    bool               overwrites{false}; ///< it replaces the destination rather than adding to it
    bool               understood{false}; ///< a rule applied; an unknown write poisons the conclusion
};

} // namespace

std::vector<std::string> AntisymmetryInference::explain() const {
    if (_num_tagged == 0) {
        return {};
    }
    return {
        fmt::format("AntisymmetryInference: tagged {} tensor(s) antisymmetric, from {} candidate write(s)", _num_tagged, _num_candidates)};
}

void AntisymmetryInference::reset_stats() {
    _num_candidates = 0;
    _num_tagged     = 0;
}

bool AntisymmetryInference::run(Graph &graph) {
    // Topological order so a fact settled on one tensor is available to the node
    // that consumes it. The rules chain: a detected leaf fact reaches a foldable
    // premise only by being carried forward.
    graph.topological_sort();

    auto const nodes_view = std::span<Node const>{graph.nodes()};

    // SELF-GATE. In the default pipeline this runs over every graph, and almost
    // no graph names a permutation operator. Without a fact to start from there
    // is nothing for the rules to carry: detection records a hint only where an
    // operator asked about one, so no operator means no hint means no
    // conclusion. Leaving early keeps the cost at one walk over the node list
    // rather than a rule evaluation per contraction.
    bool anything_to_do = std::ranges::any_of(nodes_view, [](Node const &node) { return read_operator_site(node).has_value(); });
    if (!anything_to_do) {
        anything_to_do = std::ranges::any_of(graph.tensors_map(), [](auto const &entry) { return entry.second.symmetry_hint != nullptr; });
    }
    if (!anything_to_do) {
        return false;
    }

    auto const guard = EscapeAnalysis::over(graph);

    auto const hint_of = [&](TensorId id) -> SymmetryDescriptor const * {
        auto const *handle = graph.find_tensor(graph.resolve_alias(id));
        return handle != nullptr ? handle->symmetry_hint.get() : nullptr;
    };

    // Every non-lifecycle write each tensor receives, in program order. A
    // tensor's contents are settled by ALL of them, not by the last one, which
    // is what the single-writer guard could not express: an accumulated
    // intermediate is exactly the shape a residual builds and it has no single
    // settling node.
    std::map<TensorId, std::vector<std::size_t>> writers;
    for (std::size_t i = 0; i < nodes_view.size(); ++i) {
        if (is_lifecycle(nodes_view[i].kind)) {
            continue;
        }
        for (auto const out : nodes_view[i].outputs) {
            writers[graph.resolve_alias(out)].push_back(i);
        }
    }

    std::map<std::size_t, Contribution>                   per_node;
    std::map<TensorId, std::vector<Contribution const *>> collected;

    for (std::size_t i = 0; i < nodes_view.size(); ++i) {
        Node const &node = nodes_view[i];
        if (is_lifecycle(node.kind) || node.outputs.size() != 1) {
            continue;
        }

        Contribution contribution;

        // ── R1: the output of a permutation operator ────────────────────────
        if (auto const site = read_operator_site(node); site.has_value()) {
            ++_num_candidates;
            contribution.overwrites = site->overwrites;
            auto const desc         = antisymmetry_of(*site);
            auto const requirement  = within_group_requirement(*site);
            if (desc.has_value() && requirement.has_value()) {
                if (requirement->empty()) {
                    // Unconditional arm: every group is a singleton, so the
                    // expansion is the full signed sum over a symmetric group and
                    // the addend is antisymmetric whatever the operand was.
                    contribution.generators = *desc;
                    contribution.understood = true;
                } else if (site->has_operand) {
                    // Conditional arm: a coset sum carries no antisymmetry of its
                    // own, but over an operand already antisymmetric within each
                    // group the result is fully antisymmetric on the operator's
                    // letters.
                    auto const *operand = hint_of(site->operand);
                    if (operand != nullptr && contains_all(*operand, *requirement)) {
                        contribution.generators = *desc;
                        contribution.understood = true;
                    } else {
                        note_skip("a coset operator whose operand is not known antisymmetric within each group",
                                  fmt::format("node #{}", node.id));
                    }
                } else {
                    note_skip("a coset operator over a contraction, whose operand is not a tensor this pass can ask about",
                              fmt::format("node #{}", node.id));
                }
            }
        }
        // ── R3: antisymmetry through a contraction ──────────────────────────
        else if (node.kind == OpKind::Einsum && node.inputs.size() >= 2) {
            auto const *desc = node.op_data.get_if<EinsumDescriptor>();
            if (desc != nullptr) {
                ++_num_candidates;
                auto const  lists       = live_index_lists(*desc);
                auto const &a_idx       = lists.a;
                auto const &b_idx       = lists.b;
                auto const &c_idx       = lists.c;
                contribution.overwrites = is_zero(live_c_prefactor(*desc));

                // C(..p..q..) = sum_links ab * A(..p..q..) * B(...). When ONE
                // operand carries both letters, in slots it is antisymmetric in,
                // and the other carries neither, swapping p and q in the output
                // swaps exactly those two slots: the carrier negates and the
                // other operand is untouched, so the addend negates.
                for (std::size_t x = 0; x + 1 < c_idx.size(); ++x) {
                    for (std::size_t y = x + 1; y < c_idx.size(); ++y) {
                        auto const out_p = sole_position(c_idx, c_idx[x]);
                        auto const out_q = sole_position(c_idx, c_idx[y]);
                        if (!out_p.has_value() || !out_q.has_value()) {
                            continue;
                        }
                        for (int which = 0; which < 2; ++which) {
                            auto const &carrier = which == 0 ? a_idx : b_idx;
                            auto const &other   = which == 0 ? b_idx : a_idx;
                            auto const  at_p    = sole_position(carrier, c_idx[x]);
                            auto const  at_q    = sole_position(carrier, c_idx[y]);
                            if (!at_p.has_value() || !at_q.has_value()) {
                                continue;
                            }
                            if (std::ranges::find(other, c_idx[x]) != other.end() || std::ranges::find(other, c_idx[y]) != other.end()) {
                                continue;
                            }
                            auto const *hint = hint_of(node.inputs[static_cast<std::size_t>(which)]);
                            if (hint != nullptr && std::ranges::find(hint->ops, SymmetryOp::swap(*at_p, *at_q, -1)) != hint->ops.end()) {
                                contribution.generators.add(SymmetryOp::swap(*out_p, *out_q, -1));
                                break;
                            }
                        }
                    }
                }
                contribution.understood = !contribution.generators.empty();
            }
        }
        // ── R2: division by an invariant ────────────────────────────────────
        else if (node.kind == OpKind::DirectDivision && node.inputs.size() >= 2) {
            auto const *edesc = node.op_data.get_if<ElementwiseBinaryDescriptor>();
            if (edesc != nullptr) {
                ++_num_candidates;
                contribution.overwrites = is_zero(live_beta(*edesc));
                auto const *numerator   = hint_of(node.inputs[0]);
                auto const *divisor     = hint_of(node.inputs[1]);
                if (numerator != nullptr && divisor != nullptr) {
                    // A quotient keeps exactly those of the numerator's
                    // antisymmetries whose permutation leaves the divisor alone.
                    for (auto const &op : numerator->ops) {
                        if (op.sign < 0 && std::ranges::find(divisor->ops, as_invariance(op)) != divisor->ops.end()) {
                            contribution.generators.add(op);
                        }
                    }
                }
                contribution.understood = !contribution.generators.empty();
            }
        }
        // ── R4: a linear combination ────────────────────────────────────────
        else if (node.kind == OpKind::Axpby && node.inputs.size() >= 1) {
            auto const *adesc = node.op_data.get_if<AxpbyDescriptor>();
            if (adesc != nullptr) {
                ++_num_candidates;
                contribution.overwrites = is_zero(live_beta(*adesc));
                // Scaling preserves antisymmetry, so what this write ADDS carries
                // whatever its source carries. The destination's own prior
                // contents are a separate contribution, already recorded.
                if (auto const *source = hint_of(node.inputs[0]); source != nullptr) {
                    for (auto const &op : source->ops) {
                        if (op.sign < 0) {
                            contribution.generators.add(op);
                        }
                    }
                }
                contribution.understood = !contribution.generators.empty();
            }
        }

        per_node[i] = std::move(contribution);
        collected[graph.resolve_alias(node.outputs[0])].push_back(&per_node[i]);

        // Settle the tensor once its LAST write has been seen. Waiting until the
        // end of the loop would keep the fact from the nodes that consume it.
        TensorId const out  = graph.resolve_alias(node.outputs[0]);
        auto const    &list = writers[out];
        if (list.empty() || list.back() != i) {
            continue;
        }

        auto *handle = graph.find_tensor(out);
        if (handle == nullptr || !handle->is_intermediate) {
            note_skip("the destination is not a graph-owned intermediate", fmt::format("tensor #{}", out));
            continue;
        }
        if (guard.touched_by_subtree(out)) {
            note_skip("a child sub-graph writes the destination, so this graph does not settle it", fmt::format("tensor #{}", out));
            continue;
        }

        auto const &contributions = collected[out];
        if (contributions.size() != list.size()) {
            continue; // a write this pass did not visit; cannot conclude
        }
        if (!contributions.front()->overwrites) {
            note_skip("the first write accumulates, so the destination's prior contents are unaccounted for",
                      fmt::format("tensor #{}", out));
            continue;
        }
        if (std::ranges::any_of(contributions, [](Contribution const *c) { return !c->understood; })) {
            note_skip("a write contributes something this pass cannot characterize", fmt::format("tensor #{}", out));
            continue;
        }

        // The INTERSECTION. A sum is antisymmetric under exactly the permutations
        // every addend is antisymmetric under; one indifferent addend is enough
        // to destroy the property for the whole.
        SymmetryDescriptor settled = contributions.front()->generators;
        for (std::size_t c = 1; c < contributions.size(); ++c) {
            SymmetryDescriptor both;
            for (auto const &op : settled.ops) {
                if (std::ranges::find(contributions[c]->generators.ops, op) != contributions[c]->generators.ops.end()) {
                    both.add(op);
                }
            }
            settled = std::move(both);
        }
        if (settled.empty()) {
            // Distinguished from "cannot characterize" above, because it means
            // something different and points somewhere else. Every write was
            // understood; they simply do not agree on which axes their addends
            // are antisymmetric in, so the sum is antisymmetric in none. That is
            // usually a fact about the DATA rather than a gap in the rules, and
            // reading it as a missing capability sends the reader hunting for a
            // rule that would not help.
            if (contributions.size() > 1) {
                note_skip("the writes are each antisymmetric, but in different axes, so their sum is antisymmetric in none",
                          fmt::format("tensor #{}", out));
            }
            continue;
        }
        if (handle->symmetry_hint != nullptr && contains_all(*handle->symmetry_hint, settled)) {
            continue;
        }

        SymmetryDescriptor merged = handle->symmetry_hint != nullptr ? *handle->symmetry_hint : SymmetryDescriptor{};
        for (auto const &op : settled.ops) {
            if (std::ranges::find(merged.ops, op) == merged.ops.end()) {
                merged.add(op);
            }
        }
        handle->symmetry_hint = std::make_shared<SymmetryDescriptor>(std::move(merged));
        ++_num_tagged;
    }

    // Annotation only; the node list is untouched.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
