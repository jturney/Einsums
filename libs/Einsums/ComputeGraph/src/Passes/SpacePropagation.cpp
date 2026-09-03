//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SpacePropagation.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Soundness context for one inference run on one graph, the same rule
/// SymmetryPropagation uses. An annotation is a claim about what a tensor's
/// axes range over for the whole life of the graph, so we make it only when
/// nothing can contradict it after the producing op:
///   - the tensor has exactly one writer in this graph (a second writer could
///     bind the slots to something else entirely), and
///   - the tensor isn't referenced by a child sub-graph (a nested loop /
///     conditional body writes without this graph's node list showing it, a
///     Loop node doesn't list its body's writes).
/// The guards make the pass strictly conservative and therefore safe to
/// recurse into loop bodies.
/// Both conditions in one call: exactly one value-writer in this graph, and no
/// descendant sub-graph touching the buffer. Shared with SymmetryPropagation,
/// LoopInvariantHoisting and the region framework rather than counted a fourth
/// time here, because several derivations of one relation disagreeing in the
/// corner nobody tested is this module's signature bug.
using InferGuard = EscapeAnalysis;

/// What one rule did with one node. A rule declines far more often than it
/// fires, and the interesting declines (operands that disagree about a letter)
/// have to reach `note_skip`, which is protected on the pass. So a rule reports
/// the reason and `run()` records it.
struct RuleResult {
    bool        applied{false}; ///< True when an annotation was written.
    std::string skip_reason;    ///< Short aggregated phrase, empty when there is nothing to record.
    std::string skip_detail;    ///< Per-candidate specifics, emitted only at verbosity 3.
};

/// A rule that examined the node and declined it for a reason worth counting.
RuleResult declined(std::string reason, std::string detail) {
    return RuleResult{.applied = false, .skip_reason = std::move(reason), .skip_detail = std::move(detail)};
}

/// Try to write an inferred annotation onto a graph-owned tensor handle.
/// Returns true when the handle took on a new annotation.
bool apply_inferred(Graph &graph, TensorId out_tid, std::vector<SpaceId> spaces, InferGuard const &guard) {
    if (spaces.empty()) {
        return false; // Nothing resolved, and a partial annotation is never written.
    }
    auto &handle = graph.tensor(out_tid);
    if (!handle.is_intermediate) {
        return false; // Never mutate user-owned tensor state.
    }
    if (!guard.stable(out_tid)) {
        return false; // Could be rebound later / by a child body, don't annotate.
    }
    if (spaces.size() != handle.rank) {
        return false; // An annotation is either one space per axis or nothing at all.
    }
    if (!handle.spaces.empty()) {
        if (!handle.spaces_inferred) {
            return false; // A declaration is authoritative; inference never argues with it.
        }
        if (handle.spaces == spaces) {
            return false; // Already says this, so a re-run reports no new inference.
        }
    }

    handle.spaces          = std::move(spaces);
    handle.spaces_inferred = true;
    return true;
}

/// Bind an operand's index letters to the spaces annotated on its slots.
/// Returns false when two slots of the operand claim different spaces for one
/// letter, which for a repeated (diagonal) letter is a genuine conflict.
bool bind_operand(std::vector<std::string> const &indices, std::vector<SpaceId> const &spaces,
                  std::vector<std::pair<std::string, SpaceId>> &bound) {
    if (spaces.empty() || spaces.size() != indices.size()) {
        return true; // Unannotated (or malformed): contributes nothing, conflicts with nothing.
    }
    for (std::size_t slot = 0; slot < indices.size(); ++slot) {
        SpaceId const id = spaces[slot];
        if (!id.valid()) {
            continue; // A partially annotated tensor: this axis simply says nothing.
        }
        auto const existing = std::ranges::find_if(bound, [&](auto const &entry) { return entry.first == indices[slot]; });
        if (existing == bound.end()) {
            bound.emplace_back(indices[slot], id);
        } else if (existing->second != id) {
            return false;
        }
    }
    return true;
}

/// Rule: an einsum's output slot takes the space of the letter that produced
/// it. The letters are bound from the INPUT operands' current annotations
/// rather than from the node's captured ``letter_spaces`` map, because the map
/// is a snapshot of what capture could see: it holds nothing for a program
/// annotated after capture, and nothing an upstream node's inference added
/// during this very sweep. Reading the operands is what makes this a
/// propagation rather than a replay of capture. The output's own annotation is
/// deliberately not part of the map, so a declaration constrains nothing and an
/// earlier guess cannot conflict with its own refinement.
RuleResult propagate_einsum(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Einsum) {
        return {};
    }
    if (node.inputs.empty() || node.inputs.size() > 2 || node.outputs.size() != 1) {
        return {};
    }

    auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
    if (desc == nullptr) {
        return {};
    }

    std::vector<std::pair<std::string, SpaceId>>          letters;
    std::array<std::vector<std::string> const *, 2> const indices{&desc->spec.a_indices, &desc->spec.b_indices};
    for (std::size_t operand = 0; operand < node.inputs.size(); ++operand) {
        auto const *in = graph.find_tensor(node.inputs[operand]);
        if (in == nullptr) {
            continue;
        }
        if (!bind_operand(*indices[operand], in->spaces, letters)) {
            return declined("operands disagree about a letter's space",
                            fmt::format("einsum writing '{}' binds one letter to two spaces", graph.tensor(node.outputs[0]).name));
        }
    }

    auto inferred = detail::spaces_from_letters(desc->spec.c_indices, letters);
    return RuleResult{.applied = apply_inferred(graph, node.outputs[0], std::move(inferred), guard)};
}

/// Rule: ``C = α·A``. Scaling changes values, never what an axis ranges over,
/// so the output's slots are the input's slots.
RuleResult propagate_scale(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Scale) {
        return {};
    }
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return {};
    }

    auto const *in = graph.find_tensor(node.inputs[0]);
    if (in == nullptr || in->spaces.empty()) {
        return {};
    }
    return RuleResult{.applied = apply_inferred(graph, node.outputs[0], in->spaces, guard)};
}

/// Rule: a permute reorders axes, so the output's slots are the input's slots
/// in the order the node's own index letters name. Unlike the symmetry rule
/// there is no restriction on rank and none on beta: a space is a property of
/// an axis, not of the values in it, so accumulating onto a destination cannot
/// change what its axes range over.
RuleResult propagate_permute(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Permute && node.kind != OpKind::Transpose) {
        return {};
    }
    if (node.inputs.size() != 1 || node.outputs.size() != 1) {
        return {};
    }

    auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
    if (desc == nullptr) {
        return {};
    }

    auto const *in = graph.find_tensor(node.inputs[0]);
    if (in == nullptr || in->spaces.empty()) {
        return {};
    }

    std::vector<std::pair<std::string, SpaceId>> letters;
    if (!bind_operand(desc->a_indices, in->spaces, letters)) {
        return declined("operands disagree about a letter's space",
                        fmt::format("permute of '{}' binds one letter to two spaces", in->name));
    }

    auto inferred = detail::spaces_from_letters(desc->c_indices, letters);
    return RuleResult{.applied = apply_inferred(graph, node.outputs[0], std::move(inferred), guard)};
}

/// Rule: ``Y = α·X + β·Y`` and its friends. Every operand of a linear
/// combination indexes the same axes, so the output takes the spaces its
/// annotated inputs agree on, slot by slot. Inputs that disagree are a
/// cross-space bug in the source program; the pass declines and counts it, and
/// leaves the diagnosis to a validation pass.
RuleResult propagate_linear_combination(Graph &graph, Node const &node, InferGuard const &guard) {
    if (node.kind != OpKind::Axpby) {
        return {};
    }
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return {};
    }

    std::vector<SpaceId> agreed;
    for (auto const tid : node.inputs) {
        auto const *in = graph.find_tensor(tid);
        if (in == nullptr || in->spaces.empty()) {
            continue; // Unannotated inputs neither contribute nor block.
        }
        if (agreed.empty()) {
            agreed = in->spaces;
            continue;
        }
        if (agreed != in->spaces) {
            return declined("inputs of a linear combination disagree about their spaces",
                            fmt::format("axpby writing '{}' reads operands annotated differently", graph.tensor(node.outputs[0]).name));
        }
    }

    return RuleResult{.applied = apply_inferred(graph, node.outputs[0], std::move(agreed), guard)};
}

} // namespace

std::vector<std::string> SpacePropagation::explain() const {
    if (_num_inferred == 0) {
        return {};
    }
    return {fmt::format("SpacePropagation: inferred index spaces on {} tensor(s)", _num_inferred)};
}

void SpacePropagation::reset_stats() {
    _num_inferred = 0;
}

bool SpacePropagation::run(Graph &graph) {
    PassCounter const inferred_count{_num_inferred};
    graph.topological_sort();

    auto const &nodes = graph.nodes();

    // The soundness guard for this graph: writer counts plus the buffers child
    // sub-graphs reference.
    InferGuard const guard = EscapeAnalysis::over(graph);

    // One sweep in topological order is a fixpoint for forward propagation: a
    // node is visited after every node that writes its inputs, so an
    // intermediate annotated by this sweep is already annotated when its
    // consumer is examined and a chain resolves end to end in one run.
    for (auto const &node : nodes) {
        std::array<RuleResult, 4> const results{propagate_einsum(graph, node, guard), propagate_scale(graph, node, guard),
                                                propagate_permute(graph, node, guard), propagate_linear_combination(graph, node, guard)};
        for (auto const &result : results) {
            if (result.applied) {
                ++_num_inferred;
            }
            if (!result.skip_reason.empty()) {
                note_skip(result.skip_reason, result.skip_detail);
            }
        }
    }

    if (inferred_count.moved()) {
        EINSUMS_LOG_INFO("SpacePropagation: inferred index spaces on {} tensor(s)", _num_inferred);
        report(1, fmt::format("inferred index spaces on {} intermediate tensor(s)", _num_inferred));
    }

    // Analysis pass, never changes the node list.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
