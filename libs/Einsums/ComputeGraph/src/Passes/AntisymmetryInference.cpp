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
    bool                             overwrites{false}; ///< destination prefactor is zero
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
    // The soundness guard SymmetryPropagation uses, shared rather than counted a
    // second time: exactly one value-writer in this graph, and no descendant
    // sub-graph touching the buffer. Without it a later overwrite could destroy
    // the structure this pass just promised.
    auto const guard = EscapeAnalysis::over(graph);

    for (auto const &node : graph.nodes()) {
        auto const site = read_operator_site(node);
        if (!site.has_value() || node.outputs.size() != 1) {
            continue;
        }
        ++_num_candidates;

        if (!site->overwrites) {
            note_skip("the node accumulates, so its output is the previous contents plus an antisymmetric part",
                      fmt::format("node #{}", node.id));
            continue;
        }
        if (!all_groups_singleton(site->operators)) {
            note_skip("a group names more than one letter, and a coset sum carries no antisymmetry of its own",
                      fmt::format("node #{}", node.id));
            continue;
        }

        auto const desc = antisymmetry_of(*site);
        if (!desc.has_value()) {
            note_skip("the operator's letters do not each name exactly one addressable output axis", fmt::format("node #{}", node.id));
            continue;
        }

        TensorId const out    = graph.resolve_alias(node.outputs[0]);
        auto          *handle = graph.find_tensor(out);
        if (handle == nullptr || !handle->is_intermediate) {
            note_skip("the destination is not a graph-owned intermediate", fmt::format("node #{}", node.id));
            continue;
        }
        if (!guard.stable(out)) {
            note_skip("the destination is written more than once, or by a child sub-graph", fmt::format("node #{}", node.id));
            continue;
        }
        if (handle->symmetry_hint != nullptr && *handle->symmetry_hint == *desc) {
            continue; // already known, and re-running must not count it twice
        }

        handle->symmetry_hint = std::make_shared<SymmetryDescriptor>(*desc);
        ++_num_tagged;
    }

    // Annotation only; the node list is untouched.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
