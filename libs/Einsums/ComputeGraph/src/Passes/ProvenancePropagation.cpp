//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ProvenancePropagation.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Whether @p kind produces an output holding the same elements as its input, so a tag on the
/// input describes the output too.
///
/// Only the axis reorderings. See the header for why a View is not here: a slice of a Kronecker
/// delta is an identity only when it is a square block on the diagonal, and this pass has no way
/// to know that from the node alone.
bool preserves_identity(OpKind kind) {
    return kind == OpKind::Permute || kind == OpKind::Transpose || kind == OpKind::HPTTPermute;
}

} // namespace

std::vector<std::string> ProvenancePropagation::explain() const {
    if (_num_propagated == 0) {
        return {};
    }
    return {fmt::format("ProvenancePropagation: carried a provenance tag onto {} tensor(s)", _num_propagated)};
}

bool ProvenancePropagation::run(Graph &graph) {
    graph.topological_sort();

    // Forward over program order, so a chain of permutes carries a tag the whole way in one
    // sweep rather than needing one sweep per hop.
    for (auto const &node : graph.nodes()) {
        if (!preserves_identity(node.kind) || node.inputs.empty() || node.outputs.empty()) {
            continue;
        }

        TensorHandle const *source = graph.find_tensor(node.inputs[0]);
        TensorHandle       *target = graph.find_tensor(node.outputs[0]);
        if (source == nullptr || target == nullptr || !source->tag.valid()) {
            continue;
        }

        // A permute writing back into its own source says nothing new.
        if (graph.resolve_alias(node.inputs[0]) == graph.resolve_alias(node.outputs[0])) {
            continue;
        }

        if (target->tag.valid()) {
            if (target->tag != source->tag) {
                // A declaration is authoritative and this pass does not overrule one. Reported
                // rather than resolved, because a caller who tagged both ends differently means
                // one of the two, and guessing which would be worse than saying nothing.
                note_skip("the output already carries a different provenance tag",
                          fmt::format("node '{}' would carry '{}' onto '{}', which is declared '{}'", node.label, source->tag.name,
                                      target->name, target->tag.name));
            }
            continue;
        }

        // A tag only describes the WHOLE tensor, so an output of a different shape is not the
        // same object however the node is labelled. Cheap, and it catches a rank-reducing or
        // rank-changing permute that no other check here would.
        if (source->dims != target->dims && source->rank != target->rank) {
            note_skip("the output has a different shape from the tagged input",
                      fmt::format("node '{}': '{}' is rank {} and '{}' is rank {}", node.label, source->name, source->rank, target->name,
                                  target->rank));
            continue;
        }

        graph.annotate_tag(node.outputs[0], source->tag);
        ++_num_propagated;
        report(2, fmt::format("carried tag '{}' from '{}' to '{}' across {}", source->tag.name, source->name, target->name,
                              op_kind_name(node.kind)));
    }

    if (_num_propagated != 0) {
        report(1, fmt::format("carried a provenance tag onto {} tensor(s)", _num_propagated));
        EINSUMS_LOG_INFO("ProvenancePropagation: carried a provenance tag onto {} tensor(s)", _num_propagated);
    }
    // Annotations only; the node set is untouched, which the pass manager checks.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
