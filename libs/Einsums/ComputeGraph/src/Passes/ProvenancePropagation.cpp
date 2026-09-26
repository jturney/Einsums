//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ProvenancePropagation.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
#include <cstddef>
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

/// Whether @p node overwrites its output with exactly its source, relabelled: the LIVE alpha is one,
/// the live beta zero, and no permutation operator turns it into a signed sum.
bool is_plain_copy(Node const &node) {
    if (auto const *desc = node.op_data.get_if<PermuteDescriptor>()) {
        bool const unscaled = desc->params != nullptr
                                  ? is_one(desc->params->alpha) && is_zero(desc->params->beta)
                                  : desc->alpha == std::complex<double>{1.0, 0.0} && desc->beta == std::complex<double>{};
        return unscaled && desc->operators.empty();
    }
    // A transpose carries no scalars; anything else of these kinds is not provably a copy.
    return node.kind == OpKind::Transpose;
}

/// Give every handle of @p graph and its descendants naming the buffer @p ptr the tag @p tag,
/// where it has none. Returns how many were annotated.
std::size_t carry_into(Graph &graph, void const *ptr, ProvenanceTag const &tag) {
    std::size_t           carried = 0;
    std::vector<TensorId> targets;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == ptr && !handle.tag.valid()) {
            targets.push_back(id);
        }
    }
    // Sorted, because tensors_map is unordered and an annotation order that varies between runs
    // makes every pass reading it vary with it.
    std::ranges::sort(targets);
    for (TensorId const id : targets) {
        graph.annotate_tag(id, tag);
        ++carried;
    }
    graph.for_each_subgraph([&](Graph &child) { carried += carry_into(child, ptr, tag); });
    return carried;
}

} // namespace

std::vector<std::string> ProvenancePropagation::explain() const {
    if (_num_propagated == 0) {
        return {};
    }
    return {fmt::format("ProvenancePropagation: carried a provenance tag onto {} tensor(s)", _num_propagated)};
}

bool ProvenancePropagation::run(Graph &graph) {
    PassCounter const propagated{_num_propagated};
    graph.topological_sort();

    // A body's handle for a caller's tensor is the SAME tensor, not a view of it, so a tag
    // declared on the enclosing graph describes it. Carried here because the driver runs a parent
    // before it descends: without it a caller who tags an amplitude on the graph and captures the
    // iteration as a loop body has tagged something no pass reading that body can see, which is
    // exactly the shape the region rewrites descended into bodies to serve. Declared beats
    // inherited, as everywhere else in this pass: a body that already carries a tag keeps it.
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (!handle.tag.valid() || handle.tensor_ptr == nullptr) {
            continue;
        }
        ProvenanceTag const tag = handle.tag;
        void const *const   ptr = handle.tensor_ptr;
        graph.for_each_subgraph([&](Graph &child) { _num_propagated += carry_into(child, ptr, tag); });
    }

    // Forward over program order, so a chain of permutes carries a tag the whole way in one
    // sweep rather than needing one sweep per hop.
    auto const copy_writers = EscapeAnalysis::over(graph);
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
        if (graph.buffer_of(node.inputs[0]) == graph.buffer_of(node.outputs[0])) {
            continue;
        }

        // A tag says what the WHOLE tensor is, so only a plain copy carries it: an overwrite by
        // exactly the source (no scale, no accumulation, no permutation operator), into a tensor
        // nothing else writes, here or in a body. A scaled or accumulated delta is not a delta,
        // and one a later node overwrites stops being one; DeltaElimination acted on all three.
        if (!is_plain_copy(node) || copy_writers.writer_count(node.outputs[0]) != 1 ||
            copy_writers.subtree_writer_count(node.outputs[0]) != 1) {
            note_skip("the node is not a plain copy into a tensor it alone writes", fmt::format("node '{}'", node.label));
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
        // same object however the node is labelled. A relabeling keeps the rank and permutes the
        // extents; anything else (a rank change, a different size along some axis) is not one.
        auto sorted_dims = [](std::vector<std::size_t> dims) {
            std::ranges::sort(dims);
            return dims;
        };
        if (source->rank != target->rank || sorted_dims(source->dims) != sorted_dims(target->dims)) {
            note_skip("the output has a different shape from the tagged input",
                      fmt::format("node '{}': '{}' is rank {} and '{}' is rank {}", node.label, source->name, source->rank, target->name,
                                  target->rank));
            continue;
        }

        graph.annotate_tag(node.outputs[0], source->tag);
        ++_num_propagated;
        report(2, fmt::format("carried tag '{}' from '{}' to '{}' across {}", source->tag.name, source->name, target->name, node.kind));
    }

    // This graph's own count: the member is the total over every sub-graph visited so far.
    if (propagated.moved()) {
        report(1, fmt::format("carried a provenance tag onto {} tensor(s)", propagated.delta()));
        EINSUMS_LOG_INFO("ProvenancePropagation: carried a provenance tag onto {} tensor(s)", propagated.delta());
    }
    // Annotations only; the node set is untouched, which the pass manager checks.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
