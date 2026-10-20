//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp>
#include <Einsums/ComputeGraph/SymmetryRules.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <memory>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

// Soundness context for an inference run on one graph. A symmetry tag is
// only ever a *guarantee* about the tensor's final contents, so we infer
// it only when nothing can invalidate it after the producing op:
//   - the tensor has exactly one writer in this graph (no later overwrite
//     could destroy the inferred structure), and
//   - the tensor isn't referenced by a child sub-graph (a nested loop /
//     conditional body could write it without this graph's node list
//     showing it, a Loop node doesn't list its body's writes).
// Without these guards a stale tag could claim symmetry the data no longer
// has. They make the pass strictly conservative and therefore safe to
// recurse into loop bodies.
// EscapeAnalysis::stable answers both in one call: exactly one value-writer in this graph, and no
// descendant sub-graph touching the buffer. Shared with LoopInvariantHoisting
// and the region framework so the derivations cannot disagree.

/// Try to push an inferred descriptor to a graph-owned tensor handle.
/// Returns true if the descriptor was applied (either taking on a new value
/// or overwriting ``nullptr`` on the backing tensor).
bool apply_inferred(Graph &graph, TensorId out_tid, SymmetryDescriptor desc, EscapeAnalysis const &guard) {
    auto &handle = graph.tensor(out_tid);
    if (!handle.is_intermediate)
        return false; // Never mutate user-owned tensor state.
    if (!guard.stable(out_tid))
        return false; // Could be overwritten later / by a child body, don't tag.
    // Skip if the tensor already has an equivalent descriptor, avoids
    // redundant work and spurious "new inference" reports on re-runs.
    if (handle.symmetry_hint && *handle.symmetry_hint == desc)
        return false;

    handle.symmetry_hint = std::make_shared<SymmetryDescriptor>(desc);
    if (handle.set_symmetry_fn)
        handle.set_symmetry_fn(std::move(desc));
    return true;
}

} // namespace

std::vector<std::string> SymmetryPropagation::explain() const {
    if (_num_inferred == 0) {
        return {};
    }
    return {fmt::format("SymmetryPropagation: inferred symmetry on {} tensor(s)", _num_inferred)};
}

void SymmetryPropagation::reset_stats() {
    _num_inferred = 0;
}

bool SymmetryPropagation::transform(Graph &graph) {
    PassCounter const inferred{_num_inferred};
    graph.topological_sort();

    auto const &nodes = graph.nodes();

    // Build the soundness guard for this graph: writer counts + the set of
    // tensor pointers referenced by child sub-graphs. Only single-writer,
    // not-used-by-a-child tensors get tagged.
    // Lifecycle nodes (Alloc/Free/Materialize/Initialize) list the tensor as an
    // output but don't write a *value* that could invalidate an inferred
    // symmetry; a freshly created or zeroed tensor is then filled by exactly one
    // real op. The shared analysis excludes them for the same reason.
    auto const guard = EscapeAnalysis::over(graph);

    // The rules themselves live in SymmetryRules.hpp, shared with every pass that asks what a
    // tensor's symmetry is; this pass adds only the guard and the tagging.
    for (auto const &node : nodes) {
        if (node.outputs.size() != 1)
            continue;
        if (auto desc = output_symmetry(graph, node); desc && apply_inferred(graph, node.outputs[0], std::move(*desc), guard))
            ++_num_inferred;
    }

    if (inferred.moved()) {
        EINSUMS_LOG_INFO("SymmetryPropagation: inferred symmetry on {} tensor(s)", _num_inferred);
        report(1, fmt::format("inferred symmetry on {} intermediate tensor(s) (enables rank-2 BLAS dispatch)", _num_inferred));
    }

    // Analysis pass, never changes the node list.
    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
