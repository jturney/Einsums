//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Count value-writers across @p graph and every descendant, keyed by the
/// underlying pointer. Recursive because a loop inside a conditional inside a
/// loop is three levels of the same question, and the pointer is what survives
/// the crossing: each sub-graph keeps its own tensor table and its own ids.
void count_subtree_writers(Graph const &graph, std::unordered_map<void const *, int> &writers) {
    for (auto const &node : graph.nodes()) {
        if (is_lifecycle(node.kind)) {
            continue;
        }
        for (auto const tid : node.outputs) {
            // Resolve first: a write through a view of T is a write to T, and
            // counting the view object instead would report T as single-writer
            // while a different node overwrote it every iteration.
            auto const hit = graph.tensors_map().find(graph.resolve_alias(tid));
            if (hit != graph.tensors_map().end() && hit->second.tensor_ptr != nullptr) {
                writers[hit->second.tensor_ptr]++;
            }
        }
    }
    graph.for_each_subgraph([&](Graph const &sub) { count_subtree_writers(sub, writers); });
}

} // namespace

std::string_view escape_reason(Escape reason) {
    switch (reason) {
    case Escape::Dissolvable:
        return "nothing outside the region can observe it";
    case Escape::UserOwned:
        return "the tensor is user-owned, not a graph intermediate";
    case Escape::WrittenOutside:
        return "a node outside the region writes it";
    case Escape::ReadOutside:
        return "a node outside the region reads it";
    case Escape::AliasedFromOutside:
        return "another tensor over the same buffer escapes the region";
    case Escape::TouchedBySubgraph:
        return "a loop body or conditional branch touches its buffer";
    case Escape::Unknown:
        return "the graph does not know this tensor";
    }
    return "unclassified";
}

EscapeAnalysis EscapeAnalysis::over(Graph const &graph) {
    EscapeAnalysis out;
    out._graph = &graph;

    for (auto const &node : graph.nodes()) {
        bool const lifecycle = is_lifecycle(node.kind);
        for (auto const tid : node.outputs) {
            auto const root = graph.resolve_alias(tid);
            out._any_writers[root].push_back(node.id);
            if (!lifecycle) {
                out._value_writers[root].push_back(node.id);
            }
        }
        for (auto const tid : node.inputs) {
            auto const root = graph.resolve_alias(tid);
            out._any_readers[root].push_back(node.id);
            if (!lifecycle) {
                out._value_readers[root].push_back(node.id);
            }
        }
    }

    // Every id grouped by the buffer it resolves to. Built from the tensor table
    // rather than from the node list, because a view nothing has used yet still
    // aliases the buffer and a region that dissolved it would be wrong the moment
    // a later pass gave it a reader.
    for (auto const &[id, handle] : graph.tensors_map()) {
        out._by_root[graph.resolve_alias(id)].push_back(id);
    }
    // tensors_map is unordered, so sort. A region rewrite driven off an
    // unordered walk is a rewrite that varies between runs, and the Kahn FIFO
    // bug is this module's standing reminder that "some valid order" and "the
    // same valid order every time" are different requirements.
    for (auto &[root, ids] : out._by_root) {
        std::ranges::sort(ids);
    }

    count_subtree_writers(graph, out._subtree_writers);
    graph.collect_subtree_referenced_ptrs(out._subtree_ptrs);
    return out;
}

int EscapeAnalysis::writer_count(TensorId id) const {
    auto const hit = _value_writers.find(_graph->resolve_alias(id));
    return hit == _value_writers.end() ? 0 : static_cast<int>(hit->second.size());
}

int EscapeAnalysis::subtree_writer_count(TensorId id) const {
    auto const handle = _graph->tensors_map().find(_graph->resolve_alias(id));
    if (handle == _graph->tensors_map().end() || handle->second.tensor_ptr == nullptr) {
        return 0;
    }
    auto const hit = _subtree_writers.find(handle->second.tensor_ptr);
    return hit == _subtree_writers.end() ? 0 : hit->second;
}

bool EscapeAnalysis::touched_by_subtree(TensorId id) const {
    auto const handle = _graph->tensors_map().find(_graph->resolve_alias(id));
    if (handle == _graph->tensors_map().end()) {
        return true; // cannot prove otherwise
    }
    // A tensor with no pointer is a deferred shell whose storage is not attached
    // yet, so there is nothing to compare against the subtree's pointers. Report
    // it as untouched: the pointer set cannot contain it either, and reporting
    // "touched" would decline every rewrite over a graph built from declare_*,
    // which is every graph the save/load path produces.
    if (handle->second.tensor_ptr == nullptr) {
        return false;
    }
    return _subtree_ptrs.count(handle->second.tensor_ptr) != 0;
}

bool EscapeAnalysis::stable(TensorId id) const {
    return writer_count(id) == 1 && !touched_by_subtree(id);
}

std::vector<TensorId> EscapeAnalysis::aliases_of(TensorId id) const {
    auto const hit = _by_root.find(_graph->resolve_alias(id));
    return hit == _by_root.end() ? std::vector<TensorId>{id} : hit->second;
}

Escape EscapeAnalysis::classify(TensorId id, std::unordered_set<NodeId> const &region) const {
    auto const self = _graph->tensors_map().find(id);
    if (self == _graph->tensors_map().end()) {
        return Escape::Unknown;
    }
    if (!self->second.is_intermediate) {
        return Escape::UserOwned;
    }

    auto const outside = [&region](std::unordered_map<TensorId, std::vector<NodeId>> const &map, TensorId root) {
        auto const hit = map.find(root);
        if (hit == map.end()) {
            return false;
        }
        return std::ranges::any_of(hit->second, [&region](NodeId nid) { return region.count(nid) == 0; });
    };

    auto const root = _graph->resolve_alias(id);

    // A sibling view of the same buffer that the user owns takes the whole buffer
    // out of reach, whatever the nodes do: the caller holds a tensor and expects a
    // value in it, and dissolving the writer would leave it holding whatever was
    // there before.
    for (auto const alias : aliases_of(id)) {
        if (alias == id) {
            continue;
        }
        auto const other = _graph->tensors_map().find(alias);
        if (other != _graph->tensors_map().end() && !other->second.is_intermediate) {
            return Escape::AliasedFromOutside;
        }
    }

    // VALUE writers and readers, so a lifecycle node outside the region does not
    // make a tensor escape it. That distinction is load-bearing rather than
    // tidy: `Graph::create_*` and `declare_*` put an Alloc ahead of the first
    // real write, Alloc is not raisable and so is never inside a region, and
    // counting its mention would make every graph-owned intermediate
    // undissolvable - which is every intermediate a rewrite exists to dissolve.
    //
    // The consequence is worth stating because a caller has to handle it: a
    // rewrite that dissolves an intermediate leaves that intermediate's Alloc
    // and Free behind, naming a tensor nothing writes any more. They are dead
    // rather than wrong, and `DeadNodeElimination` removes them; a region
    // rewrite must not assume it owns them, because they sit outside it.
    //
    // Writes before reads: both decline, but an outside writer means the region
    // does not own this value at all, while an outside reader usually means the
    // region merely needs to grow, and the two call for different responses.
    if (outside(_value_writers, root)) {
        return Escape::WrittenOutside;
    }
    if (outside(_value_readers, root)) {
        return Escape::ReadOutside;
    }
    if (touched_by_subtree(id)) {
        return Escape::TouchedBySubgraph;
    }
    return Escape::Dissolvable;
}

EINSUMS_NAMESPACE_END(compute_graph)
