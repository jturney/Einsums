//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/UsageAnalysis.hpp>

#include <algorithm>
#include <unordered_set>

namespace einsums::compute_graph {

namespace {

size_t scan(std::vector<TensorUse> const &uses, bool want_write, bool ignore_alloc, bool ignore_free, bool include_subtree, bool last) {
    size_t found = TensorUsage::npos;
    for (auto const &u : uses) {
        if (want_write && !u.is_write)
            continue;
        if (ignore_alloc && u.kind == OpKind::Alloc)
            continue;
        if (ignore_free && u.kind == OpKind::Free)
            continue;
        if (!include_subtree && u.via_subtree)
            continue;
        if (!last)
            return u.pos;
        found = u.pos;
    }
    return found;
}

} // namespace

size_t TensorUsage::last_use(bool ignore_free, bool include_subtree) const {
    return scan(uses, /*want_write=*/false, /*ignore_alloc=*/false, ignore_free, include_subtree, /*last=*/true);
}

size_t TensorUsage::first_use(bool include_subtree) const {
    return scan(uses, /*want_write=*/false, /*ignore_alloc=*/false, /*ignore_free=*/false, include_subtree, /*last=*/false);
}

size_t TensorUsage::first_writer(bool ignore_alloc, bool include_subtree) const {
    return scan(uses, /*want_write=*/true, ignore_alloc, /*ignore_free=*/false, include_subtree, /*last=*/false);
}

size_t TensorUsage::last_writer(bool include_subtree) const {
    return scan(uses, /*want_write=*/true, /*ignore_alloc=*/false, /*ignore_free=*/false, include_subtree, /*last=*/true);
}

bool TensorUsage::has_writer_before(size_t pos) const {
    for (auto const &u : uses) {
        if (u.pos >= pos)
            return false; // uses are ascending; nothing earlier remains
        if (u.is_write)
            return true;
    }
    return false;
}

size_t TensorUsage::reads() const {
    return static_cast<size_t>(std::count_if(uses.begin(), uses.end(), [](TensorUse const &u) { return !u.is_write; }));
}

size_t TensorUsage::writes() const {
    return static_cast<size_t>(std::count_if(uses.begin(), uses.end(), [](TensorUse const &u) { return u.is_write; }));
}

UsageAnalysis UsageAnalysis::build(Graph &graph) {
    UsageAnalysis           ua;
    auto const             &nodes = graph.nodes();
    Graph::EffectiveIoCache cache;

    ua._node_count = nodes.size();

    for (size_t i = 0; i < nodes.size(); i++) {
        Node const &node = nodes[i];
        auto const  pos  = static_cast<std::uint32_t>(i);

        // The node's own lists first, then whatever effective-IO adds from
        // sub-graphs. Recording own-list ids into a set keeps the subtree
        // pass from double-reporting the same buffer at the same position.
        std::unordered_set<TensorId> own_read, own_written;
        for (auto const tid : node.inputs) {
            TensorId const owner = graph.resolve_alias(tid);
            if (own_read.insert(owner).second) {
                ua._table[owner].uses.push_back(TensorUse{pos, node.kind, /*is_write=*/false, /*via_subtree=*/false});
            }
        }
        for (auto const tid : node.outputs) {
            TensorId const owner = graph.resolve_alias(tid);
            if (own_written.insert(owner).second) {
                ua._table[owner].uses.push_back(TensorUse{pos, node.kind, /*is_write=*/true, /*via_subtree=*/false});
            }
        }

        if (node.kind == OpKind::Loop || node.kind == OpKind::Conditional) {
            auto [eff_in, eff_out] = graph.effective_io_cached(node, cache);
            for (auto const tid : eff_in) {
                TensorId const owner = graph.resolve_alias(tid);
                if (own_read.insert(owner).second) {
                    ua._table[owner].uses.push_back(TensorUse{pos, node.kind, /*is_write=*/false, /*via_subtree=*/true});
                }
            }
            for (auto const tid : eff_out) {
                TensorId const owner = graph.resolve_alias(tid);
                if (own_written.insert(owner).second) {
                    ua._table[owner].uses.push_back(TensorUse{pos, node.kind, /*is_write=*/true, /*via_subtree=*/true});
                }
            }
        }
    }

    return ua;
}

TensorUsage const *UsageAnalysis::find(Graph const &graph, TensorId id) const {
    return find_owner(graph.resolve_alias(id));
}

TensorUsage const *UsageAnalysis::find_owner(TensorId owner) const {
    auto it = _table.find(owner);
    return it == _table.end() ? nullptr : &it->second;
}

} // namespace einsums::compute_graph
