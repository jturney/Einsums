//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

bool FreeInsertion::run(Graph &graph) {
    _num_freed = 0;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    if (nodes.empty())
        return false;

    // Find the last node that references each intermediate tensor.
    // A tensor is "referenced" if it appears in a node's inputs or outputs.
    std::unordered_map<TensorId, size_t> last_use; // TensorId → last node index

    // Resolve a TensorId through any chain of aliases to the underlying owner.
    // Aliases (View outputs) read/write through the parent's storage, so a use
    // of the alias is logically a use of the owner — extend the owner's
    // lifetime accordingly.
    auto resolve_owner = [&](TensorId id) {
        for (int hops = 0; hops < 32; ++hops) {
            auto it = tensors.find(id);
            if (it == tensors.end() || it->second.aliases == 0)
                return id;
            id = it->second.aliases;
        }
        return id;
    };

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        for (auto tid : nodes[idx].inputs) {
            TensorId const owner = resolve_owner(tid);
            auto           it    = tensors.find(owner);
            if (it != tensors.end() && it->second.is_intermediate) {
                last_use[owner] = idx;
            }
        }
        for (auto tid : nodes[idx].outputs) {
            TensorId const owner = resolve_owner(tid);
            auto           it    = tensors.find(owner);
            if (it != tensors.end() && it->second.is_intermediate) {
                last_use[owner] = idx;
            }
        }
    }

    if (last_use.empty())
        return false;

    // Build insertion list: for each intermediate, insert a Free node after its last use.
    struct Insertion {
        size_t   position; // Insert AFTER this node index
        TensorId tid;
        size_t   bytes;
    };
    std::vector<Insertion> insertions;

    for (auto const &[tid, last_idx] : last_use) {
        auto it = tensors.find(tid);
        if (it == tensors.end())
            continue;

        auto const &handle = it->second;

        // Only free intermediates that have a release function and are above size threshold.
        // Small tensors are kept alive to avoid alloc/free overhead in re-executed graphs.
        if (!handle.is_intermediate || !handle.release_fn)
            continue;
        if (handle.total_bytes() < _min_bytes)
            continue;

        // Aliasing tensors (Views) don't own their storage — the parent does.
        // Freeing the alias would tear down the view object but leave the
        // parent's data alone; we'd then crash on the next re-execute when
        // the View executor tries to re-emplace into a destroyed holder.
        if (handle.aliases != 0)
            continue;

        // Don't free tensors that are already Free'd (avoid duplicates on re-run)
        bool already_freed = false;
        for (size_t idx = last_idx + 1; idx < nodes.size(); idx++) {
            if (nodes[idx].kind == OpKind::Free) {
                for (auto in_tid : nodes[idx].inputs) {
                    if (in_tid == tid) {
                        already_freed = true;
                        break;
                    }
                }
            }
            if (already_freed)
                break;
        }
        if (already_freed)
            continue;

        insertions.push_back({.position = last_idx, .tid = tid, .bytes = handle.total_bytes()});
    }

    if (insertions.empty())
        return false;

    // Sort by position descending so inserts don't shift earlier positions.
    std::ranges::sort(insertions, [](Insertion const &a, Insertion const &b) { return a.position > b.position; });

    for (auto const &ins : insertions) {
        auto it = tensors.find(ins.tid);
        if (it == tensors.end())
            continue;

        auto const  &handle = it->second;
        auto         rel_fn = handle.release_fn;
        size_t const bytes  = ins.bytes;

        Node free_node;
        free_node.kind    = OpKind::Free;
        free_node.label   = fmt::format("free({})", handle.name);
        free_node.inputs  = {ins.tid}; // Dependency: runs after last consumer
        free_node.outputs = {};

        free_node.execute = [rel_fn, bytes, name = handle.name]() {
            if (rel_fn) {
                rel_fn();
                EINSUMS_LOG_DEBUG("FreeInsertion: released '{}' ({} bytes)", name, bytes);
            }
        };

        free_node.estimated_bytes = bytes;

        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(ins.position + 1), std::move(free_node));
        _num_freed++;
    }

    if (_num_freed > 0) {
        graph.mark_sorted();
        EINSUMS_LOG_INFO("FreeInsertion: inserted {} Free nodes", _num_freed);
    }

    return _num_freed > 0;
}

} // namespace einsums::compute_graph::passes
