//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

bool IOPrefetch::run(Graph &graph) {
    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        _num_prefetched = 0;
        return false;
    }

    _num_prefetched = 0;

    // Collect DiskRead positions in reverse order so that moves don't
    // invalidate positions of earlier entries in our list.
    std::vector<size_t> read_positions;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (nodes[idx].kind == OpKind::DiskRead) {
            read_positions.push_back(idx);
        }
    }

    if (read_positions.empty())
        return false;

    // Process from last to first so index shifts don't affect earlier entries.
    for (unsigned long pos : std::views::reverse(read_positions)) {
        if (pos == 0)
            continue; // Already at the beginning

        // Build writer map fresh (positions may have shifted from previous moves).
        std::unordered_map<TensorId, size_t> last_writer;
        for (size_t idx = 0; idx < nodes.size(); idx++) {
            for (auto tid : nodes[idx].outputs) {
                last_writer[tid] = idx;
            }
        }

        // Find the DiskRead — it may have shifted due to earlier moves.
        // Search for it by checking kind at the expected position.
        // If it moved, scan forward to find it.
        while (pos < nodes.size() && nodes[pos].kind != OpKind::DiskRead)
            pos++;
        if (pos >= nodes.size())
            continue;

        // Compute earliest legal position: after all predecessors.
        size_t earliest = 0;
        for (auto tid : nodes[pos].inputs) {
            auto wit = last_writer.find(tid);
            if (wit != last_writer.end() && wit->second < pos) {
                earliest = std::max(earliest, wit->second + 1);
            }
        }

        if (earliest < pos) {
            EINSUMS_LOG_INFO("IOPrefetch: moved '{}' from position {} to {} (prefetched by {} nodes)", nodes[pos].label, pos, earliest,
                             pos - earliest);

            std::rotate(nodes.begin() + static_cast<ptrdiff_t>(earliest), nodes.begin() + static_cast<ptrdiff_t>(pos),
                        nodes.begin() + static_cast<ptrdiff_t>(pos) + 1);
            _num_prefetched++;
        }
    }

    if (_num_prefetched > 0) {
        graph.mark_sorted();
    }

    return _num_prefetched > 0;
}

} // namespace einsums::compute_graph::passes
