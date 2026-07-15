//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph {

class Graph;

/**
 * @brief One read or write of a tensor buffer by one node.
 *
 * Position-keyed against the graph's current node order (position IS program
 * order in this IR). @c via_subtree distinguishes uses that only exist through
 * a control-flow node's sub-graph (Loop bodies, Conditional branches, found
 * via effective-IO expansion) from uses recorded on the node's own
 * input/output lists; consumers that historically scanned raw lists filter on
 * it to keep their exact semantics.
 */
struct TensorUse {
    std::uint32_t pos;         ///< Node position in the current (sorted) node order
    OpKind        kind;        ///< The using node's kind (for lifecycle filtering: Alloc/Free/Materialize/Initialize)
    bool          is_write;    ///< True if the node writes the buffer, false if it reads
    bool          via_subtree; ///< True if the use surfaced only through effective-IO (Loop/Conditional body)
};

/**
 * @brief Every use of one owning buffer, in ascending position order.
 *
 * Keyed by the OWNER TensorId (alias chains resolved), so a write through a
 * view and a read of its parent land in the same entry.
 *
 * The filter parameters on the query helpers exist because the passes this
 * table replaced disagreed on what "counts": FreeInsertion ignores Free nodes
 * when computing last use (a Free's input is a scheduling edge, not a use) and
 * ignores Alloc when finding the first writer (creation, not data); the
 * program-order validator wants every use. Encoding the policy at the call
 * site keeps one canonical scan serving all of them.
 */
struct EINSUMS_EXPORT TensorUsage {
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    std::vector<TensorUse> uses; ///< Ascending by pos; one entry per (node, direction)

    /// Position of the last read-or-write. @p ignore_free skips Free nodes;
    /// @p include_subtree admits uses inside Loop/Conditional bodies.
    [[nodiscard]] size_t last_use(bool ignore_free = false, bool include_subtree = true) const;

    /// Position of the first read-or-write (same filters as last_use).
    [[nodiscard]] size_t first_use(bool include_subtree = true) const;

    /// Position of the first writing node. @p ignore_alloc skips Alloc nodes
    /// (buffer creation, not a data write).
    [[nodiscard]] size_t first_writer(bool ignore_alloc = false, bool include_subtree = true) const;

    /// Position of the last writing node.
    [[nodiscard]] size_t last_writer(bool include_subtree = true) const;

    /// True if any writer sits at a position strictly below @p pos.
    [[nodiscard]] bool has_writer_before(size_t pos) const;

    /// Number of reads / writes (all kinds, subtree included).
    [[nodiscard]] size_t reads() const;
    [[nodiscard]] size_t writes() const;
};

/**
 * @brief Graph-wide reader/writer/liveness index: one forward scan, shared by
 *        every pass that used to build its own.
 *
 * Before this existed, ~14 passes each rebuilt some slice of this table per
 * run (several inside per-candidate loops, turning O(n) questions into O(n²)
 * scans), with three different alias conventions between them. Build once,
 * query everywhere.
 *
 * Canonical semantics: TensorIds are resolved through Graph::resolve_alias,
 * control-flow nodes contribute their sub-graphs' IO via effective-IO
 * expansion (marked @c via_subtree). Positions refer to the node order at
 * build time - the table is invalidated by the same mutation-declaration
 * points as DependencyInfo (add_node / mark_sorted / topological_sort), so
 * the cached copy on Graph is only valid until the next declared mutation.
 */
class EINSUMS_EXPORT UsageAnalysis {
  public:
    /// Scan @p graph and build the full table (no caching; see Graph::usage()
    /// for the cached accessor). Also usable standalone for before/after
    /// snapshots (the pass program-order validator).
    static UsageAnalysis build(Graph &graph);

    /// Usage of the buffer @p id belongs to (alias chain resolved here), or
    /// nullptr if the buffer is never used by any node.
    [[nodiscard]] TensorUsage const *find(Graph const &graph, TensorId id) const;

    /// Direct access when @p owner is already alias-resolved.
    [[nodiscard]] TensorUsage const *find_owner(TensorId owner) const;

    /// Owner-keyed table (alias chains resolved).
    [[nodiscard]] std::unordered_map<TensorId, TensorUsage> const &table() const { return _table; }

    /// Number of nodes the table was built from (staleness defense).
    [[nodiscard]] size_t node_count() const { return _node_count; }

  private:
    std::unordered_map<TensorId, TensorUsage> _table;
    size_t                                    _node_count{0};
};

} // namespace einsums::compute_graph
