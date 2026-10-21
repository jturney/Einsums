//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file RunMetrics.hpp
/// @brief What one replay of a graph did, measured as it ran: time per node, bytes per backing,
///        memory held.

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief One node's share of a replay.
 *
 * A node in a loop body ran once per iteration: @ref ms is the sum and @ref runs the count. A
 * control-flow node's own time includes its body's, whose nodes have entries of their own.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") NodeRun {
    /// The name of the graph or body the node is in.
    APIARY_EXPOSE APIARY_READONLY std::string graph;
    /// The node's id within that graph.
    APIARY_EXPOSE APIARY_READONLY NodeId id{0};
    /// The node's kind.
    APIARY_EXPOSE APIARY_READONLY OpKind kind{OpKind::Custom};
    /// The node's label.
    APIARY_EXPOSE APIARY_READONLY std::string label;
    /// Wall-clock milliseconds, summed over its runs.
    APIARY_EXPOSE APIARY_READONLY double ms{0.0};
    /// The thread width of its last run; 0 when the executor did not place it at a width.
    APIARY_EXPOSE APIARY_READONLY unsigned width{0};
    /// How many times it ran in the replay.
    APIARY_EXPOSE APIARY_READONLY std::size_t runs{0};
};

/**
 * @brief What one backing's storage moved in a replay, counted where it moved.
 *
 * Every read and write of a backing goes through the graph's fill and store, whichever pass put
 * them there: a whole load, a slab a contraction streams, a box a loop's view names, a whole
 * store after the last writer, a slab a streamed producer writes. A read or a write counts once
 * per call, whatever its size.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") BackingRun {
    /// The backed tensor's name.
    APIARY_EXPOSE APIARY_READONLY std::string tensor;
    /// The storage, as @ref TensorBacking::describe names it.
    APIARY_EXPOSE APIARY_READONLY std::string storage;
    /// Bytes read from the storage.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t bytes_read{0};
    /// Bytes written to it.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t bytes_written{0};
    /// Reads, each a slab or a whole load.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t reads{0};
    /// Writes, each a slab or a whole store.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t writes{0};
    /// Publishes of an update backing: one per replay that reached its end.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t publishes{0};
};

/**
 * @brief What the last replay of a graph did, as @ref Graph::last_run reports it.
 *
 * Measured, beside what the passes planned: @ref peak_bytes is what
 * @ref passes::MemoryPlanning::peak_memory predicts, and a backing's bytes are what
 * SlabStreaming's report estimates.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") RunMetrics {
    /// Wall-clock milliseconds of the whole replay.
    APIARY_EXPOSE APIARY_READONLY double wall_ms{0.0};
    /// Every node the replay ran, the graph's own and its bodies'. Each graph's nodes are in the
    /// order they first ran, and a body's come before those of the graph holding it, since a body's
    /// replay finishes first.
    APIARY_EXPOSE APIARY_READONLY std::vector<NodeRun> nodes;
    /// One entry per backing in the graph and its bodies, in name order.
    APIARY_EXPOSE APIARY_READONLY std::vector<BackingRun> backings;
    /// The most bytes of graph-managed tensors held at once: deferred tensors the graph allocated
    /// and freed, plus @ref arena_bytes. Not the process's memory: BLAS scratch is outside it, and
    /// the caller's own tensors are in @ref resident_bytes.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t peak_bytes{0};
    /// The arena MemoryPlanning placed intermediates in, held for the whole replay; 0 without one.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t arena_bytes{0};
    /// What the replay held from its start to its end outside @ref peak_bytes: the caller's
    /// tensors and any intermediate given storage outside a lifecycle node (@ref
    /// Graph::resident_bytes). The program held @ref peak_bytes plus this at its peak.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t resident_bytes{0};
};

EINSUMS_NAMESPACE_END(compute_graph)
