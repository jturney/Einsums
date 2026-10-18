//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file BackedWrites.hpp
/// @brief Who writes a backed tensor: an operation in memory, or a node writing its storage.
///
/// A write-backed tensor is produced either in memory, by ordinary operations, and then stored
/// whole, or a slab at a time straight into its storage. Which of the two a graph does decides
/// where its readers read from, whether it is filled from storage before them, and whether it is
/// stored after its last writer, so the verifier, Materialization and SlabStreaming all ask the
/// same two questions here.

#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;
struct Node;

namespace detail {

/**
 * @brief Whether an operation in @p graph or any graph below it writes @p object in memory.
 *
 * An operation writes the object when one of its outputs resolves, through views, to the buffer
 * of a handle over @p object. View nodes (which alias and write nothing), lifecycle nodes, and
 * nodes that read or write storage do not count.
 *
 * @param[in] graph Where to look, with every graph below it.
 * @param[in] object The tensor object, the identity a body and its parent share.
 * @param[in] skip_subtree A graph below @p graph to leave out, with everything below it, or null.
 * @param[in] skip_node A node to leave out, or null.
 */
[[nodiscard]] EINSUMS_EXPORT bool written_in_memory(Graph const &graph, void const *object, Graph const *skip_subtree = nullptr,
                                                    Node const *skip_node = nullptr);

/// @brief Whether a node writing storage in @p graph or below it writes @p backing's storage.
[[nodiscard]] EINSUMS_EXPORT bool written_to_storage(Graph const &graph, TensorBacking const &backing);

/**
 * @brief The disk descriptor every node that reads or writes @p backing's storage carries.
 *
 * One key per storage, whichever tensor and slab the node moves, so the scheduler orders every
 * read of the storage after every write to it, across loops, through the named-resource hazards.
 */
[[nodiscard]] EINSUMS_EXPORT DiskIODescriptor backing_io(TensorBacking const &backing, TensorId tensor, std::size_t bytes);

/// @brief The named-resource key of @p backing's storage: the one every node carrying backing_io's descriptor reads or writes.
[[nodiscard]] EINSUMS_EXPORT std::string storage_key(TensorBacking const &backing);

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
