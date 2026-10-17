//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Backing.hpp
/// @brief A ``.etn`` entry as the storage behind a tensor a graph reads but never holds whole.

#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <string>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN()
namespace APIARY_MODULE("io") tensor_io {

/// The kind name an ``.etn`` backing carries and a saved graph records.
inline constexpr std::string_view etn_backing_kind = "tensor_io.etn";

/**
 * @brief The backing for one entry of a ``.etn`` file.
 * @param[in] path The file. Stored as given: a relative path is resolved against the working
 *            directory each time the graph reads or writes it, and again when a saved graph is
 *            loaded.
 * @param[in] entry The entry's name within the file.
 * @param[in] write False (the default) for an input the graph reads: the entry must exist, and
 *            its extents and element type are read from it now. True for a tensor the graph
 *            produces and stores: the file and entry are created, or reused when the entry already
 *            holds a tensor of the same shape and type, when the backing is annotated on its
 *            tensor, which supplies that shape and type.
 * @return The backing, for @ref compute_graph::Graph::annotate_backing.
 * @throws std::runtime_error when a read backing's file cannot be opened or holds no such entry.
 * @throws std::invalid_argument when a read backing's entry holds integers, which no graph tensor
 *         does, or, at annotation, when a write backing's entry already holds another shape or
 *         type.
 *
 * Each read or write opens the file, checks that the entry still has the extents and element
 * type it had when the backing was built, and moves the slab: a file rewritten between replays
 * is used afresh, and one rewritten with a different shape is refused rather than read or
 * written as the wrong elements.
 *
 * @code
 * auto &B = graph.declare_runtime_tensor<double>("B_vv", {naux, nvir, nvir});
 * graph.annotate_backing(B, tensor_io::etn_backing("ints.etn", "B_vv"));         // an input
 * auto &T = graph.declare_runtime_tensor<double>("T2", {nocc, nocc, nvir, nvir});
 * graph.annotate_backing(T, tensor_io::etn_backing("amps.etn", "T2", true));     // produced and stored
 * @endcode
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_RENAME("etn") [[nodiscard]] EINSUMS_EXPORT compute_graph::TensorBacking
                                                                etn_backing(std::string path, std::string entry, bool write = false);

} // namespace tensor_io
EINSUMS_NAMESPACE_END()
