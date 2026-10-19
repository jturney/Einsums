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
#include <utility>
#include <vector>

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
 * @param[in] order For a write backing, the order to store a new entry's axes in: the declared
 *            axis at each stored position (see @ref TensorEntry::stored_order). Empty, the default,
 *            leaves it to the graph, which stores a new entry so the boxes it moves are long
 *            contiguous runs of the file. Give it when another run will read the entry in boxes of
 *            its own. An entry that already exists has to be stored in this order to be reused.
 * @return The backing, for @ref compute_graph::Graph::annotate_backing.
 * @throws std::runtime_error when a read backing's file cannot be opened or holds no such entry.
 * @throws std::invalid_argument when a read backing's entry holds integers, which no graph tensor
 *         does, or, at annotation, when a write backing's entry already holds another shape or
 *         type, or is stored in another order than @p order asks for.
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
etn_backing(std::string path, std::string entry, bool write = false, std::vector<std::size_t> order = std::vector<std::size_t>{});

/**
 * @brief A ``.etn`` file named once, for the backings of its entries.
 *
 * Holds only the path: nothing is opened until a backing is built, and each backing reopens the
 * file by path at every slab as @ref etn_backing's do. A program that keeps several tensors in one
 * file names it here rather than in every backing.
 *
 * @code
 * ints = einsums.io.EtnFile("ints.etn")
 * B = graph.declare_tensor("Bvv", backing=ints.entry("Bvv"))
 * T = graph.declare_tensor("T2", dims, backing=ints.entry("T2", write=True))
 * @endcode
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE EtnFile {
  public:
    /// @brief Name the file.
    /// @param[in] path The file, stored as given; see @ref etn_backing.
    APIARY_EXPOSE explicit EtnFile(std::string path) : _path(std::move(path)) {}

    /// @brief The file's path.
    /// @return The path, as given.
    APIARY_EXPOSE APIARY_GETTER("path") [[nodiscard]] std::string const &path() const noexcept { return _path; }

    /**
     * @brief The backing for one entry of this file: @ref etn_backing with this file's path.
     * @param[in] entry The entry's name within the file.
     * @param[in] write False for an input the graph reads, true for a tensor it produces and stores.
     * @param[in] order For a write backing, the order to store a new entry's axes in.
     * @return The backing.
     * @throws std::runtime_error As @ref etn_backing.
     * @throws std::invalid_argument As @ref etn_backing.
     */
    APIARY_EXPOSE [[nodiscard]] compute_graph::TensorBacking entry(std::string entry, bool write = false,
                                                                   std::vector<std::size_t> order = std::vector<std::size_t>{}) const {
        return etn_backing(_path, std::move(entry), write, std::move(order));
    }

  private:
    std::string _path;
};

} // namespace tensor_io
EINSUMS_NAMESPACE_END()
