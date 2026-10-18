//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Entries.hpp
/// @brief Copy, rename and swap the entries of ``.etn`` files, now or as a step of a graph.
///
/// Each runs at once outside a capture. Inside one it records a node that runs at every replay,
/// ordered after every read and write of the entries it names that comes before it in the graph,
/// and before every one after it, including the reads and writes of tensors backed by them.
///
/// @code
/// // One iteration reads "t2" and writes "t2_next"; the swap makes the next value current.
/// {
///     cg::CaptureGuard guard(graph);
///     ...
///     tensor_io::swap_entries("amps.etn", "t2", "t2_next");
/// }
/// for (int iter = 0; iter < max_iter; ++iter) {
///     graph.execute();
/// }
/// @endcode

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN()
namespace APIARY_MODULE("io") tensor_io {

/**
 * @brief Copy the entry @p src_entry of @p src_path into the entry @p dst_entry of @p dst_path.
 *
 * The destination file is created when it does not exist, and the destination entry, with the
 * source's extents, element type and stored order, when it does not exist. An existing one must
 * have the same extents and element type. The elements pass through a buffer of at most
 * ``--einsums:work-buffer-size``, so neither entry is held whole. See @ref TensorFile::copy.
 * @param[in] src_path The file holding the entry to copy.
 * @param[in] src_entry The entry to copy.
 * @param[in] dst_path The file to copy it into, which may be @p src_path.
 * @param[in] dst_entry The entry to copy it into.
 * @throws std::runtime_error when the source file or entry is missing.
 * @throws std::invalid_argument when the destination entry exists with other extents or another
 *         element type.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_RENAME("copy") EINSUMS_EXPORT void copy_entry(std::string src_path, std::string src_entry, std::string dst_path,
                                                                   std::string dst_entry);

/**
 * @brief Give the entry @p old_name of @p path the name @p new_name.
 *
 * Only the file's entry table changes, so a rename costs the same whatever the entry holds. See
 * @ref TensorFile::rename.
 * @throws std::runtime_error when the file or the entry is missing.
 * @throws std::invalid_argument when an entry is already named @p new_name, or no entry can hold it.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_RENAME("rename") EINSUMS_EXPORT void rename_entry(std::string path, std::string old_name, std::string new_name);

/**
 * @brief Exchange the names of the entries @p first and @p second of @p path.
 *
 * Only the file's entry table changes. A program that reads one entry and writes the next value
 * beside it swaps the two after each step, so the same entry always holds the current value and
 * one graph serves every step. See @ref TensorFile::swap.
 * @throws std::runtime_error when the file or either entry is missing.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_RENAME("swap") EINSUMS_EXPORT void swap_entries(std::string path, std::string first, std::string second);

} // namespace tensor_io
EINSUMS_NAMESPACE_END()
