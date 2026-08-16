//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Get.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <filesystem>

/*
 * The Tensor module's options, plus the one derived path everything that
 * touches the scratch file wants.
 */

EINSUMS_NAMESPACE_BEGIN(detail)

/// The system temporary directory, asked for once when the option is registered.
EINSUMS_EXPORT std::string default_scratch_dir();

/// `einsums.<pid>.h5`, so two processes never share a scratch file.
EINSUMS_EXPORT std::string default_hdf5_file_name();

EINSUMS_NAMESPACE_END(detail)

EINSUMS_NAMESPACE_BEGIN(option)

/// Where the scratch file for disk-backed tensors lives.
inline constinit cl::ConfigOption<std::string> ScratchDir = cl::config_opt_computed<std::string>(
    "einsums:scratch-dir", "The scratch directory for Einsums tensor files.", "Tensor Options", &detail::default_scratch_dir);

/// The scratch file's name. Defaults to einsums.[pid].h5, where [pid] is the
/// process id, so a second process does not adopt the first one's file.
inline constinit cl::ConfigOption<std::string> Hdf5FileName = cl::config_opt_computed<std::string>(
    "einsums:hdf5-file-name",
    "The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.", "Tensor Options",
    &detail::default_hdf5_file_name);

/// Remove the per-process scratch file on exit.
///
/// It must default to true: otherwise these files accumulate indefinitely in
/// the scratch directory and, through PID reuse, a later process inherits a
/// stale one. Passing --einsums:no-delete-hdf5-files keeps them.
inline constinit cl::ConfigOption<bool> DeleteHdf5Files =
    cl::config_flag("einsums:delete-hdf5-files", "Clean up the HDF5 scratch file on exit.", "Tensor Options", true);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// The full path of the HDF5 scratch file this process uses.
EINSUMS_EXPORT std::filesystem::path hdf5_scratch_path();

EINSUMS_NAMESPACE_END()
