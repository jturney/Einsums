//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

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

/*
 * The descriptors below store the address of their provider, and a descriptor
 * is constant-initialized. An exported function's address resolves through the
 * import table on Windows and so is not known until the loader runs, which
 * leaves it unusable as a constant initializer in anything that consumes the
 * library. These forward to the exported definitions from whatever binary
 * includes this header, where the address is an ordinary link-time constant.
 */
inline std::string scratch_dir_provider() {
    return default_scratch_dir();
}

inline std::string hdf5_file_name_provider() {
    return default_hdf5_file_name();
}

EINSUMS_NAMESPACE_END(detail)

EINSUMS_NAMESPACE_BEGIN(option)

/// Where the scratch file for disk-backed tensors lives.
inline constinit cl::ConfigOption<std::string> ScratchDir = cl::config_opt_computed<std::string>(
    "einsums:scratch-dir", "The scratch directory for Einsums tensor files.", "Tensor Options", &detail::scratch_dir_provider, "DIR");

/// The scratch file's name. Defaults to einsums.[pid].h5, where [pid] is the
/// process id, so a second process does not adopt the first one's file.
inline constinit cl::ConfigOption<std::string> Hdf5FileName = cl::config_opt_computed<std::string>(
    "einsums:hdf5-file-name",
    "The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.", "Tensor Options",
    &detail::hdf5_file_name_provider, "filename");

/// Remove the per-process scratch file on exit.
///
/// It must default to true: otherwise these files accumulate indefinitely in
/// the scratch directory and, through PID reuse, a later process inherits a
/// stale one. Passing --einsums:no-delete-hdf5-files keeps them.
inline constinit cl::ConfigOption<bool> DeleteHdf5Files =
    cl::config_flag("einsums:delete-hdf5-files", "Clean up the HDF5 scratch file on exit.", "Tensor Options", true);

/// Back large tensor buffers with transparent huge pages.
///
/// A tensor's storage is advised to the kernel (`madvise(MADV_HUGEPAGE)` on
/// Linux) before its first touch, so the zero-fill that follows faults 2 MB
/// pages instead of 4 KB ones. Off by default, because the effect is
/// shape-dependent in both directions: on the TCB intensli shapes it was worth
/// +10 to +23 percent where a contraction's inner walk spans many pages, and
/// cost 16 to 31 percent where an operand's strides are multiples of 32 KB or
/// more, since huge pages fix the low 21 address bits and such strides then
/// alias on the same DRAM channel and L2 sets that 4 KB page placement used to
/// scatter. Only buffers of a few huge pages or more are advised. No effect off
/// Linux or when THP is disabled.
inline constinit cl::ConfigOption<bool> TensorHugePages =
    cl::config_flag("einsums:tensor:huge-pages", "Back large tensor buffers with transparent huge pages where the OS offers them.",
                    "Tensor Options", false);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// The full path of the HDF5 scratch file this process uses.
EINSUMS_EXPORT std::filesystem::path hdf5_scratch_path();

/**
 * @brief Give the Tensor module's options their command-line presence. Idempotent.
 *
 * Run from a namespace-scope initializer rather than from the module's
 * argument hook, because the hook fires part-way through `initialize()` and
 * anything that enumerates the registry earlier - the Python binding layer
 * building argv, for one - would not see these options at all.
 */
EINSUMS_EXPORT int register_Einsums_Tensor_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Tensor = register_Einsums_Tensor_options();
}

EINSUMS_NAMESPACE_END()
