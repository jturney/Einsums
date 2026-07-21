//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 *
 * If this module does need to be initialized, make sure this header is referenced at least somewhere
 * in your public code. If it is not, then the initialization routines will not occur. Feel free
 * to change the namespaces of these functions. Just remember to also change the namespaces in the
 * corresponding code files.
 */

namespace einsums {

// NOLINTBEGIN
EINSUMS_EXPORT int setup_Einsums_Tensor();

EINSUMS_EXPORT void add_Einsums_Tensor_arguments();
EINSUMS_EXPORT void initialize_Einsums_Tensor();
EINSUMS_EXPORT void finalize_Einsums_Tensor();
// NOLINTEND

/// Open an existing HDF5 file for read/write. Returns true on success; returns
/// false (without terminating) when the file exists but is not a usable HDF5
/// file - e.g. a stale or corrupt scratch file left by an earlier process whose
/// PID has since been reused. Callers should recreate the file in that case.
EINSUMS_EXPORT bool open_hdf5_file(std::string const &fname);
EINSUMS_EXPORT void create_hdf5_file(std::string const &fname);

namespace detail {

static int initialize_module_Einsums_Tensor = setup_Einsums_Tensor(); // NOLINT(bugprone-throwing-static-initialization)

}

} // namespace einsums