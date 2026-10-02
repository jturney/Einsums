//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN()

// NOLINTBEGIN
EINSUMS_EXPORT int setup_Einsums_Tensor();

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

EINSUMS_NAMESPACE_END()