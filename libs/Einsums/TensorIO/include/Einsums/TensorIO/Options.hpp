//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Options/Get.hpp>

/*
 * TensorIO's option, declared where .etn files are committed.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Sync an .etn file to disk before and after each commit rewrites its header, so that a power
/// loss, and not only a killed process, leaves the file as its last commit left it.
inline constinit cl::ConfigOption<bool> IoSync =
    cl::config_flag("einsums:io:sync", "Sync .etn files to disk at every commit, so a power loss leaves the last commit", "TensorIO", true);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give TensorIO's option its command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_TensorIO_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_TensorIO = register_Einsums_TensorIO_options();
}

EINSUMS_NAMESPACE_END()
