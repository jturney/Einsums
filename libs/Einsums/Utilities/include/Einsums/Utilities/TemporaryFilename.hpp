//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <filesystem>

namespace einsums {

/**
 * @brief Generate a unique path in the system temporary directory.
 *
 * The path is not created; only its name is reserved by uniqueness. Useful for
 * scratch files in tests (notably TensorIO round-trips) without hand-rolling a
 * name that collides under parallel ctest.
 *
 * @return A unique filesystem path. Converts to ``pathlib.Path`` in Python.
 */
APIARY_EXPOSE EINSUMS_EXPORT std::filesystem::path make_temp_path();

} // namespace einsums