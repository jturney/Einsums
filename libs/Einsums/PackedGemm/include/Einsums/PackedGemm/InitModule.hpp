//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

namespace einsums {

/**
 * @brief Registers the initialization and finalization functions with the runtime manager.
 */
EINSUMS_EXPORT int setup_Einsums_PackedGemm();

/**
 * @brief Initialize the PackedGemm backend.
 */
EINSUMS_EXPORT void initialize_Einsums_PackedGemm();

/**
 * @brief Finalize the PackedGemm backend.
 */
EINSUMS_EXPORT void finalize_Einsums_PackedGemm();

namespace detail {
static int initialize_module_Einsums_PackedGemm = setup_Einsums_PackedGemm();

} // namespace detail

} // namespace einsums
