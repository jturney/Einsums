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
 */

namespace einsums {

EINSUMS_EXPORT int init_Einsums_BufferAllocator();

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments();
EINSUMS_EXPORT void initialize_Einsums_BufferAllocator();
EINSUMS_EXPORT void finalize_Einsums_BufferAllocator();

namespace detail {

static int initialize_module_Einsums_BufferAllocator = init_Einsums_BufferAllocator();

}

} // namespace einsums