//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN()

EINSUMS_EXPORT int init_Einsums_BufferAllocator();

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments();
EINSUMS_EXPORT void initialize_Einsums_BufferAllocator();
EINSUMS_EXPORT void finalize_Einsums_BufferAllocator();

namespace detail {

static int initialize_module_Einsums_BufferAllocator = init_Einsums_BufferAllocator();

}

EINSUMS_NAMESPACE_END()