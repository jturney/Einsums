//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// This file can be included multiple times and does not use #pragma once.

#include <einsums/config/compiler_specific.hpp>

// re-enable warnings about dependent classes not being exported from the dll
#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(pop)
#endif
