//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config/CompilerSpecific.hpp>

// re-enable warnings about dependent classes not being exported from the dll
#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(pop)
#endif
