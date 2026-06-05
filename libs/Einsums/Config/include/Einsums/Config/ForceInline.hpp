//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>

#if !defined(EINSUMS_FORCEINLINE)
#    if defined(__NVCC__) || defined(__CUDACC__)
/// Marks a function to be forced inline.
/// @versionadded{1.0.0}
#        define EINSUMS_FORCEINLINE inline
#    elif defined(EINSUMS_MSVC)
#        define EINSUMS_FORCEINLINE __forceinline
#    elif defined(__GNUC__)
#        define EINSUMS_FORCEINLINE inline __attribute__((__always_inline__))
#    else
#        define EINSUMS_FORCEINLINE inline
#    endif
#endif
