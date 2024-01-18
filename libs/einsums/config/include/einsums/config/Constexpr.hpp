//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/CompilerSpecific.hpp>
#include <einsums/config/defines.hpp>

/// This macro evaluates to ``inline constexpr`` for host code and
/// ``__device__ static const`` for device code with NVCC
#if defined(EINSUMS_COMPUTE_DEVICE_CODE) && defined(__NVCC__)
#    define EINSUMS_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE EINSUMS_DEVICE static const
#else
#    define EINSUMS_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE inline constexpr
#endif
