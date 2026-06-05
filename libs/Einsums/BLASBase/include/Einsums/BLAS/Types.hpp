//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLASBase/Defines.hpp>

namespace einsums::blas {

/**
 * @typedef int_t
 *
 * Represents the standard BLAS integer type. Its size depends on the BLAS interface used.
 *
 * @versionadded{1.0.0}
 */
/**
 * @typedef euint_t
 *
 * Represents the standard unsigned BLAS integer type. Its size depends on the BLAS interface used.
 *
 * @versionadded{1.0.0}
 */
/**
 * @typedef elong
 *
 * Represents a potentially longer integer type for the BLAS interface. Its size depends on the BLAS interface used.
 *
 * @versionadded{1.0.0}
 */

#if defined(EINSUMS_BLAS_INTERFACE_ILP64)
using int_t   = long long int;
using euint_t = unsigned long long int;
using elong   = long long int;
#elif defined(EINSUMS_BLAS_INTERFACE_LP64)
using int_t   = int;          // NOLINT(readability-identifier-naming)
using euint_t = unsigned int; // NOLINT(readability-identifier-naming)
using elong   = long int;     // NOLINT(readability-identifier-naming)
#else
#    warning Unknown BLAS interface type. Defaulting to LP64.
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#endif

} // namespace einsums::blas
