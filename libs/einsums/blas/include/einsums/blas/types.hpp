//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::blas {

#if defined(EINSUMS_BLAS_INTERFACE_ILP64)
using int_t   = long long int;
using euint_t = unsigned long long int;
using elong   = long long int;
#elif defined(EINSUMS_BLAS_INTERFACE_LP64)
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#else
#    warning Unknown BLAS interface type. Defaulting to LP64.
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#endif

} // namespace einsums::blas