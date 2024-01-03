//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums {

#if defined(EINSUMS_LINALG_INT_INTERFACE_ILP64)
using eint  = long long int;          // NOLINT
using euint = unsigned long long int; // NOLINT
using elong = long long int;          // NOLINT
#else
using eint  = int;          // NOLINT
using euint = unsigned int; // NOLINT
using elong = long int;     // NOLINT
#endif

} // namespace einsums