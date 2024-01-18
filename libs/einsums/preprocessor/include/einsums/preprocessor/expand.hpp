//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// The EINSUMS_PP_EXPAND macro performs a double macro-expansion on its argument.
/// \param X Token to be expandd twice
///
/// This macro can be used to prouce a delayed preprocessor expansion
///
/// Example:
/// \code
/// #define MACRO(a, b, c) (a)(b)(c)
/// #define ARGS() (1, 2, 3)
///
/// EINSUMS_PP_EXPAND(MACRO ARGS()) // expands to (1)(2)(3)
/// \endcode
#define EINSUMS_PP_EXPAND(X) X
