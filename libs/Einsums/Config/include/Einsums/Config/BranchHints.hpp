//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(__GNUC__)
/**
 * Hint at the compiler that \c expr is likely to be true.
 *
 * @param expr The expression to hint at.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_LIKELY(expr) __builtin_expect(static_cast<bool>(expr), true)
/**
 * Hint at the compiler that \c expr is likely to be false.
 *
 * @param expr The expression to hint at.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_UNLIKELY(expr) __builtin_expect(static_cast<bool>(expr), false)
#else
#    define EINSUMS_LIKELY(expr)   expr
#    define EINSUMS_UNLIKELY(expr) expr
#endif
