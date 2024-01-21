//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)

/// Hint at the compiler that \c expr is likely to be true.
#    define EINSUMS_LIKELY(expr)
/// Hint at the compiler that \c expr is likely to be false.
#    define EINSUMS_UNLIKELY(expr)

#else

#    if defined(__GNUC__)
#        define EINSUMS_LIKELY(expr)   __builtin_expect(static_cast<bool>(expr), true)
#        define EINSUMS_UNLIKELY(expr) __builtin_expect(static_cast<bool>(expr), false)
#    else
#        define EINSUMS_LIKELY(expr)   expr
#        define EINSUMS_UNLIKELY(expr) expr
#    endif
#endif
