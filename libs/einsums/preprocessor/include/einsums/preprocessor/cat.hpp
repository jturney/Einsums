// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------
//

#pragma once

#if defined(DOXYGEN)
/// Concatenates the tokens \c A and \c B into a single token. Evaluates to \c AB
/// \param A First token
/// \param B Second token
# define EINSUMS_PP_CAT(A, B)
#else

# include <einsums/preprocessor/config.hpp>

# if ~EINSUMS_PP_CONFIG_FLAGS() & EINSUMS_PP_CONFIG_MWCC()
#  define EINSUMS_PP_CAT(a, b) EINSUMS_PP_CAT_I(a, b)
# else
#  define EINSUMS_PP_CAT(a, b) EINSUMS_PP_CAT_OO((a, b))
#  define EINSUMS_PP_CAT_OO(par) EINSUMS_PP_CAT_I##par
# endif
#
# if (~EINSUMS_PP_CONFIG_FLAGS() & EINSUMS_PP_CONFIG_MSVC()) ||                                    \
     (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700)
#  define EINSUMS_PP_CAT_I(a, b) a##b
# else
#  define EINSUMS_PP_CAT_I(a, b) EINSUMS_PP_CAT_II(~, a##b)
#  define EINSUMS_PP_CAT_II(p, res) res
# endif

#endif
