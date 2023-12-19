//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/preprocessor/Cat.hpp>

/// \cond NOINTERNAL
#define EINSUMS_PP_DETAILS_APPLY(macro, args)            EINSUMS_PP_DETAILS_APPLY_I(macro, args)
#define EINSUMS_PP_DETAILS_APPLY_I(macro, args)          macro args
#define EINSUMS_PP_DETAILS_STRIP_PARENS_I(...)           1, 1
#define EINSUMS_PP_DETAILS_EVAL(test, x)                 EINSUMS_PP_DETAILS_EVAL_I(test, x)
#define EINSUMS_PP_DETAILS_EVAL_I(test, x)               EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS(EINSUMS_PP_DETAILS_TEST_ARITY test, x)
#define EINSUMS_PP_DETAILS_TEST_ARITY(...)               EINSUMS_PP_DETAILS_APPLY(EINSUMS_PP_DETAILS_TEST_ARITY_I, (__VA_ARGS__, 2, 1, 0))
#define EINSUMS_PP_DETAILS_TEST_ARITY_I(a, b, c, ...)    c
#define EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS(cond, x)   EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_I(cond, x)
#define EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_I(cond, x) EINSUMS_PP_CAT(EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_, cond)(x)
#define EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_1(x)       x
#define EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_2(x)       EINSUMS_PP_DETAILS_APPLY(EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_2_I, x)
#define EINSUMS_PP_DETAILS_MAYBE_STRIP_PARENS_2_I(...)   __VA_ARGS__
/// \endcond

//==============================================================================
/*!
 * For any symbol \c X, this macro returns the same symbol from which potential
 * outer parens have been removed. If no outer parens are found, this macros
 * evaluates to \c X itself without error.
 *
 * The original implementation of this macro is from Steven Watanbe as shown
 * in http://boost.2283326.n4.nabble.com/preprocessor-removing-parentheses-td2591973.html#a2591976
 *
 * \param X Symbol to strip parens from
 *
 * \par Example Usage:
 *
 * \code
 * EINSUMS_PP_STRIP_PARENS(no_parens)
 * EINSUMS_PP_STRIP_PARENS((with_parens))
 * \endcode
 *
 * This produces the following output
 * \code
 * no_parens
 * with_parens
 * \endcode
 */
//==============================================================================
#define EINSUMS_PP_STRIP_PARENS(X)                                                                                                         \
    EINSUMS_PP_DETAILS_EVAL((EINSUMS_PP_DETAILS_STRIP_PARENS_I X), X)                                                                      \
    /**/
