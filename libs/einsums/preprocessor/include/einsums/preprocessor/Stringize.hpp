//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)
/// The \a EINSUMS_PP_STRINGIZE macro stringizes its argument after it has been expanded.
///
/// \param X The text to be converted to a string literal
///
/// The passed argument \c X will expand to \c "X". Note that the stringizing
/// operator (#) prevents arguments from expanding. This macro circumvents this
/// shortcoming.
#    define EINSUMS_PP_STRINGIZE(X)
#else

#    include <einsums/preprocessor/Config.hpp>

#    if EINSUMS_PP_CONFIG_FLAGS() & EINSUMS_PP_CONFIG_MSVC()
#        define EINSUMS_PP_STRINGIZE(text)  EINSUMS_PP_STRINGIZE_A((text))
#        define EINSUMS_PP_STRINGIZE_A(arg) EINSUMS_PP_STRINGIZE_I arg
#    elif EINSUMS_PP_CONFIG_FLAGS() & EINSUMS_PP_CONFIG_MWCC()
#        define EINSUMS_PP_STRINGIZE(text)   EINSUMS_PP_STRINGIZE_OO((text))
#        define EINSUMS_PP_STRINGIZE_OO(par) EINSUMS_PP_STRINGIZE_I##par
#    else
#        define EINSUMS_PP_STRINGIZE(text) EINSUMS_PP_STRINGIZE_I(text)
#    endif

#    define EINSUMS_PP_STRINGIZE_I(text) #text

#endif
