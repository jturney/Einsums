//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/preprocessor/stringize.hpp>

#include <einsums/assertion/current_function.hpp>
#include <einsums/assertion/evaluate_assert.hpp>
#include <einsums/assertion/source_location.hpp>

#if defined(EINSUMS_COMPUTE_DEVICE_CODE)
#    include <assert.h>
#endif
#include <exception>
#include <string>
#include <type_traits>

namespace einsums::detail {
/// The signature for an assertion handler
using assertion_handler_type = void (*)(source_location const &loc, char const *expr, std::string const &msg);

/// Set the assertion handler to be used within a program. If the handler has been
/// set already once, the call to this function will be ignored.
/// \note This function is not thread safe
EINSUMS_EXPORT void set_assertion_handler(assertion_handler_type handler);
} // namespace einsums::detail

#if defined(DOXYGEN)
/// \def EINSUMS_ASSERT(expr, msg)
/// \brief This macro asserts that \a expr evaluates to true.
///
/// \param expr The expression to assert on. This can either be an expression
///             that's convertible to bool or a callable which returns bool
/// \param msg The optional message that is used to give further information if
///             the assert fails. This should be convertible to a std::string
///
/// If \p expr evaluates to false, The source location and \p msg is being
/// printed along with the expression and additional. Afterwards the program is
/// being aborted. The assertion handler can be customized by calling
/// einsums::detail::set_assertion_handler().
///
/// Asserts are enabled if \a EINSUMS_DEBUG is set. This is the default for
/// `CMAKE_BUILD_TYPE=Debug`
#    define EINSUMS_ASSERT(expr)

/// \see EINSUMS_ASSERT
#    define EINSUMS_ASSERT_MSG(expr, msg)
#else
/// \cond NOINTERNAL
#    define EINSUMS_ASSERT_(expr, msg)                                                                                 \
        (!!(expr) ? void()                                                                                             \
                  : ::einsums::detail::handle_assert(                                                                  \
                        ::einsums::detail::source_location{__FILE__, static_cast<unsigned>(__LINE__),                  \
                                                           EINSUMS_ASSERT_CURRENT_FUNCTION},                           \
                        EINSUMS_PP_STRINGIZE(expr), msg)) /**/

#    if defined(EINSUMS_DEBUG)
#        if defined(EINSUMS_COMPUTE_DEVICE_CODE)
#            define EINSUMS_ASSERT(expr)          assert(expr)
#            define EINSUMS_ASSERT_MSG(expr, msg) EINSUMS_ASSERT(expr)
#        else
#            define EINSUMS_ASSERT(expr)          EINSUMS_ASSERT_(expr, std::string())
#            define EINSUMS_ASSERT_MSG(expr, msg) EINSUMS_ASSERT_(expr, msg)
#        endif
#    else
#        define EINSUMS_ASSERT(expr)
#        define EINSUMS_ASSERT_MSG(expr, msg)
#    endif

#    define EINSUMS_UNREACHABLE                                                                                        \
        EINSUMS_ASSERT_(false, "This code is meant to be unreachable. If you are seeing this error "                   \
                               "message it means that you have found a bug in einsums. Please report "                 \
                               "it on https://github.com/Einsums/Einsums/issues.");                                    \
        std::terminate()
#endif
