//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assertion/evaluate_assert.hpp>
#include <einsums/preprocessor/stringize.hpp>

#include <fmt/format.h>

#include <exception>
#include <source_location>
#include <string>
#include <type_traits>

namespace einsums::detail {

using assertion_handler_type = void (*)(std::source_location const &loc, const char *expr, std::string const &msg);

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
/// einsums::assertion::set_assertion_handler().
///
/// Asserts are enabled if \a EINSUMS_DEBUG is set. This is the default for
/// `CMAKE_BUILD_TYPE=Debug`
#    define EINSUMS_ASSERT(expr)

/// \see EINSUMS_ASSERT
#    define EINSUMS_ASSERT_MSG(expr, msg)
#else
/// \cond NOINTERNAL
#    define EINSUMS_ASSERT_(expr, ...)                                                                                                     \
        (!!(expr) ? void()                                                                                                                 \
                  : ::einsums::detail::handle_assert(std::source_location::current(), EINSUMS_PP_STRINGIZE(expr),                          \
                                                     fmt::format(__VA_ARGS__))) /**/

#    define EINSUMS_ASSERT_LOCKED_(l, expr, ...)                                                                                           \
        (!!(expr) ? void()                                                                                                                 \
                  : ((l).unlock(), ::einsums::detail::handle_assert(std::source_location::current(), EINSUMS_PP_STRINGIZE(expr),           \
                                                                    fmt::format(__VA_ARGS__)))) /**/

#    if defined(EINSUMS_DEBUG)
#        if defined(EINSUMS_COMPUTE_DEVICE_CODE)
#            define EINSUMS_ASSERT(expr)                    assert(expr)
#            define EINSUMS_ASSERT_MSG(expr, ...)           EINSUMS_ASSERT(expr)
#            define EINSUMS_ASSERT_LOCKED(l, expr)          assert(expr)
#            define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...) EINSUMS_ASSERT(expr)
#        else
#            define EINSUMS_ASSERT(expr)                    EINSUMS_ASSERT_(expr, "")
#            define EINSUMS_ASSERT_MSG(expr, ...)           EINSUMS_ASSERT_(expr, __VA_ARGS__)
#            define EINSUMS_ASSERT_LOCKED(l, expr)          EINSUMS_ASSERT_LOCKED_(l, expr, "")
#            define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...) EINSUMS_ASSERT_LOCKED_(l, expr, __VA_ARGS__)
#        endif
#    else
#        define EINSUMS_ASSERT(expr)
#        define EINSUMS_ASSERT_MSG(expr, ...)
#        define EINSUMS_ASSERT_LOCKED(l, expr)
#        define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...)
#    endif

#    define EINSUMS_UNREACHABLE                                                                                                            \
        EINSUMS_ASSERT_(false, "This code is meant to be unreachable. If you are seeing this error "                                       \
                               "message it means that you have found a bug in EINSUMS. Please report it "                                  \
                               "on the issue tracker: https://github.com/STEllAR-GROUP/einsums/issues.");                                  \
        std::terminate()
#endif