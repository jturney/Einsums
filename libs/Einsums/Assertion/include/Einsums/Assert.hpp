//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assertion/EvaluateAssert.hpp>
#include <Einsums/Preprocessor/Stringify.hpp>

#include <fmt/format.h>

#include <source_location>
#include <string>

#ifdef EINSUMS_COMPUTE_DEVICE_CODE
#    include <cassert>
#endif

namespace einsums::detail {

/**
 * Thrown when an assertion fails.
 *
 * @versionadded{2.0.0}
 */
struct assertion_error : std::logic_error { // NOLINT(readability-identifier-naming)
  public:
    using std::logic_error::logic_error;
};

/**
 * @typedef assertion_handler_type
 *
 * @brief The type for assertion handlers.
 *
 * @versionadded{1.0.0}
 */
// NOLINTNEXTLINE(readability-identifier-naming)
using assertion_handler_type = void (*)(std::source_location const &loc, char const *expr, std::string const &msg);

/**
 * @brief The default assertion handler. It prints line information whenever an assertion fails.
 *
 * @param[in] loc Where the assertion failed.
 * @param[in] expr The expression that failed.
 * @param[in] msg An extra diagnostic message.
 *
 * @throws assertion_error Always throws.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Throws an exception that can be handled rather than calling exit.
 * @endversion
 */
[[noreturn]] EINSUMS_EXPORT void default_assertion_handler(std::source_location const &loc, char const *expr, std::string const &msg);

/**
 * @brief Sets the assertion hanlder to a user-defined handler.
 *
 * @param[in] handler The new handler to use.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void set_assertion_handler(assertion_handler_type handler);

} // namespace einsums::detail

/// \cond NOINTERNAL
#define EINSUMS_ASSERT_(expr, ...)                                                                                                         \
    ((bool)(expr)                                                                                                                          \
         ? void()                                                                                                                          \
         : ::einsums::detail::handle_assert(std::source_location::current(), EINSUMS_PP_STRINGIFY(expr), fmt::format(__VA_ARGS__))) /**/

#define EINSUMS_ASSERT_LOCKED_(l, expr, ...)                                                                                               \
    ((bool)(expr) ? void()                                                                                                                 \
                  : ((l).unlock(), ::einsums::detail::handle_assert(std::source_location::current(), EINSUMS_PP_STRINGIFY(expr),           \
                                                                    fmt::format(__VA_ARGS__)))) /**/

#if defined(EINSUMS_DEBUG)
#    if defined(EINSUMS_COMPUTE_DEVICE_CODE)
/**
 * @def EINSUMS_ASSERT(expr)
 *
 * @brief This macro asserts that @p expr evaluates to true, but does not have a custom message.
 *
 * @param[in] expr The expression to test.
 *
 * @sa EINSUMS_ASSERT_MSG
 *
 * @throws assertion_error If the expression is false.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Throws an exception instead of calling exit when the expression is false.
 * @endversion
 */
#        define EINSUMS_ASSERT(expr) assert(expr)
/** \def EINSUMS_ASSERT_MSG(expr, msg)
 * \brief This macro asserts that \p expr evaluates to true.
 *
 * \param[in] expr The expression to assert on. This can either be an expression
 *             that's convertible to bool or a callable which returns bool
 * \param[in] msg The optional message that is used to give further information if
 *             the assert fails. This should be convertible to a std::string
 *
 * If \p expr evaluates to false, The source location and \p msg is
 * printed along with the expression and additional. Afterwards the program is
 * aborted. The assertion handler can be customized by calling
 * einsums::assertion::set_assertion_handler().
 *
 * Asserts are enabled if \a EINSUMS_DEBUG is set. This is the default for
 * `CMAKE_BUILD_TYPE=Debug`
 *
 * @throws assertion_error If the expression is false.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Throws an exception instead of calling exit when the expression is false.
 * @endversion
 */
#        define EINSUMS_ASSERT_MSG(expr, ...)           EINSUMS_ASSERT(expr)
#        define EINSUMS_ASSERT_LOCKED(l, expr)          assert(expr)
#        define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...) EINSUMS_ASSERT(expr)
#    else
#        define EINSUMS_ASSERT(expr)                    EINSUMS_ASSERT_(expr, "")
#        define EINSUMS_ASSERT_MSG(expr, ...)           EINSUMS_ASSERT_(expr, __VA_ARGS__)
#        define EINSUMS_ASSERT_LOCKED(l, expr)          EINSUMS_ASSERT_LOCKED_(l, expr, "")
#        define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...) EINSUMS_ASSERT_LOCKED_(l, expr, __VA_ARGS__)
#    endif
#else
#    define EINSUMS_ASSERT(expr)
#    define EINSUMS_ASSERT_MSG(expr, ...)
#    define EINSUMS_ASSERT_LOCKED(l, expr)
#    define EINSUMS_ASSERT_LOCKED_MSG(l, expr, ...)
#endif

#define EINSUMS_UNREACHABLE                                                                                                                \
    EINSUMS_ASSERT_(false, "This code is meant to be unreachable. If you are seeing this error "                                           \
                           "message it means that you have found a bug in Einsums. Please report it "                                      \
                           "on the issue tracker: https://github.com/Einsums/Einsums/issues.");                                            \
    std::terminate()
/// \endcond NOINTERNAL