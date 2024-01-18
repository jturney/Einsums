//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/errors/error.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception.hpp>
#include <einsums/errors/exception_info.hpp>
// #include <einsums/modules/logging.hpp>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#include <exception>
#include <string>
#include <system_error>
#include <utility>

namespace einsums {
///////////////////////////////////////////////////////////////////////////
/// Construct a einsums::exception from a \a einsums::error.
///
/// \param e    The parameter \p e holds the einsums::error code the new
///             exception should encapsulate.
exception::exception(error e) : std::system_error(make_error_code(e, throwmode::plain)) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
    if (e != einsums::error::success) {
        // LERR_(error).format("created exception: {}", this->what());
    }
}

/// Construct a einsums::exception from a std#system_error.
exception::exception(std::system_error const &e) : std::system_error(e) {
    // LERR_(error).format("created exception: {}", this->what());
}

/// Construct a einsums::exception from a std#system#error_code.
exception::exception(std::error_code const &e) : std::system_error(e) {
    // LERR_(error).format("created exception: {}", this->what());
}

/// Construct a einsums::exception from a \a einsums::error and an error message.
///
/// \param e      The parameter \p e holds the einsums::error code the new
///               exception should encapsulate.
/// \param msg    The parameter \p msg holds the error message the new
///               exception should encapsulate.
/// \param mode   The parameter \p mode specifies whether the returned
///               einsums::error_code belongs to the error category
///               \a einsums_category (if mode is \a plain, this is the
///               default) or to the category \a einsums_category_rethrow
///               (if mode is \a rethrow).
exception::exception(error e, char const *msg, throwmode mode)
    : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
    if (e != einsums::error::success) {
        // LERR_(error).format("created exception: {}", this->what());
    }
}

/// Construct a einsums::exception from a \a einsums::error and an error message.
///
/// \param e      The parameter \p e holds the einsums::error code the new
///               exception should encapsulate.
/// \param msg    The parameter \p msg holds the error message the new
///               exception should encapsulate.
/// \param mode   The parameter \p mode specifies whether the returned
///               einsums::error_code belongs to the error category
///               \a einsums_category (if mode is \a plain, this is the
///               default) or to the category \a einsums_category_rethrow
///               (if mode is \a rethrow).
exception::exception(error e, std::string const &msg, throwmode mode)
    : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
    if (e != einsums::error::success) {
        // LERR_(error).format("created exception: {}", this->what());
    }
}

/// Destruct a einsums::exception
///
/// \throws nothing
exception::~exception() noexcept {
}

/// The function \a get_error() returns the einsums::error code stored
/// in the referenced instance of a einsums::exception. It returns
/// the einsums::error code this exception instance was constructed
/// from.
///
/// \throws nothing
auto exception::get_error() const noexcept -> error {
    return static_cast<error>(this->std::system_error::code().value());
}

/// The function \a get_error_code() returns a einsums::error_code which
/// represents the same error condition as this einsums::exception instance.
///
/// \param mode   The parameter \p mode specifies whether the returned
///               einsums::error_code belongs to the error category
///               \a einsums_category (if mode is \a plain, this is the
///               default) or to the category \a einsums_category_rethrow
///               (if mode is \a rethrow).
auto exception::get_error_code(throwmode mode) const noexcept -> error_code {
    (void)mode;
    return {this->std::system_error::code().value(), *this};
}

namespace detail {
static custom_exception_info_handler_type custom_exception_info_handler;

void set_custom_exception_info_handler(custom_exception_info_handler_type f) {
    custom_exception_info_handler = f;
}

static pre_exception_handler_type pre_exception_handler;

void set_pre_exception_handler(pre_exception_handler_type f) {
    pre_exception_handler = f;
}
} // namespace detail
} // namespace einsums

namespace einsums::detail {
template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e, std::string const &func,
                                                    std::string const &file, long line) -> std::exception_ptr {
    // create a std::exception_ptr object encapsulating the Exception to
    // be thrown and annotate it with all the local information we have
    try {
        throw_with_info(e, std::move(einsums::exception_info().set(einsums::detail::throw_function(func),
                                                                   einsums::detail::throw_file(file),
                                                                   einsums::detail::throw_line(line))));
    } catch (...) {
        return std::current_exception();
    }

    // need this return to silence a warning with icc
    EINSUMS_ASSERT(false); // -V779
    return {};
}

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e) -> std::exception_ptr {
    // create a std::exception_ptr object encapsulating the Exception to
    // be thrown and annotate it with all the local information we have
    try {
        einsums::throw_with_info(e);
    } catch (...) {
        return std::current_exception();
    }

    // need this return to silence a warning with icc
    EINSUMS_ASSERT(false); // -V779
    return {};
}

template EINSUMS_EXPORT std::exception_ptr construct_lightweight_exception(einsums::thread_interrupted const &);

template <typename Exception>
EINSUMS_EXPORT auto construct_custom_exception(Exception const &e, std::string const &func, std::string const &file,
                                               long line, std::string const &auxinfo) -> std::exception_ptr {
    if (!custom_exception_info_handler) {
        return construct_lightweight_exception(e, func, file, line);
    }

    // create a std::exception_ptr object encapsulating the Exception to
    // be thrown and annotate it with information provided by the hook
    try {
        throw_with_info(e, custom_exception_info_handler(func, file, line, auxinfo));
    } catch (...) {
        return std::current_exception();
    }

    // need this return to silence a warning with icc
    EINSUMS_ASSERT(false); // -V779
    return {};
}

///////////////////////////////////////////////////////////////////////////
template <typename Exception>
inline auto is_of_lightweight_einsums_category(Exception const &) -> bool {
    return false;
}

inline auto is_of_lightweight_einsums_category(einsums::exception const &e) -> bool {
    return e.get_error_code().category() == get_lightweight_einsums_category();
}

///////////////////////////////////////////////////////////////////////////
auto access_exception(error_code const &e) -> std::exception_ptr {
    return e._exception;
}

template <typename Exception>
EINSUMS_EXPORT auto get_exception(Exception const &e, std::string const &func, std::string const &file, long line,
                                  std::string const &auxinfo) -> std::exception_ptr {
    if (is_of_lightweight_einsums_category(e)) {
        return construct_lightweight_exception(e, func, file, line);
    }

    return construct_custom_exception(e, func, file, line, auxinfo);
}

template <typename Exception>
EINSUMS_EXPORT void throw_exception(Exception const &e, std::string const &func, std::string const &file, long line) {
    if (pre_exception_handler) {
        pre_exception_handler();
    }

    std::rethrow_exception(get_exception(e, func, file, line));
}

///////////////////////////////////////////////////////////////////////////
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::exception const &, std::string const &,
                                                         std::string const &, long, std::string const &);

template EINSUMS_EXPORT std::exception_ptr get_exception(std::system_error const &, std::string const &,
                                                         std::string const &, long, std::string const &);

template EINSUMS_EXPORT std::exception_ptr get_exception(std::exception const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::detail::std_exception const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::bad_exception const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::detail::bad_exception const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::bad_typeid const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::detail::bad_typeid const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::bad_cast const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::detail::bad_cast const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::bad_alloc const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(einsums::detail::bad_alloc const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::logic_error const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::runtime_error const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::out_of_range const &, std::string const &,
                                                         std::string const &, long, std::string const &);
template EINSUMS_EXPORT std::exception_ptr get_exception(std::invalid_argument const &, std::string const &,
                                                         std::string const &, long, std::string const &);

///////////////////////////////////////////////////////////////////////////
template EINSUMS_EXPORT void throw_exception(einsums::exception const &, std::string const &, std::string const &,
                                             long);

template EINSUMS_EXPORT void throw_exception(std::system_error const &, std::string const &, std::string const &, long);

template EINSUMS_EXPORT void throw_exception(std::exception const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(einsums::detail::std_exception const &, std::string const &,
                                             std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::bad_exception const &, std::string const &, std::string const &,
                                             long);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_exception const &, std::string const &,
                                             std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::bad_typeid const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_typeid const &, std::string const &,
                                             std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::bad_cast const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_cast const &, std::string const &,
                                             std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::bad_alloc const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_alloc const &, std::string const &,
                                             std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::logic_error const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::runtime_error const &, std::string const &, std::string const &,
                                             long);
template EINSUMS_EXPORT void throw_exception(std::out_of_range const &, std::string const &, std::string const &, long);
template EINSUMS_EXPORT void throw_exception(std::invalid_argument const &, std::string const &, std::string const &,
                                             long);
} // namespace einsums::detail

///////////////////////////////////////////////////////////////////////////////
namespace einsums {

///////////////////////////////////////////////////////////////////////////
/// Return the error message.
auto get_error_what(einsums::exception_info const &xi) -> std::string {
    // Try a cast to std::exception - this should handle boost.system
    // error codes in addition to the standard library exceptions.
    auto const *se = dynamic_cast<std::exception const *>(&xi);
    return se ? se->what() : std::string("<unknown>");
}

///////////////////////////////////////////////////////////////////////////
auto get_error(einsums::exception const &e) -> error {
    return static_cast<einsums::error>(e.get_error());
}

auto get_error(einsums::error_code const &e) -> error {
    return static_cast<einsums::error>(e.value());
}

auto get_error(std::exception_ptr const &e) -> error {
    try {
        std::rethrow_exception(e);
    } catch (einsums::thread_interrupted const &) {
        return einsums::error::thread_cancelled;
    } catch (einsums::exception const &he) {
        return he.get_error();
    } catch (std::system_error const &e) {
        int code = e.code().value();
        if (code < static_cast<int>(einsums::error::success) || code >= static_cast<int>(einsums::error::last_error))
            code |= static_cast<int>(einsums::error::system_error_flag);
        return static_cast<einsums::error>(code);
    } catch (...) {
        return einsums::error::unknown_error;
    }
}

/// Return the function name from which the exception was thrown.
auto get_error_function_name(einsums::exception_info const &xi) -> std::string {
    std::string const *function = xi.get<einsums::detail::throw_function>();
    if (function)
        return *function;

    return {};
}

/// Return the (source code) file name of the function from which the
/// exception was thrown.
auto get_error_file_name(einsums::exception_info const &xi) -> std::string {
    std::string const *file = xi.get<einsums::detail::throw_file>();
    if (file)
        return *file;

    return "<unknown>";
}

/// Return the line number in the (source code) file of the function from
/// which the exception was thrown.
auto get_error_line_number(einsums::exception_info const &xi) -> long {
    long const *line = xi.get<einsums::detail::throw_line>();
    if (line)
        return *line;
    return -1;
}

} // namespace einsums
