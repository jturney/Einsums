//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception_fwd.hpp>
#include <einsums/errors/exception_info.hpp>

#include <exception>
#include <functional>
#include <string>
#include <system_error>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

////////////////////////////////////////////////////////////////////////////////
namespace einsums {

////////////////////////////////////////////////////////////////////////////////
/// \brief A einsums::exception is the main exception type used by einsums to
///        report errors.
///
/// The einsums::exception type is the main exception type  used by einsums to
/// report errors. Any exceptions thrown by functions in the einsums library
/// are either of this type or of a type derived from it. This implies that
/// it is always safe to use this type only in catch statements guarding
/// einsums library calls.
struct EINSUMS_EXPORT exception : public std::system_error {

    /**
     * @brief Construct a einsums::exception from an \a einsums::error .
     *
     * @param e The parameter \p e holds the einsums::error code the new exception should encapsulate.
     */
    explicit exception(error e = einsums::error::success);

    /// Construct a einsums::exception from a std#system_error.
    explicit exception(std::system_error const &e);

    /// Construct a einsums::exception from a std#system#error_code. This
    /// constructor is required to compensate for the changes introduced as
    /// a resolution to LWG3162 (https://cplusplus.github.io/LWG/issue3162).
    explicit exception(std::error_code const &e);

    /// Construct a einsums::exception from a \a einsums::error and an error message.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    exception(error e, char const *msg, throwmode mode = throwmode::plain);

    /// Construct a einsums::exception from a \a einsums::error and an error message.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    exception(error e, std::string const &msg, throwmode mode = throwmode::plain);

    /// Destruct a einsums::exception
    ///
    /// \throws nothing
    ~exception() noexcept;

    /// The function \a get_error() returns the einsums::error code stored
    /// in the referenced instance of a einsums::exception. It returns
    /// the einsums::error code this exception instance was constructed
    /// from.
    ///
    /// \throws nothing
    [[nodiscard]] auto get_error() const noexcept -> error;

    /// The function \a get_error_code() returns a einsums::error_code which
    /// represents the same error condition as this einsums::exception instance.
    ///
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    [[nodiscard]] auto get_error_code(throwmode mode = throwmode::plain) const noexcept -> error_code;
};

////////////////////////////////////////////////////////////////////////////////

namespace detail {

using custom_exception_info_handler_type =
    std::function<einsums::exception_info(std::string const &, std::string const &, long, std::string const &)>;

EINSUMS_EXPORT void set_custom_exception_info_handler(custom_exception_info_handler_type f);

using pre_exception_handler_type = std::function<void()>;

EINSUMS_EXPORT void set_pre_exception_handler(pre_exception_handler_type f);

} // namespace detail

///////////////////////////////////////////////////////////////////////////
/// \brief A einsums::thread_interrupted is the exception type used by einsums to
///        interrupt a running einsums thread.
///
/// The \a einsums::thread_interrupted type is the exception type used by einsums to
/// interrupt a running thread.
///
/// A running thread can be interrupted by invoking the interrupt() member
/// function of the corresponding einsums::thread object. When the interrupted
/// thread next executes one of the specified interruption points (or if it
/// is currently blocked whilst executing one) with interruption enabled,
/// then a einsums::thread_interrupted exception will be thrown in the interrupted
/// thread. If not caught, this will cause the execution of the interrupted
/// thread to terminate. As with any other exception, the stack will be
/// unwound, and destructors for objects of automatic storage duration will
/// be executed.
///
/// If a thread wishes to avoid being interrupted, it can create an instance
/// of \a einsums::this_thread::disable_interruption. Objects of this class disable
/// interruption for the thread that created them on construction, and
/// restore the interruption state to whatever it was before on destruction.
///
/// \code
///     void f()
///     {
///         // interruption enabled here
///         {
///             einsums::this_thread::disable_interruption di;
///             // interruption disabled
///             {
///                 einsums::this_thread::disable_interruption di2;
///                 // interruption still disabled
///             } // di2 destroyed, interruption state restored
///             // interruption still disabled
///         } // di destroyed, interruption state restored
///         // interruption now enabled
///     }
/// \endcode
///
/// The effects of an instance of \a einsums::this_thread::disable_interruption can be
/// temporarily reversed by constructing an instance of
/// \a einsums::this_thread::restore_interruption, passing in the
/// \a einsums::this_thread::disable_interruption object in question. This will restore
/// the interruption state to what it was when the
/// \a einsums::this_thread::disable_interruption
/// object was constructed, and then disable interruption again when the
/// \a einsums::this_thread::restore_interruption object is destroyed.
///
/// \code
///     void g()
///     {
///         // interruption enabled here
///         {
///             einsums::this_thread::disable_interruption di;
///             // interruption disabled
///             {
///                 einsums::this_thread::restore_interruption ri(di);
///                 // interruption now enabled
///             } // ri destroyed, interruption disable again
///         } // di destroyed, interruption state restored
///         // interruption now enabled
///     }
/// \endcode
///
/// At any point, the interruption state for the current thread can be
/// queried by calling \a einsums::this_thread::interruption_enabled().
struct EINSUMS_EXPORT thread_interrupted : std::exception {};

/// \cond NODETAIL
namespace detail {
// Stores the information about the function name the exception has been
// raised in. This information will show up in error messages under the
// [function] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_function, std::string);

// Stores the information about the source file name the exception has
// been raised in. This information will show up in error messages under
// the [file] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_file, std::string);

// Stores the information about the source file line number the exception
// has been raised at. This information will show up in error messages
// under the [line] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_line, long);

struct EINSUMS_EXPORT std_exception : std::exception {
  private:
    std::string _what;

  public:
    explicit std_exception(std::string const &w) : _what(w) {}

    ~std_exception() noexcept {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }
};

struct EINSUMS_EXPORT bad_alloc : std::bad_alloc {
  private:
    std::string _what;

  public:
    explicit bad_alloc(std::string const &w) : _what(w) {}

    ~bad_alloc() noexcept {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }
};

struct EINSUMS_EXPORT bad_exception : std::bad_exception {
  private:
    std::string _what;

  public:
    explicit bad_exception(std::string const &w) : _what(w) {}

    ~bad_exception() noexcept {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }
};

struct EINSUMS_EXPORT bad_cast : std::bad_cast {
  private:
    std::string _what;

  public:
    explicit bad_cast(std::string const &w) : _what(w) {}

    ~bad_cast() noexcept {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }
};

struct EINSUMS_EXPORT bad_typeid : std::bad_typeid {
  private:
    std::string _what;

  public:
    explicit bad_typeid(std::string const &w) : _what(w) {}

    ~bad_typeid() noexcept {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }
};

template <typename Exception>
EINSUMS_EXPORT auto get_exception(einsums::exception const &e, std::string const &func, std::string const &file,
                                  long line, std::string const &auxinfo) -> std::exception_ptr;

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e) -> std::exception_ptr;
} // namespace detail
/// \endcond

///////////////////////////////////////////////////////////////////////////
// Extract elements of the diagnostic information embedded in the given
// exception.

/// \brief Return the error message of the thrown exception.
///
/// The function \a einsums::get_error_what can be used to extract the
/// diagnostic information element representing the error message as stored
/// in the given exception instance.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \returns    The error message stored in the exception
///             If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error()
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_config(), \a einsums::get_error_state()
///
EINSUMS_EXPORT auto get_error_what(exception_info const &xi) -> std::string;

/// \cond NOINTERNAL
template <typename E>
auto get_error_what(E const &e) -> std::string {
    return invoke_with_exception_info(
        e, [](exception_info const *xi) { return xi ? get_error_what(*xi) : std::string("<unknown>"); });
}

inline auto get_error_what(einsums::error_code const &e) -> std::string {
    // if this is a lightweight error_code, return canned response
    if (e.category() == detail::get_lightweight_einsums_category())
        return e.message();

    return get_error_what<einsums::error_code>(e);
}

inline auto get_error_what(std::exception const &e) -> std::string {
    return e.what();
}
/// \endcond

/// \brief Return the error code value of the exception thrown.
///
/// The function \a einsums::get_error can be used to extract the
/// diagnostic information element representing the error value code as
/// stored in the given exception instance.
///
/// \param e    The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception,
///             \a einsums::error_code, or \a std::exception_ptr.
///
/// \returns    The error value code of the process where the exception was
///             thrown.
///
/// \throws     nothing
///
/// \see        \a einsums::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT auto get_error(einsums::exception const &e) -> error;

/// \copydoc get_error(einsums::exception const& e)
EINSUMS_EXPORT auto get_error(einsums::error_code const &e) -> error;

/// \cond NOINTERNAL
EINSUMS_EXPORT auto get_error(std::exception_ptr const &e) -> error;
/// \endcond

/// \brief Return the function name from which the exception was thrown.
///
/// The function \a einsums::get_error_function_name can be used to extract the
/// diagnostic information element representing the name of the function
/// as stored in the given exception instance.
///
/// \returns    The name of the function from which the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id()
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT auto get_error_function_name(einsums::exception_info const &xi) -> std::string;

/// \cond NOINTERNAL
template <typename E>
auto get_error_function_name(E const &e) -> std::string {
    return invoke_with_exception_info(
        e, [](exception_info const *xi) { return xi ? get_error_function_name(*xi) : std::string("<unknown>"); });
}
/// \endcond

/// \brief Return the (source code) file name of the function from which
///        the exception was thrown.
///
/// The function \a einsums::get_error_file_name can be used to extract the
/// diagnostic information element representing the name of the source file
/// as stored in the given exception instance.
///
/// \returns    The name of the source file of the function from which the
///             exception was thrown. If the exception instance does
///             not hold this information, the function will return an empty
///             string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT auto get_error_file_name(einsums::exception_info const &xi) -> std::string;

/// \cond NOINTERNAL
template <typename E>
auto get_error_file_name(E const &e) -> std::string {
    return invoke_with_exception_info(
        e, [](exception_info const *xi) { return xi ? get_error_file_name(*xi) : std::string("<unknown>"); });
}
/// \endcond

/// \brief Return the line number in the (source code) file of the function
///        from which the exception was thrown.
///
/// The function \a einsums::get_error_line_number can be used to extract the
/// diagnostic information element representing the line number
/// as stored in the given exception instance.
///
/// \returns    The line number of the place where the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return -1.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     nothing
///
/// \see        \a einsums::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name()
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT auto get_error_line_number(einsums::exception_info const &xi) -> long;

/// \cond NOINTERNAL
template <typename E>
auto get_error_line_number(E const &e) -> long {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_line_number(*xi) : -1; });
}
/// \endcond
} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>

#include <einsums/errors/throw_exception.hpp>
