//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/exception_fwd.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

///////////////////////////////////////////////////////////////////////////////
namespace einsums {

/// \cond NODETAIL
namespace detail {

EINSUMS_EXPORT auto access_exception(error_code const &) -> std::exception_ptr;

///////////////////////////////////////////////////////////////////////////////
struct command_line_error : std::logic_error {
    explicit command_line_error(char const *msg) : std::logic_error(msg) {}
    explicit command_line_error(std::string const &msg) : std::logic_error(msg) {}
};

} // namespace detail
  /// \endcond

///////////////////////////////////////////////////////////////////////////
/// \brief Returns generic einsums error category used for new errors.
EINSUMS_EXPORT auto get_einsums_category() -> std::error_category const &;

/// \brief Returns generic einsums error category used for errors re-thrown
///        after the exception has been de-serialized.
EINSUMS_EXPORT auto get_einsums_rethrow_category() -> std::error_category const &;

/// \cond NOINTERNAL
namespace detail {
EINSUMS_EXPORT auto get_lightweight_einsums_category() -> std::error_category const &;

EINSUMS_EXPORT auto get_einsums_category(throwmode mode) -> std::error_category const &;

inline auto make_system_error_code(error e, throwmode mode = throwmode::plain) -> std::error_code {
    return {static_cast<int>(e), get_einsums_category(mode)};
}

///////////////////////////////////////////////////////////////////////////
inline auto make_error_condition(error e, throwmode mode) -> std::error_condition {
    return {static_cast<int>(e), get_einsums_category(mode)};
}
} // namespace detail
/// \endcond

///////////////////////////////////////////////////////////////////////////
/// \brief A einsums::error_code represents an arbitrary error condition.
///
/// The class einsums::error_code describes an object used to hold error code
/// values, such as those originating from the operating system or other
/// low-level application program interfaces.
///
/// \note Class einsums::error_code is an adjunct to error reporting by
/// exception
///
class error_code : public std::error_code //-V690
{
  public:
    /// Construct an object of type error_code.
    ///
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    ///
    /// \throws nothing
    explicit error_code(throwmode mode = throwmode::plain)
        : std::error_code(detail::make_system_error_code(einsums::error::success, mode)) {}

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    ///
    /// \throws nothing
    EINSUMS_EXPORT explicit error_code(error e, throwmode mode = throwmode::plain);

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param func   The name of the function where the error was raised.
    /// \param file   The file name of the code where the error was raised.
    /// \param line   The line number of the code line where the error was
    ///               raised.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    ///
    /// \throws nothing
    EINSUMS_EXPORT error_code(error e, char const *func, char const *file, long line,
                              throwmode mode = throwmode::plain);

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    ///
    /// \throws std#bad_alloc (if allocation of a copy of
    ///         the passed string fails).
    EINSUMS_EXPORT error_code(error e, char const *msg, throwmode mode = throwmode::plain);

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param func   The name of the function where the error was raised.
    /// \param file   The file name of the code where the error was raised.
    /// \param line   The line number of the code line where the error was
    ///               raised.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this is the
    ///               default) or to the category \a einsums_category_rethrow
    ///               (if mode is \a throwmode::rethrow).
    ///
    /// \throws std#bad_alloc (if allocation of a copy of
    ///         the passed string fails).
    EINSUMS_EXPORT error_code(error e, char const *msg, char const *func, char const *file, long line,
                              throwmode mode = throwmode::plain);

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this is the
    ///               default) or to the category \a einsums_category_rethrow
    ///               (if mode is \a throwmode::rethrow).
    ///
    /// \throws std#bad_alloc (if allocation of a copy of
    ///         the passed string fails).
    EINSUMS_EXPORT error_code(error e, std::string const &msg, throwmode mode = throwmode::plain);

    /// Construct an object of type error_code.
    ///
    /// \param e      The parameter \p e holds the einsums::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param func   The name of the function where the error was raised.
    /// \param file   The file name of the code where the error was raised.
    /// \param line   The line number of the code line where the error was
    ///               raised.
    /// \param mode   The parameter \p mode specifies whether the constructed
    ///               einsums::error_code belongs to the error category
    ///               \a einsums_category (if mode is \a throwmode::plain, this
    ///               is the default) or to the category \a
    ///               einsums_category_rethrow (if mode is \a throwmode::rethrow).
    ///
    /// \throws std#bad_alloc (if allocation of a copy of
    ///         the passed string fails).
    EINSUMS_EXPORT error_code(error e, std::string const &msg, char const *func, char const *file, long line,
                              throwmode mode = throwmode::plain);

    /// Return a reference to the error message stored in the einsums::error_code.
    ///
    /// \throws nothing
    [[nodiscard]] EINSUMS_EXPORT auto get_message() const -> std::string;

    /// \brief Clear this error_code object.
    /// The postconditions of invoking this method are
    /// * value() == einsums::error::success and category() == einsums::get_einsums_category()
    void clear() {
        this->std::error_code::assign(static_cast<int>(einsums::error::success), get_einsums_category());
        _exception = std::exception_ptr();
    }

    /// Copy constructor for error_code
    ///
    /// \note This function maintains the error category of the left hand
    ///       side if the right hand side is a success code.
    EINSUMS_EXPORT error_code(error_code const &rhs);

    /// Assignment operator for error_code
    ///
    /// \note This function maintains the error category of the left hand
    ///       side if the right hand side is a success code.
    EINSUMS_EXPORT auto operator=(error_code const &rhs) -> error_code &;

  private:
    friend auto detail::access_exception(error_code const &) -> std::exception_ptr;
    friend class exception;
    friend auto make_error_code(std::exception_ptr const &) -> error_code;

    EINSUMS_EXPORT error_code(int err, einsums::exception const &e);
    EINSUMS_EXPORT explicit error_code(std::exception_ptr const &e);

    std::exception_ptr _exception;
};

/// @{
/// \brief Returns a new error_code constructed from the given parameters.
inline auto make_error_code(error e, throwmode mode = throwmode::plain) -> error_code {
    return error_code(e, mode);
}
inline auto make_error_code(error e, char const *func, char const *file, long line, throwmode mode = throwmode::plain)
    -> error_code {
    return {e, func, file, line, mode};
}

/// \brief Returns error_code(e, msg, mode).
inline auto make_error_code(error e, char const *msg, throwmode mode = throwmode::plain) -> error_code {
    return {e, msg, mode};
}
inline auto make_error_code(error e, char const *msg, char const *func, char const *file, long line,
                            throwmode mode = throwmode::plain) -> error_code {
    return {e, msg, func, file, line, mode};
}

/// \brief Returns error_code(e, msg, mode).
inline auto make_error_code(error e, std::string const &msg, throwmode mode = throwmode::plain) -> error_code {
    return {e, msg, mode};
}
inline auto make_error_code(error e, std::string const &msg, char const *func, char const *file, long line,
                            throwmode mode = throwmode::plain) -> error_code {
    return {e, msg, func, file, line, mode};
}
inline auto make_error_code(std::exception_ptr const &e) -> error_code {
    return error_code(e);
}
///@}

/// \brief Returns error_code(einsums::error:success, "success", mode).
inline auto make_success_code(throwmode mode = throwmode::plain) -> error_code {
    return error_code(mode);
}
} // namespace einsums

#include <einsums/errors/throw_exception.hpp>
