#pragma once

#include "einsums/_Common.hpp"

#include <source_location>

namespace einsums {

namespace detail {

/**
 * @brief Remove identifying information from a path, such as directory structure and usernames.
 *
 * If EINSUMS_ANONYMIZE is disabled, this simply returns the path passed in.
 * Otherwise, this starts by checking for the src/ directory, the tests/ directory, or the timing/ directory.
 * If any of those are found, then all directories above those are removed, and /git/ is prepended to the
 * resulting string, since these directories should only exist in a local Git repository for Einsums.
 * If none of those are found, then it looks for include/, since this is where Einsums' headers will be
 * made available once installed. If it finds this, then it removes all other directories from before this
 * and prepends /install/ to the string, since the error is likely thrown from an installed file.
 * If none of those are found, then it defaults to just returning the file path without any transformations.
 */
EINSUMS_EXPORT std::string anonymize(std::string fpath);
} // namespace detail

/**
 * @struct EinsumsException
 *
 * @brief Represents an exception that includes information on where it was thrown from.
 */
struct EinsumsException : std::exception {
  private:
    std::string _what;

  public:
    EinsumsException(const char *file, const char *line, const char *function, const char *message) {
        _what += file;
        _what += ":";
        _what += line;
        _what += ":\nIn function ";
        _what += function;
        _what += ":\n";
        _what += message;
    }

    EinsumsException(const char *file, const char *line, const char *function, const std::string &message) {
        _what += file;
        _what += ":";
        _what += line;
        _what += ":\nIn function ";
        _what += function;
        _what += ":\n";
        _what += message;
    }

    const char *what() const noexcept override { return _what.c_str(); }

    EinsumsException &operator=(const EinsumsException &other) noexcept {
        _what = other._what;
        return *this;
    }
};

} // namespace einsums

/**
 * @def EINSUMSEXCEPTION
 *
 * @brief Create an EinsumsException with the specified message.
 *
 * This creates the exception and initializes it with the file it was thrown from, the line it was
 * thrown from within the file, and the name of the function that threw it. It also adds a user-specified
 * message.
 */
#define EINSUMSEXCEPTION(what)                                                                                                             \
    einsums::EinsumsException(einsums::detail::anonymize(__FILE__).c_str(), EINSUMS_STRINGIFY(__LINE__),                                   \
                              std::source_location::current().function_name(), (what))
