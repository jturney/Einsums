//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
// clang-format on

namespace einsums::program_options {

inline auto strip_prefixes(std::string const &text) -> std::string {
    // "--for-bar" -> "foo-bar"
    std::string::size_type i = text.find_first_not_of("-/");
    if (i == std::string::npos) {
        return text;
    } else {
        return text.substr(i);
    }
}

/** Base clas for all errors in this module. */
class EINSUMS_ALWAYS_EXPORT Error : public std::logic_error {
  public:
    Error(std::string const &xwhat) : std::logic_error(xwhat) {}
};

/** Error thrown when there are too man positional options. This is a programming error. */
class EINSUMS_ALWAYS_EXPORT TooManyPositionalOptionsError : public Error {
  public:
    TooManyPositionalOptionsError() : Error("too many positional options have been specified on the command line") {}
};

/** Class thrown when there are programming error related to style */
class EINSUMS_ALWAYS_EXPORT InvalidCommandLineStyle : public Error {
  public:
    InvalidCommandLineStyle(std::string const &msg) : Error(msg) {}
};

/** Class thrown if config file can not be read */
class EINSUMS_ALWAYS_EXPORT ReadingFile : public Error {
  public:
    ReadingFile(char const *filename) : Error(std::string("can not read options configuration file '").append(filename).append("'")) {}
};

/** Base class for most exceptions in the library.
 *
 *  Substitutes the values for the parameter name
 *      placeholders in the template to create the human
 *      readable error message
 *
 *  Placeholders are surrounded by % signs: %example%
 *      Poor man's version of format
 *
 *  If a parameter name is absent, perform default substitutions
 *      instead so ugly placeholders are never left in-place.
 *
 *  Options are displayed in "canonical" form
 *      This is the most unambiguous form of the
 *      *parsed* option name and would correspond to
 *      option_description::format_name()
 *      i.e. what is shown by print_usage()
 *
 *  The "canonical" form depends on whether the option is
 *      specified in short or long form, using dashes or slashes
 *      or without a prefix (from a configuration file)
 *
 */
class EINSUMS_ALWAYS_EXPORT ErrorWithOptionName : public Error {
  protected:
    /** can be
     *     0 = no prefix (config file option)
     *     AllowLong
     *     AllowDashForShort
     *     AllowSlashForShort
     *     AllowLongDisguise
     */
    int _option_style;

    /** substitutions from placeholds to values */
    std::map<std::string, std::string> _substitutions;
    using StringPair = std::pair<std::string, std::string>;
    std::map<std::string, StringPair> _substitution_defaults;

  public:
    /** template with placeholders */
    std::string _error_template;

    ErrorWithOptionName(std::string const &template_, std::string const &option_name = "", std::string const &original_token = "",
                        int option_style = 0);

    ~ErrorWithOptionName() noexcept = default;

    void set_substitute(std::string const &parameter_name, std::string const &value) { _substitutions[parameter_name] = value; }

    void set_substitute_default(std::string const &parameter_name, std::string const &from, std::string const &to) {
        _substitution_defaults[parameter_name] = std::make_pair(from, to);
    }

    void add_context(std::string const &option_name, std::string const &original_token, int option_style) {
        set_option_name(option_name);
        set_original_token(original_token);
        set_prefix(option_style);
    }

    void set_prefix(int option_style) { _option_style = option_style; }

    virtual void set_option_name(std::string const &option_name) { set_substitute("option", option_name); }

    [[nodiscard]] auto get_option_name() const -> std::string { return get_canonical_option_name(); }

    void set_original_token(std::string const &original_token) { set_substitute("original_token", original_token); }

    [[nodiscard]] auto what() const noexcept -> char const * override;

  protected:
    mutable std::string _message;

    virtual void substitute_placeholders(std::string const &error_template) const;

    void replace_token(std::string const &from, std::string const &to) const;

    auto get_canonical_option_name() const -> std::string;
    auto get_canonical_option_prefix() const -> std::string;
};

class EINSUMS_ALWAYS_EXPORT MultipleValue : public ErrorWithOptionName {
  public:
    MultipleValue() : ErrorWithOptionName("option '%canonical_option%' only takes a single argument") {}

    ~MultipleValue() noexcept = default;
};

class EINSUMS_ALWAYS_EXPORT RequiredOption : public ErrorWithOptionName {
  public:
    RequiredOption(std::string const &option_name)
        : ErrorWithOptionName("the option '%canonical_option%' is required but missing", "", option_name) {}

    ~RequiredOption() noexcept = default;
};

/** Base class of un-parsable options,
 *  when the desired option cannot be identified.
 *
 *
 *  It makes no sense to have an option name, when we can't match an option to the
 *      parameter
 *
 *  Having this a part of the error_with_option_name hierarchy makes error
 *      handling a lot easier, even if the name indicates some sort of
 *      conceptual dissonance!
 *
 **/
class EINSUMS_ALWAYS_EXPORT ErrorWithNoOptionName : public ErrorWithOptionName {
  public:
    ErrorWithNoOptionName(std::string const &template_, std::string const &original_token = "")
        : ErrorWithOptionName(template_, "", original_token) {}

    void set_option_name(std::string const &) override {}

    ~ErrorWithNoOptionName() noexcept = default;
};

class EINSUMS_ALWAYS_EXPORT UnknownOption : public ErrorWithNoOptionName {
  public:
    UnknownOption(std::string const &original_token = "")
        : ErrorWithNoOptionName("unrecognised option '%canonical_option%'", original_token) {}

    ~UnknownOption() noexcept = default;
};

class EINSUMS_ALWAYS_EXPORT AmbiguousOption : public ErrorWithNoOptionName {
  public:
    AmbiguousOption(std::vector<std::string> const &xalternatives)
        : ErrorWithNoOptionName("option '%canonical_option%' is ambiguous"), _alternatives(xalternatives) {}

    ~AmbiguousOption() noexcept = default;

    auto alternatives() const noexcept -> std::vector<std::string> const & { return _alternatives; }

  protected:
    void substitute_placeholders(std::string const &error_template) const override;

  private:
    std::vector<std::string> _alternatives;
};

class EINSUMS_ALWAYS_EXPORT InvalidSyntax : public ErrorWithOptionName {
  public:
    enum class Kind {
        LongNotAllowed = 30,
        LongAdjacentNotAllowed,
        ShortAdjacentNotAllowed,
        EmptyAdjacentParameter,
        MissingParameters,
        ExtraParameter,
        UnrecognizedLine
    };

    InvalidSyntax(Kind kind, std::string const &option_name = "", std::string const &original_token = "", int option_style = 0)
        : ErrorWithOptionName(get_template(kind), option_name, original_token, option_style), _kind(kind) {}

    ~InvalidSyntax() noexcept = default;

    auto kind() const -> Kind { return _kind; }

    virtual auto tokens() const -> std::string { return get_option_name(); }

  protected:
    auto get_template(Kind kind) -> std::string;
    Kind _kind;
};

class EINSUMS_ALWAYS_EXPORT InvalidConfigFileSyntax : public InvalidSyntax {
  public:
    InvalidConfigFileSyntax(std::string const &invalid_line, Kind kind) : InvalidSyntax(kind) {
        _substitutions["invalid_line"] = invalid_line;
    }

    ~InvalidConfigFileSyntax() noexcept {}

    /** Convenience functions for backwards compatibility */
    auto tokens() const -> std::string override {
        auto it = _substitutions.find("invalid_line");
        if (it != _substitutions.end()) {
            return it->second;
        }
        return "<unknown>";
    }
};

class EINSUMS_ALWAYS_EXPORT InvalidCommandLineSyntax : public InvalidSyntax {
  public:
    InvalidCommandLineSyntax(Kind kind, std::string const &option_name = "", std::string const &original_token = "", int option_style = 0)
        : InvalidSyntax(kind, option_name, original_token, option_style) {}
    ~InvalidCommandLineSyntax() noexcept {}
};

/** Class thrown when value of option is incorrect. */
class EINSUMS_ALWAYS_EXPORT ValidationError : public ErrorWithOptionName {
  public:
    enum class Kind { MultipleValuesNotAllowed = 30, AtLeastOneValueRequired, InvalidBoolValue, InvalidOptionValue, InvalidOption };

  public:
    ValidationError(Kind kind, std::string const &option_name = "", std::string const &original_token = "", int option_style = 0)
        : ErrorWithOptionName(get_template(kind), option_name, original_token, option_style), _kind(kind) {}

    ~ValidationError() noexcept {}

    auto kind() const -> Kind { return _kind; }

  protected:
    /** Used to convert kind_t to a related error text */
    auto get_template(Kind kind) -> std::string;
    Kind _kind;
};

/** Class thrown if there is an invalid option value given */
class EINSUMS_ALWAYS_EXPORT InvalidOptionValue : public ValidationError {
  public:
    InvalidOptionValue(std::string const &value);
    InvalidOptionValue(std::wstring const &value);
};

/** Class thrown if there is an invalid bool value given */
class EINSUMS_ALWAYS_EXPORT InvalidBoolValue : public ValidationError {
  public:
    InvalidBoolValue(std::string const &value);
};

} // namespace einsums::program_options

#include <einsums/config/WarningsSuffix.hpp>
