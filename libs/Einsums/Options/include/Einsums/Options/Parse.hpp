//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Source.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/*
 * The driver's half of the module: reading a config file, applying the
 * environment, walking argv, and rendering help. Only the runtime's
 * initialization path and the module's own tests include this.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

struct OptionBase;

// -------------------------- Enumeration ----------------------------------- //

/**
 * @brief One registered option, flattened into data.
 *
 * What a descriptor declares, minus the C++ type: enough for a caller that
 * never names the option in source - a binding layer, a documentation
 * generator - to spell the flag, know what kind of value it takes, and report
 * what it defaults to.
 *
 * @versionadded{2.0.0}
 */
struct RegisteredOption {
    /// The command-line long name, e.g. `einsums:log:level`.
    std::string name;
    /// The config key derived from @ref name.
    std::string key;
    /// The `--no-` spelling, for a flag; empty for a value option.
    std::string negated_name;
    /// The `--help` description and heading.
    std::string help;
    std::string category;
    /// The placeholder shown in help for a value option, e.g. `PORT`.
    std::string value_name;

    OptionKind kind = OptionKind::Value;
    OptionType type = OptionType::Bool;

    /// The declared default, in whichever member @ref type selects.
    bool         default_bool   = false;
    std::int64_t default_int    = 0;
    double       default_double = 0.0;
    std::string  default_string;

    /// True when the default came from a provider run at registration rather
    /// than from a literal, so the value above is this process's answer and
    /// not something a generator may print as the declared default.
    bool computed_default = false;
};

/**
 * @brief Every option a descriptor has registered, in registration order.
 *
 * The generated `--no-` twin of a flag is not listed: it is a spelling of the
 * option already named here, not a second option. Keys reached only through
 * the dynamic API are not listed either, since no descriptor claims them.
 *
 * This is the enumeration the Python binding layer builds its argv from, so
 * that the flag spellings live in exactly one place.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::vector<RegisteredOption> registered_options();

// -------------------------- Config reader --------------------------------- //

/**
 * @brief Read a `key = value` or flat-JSON config file into a map.
 *
 * Keys are lower-cased; a missing or unreadable file yields an empty map.
 */
EINSUMS_EXPORT std::map<std::string, std::string, std::less<>> read_config(std::string_view path);

// -------------------------- Help / Version -------------------------------- //

/// The width to wrap help text at, honouring COLUMNS and the terminal size.
EINSUMS_EXPORT std::size_t terminal_width();

/// Render the full `--help` text.
EINSUMS_EXPORT std::string format_help(std::string_view prog);

EINSUMS_EXPORT void print_help(std::string_view prog, std::FILE *out = stdout);

EINSUMS_EXPORT void print_version(std::string_view prog, std::string_view ver, std::FILE *out = stdout);

namespace detail {

/// The registered option carrying this long name, or null.
EINSUMS_EXPORT OptionBase *find_long(std::string_view name);

/// The registered option carrying this short name, or null.
EINSUMS_EXPORT OptionBase *find_short(char c);

/// Every registered positional option, in declaration order.
EINSUMS_EXPORT std::vector<OptionBase *> positional_options();

/// The first long name declared by two different options, if any.
EINSUMS_EXPORT std::optional<std::string> duplicate_long_name();

/// Apply every environment variable that a registered option names.
EINSUMS_EXPORT bool apply_environment(std::string &error);

} // namespace detail

// -------------------------- Parser ---------------------------------------- //

EINSUMS_EXPORT ParseResult parse_internal(std::span<std::string const> args, char const *programName, std::string_view version,
                                          std::map<std::string, std::string, std::less<>> const *config,
                                          std::vector<std::string>                              *unknown_args = nullptr);

/**
 * @brief Parse command-line arguments into the previously registered options.
 *
 * Values are resolved in increasing order of precedence: the programmer's
 * `Default(...)`, then any environment variable the option names, then the
 * command line.
 *
 * @param args command-line arguments, argv[0] included
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse(std::span<std::string const> args, char const *programName = nullptr, std::string_view version = {},
                         std::vector<std::string> *unknown_args = nullptr) {
    return parse_internal(args, programName, version, nullptr, unknown_args);
}

/**
 * @brief Parse command-line arguments, consulting a config file first.
 *
 * Precedence runs default < config file < environment < command line.
 *
 * @param args command-line arguments, argv[0] included
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param config_path key=value or simple json config file read before the environment
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse_with_config(std::span<std::string const> args, char const *programName = nullptr, std::string_view version = {},
                                     std::string_view config_path = {}, std::vector<std::string> *unknown_args = nullptr) {
    auto const kv = read_config(config_path);
    return parse_internal(args, programName, version, &kv, unknown_args);
}

EINSUMS_NAMESPACE_END(cl)
