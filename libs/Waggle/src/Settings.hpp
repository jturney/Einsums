//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Types.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

WAGGLE_NAMESPACE_BEGIN

/// Reads an environment variable: its value, or nothing if unset.
using EnvironmentReader = std::function<std::optional<std::string>(char const *name)>;

/**
 * @brief The process's settings and who set them.
 *
 * Three layers, highest first: values a library set explicitly with @ref configure, then the
 * ``WAGGLE_*`` environment variables, then the defaults in @ref Settings. Libraries share one
 * profiler, so when two set one setting to different values the first keeps it: the second is
 * reported, never applied, and no library's choice changes under it.
 */
class WAGGLE_EXPORT SettingsStore {
  public:
    /// The process environment.
    static auto process_environment() -> EnvironmentReader;

    /// Take every ``WAGGLE_*`` variable @p getenv knows, below anything set explicitly. A value that
    /// does not parse is skipped and its message added to the result.
    auto apply_environment(EnvironmentReader const &getenv) -> std::vector<std::string>;

    /// Set what @p update holds. A setting another call already set explicitly to a different value
    /// keeps that value; each such refusal is described in the result.
    auto configure(SettingsUpdate const &update) -> std::vector<std::string>;

    /// Set what @p update holds whoever set it before, and count it as explicit. For tests and tools
    /// that must restore a value; libraries use @ref configure.
    void override_settings(SettingsUpdate const &update);

    /// Add to @p update the setting @p key names (``report_file``, ``port``, ...) set to @p value,
    /// written as on a command line. False, leaving @p update alone, for an unknown name or a value
    /// that does not parse.
    static auto add_from_text(SettingsUpdate &update, std::string_view key, std::string const &value) -> bool;

    /// The current value of setting @p key as text, or nothing for an unknown name.
    [[nodiscard]] auto text(std::string_view key) const -> std::optional<std::string>;

    /// The settings in force.
    [[nodiscard]] auto current() const -> Settings const & { return _settings; }

  private:
    Settings _settings;

    /// Which settings @ref configure has set, in @ref SettingsUpdate's member order.
    bool _explicit[10]{}; // NOLINT(modernize-avoid-c-arrays)
};

WAGGLE_NAMESPACE_END
