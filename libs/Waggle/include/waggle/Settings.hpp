//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <waggle/Config.hpp>

WAGGLE_NAMESPACE_BEGIN

/// Everything the profiler can be told, with the values it uses when nobody says otherwise.
struct Settings {
    bool         record{true};               ///< Record zones and annotations.
    bool         report{true};               ///< Write the text report at the last finalize.
    std::string  report_file{"profile.txt"}; ///< Where the report goes.
    bool         report_append{false};       ///< Append the report to @ref report_file rather than replace it.
    bool         report_detailed{false};     ///< Report every zone's min, max and counters.
    std::string  save{};                     ///< Write a session file here at the last finalize; empty for none.
    bool         server{false};              ///< Run the live-viewing server.
    std::int64_t port{19216};                ///< The server's port.
    bool         wait_for_viewer{false};     ///< Hold the program until a viewer connects.
    std::int64_t max_distinct_children{256}; ///< Distinct child names per node before the rest fold into "(other)"; 0 for no limit.
};

/// Some settings to change; an empty member leaves that setting alone.
struct SettingsUpdate {
    std::optional<bool>         record;
    std::optional<bool>         report;
    std::optional<std::string>  report_file;
    std::optional<bool>         report_append;
    std::optional<bool>         report_detailed;
    std::optional<std::string>  save;
    std::optional<bool>         server;
    std::optional<std::int64_t> port;
    std::optional<bool>         wait_for_viewer;
    std::optional<std::int64_t> max_distinct_children;
};

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

    /// The settings in force.
    [[nodiscard]] auto current() const -> Settings const & { return _settings; }

  private:
    Settings _settings;

    /// Which settings @ref configure has set, in @ref SettingsUpdate's member order.
    bool _explicit[10]{}; // NOLINT(modernize-avoid-c-arrays)
};

WAGGLE_NAMESPACE_END
