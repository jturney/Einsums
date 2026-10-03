//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// The values Waggle's C++ interface passes: settings, a client's description, a diagnostic's level.
/// All of it header-only, crossing the C interface field by field.

#include <Waggle/Config.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

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

/// How much a profiler message matters.
enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

/// Answers a viewer request: receives the request's params as a JSON object, returns JSON.
using RequestHandler = std::function<std::string(std::string const &params)>;

/// Produces one JSON value for a session file.
using SessionSection = std::function<std::string()>;

/// A library using the profiler, as viewers and session files name it.
struct ClientInfo {
    std::string name;
    std::string version{};
    std::string git_commit{};
    std::string git_branch{};
    bool        git_dirty{false};
    std::string build_type{};
};

/// Receives the profiler's own messages: a server that could not bind, a setting refused.
using DiagnosticHandler = std::function<void(DiagnosticLevel level, std::string_view message)>;

WAGGLE_NAMESPACE_END
