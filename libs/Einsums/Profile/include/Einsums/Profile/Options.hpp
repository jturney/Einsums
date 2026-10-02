//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <cstdint>

// The profiler's options, declared where the profiler reads them.

EINSUMS_NAMESPACE_BEGIN(option)

/// Stop recording zones and annotations; a large speedup for many small operations.
inline constinit cl::ConfigOption<bool> ProfileDisable = cl::config_flag(
    "einsums:profile:disable", "Do not record profiling zones or annotations (large speedup for small operations)", "Profile", false);

/// Write the text report at shutdown.
inline constinit cl::ConfigOption<bool> ProfileReport =
    cl::config_flag("einsums:profile:report", "Generate a profile report on exit", "Profile", true);

/// Where that report goes.
inline constinit cl::ConfigOption<std::string> ProfileFilename =
    cl::config_opt<std::string>("einsums:profile:filename", "Profile report file name", "Profile", "profile.txt", "filename");

/// Append to the report file rather than truncating it.
inline constinit cl::ConfigOption<bool> ProfileAppend =
    cl::config_flag("einsums:profile:append", "Append to the profile file instead of truncating it", "Profile", true);

/// Report every zone rather than the summary.
inline constinit cl::ConfigOption<bool> ProfileDetailed =
    cl::config_flag("einsums:profile:detailed", "Print a detailed profile report", "Profile", false);

/// Write the session as JSON for the profile viewer. Empty means do not.
inline constinit cl::ConfigOption<std::string> ProfileSave = cl::config_opt<std::string>(
    "einsums:profile:save", "Save the profile session as JSON for the profile viewer", "Profile", "", "filename");

/// Enable the profile server for remote viewing. It is single-threaded and unauthenticated: local use only.
inline constinit cl::ConfigOption<bool> ProfileServer =
    cl::config_flag("einsums:profile:server", "Enable the profile server for remote viewing", "Profile", false);

/// The port the profile server listens on.
inline constinit cl::ConfigOption<std::int64_t> ProfilePort =
    cl::config_opt<std::int64_t>("einsums:profile:port", "Profile server port", "Profile", 19216, "PORT");

/// Block at startup until a viewer connects.
inline constinit cl::ConfigOption<bool> ProfileWaitForViewer =
    cl::config_flag("einsums:profile:wait-for-viewer", "Wait for the profiler viewer to connect before running", "Profile", false);

/// Distinct zone names one parent keeps as separate nodes. Past it, a new name joins that parent's
/// "(other)" node, so names built at run time (a graph's, a loop's) cannot grow the tree without
/// bound. Zero means no limit.
inline constinit cl::ConfigOption<std::int64_t> ProfileMaxDistinctChildren = cl::config_opt<std::int64_t>(
    "einsums:profile:max-distinct-children", "Distinct zone names kept per parent before the rest fold into \"(other)\"; 0 means no limit",
    "Profile", 256, "N");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// @brief The three options the Profiler reads when it constructs itself.
///
/// Out of line on purpose. The Profiler is constructed from a header, and a TU outside the library
/// gets its own copy of each descriptor, never registered, so an inline read there would always
/// return the default.
/// @{
EINSUMS_EXPORT bool profile_recording_disabled();
EINSUMS_EXPORT bool profile_server_enabled();
EINSUMS_EXPORT std::int64_t profile_server_port();
/// @}

EINSUMS_NAMESPACE_END()

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the profiler's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_Profile_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Profile = register_Einsums_Profile_options();
}

EINSUMS_NAMESPACE_END()
