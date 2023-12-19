//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/Config.hpp>
#include <einsums/config/ConfigStrings.hpp>
#include <einsums/config/Version.hpp>
#include <einsums/preprocessor/Stringize.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace einsums {

/// Returns the major einsums version
constexpr std::uint8_t major_version() {
    return EINSUMS_VERSION_MAJOR;
}

// Returns the minor einsums version.
constexpr std::uint8_t minor_version() {
    return EINSUMS_VERSION_MINOR;
}

// Returns the sub-minor/patch-level einsums version.
constexpr std::uint8_t patch_version() {
    return EINSUMS_VERSION_PATCH;
}

// Returns the full einsums version.
constexpr std::uint32_t full_version() {
    return EINSUMS_VERSION_FULL;
}

// Returns the full einsums version.
EINSUMS_EXPORT std::string full_version_as_string();

// Returns the tag.
constexpr std::string_view tag() {
    return EINSUMS_VERSION_TAG;
}

// Return the einsums configuration information.
EINSUMS_EXPORT std::string configuration_string();

// Returns the einsums version string.
EINSUMS_EXPORT std::string build_string();

// Returns the einsums build type ('Debug', 'Release', etc.)
constexpr std::string_view build_type() {
    return EINSUMS_PP_STRINGIZE(EINSUMS_BUILD_TYPE);
}

// Returns the einsums build date and time
EINSUMS_EXPORT std::string build_date_time();

// Returns the einsums full build information string.
EINSUMS_EXPORT std::string full_build_string();

// Returns the copyright string.
constexpr std::string_view copyright() {
    char const *const copyright = "Einsums\n\n"
                                  "Copyright (c) The Einsums Developers,\n"
                                  "https://github.com/E/Einsums\n\n"
                                  "Copyright (c) The Einsums Developers. All rights reserved.\n"
                                  "Licensed under the MIT License. See LICENSE.txt in the project root for license information.\n";
    return copyright;
}

// Returns the full version string.
EINSUMS_EXPORT std::string complete_version();

} // namespace einsums