//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/config/ConfigStrings.hpp>
#include <einsums/config/version.hpp>
#include <einsums/preprocessor/stringize.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace einsums {

/// Returns the major einsums version
constexpr auto major_version() -> std::uint8_t {
    return EINSUMS_VERSION_MAJOR;
}

// Returns the minor einsums version.
constexpr auto minor_version() -> std::uint8_t {
    return EINSUMS_VERSION_MINOR;
}

// Returns the sub-minor/patch-level einsums version.
constexpr auto patch_version() -> std::uint8_t {
    return EINSUMS_VERSION_PATCH;
}

// Returns the full einsums version.
constexpr auto full_version() -> std::uint32_t {
    return EINSUMS_VERSION_FULL;
}

// Returns the full einsums version.
EINSUMS_EXPORT auto full_version_as_string() -> std::string;

// Returns the tag.
constexpr auto tag() -> std::string_view {
    return EINSUMS_VERSION_TAG;
}

// Return the einsums configuration information.
EINSUMS_EXPORT auto configuration_string() -> std::string;

// Returns the einsums version string.
EINSUMS_EXPORT auto build_string() -> std::string;

// Returns the einsums build type ('Debug', 'Release', etc.)
constexpr auto build_type() -> std::string_view {
    return EINSUMS_PP_STRINGIZE(EINSUMS_BUILD_TYPE);
}

// Returns the einsums build date and time
EINSUMS_EXPORT auto build_date_time() -> std::string;

// Returns the einsums full build information string.
EINSUMS_EXPORT auto full_build_string() -> std::string;

// Returns the copyright string.
constexpr auto copyright() -> std::string_view {
    char const *const copyright =
        "Einsums\n\n"
        "Copyright (c) The Einsums Developers,\n"
        "https://github.com/E/Einsums\n\n"
        "Copyright (c) The Einsums Developers. All rights reserved.\n"
        "Licensed under the MIT License. See LICENSE.txt in the project root for license information.\n";
    return copyright;
}

// Returns the full version string.
EINSUMS_EXPORT auto complete_version() -> std::string;

} // namespace einsums