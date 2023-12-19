//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/Version.hpp>

#include <fmt/format.h>

#include <sstream>

namespace einsums {

std::string full_version_as_string() {
    return fmt::format("{}.{}.{}", EINSUMS_VERSION_MAJOR, EINSUMS_VERSION_MINOR, EINSUMS_VERSION_PATCH);
}

std::string full_build_string() {
    std::ostringstream strm;
    strm << "{config}:\n"
         << configuration_string() << "{version}: " << build_string() << "\n"
         << "{build-type}: " << build_type() << "\n"
         << "{date}: " << build_date_time() << "\n";

    return strm.str();
}

std::string configuration_string() {
    std::ostringstream strm;

    strm << "Einsums:\n";

    char const *const *p = einsums::config_strings;
    while (*p)
        strm << "  " << *p++ << "\n";
    strm << "\n";

    return strm.str();
}

std::string build_string() {
    return fmt::format("V{}{}, Git: {:.10}", full_version_as_string(), EINSUMS_VERSION_TAG, EINSUMS_HAVE_GIT_COMMIT);
}

std::string complete_version() {
    std::string version = fmt::format("Version:\n"
                                      "  Einsums: {}\n"
                                      "\n"
                                      "Build:\n"
                                      "  Type: {}\n"
                                      "  Date: {}\n",
                                      build_string(), build_type(), build_date_time());

    return version;
}

std::string build_date_time() {
    return std::string(__DATE__) + " " + __TIME__;
}

} // namespace einsums