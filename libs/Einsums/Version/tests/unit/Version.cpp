//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Version.hpp>

#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;

// full_version_as_string() left out the tag, so an alpha reported itself as the final release:
// Python's einsums.__version__ said 2.0.0 for 2.0.0a1.
TEST_CASE("The version string carries the tag", "[Version]") {
    std::string const numeric =
        std::to_string(major_version()) + "." + std::to_string(minor_version()) + "." + std::to_string(patch_version());
    CHECK(full_version_as_string() == numeric + std::string(tag()));
}

TEST_CASE("The build string names the version once", "[Version]") {
    std::string const build           = build_string();
    std::string const expected_prefix = "v" + full_version_as_string() + ", Git: ";
    CHECK(build.starts_with(expected_prefix));
}
