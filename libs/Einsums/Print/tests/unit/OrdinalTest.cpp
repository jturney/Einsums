//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Formatting ordinals", "[print]") {
    using namespace einsums;

    std::string formatted = fmt::format("{}", print::ordinal{1});

    REQUIRE(formatted == "1st");
}
