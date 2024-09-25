//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/init.hpp>
#include <einsums/string_util/from_string.hpp>

#include <einsums/testing.hpp>

TEST_CASE("get entry", "[init_runtime]") {
    std::string val = einsums::detail::get_config_entry("einsums.pu_step", "42");
    CHECK(!val.empty());
    CHECK(einsums::string_util::from_string<int>(val) == 1);
}