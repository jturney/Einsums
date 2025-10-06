//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TypeSupport/TypeName.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Type Names") {
    INFO(einsums::detail::get_type_name_string_view<int>());

    REQUIRE(einsums::detail::get_type_name_string_view<int>() == "int");
}