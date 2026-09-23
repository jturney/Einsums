//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/StringUtil/StringOps.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

TEST_CASE("difference removes one character per match", "[StringUtil]") {
    // Each character of the second string cancels one occurrence in the first. It used to call
    // erase(index), which removes everything from index to the end: "ij" less "ki" came out empty,
    // and the permute that relies on this accepted index strings that named different axes.
    CHECK(einsums::difference("ij", "ki") == "j");
    CHECK(einsums::difference("ijk", "kij").empty());
    CHECK(einsums::difference("abc", "b") == "ac");
    CHECK(einsums::difference("aab", "a") == "ab");
    CHECK(einsums::difference("abc", "") == "abc");
    CHECK(einsums::difference("", "abc").empty());
}

TEST_CASE("reverse", "[StringUtil]") {
    CHECK(einsums::reverse("ijk") == "kji");
    CHECK(einsums::reverse("").empty());
}

TEST_CASE("find_char_with_position", "[StringUtil]") {
    std::vector<int> positions(3);
    einsums::find_char_with_position("kij", "ijk", &positions);
    CHECK(positions == std::vector<int>{2, 0, 1});
}
