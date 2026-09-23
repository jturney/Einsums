//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/StringUtil/StringOps.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN()

std::string difference(std::string const &st1, std::string const &st2) {
    std::string out = st1;

    for (char i : st2) {
        size_t index = out.find(i);
        if (index < out.size()) {
            // One character. erase(index) alone removes everything from index to the end, which
            // emptied the result on the first match and made any two index strings look alike.
            out.erase(index, 1);
        }
    }

    return out;
}

std::string reverse(std::string const &str) {
    return std::string{str.rbegin(), str.rend()};
}

EINSUMS_NAMESPACE_END()