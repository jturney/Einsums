//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdio>
#include <string>

namespace einsums {

/// @brief Escape a string for safe embedding in a JSON value.
///
/// Handles quotes, backslashes, and control characters (< 0x20).
inline auto json_escape(std::string const &s) -> std::string {
    std::string out;
    out.reserve(s.size());
    for (char const c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                // NOLINTNEXTLINE(modernize-avoid-c-arrays)
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", static_cast<int>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

} // namespace einsums
