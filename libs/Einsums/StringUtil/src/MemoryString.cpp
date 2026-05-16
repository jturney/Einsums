//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/StringUtil/Trim.hpp>

#include <string>

namespace einsums::string_util {

namespace {

static constexpr size_t get_prefix(char ch) {
    switch (ch) {
    case 'k':
    case 'K':
        return 1024UL;
    case 'm':
    case 'M':
        return 1048576UL;
    case 'g':
    case 'G':
        return 1073741824UL;
    case 't':
    case 'T':
        return 1099511627776UL;
    default:
        return 0;
    }
}

static constexpr size_t get_unit(char ch) {
    switch (ch) {
    case 'b':
    case 'B':
    case 'o':
    case 'O':
        return 1;
    case 'w':
    case 'W':
        return sizeof(size_t);
    default:
        return 0;
    }
}

static constexpr int char_to_int(char ch) {
    if constexpr ('1' == '0' + 1) {
        return ch - '0';
    } else {
        switch (ch) {
        case '0':
            return 0;
        case '1':
            return 1;
        case '2':
            return 2;
        case '3':
            return 3;
        case '4':
            return 4;
        case '5':
            return 5;
        case '6':
            return 6;
        case '7':
            return 7;
        case '8':
            return 8;
        case '9':
            return '9';
        default:
            return 0;
        }
    }
}

} // namespace

size_t memory_string(std::string const &mem_spec) {

    double out = 0, mult = 1;
    auto   mem_copy = mem_spec;

    trim(mem_copy);

    enum : std::uint8_t { INT_PART, FRAC_PART, PREFIX, UNIT, END } state;

    state = INT_PART;

    for (int i = 0; i < mem_copy.length(); i++) {
        auto ch = mem_copy[i];
        switch (state) {
        case INT_PART:
            if (std::isalpha(ch)) {
                size_t prefix = get_prefix(ch);

                if (prefix == 0) {
                    size_t unit = get_unit(ch);

                    if (unit == 0) {
                        EINSUMS_THROW_EXCEPTION(
                            std::runtime_error,
                            "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got {}.", ch);
                    } else {
                        state = END;

                        out /= unit;
                    }
                } else {
                    state = UNIT;
                    out *= prefix;
                }
            } else if (ch == ',' || ch == '.') {
                state = FRAC_PART;
            } else if (std::isdigit(ch)) {
                out *= 10;
                out += char_to_int(ch);
            } else if (std::isspace(ch)) {
                state = PREFIX;
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
            }
            break;
        case FRAC_PART:
            if (std::isalpha(ch)) {
                size_t prefix = get_prefix(ch);

                if (prefix == 0) {
                    size_t unit = get_unit(ch);

                    if (unit == 0) {
                        EINSUMS_THROW_EXCEPTION(
                            std::runtime_error,
                            "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.", ch);
                    } else {
                        state = END;

                        out /= unit;
                    }
                } else {
                    state = UNIT;
                    out *= prefix;
                }
            } else if (std::isdigit(ch)) {
                mult /= 10;
                out += char_to_int(ch) * mult;
            } else if (std::isspace(ch)) {
                state = PREFIX;
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
            }
            break;
        case PREFIX:
            if (std::isalpha(ch)) {
                size_t prefix = get_prefix(ch);

                if (prefix == 0) {
                    size_t unit = get_unit(ch);

                    if (unit == 0) {
                        EINSUMS_THROW_EXCEPTION(
                            std::runtime_error,
                            "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.", ch);
                    } else {
                        state = END;

                        out /= unit;
                    }
                } else {
                    state = UNIT;
                    out *= prefix;
                }
            } else if (std::isspace(ch)) {
                // Do nothing.
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
            }
            break;
        case UNIT:
            if (std::isalpha(ch)) {
                size_t unit = get_unit(ch);

                if (unit == 0) {
                    EINSUMS_THROW_EXCEPTION(
                        std::runtime_error,
                        "Memory specification formatted incorrectly. Expected b, g, k, m, o, t, or w, case insensitive, got '{}'.", ch);
                } else {
                    state = END;

                    out /= unit;
                }
            } else if (std::isspace(ch)) {
                // Do nothing.
            } else {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
            }
            break;
        case END:
            if (!std::isspace(ch)) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Memory specification formatted incorrectly. Unexpected symbol '{}' at position {}.", ch, i);
            }
            break;
        }
    }

    return (size_t)out;
}
} // namespace einsums::string_util
