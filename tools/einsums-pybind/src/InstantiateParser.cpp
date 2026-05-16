//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "InstantiateParser.hpp"

#include <cctype>

namespace einsums::pybind {

namespace {

// Trim ASCII whitespace from both ends of a string view.
std::string trim(std::string const &s) {
    std::size_t begin = 0;
    std::size_t end   = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
}

// Split `s` on `sep` while respecting ``< > ( )`` nesting depth so that
// commas inside template arguments or pseudo-grouping parens don't tear
// the payload apart.
std::vector<std::string> split_balanced(std::string const &s, char sep) {
    std::vector<std::string> parts;
    std::string              current;
    int                      angle = 0;
    int                      paren = 0;
    for (char const c : s) {
        if (c == '<') {
            ++angle;
        } else if (c == '>') {
            if (angle > 0) {
                --angle;
            }
        } else if (c == '(') {
            ++paren;
        } else if (c == ')') {
            if (paren > 0) {
                --paren;
            }
        }
        if (c == sep && angle == 0 && paren == 0) {
            parts.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty() || !parts.empty()) {
        parts.push_back(trim(current));
    }
    return parts;
}

// Pull out the keyword and parenthesized payload from a token like
// ``T(float, double)``. Returns an empty group if the token doesn't fit
// the ``IDENTIFIER(...)`` shape.
ParamGroup extract_param_group(std::string const &token) {
    ParamGroup        out;
    std::size_t const open = token.find('(');
    if (open == std::string::npos || token.empty() || token.back() != ')') {
        return out;
    }
    out.keyword             = trim(token.substr(0, open));
    std::string const inner = token.substr(open + 1, token.size() - open - 2);
    out.values              = split_balanced(inner, ',');
    return out;
}

} // namespace

InstantiateSpec parse_instantiate(std::string const &payload) {
    InstantiateSpec out;
    auto const      tokens = split_balanced(trim(payload), ',');
    if (tokens.empty()) {
        return out;
    }
    out.class_name = tokens.front();
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        ParamGroup g = extract_param_group(tokens[i]);
        if (!g.keyword.empty() && !g.values.empty()) {
            out.groups.push_back(std::move(g));
        }
    }
    return out;
}

InstantiateAsSpec parse_instantiate_as(std::string const &py_name, std::string const &type_expr) {
    InstantiateAsSpec out;
    out.py_name             = trim(py_name);
    std::string const expr  = trim(type_expr);
    std::size_t const open  = expr.find('<');
    std::size_t const close = expr.rfind('>');
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        out.class_name = expr; // not actually templated — Phase 8 may diagnose
        return out;
    }
    out.class_name = trim(expr.substr(0, open));
    out.type_args  = trim(expr.substr(open + 1, close - open - 1));
    return out;
}

std::vector<std::string> cross_product(std::vector<std::vector<std::string>> const &lists) {
    std::vector<std::string> result = {""};
    for (auto const &list : lists) {
        std::vector<std::string> next;
        next.reserve(result.size() * list.size());
        for (auto const &acc : result) {
            for (auto const &item : list) {
                if (acc.empty()) {
                    next.push_back(item);
                } else {
                    std::string combined = acc;
                    combined += ", ";
                    combined += item;
                    next.push_back(std::move(combined));
                }
            }
        }
        result = std::move(next);
    }
    return result;
}

std::string sanitize_python_name(std::string const &base, std::string const &type_args) {
    std::string out             = base;
    bool        last_underscore = !out.empty() && out.back() == '_';
    auto        append          = [&](char ch) {
        bool const ident = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
        if (ident) {
            out += ch;
            last_underscore = false;
        } else if (!last_underscore) {
            out += '_';
            last_underscore = true;
        }
    };
    append('_');
    for (char const ch : type_args) {
        append(ch);
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out;
}

} // namespace einsums::pybind
