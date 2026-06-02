//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "MacroScanner.hpp"

#include <cctype>
#include <string>
#include <vector>

#include "DocExtractor.hpp" // clean_raw_comment

namespace einsums::pybind {
namespace {

std::vector<llvm::StringRef> split_lines(llvm::StringRef s) {
    std::vector<llvm::StringRef> out;
    while (true) {
        auto [head, tail] = s.split('\n');
        out.push_back(head);
        if (tail.data() == nullptr) {
            break;
        }
        s = tail;
    }
    return out;
}

// If `line` is a ``#define NAME`` (allowing leading whitespace and spaces
// after ``#``), return NAME and fill function-like info; otherwise "".
std::string define_name(llvm::StringRef line, bool &func_like, std::vector<std::string> &params) {
    llvm::StringRef t = line.ltrim();
    if (!t.consume_front("#")) {
        return "";
    }
    t = t.ltrim();
    if (!t.consume_front("define")) {
        return "";
    }
    // Require whitespace between ``define`` and the name (reject ``#defined``).
    if (t.empty() || !(t.front() == ' ' || t.front() == '\t')) {
        return "";
    }
    t             = t.ltrim();
    std::size_t k = 0;
    while (k < t.size() && (std::isalnum(static_cast<unsigned char>(t[k])) != 0 || t[k] == '_')) {
        ++k;
    }
    if (k == 0) {
        return "";
    }
    std::string name = t.substr(0, k).str();
    // Function-like only when ``(`` directly abuts the name (no space).
    if (k < t.size() && t[k] == '(') {
        func_like = true;
        std::string cur;
        std::size_t p = k + 1;
        for (; p < t.size() && t[p] != ')'; ++p) {
            if (t[p] == ',') {
                params.push_back(llvm::StringRef(cur).trim().str());
                cur.clear();
            } else {
                cur += t[p];
            }
        }
        std::string const last = llvm::StringRef(cur).trim().str();
        if (!last.empty()) {
            params.push_back(last);
        }
    }
    return name;
}

} // namespace

std::vector<BoundMacro> scan_macros(llvm::StringRef source) {
    std::vector<BoundMacro>            out;
    std::vector<llvm::StringRef> const lines = split_lines(source);
    std::size_t const                  n     = lines.size();

    std::size_t i = 0;
    while (i < n) {
        llvm::StringRef const lt = lines[i].ltrim();
        std::string           raw_comment;
        std::size_t           after = i;

        if (lt.starts_with("/**") || lt.starts_with("/*!")) {
            std::size_t j = i;
            for (; j < n; ++j) {
                raw_comment += lines[j].str();
                raw_comment += '\n';
                if (lines[j].contains("*/")) {
                    break;
                }
            }
            after = j + 1;
        } else if (lt.starts_with("///")) {
            std::size_t j = i;
            for (; j < n && lines[j].ltrim().starts_with("///"); ++j) {
                raw_comment += lines[j].str();
                raw_comment += '\n';
            }
            after = j;
        } else {
            ++i;
            continue;
        }

        // Find the #define the comment documents: skip blank lines and any
        // preprocessor conditional scaffolding (``#if``/``#ifdef``/``#ifndef``/
        // ``#elif``/``#else``/``#endif``) between the doc comment and the
        // ``#define``. This handles the common pattern of a doc comment placed
        // before a conditional definition block, e.g.
        //   /** ... */
        //   #if EINSUMS_ACTIVE_LOG_LEVEL <= 0
        //   #    define EINSUMS_LOG_TRACE(...) ...
        //   #else
        //   #    define EINSUMS_LOG_TRACE(...)
        //   #endif
        // Stop at the first non-blank, non-conditional line; if it is not a
        // ``#define`` the comment documents something else (or nothing).
        std::size_t k = after;
        while (k < n) {
            llvm::StringRef const trimmed = lines[k].trim();
            if (trimmed.empty()) {
                ++k;
                continue;
            }
            llvm::StringRef d = lines[k].ltrim();
            if (d.consume_front("#")) {
                d = d.ltrim();
                if (d.starts_with("if") || d.starts_with("elif") || d.starts_with("else") || d.starts_with("endif")) {
                    ++k;
                    continue;
                }
            }
            break;
        }
        if (k < n) {
            // Join backslash-continued lines so a multi-line ``#define`` (e.g.
            // the variadic ``EINSUMS_PP_ARGN_(_1, ..., \``) yields a complete
            // parameter list rather than a stray trailing ``\``.
            std::string joined = lines[k].str();
            std::size_t c      = k;
            while (!joined.empty() && joined.back() == '\\') {
                joined.pop_back();
                if (++c >= n) {
                    break;
                }
                joined += lines[c].str();
            }
            bool                     func_like = false;
            std::vector<std::string> params;
            std::string const        name = define_name(joined, func_like, params);
            if (!name.empty()) {
                std::string doc = clean_raw_comment(raw_comment);
                // Honor Doxygen visibility markers: ``\cond``/``@cond`` and
                // ``\internal``/``@internal`` mean "do not document". (The
                // ``\cond NOINTERNAL`` guard on internal helper macros like
                // ``EINSUMS_ASSERT_`` is the common case.) Also skip a comment
                // that cleaned to nothing.
                bool const internal = doc.empty() || doc.find("\\cond") != std::string::npos || doc.find("@cond") != std::string::npos ||
                                      doc.find("\\internal") != std::string::npos || doc.find("@internal") != std::string::npos;
                if (!internal) {
                    BoundMacro m;
                    m.name             = name;
                    m.qualified_name   = name; // macros are not namespaced
                    m.doc              = std::move(doc);
                    m.is_function_like = func_like;
                    m.params           = std::move(params);
                    out.push_back(std::move(m));
                }
            }
        }
        i = after;
    }
    return out;
}

} // namespace einsums::pybind
