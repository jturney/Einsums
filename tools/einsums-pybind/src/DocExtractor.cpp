//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "DocExtractor.hpp"

#include <array>
#include <string>
#include <vector>

#include "clang/AST/RawCommentList.h"
#include "llvm/ADT/StringRef.h"

namespace einsums::pybind {

namespace {

// Strip a leading ``///``, ``//!``, ``/**``, or ``*`` plus surrounding
// whitespace from a single line of a doxygen comment. Prefixes are
// checked in length-descending order so ``///`` matches before ``//``.
llvm::StringRef strip_comment_markers(llvm::StringRef line) {
    static constexpr std::array<llvm::StringLiteral, 7> k_prefixes = {
        llvm::StringLiteral{"///"}, llvm::StringLiteral{"//!"}, llvm::StringLiteral{"/**"}, llvm::StringLiteral{"/*!"},
        llvm::StringLiteral{"//"},  llvm::StringLiteral{"/*"},  llvm::StringLiteral{"*"},
    };
    line = line.ltrim();
    for (auto const &prefix : k_prefixes) {
        if (line.starts_with(prefix)) {
            line = line.drop_front(prefix.size());
            break;
        }
    }
    if (line.ends_with("*/")) {
        line = line.drop_back(2);
    }
    return line.trim();
}

// True for "banner" rows of repeating decoration — e.g.
// ``///////////////////////////``, ``****************/``, ``=========``,
// ``-------``. These add no documentation value and otherwise survive
// marker stripping (a row of slashes only loses the first ``///``).
// We treat them as empty lines so leading/trailing trims drop them and
// interior banners collapse to a blank line.
bool is_banner_line(llvm::StringRef line) {
    if (line.empty()) {
        return false;
    }
    for (char const c : line) {
        if (c != '/' && c != '*' && c != '=' && c != '-' && c != '_' && c != '#' && c != ' ' && c != '\t') {
            return false;
        }
    }
    return true;
}

} // namespace

std::string extract_doc(clang::Decl const *decl, clang::ASTContext &ctx) {
    clang::RawComment const *raw = ctx.getRawCommentForDeclNoCache(decl);
    if (raw == nullptr) {
        return "";
    }
    llvm::StringRef const    text = raw->getRawText(ctx.getSourceManager());
    std::vector<std::string> lines;
    llvm::StringRef          remaining = text;
    while (!remaining.empty()) {
        auto [head, tail]       = remaining.split('\n');
        llvm::StringRef cleaned = strip_comment_markers(head);
        if (is_banner_line(cleaned)) {
            cleaned = llvm::StringRef{};
        }
        if (!cleaned.empty() || !lines.empty()) {
            lines.emplace_back(cleaned.str());
        }
        if (tail.data() == nullptr) {
            break;
        }
        remaining = tail;
    }
    while (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }
    std::string out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i != 0) {
            out += '\n';
        }
        out += lines[i];
    }
    return out;
}

} // namespace einsums::pybind
