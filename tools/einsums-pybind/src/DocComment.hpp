//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// DocComment — parse a raw Doxygen doc-comment body into structured,
// reST-ready fields.
//
// DocExtractor (DocExtractor.hpp) hands us the comment text with the ///,
// /**, */, and leading-* markers already stripped, but otherwise verbatim:
// `@brief`, `@param x ...`, `@versionadded{1.0.0}`, `@f$ ... @f$` math,
// `@code ... @endcode`, and so on. The docs renderer wants those split into
// a brief line, a detail body, and per-parameter / return / throws entries,
// with the inline and block Doxygen commands converted to reStructuredText.
//
// This parser is intentionally scoped to the command set that actually
// appears in the Einsums headers (see the grep audit in the commit that
// introduced it), not the whole of Doxygen. Unknown `@command` tokens are
// passed through with the leading marker stripped rather than dropped, so
// nothing silently vanishes.
//
// Output is consumed by DocsJson (the `doc_structured` object) and is the
// shared normalizer Option 2 (a Breathe replacement) will reuse against
// Doxygen XML doc bodies.

#include <string>
#include <vector>

namespace einsums::pybind {

// A named doc entry: @param / @tparam / @throws. `name` is the parameter or
// exception-type name; `description` is reST-ready text (inline commands
// already converted).
struct DocEntry {
    std::string name;
    std::string description;
};

// Structured form of a doc comment. Every text field is reST-ready: inline
// Doxygen commands (`@c`, `@p`, `@ref`, `@f$math@f$`, ...) are converted and
// block constructs (`@code`, `@versionadded`, `@note`, ...) are rendered as
// reST directives inside `detail`.
struct DocComment {
    std::string           brief;   // one-line summary (from @brief or leading paragraph)
    std::string           detail;  // remaining prose + converted block directives
    std::vector<DocEntry> params;  // @param
    std::vector<DocEntry> tparams; // @tparam (C++ template params; usually omitted from Python pages)
    std::string           returns; // @return / @returns
    std::vector<DocEntry> throws_; // @throws / @throw / @exception  (name = exception type)

    [[nodiscard]] bool empty() const {
        return brief.empty() && detail.empty() && params.empty() && tparams.empty() && returns.empty() && throws_.empty();
    }
};

// Parse `raw` (a marker-stripped Doxygen comment body) into structured form.
DocComment parse_doc_comment(std::string const &raw);

} // namespace einsums::pybind
