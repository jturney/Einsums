//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// DocsJson — serialize the binding IR to a documentation-oriented JSON
// document.
//
// This is the `--emit-docs-json` mode of einsums-pybind. Where the C++
// emitter (Emitter.hpp) turns the IR into pybind11 and the .pyi emitter
// (PyiEmitter.hpp) turns it into type stubs, this turns it into a stable,
// tool-agnostic JSON description of the Python-facing surface — the input
// a documentation generator consumes instead of hand-written reST.
//
// The schema is intentionally a faithful, lossless-ish projection of the
// IR (see IR.hpp): every doc-relevant field a renderer might want is
// present, in both its C++ form and its Python-stub form, so the renderer
// — not this tool — decides what to show. The top-level shape is:
//
//   {
//     "schema_version": 1,
//     "module": "einsums",
//     "classes":   [ <class>...   ],
//     "functions": [ <function>... ],
//     "enums":     [ <enum>...     ]
//   }
//
// See emit_docs_json's implementation for the per-entity field list. The
// schema is versioned via "schema_version" so a downstream renderer can
// detect incompatible changes.

#include <string>

#include "IR.hpp"

namespace einsums::pybind {

// Current docs-JSON schema version. Bump on any incompatible change to the
// emitted shape so downstream renderers can guard.
// v2 adds a per-entity `doc_structured` object (brief/detail/params/
// tparams/returns/throws), reST-ready, parsed from the raw Doxygen `doc`.
inline constexpr int k_docs_json_schema_version = 2;

// Serialize `module_` to a pretty-printed JSON string per the schema above.
// `module_name` is recorded as the document's "module" field (the Python
// import name, e.g. "einsums").
std::string emit_docs_json(Module const &module_, std::string const &module_name);

} // namespace einsums::pybind
