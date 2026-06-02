//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <vector>

#include "IR.hpp"
#include "llvm/ADT/StringRef.h"

namespace einsums::pybind {

// Scan raw header source text for *documented* preprocessor macros: a doc
// comment (``/** ... */`` / ``/*! ... */`` or a run of ``///``) immediately
// followed by a ``#define NAME`` (optionally function-like ``NAME(args)``).
//
// Reads raw source text — all preprocessor branches — so a macro documented
// inside a compiler-specific ``#if`` branch is still found regardless of which
// branch the current build would take. Undocumented ``#define``s are ignored,
// mirroring the "document only documented entities" rule for the rest of the
// docs-mode surface. Macros are not AST declarations, so this is the only way
// to recover their documentation.
std::vector<BoundMacro> scan_macros(llvm::StringRef source);

} // namespace einsums::pybind
