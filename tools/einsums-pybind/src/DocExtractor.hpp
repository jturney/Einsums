//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

namespace einsums::pybind {

// Returns the cleaned doxygen comment associated with `decl`, or an empty
// string if the declaration has no doc-comment. Cleaning strips ``///``,
// ``/**``, ``*/``, and leading ``*`` markers but otherwise preserves the
// text verbatim — Phase 2 keeps escaping/markdown conversion out of scope
// (Phase 3 polish handles ``\param`` / ``\return`` reflow).
std::string extract_doc(clang::Decl const *decl, clang::ASTContext &ctx);

} // namespace einsums::pybind
