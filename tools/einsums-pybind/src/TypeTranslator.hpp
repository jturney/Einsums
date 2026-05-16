//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

namespace einsums::pybind {

// Phase-2 stub: returns a pretty-printed C++ form of `type` suitable for
// emission in pybind11 binding code. Currently relies on Clang's own
// PrintingPolicy; Phase 3 will extend this to handle holder rewrites
// (e.g. unique_ptr<T> -> shared_ptr<T> when an EINSUMS_PYBIND_HOLDER
// directive is in effect) and pybind11-specific type substitutions.
std::string translate_type(clang::QualType type, clang::ASTContext const &ctx);

// Best-effort Python-stub form of `type`, suitable for emission in a
// `.pyi` file consumed by pyright. Maps fundamentals to Python builtins,
// std containers/optional/pair/tuple/variant/function to their typing
// equivalents, and strips cv/ref/ptr qualifiers. Anything unknown (most
// notably bound class types) is returned as the canonical qualified C++
// name; a post-pass over the IR resolves those against bound classes.
std::string translate_python_type(clang::QualType type, clang::ASTContext const &ctx);

// Best-effort translation of a captured default-argument expression to
// its Python-literal form. Strips integer/float suffixes, rewrites
// `nullptr` / `std::nullopt` to `None`, `true`/`false` to `True`/`False`,
// and falls back to the verbatim text when no rewrite applies.
std::string translate_python_default(std::string const &cpp_default);

// String-based variant of `translate_python_type` for callers that
// already have a printed C++ type name (no `clang::QualType` access).
// Same recursion rules: maps fundamentals, `std::vector<T>` →
// `list[T_py]`, `std::pair<A,B>` → `tuple[A_py, B_py]`, etc.
std::string translate_python_type_string(std::string const &cpp_type);

} // namespace einsums::pybind
