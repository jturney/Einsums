//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "IR.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace einsums::pybind {

// Walks a translation unit and builds a Module IR populated only with
// declarations carrying at least one EINSUMS_PYBIND_* annotation. Other
// declarations are ignored entirely.
//
// Class scope is tracked via a stack so that methods, fields, and nested
// types attach to the right BoundClass. Templates are detected and the
// is_template flag is set, but full instantiation handling is deferred to
// Phase 4.
class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(clang::ASTContext &ctx) : _context(ctx) {}

    /// Set the list of source-include header paths (as passed via
    /// ``--source-include`` on the codegen command line). When non-empty,
    /// the visitor only binds declarations whose source location resolves
    /// to a file ending in one of these relative paths — transitive
    /// includes from other modules' headers are skipped to avoid
    /// duplicate bindings (the owning module's codegen run handles them).
    void set_module_header_filter(std::vector<std::string> const &headers) { _module_headers = headers; }

    Module take() && { return std::move(_module); }

    [[nodiscard]] int error_count() const { return _error_count; }

    // Override the *Traverse* hooks for class-like records so we can push
    // and pop the scope stack around the recursive descent into members.
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool TraverseCXXRecordDecl(clang::CXXRecordDecl *decl);
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool TraverseClassTemplateDecl(clang::ClassTemplateDecl *decl);
    // Namespace traversal pushes/pops the inherited submodule directive
    // stack so entities inside a ``namespace EINSUMS_PYBIND_MODULE("foo")
    // bar { ... }`` block inherit ``module:foo`` unless they declare their
    // own override.
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool TraverseNamespaceDecl(clang::NamespaceDecl *decl);

    // *Visit* hooks fire on the way down and produce IR records.
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool VisitCXXMethodDecl(clang::CXXMethodDecl *decl);
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool VisitFunctionDecl(clang::FunctionDecl *decl);
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool VisitFieldDecl(clang::FieldDecl *decl);
    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    bool VisitEnumDecl(clang::EnumDecl *decl);

  private:
    clang::ASTContext        &_context;
    Module                    _module;
    std::vector<BoundClass *> _class_stack; // current containing class chain (top = innermost)
    int                       _error_count = 0;

    [[nodiscard]] BoundClass *current_class() const { return _class_stack.empty() ? nullptr : _class_stack.back(); }

    // Returns false if the decl carries no einsums_pybind annotation, in
    // which case visitors should leave it alone.
    [[nodiscard]] bool has_any_pybind_annotation(clang::Decl const *decl) const;

    // Returns true if the decl's source location is in one of the module's
    // own headers (per ``--source-include`` flags). When the filter is
    // empty all decls pass — preserves behaviour for callers that don't
    // set a filter.
    [[nodiscard]] bool decl_in_module_headers(clang::Decl const *decl) const;

    std::vector<std::string> _module_headers;

    // Stack of inherited submodule paths from enclosing
    // EINSUMS_PYBIND_MODULE-annotated namespaces. The innermost value is at
    // the back; empty when no enclosing namespace carries a directive.
    // ``fill_common`` injects the back of this stack as a synthetic
    // ``module`` directive on entities that don't have one of their own.
    std::vector<std::string> _module_stack;

    // Common metadata population — qualified name, doc, location, parsed
    // directives. Used by every BoundEntityCommon-derived struct.
    void fill_common(BoundEntityCommon &entity, clang::NamedDecl const *decl);
};

} // namespace einsums::pybind
