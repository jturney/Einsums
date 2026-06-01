//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Visitor.hpp"

#include "AnnotationParser.hpp"
#include "DocExtractor.hpp"
#include "InstantiateParser.hpp"
#include "TypeTranslator.hpp"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

namespace einsums::pybind {

namespace {

// Convert a clang SourceLocation into the plain-string form the IR uses.
SourceLocation to_ir_location(clang::SourceLocation loc, clang::SourceManager const &sm) {
    SourceLocation out;
    out.file   = sm.getFilename(loc).str();
    out.line   = sm.getSpellingLineNumber(loc);
    out.column = sm.getSpellingColumnNumber(loc);
    return out;
}

DirectiveList parse_attached_directives(clang::Decl const *decl) {
    DirectiveList out;
    for (auto const *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
        if (auto d = parse_annotation(attr->getAnnotation())) {
            out.push_back(std::move(*d));
        }
    }
    return out;
}

// Build the BoundParam list for a function/method declaration. Default
// argument expressions are captured as their pretty-printed source form.
std::vector<BoundParam> build_params(clang::FunctionDecl const *fn, clang::ASTContext const &ctx) {
    std::vector<BoundParam> params;
    params.reserve(fn->getNumParams());
    for (clang::ParmVarDecl const *p : fn->parameters()) {
        BoundParam bp;
        bp.name           = p->getNameAsString();
        bp.type           = translate_type(p->getType(), ctx);
        bp.type_canonical = translate_type(p->getType().getCanonicalType(), ctx);
        bp.py_type        = translate_python_type(p->getType(), ctx);
        if (p->hasDefaultArg() && !p->hasUninstantiatedDefaultArg()) {
            if (clang::Expr const *def = p->getDefaultArg()) {
                std::string              printed;
                llvm::raw_string_ostream os(printed);
                clang::PrintingPolicy    policy(ctx.getLangOpts());
                policy.SuppressTagKeyword     = true;
                policy.FullyQualifiedName     = true;
                policy.SuppressUnwrittenScope = false;
                def->printPretty(os, nullptr, policy);
                bp.default_value    = printed;
                bp.default_value_py = translate_python_default(printed);
            }
        }
        params.push_back(std::move(bp));
    }
    return params;
}

// Substitute `{name}` placeholders in `template_str` with values from
// `bindings` (template-param-name -> concrete-arg). Unknown placeholders
// are left untouched so the user notices them in the generated Python
// identifier and can fix the template.
std::string substitute_name_template(std::string const &template_str, std::vector<std::string> const &param_names,
                                     std::vector<std::string> const &combo_values) {
    std::string out = template_str;
    for (std::size_t i = 0; i < param_names.size() && i < combo_values.size(); ++i) {
        std::string const  placeholder = "{" + param_names[i] + "}";
        std::string const &value       = combo_values[i];
        // Sanitize the value so the resulting identifier is valid Python.
        std::string clean;
        bool        last_underscore = false;
        for (char const ch : value) {
            bool const is_ident = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
            if (is_ident) {
                clean += ch;
                last_underscore = false;
            } else if (!last_underscore && !clean.empty()) {
                clean += '_';
                last_underscore = true;
            }
        }
        while (!clean.empty() && clean.back() == '_') {
            clean.pop_back();
        }
        std::size_t pos = 0;
        while ((pos = out.find(placeholder, pos)) != std::string::npos) {
            out.replace(pos, placeholder.size(), clean);
            pos += clean.size();
        }
    }
    return out;
}

// Split a per-instantiation comma-joined string back into individual
// values, respecting <> nesting. e.g. "float, 2" -> ["float", "2"];
// "std::complex<float>, 3" -> ["std::complex<float>", "3"].
std::vector<std::string> split_combo(std::string const &combo) {
    std::vector<std::string> parts;
    std::string              current;
    int                      angle = 0;
    for (char const c : combo) {
        if (c == '<') {
            ++angle;
        } else if (c == '>') {
            if (angle > 0) {
                --angle;
            }
        }
        if (c == ',' && angle == 0) {
            // trim leading/trailing space
            std::size_t b = 0;
            std::size_t e = current.size();
            while (b < e && current[b] == ' ') {
                ++b;
            }
            while (e > b && current[e - 1] == ' ') {
                --e;
            }
            parts.push_back(current.substr(b, e - b));
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        std::size_t b = 0;
        std::size_t e = current.size();
        while (b < e && current[b] == ' ') {
            ++b;
        }
        while (e > b && current[e - 1] == ' ') {
            --e;
        }
        parts.push_back(current.substr(b, e - b));
    }
    return parts;
}

// Parse a stringified macro-argument list of double-quoted identifiers,
// e.g. ``"trans_a", "trans_b"`` -> ``{"trans_a", "trans_b"}``. The
// payload comes from ``#__VA_ARGS__`` so each name is wrapped in
// embedded double quotes that need to be stripped. Whitespace around
// commas is tolerated; nothing else is.
std::vector<std::string> parse_quoted_string_list(std::string const &payload) {
    std::vector<std::string> out;
    std::string              current;
    bool                     in_quotes = false;
    for (char const c : payload) {
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (in_quotes) {
            current += c;
        } else if (c == ',') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

// Match the @instantiate(_template) groups against the class's actual
// template-parameter names and reorder them positionally. Returns the
// per-position value lists ready for cross_product. On any mismatch
// (unknown keyword, duplicate, missing parameter, wrong arity, mangled
// class name) emits a clear diagnostic and returns nullopt.
//
// This is the guard against random macro expansion: if some upstream
// header has ``#define Element ...`` and the user wrote
// ``Element(float, double)``, the preprocessor mangles it before our
// macro stringifies, yielding a payload whose group keyword is no
// longer ``Element``. We reject anything we don't recognize.
std::optional<std::vector<std::vector<std::string>>>
match_groups_to_template_params(clang::Decl const *decl, clang::ASTContext const &ctx, std::string const &directive_name,
                                std::string const &class_name_in_payload, std::string const &actual_class_name,
                                std::vector<ParamGroup> const &groups, std::vector<std::string> const &template_param_names) {
    auto report_error = [&](std::string const &msg) {
        clang::SourceManager const &sm  = ctx.getSourceManager();
        clang::SourceLocation const loc = decl->getLocation();
        llvm::errs() << sm.getFilename(loc) << ":" << sm.getSpellingLineNumber(loc) << ":" << sm.getSpellingColumnNumber(loc)
                     << ": error: einsums-pybind: @" << directive_name << " on " << actual_class_name << ": " << msg << "\n";
    };

    bool ok = true;
    if (!class_name_in_payload.empty() && class_name_in_payload != actual_class_name) {
        report_error("class name '" + class_name_in_payload + "' in directive payload does not match the annotated class '" +
                     actual_class_name + "' (possible macro expansion?)");
        ok = false;
    }
    if (groups.size() != template_param_names.size()) {
        report_error("expected " + std::to_string(template_param_names.size()) + " parameter list(s) (one per template parameter, got " +
                     std::to_string(groups.size()) + ")");
        ok = false;
    }

    std::vector<std::vector<std::string>> ordered(template_param_names.size());
    std::vector<bool>                     seen(template_param_names.size(), false);
    for (auto const &g : groups) {
        auto const it = std::find(template_param_names.begin(), template_param_names.end(), g.keyword);
        if (it == template_param_names.end()) {
            std::string known;
            for (std::size_t i = 0; i < template_param_names.size(); ++i) {
                if (i != 0) {
                    known += ", ";
                }
                known += template_param_names[i];
            }
            report_error("unknown parameter keyword '" + g.keyword + "' (template parameters are: " + known +
                         ") — possible macro expansion or typo");
            ok = false;
            continue;
        }
        auto const idx = static_cast<std::size_t>(std::distance(template_param_names.begin(), it));
        if (seen[idx]) {
            report_error("parameter keyword '" + g.keyword + "' specified more than once");
            ok = false;
            continue;
        }
        seen[idx]    = true;
        ordered[idx] = g.values;
    }
    if (!ok) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (!seen[i]) {
            report_error("missing parameter list for template parameter '" + template_param_names[i] + "'");
            return std::nullopt;
        }
    }
    return ordered;
}

// Resolve every @instantiate / @instantiate_as / @instantiate_template
// directive on a class into the cross-product of concrete instantiations
// the emitter will use.
std::vector<BoundInstantiation> collect_instantiations(clang::NamedDecl const *decl, clang::ASTContext const &ctx,
                                                       DirectiveList const            &directives,
                                                       std::vector<std::string> const &template_param_names, int &error_count) {
    std::vector<BoundInstantiation> out;
    std::string const               base = decl->getNameAsString();
    for (auto const &d : directives) {
        if (d.name == "instantiate" && !d.args.empty()) {
            InstantiateSpec const spec = parse_instantiate(d.args.front());
            auto const            ordered =
                match_groups_to_template_params(decl, ctx, d.name, spec.class_name, base, spec.groups, template_param_names);
            if (!ordered) {
                ++error_count;
                continue;
            }
            for (auto const &combo : cross_product(*ordered)) {
                BoundInstantiation inst;
                inst.type_args = combo;
                inst.py_name   = sanitize_python_name(base, combo);
                out.push_back(std::move(inst));
            }
        } else if (d.name == "instantiate_as" && d.args.size() == 2) {
            InstantiateAsSpec const spec = parse_instantiate_as(d.args[0], d.args[1]);
            BoundInstantiation      inst;
            inst.py_name   = spec.py_name;
            inst.type_args = spec.type_args;
            out.push_back(std::move(inst));
        } else if (d.name == "instantiate_template" && d.args.size() == 2) {
            std::string const     name_template = d.args[0];
            InstantiateSpec const spec          = parse_instantiate(d.args[1]);
            auto const            ordered =
                match_groups_to_template_params(decl, ctx, d.name, spec.class_name, base, spec.groups, template_param_names);
            if (!ordered) {
                ++error_count;
                continue;
            }
            for (auto const &combo : cross_product(*ordered)) {
                BoundInstantiation inst;
                inst.type_args                       = combo;
                std::vector<std::string> const split = split_combo(combo);
                inst.py_name                         = substitute_name_template(name_template, template_param_names, split);
                out.push_back(std::move(inst));
            }
        }
    }
    return out;
}

// Names of each template parameter on a templated class — e.g.
// ``template <typename T, size_t rank, typename Alloc>`` returns
// ``["T", "rank", "Alloc"]``. Used by INSTANTIATE_TEMPLATE to substitute
// `{T}`/`{rank}` placeholders against concrete instantiation values.
std::vector<std::string> collect_template_param_names(clang::CXXRecordDecl const *decl) {
    std::vector<std::string>        out;
    clang::ClassTemplateDecl const *tmpl = decl->getDescribedClassTemplate();
    if (tmpl == nullptr) {
        return out;
    }
    clang::TemplateParameterList const *params = tmpl->getTemplateParameters();
    if (params == nullptr) {
        return out;
    }
    out.reserve(params->size());
    for (clang::NamedDecl const *p : *params) {
        out.push_back(p->getNameAsString());
    }
    return out;
}

// Qualified base-class names in declaration order (public bases only —
// pybind11 multiple-inheritance binding works on public bases).
std::vector<std::string> collect_bases(clang::CXXRecordDecl const *decl, clang::ASTContext const &ctx) {
    std::vector<std::string> bases;
    if (!decl->hasDefinition()) {
        return bases;
    }
    for (clang::CXXBaseSpecifier const &base : decl->bases()) {
        if (base.getAccessSpecifier() != clang::AS_public) {
            continue;
        }
        bases.push_back(translate_type(base.getType(), ctx));
    }
    return bases;
}

} // namespace

bool Visitor::has_any_pybind_annotation(clang::Decl const *decl) const {
    for (auto const *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
        if (attr->getAnnotation().starts_with(k_annotation_prefix)) {
            return true;
        }
    }
    return false;
}

bool Visitor::passes_docs_filter(clang::NamedDecl const *decl) const {
    if (!decl_in_module_headers(decl)) {
        return false;
    }
    // Skip internal/implementation and anonymous namespaces.
    std::string const qn = decl->getQualifiedNameAsString();
    if (qn.find("detail::") != std::string::npos || qn.find("impl::") != std::string::npos ||
        qn.find("(anonymous namespace)") != std::string::npos) {
        return false;
    }
    // Document only entities that carry a doc comment, and never ones marked
    // @internal. (The filter can be relaxed to EXTRACT_ALL-style later.)
    std::string const doc = extract_doc(decl, _context);
    if (doc.empty()) {
        return false;
    }
    if (doc.find("@internal") != std::string::npos || doc.find("\\internal") != std::string::npos) {
        return false;
    }
    return true;
}

bool Visitor::passes_member_filter(clang::NamedDecl const *decl) const {
    if (decl->getAccess() != clang::AS_public) {
        return false;
    }
    if (!decl_in_module_headers(decl)) {
        return false;
    }
    std::string const doc = extract_doc(decl, _context);
    return doc.find("@internal") == std::string::npos && doc.find("\\internal") == std::string::npos;
}

bool Visitor::decl_in_module_headers(clang::Decl const *decl) const {
    if (_module_headers.empty()) {
        return true; // no filter → bind everything (legacy behaviour)
    }
    clang::SourceManager const &sm  = _context.getSourceManager();
    clang::SourceLocation const loc = sm.getFileLoc(decl->getLocation());
    if (loc.isInvalid()) {
        return false;
    }
    llvm::StringRef const filename = sm.getFilename(loc);
    if (filename.empty()) {
        return false;
    }
    for (auto const &header : _module_headers) {
        // Match relative-suffix: the codegen passes paths like
        // "Einsums/Tensor/RuntimeTensor.hpp" via --source-include; the
        // decl's filename is absolute. Compare via ends_with so we
        // match without depending on the absolute-path layout.
        if (filename.ends_with(header)) {
            return true;
        }
    }
    return false;
}

void Visitor::fill_common(BoundEntityCommon &entity, clang::NamedDecl const *decl) {
    entity.name           = decl->getNameAsString();
    entity.qualified_name = decl->getQualifiedNameAsString();
    entity.directives     = parse_attached_directives(decl);
    entity.doc            = extract_doc(decl, _context);
    entity.location       = to_ir_location(decl->getLocation(), _context.getSourceManager());

    // Inherit the enclosing-namespace submodule path when the entity
    // doesn't carry its own ``module`` directive. Per-entity directives
    // win on override; an explicit empty string is not supported (the
    // user can't currently opt back out of an inherited submodule, which
    // is the simplest predictable rule).
    if (!_module_stack.empty()) {
        bool has_own_module = false;
        for (auto const &d : entity.directives) {
            if (d.name == "module") {
                has_own_module = true;
                break;
            }
        }
        if (!has_own_module) {
            Directive inherited;
            inherited.name = "module";
            inherited.args.push_back(_module_stack.back());
            entity.directives.push_back(std::move(inherited));
        }
    }

    // Materialize the resolved submodule path (own or inherited) as a
    // typed field so downstream emitters don't need to scan directives.
    for (auto const &d : entity.directives) {
        if (d.name == "module" && !d.args.empty()) {
            entity.submodule = d.args.front();
            break;
        }
    }
}

bool Visitor::TraverseNamespaceDecl(clang::NamespaceDecl *decl) {
    // Look for an EINSUMS_PYBIND_MODULE directive on this namespace
    // declaration. Anonymous namespaces and re-opened blocks without the
    // directive don't push anything; the previous innermost value remains
    // in effect for nested entities.
    bool                pushed = false;
    DirectiveList const dirs   = parse_attached_directives(decl);
    for (auto const &d : dirs) {
        if (d.name == "module" && !d.args.empty()) {
            _module_stack.push_back(d.args.front());
            pushed = true;
            break;
        }
    }

    bool const result = clang::RecursiveASTVisitor<Visitor>::TraverseNamespaceDecl(decl);

    if (pushed) {
        _module_stack.pop_back();
    }
    return result;
}

bool Visitor::TraverseCXXRecordDecl(clang::CXXRecordDecl *decl) {
    // Skip implicit/injected duplicates; they re-trigger every annotation
    // on the canonical decl and would double-count.
    if (decl->isImplicit()) {
        return true;
    }
    // Anonymous structs and forward declarations don't carry binding info.
    if (!decl->hasDefinition() || decl->getDeclName().isEmpty()) {
        return clang::RecursiveASTVisitor<Visitor>::TraverseCXXRecordDecl(decl);
    }
    bool const wanted = _docs_mode ? passes_docs_filter(decl) : has_any_pybind_annotation(decl);
    if (!wanted) {
        return clang::RecursiveASTVisitor<Visitor>::TraverseCXXRecordDecl(decl);
    }

    // Out-of-module annotated classes: capture a minimal record purely
    // for cross-module name resolution in the .pyi emitter (so a runtime
    // signature mentioning ``GeneralRuntimeTensor<float, ...>`` from
    // another module's headers can resolve to ``RuntimeTensorF``). No
    // methods, fields, or properties are walked; the C++ emitter skips
    // these via ``is_external``.
    if (!decl_in_module_headers(decl)) {
        BoundClass external;
        fill_common(external, decl);
        external.is_external          = true;
        external.is_template          = decl->getDescribedClassTemplate() != nullptr;
        external.template_param_names = collect_template_param_names(decl);
        external.instantiations = collect_instantiations(decl, _context, external.directives, external.template_param_names, _error_count);
        _module.classes.push_back(std::move(external));
        return clang::RecursiveASTVisitor<Visitor>::TraverseCXXRecordDecl(decl);
    }

    BoundClass cls;
    fill_common(cls, decl);
    cls.is_template          = decl->getDescribedClassTemplate() != nullptr;
    cls.template_param_names = collect_template_param_names(decl);
    cls.bases                = collect_bases(decl, _context);
    cls.instantiations       = collect_instantiations(decl, _context, cls.directives, cls.template_param_names, _error_count);

    BoundClass *parent = current_class();
    // target is mutated later through the class stack as methods/fields
    // attach to it, so it must remain a pointer to non-const.
    // NOLINTNEXTLINE(misc-const-correctness)
    BoundClass *target = nullptr;
    if (parent != nullptr) {
        parent->nested_classes.push_back(std::move(cls));
        target = &parent->nested_classes.back();
    } else {
        _module.classes.push_back(std::move(cls));
        target = &_module.classes.back();
    }

    _class_stack.push_back(target);
    bool const result = clang::RecursiveASTVisitor<Visitor>::TraverseCXXRecordDecl(decl);
    _class_stack.pop_back();
    return result;
}

bool Visitor::TraverseClassTemplateDecl(clang::ClassTemplateDecl *decl) {
    // Templates reach us through their templated-decl shell. Delegate to
    // TraverseCXXRecordDecl on the underlying record so the same scope
    // tracking applies; Phase 4 will branch here for cross-product
    // expansion.
    if (clang::CXXRecordDecl *templated = decl->getTemplatedDecl()) {
        return TraverseCXXRecordDecl(templated);
    }
    return clang::RecursiveASTVisitor<Visitor>::TraverseClassTemplateDecl(decl);
}

bool Visitor::VisitCXXMethodDecl(clang::CXXMethodDecl *decl) {
    if (decl->isImplicit()) {
        return true;
    }
    BoundClass *cls = current_class();
    if (cls == nullptr) {
        return true; // method outside an exposed class — nothing to attach to
    }
    if (_docs_mode ? !passes_member_filter(decl) : (!has_any_pybind_annotation(decl) || !decl_in_module_headers(decl))) {
        return true;
    }

    BoundMethod method;
    fill_common(method, decl);
    method.return_type           = translate_type(decl->getReturnType(), _context);
    method.return_type_canonical = translate_type(decl->getReturnType().getCanonicalType(), _context);
    method.return_py_type        = translate_python_type(decl->getReturnType(), _context);
    method.params                = build_params(decl, _context);
    method.is_const              = decl->isConst();
    method.is_static             = decl->isStatic();
    method.is_virtual            = decl->isVirtual();
    method.is_pure_virtual       = decl->isPureVirtual();
    method.is_constructor        = clang::isa<clang::CXXConstructorDecl>(decl);
    method.is_destructor         = clang::isa<clang::CXXDestructorDecl>(decl);
    method.is_operator           = decl->isOverloadedOperator();
    method.is_deleted            = decl->isDeleted();
    // Member function templates: capture their template parameters so the
    // renderer can emit a ``template <...>`` clause (and so docs collects the
    // param names for nitpick suppression — they are never xref targets).
    if (auto const *ftpl = decl->getDescribedFunctionTemplate()) {
        method.is_template = true;
        if (auto const *plist = ftpl->getTemplateParameters()) {
            method.template_param_names.reserve(plist->size());
            for (clang::NamedDecl const *p : *plist) {
                method.template_param_names.push_back(p->getNameAsString());
            }
        }
    }

    // Honor EINSUMS_PYBIND_VARIADIC_FROM: record the named template
    // parameter and the per-element type so the emitter can expand the
    // pack at per-instantiation time. The directive only makes sense
    // inside a templated class.
    for (auto const &d : method.directives) {
        if (d.name == "variadic_from" && d.args.size() == 2) {
            method.has_variadic_pack     = true;
            method.variadic_from_param   = d.args[0];
            method.variadic_element_type = d.args[1];
        }
    }

    if (method.is_constructor) {
        method.return_type.clear();
        cls->ctors.push_back(std::move(method));
    } else {
        cls->methods.push_back(std::move(method));
    }
    return true;
}

bool Visitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
    // Skip methods (handled by VisitCXXMethodDecl) and template
    // instantiations (the primary template was already processed).
    if (clang::isa<clang::CXXMethodDecl>(decl)) {
        return true;
    }
    if (decl->isImplicit() || decl->isTemplateInstantiation()) {
        return true;
    }
    if (_docs_mode ? !passes_docs_filter(decl) : (!has_any_pybind_annotation(decl) || !decl_in_module_headers(decl))) {
        return true;
    }

    BoundFunction fn;
    fill_common(fn, decl);
    fn.return_type           = translate_type(decl->getReturnType(), _context);
    fn.return_type_canonical = translate_type(decl->getReturnType().getCanonicalType(), _context);
    fn.return_py_type        = translate_python_type(decl->getReturnType(), _context);
    fn.params                = build_params(decl, _context);
    fn.is_template           = decl->getDescribedFunctionTemplate() != nullptr;
    // Resolve EINSUMS_PYBIND_INSTANTIATE_AS directives on a templated free
    // function into per-instantiation BoundInstantiation entries. Each
    // directive specifies an explicit Python name and the C++ type
    // arguments; the emitter writes one m.def(py_name, &qualified<args>)
    // call per entry. Multiple instantiations sharing a Python name
    // produce a pybind11 overload set on the Python side.
    if (fn.is_template) {
        // Capture the function template's parameter names (e.g.
        // ``["AType", "BType", "CType"]``). The emitter splits each
        // instantiation's type_args by comma and pairs them with these
        // names to substitute concrete types into the parameter list
        // when emitting a static_cast<> to disambiguate overloads.
        if (auto const *ftpl = decl->getDescribedFunctionTemplate()) {
            if (auto const *plist = ftpl->getTemplateParameters()) {
                fn.template_param_names.reserve(plist->size());
                for (clang::NamedDecl const *p : *plist) {
                    fn.template_param_names.push_back(p->getNameAsString());
                }
            }
        }
        // Collect EINSUMS_PYBIND_TEMPLATE_KWARGS first so it's available
        // when expanding INSTANTIATE_BOOLS below. The directive carries a
        // single free-form arg holding the stringified macro arguments,
        // e.g. ``"trans_a", "trans_b"`` — quoted, comma-separated.
        for (auto const &d : fn.directives) {
            if (d.name == "template_kwargs" && !d.args.empty()) {
                fn.template_kwargs = parse_quoted_string_list(d.args.front());
                break;
            }
        }
        for (auto const &d : fn.directives) {
            if (d.name == "instantiate_as" && d.args.size() == 2) {
                // For free functions the directive payload is just the
                // comma-separated template-argument list (no enclosing
                // class wrapper), so we don't pass through
                // parse_instantiate_as (which assumes ``Class<args>``
                // shape and strips the angle brackets).
                BoundInstantiation inst;
                inst.py_name   = d.args[0];
                inst.type_args = d.args[1];
                fn.instantiations.push_back(std::move(inst));
            } else if (d.name == "instantiate_bools" && d.args.size() == 2) {
                // Expand 2^N false/true combinations and prepend each to
                // the non-bool tail. Lexicographic order: first kwarg is
                // the high bit (matches the leading template parameter).
                std::size_t const n = fn.template_kwargs.size();
                if (n == 0) {
                    clang::SourceManager const &sm  = _context.getSourceManager();
                    clang::SourceLocation const loc = decl->getLocation();
                    llvm::errs() << sm.getFilename(loc) << ":" << sm.getSpellingLineNumber(loc) << ":" << sm.getSpellingColumnNumber(loc)
                                 << ": error: einsums-pybind: @instantiate_bools on " << fn.qualified_name
                                 << ": requires EINSUMS_PYBIND_TEMPLATE_KWARGS to declare bool kwarg names\n";
                    ++_error_count;
                    continue;
                }
                std::string const &tail   = d.args[1];
                std::size_t const  combos = std::size_t{1} << n;
                for (std::size_t mask = 0; mask < combos; ++mask) {
                    std::string prefix;
                    for (std::size_t k = 0; k < n; ++k) {
                        if (k != 0) {
                            prefix += ", ";
                        }
                        bool const v = ((mask >> (n - 1 - k)) & 1U) != 0U;
                        prefix += v ? "true" : "false";
                    }
                    BoundInstantiation inst;
                    inst.py_name = d.args[0];
                    if (tail.empty()) {
                        inst.type_args = std::move(prefix);
                    } else {
                        inst.type_args = std::move(prefix);
                        inst.type_args += ", ";
                        inst.type_args += tail;
                    }
                    fn.instantiations.push_back(std::move(inst));
                }
            }
        }
    }
    _module.functions.push_back(std::move(fn));
    return true;
}

bool Visitor::VisitFieldDecl(clang::FieldDecl *decl) {
    BoundClass *cls = current_class();
    if (cls == nullptr) {
        return true;
    }
    if (decl->isImplicit()) {
        return true;
    }
    if (_docs_mode ? !passes_member_filter(decl) : (!has_any_pybind_annotation(decl) || !decl_in_module_headers(decl))) {
        return true;
    }

    BoundField field;
    fill_common(field, decl);
    field.type      = translate_type(decl->getType(), _context);
    field.py_type   = translate_python_type(decl->getType(), _context);
    field.is_static = false; // FieldDecl is non-static by definition
    cls->fields.push_back(std::move(field));
    return true;
}

bool Visitor::VisitEnumDecl(clang::EnumDecl *decl) {
    if (decl->isImplicit() || !decl->isComplete()) {
        return true;
    }
    bool const member = current_class() != nullptr;
    if (_docs_mode ? !(member ? passes_member_filter(decl) : passes_docs_filter(decl))
                   : (!has_any_pybind_annotation(decl) || !decl_in_module_headers(decl))) {
        return true;
    }

    BoundEnum e;
    fill_common(e, decl);
    e.is_scoped          = decl->isScoped();
    e.underlying_type    = translate_type(decl->getIntegerType(), _context);
    e.underlying_py_type = translate_python_type(decl->getIntegerType(), _context);
    for (clang::EnumConstantDecl const *ec : decl->enumerators()) {
        BoundEnumerator entry;
        entry.name  = ec->getNameAsString();
        entry.value = ec->getInitVal().getExtValue();
        entry.doc   = extract_doc(ec, _context);
        e.enumerators.push_back(std::move(entry));
    }

    if (BoundClass *cls = current_class()) {
        cls->nested_enums.push_back(std::move(e));
    } else {
        _module.enums.push_back(std::move(e));
    }
    return true;
}

bool Visitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl) {
    // Typedefs/using-aliases are captured only in docs mode, primarily so a
    // ``cpp:type`` declaration exists for them — otherwise references to the
    // alias in function signatures (e.g. ``int_t``) dangle. We don't require
    // a doc comment (an undocumented public alias is still a real ref target),
    // but we keep the namespace/internal hygiene of the docs filter.
    if (!_docs_mode || decl->isImplicit()) {
        return true;
    }
    if (!decl_in_module_headers(decl)) {
        return true;
    }
    std::string const qn = decl->getQualifiedNameAsString();
    if (qn.find("detail::") != std::string::npos || qn.find("impl::") != std::string::npos ||
        qn.find("(anonymous namespace)") != std::string::npos) {
        return true;
    }
    // Skip member aliases of a template specialization (qualified name like
    // ``Trait<type-parameter-0-0>::type``) — partial specializations bypass
    // the class-stack scope tracking, and their template-instantiation scope
    // confuses the cpp domain.
    if (qn.find('<') != std::string::npos) {
        return true;
    }
    // Only document namespace-scope aliases. Member type aliases
    // (``struct Trait { using type = ...; }``) are implementation detail and
    // their template-instantiation scope (``Trait<type-parameter-0-0>``)
    // confuses the cpp domain.
    if (current_class() != nullptr) {
        return true;
    }
    std::string const doc = extract_doc(decl, _context);
    if (doc.find("@internal") != std::string::npos || doc.find("\\internal") != std::string::npos) {
        return true;
    }

    BoundTypedef td;
    fill_common(td, decl);
    td.underlying_type = translate_type(decl->getUnderlyingType(), _context);
    // Alias templates (``template <...> using X = ...``): capture the
    // template parameters so the renderer emits a correct ``template <...>``
    // prefix on the cpp:type directive.
    if (auto const *at = clang::dyn_cast<clang::TypeAliasDecl>(decl)) {
        if (auto const *tat = at->getDescribedAliasTemplate()) {
            td.is_template = true;
            if (auto const *plist = tat->getTemplateParameters()) {
                td.template_param_names.reserve(plist->size());
                for (clang::NamedDecl const *p : *plist) {
                    td.template_param_names.push_back(p->getNameAsString());
                }
            }
        }
    }
    _module.typedefs.push_back(std::move(td));
    return true;
}

bool Visitor::VisitConceptDecl(clang::ConceptDecl *decl) {
    if (!_docs_mode || decl->isImplicit() || !decl_in_module_headers(decl)) {
        return true;
    }
    std::string const qn = decl->getQualifiedNameAsString();
    if (qn.find("detail::") != std::string::npos || qn.find("impl::") != std::string::npos) {
        return true;
    }
    std::string const doc = extract_doc(decl, _context);
    if (doc.find("@internal") != std::string::npos || doc.find("\\internal") != std::string::npos) {
        return true;
    }

    BoundConcept c;
    fill_common(c, decl);
    if (auto const *plist = decl->getTemplateParameters()) {
        c.template_param_names.reserve(plist->size());
        for (clang::NamedDecl const *p : *plist) {
            c.template_param_names.push_back(p->getNameAsString());
        }
    }
    _module.concepts.push_back(std::move(c));
    return true;
}

} // namespace einsums::pybind
