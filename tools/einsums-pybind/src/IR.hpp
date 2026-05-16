//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// IR — internal representation of declarations the codegen tool will bind.
//
// The Visitor (Visitor.hpp) populates this IR from the Clang AST; the
// Emitter (Phase 3) consumes it and produces pybind11 C++. Keeping the IR
// independent of clang::* types means the emitter doesn't need to drag in
// libtooling headers, and unit tests against the IR can construct fixtures
// directly without spinning up a ClangTool.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace einsums::pybind {

struct SourceLocation {
    std::string file;
    unsigned    line   = 0;
    unsigned    column = 0;
};

// One parsed annotation directive. The AnnotationParser splits the raw
// "einsums_pybind:<name>[:<arg>[:<arg>...]]" payload into this form so the
// emitter can switch on `name` cleanly.
struct Directive {
    std::string              name;
    std::vector<std::string> args;
};

using DirectiveList = std::vector<Directive>;

// Common metadata every bound declaration carries. Inherited by the
// concrete bound-* structs below so the emitter can write generic code
// against this base when convenient.
struct BoundEntityCommon {
    std::string    name;           // unqualified
    std::string    qualified_name; // ::ns::Class::method
    DirectiveList  directives;
    std::string    doc; // raw doxygen text or empty
    SourceLocation location;
    /// Resolved Python submodule path (from a per-entity or inherited
    /// EINSUMS_PYBIND_MODULE directive on an enclosing namespace). Empty
    /// when the entity belongs to the top-level module. The .pyi
    /// emitter uses this to decide which `<module>.pyi` file an
    /// entity belongs to (e.g. einsums.linalg, einsums.graph).
    std::optional<std::string> submodule;
};

// A single function parameter. Default-value text is captured verbatim
// from the AST so the emitter can re-emit it as a `py::arg("x") = ...`.
//
// `py_type` and `default_value_py` carry the Python-stub forms — a
// best-effort translation populated alongside the C++ form. The .pyi
// emitter consumes these; the C++ emitter ignores them.
struct BoundParam {
    std::string name;
    std::string type;
    /// Canonical (typedef-expanded) C++ type, populated alongside
    /// ``type``. Same purpose as ``BoundFunction::return_type_canonical``
    /// — lets the .pyi emitter substitute and resolve through typedef
    /// aliases like ``RuntimeTensor<T>`` ↔ ``GeneralRuntimeTensor<T,
    /// std::allocator<T>>``.
    std::string                type_canonical;
    std::string                py_type;
    std::optional<std::string> default_value;
    std::optional<std::string> default_value_py;
};

struct BoundField : BoundEntityCommon {
    std::string type;
    std::string py_type;
    bool        is_static = false;
};

struct BoundMethod : BoundEntityCommon {
    std::string return_type;
    /// Canonical (typedef-expanded) form of the return type. Same role
    /// as ``BoundFunction::return_type_canonical``: the .pyi emitter
    /// substitutes member-template bindings on this and re-resolves so
    /// per-directive overloads emit concrete return types.
    std::string             return_type_canonical;
    std::string             return_py_type;
    std::vector<BoundParam> params;
    bool                    is_const        = false;
    bool                    is_static       = false;
    bool                    is_virtual      = false;
    bool                    is_pure_virtual = false;
    bool                    is_constructor  = false;
    bool                    is_destructor   = false;
    bool                    is_operator     = false;
    bool                    is_deleted      = false;

    // Set by EINSUMS_PYBIND_VARIADIC_FROM: the last parameter is a pack
    // expansion whose arity comes from the named template parameter, and
    // each expanded slot has type ``variadic_element_type``. When emitting
    // a per-instantiation binding, the pack slot is replaced with N copies
    // of ``(variadic_element_type, dim_<i>)``.
    bool        has_variadic_pack = false;
    std::string variadic_from_param;
    std::string variadic_element_type;
};

struct BoundEnumerator {
    std::string  name;
    std::int64_t value = 0;
    std::string  doc;
};

struct BoundEnum : BoundEntityCommon {
    bool                         is_scoped = false;
    std::string                  underlying_type;
    std::string                  underlying_py_type;
    std::vector<BoundEnumerator> enumerators;
};

// One concrete template instantiation requested by an @instantiate or
// @instantiate_as directive. The emitter produces both an explicit
// ``template class Foo<args>;`` declaration and a per-instantiation
// ``py::class_<Foo<args>>(m, py_name)`` binding block from this.
struct BoundInstantiation {
    std::string py_name;   // Python identifier (sanitized or user-supplied)
    std::string type_args; // ready-to-paste between < and >, e.g. "float, 2"
};

// One logical Python-facing entry on a free function or method. Computed
// by a post-IR pass (PythonOverloads.hpp) that groups raw
// BoundInstantiation entries according to the merge rules pybind11
// expects:
//
//   * NonTemplate          — function has no templates; one m.def, one stub
//   * SingleInstantiation  — one instantiation with this py_name; one m.def
//   * DtypeDispatcher      — N instantiations sharing arg signature, only
//                            return type varies; collapse into one m.def
//                            taking ``dtype="..."`` kwarg
//   * TemplateKwargsDispatcher
//                          — 2^N instantiations from EINSUMS_PYBIND_INSTANTIATE_BOOLS
//                            collapse into one m.def with N bool kwargs
//   * OverloadSet          — multiple instantiations with the same py_name
//                            that DON'T merge into a dispatcher; pybind11
//                            picks at runtime via overload resolution
//
// Both the C++ emitter and the .pyi emitter consume this view so the
// merge rules live in one place.
struct PythonOverload {
    enum class Kind : std::uint8_t {
        NonTemplate,
        SingleInstantiation,
        DtypeDispatcher,
        TemplateKwargsDispatcher,
        OverloadSet,
    };

    Kind        kind = Kind::NonTemplate;
    std::string py_name;

    // Indices into the owning entity's `instantiations` vector that make up
    // this logical entry. Empty for NonTemplate (which uses the function
    // itself directly).
    std::vector<std::size_t> instantiation_indices;

    // For DtypeDispatcher: the C++ scalar type each instantiation
    // contributes (parallel to instantiation_indices), plus the picked
    // default dtype string (e.g. "float64").
    std::vector<std::string> dtype_values;
    std::string              default_dtype;

    // For TemplateKwargsDispatcher: parallel to f.template_kwargs.
    std::vector<std::string> kwarg_names;
};

// A property aggregated from @getter / @setter directives on class
// methods. Computed by a post-IR pass (Properties.hpp) that walks
// BoundClass.methods. The .pyi emitter consumes this directly; the
// pybind11 C++ emitter still derives the same merge inline (kept
// separate to avoid churning the working emit code).
struct BoundProperty {
    std::string py_name; // Python attribute name from @getter("name")
    std::string type;    // Getter's C++ return type, ref/cv stripped
    std::string py_type; // Python form (translated from type)
    std::string doc;     // Doxygen text from getter
    bool        has_setter = false;
    // Indices into BoundClass.methods. setter_index is valid only when
    // has_setter is true.
    std::size_t getter_index = 0;
    std::size_t setter_index = 0;
};

struct BoundClass : BoundEntityCommon {
    bool is_template = false;
    /// True for annotated classes seen in headers OUTSIDE the current
    /// module's source filter. Captured purely for cross-module name
    /// resolution in the .pyi emitter — the C++ emitter ignores them
    /// (their bindings live in the owning module's TU).
    bool                            is_external = false;
    std::vector<std::string>        template_param_names; // e.g. ["T", "rank"] for ``template <typename T, size_t rank>``
    std::vector<std::string>        bases;
    std::vector<BoundMethod>        ctors;
    std::vector<BoundMethod>        methods;
    std::vector<BoundField>         fields;
    std::vector<BoundEnum>          nested_enums;
    std::vector<BoundClass>         nested_classes;
    std::vector<BoundInstantiation> instantiations;
    /// @getter/@setter pairs collapsed into properties. Computed by the
    /// post-IR pass in Properties.hpp; empty until the pass has run.
    std::vector<BoundProperty> properties;
};

struct BoundFunction : BoundEntityCommon {
    std::string return_type;
    /// Canonical (typedef-expanded) form of the return type. For
    /// per-instantiation .pyi emission we substitute and look up the
    /// canonical form so a function returning the typedef alias
    /// ``RuntimeTensor<T>`` resolves to the same Python class as the
    /// underlying ``GeneralRuntimeTensor<T, std::allocator<T>>``.
    std::string             return_type_canonical;
    std::string             return_py_type;
    std::vector<BoundParam> params;
    bool                    is_template = false;
    /// Names of the function template's template parameters (e.g.
    /// ``["AType", "BType", "CType"]`` for
    /// ``template <BasicTensorConcept AType, …>``). Used by the emitter
    /// to substitute concrete types from per-instantiation type_args
    /// into return/parameter types when emitting a static_cast<>
    /// to disambiguate overloads.
    std::vector<std::string> template_param_names;
    /// Python kwarg names for the leading bool template parameters,
    /// from ``EINSUMS_PYBIND_TEMPLATE_KWARGS``. Empty for functions
    /// without that directive. The emitter generates a runtime
    /// dispatcher with these as keyword-only arguments when non-empty.
    std::vector<std::string> template_kwargs;
    /// One per ``EINSUMS_PYBIND_INSTANTIATE_AS`` directive on a templated
    /// free function. Empty for non-templated functions and for templated
    /// functions without explicit instantiation directives (those skip
    /// emission with a TODO comment).
    std::vector<BoundInstantiation> instantiations;
    /// Logical Python-facing entries for this function, computed by a
    /// post-IR pass (see PythonOverloads.hpp). The C++ emitter dispatches
    /// on each entry's `kind` to choose dispatcher style; the .pyi
    /// emitter renders one `def` (with `@overload` decorators if needed)
    /// per entry. Empty when the pass hasn't run yet.
    std::vector<PythonOverload> python_overloads;
};

struct Module {
    std::vector<BoundClass>    classes;
    std::vector<BoundFunction> functions;
    std::vector<BoundEnum>     enums;
};

// Deterministic textual dump of the IR for golden-output testing and
// diagnostics. Phase 3's emitter produces the actual pybind11 C++; this
// function exists so Phase 2 has something testable.
std::string dump(Module const &module_);

} // namespace einsums::pybind
