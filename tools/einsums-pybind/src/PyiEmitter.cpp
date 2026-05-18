//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "PyiEmitter.hpp"

#include <cctype>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "PythonOverloads.hpp"
#include "TypeTranslator.hpp"

namespace einsums::pybind {

namespace {

// ── Directive helpers (kept tiny and standalone so PyiEmitter doesn't
// need to share DirectiveView with the C++ emitter) ─────────────────────

bool has_directive(DirectiveList const &dirs, std::string const &name) {
    for (auto const &d : dirs) {
        if (d.name == name) {
            return true;
        }
    }
    return false;
}

std::string first_directive_arg(DirectiveList const &dirs, std::string const &name) {
    for (auto const &d : dirs) {
        if (d.name == name && !d.args.empty()) {
            return d.args.front();
        }
    }
    return {};
}

bool is_hidden(BoundEntityCommon const &e) {
    return has_directive(e.directives, "hide");
}

// Split `text` on '\n' into individual lines (no trailing newline kept).
std::vector<std::string> split_lines(std::string const &text) {
    std::vector<std::string> out;
    std::size_t              start = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            out.push_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(text.substr(start));
    return out;
}

// Emit a Python triple-quoted docstring at the given indent level.
// Picks `"""` unless the doc itself contains that sequence, in which
// case it falls back to `'''`. Multi-line docs span multiple lines so
// the rendered text stays readable in editor hovers.
void emit_docstring(std::ostringstream &os, std::string const &doc, std::string const &indent) {
    if (doc.empty()) {
        return;
    }
    char const *q = (doc.find("\"\"\"") == std::string::npos) ? "\"\"\"" : "'''";

    auto lines = split_lines(doc);
    while (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }
    if (lines.empty()) {
        return;
    }
    if (lines.size() == 1) {
        os << indent << q << lines[0] << q << "\n";
        return;
    }
    os << indent << q << lines[0] << "\n";
    for (std::size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].empty()) {
            os << "\n";
        } else {
            os << indent << lines[i] << "\n";
        }
    }
    os << indent << q << "\n";
}

std::string py_name_for(BoundEntityCommon const &e) {
    auto const renamed = first_directive_arg(e.directives, "rename");
    return renamed.empty() ? e.name : renamed;
}

// ── Type resolution against bound classes/enums ─────────────────────────

// Build the set of qualified C++ names that have a Python binding (and
// the per-instantiation Python names for templated classes). Used to
// rewrite parameter/return Python-type strings: the TypeTranslator only
// gets us as far as the canonical C++ qualified name for unknown types;
// here we resolve those to the actual Python identifier so the stub
// references something pyright can find.
struct NameMap {
    // canonical C++ -> Python identifier
    std::unordered_map<std::string, std::string> cpp_to_py;
    // canonical C++ qualified names of known bound classes (for prefix
    // matching when the type carries template-arg sugar like
    // "::ns::Tensor<float, 2>").
    std::unordered_set<std::string> bound_class_keys;

    void add(std::string const &cpp, std::string const &py) {
        std::string key = cpp;
        // Strip leading "::" so users can mix qualified and unqualified.
        if (key.size() >= 2 && key[0] == ':' && key[1] == ':') {
            key.erase(0, 2);
        }
        if (cpp_to_py.find(key) == cpp_to_py.end()) {
            cpp_to_py[key] = py;
        }
    }
};

// NOLINTNEXTLINE(misc-no-recursion)
void collect_class_names(BoundClass const &cls, NameMap &out, std::string const &parent_py = {}) {
    if (is_hidden(cls)) {
        return;
    }
    std::string py_class;
    if (cls.is_template) {
        // Templated class: each instantiation contributes one Python
        // class. The canonical C++ form is "qualified_name<args>".
        for (auto const &inst : cls.instantiations) {
            std::string const key    = cls.qualified_name + "<" + inst.type_args + ">";
            std::string const dotted = parent_py.empty() ? inst.py_name : (parent_py + "." + inst.py_name);
            out.add(key, dotted);
            out.bound_class_keys.insert(cls.qualified_name);
        }
        // For nested-enum naming, no single parent name applies to a
        // templated class — we don't try to dot-prefix nested entities
        // under a templated parent (rare and ambiguous).
        py_class = parent_py;
    } else {
        py_class = parent_py.empty() ? py_name_for(cls) : (parent_py + "." + py_name_for(cls));
        out.add(cls.qualified_name, py_class);
    }
    // Nested enums are lifted to module-level pybind registrations by
    // Emitter.cpp (``py::enum_<Parent::Inner>(_sub_x, "Inner")``), so at
    // runtime they're reachable as ``submodule.Inner`` — NOT
    // ``Parent.Inner``. Register them in the name map without the parent
    // prefix so type annotations resolve correctly.
    for (auto const &e : cls.nested_enums) {
        if (is_hidden(e)) {
            continue;
        }
        out.add(e.qualified_name, py_name_for(e));
    }
    for (auto const &n : cls.nested_classes) {
        collect_class_names(n, out, py_class);
    }
}

void collect_enum_names(BoundEnum const &e, NameMap &out) {
    if (is_hidden(e)) {
        return;
    }
    out.add(e.qualified_name, py_name_for(e));
}

NameMap build_name_map(Module const &module_) {
    NameMap map;
    for (auto const &c : module_.classes) {
        collect_class_names(c, map);
    }
    for (auto const &e : module_.enums) {
        collect_enum_names(e, map);
    }
    return map;
}

// Map a raw C++ type token to its Python equivalent, or empty when no
// mapping applies. Mirrors the fundamental table in TypeTranslator.cpp
// plus a few common templated forms (std::complex, std::string,
// std::string_view) so PyiEmitter can normalize values that arrive
// post-Visitor (template-arg substitution where the value comes
// straight from the @instantiate_as directive).
// NOLINTNEXTLINE(misc-no-recursion)
std::string cpp_scalar_to_py(std::string const &cpp) {
    static std::unordered_map<std::string, std::string> const table = {
        {"void", "None"},          {"bool", "bool"},
        {"char", "str"},           {"signed char", "int"},
        {"unsigned char", "int"},  {"short", "int"},
        {"unsigned short", "int"}, {"int", "int"},
        {"unsigned", "int"},       {"unsigned int", "int"},
        {"long", "int"},           {"unsigned long", "int"},
        {"long long", "int"},      {"unsigned long long", "int"},
        {"size_t", "int"},         {"ssize_t", "int"},
        {"ptrdiff_t", "int"},      {"int8_t", "int"},
        {"uint8_t", "int"},        {"int16_t", "int"},
        {"uint16_t", "int"},       {"int32_t", "int"},
        {"uint32_t", "int"},       {"int64_t", "int"},
        {"uint64_t", "int"},       {"float", "float"},
        {"double", "float"},       {"long double", "float"},
        {"std::string", "str"},    {"std::string_view", "str"},
    };
    if (auto const it = table.find(cpp); it != table.end()) {
        return it->second;
    }
    // Strip leading "::" for tolerance.
    if (cpp.size() >= 2 && cpp[0] == ':' && cpp[1] == ':') {
        return cpp_scalar_to_py(cpp.substr(2));
    }
    // Common templated forms — match by prefix + trailing '>'.
    auto match_template = [&cpp](char const *prefix) -> bool {
        std::size_t const plen = std::char_traits<char>::length(prefix);
        return cpp.size() > plen && cpp.compare(0, plen, prefix) == 0 && cpp.back() == '>';
    };
    if (match_template("std::complex<") || match_template("complex<")) {
        return "complex";
    }
    if (match_template("std::vector<") || match_template("std::list<") || match_template("std::deque<")) {
        // Fall through to general handling — caller may want
        // ``list[T_py]``. For substitution-value purposes the simplest
        // useful answer is ``list``.
        return "list";
    }
    return {};
}

// Resolve unbound-name fall-throughs in py_type strings against the
// name map. The TypeTranslator emits the canonical C++ name when it
// can't translate (most often a bound class). We rewrite those tokens
// to the matching Python identifier; remaining unknown identifiers
// stay as ``Any`` rather than dangling C++ names.
std::string resolve_py_type(std::string const &py_type, NameMap const &names) {
    if (py_type.empty()) {
        return "None";
    }
    // Builtin Python types pass through untouched (covered by
    // translate_python_type's mapping). For everything else, walk
    // identifier tokens and try to resolve.
    auto is_ident_start = [](char c) { return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_'; };
    auto is_ident_cont  = [](char c) { return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_' || c == ':'; };

    std::string out;
    out.reserve(py_type.size());
    std::size_t i = 0;
    while (i < py_type.size()) {
        char const c = py_type[i];
        if (is_ident_start(c)) {
            std::size_t j = i + 1;
            while (j < py_type.size() && is_ident_cont(py_type[j])) {
                ++j;
            }
            std::string tok = py_type.substr(i, j - i);
            // Strip leading "::" if any.
            if (tok.size() >= 2 && tok[0] == ':' && tok[1] == ':') {
                tok.erase(0, 2);
            }
            // Look up the whole token first; this catches both the
            // simple bound class case and dotted qualified names. For
            // template-bearing names like "Tensor<float, 2>" the
            // tokenizer splits at '<', so we'd only see "Tensor" — for
            // those we look up the qualified name + the template args
            // already in the surrounding text. As a fallback, leave
            // the token as-is and pyright will surface the issue.
            auto const it = names.cpp_to_py.find(tok);
            if (it != names.cpp_to_py.end()) {
                out += it->second;
            } else {
                out += py_type.substr(i, j - i);
            }
            i = j;
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

// Strip cv/ref/ptr qualifiers from a C++ type string. Used before
// name-map lookups so post-substitution forms like
// ``einsums::Tensor<float> &`` match the map key
// ``einsums::Tensor<float>``.
std::string strip_cpp_qualifiers(std::string const &s) {
    std::string out        = s;
    auto        trim_edges = [&]() {
        while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) {
            out.pop_back();
        }
        std::size_t start = 0;
        while (start < out.size() && (out[start] == ' ' || out[start] == '\t')) {
            ++start;
        }
        if (start > 0) {
            out.erase(0, start);
        }
    };
    trim_edges();
    bool changed = true;
    while (changed) {
        changed = false;
        while (!out.empty() && (out.back() == '*' || out.back() == '&')) {
            out.pop_back();
            trim_edges();
            changed = true;
        }
        for (auto const &kw : {std::string{"const "}, std::string{"volatile "}}) {
            if (out.size() >= kw.size() && out.compare(0, kw.size(), kw) == 0) {
                out.erase(0, kw.size());
                trim_edges();
                changed = true;
            }
        }
        for (auto const &sfx : {std::string{" const"}, std::string{" volatile"}}) {
            if (out.size() >= sfx.size() && out.compare(out.size() - sfx.size(), sfx.size(), sfx) == 0) {
                out.erase(out.size() - sfx.size());
                trim_edges();
                changed = true;
            }
        }
    }
    return out;
}

// Try to resolve a full templated C++ name like "ns::Tensor<float, 2>"
// against per-instantiation Python names. Returns empty string if no
// exact match. Strips cv/ref/ptr qualifiers first so a post-substitution
// ``Tensor<float> &`` matches the map key ``Tensor<float>``.
std::string resolve_instantiation_name(std::string const &py_type, NameMap const &names) {
    std::string key = strip_cpp_qualifiers(py_type);
    if (key.empty() || key.find('<') == std::string::npos) {
        return {};
    }
    if (key.size() >= 2 && key[0] == ':' && key[1] == ':') {
        key.erase(0, 2);
    }
    auto const it = names.cpp_to_py.find(key);
    if (it != names.cpp_to_py.end()) {
        return it->second;
    }
    return {};
}

// True when `s` looks like an unresolved C++ name leaking into the
// stub: contains either a `::` qualifier or `<>` template syntax that
// pyright can't interpret. We collapse those to ``Any`` rather than
// emit invalid Python.
bool looks_like_unresolved_cpp(std::string const &s) {
    if (s.find("::") != std::string::npos) {
        return true;
    }
    // Bare angle brackets are template syntax. Allow them only inside
    // bracketed forms that python typing actually supports — those use
    // `[`/`]` instead, so any `<` here is C++.
    return s.find('<') != std::string::npos;
}

// Forward declaration — strip_value_type and expand_value_type_refs
// share the helper below.
std::string strip_value_type(std::string const &s);

// Replace every cpp_to_py-mapped class name (the FULL instantiation
// key, e.g. ``einsums::GeneralRuntimeTensor<float, std::allocator<float>>``)
// in `s` with its Python identifier. Used as a final pre-pass before
// `resolve()` so nested types like ``std::tuple<Class<args>, ...>``
// reduce to ``tuple[PyName, ...]`` instead of falling through to ``Any``.
std::string inline_bound_class_names(std::string s, NameMap const &names) {
    if (s.empty()) {
        return s;
    }
    // Sort keys by descending length so longer names match first (avoids
    // a substring of one binding eating the prefix of another).
    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(names.cpp_to_py.size());
    for (auto const &[k, v] : names.cpp_to_py) {
        if (k.find('<') != std::string::npos) {
            entries.emplace_back(k, v);
        }
    }
    std::sort(entries.begin(), entries.end(), [](auto const &a, auto const &b) { return a.first.size() > b.first.size(); });
    for (auto const &[needle, replacement] : entries) {
        std::size_t pos = 0;
        while ((pos = s.find(needle, pos)) != std::string::npos) {
            s.replace(pos, needle.size(), replacement);
            pos += replacement.size();
        }
    }
    return s;
}

// Walk `s` and replace every ``typename Class<args>::ValueType``
// occurrence with ``args.first``. Handles nested templates (the class
// name may itself carry `<>` arguments). One pass, left-to-right; the
// resulting text may still need additional resolve() processing for
// outer-level typedefs and bound-class lookups.
std::string expand_value_type_refs(std::string const &s) {
    static constexpr std::string_view k_typename = "typename ";
    static constexpr std::string_view k_value    = "::ValueType";
    std::string                       out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, k_typename.size(), k_typename) != 0) {
            out += s[i];
            ++i;
            continue;
        }
        // Scan from end-of-"typename " forward, tracking '<'/'>' depth,
        // until we hit a depth-0 "::ValueType".
        std::size_t const class_begin = i + k_typename.size();
        std::size_t       j           = class_begin;
        int               depth       = 0;
        bool              matched     = false;
        while (j < s.size()) {
            char const c = s[j];
            if (c == '<') {
                ++depth;
            } else if (c == '>') {
                if (depth > 0) {
                    --depth;
                }
            } else if (depth == 0 && j + k_value.size() <= s.size() && s.compare(j, k_value.size(), k_value) == 0) {
                std::string const cls_text{s.data() + class_begin, j - class_begin};
                std::string const vt = strip_value_type(std::string{k_typename} + cls_text + std::string{k_value});
                if (!vt.empty()) {
                    out += vt;
                    i       = j + k_value.size();
                    matched = true;
                    break;
                }
                break;
            }
            ++j;
        }
        if (!matched) {
            out += s[i];
            ++i;
        }
    }
    return out;
}

// Recognize the ``typename <Class>::ValueType`` pattern (after template
// arg substitution) and return the inner value-type. Einsums tensor
// classes all follow ``Tensor<T, ...>`` shape where ``ValueType`` is
// the first template argument, so resolving via "first arg of <Class>"
// gives the right scalar type for the instantiation.
std::string strip_value_type(std::string const &s) {
    auto trim_view = [](std::string_view v) {
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
            v.remove_prefix(1);
        }
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) {
            v.remove_suffix(1);
        }
        return v;
    };
    std::string_view sv = trim_view(std::string_view{s});
    if (sv.size() >= 9 && sv.substr(0, 9) == "typename ") {
        sv = trim_view(sv.substr(9));
    }
    static constexpr std::string_view k_suffix = "::ValueType";
    if (sv.size() <= k_suffix.size() || sv.substr(sv.size() - k_suffix.size()) != k_suffix) {
        return {};
    }
    sv            = sv.substr(0, sv.size() - k_suffix.size());
    auto const lt = sv.find('<');
    if (lt == std::string_view::npos) {
        return {};
    }
    int depth = 0;
    for (std::size_t i = lt + 1; i < sv.size(); ++i) {
        char const c = sv[i];
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            if (depth == 0) {
                return std::string{trim_view(sv.substr(lt + 1, i - lt - 1))};
            }
            --depth;
        } else if (c == ',' && depth == 0) {
            return std::string{trim_view(sv.substr(lt + 1, i - lt - 1))};
        }
    }
    return {};
}

// Whole-token identifier substitution on a raw C++ type string.
// Unlike substitute_python_template_params (which pre-resolves each
// value through the Python translator), this preserves values verbatim
// so the caller can run the result through the full resolver. Used to
// build per-instantiation parameter annotations from raw param types.
std::string substitute_cpp_idents(std::string const &type, std::vector<std::string> const &names, std::vector<std::string> const &values) {
    if (names.empty() || values.empty()) {
        return type;
    }
    std::unordered_map<std::string, std::string> map;
    for (std::size_t i = 0; i < names.size() && i < values.size(); ++i) {
        map[names[i]] = values[i];
    }
    auto        is_ident_start = [](char c) { return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_'; };
    auto        is_ident_cont  = [](char c) { return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_'; };
    std::string out;
    out.reserve(type.size());
    std::size_t i = 0;
    while (i < type.size()) {
        char const c = type[i];
        if (is_ident_start(c)) {
            std::size_t j = i + 1;
            while (j < type.size() && is_ident_cont(type[j])) {
                ++j;
            }
            std::string const tok = type.substr(i, j - i);
            auto const        it  = map.find(tok);
            if (it != map.end()) {
                out += it->second;
            } else {
                out += tok;
            }
            i = j;
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

// Final resolution: try the templated-instantiation lookup first, then
// run the full TypeTranslator string pipeline (handles std::vector<T>
// → list[T_py], std::optional<T> → T | None, etc. recursively), then
// fall back to identifier rewriting + fundamental-type table.
// Anything still containing C++ syntax collapses to Any so the stub
// stays valid Python.
// NOLINTNEXTLINE(misc-no-recursion)
std::string resolve(std::string const &py_type, NameMap const &names) {
    if (auto inst = resolve_instantiation_name(py_type, names); !inst.empty()) {
        return inst;
    }
    // ``typename Class<args>::ValueType`` resolves to the first template
    // arg of ``Class<args>`` — Einsums tensor types follow that shape.
    if (auto vt = strip_value_type(py_type); !vt.empty()) {
        return resolve(vt, names);
    }
    // Try the full string translator first — picks up std::vector<T>,
    // std::optional<T>, std::pair<A,B>, etc. recursively. Returns the
    // input unchanged when no rule applies, so we still need the
    // fallthrough below for bound-name rewrites and unresolved leaks.
    std::string recursed = translate_python_type_string(py_type);
    if (recursed != py_type) {
        if (looks_like_unresolved_cpp(recursed)) {
            // The canonicalizer reduced ``py_type`` (e.g. stripped
            // ``const`` and ``&``) but the result still looks like a
            // C++ qualified name. Try the bound-class name map before
            // giving up — that's how a parameter typed as
            // ``Slab const &`` resolves to the Python ``Slab`` rather
            // than the catch-all ``Any``.
            std::string const mapped = resolve_py_type(recursed, names);
            if (!looks_like_unresolved_cpp(mapped)) {
                return mapped;
            }
            return "Any";
        }
        return recursed;
    }
    if (auto scalar = cpp_scalar_to_py(py_type); !scalar.empty()) {
        return scalar;
    }
    std::string out = resolve_py_type(py_type, names);
    if (looks_like_unresolved_cpp(out)) {
        return "Any";
    }
    return out;
}

// ── Param / signature emission ──────────────────────────────────────────

// One parameter slot in a Python signature.
struct PyParam {
    std::string name;
    std::string annotation;
    std::string default_value; // empty -> no default
};

// Pick a placeholder name for unnamed params so the stub stays valid
// Python. Slot index is stable across overloads.
std::string param_name_for(BoundParam const &p, std::size_t slot) {
    if (!p.name.empty()) {
        return p.name;
    }
    return "arg" + std::to_string(slot);
}

// True when a candidate default-value text isn't valid Python and so
// should be replaced with ``...`` (Python ellipsis is allowed as a stub
// placeholder default per PEP 484). Catches unresolved C++ leftovers
// like ``typename T::ValueType{0}`` and brace-init expressions.
bool default_needs_ellipsis(std::string const &v) {
    if (v.empty()) {
        return false;
    }
    if (v.find("::") != std::string::npos) {
        return true;
    }
    if (v.find("typename") != std::string::npos) {
        return true;
    }
    // Bare brace-init like ``{0}`` or ``{0, 1}`` isn't valid as a
    // Python default expression.
    if (v.front() == '{' && v.back() == '}') {
        return true;
    }
    if (v.find('{') != std::string::npos || v.find('}') != std::string::npos) {
        return true;
    }
    return false;
}

PyParam build_param(BoundParam const &p, std::size_t slot, NameMap const &names) {
    PyParam out;
    out.name       = param_name_for(p, slot);
    out.annotation = resolve(p.py_type.empty() ? std::string{"Any"} : p.py_type, names);
    if (p.default_value_py) {
        out.default_value = default_needs_ellipsis(*p.default_value_py) ? std::string{"..."} : *p.default_value_py;
    }
    return out;
}

void emit_signature(std::ostringstream &os, std::vector<PyParam> const &params, std::string const &ret) {
    os << "(";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << params[i].name << ": " << params[i].annotation;
        if (!params[i].default_value.empty()) {
            os << " = " << params[i].default_value;
        }
    }
    os << ") -> " << ret;
}

// ── Free-function emission ──────────────────────────────────────────────

void emit_function(std::ostringstream &os, BoundFunction const &f, NameMap const &names);

// Forward decls — defined further down (with the member-template
// helpers) but the dtype-dispatcher emission below needs them.
std::string format_dtype_literal(std::vector<std::string> const &aliases);
bool        literal_covers_default(std::vector<std::string> const &aliases, std::string const &default_dtype);

// One overload emission of a templated free function for a given
// BoundInstantiation. Builds a TemplateArgBindings-style map by pairing
// f.template_param_names with the instantiation's split type_args, then
// substitutes template params in each parameter's py_type. (Param py_type
// strings are produced from the unresolved primary template's signature,
// so without substitution we'd emit "AType" / "BType" placeholders.)
//
// Substitution is whole-token: a param annotated as ``Tensor<AType, 2>``
// gets ``AType`` rewritten to the instantiation's first type arg.
std::string substitute_python_template_params(std::string s, std::vector<std::string> const &param_names,
                                              std::vector<std::string> const &values, NameMap const &names) {
    if (param_names.empty() || values.empty()) {
        return s;
    }
    std::unordered_map<std::string, std::string> map;
    for (std::size_t i = 0; i < param_names.size() && i < values.size(); ++i) {
        // Resolve the value through scalar-type mapping → bound-name
        // map → fallthrough. Order matters: ``double`` must map to
        // ``float`` (Python), not pass through as ``double``.
        map[param_names[i]] = resolve(values[i], names);
    }

    auto is_ident_start = [](char c) { return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_'; };
    auto is_ident_cont  = [](char c) { return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_'; };

    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        char const c = s[i];
        if (is_ident_start(c)) {
            std::size_t j = i + 1;
            while (j < s.size() && is_ident_cont(s[j])) {
                ++j;
            }
            std::string const tok = s.substr(i, j - i);
            auto const        it  = map.find(tok);
            if (it != map.end()) {
                out += it->second;
            } else {
                out += tok;
            }
            i = j;
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

// Split a per-instantiation comma list (mirrors PythonOverloads's
// helper but kept local to avoid cross-link shenanigans). Respects
// `<>` nesting.
std::vector<std::string> split_targs(std::string const &combo) {
    std::vector<std::string> parts;
    std::string              cur;
    int                      depth = 0;
    for (char const c : combo) {
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            if (depth > 0) {
                --depth;
            }
        }
        if (c == ',' && depth == 0) {
            // trim
            std::size_t b = 0, e = cur.size();
            while (b < e && cur[b] == ' ') {
                ++b;
            }
            while (e > b && cur[e - 1] == ' ') {
                --e;
            }
            parts.push_back(cur.substr(b, e - b));
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        std::size_t b = 0, e = cur.size();
        while (b < e && cur[b] == ' ') {
            ++b;
        }
        while (e > b && cur[e - 1] == ' ') {
            --e;
        }
        parts.push_back(cur.substr(b, e - b));
    }
    return parts;
}

// Build the resolved Python signature for one instantiation of a
// templated free function. For per-instantiation use we substitute
// template params on the RAW C++ types (preserving forms like
// ``typename T::ValueType``), then run the result through the full
// resolver — this gives concrete types per overload (e.g. ``factor:
// float`` for the float instantiation, ``A: RuntimeTensorF``, etc.)
// instead of bare ``Any`` for everything that touched a typename.
std::pair<std::vector<PyParam>, std::string> resolve_instantiation_signature(BoundFunction const &f, BoundInstantiation const &inst,
                                                                             NameMap const &names) {
    auto const           values = split_targs(inst.type_args);
    std::vector<PyParam> params;
    params.reserve(f.params.size());
    // When substituting on the canonical (typedef-expanded) form,
    // clang reports template parameters as ``type-parameter-0-0``,
    // ``type-parameter-0-1``, etc. (depth-positional internal names)
    // — the source template parameter NAME (e.g. ``T``) is gone. Map
    // those positional names to the same instantiation values.
    std::vector<std::string> canonical_param_names;
    canonical_param_names.reserve(f.template_param_names.size());
    for (std::size_t i = 0; i < f.template_param_names.size(); ++i) {
        canonical_param_names.push_back("type-parameter-0-" + std::to_string(i));
    }

    // Helper: try resolving the as-written substituted form first, then
    // fall back to the canonical (typedef-expanded) form. The
    // typedef-expanded form is what the cross-module name map keys on,
    // so this catches aliases like ``RuntimeTensor<T>`` →
    // ``GeneralRuntimeTensor<T, std::allocator<T>>``.
    auto resolve_with_canonical = [&](std::string const &as_written, std::string const &canonical) -> std::string {
        std::string subbed = substitute_cpp_idents(as_written, f.template_param_names, values);
        // Inline ``typename X::ValueType`` references (which appear
        // nested inside std::tuple<...>, std::vector<...>, etc.) so the
        // outer name-map lookup sees concrete types.
        subbed = expand_value_type_refs(subbed);
        // Pre-resolve any bound-class instantiations we DO know about so
        // nested types (e.g. ``std::tuple<RuntimeTensor<float>, ...>``)
        // emit ``tuple[RuntimeTensorF, ...]`` rather than falling through
        // to ``Any``.
        subbed         = inline_bound_class_names(subbed, names);
        std::string r1 = resolve(subbed, names);
        if (r1 != "Any" && !looks_like_unresolved_cpp(r1)) {
            return r1;
        }
        if (canonical.empty() || canonical == as_written) {
            return r1;
        }
        // Substitute in canonical form using BOTH the source-level
        // template-param names (in case clang preserved them) and the
        // depth-positional ``type-parameter-0-N`` names (the usual
        // case for canonicalized types).
        std::string subbed_canon = substitute_cpp_idents(canonical, f.template_param_names, values);
        subbed_canon             = substitute_cpp_idents(subbed_canon, canonical_param_names, values);
        subbed_canon             = expand_value_type_refs(subbed_canon);
        subbed_canon             = inline_bound_class_names(subbed_canon, names);
        // ``type-parameter-0-N`` contains hyphens, so identifier-token
        // substitution above won't catch it (the lexer splits at '-').
        // Do a literal whole-string replace for those.
        for (std::size_t i = 0; i < canonical_param_names.size() && i < values.size(); ++i) {
            std::string const &needle      = canonical_param_names[i];
            std::string const &replacement = values[i];
            std::size_t        pos         = 0;
            while ((pos = subbed_canon.find(needle, pos)) != std::string::npos) {
                subbed_canon.replace(pos, needle.size(), replacement);
                pos += replacement.size();
            }
        }
        std::string r2 = resolve(subbed_canon, names);
        if (r2 != "Any" && !looks_like_unresolved_cpp(r2)) {
            return r2;
        }
        return r1;
    };

    for (std::size_t i = 0; i < f.params.size(); ++i) {
        PyParam slot;
        slot.name       = param_name_for(f.params[i], i);
        slot.annotation = resolve_with_canonical(f.params[i].type, f.params[i].type_canonical);

        if (f.params[i].default_value_py) {
            slot.default_value = substitute_python_template_params(*f.params[i].default_value_py, f.template_param_names, values, names);
            if (default_needs_ellipsis(slot.default_value)) {
                slot.default_value = "...";
            }
        }
        params.push_back(std::move(slot));
    }
    std::string ret = resolve_with_canonical(f.return_type.empty() ? std::string{"void"} : f.return_type, f.return_type_canonical);
    return {std::move(params), std::move(ret)};
}

void emit_overload_decorator(std::ostringstream &os) {
    os << "@overload\n";
}

void emit_function(std::ostringstream &os, BoundFunction const &f, NameMap const &names) {
    if (is_hidden(f)) {
        return;
    }
    if (f.python_overloads.empty()) {
        // Templated function with no @instantiate_as: nothing to emit.
        return;
    }

    // Build one signature string per logical entry, then dedupe by
    // signature within the same py_name. Two entries that differ only
    // by C++ types collapsed to the same Python types (e.g. float and
    // double both → Python ``float``) are indistinguishable in a stub
    // and would otherwise produce duplicate `def` lines.
    struct Rendered {
        std::string py_name;
        std::string signature; // "(arg: T, ...) -> Ret"
    };
    std::vector<Rendered> entries;

    auto render_signature = [](std::vector<PyParam> const &params, std::string const &ret) {
        std::ostringstream s;
        emit_signature(s, params, ret);
        return s.str();
    };

    auto render_template_kwargs_signature = [](std::vector<PyParam> const &params, std::vector<std::string> const &kwargs,
                                               std::string const &ret) {
        std::ostringstream s;
        s << "(";
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (i != 0) {
                s << ", ";
            }
            s << params[i].name << ": " << params[i].annotation;
            if (!params[i].default_value.empty()) {
                s << " = " << params[i].default_value;
            }
        }
        if (!params.empty()) {
            s << ", ";
        }
        s << "*";
        for (auto const &kw : kwargs) {
            s << ", " << kw << ": bool = False";
        }
        s << ") -> " << ret;
        return s.str();
    };

    for (auto const &ov : f.python_overloads) {
        Rendered rendered;
        switch (ov.kind) {
        case PythonOverload::Kind::NonTemplate: {
            std::vector<PyParam> params;
            params.reserve(f.params.size());
            for (std::size_t i = 0; i < f.params.size(); ++i) {
                params.push_back(build_param(f.params[i], i, names));
            }
            std::string const ret = resolve(f.return_py_type.empty() ? std::string{"None"} : f.return_py_type, names);
            rendered.py_name      = py_name_for(f);
            rendered.signature    = render_signature(params, ret);
            break;
        }
        case PythonOverload::Kind::SingleInstantiation:
        case PythonOverload::Kind::OverloadSet: {
            auto const &inst   = f.instantiations[ov.instantiation_indices.front()];
            auto [params, ret] = resolve_instantiation_signature(f, inst, names);
            rendered.py_name   = ov.py_name;
            rendered.signature = render_signature(params, ret);
            break;
        }
        case PythonOverload::Kind::DtypeDispatcher: {
            // Emit ONE @overload per dtype with a ``Literal["..."]``-
            // typed dtype kwarg and the concrete return type, plus a
            // final fallback overload typed ``dtype: str`` returning
            // the union for non-literal callers. The default value
            // sits on the overload whose aliases include the picked
            // default_dtype, so ``fn(x)`` (no dtype) still resolves to
            // the concrete return type rather than the union.
            auto const &first_inst = f.instantiations[ov.instantiation_indices.front()];
            auto [base_params, _]  = resolve_instantiation_signature(f, first_inst, names);
            (void)_;

            // Build per-instantiation (resolved_return, aliases).
            std::vector<std::pair<std::string, std::vector<std::string>>> per_inst;
            std::vector<std::string>                                      ret_options;
            for (std::size_t const idx : ov.instantiation_indices) {
                auto [_p, r] = resolve_instantiation_signature(f, f.instantiations[idx], names);
                (void)_p;
                std::string const dtype_value = ov.dtype_values[per_inst.size()];
                per_inst.emplace_back(r, dtype_aliases_for(dtype_value));
                if (std::find(ret_options.begin(), ret_options.end(), r) == ret_options.end()) {
                    ret_options.push_back(r);
                }
            }
            std::string ret_union;
            if (ret_options.empty()) {
                ret_union = "Any";
            } else if (ret_options.size() == 1) {
                ret_union = ret_options.front();
            } else {
                for (std::size_t i = 0; i < ret_options.size(); ++i) {
                    if (i != 0) {
                        ret_union += " | ";
                    }
                    ret_union += ret_options[i];
                }
            }

            // Emit one Literal[...] overload per instantiation.
            for (auto const &[ret_t, aliases] : per_inst) {
                auto    params = base_params;
                PyParam dtype;
                dtype.name       = "dtype";
                dtype.annotation = format_dtype_literal(aliases);
                if (literal_covers_default(aliases, ov.default_dtype)) {
                    dtype.default_value = "\"" + ov.default_dtype + "\"";
                }
                params.push_back(std::move(dtype));
                Rendered r;
                r.py_name   = ov.py_name;
                r.signature = render_signature(params, ret_t);
                entries.push_back(std::move(r));
            }
            // Plus the str fallback so non-literal dtype values still
            // type-check (return widens to the union there).
            {
                auto    params = base_params;
                PyParam dtype;
                dtype.name          = "dtype";
                dtype.annotation    = "str";
                dtype.default_value = "\"" + ov.default_dtype + "\"";
                params.push_back(std::move(dtype));
                Rendered r;
                r.py_name   = ov.py_name;
                r.signature = render_signature(params, ret_union);
                entries.push_back(std::move(r));
            }
            continue; // skip the bottom dedupe-and-push, we pushed directly
        }
        case PythonOverload::Kind::TemplateKwargsDispatcher: {
            auto const &inst   = f.instantiations[ov.instantiation_indices.front()];
            auto [params, ret] = resolve_instantiation_signature(f, inst, names);
            rendered.py_name   = ov.py_name;
            rendered.signature = render_template_kwargs_signature(params, ov.kwarg_names, ret);
            break;
        }
        }
        // Skip exact duplicates from prior entries.
        bool duplicate = false;
        for (auto const &prev : entries) {
            if (prev.py_name == rendered.py_name && prev.signature == rendered.signature) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            entries.push_back(std::move(rendered));
        }
    }

    // Count remaining entries per py_name to decide on @overload.
    std::unordered_map<std::string, std::size_t> per_name;
    for (auto const &e : entries) {
        ++per_name[e.py_name];
    }
    // Use the function's docstring on every overload — pybind11 attaches
    // the same doc to each pybind11 overload, and pyright surfaces the
    // first match on hover. (Per-instantiation docs aren't tracked.)
    for (auto const &e : entries) {
        if (per_name[e.py_name] > 1) {
            emit_overload_decorator(os);
        }
        if (f.doc.empty()) {
            os << "def " << e.py_name << e.signature << ": ...\n";
        } else {
            os << "def " << e.py_name << e.signature << ":\n";
            emit_docstring(os, f.doc, "    ");
            os << "    ...\n";
        }
    }
}

// ── Class emission ──────────────────────────────────────────────────────

// One emitted Python class: either a non-template class (one) or one
// per instantiation of a templated class.
struct ClassEmission {
    std::string py_name;
    // For templated classes, the substitution map T->concrete used to
    // resolve member signatures. Empty for non-template classes.
    std::vector<std::string> tparam_names;
    std::vector<std::string> tparam_values;
};

// Trim leading/trailing whitespace.
std::string trim_ws(std::string s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

// Split a comma-separated payload respecting `<>` nesting. Mirrors the
// emitter's `split_instantiation_args` but lives here to avoid pulling
// in the whole emit-time helper graph.
std::vector<std::string> split_member_as_args(std::string const &payload) {
    std::vector<std::string> out;
    std::string              cur;
    int                      depth = 0;
    for (char const c : payload) {
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            if (depth > 0) {
                --depth;
            }
        }
        if (c == ',' && depth == 0) {
            out.push_back(trim_ws(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.push_back(trim_ws(cur));
    }
    return out;
}

// Substitute template-name → concrete-type bindings on a raw C++ type
// string and resolve the result through the full pipeline: canonical-
// form fallback, ``typename X::ValueType`` expansion, bound-class
// inlining, scalar mapping, and the recursive python translator.
// Anything that survives all of those checks while still containing
// C++ syntax collapses to ``Any``.
std::string resolve_substituted_type(std::string const &as_written, std::string const &canonical,
                                     std::vector<std::string> const &names_to_subst, std::vector<std::string> const &values,
                                     NameMap const &names) {
    std::vector<std::string> canonical_param_names;
    canonical_param_names.reserve(names_to_subst.size());
    for (std::size_t i = 0; i < names_to_subst.size(); ++i) {
        canonical_param_names.push_back("type-parameter-0-" + std::to_string(i));
    }
    std::string subbed = substitute_cpp_idents(as_written, names_to_subst, values);
    subbed             = expand_value_type_refs(subbed);
    subbed             = inline_bound_class_names(subbed, names);
    std::string r1     = resolve(subbed, names);
    if (r1 != "Any" && !looks_like_unresolved_cpp(r1)) {
        return r1;
    }
    if (canonical.empty() || canonical == as_written) {
        return r1;
    }
    std::string subbed_canon = substitute_cpp_idents(canonical, names_to_subst, values);
    subbed_canon             = substitute_cpp_idents(subbed_canon, canonical_param_names, values);
    subbed_canon             = expand_value_type_refs(subbed_canon);
    for (std::size_t i = 0; i < canonical_param_names.size() && i < values.size(); ++i) {
        std::string const &needle      = canonical_param_names[i];
        std::string const &replacement = values[i];
        std::size_t        pos         = 0;
        while ((pos = subbed_canon.find(needle, pos)) != std::string::npos) {
            subbed_canon.replace(pos, needle.size(), replacement);
            pos += replacement.size();
        }
    }
    subbed_canon   = inline_bound_class_names(subbed_canon, names);
    std::string r2 = resolve(subbed_canon, names);
    if (r2 != "Any" && !looks_like_unresolved_cpp(r2)) {
        return r2;
    }
    return r1;
}

// Format a list of dtype string aliases into a Python ``Literal[...]``
// annotation (e.g. ``["float32", "float", "f4"]`` →
// ``Literal["float32", "float", "f4"]``). Returns just ``str`` when
// the alias list is empty so the caller can use it directly.
std::string format_dtype_literal(std::vector<std::string> const &aliases) {
    if (aliases.empty()) {
        return "str";
    }
    std::string out = "Literal[";
    for (std::size_t i = 0; i < aliases.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += "\"" + aliases[i] + "\"";
    }
    out += "]";
    return out;
}

// True when any element of ``aliases`` matches ``default``. Used to
// decide which Literal overload carries the dtype default value.
bool literal_covers_default(std::vector<std::string> const &aliases, std::string const &default_dtype) {
    for (auto const &a : aliases) {
        if (a == default_dtype) {
            return true;
        }
    }
    return false;
}

// One Python-facing binding produced by an EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS
// directive on a templated method. The C++ emitter writes one
// ``.def(...)`` per directive (or merges into a dtype dispatcher when
// the group qualifies); the .pyi emitter mirrors that by emitting one
// stub per directive with the per-binding template substitution
// applied.
struct MemberAsBinding {
    std::string              py_name;       // Python identifier for this directive.
    std::vector<std::string> tparam_names;  // method-template parameter names.
    std::vector<std::string> tparam_values; // concrete types for each name.
};

// Collect every EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS directive on a
// method as a separate binding. The directive shape is:
//   ``instantiate_member_as`` with args[0]=py_name, args[1]="U=float, V=int"
// (one or more ``Name=Type`` assignments, comma-separated, respecting
// ``<>`` nesting in the type).
std::vector<MemberAsBinding> collect_member_as_bindings(BoundMethod const &m) {
    std::vector<MemberAsBinding> out;
    for (auto const &d : m.directives) {
        if (d.name != "instantiate_member_as" || d.args.size() < 2) {
            continue;
        }
        MemberAsBinding mb;
        mb.py_name          = d.args[0];
        auto const &payload = d.args[1];
        auto const  pieces  = split_member_as_args(payload);
        for (auto const &piece : pieces) {
            auto const eq = piece.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            std::string const lhs = trim_ws(piece.substr(0, eq));
            std::string const rhs = trim_ws(piece.substr(eq + 1));
            if (!lhs.empty() && !rhs.empty()) {
                mb.tparam_names.push_back(lhs);
                mb.tparam_values.push_back(rhs);
            }
        }
        if (!mb.tparam_names.empty()) {
            out.push_back(std::move(mb));
        }
    }
    return out;
}

// Forward decls — these helpers are defined further down in the file
// but `emit_method_signature` needs them for the rich-comparison
// LSP-widening path.
std::string operator_dunder(BoundMethod const &m);
bool        is_rich_comparison_dunder(std::string const &dunder);

void emit_method_signature(std::ostringstream &os, BoundMethod const &m, NameMap const &names, ClassEmission const &ce, bool is_init,
                           std::vector<std::string> const &extra_names = {}, std::vector<std::string> const &extra_values = {}) {
    // Build a combined substitution map: enclosing class's template
    // params first, then the method's own per-directive bindings
    // (one INSTANTIATE_MEMBER_AS directive's worth at a time, supplied
    // by the caller via ``extra_names``/``extra_values``).
    std::vector<std::string> all_names  = ce.tparam_names;
    std::vector<std::string> all_values = ce.tparam_values;
    all_names.insert(all_names.end(), extra_names.begin(), extra_names.end());
    all_values.insert(all_values.end(), extra_values.begin(), extra_values.end());

    auto resolve_for_method = [&](std::string const &as_written, std::string const &canonical) {
        return resolve_substituted_type(as_written, canonical, all_names, all_values, names);
    };

    auto subst = [&](std::string s) { return substitute_python_template_params(std::move(s), all_names, all_values, names); };

    std::vector<PyParam> params;
    params.reserve(m.params.size() + 1);
    if (!m.is_static) {
        PyParam self;
        self.name       = "self";
        self.annotation = ""; // self has no annotation
        params.push_back(std::move(self));
    }
    // Variadic-pack methods (EINSUMS_PYBIND_VARIADIC_FROM): the last
    // C++ parameter is a pack, but Python only sees N typed args of
    // ``variadic_element_type``. We fold the pack down to a single
    // ``*name: <element_type>`` slot for the stub. (pybind11 itself
    // expands per-instantiation to N positional ints; ``*args`` covers
    // every arity uniformly.)
    // For rich-comparison dunders the parameter type must be widened
    // to ``object`` (the type of ``object.__eq__``'s parameter) so the
    // stub doesn't violate LSP and trip pyright's
    // ``reportIncompatibleMethodOverride``.
    bool const widen_to_object = is_rich_comparison_dunder(operator_dunder(m));

    std::size_t const param_count = (m.has_variadic_pack && !m.params.empty()) ? m.params.size() - 1 : m.params.size();
    for (std::size_t i = 0; i < param_count; ++i) {
        PyParam slot;
        slot.name = param_name_for(m.params[i], i);
        // Same substitute-then-resolve flow the free-function path uses,
        // so per-instantiation member-template bindings (e.g. ``T=float``
        // from INSTANTIATE_MEMBER_AS) substitute on the raw C++ type and
        // re-translate, instead of just patching the already-translated
        // py_type (which collapses unresolved typenames to ``Any``).
        slot.annotation = resolve_for_method(m.params[i].type, m.params[i].type_canonical);
        if (m.params[i].default_value_py) {
            slot.default_value = subst(*m.params[i].default_value_py);
            if (default_needs_ellipsis(slot.default_value)) {
                slot.default_value = "...";
            }
        }
        if (widen_to_object) {
            slot.annotation = "object";
        }
        params.push_back(std::move(slot));
    }
    std::string ret = is_init ? std::string{"None"}
                              : resolve_for_method(m.return_type.empty() ? std::string{"void"} : m.return_type, m.return_type_canonical);

    os << "(";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        if (params[i].annotation.empty()) {
            os << params[i].name; // bare ``self``
        } else {
            os << params[i].name << ": " << params[i].annotation;
            if (!params[i].default_value.empty()) {
                os << " = " << params[i].default_value;
            }
        }
    }
    if (m.has_variadic_pack && !m.params.empty()) {
        if (!params.empty()) {
            os << ", ";
        }
        std::string const pack_name = m.params.back().name.empty() ? std::string{"args"} : m.params.back().name;
        std::string const element_type =
            subst(resolve(m.variadic_element_type.empty() ? std::string{"int"} : m.variadic_element_type, names));
        // The variadic_element_type comes from the C++ source verbatim
        // (e.g. ``int``); resolve() handles the few simple cases. For
        // anything more exotic we fall back to ``int`` since pybind11's
        // typical variadic pattern is dim packs.
        os << "*" << pack_name << ": " << element_type;
    }
    os << ") -> " << ret;
}

// Look up the operator dunder name from a method's @operator directive.
// Returns empty for non-operator methods or when the directive is absent.
std::string operator_dunder(BoundMethod const &m) {
    return first_directive_arg(m.directives, "operator");
}

// Rich-comparison dunders inherit a parameter type of ``object`` from
// ``object.__eq__`` / etc., so any narrower override triggers
// pyright's LSP-violation diagnostic. Detect them so we can override
// the param annotation in the stub.
bool is_rich_comparison_dunder(std::string const &dunder) {
    return dunder == "__eq__" || dunder == "__ne__" || dunder == "__lt__" || dunder == "__le__" || dunder == "__gt__" || dunder == "__ge__";
}

// True when the method participates in a property (and so should NOT
// be emitted as a regular def).
bool method_in_property(BoundMethod const &m) {
    return has_directive(m.directives, "getter") || has_directive(m.directives, "setter");
}

void emit_class(std::ostringstream &os, BoundClass const &cls, NameMap const &names);

// NOLINTNEXTLINE(misc-no-recursion)
void emit_class_body(std::ostringstream &os, BoundClass const &cls, ClassEmission const &ce, NameMap const &names) {
    bool wrote_anything = false;

    // Fields
    for (auto const &f : cls.fields) {
        if (is_hidden(f)) {
            continue;
        }
        std::string const py_type = substitute_python_template_params(resolve(f.py_type.empty() ? std::string{"Any"} : f.py_type, names),
                                                                      ce.tparam_names, ce.tparam_values, names);
        os << "    " << py_name_for(f) << ": " << py_type << "\n";
        wrote_anything = true;
    }

    // Properties
    for (auto const &p : cls.properties) {
        std::string const py_type = substitute_python_template_params(resolve(p.py_type.empty() ? std::string{"Any"} : p.py_type, names),
                                                                      ce.tparam_names, ce.tparam_values, names);
        os << "    @property\n";
        if (p.doc.empty()) {
            os << "    def " << p.py_name << "(self) -> " << py_type << ": ...\n";
        } else {
            os << "    def " << p.py_name << "(self) -> " << py_type << ":\n";
            emit_docstring(os, p.doc, "        ");
            os << "        ...\n";
        }
        if (p.has_setter) {
            os << "    @" << p.py_name << ".setter\n";
            os << "    def " << p.py_name << "(self, value: " << py_type << ") -> None: ...\n";
        }
        wrote_anything = true;
    }

    // Constructors → __init__
    bool const has_multiple_ctors = cls.ctors.size() > 1;
    for (auto const &ctor : cls.ctors) {
        if (is_hidden(ctor) || ctor.is_deleted) {
            continue;
        }
        if (has_multiple_ctors) {
            os << "    @overload\n";
        }
        os << "    def __init__";
        emit_method_signature(os, ctor, names, ce, /*is_init=*/true);
        if (ctor.doc.empty()) {
            os << ": ...\n";
        } else {
            os << ":\n";
            emit_docstring(os, ctor.doc, "        ");
            os << "        ...\n";
        }
        wrote_anything = true;
    }

    // Methods (skip property accessors and destructor).
    //
    // EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS fans out one Python method
    // per directive, each with its own template-arg substitution. We
    // pre-compute the Python name for every emission (one per regular
    // method, N per method-with-INSTANTIATE_MEMBER_AS) so the @overload
    // decorator counts cover both.
    auto method_emit_name = [&](BoundMethod const &m, std::string const &fallback) {
        std::string const op = operator_dunder(m);
        return op.empty() ? fallback : op;
    };

    std::unordered_map<std::string, std::size_t> name_count;
    for (auto const &m : cls.methods) {
        if (is_hidden(m) || m.is_destructor || m.is_deleted) {
            continue;
        }
        if (method_in_property(m)) {
            continue;
        }
        auto const member_as = collect_member_as_bindings(m);
        if (member_as.empty()) {
            ++name_count[method_emit_name(m, py_name_for(m))];
            continue;
        }
        for (auto const &b : member_as) {
            ++name_count[method_emit_name(m, b.py_name)];
        }
    }

    // Render a single method signature into a string so we can dedupe
    // member-as overloads whose per-binding substitutions collapse to
    // the same Python signature (e.g. ``U=float`` and ``U=double``
    // both produce ``-> float``). The free-function emit path does the
    // same.
    auto render_method_sig = [&](BoundMethod const &m, std::vector<std::string> const &extra_names,
                                 std::vector<std::string> const &extra_values) {
        std::ostringstream s;
        emit_method_signature(s, m, names, ce, /*is_init=*/false, extra_names, extra_values);
        return s.str();
    };

    auto emit_one_method = [&](BoundMethod const &m, std::string const &py, std::string const &signature) {
        bool const ovl = name_count[py] > 1;
        if (ovl) {
            os << "    @overload\n";
        }
        if (m.is_static) {
            os << "    @staticmethod\n";
        }
        os << "    def " << py << signature;
        if (m.doc.empty()) {
            os << ": ...\n";
        } else {
            os << ":\n";
            emit_docstring(os, m.doc, "        ");
            os << "        ...\n";
        }
    };

    for (auto const &m : cls.methods) {
        if (is_hidden(m) || m.is_destructor || m.is_deleted) {
            continue;
        }
        if (method_in_property(m)) {
            continue;
        }
        auto const member_as = collect_member_as_bindings(m);
        if (member_as.empty()) {
            std::string const py = method_emit_name(m, py_name_for(m));
            emit_one_method(m, py, render_method_sig(m, {}, {}));
            wrote_anything = true;
            continue;
        }

        // Group bindings by py_name (preserving directive order). Each
        // group is a candidate for the dtype-dispatcher collapse: when
        // multiple directives share a Python name AND a resolved arg
        // signature AND a recognized scalar dtype in their bindings,
        // pybind11 can't tell them apart at runtime — we must merge
        // them into one ``def`` taking ``dtype="..."``. Otherwise emit
        // as @overload entries (same logic as before).
        std::vector<std::string>                                    py_names_in_order;
        std::map<std::string, std::vector<MemberAsBinding const *>> groups;
        for (auto const &b : member_as) {
            std::string const py = method_emit_name(m, b.py_name);
            if (groups.find(py) == groups.end()) {
                py_names_in_order.push_back(py);
            }
            groups[py].push_back(&b);
        }

        // Find the first dtype-recognizable value in a binding's
        // tparam_values, or empty when none qualify.
        auto dtype_value_for = [](MemberAsBinding const &b) {
            for (auto const &v : b.tparam_values) {
                if (!dtype_aliases_for(v).empty()) {
                    return v;
                }
            }
            return std::string{};
        };

        // Render the *arg-only* part of a method signature (i.e.
        // ignoring the return type) so we can compare across bindings
        // in a group. Trailing ``-> Ret`` is stripped.
        auto strip_return_type = [](std::string const &sig) {
            std::size_t const arrow = sig.rfind(") -> ");
            if (arrow == std::string::npos) {
                return sig;
            }
            return sig.substr(0, arrow + 1); // keep the closing ')'
        };

        for (auto const &py : py_names_in_order) {
            auto const &group = groups[py];

            bool        same_args = group.size() > 1;
            std::string ref_args;
            if (same_args) {
                ref_args = strip_return_type(render_method_sig(m, group.front()->tparam_names, group.front()->tparam_values));
                for (std::size_t k = 1; k < group.size(); ++k) {
                    if (strip_return_type(render_method_sig(m, group[k]->tparam_names, group[k]->tparam_values)) != ref_args) {
                        same_args = false;
                        break;
                    }
                }
            }
            bool all_have_dtype = same_args;
            if (all_have_dtype) {
                for (auto const *b : group) {
                    if (dtype_value_for(*b).empty()) {
                        all_have_dtype = false;
                        break;
                    }
                }
            }

            if (same_args && all_have_dtype) {
                // ── Dtype dispatcher (N+1 @overloads) ─────────────
                // One ``Literal["..."]`` overload per directive plus
                // a final ``str``-typed fallback. Default value lives
                // on the overload whose aliases include the picked
                // default_dtype, so calling without ``dtype`` still
                // resolves to a concrete return type rather than the
                // union (which the fallback uses).
                std::vector<std::string>                                      dtype_values;
                std::vector<std::pair<std::string, std::vector<std::string>>> per_inst;
                std::vector<std::string>                                      ret_options;
                for (auto const *b : group) {
                    std::string const dval = dtype_value_for(*b);
                    dtype_values.push_back(dval);
                    std::string const sig   = render_method_sig(m, b->tparam_names, b->tparam_values);
                    auto const        arrow = sig.rfind(" -> ");
                    std::string       ret_t = (arrow != std::string::npos) ? sig.substr(arrow + 4) : std::string{"Any"};
                    per_inst.emplace_back(ret_t, dtype_aliases_for(dval));
                    if (std::find(ret_options.begin(), ret_options.end(), ret_t) == ret_options.end()) {
                        ret_options.push_back(ret_t);
                    }
                }
                std::string const default_dtype = pick_default_dtype(dtype_values);
                std::string       ret_union;
                if (ret_options.empty()) {
                    ret_union = "Any";
                } else if (ret_options.size() == 1) {
                    ret_union = ret_options.front();
                } else {
                    for (std::size_t i = 0; i < ret_options.size(); ++i) {
                        if (i != 0) {
                            ret_union += " | ";
                        }
                        ret_union += ret_options[i];
                    }
                }

                // ``args_only`` is the leading "(self, ...)" — we
                // splice the dtype kwarg before the closing ')'.
                std::string const first_sig = render_method_sig(m, group.front()->tparam_names, group.front()->tparam_values);
                std::string const args_only = strip_return_type(first_sig);
                std::string const args_open = args_only.substr(0, args_only.size() - 1); // drop trailing ')'
                std::string       args_pref = args_open;
                if (args_pref.size() > 1 && args_pref.back() != '(') {
                    args_pref += ", ";
                }

                // name_count was incremented once per directive; the
                // dispatcher emits `group.size() + 1` lines (N
                // Literal overloads plus the fallback) — bump up so
                // every emitted line gets the @overload decorator.
                name_count[py] += 1; // for the fallback

                for (auto const &[ret_t, aliases] : per_inst) {
                    std::string sig = args_pref + "dtype: " + format_dtype_literal(aliases);
                    if (literal_covers_default(aliases, default_dtype)) {
                        sig += " = \"" + default_dtype + "\"";
                    }
                    sig += ") -> " + ret_t;
                    emit_one_method(m, py, sig);
                }
                std::string fallback_sig = args_pref + "dtype: str = \"" + default_dtype + "\") -> " + ret_union;
                emit_one_method(m, py, fallback_sig);
                wrote_anything = true;
                continue;
            }

            // ── Plain @overload set ──────────────────────────────
            // Dedupe identical post-resolution signatures (e.g. binding
            // sets that collapse to the same Python types after scalar
            // mapping). Adjust name_count so the @overload decorator
            // only fires when distinct signatures remain.
            std::vector<std::string>                         seen;
            std::vector<std::pair<std::string, std::string>> to_emit;
            for (auto const *b : group) {
                std::string const sig = render_method_sig(m, b->tparam_names, b->tparam_values);
                if (std::find(seen.begin(), seen.end(), sig) != seen.end()) {
                    if (name_count[py] > 0) {
                        --name_count[py];
                    }
                    continue;
                }
                seen.push_back(sig);
                to_emit.emplace_back(py, sig);
            }
            for (auto const &[emit_py, emit_sig] : to_emit) {
                emit_one_method(m, emit_py, emit_sig);
            }
            wrote_anything = true;
        }
    }

    // Nested enums are emitted at module scope (matching Emitter.cpp's
    // pybind registration) — see the top-level emission loop. Skipping
    // them here so the parent class body doesn't shadow them with a
    // wrong-scope copy.

    // Nested classes
    for (auto const &n : cls.nested_classes) {
        if (is_hidden(n)) {
            continue;
        }
        std::ostringstream nested;
        emit_class(nested, n, names);
        // Indent each line by 4 spaces to nest under the parent class.
        std::string const text = nested.str();
        std::size_t       pos  = 0;
        while (pos < text.size()) {
            std::size_t const eol = text.find('\n', pos);
            if (eol == std::string::npos) {
                if (pos < text.size()) {
                    os << "    " << text.substr(pos) << "\n";
                }
                break;
            }
            if (eol > pos) {
                os << "    " << text.substr(pos, eol - pos) << "\n";
            } else {
                os << "\n";
            }
            pos = eol + 1;
        }
        wrote_anything = true;
    }

    if (!wrote_anything) {
        os << "    ...\n";
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
void emit_class(std::ostringstream &os, BoundClass const &cls, NameMap const &names) {
    if (is_hidden(cls)) {
        return;
    }
    // External annotated classes are recorded only so the type
    // translator can map cross-module C++ names to their Python
    // identifiers. Their full definitions live in the owning module's
    // own .pyi fragment; re-emitting empty shells here would produce
    // duplicate `class Foo: ...` entries once the aggregator
    // concatenates fragments.
    if (cls.is_external) {
        return;
    }

    // Exception classes (registered via py::register_exception): emit
    // as a Python Exception subclass. We don't try to bind methods.
    if (has_directive(cls.directives, "exception")) {
        os << "class " << py_name_for(cls) << "(Exception): ...\n\n";
        return;
    }

    auto bases_clause = [&](ClassEmission const &ce) {
        // Filter to bases that actually resolve to a Python identifier.
        // Anything that falls back to Any or stays as a raw C++ name
        // would just confuse pyright (and would emit `(Any, Any)` for
        // multiple unbound bases).
        std::vector<std::string> resolved;
        for (auto const &b : cls.bases) {
            std::string const py = resolve(b, names);
            if (py == b || py == "Any" || looks_like_unresolved_cpp(py)) {
                continue;
            }
            // Dedupe — multiple bases mapped to the same Python name
            // (e.g. via inheritance through several typedefs) become one.
            bool already = false;
            for (auto const &r : resolved) {
                if (r == py) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                resolved.push_back(py);
            }
        }
        std::string out;
        if (!resolved.empty()) {
            out += "(";
            for (std::size_t i = 0; i < resolved.size(); ++i) {
                if (i != 0) {
                    out += ", ";
                }
                out += resolved[i];
            }
            out += ")";
        }
        (void)ce;
        return out;
    };

    if (!cls.is_template) {
        ClassEmission ce;
        ce.py_name = py_name_for(cls);
        os << "class " << ce.py_name << bases_clause(ce) << ":\n";
        emit_docstring(os, cls.doc, "    ");
        emit_class_body(os, cls, ce, names);
        os << "\n";
        return;
    }

    // Templated class: one Python class per instantiation. The
    // BoundInstantiation.py_name is what Python sees. The class doc is
    // shared across instantiations (no per-instantiation doc tracked).
    for (auto const &inst : cls.instantiations) {
        ClassEmission ce;
        ce.py_name       = inst.py_name;
        ce.tparam_names  = cls.template_param_names;
        ce.tparam_values = split_targs(inst.type_args);
        os << "class " << ce.py_name << bases_clause(ce) << ":\n";
        emit_docstring(os, cls.doc, "    ");
        emit_class_body(os, cls, ce, names);
        os << "\n";
    }
}

// ── Top-level enum emission ─────────────────────────────────────────────

void emit_enum(std::ostringstream &os, BoundEnum const &e) {
    if (is_hidden(e)) {
        return;
    }
    os << "class " << py_name_for(e) << "(enum.IntEnum):\n";
    for (auto const &v : e.enumerators) {
        os << "    " << v.name << " = " << v.value << "\n";
    }
    os << "\n";
}

} // namespace

std::string emit_pyi(Module const &module_, PyiOptions const &opts) {
    std::ostringstream os;
    if (!opts.banner.empty()) {
        // Stubs are Python; comments use ``#``.
        os << "# " << opts.banner << "\n";
    }
    os << "# This file is auto-generated by einsums-pybind. Do not edit.\n";
    os << "# pyright: reportInvalidTypeForm=false\n";
    os << "\n";
    os << "from __future__ import annotations\n";
    os << "from typing import Any, Literal, overload\n";
    os << "from collections.abc import Callable\n";
    os << "import enum\n";
    os << "import numpy\n";
    os << "import numpy.typing\n";
    os << "\n";

    NameMap const names = build_name_map(module_);

    // Collect every nested enum (recursively, through nested classes) so
    // we can emit them at module scope alongside top-level enums —
    // mirroring Emitter.cpp's ``py::enum_<Parent::Inner>(_sub_x, "Inner")``
    // placement. The parent class's body skips them.
    std::vector<BoundEnum const *>                nested_enums;
    std::function<void(BoundClass const &)> const collect_nested = [&](BoundClass const &c) {
        for (auto const &e : c.nested_enums) {
            if (!is_hidden(e)) {
                nested_enums.push_back(&e);
            }
        }
        for (auto const &n : c.nested_classes) {
            collect_nested(n);
        }
    };
    for (auto const &c : module_.classes) {
        collect_nested(c);
    }

    // Group entities by submodule path so the aggregator can route
    // each block to the right per-submodule .pyi file. The sentinel
    // ``# %%submodule: <name>`` marks the start of a block; an empty
    // name means the entity binds at the top level (einsums._core).
    auto submodule_of = [](BoundEntityCommon const &e) -> std::string { return e.submodule.value_or(std::string{}); };

    // First pass: collect distinct submodule paths in stable order
    // (top-level first, then in insertion order of first sighting).
    std::vector<std::string>        submodules{""};
    std::unordered_set<std::string> seen{""};
    auto                            note = [&](std::string const &sm) {
        if (seen.insert(sm).second) {
            submodules.push_back(sm);
        }
    };
    for (auto const &e : module_.enums) {
        note(submodule_of(e));
    }
    for (BoundEnum const *e : nested_enums) {
        note(submodule_of(*e));
    }
    for (auto const &c : module_.classes) {
        note(submodule_of(c));
    }
    for (auto const &f : module_.functions) {
        note(submodule_of(f));
    }

    // Second pass: emit each submodule's block, prefixed with the
    // sentinel so the aggregator can split.
    for (auto const &sm : submodules) {
        std::ostringstream block;
        for (auto const &e : module_.enums) {
            if (submodule_of(e) == sm) {
                emit_enum(block, e);
            }
        }
        for (BoundEnum const *e : nested_enums) {
            if (submodule_of(*e) == sm) {
                emit_enum(block, *e);
            }
        }
        for (auto const &c : module_.classes) {
            if (submodule_of(c) == sm) {
                emit_class(block, c, names);
            }
        }
        for (auto const &f : module_.functions) {
            if (submodule_of(f) == sm) {
                emit_function(block, f, names);
            }
        }
        std::string const text = block.str();
        if (text.empty()) {
            continue;
        }
        os << "# %%submodule: " << sm << "\n";
        os << text;
    }
    return os.str();
}

} // namespace einsums::pybind
