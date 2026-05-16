//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Emitter.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "PythonOverloads.hpp"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/raw_ostream.h"

namespace einsums::pybind {

namespace {

// API strings for the active target binding library. Built once per
// emit() call; threaded through helpers that touch the API surface.
struct Backend {
    std::string              name;            // "pybind11" / "nanobind" — for diagnostics
    std::string              ns;              // "py" / "nb"
    std::string              ns_full;         // "pybind11" / "nanobind"
    std::string              module_macro;    // "PYBIND11_MODULE" / "NB_MODULE"
    std::vector<std::string> headers;         // includes for the binding TU
    std::string              rvp_prefix;      // "py::return_value_policy::" / "nb::rv_policy::"
    bool                     supports_buffer; // pybind11: true, nanobind: false (uses ndarray)
};

Backend make_backend(Target t) {
    Backend b;
    if (t == Target::Nanobind) {
        b.name         = "nanobind";
        b.ns           = "nb";
        b.ns_full      = "nanobind";
        b.module_macro = "NB_MODULE";
        // Nanobind splits STL bindings across per-type headers; pull in
        // the ones most modules need. Adding more is cheap; missing
        // ones produce clear errors at compile time.
        b.headers = {
            "<nanobind/nanobind.h>",   "<nanobind/operators.h>",   "<nanobind/stl/string.h>",
            "<nanobind/stl/vector.h>", "<nanobind/stl/complex.h>", "<nanobind/ndarray.h>",
        };
        b.rvp_prefix      = "nb::rv_policy::";
        b.supports_buffer = false;
    } else {
        b.name            = "pybind11";
        b.ns              = "py";
        b.ns_full         = "pybind11";
        b.module_macro    = "PYBIND11_MODULE";
        b.headers         = {"<pybind11/pybind11.h>", "<pybind11/complex.h>",   "<pybind11/functional.h>",
                             "<pybind11/numpy.h>",    "<pybind11/operators.h>", "<pybind11/stl.h>"};
        b.rvp_prefix      = "py::return_value_policy::";
        b.supports_buffer = true;
    }
    return b;
}

// ---- Directive lookup -----------------------------------------------------
// Directives are stored as a flat list to preserve order. The emitter
// usually wants O(1) "is X present?" or "what's the arg of X?" lookups,
// which a small string-keyed map handles cleanly.
class DirectiveView {
  public:
    explicit DirectiveView(DirectiveList const &dirs) {
        for (auto const &d : dirs) {
            _by_name[d.name].push_back(&d);
        }
    }

    [[nodiscard]] bool has(llvm::StringRef name) const { return _by_name.count(name.str()) != 0; }

    [[nodiscard]] Directive const *first(llvm::StringRef name) const {
        auto it = _by_name.find(name.str());
        if (it == _by_name.end() || it->second.empty()) {
            return nullptr;
        }
        return it->second.front();
    }

    [[nodiscard]] std::vector<Directive const *> all(llvm::StringRef name) const {
        auto it = _by_name.find(name.str());
        if (it == _by_name.end()) {
            return {};
        }
        return it->second;
    }

  private:
    std::map<std::string, std::vector<Directive const *>> _by_name;
};

// Return the Python identifier for a bound entity, honoring @rename.
std::string py_name_for(BoundEntityCommon const &e) {
    DirectiveView const v(e.directives);
    if (auto const *d = v.first("rename"); d != nullptr && !d->args.empty()) {
        return d->args.front();
    }
    return e.name;
}

// Submodule name from a @module directive, or empty if the entity binds
// at the top-level module. Dotted names ("tensor.algebra") select a
// nested submodule, with each component its own def_submodule call.
std::string submodule_path_for(BoundEntityCommon const &e) {
    DirectiveView const v(e.directives);
    if (auto const *d = v.first("module"); d != nullptr && !d->args.empty()) {
        return d->args.front();
    }
    return "";
}

// Convert a (possibly dotted) submodule path into the C++ variable name
// the emitter binds it under. Empty path -> "m" (the top-level module).
std::string submodule_var_name(std::string const &path) {
    if (path.empty()) {
        return "m";
    }
    std::string out = "_sub_";
    for (char const c : path) {
        out += (c == '.') ? '_' : c;
    }
    return out;
}

// Stringify a docstring as a C++ raw string literal — pybind11 takes the
// docstring as a const char *, and a raw literal sidesteps escape gymnastics
// for embedded quotes / backslashes / newlines.
std::string as_cpp_doc(std::string const &doc) {
    if (doc.empty()) {
        return "";
    }
    return "R\"einsums(" + doc + ")einsums\"";
}

// ---- Emission helpers -----------------------------------------------------

bool has_directive(BoundEntityCommon const &e, llvm::StringRef name) {
    return DirectiveView(e.directives).has(name);
}

bool is_hidden(BoundEntityCommon const &e) {
    return has_directive(e, "hide");
}

void emit_function_arg_modifiers(llvm::raw_string_ostream &os, BoundEntityCommon const &e, Backend const &b) {
    DirectiveView const v(e.directives);
    if (auto const *d = v.first("rvp"); d != nullptr && !d->args.empty()) {
        os << ", " << b.rvp_prefix << d->args.front();
    }
    for (Directive const *d : v.all("keep_alive")) {
        if (d->args.size() == 2) {
            os << ", " << b.ns << "::keep_alive<" << d->args[0] << ", " << d->args[1] << ">()";
        }
    }
    if (v.has("release_gil")) {
        os << ", " << b.ns << "::call_guard<" << b.ns << "::gil_scoped_release>()";
    }
}

void emit_doc_arg(llvm::raw_string_ostream &os, BoundEntityCommon const &e) {
    DirectiveView const v(e.directives);
    if (auto const *d = v.first("doc"); d != nullptr && !d->args.empty()) {
        os << ", " << as_cpp_doc(d->args.front());
        return;
    }
    if (!e.doc.empty()) {
        os << ", " << as_cpp_doc(e.doc);
    }
}

void emit_param_args(llvm::raw_string_ostream &os, std::vector<BoundParam> const &params, Backend const &b) {
    for (auto const &p : params) {
        os << ", " << b.ns << "::arg(\"";
        os << (p.name.empty() ? "arg" : p.name);
        os << "\")";
        if (p.default_value) {
            os << " = " << *p.default_value;
        }
    }
}

// Generate the C++ pointer-to-member expression for a method, including
// the static_cast<> needed to disambiguate overloads. For unambiguous
// methods the cast is harmless; for overloads it's required. Static
// methods take a free-function pointer cast (no Class::*) since they
// have no implicit ``this``.
std::string method_pointer(std::string const &class_qual, BoundMethod const &m) {
    std::string sig;
    sig += m.return_type;
    sig += " (";
    if (!m.is_static) {
        sig += class_qual;
        sig += "::*)(";
    } else {
        sig += "*)(";
    }
    for (std::size_t i = 0; i < m.params.size(); ++i) {
        if (i != 0) {
            sig += ", ";
        }
        sig += m.params[i].type;
    }
    sig += ")";
    if (m.is_const) {
        sig += " const";
    }
    std::string out = "static_cast<";
    out += sig;
    out += ">(&";
    out += class_qual;
    out += "::";
    out += m.name;
    out += ")";
    return out;
}

void emit_field(llvm::raw_string_ostream &os, std::string const &class_qual, BoundField const &f) {
    if (is_hidden(f)) {
        return;
    }
    DirectiveView const v(f.directives);
    std::string const   py   = py_name_for(f);
    char const         *kind = v.has("readonly") ? "def_readonly" : "def_readwrite";
    os << "        ." << kind << "(\"" << py << "\", &" << class_qual << "::" << f.name << "";
    emit_doc_arg(os, f);
    os << ")\n";
}

// Forward decls — these helpers live further down in the TU but
// emit_method needs them for the EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS
// fan-out / dtype-dispatcher path. dtype_aliases_for / pick_default_dtype
// live in PythonOverloads.hpp.
using TemplateArgBindings = std::map<std::string, std::string>;
TemplateArgBindings parse_member_as_bindings(std::string const &payload);
std::string         substitute_template_params(std::string const &type, TemplateArgBindings const &bindings);

// Bind a method as either a plain .def, an operator, or merge with a
// matching getter/setter pair into a .def_property. The caller passes the
// full method list so this helper can find the partner accessor.
void emit_method(llvm::raw_string_ostream &os, std::string const &class_qual, BoundMethod const &m,
                 std::vector<BoundMethod> const &siblings, Backend const &b) {
    if (is_hidden(m) || m.is_constructor || m.is_destructor || m.is_deleted) {
        return;
    }

    DirectiveView const v(m.directives);

    // Property emission: a method tagged @getter("X") merges with its
    // matching @setter("X") into a single .def_property. Without a
    // matching setter we emit .def_property_readonly. We emit only on
    // the getter pass and skip the setter; the inverse pass is a no-op.
    if (auto const *getter = v.first("getter"); getter != nullptr && !getter->args.empty()) {
        std::string const  prop   = getter->args.front();
        BoundMethod const *setter = nullptr;
        for (auto const &cand : siblings) {
            DirectiveView const cv(cand.directives);
            auto const         *sd = cv.first("setter");
            if (sd != nullptr && !sd->args.empty() && sd->args.front() == prop) {
                setter = &cand;
                break;
            }
        }
        if (setter == nullptr) {
            os << "        .def_property_readonly(\"" << prop << "\", " << method_pointer(class_qual, m);
        } else {
            os << "        .def_property(\"" << prop << "\", " << method_pointer(class_qual, m) << ", "
               << method_pointer(class_qual, *setter);
        }
        emit_doc_arg(os, m);
        os << ")\n";
        return;
    }
    if (v.has("setter")) {
        return; // emitted (or skipped) by the getter pass above
    }

    // Operator emission: pybind11 / nanobind both map Python dunder
    // names back to C++ operators via the explicit string form.
    if (auto const *op = v.first("operator"); op != nullptr && !op->args.empty()) {
        os << "        .def(\"" << op->args.front() << "\", " << method_pointer(class_qual, m);
        emit_param_args(os, m.params, b);
        emit_function_arg_modifiers(os, m, b);
        emit_doc_arg(os, m);
        os << ")\n";
        return;
    }

    // Member-template fan-out (EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS):
    // each directive defines one binding for a templated method with its
    // own pinned bindings. Multiple directives stack — when they share a
    // py_name AND only differ in return type (the dtype-dispatcher case),
    // collapse into one dispatcher .def taking a ``dtype="..."`` kwarg.
    // Otherwise emit each as its own binding (pybind11 overload set).
    auto const member_as_dirs = v.all("instantiate_member_as");
    if (!member_as_dirs.empty()) {
        // Group by py_name in directive order.
        std::vector<std::string>                                py_names;
        std::map<std::string, std::vector<TemplateArgBindings>> groups;
        for (auto const *d : member_as_dirs) {
            if (d->args.size() != 2) {
                continue;
            }
            std::string const         py_inst  = d->args[0];
            TemplateArgBindings const bindings = parse_member_as_bindings(d->args[1]);
            if (groups.find(py_inst) == groups.end()) {
                py_names.push_back(py_inst);
            }
            groups[py_inst].push_back(bindings);
        }

        auto resolved_arg_sig = [&](TemplateArgBindings const &bindings) {
            std::string sig = "(";
            for (std::size_t i = 0; i < m.params.size(); ++i) {
                if (i != 0) {
                    sig += ',';
                }
                sig += substitute_template_params(m.params[i].type, bindings);
            }
            sig += ')';
            return sig;
        };

        auto dtype_value_in = [](TemplateArgBindings const &bindings) {
            for (auto const &[k, v] : bindings) {
                if (!dtype_aliases_for(v).empty()) {
                    return v;
                }
            }
            return std::string{};
        };

        for (auto const &py_inst : py_names) {
            auto const &group = groups[py_inst];

            // Detect dtype-dispatcher case: same arg signature across the
            // group, every instance has a recognized dtype binding.
            bool same_sig = group.size() > 1;
            if (same_sig) {
                std::string const ref = resolved_arg_sig(group.front());
                for (std::size_t k = 1; k < group.size(); ++k) {
                    if (resolved_arg_sig(group[k]) != ref) {
                        same_sig = false;
                        break;
                    }
                }
            }
            bool all_dtype_known = same_sig;
            if (all_dtype_known) {
                for (auto const &bindings : group) {
                    if (dtype_value_in(bindings).empty()) {
                        all_dtype_known = false;
                        break;
                    }
                }
            }

            if (same_sig && all_dtype_known) {
                // ── Dtype dispatcher ──────────────────────────────────────
                // Use the first instance's resolved param types in the lambda
                // signature; pybind11 will accept any Python args convertible
                // to that signature (string / list / etc., independent of
                // dtype since the runtime args don't depend on T).
                TemplateArgBindings const &first = group.front();
                std::vector<std::string>   dtype_values;
                dtype_values.reserve(group.size());
                for (auto const &b2 : group) {
                    dtype_values.push_back(dtype_value_in(b2));
                }
                std::string const default_dtype = pick_default_dtype(dtype_values);
                std::string const method_kind   = m.is_static ? "def_static" : "def";

                os << "        ." << method_kind << "(\n";
                os << "            \"" << py_inst << "\",\n";
                os << "            [](";
                if (!m.is_static) {
                    os << class_qual << " &self";
                }
                for (std::size_t i = 0; i < m.params.size(); ++i) {
                    if (!m.is_static || i != 0) {
                        os << ", ";
                    }
                    os << substitute_template_params(m.params[i].type, first) << " "
                       << (m.params[i].name.empty() ? std::string{"arg"} + std::to_string(i) : m.params[i].name);
                }
                if (!m.is_static || !m.params.empty()) {
                    os << ", ";
                }
                os << "std::string const &dtype) -> " << b.ns << "::object {\n";

                for (auto const &bindings : group) {
                    std::string const        dval    = dtype_value_in(bindings);
                    std::vector<std::string> aliases = dtype_aliases_for(dval);
                    os << "                if (";
                    for (std::size_t i = 0; i < aliases.size(); ++i) {
                        if (i != 0) {
                            os << " || ";
                        }
                        os << "dtype == \"" << aliases[i] << "\"";
                    }
                    os << ") {\n";
                    // Build the explicit method-template instantiation by
                    // substituting bindings into the return type — that gives
                    // us the concrete tensor type, but we still need to pick
                    // the right overload via &self.template method<...>.
                    // We rely on the resolved return-type being unique per
                    // binding so static_cast<> on the method pointer
                    // disambiguates.
                    BoundMethod resolved = m;
                    resolved.return_type = substitute_template_params(m.return_type, bindings);
                    for (auto &p : resolved.params) {
                        p.type = substitute_template_params(p.type, bindings);
                    }
                    // Build the static_cast<>'d method-pointer expression; for
                    // non-static methods, the call site needs (self.*ptr)(args)
                    // so we wrap the cast accordingly.
                    std::string const cast_kind = m.is_static ? std::string{"*"} : (class_qual + "::*");
                    if (m.is_static) {
                        os << "                    return " << b.ns << "::cast(static_cast<" << resolved.return_type << " (*)(";
                    } else {
                        os << "                    return " << b.ns << "::cast((self.*static_cast<" << resolved.return_type << " ("
                           << class_qual << "::*)(";
                    }
                    for (std::size_t i = 0; i < resolved.params.size(); ++i) {
                        if (i != 0) {
                            os << ", ";
                        }
                        os << resolved.params[i].type;
                    }
                    os << ")";
                    if (m.is_const) {
                        os << " const";
                    }
                    os << ">(&" << class_qual << "::" << m.name << ")";
                    if (m.is_static) {
                        os << "(";
                    } else {
                        os << ")(";
                    }
                    bool first_call_arg = true;
                    if (m.is_static) {
                        first_call_arg = true;
                    }
                    for (std::size_t i = 0; i < m.params.size(); ++i) {
                        if (!first_call_arg) {
                            os << ", ";
                        }
                        first_call_arg = false;
                        os << (m.params[i].name.empty() ? std::string{"arg"} + std::to_string(i) : m.params[i].name);
                    }
                    os << ")";
                    // For reference-returning methods (e.g. ``T &declare_*``),
                    // hand the python side a non-owning view. Use ``reference``
                    // here (NOT ``reference_internal`` — that policy needs a
                    // direct-method binding to know its parent). The
                    // ``keep_alive<0, 1>`` on the .def below ties the returned
                    // tensor's lifetime to the parent class instance.
                    bool const returns_ref = resolved.return_type.find('&') != std::string::npos;
                    if (returns_ref) {
                        os << ", " << b.ns << "::return_value_policy::reference";
                    }
                    os << ");\n";
                    os << "                }\n";
                }
                os << "                throw " << b.ns << "::value_error(\"unsupported dtype: \" + dtype);\n";
                os << "            }";
                emit_param_args(os, m.params, b);
                os << ", " << b.ns << "::arg(\"dtype\") = \"" << default_dtype << "\"";
                // For reference-returning methods, tie the returned tensor's
                // lifetime to the bound class instance (lambda arg 1).
                if (m.return_type.find('&') != std::string::npos && !m.is_static) {
                    os << ", " << b.ns << "::keep_alive<0, 1>()";
                }
                emit_function_arg_modifiers(os, m, b);
                emit_doc_arg(os, m);
                os << ")\n";
            } else {
                // ── Plain per-binding emission ───────────────────────────
                for (auto const &bindings : group) {
                    BoundMethod resolved = m;
                    resolved.return_type = substitute_template_params(m.return_type, bindings);
                    for (auto &p : resolved.params) {
                        p.type = substitute_template_params(p.type, bindings);
                        if (p.default_value) {
                            p.default_value = substitute_template_params(*p.default_value, bindings);
                        }
                    }
                    std::string const method_kind = resolved.is_static ? "def_static" : "def";
                    std::string const cast_kind   = resolved.is_static ? std::string{"*"} : (class_qual + "::*");
                    os << "        ." << method_kind << "(\"" << py_inst << "\", static_cast<" << resolved.return_type << " (" << cast_kind
                       << ")(";
                    for (std::size_t i = 0; i < resolved.params.size(); ++i) {
                        if (i != 0) {
                            os << ", ";
                        }
                        os << resolved.params[i].type;
                    }
                    os << ")";
                    if (resolved.is_const) {
                        os << " const";
                    }
                    os << ">(&" << class_qual << "::" << resolved.name << ")";
                    emit_param_args(os, resolved.params, b);
                    emit_function_arg_modifiers(os, m, b);
                    emit_doc_arg(os, m);
                    os << ")\n";
                }
            }
        }
        return;
    }

    std::string const py = py_name_for(m);
    if (m.is_static) {
        os << "        .def_static(\"" << py << "\", ";
    } else {
        os << "        .def(\"" << py << "\", ";
    }
    os << method_pointer(class_qual, m);
    emit_param_args(os, m.params, b);
    emit_function_arg_modifiers(os, m, b);
    emit_doc_arg(os, m);
    os << ")\n";
}

// Mapping from a templated class's template parameter NAME ("rank") to
// the value used by the current instantiation ("2"). Empty for
// non-templated classes; used by EINSUMS_PYBIND_VARIADIC_FROM expansion
// to look up arity at per-instantiation emit time and by
// substitute_template_params() to resolve dependent types in method
// signatures.
using TemplateArgBindings = std::map<std::string, std::string>;

// Lexically substitute identifier-like tokens in a type string using the
// given bindings. Used to resolve dependent types in method/ctor parameter
// and return types when emitting a per-instantiation binding. For example,
// with bindings {"T": "float", "rank": "2"}, the string "T const &" becomes
// "float const &" and "Dim<rank>" becomes "Dim<2>". Matches whole-word
// occurrences only, so substituting "T" doesn't mangle "Tensor" or
// "TVector".
//
// Handles bindings that themselves contain non-identifier characters (the
// common case is ``Alloc → std::allocator<float>``) — substitution is
// single-pass over the input, so no risk of re-substituting tokens inside
// the replacement text.
// Trim ASCII whitespace from both ends of a view. Inline because adding
// llvm::StringRef::trim() would pull in another header for one use.
std::string trim(std::string s) {
    auto const ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!s.empty() && ws(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && ws(s.back())) {
        s.pop_back();
    }
    return s;
}

// Parse a comma-separated ``key = value`` payload (Plan C protocol
// directives carry their config this way). Whitespace around keys/values
// is trimmed. Values may contain template angle brackets (``Foo<T, U>``)
// — the splitter respects ``<>`` nesting so commas inside templates
// don't break key/value boundaries.
std::map<std::string, std::string> parse_kv_payload(std::string const &payload) {
    std::map<std::string, std::string> out;
    std::string                        current;
    int                                angle      = 0;
    auto                               flush_pair = [&](std::string const &piece) {
        std::string const eq_split = piece;
        std::size_t const eq       = eq_split.find('=');
        if (eq == std::string::npos) {
            return;
        }
        std::string const k = trim(eq_split.substr(0, eq));
        std::string const v = trim(eq_split.substr(eq + 1));
        if (!k.empty() && !v.empty()) {
            out[k] = v;
        }
    };
    for (char const c : payload) {
        if (c == '<') {
            ++angle;
        } else if (c == '>' && angle > 0) {
            --angle;
        }
        if (c == ',' && angle == 0) {
            flush_pair(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        flush_pair(current);
    }
    return out;
}

// Parse `instantiate_member` directive payloads into name → type bindings.
// Each directive arg is a single ``Name = Type`` assignment; multiple
// directives stack. The codegen reads these for ctors/methods that have
// member templates of their own (distinct from the enclosing class's
// template parameters), so pybind11's static_cast<>'d signature
// resolves to a concrete instantiation.
TemplateArgBindings collect_member_bindings(DirectiveList const &directives) {
    TemplateArgBindings out;
    DirectiveView const v(directives);
    for (auto const *d : v.all("instantiate_member")) {
        if (d->args.empty()) {
            continue;
        }
        std::string const &payload = d->args.front();
        auto const         eq      = payload.find('=');
        if (eq == std::string::npos) {
            continue; // malformed; skip rather than emit a broken binding
        }
        std::string const lhs = trim(payload.substr(0, eq));
        std::string const rhs = trim(payload.substr(eq + 1));
        if (!lhs.empty() && !rhs.empty()) {
            out[lhs] = rhs;
        }
    }
    return out;
}

// Parse one `instantiate_member_as` directive's tail. The tail is a
// comma-separated list of ``Name = Type`` assignments where Type may
// contain template angles (commas inside <> are not separators). Returns
// the bindings map; empty when no valid pairs were found.
TemplateArgBindings parse_member_as_bindings(std::string const &payload) {
    TemplateArgBindings            out;
    std::vector<std::string> const pieces = split_instantiation_args(payload);
    for (auto const &piece : pieces) {
        auto const eq = piece.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string const lhs = trim(piece.substr(0, eq));
        std::string const rhs = trim(piece.substr(eq + 1));
        if (!lhs.empty() && !rhs.empty()) {
            out[lhs] = rhs;
        }
    }
    return out;
}

std::string substitute_template_params(std::string const &type, TemplateArgBindings const &bindings) {
    if (bindings.empty()) {
        return type;
    }
    auto is_ident_start = [](char c) { return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_'; };
    auto is_ident_cont  = [](char c) { return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_'; };

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
            auto const        it  = bindings.find(tok);
            if (it != bindings.end()) {
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

// Emit one py::class_<> block, given the concrete C++ type to bind and the
// Python identifier to expose it as. The non-template path passes the
// class's own qualified name; the template path calls this once per
// instantiation with the substituted type and per-instantiation py_name.
//
// `bound_class_names` is the set of fully-qualified class names this run
// will bind. Only bases in that set are emitted on the py::class_<>
// declaration — pybind11 errors at import time on unknown bases, so
// silently filtering keeps the user from having to slap
// EINSUMS_PYBIND_NO_BASES on every class with internal-only inheritance.
void emit_class_body(llvm::raw_string_ostream &os, BoundClass const &cls, std::string const &concrete_type, std::string const &py_name,
                     std::unordered_set<std::string> const &bound_class_names, TemplateArgBindings const &arg_bindings, Backend const &b,
                     std::string const &mod_var) {
    DirectiveView const v(cls.directives);

    std::string holder;
    if (auto const *d = v.first("holder"); d != nullptr && !d->args.empty()) {
        holder = d->args.front() + "<" + concrete_type + ">";
    }

    os << "    " << b.ns << "::class_<" << concrete_type;
    if (!holder.empty()) {
        os << ", " << holder;
    }
    if (!v.has("no_bases")) {
        for (std::string const &base : cls.bases) {
            // Strip any template arguments so ``Foo<T>`` matches ``Foo``
            // in the bound set. Bases of templated classes show up here
            // as ``ns::Bar<...>`` while the bound entry is ``ns::Bar``.
            std::string base_key = base;
            if (auto angle = base_key.find('<'); angle != std::string::npos) {
                base_key = base_key.substr(0, angle);
            }
            if (bound_class_names.count(base_key) != 0U) {
                os << ", " << base;
            }
        }
    }
    os << ">(" << mod_var << ", \"" << py_name << "\"";
    if (v.has("dynamic_attr")) {
        os << ", " << b.ns << "::dynamic_attr()";
    }
    if (v.has("buffer_protocol") && b.supports_buffer) {
        os << ", " << b.ns << "::buffer_protocol()";
    }
    emit_doc_arg(os, cls);
    os << ")\n";

    // Constructors. Skip if the class is marked @nocopy / @nomove and the
    // ctor would be the implicit copy/move; the simple heuristic here is
    // arity + first-param-type-equals-class-ref. Phase 8 polishes this.
    bool const skip_copy = v.has("nocopy");
    bool const skip_move = v.has("nomove");
    for (auto const &ctor : cls.ctors) {
        if (is_hidden(ctor) || ctor.is_deleted) {
            continue;
        }
        if (skip_copy && ctor.params.size() == 1 && ctor.params.front().type.find(cls.qualified_name) != std::string::npos &&
            ctor.params.front().type.find("const ") != std::string::npos) {
            continue;
        }
        if (skip_move && ctor.params.size() == 1 && ctor.params.front().type.find(cls.qualified_name) != std::string::npos &&
            ctor.params.front().type.find("&&") != std::string::npos) {
            continue;
        }

        // EINSUMS_PYBIND_VARIADIC_FROM expansion: replace the (assumed-last)
        // pack parameter with N copies of the named element type, where N
        // is the value of the named template parameter for this
        // instantiation. Skip the ctor entirely if the binding info isn't
        // available for this class — e.g. non-template instantiation pass.
        std::vector<BoundParam> expanded_params = ctor.params;
        // Resolve any references to the class template parameters AND any
        // member-template parameters pinned via INSTANTIATE_MEMBER. The
        // member bindings are layered on top of the class bindings so they
        // can shadow if names happen to collide.
        TemplateArgBindings ctor_bindings = arg_bindings;
        for (auto const &kv : collect_member_bindings(ctor.directives)) {
            ctor_bindings[kv.first] = kv.second;
        }
        for (auto &p : expanded_params) {
            p.type = substitute_template_params(p.type, ctor_bindings);
            if (p.default_value) {
                p.default_value = substitute_template_params(*p.default_value, ctor_bindings);
            }
        }
        if (ctor.has_variadic_pack) {
            auto const it = arg_bindings.find(ctor.variadic_from_param);
            if (it == arg_bindings.end()) {
                os << "        // skipped variadic ctor: template parameter '" << ctor.variadic_from_param
                   << "' not bound for this instantiation\n";
                continue;
            }
            std::size_t arity = 0;
            try {
                arity = std::stoul(it->second);
            } catch (...) {
                os << "        // skipped variadic ctor: template parameter '" << ctor.variadic_from_param << "' value '" << it->second
                   << "' is not an integer\n";
                continue;
            }
            // Drop the trailing pack param and append N typed slots.
            if (!expanded_params.empty()) {
                expanded_params.pop_back();
            }
            for (std::size_t i = 0; i < arity; ++i) {
                BoundParam p;
                p.type = ctor.variadic_element_type;
                p.name = "dim_" + std::to_string(i);
                expanded_params.push_back(std::move(p));
            }
        }

        os << "        .def(" << b.ns << "::init<";
        for (std::size_t i = 0; i < expanded_params.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << expanded_params[i].type;
        }
        os << ">()";
        emit_param_args(os, expanded_params, b);
        emit_function_arg_modifiers(os, ctor, b);
        emit_doc_arg(os, ctor);
        os << ")\n";
    }

    // Methods (including operators / properties). For templated classes
    // the method pointers refer to the concrete instantiation, and any
    // references to the class template parameters AND any member-template
    // parameters pinned via INSTANTIATE_MEMBER are resolved here so the
    // static_cast<> in the emitted method-pointer expression compiles.
    auto methods_resolved = cls.methods;
    for (auto &m : methods_resolved) {
        TemplateArgBindings method_bindings = arg_bindings;
        for (auto const &kv : collect_member_bindings(m.directives)) {
            method_bindings[kv.first] = kv.second;
        }
        m.return_type = substitute_template_params(m.return_type, method_bindings);
        for (auto &p : m.params) {
            p.type = substitute_template_params(p.type, method_bindings);
            if (p.default_value) {
                p.default_value = substitute_template_params(*p.default_value, method_bindings);
            }
        }
    }
    for (auto const &m : methods_resolved) {
        emit_method(os, concrete_type, m, methods_resolved, b);
    }

    // Public fields. Resolve any class-template-parameter references in
    // the field type for the same reason as methods above.
    for (auto const &f : cls.fields) {
        BoundField fr = f;
        fr.type       = substitute_template_params(fr.type, arg_bindings);
        emit_field(os, concrete_type, fr);
    }

    // Buffer protocol implementations.
    //
    //   @buffer_from <fn>            — legacy: emits .def_buffer that calls
    //                                   the named function (which must
    //                                   itself return py::buffer_info — i.e.
    //                                   user code touches pybind11 directly).
    //   @buffer_protocol_std(...)    — Plan C: codegen synthesizes the
    //                                   buffer-info builder from the named
    //                                   pure-C++ accessors. User code is
    //                                   pybind11-free.
    //
    // Both are pybind11-only today; nanobind's tensor protocol uses
    // nb::ndarray<> and would be added as a parallel branch on b.name.
    if (b.supports_buffer) {
        if (auto const *bf = v.first("buffer_from"); bf != nullptr && !bf->args.empty()) {
            os << "        .def_buffer([](" << concrete_type << " &self) { return " << bf->args.front() << "(self); })\n";
        }
        if (auto const *bps = v.first("buffer_protocol_std"); bps != nullptr && !bps->args.empty()) {
            auto const fields       = parse_kv_payload(bps->args.front());
            auto const data_fn      = fields.find("data");
            auto const rank_fn      = fields.find("rank");
            auto const dim_fn       = fields.find("dim");
            auto const stride_fn    = fields.find("stride");
            auto const element_type = fields.find("element_type");
            auto const end          = fields.end();
            if (data_fn == end || rank_fn == end || dim_fn == end || stride_fn == end || element_type == end) {
                os << "        // skipped buffer_protocol_std on " << concrete_type
                   << ": missing one of {data, rank, dim, stride, element_type}\n";
            } else {
                std::string const elem = substitute_template_params(element_type->second, arg_bindings);
                os << "        .def_buffer([](" << concrete_type << " &self) {\n"
                   << "            std::vector<py::ssize_t> _shape, _strides;\n"
                   << "            std::size_t const _r = self." << rank_fn->second << "();\n"
                   << "            _shape.reserve(_r); _strides.reserve(_r);\n"
                   << "            for (std::size_t _i = 0; _i < _r; ++_i) {\n"
                   << "                _shape.push_back(static_cast<py::ssize_t>(self." << dim_fn->second << "(_i)));\n"
                   << "                _strides.push_back(static_cast<py::ssize_t>(self." << stride_fn->second << "(_i) * sizeof(" << elem
                   << ")));\n"
                   << "            }\n"
                   << "            return py::buffer_info(self." << data_fn->second << "(),\n"
                   << "                                   static_cast<py::ssize_t>(sizeof(" << elem << ")),\n"
                   << "                                   py::format_descriptor<" << elem << ">::format(),\n"
                   << "                                   static_cast<py::ssize_t>(_r), std::move(_shape), std::move(_strides));\n"
                   << "        })\n";
            }
        }
    }

    // Index protocol (Plan C, scalar pass). User provides at_element /
    // set_element returning/taking a scalar at a fully-specified
    // std::vector<int64_t> index. The codegen synthesizes
    // __getitem__/__setitem__ overloads for py::int_ and py::tuple,
    // handles negative-index normalization and bounds-checking, and
    // raises py::index_error / py::type_error appropriately.
    //
    // Slice/view-returning paths and bulk buffer assign are deferred to
    // a follow-up that lands alongside RuntimeTensorView's binding.
    if (auto const *ip = v.first("index_protocol_std"); ip != nullptr && !ip->args.empty()) {
        auto const fields       = parse_kv_payload(ip->args.front());
        auto const element_type = fields.find("element_type");
        auto const rank_fn      = fields.find("rank");
        auto const dim_fn       = fields.find("dim");
        auto const at_element   = fields.find("at_element");
        auto const set_element  = fields.find("set_element");
        auto const at_view      = fields.find("at_view");   // optional
        auto const view_type    = fields.find("view_type"); // optional, required if at_view present
        auto const end          = fields.end();
        if (element_type == end || rank_fn == end || dim_fn == end || at_element == end || set_element == end) {
            os << "        // skipped index_protocol_std on " << concrete_type
               << ": missing one of {element_type, rank, dim, at_element, set_element}\n";
        } else if (b.name == "pybind11") {
            std::string const elem     = substitute_template_params(element_type->second, arg_bindings);
            std::string const rfn      = rank_fn->second;
            std::string const dfn      = dim_fn->second;
            std::string const at_fn    = at_element->second;
            std::string const set_fn   = set_element->second;
            bool const        has_view = at_view != end && view_type != end;
            std::string const av_fn    = has_view ? at_view->second : std::string{};
            std::string const vty      = has_view ? substitute_template_params(view_type->second, arg_bindings) : std::string{};

            // __getitem__(int) — single integer index. Returns scalar
            // when the tensor is rank-1; otherwise (when at_view is
            // available) builds a sub-view collapsing axis 0.
            os << "        .def(\"__getitem__\", [](" << concrete_type << " &self, py::ssize_t idx) -> py::object {\n"
               << "            std::size_t const _r = self." << rfn << "();\n"
               << "            if (_r == 0) throw py::index_error(\"cannot index rank-0 tensor\");\n"
               << "            std::int64_t _d = static_cast<std::int64_t>(self." << dfn << "(0));\n"
               << "            std::int64_t _i = idx < 0 ? idx + _d : idx;\n"
               << "            if (_i < 0 || _i >= _d) throw py::index_error();\n"
               << "            if (_r == 1) return py::cast(self." << at_fn << "(std::vector<std::int64_t>{_i}));\n";
            if (has_view) {
                os << "            std::vector<einsums::SliceSpec> _spec(_r);\n"
                   << "            _spec[0].kind = einsums::SliceSpec::Kind::Index; _spec[0].index = _i;\n"
                   << "            for (std::size_t _k = 1; _k < _r; ++_k) _spec[_k].kind = einsums::SliceSpec::Kind::Full;\n"
                   << "            return py::cast(self." << av_fn << "(_spec));\n";
            } else {
                os << "            throw py::type_error(\"partial indexing requires at_view; not bound\");\n";
            }
            os << "        })\n";

            // __getitem__(ellipsis) — bare `t[...]` returns a full-tensor view.
            if (has_view) {
                os << "        .def(\"__getitem__\", [](" << concrete_type << " &self, py::ellipsis) -> " << vty << " {\n"
                   << "            std::size_t const _r = self." << rfn << "();\n"
                   << "            std::vector<einsums::SliceSpec> _spec(_r);\n"
                   << "            for (std::size_t _k = 0; _k < _r; ++_k) _spec[_k].kind = einsums::SliceSpec::Kind::Full;\n"
                   << "            return self." << av_fn << "(_spec);\n"
                   << "        })\n";
            }

            // __getitem__(slice) — sub-view along axis 0, full on other axes.
            if (has_view) {
                os << "        .def(\"__getitem__\", [](" << concrete_type << " &self, py::slice key) -> " << vty << " {\n"
                   << "            std::size_t const _r = self." << rfn << "();\n"
                   << "            if (_r == 0) throw py::index_error(\"cannot index rank-0 tensor\");\n"
                   << "            py::ssize_t _start = 0, _stop = 0, _step = 0, _len = 0;\n"
                   << "            if (!key.compute(static_cast<py::ssize_t>(self." << dfn << "(0)), &_start, &_stop, &_step, &_len))\n"
                   << "                throw py::error_already_set();\n"
                   << "            std::vector<einsums::SliceSpec> _spec(_r);\n"
                   << "            _spec[0].kind = einsums::SliceSpec::Kind::Range;\n"
                   << "            _spec[0].start = _start; _spec[0].stop = _stop; _spec[0].step = _step;\n"
                   << "            for (std::size_t _k = 1; _k < _r; ++_k) _spec[_k].kind = einsums::SliceSpec::Kind::Full;\n"
                   << "            return self." << av_fn << "(_spec);\n"
                   << "        })\n";
            }

            // __getitem__(tuple) — full int → scalar; mixed/partial → view.
            os << "        .def(\"__getitem__\", [](" << concrete_type << " &self, py::tuple key) -> py::object {\n"
               << "            std::size_t const _r = self." << rfn << "();\n"
               << "            bool _all_int = true;\n"
               << "            for (auto const &_it : key) {\n"
               << "                if (!py::isinstance<py::int_>(_it)) { _all_int = false; break; }\n"
               << "            }\n"
               << "            if (_all_int && key.size() == _r) {\n"
               << "                std::vector<std::int64_t> _idx; _idx.reserve(_r);\n"
               << "                for (std::size_t _k = 0; _k < _r; ++_k) {\n"
               << "                    std::int64_t _v = py::cast<std::int64_t>(key[_k]);\n"
               << "                    std::int64_t _d = static_cast<std::int64_t>(self." << dfn << "(_k));\n"
               << "                    if (_v < 0) _v += _d;\n"
               << "                    if (_v < 0 || _v >= _d) throw py::index_error();\n"
               << "                    _idx.push_back(_v);\n"
               << "                }\n"
               << "                return py::cast(self." << at_fn << "(_idx));\n"
               << "            }\n";
            if (has_view) {
                // Mixed tuple: build SliceSpec vec, expand ellipsis, pad with Full.
                os << "            std::vector<einsums::SliceSpec> _spec; _spec.reserve(_r);\n"
                   << "            std::size_t _consumed = 0;\n"
                   << "            for (std::size_t _k = 0; _k < key.size(); ++_k) {\n"
                   << "                py::handle _it = key[_k];\n"
                   << "                if (py::isinstance<py::ellipsis>(_it)) {\n"
                   << "                    std::size_t _remaining_keys = key.size() - _k - 1;\n"
                   << "                    while (_spec.size() + _remaining_keys < _r) {\n"
                   << "                        einsums::SliceSpec _f; _f.kind = einsums::SliceSpec::Kind::Full; _spec.push_back(_f);\n"
                   << "                    }\n"
                   << "                } else if (py::isinstance<py::int_>(_it)) {\n"
                   << "                    std::int64_t _v = py::cast<std::int64_t>(_it);\n"
                   << "                    std::int64_t _d = static_cast<std::int64_t>(self." << dfn << "(_spec.size()));\n"
                   << "                    if (_v < 0) _v += _d;\n"
                   << "                    if (_v < 0 || _v >= _d) throw py::index_error();\n"
                   << "                    einsums::SliceSpec _s; _s.kind = einsums::SliceSpec::Kind::Index; _s.index = _v; "
                      "_spec.push_back(_s);\n"
                   << "                } else if (py::isinstance<py::slice>(_it)) {\n"
                   << "                    py::slice _sl = py::cast<py::slice>(_it);\n"
                   << "                    py::ssize_t _start, _stop, _step, _len;\n"
                   << "                    if (!_sl.compute(static_cast<py::ssize_t>(self." << dfn << "(_spec.size())),\n"
                   << "                                     &_start, &_stop, &_step, &_len))\n"
                   << "                        throw py::error_already_set();\n"
                   << "                    einsums::SliceSpec _s; _s.kind = einsums::SliceSpec::Kind::Range;\n"
                   << "                    _s.start = _start; _s.stop = _stop; _s.step = _step; _spec.push_back(_s);\n"
                   << "                } else {\n"
                   << "                    throw py::type_error(\"unsupported index element\");\n"
                   << "                }\n"
                   << "                ++_consumed;\n"
                   << "            }\n"
                   << "            (void)_consumed;\n"
                   << "            while (_spec.size() < _r) { einsums::SliceSpec _f; _f.kind = einsums::SliceSpec::Kind::Full; "
                      "_spec.push_back(_f); }\n"
                   << "            return py::cast(self." << av_fn << "(_spec));\n";
            } else {
                os << "            throw py::type_error(\"partial/slice indexing requires at_view; not bound\");\n";
            }
            os << "        })\n";

            // __setitem__(int, T) — write scalar to rank-1 element only
            // (broadcast-set across a sub-view is deferred).
            os << "        .def(\"__setitem__\", [](" << concrete_type << " &self, py::ssize_t idx, " << elem << " value) {\n"
               << "            std::size_t const _r = self." << rfn << "();\n"
               << "            if (_r != 1) throw py::type_error(\"int-only assignment supported on rank-1 tensors here; "
               << "broadcast-set across a sub-view is not yet bound\");\n"
               << "            std::int64_t _d = static_cast<std::int64_t>(self." << dfn << "(0));\n"
               << "            std::int64_t _i = idx < 0 ? idx + _d : idx;\n"
               << "            if (_i < 0 || _i >= _d) throw py::index_error();\n"
               << "            self." << set_fn << "(std::vector<std::int64_t>{_i}, value);\n"
               << "        })\n";

            // __setitem__(tuple, T) — write scalar at full integer index;
            // partial / slice writes are deferred.
            os << "        .def(\"__setitem__\", [](" << concrete_type << " &self, py::tuple key, " << elem << " value) {\n"
               << "            std::size_t const _r = self." << rfn << "();\n"
               << "            if (key.size() != _r) throw py::type_error(\"partial assignment not yet bound\");\n"
               << "            std::vector<std::int64_t> _idx; _idx.reserve(_r);\n"
               << "            for (std::size_t _i = 0; _i < _r; ++_i) {\n"
               << "                if (!py::isinstance<py::int_>(key[_i]))\n"
               << "                    throw py::type_error(\"slice/ellipsis assignment not yet bound\");\n"
               << "                std::int64_t _v = py::cast<std::int64_t>(key[_i]);\n"
               << "                std::int64_t _d = static_cast<std::int64_t>(self." << dfn << "(_i));\n"
               << "                if (_v < 0) _v += _d;\n"
               << "                if (_v < 0 || _v >= _d) throw py::index_error();\n"
               << "                _idx.push_back(_v);\n"
               << "            }\n"
               << "            self." << set_fn << "(_idx, value);\n"
               << "        })\n";
        } else {
            os << "        // index_protocol_std: nanobind backend not yet implemented\n";
        }
    }

    // Iterator protocol (Plan C). User provides STL-style begin()/end()
    // member functions; codegen emits __iter__ via py::make_iterator with
    // a keep_alive policy so the iterator can outlive the
    // method call but not the parent. Backend-specific bits stay inside
    // the codegen lambda; user code is pure C++.
    if (auto const *it = v.first("iterator_std"); it != nullptr && !it->args.empty()) {
        auto const fields = parse_kv_payload(it->args.front());
        auto const beg    = fields.find("begin");
        auto const ed     = fields.find("end");
        auto const end_it = fields.end();
        if (beg == end_it || ed == end_it) {
            os << "        // skipped iterator_std on " << concrete_type << ": missing one of {begin, end}\n";
        } else if (b.name == "pybind11") {
            os << "        .def(\"__iter__\", [](" << concrete_type << " &self) {\n"
               << "            return py::make_iterator(self." << beg->second << "(), self." << ed->second << "());\n"
               << "        }, py::keep_alive<0, 1>())\n";
        } else {
            // Nanobind's nb::make_iterator API is shaped slightly
            // differently (taking a scope arg first); a parallel branch
            // would land here when nanobind support is wired up.
            os << "        // iterator_std: nanobind backend not yet implemented\n";
        }
    }

    os << "        ;\n\n";

    // <ns>::implicitly_convertible<Source, Class>() — emitted after the
    // class_<> chain since it's a free call against the bound type.
    // Nanobind has the same API name.
    for (auto const *d : v.all("implicit_from")) {
        if (!d->args.empty()) {
            os << "    " << b.ns << "::implicitly_convertible<" << d->args.front() << ", " << concrete_type << ">();\n\n";
        }
    }
}

// Bind a class as a Python exception type via py::register_exception<>.
// Pybind11-only — nanobind's exception API is shaped differently and is
// out of scope for the @exception directive today.
void emit_exception(llvm::raw_string_ostream &os, BoundClass const &cls, Backend const &b, std::string const &mod_var) {
    if (b.name != "pybind11") {
        os << "    // skipped exception binding for " << cls.qualified_name
           << ": @exception directive requires pybind11 target (--target pybind11)\n";
        return;
    }
    std::string const py = py_name_for(cls);
    os << "    " << b.ns << "::register_exception<" << cls.qualified_name << ">(" << mod_var << ", \"" << py << "\");\n";
}

void emit_class(llvm::raw_string_ostream &os, BoundClass const &cls, std::unordered_set<std::string> const &bound_class_names,
                Backend const &b, std::string const &mod_var) {
    if (is_hidden(cls) || cls.is_external) {
        // External annotated classes are recorded only for cross-module
        // name resolution in the .pyi emitter. Their bindings live in
        // their owning module's TU; emitting them here would cause
        // duplicate-symbol registration at import time.
        return;
    }

    // Exception classes take a totally different code path —
    // register_exception<>, no class_<> declaration.
    if (DirectiveView(cls.directives).has("exception")) {
        emit_exception(os, cls, b, mod_var);
        return;
    }

    if (cls.is_template) {
        if (cls.instantiations.empty()) {
            os << "    // skipped: templated class " << cls.qualified_name << " has no @instantiate / @instantiate_as directive\n";
            return;
        }
        for (auto const &inst : cls.instantiations) {
            std::string const concrete = cls.qualified_name + "<" + inst.type_args + ">";
            // Map template parameter NAME ("rank") -> instantiation value
            // ("2") for this instantiation. Used by VARIADIC_FROM expansion.
            TemplateArgBindings            bindings;
            std::vector<std::string> const split = split_instantiation_args(inst.type_args);
            for (std::size_t i = 0; i < cls.template_param_names.size() && i < split.size(); ++i) {
                bindings[cls.template_param_names[i]] = split[i];
            }
            emit_class_body(os, cls, concrete, inst.py_name, bound_class_names, bindings, b, mod_var);
        }
        return;
    }

    emit_class_body(os, cls, cls.qualified_name, py_name_for(cls), bound_class_names, {}, b, mod_var);
}

void emit_enum(llvm::raw_string_ostream &os, BoundEnum const &e, Backend const &b, std::string const &mod_var) {
    if (is_hidden(e)) {
        return;
    }
    std::string const py = py_name_for(e);
    os << "    " << b.ns << "::enum_<" << e.qualified_name << ">(" << mod_var << ", \"" << py << "\"";
    emit_doc_arg(os, e);
    os << ")\n";
    for (auto const &v : e.enumerators) {
        os << "        .value(\"" << v.name << "\", " << e.qualified_name << "::" << v.name;
        if (!v.doc.empty()) {
            os << ", " << as_cpp_doc(v.doc);
        }
        os << ")\n";
    }
    if (!e.is_scoped) {
        os << "        .export_values()\n";
    }
    os << "        ;\n\n";
}

// Walk the class tree and collect every "Class<args>" form requested by an
// @instantiate or @instantiate_as directive. The emitter writes a single
// ``template class Class<args>;`` per unique entry at TU scope so the
// linker has a definition to bind against.
// NOLINTNEXTLINE(misc-no-recursion)
void collect_explicit_instantiations(BoundClass const &cls, std::vector<std::string> &out) {
    if (cls.is_external) {
        return; // owning module already emitted the explicit instantiation
    }
    if (cls.is_template) {
        for (auto const &inst : cls.instantiations) {
            std::string entry = cls.qualified_name + "<" + inst.type_args + ">";
            if (std::find(out.begin(), out.end(), entry) == out.end()) {
                out.push_back(std::move(entry));
            }
        }
    }
    for (auto const &n : cls.nested_classes) {
        collect_explicit_instantiations(n, out);
    }
}

// Build the resolved C++ argument signature ("(arg0_t, arg1_t, ...)") for a
// single instantiation of a templated free function. The dispatcher path
// uses this to detect overload groups whose only difference is return type.
std::string resolved_arg_signature(BoundFunction const &f, BoundInstantiation const &inst) {
    TemplateArgBindings            bindings;
    std::vector<std::string> const split = split_instantiation_args(inst.type_args);
    for (std::size_t i = 0; i < f.template_param_names.size() && i < split.size(); ++i) {
        bindings[f.template_param_names[i]] = split[i];
    }
    std::string sig = "(";
    for (std::size_t i = 0; i < f.params.size(); ++i) {
        if (i != 0) {
            sig += ',';
        }
        sig += substitute_template_params(f.params[i].type, bindings);
    }
    sig += ')';
    return sig;
}

// Emit a single overload of a templated free function via static_cast<>.
void emit_function_overload(llvm::raw_string_ostream &os, BoundFunction const &f, BoundInstantiation const &inst, Backend const &b,
                            std::string const &mod_var) {
    TemplateArgBindings            bindings;
    std::vector<std::string> const split = split_instantiation_args(inst.type_args);
    for (std::size_t i = 0; i < f.template_param_names.size() && i < split.size(); ++i) {
        bindings[f.template_param_names[i]] = split[i];
    }
    std::string const ret_resolved = substitute_template_params(f.return_type, bindings);

    os << "    " << mod_var << ".def(\"" << inst.py_name << "\", static_cast<" << ret_resolved << " (*)(";
    for (std::size_t i = 0; i < f.params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << substitute_template_params(f.params[i].type, bindings);
    }
    os << ")>(&" << f.qualified_name << "<" << inst.type_args << ">)";

    // Substitute template parameters in default values too, e.g. a default
    // ``typename AType::ValueType{0}`` becomes ``float{0}`` for a float
    // instantiation. The binding TU has the resolved types in scope, not
    // the function-template parameter names.
    std::vector<BoundParam> resolved_params = f.params;
    for (auto &p : resolved_params) {
        if (p.default_value) {
            p.default_value = substitute_template_params(*p.default_value, bindings);
        }
    }
    emit_param_args(os, resolved_params, b);
    emit_function_arg_modifiers(os, f, b);
    emit_doc_arg(os, f);
    os << ");\n";
}

// Emit a Python-level dispatcher for a sub-group of instantiations that
// differ only in their leading bool template parameters (declared via
// EINSUMS_PYBIND_TEMPLATE_KWARGS). The lambda takes the function's normal
// runtime args plus one bool kwarg per declared template-kwarg name, and
// branches into the matching ``f<TA, TB, ...>`` call. All members of
// ``group`` must share the same non-bool template-arg tail, so their
// resolved C++ parameter types are identical and the lambda commits to a
// single concrete signature.
void emit_template_kwargs_dispatcher(llvm::raw_string_ostream &os, BoundFunction const &f,
                                     std::vector<BoundInstantiation const *> const &group, Backend const &b, std::string const &mod_var) {
    BoundInstantiation const      &first = *group.front();
    TemplateArgBindings            bindings;
    std::vector<std::string> const split = split_instantiation_args(first.type_args);
    for (std::size_t i = 0; i < f.template_param_names.size() && i < split.size(); ++i) {
        bindings[f.template_param_names[i]] = split[i];
    }
    std::size_t const n            = f.template_kwargs.size();
    std::string const ret_resolved = substitute_template_params(f.return_type, bindings);
    bool const        is_void      = ret_resolved == "void";

    os << "    " << mod_var << ".def(\n";
    os << "        \"" << first.py_name << "\",\n";
    os << "        []("; // dispatcher lambda
    for (std::size_t i = 0; i < f.params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << substitute_template_params(f.params[i].type, bindings) << " "
           << (f.params[i].name.empty() ? std::string{"arg"} : f.params[i].name);
    }
    if (!f.params.empty()) {
        os << ", ";
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (k != 0) {
            os << ", ";
        }
        os << "bool " << f.template_kwargs[k];
    }
    if (is_void) {
        os << ") -> void {\n";
    } else {
        os << ") -> " << ret_resolved << " {\n";
    }

    auto emit_call = [&](BoundInstantiation const *inst, std::string const &indent) {
        os << indent;
        if (!is_void) {
            os << "return ";
        }
        os << f.qualified_name << "<" << inst->type_args << ">(";
        for (std::size_t i = 0; i < f.params.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << (f.params[i].name.empty() ? std::string{"arg"} : f.params[i].name);
        }
        os << ");\n";
        if (is_void) {
            os << indent << "return;\n";
        }
    };

    // Iterate group in insertion order = lexicographic order of the bool
    // mask (visitor expansion guarantees this).
    for (std::size_t mask = 0; mask < group.size(); ++mask) {
        os << "            if (";
        for (std::size_t k = 0; k < n; ++k) {
            if (k != 0) {
                os << " && ";
            }
            bool const v = ((mask >> (n - 1 - k)) & 1U) != 0U;
            if (!v) {
                os << "!";
            }
            os << f.template_kwargs[k];
        }
        os << ") {\n";
        emit_call(group[mask], "                ");
        os << "            }\n";
    }
    os << "            throw " << b.ns << "::value_error(\"unreachable: invalid bool combination\");\n";
    os << "        }";

    // Resolve default-value template params to concrete types so the
    // emitted py::arg("x") = ... compiles in the binding TU. Mirrors the
    // logic in emit_function_overload.
    std::vector<BoundParam> resolved_params = f.params;
    for (auto &p : resolved_params) {
        if (p.default_value) {
            p.default_value = substitute_template_params(*p.default_value, bindings);
        }
    }
    emit_param_args(os, resolved_params, b);
    // kw_only() applies to every arg that follows, so emit once.
    os << ", " << b.ns << "::kw_only()";
    for (std::size_t k = 0; k < n; ++k) {
        os << ", " << b.ns << "::arg(\"" << f.template_kwargs[k] << "\") = false";
    }
    emit_function_arg_modifiers(os, f, b);
    emit_doc_arg(os, f);
    os << ");\n";
}

// Emit a Python-level dispatcher for an overload group whose members share
// an argument signature and differ only by return type. The first
// instantiation in the group provides the default dtype.
void emit_dtype_dispatcher(llvm::raw_string_ostream &os, BoundFunction const &f, std::vector<BoundInstantiation const *> const &group,
                           Backend const &b, std::string const &mod_var) {
    BoundInstantiation const &first = *group.front();
    std::vector<std::string>  dtype_values;
    dtype_values.reserve(group.size());
    for (auto const *inst : group) {
        dtype_values.push_back(split_instantiation_args(inst->type_args).front());
    }
    std::string const default_dtype = pick_default_dtype(dtype_values);

    os << "    " << mod_var << ".def(\n";
    os << "        \"" << first.py_name << "\",\n";
    os << "        []("; // dispatcher lambda
    for (std::size_t i = 0; i < f.params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        // Use first instantiation's resolved param types — they're all
        // identical across the group by construction.
        TemplateArgBindings            b0;
        std::vector<std::string> const s0 = split_instantiation_args(first.type_args);
        for (std::size_t k = 0; k < f.template_param_names.size() && k < s0.size(); ++k) {
            b0[f.template_param_names[k]] = s0[k];
        }
        os << substitute_template_params(f.params[i].type, b0) << " " << (f.params[i].name.empty() ? std::string{"arg"} : f.params[i].name);
    }
    if (!f.params.empty()) {
        os << ", ";
    }
    os << "std::string const &dtype) -> " << b.ns << "::object {\n";

    for (auto const *inst_ptr : group) {
        std::string const              first_arg = split_instantiation_args(inst_ptr->type_args).front();
        std::vector<std::string> const aliases   = dtype_aliases_for(first_arg);
        os << "            if (";
        for (std::size_t i = 0; i < aliases.size(); ++i) {
            if (i != 0) {
                os << " || ";
            }
            os << "dtype == \"" << aliases[i] << "\"";
        }
        os << ") {\n";
        os << "                return " << b.ns << "::cast(" << f.qualified_name << "<" << inst_ptr->type_args << ">(";
        for (std::size_t i = 0; i < f.params.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << (f.params[i].name.empty() ? std::string{"arg"} : f.params[i].name);
        }
        os << "));\n";
        os << "            }\n";
    }
    os << "            throw " << b.ns << "::value_error(\"unsupported dtype: \" + dtype);\n";
    os << "        }";
    emit_param_args(os, f.params, b);
    os << ", " << b.ns << "::arg(\"dtype\") = \"" << default_dtype << "\"";
    emit_function_arg_modifiers(os, f, b);
    emit_doc_arg(os, f);
    os << ");\n";
}

void emit_function(llvm::raw_string_ostream &os, BoundFunction const &f, Backend const &b, std::string const &mod_var) {
    if (is_hidden(f)) {
        return;
    }
    if (f.is_template && f.instantiations.empty()) {
        os << "    // skipped: templated free function " << f.qualified_name << " has no @instantiate_as directive\n";
        return;
    }

    // Walk the precomputed PythonOverload view (PythonOverloads.cpp) and
    // dispatch each entry to the matching emit_* helper. This file no
    // longer has to think about how raw instantiations group up — that
    // decision lives in compute_python_overloads().
    auto resolve = [&](std::vector<std::size_t> const &idxs) {
        std::vector<BoundInstantiation const *> out;
        out.reserve(idxs.size());
        for (std::size_t const idx : idxs) {
            out.push_back(&f.instantiations[idx]);
        }
        return out;
    };

    for (auto const &ov : f.python_overloads) {
        switch (ov.kind) {
        case PythonOverload::Kind::NonTemplate: {
            std::string const py = py_name_for(f);
            // Always wrap in a static_cast so non-templated free functions
            // that share a Python name (e.g. `cg::custom` overloaded by
            // tensor-vs-no-tensor signature) disambiguate cleanly. When
            // there is no overload the cast is a no-op for the compiler.
            os << "    " << mod_var << ".def(\"" << py << "\", static_cast<" << f.return_type << "(*)(";
            for (std::size_t i = 0; i < f.params.size(); ++i) {
                if (i > 0) {
                    os << ", ";
                }
                os << f.params[i].type;
            }
            os << ")>(&" << f.qualified_name << ")";
            emit_param_args(os, f.params, b);
            emit_function_arg_modifiers(os, f, b);
            emit_doc_arg(os, f);
            os << ");\n";
            break;
        }
        case PythonOverload::Kind::SingleInstantiation:
        case PythonOverload::Kind::OverloadSet: {
            for (auto const *inst : resolve(ov.instantiation_indices)) {
                emit_function_overload(os, f, *inst, b, mod_var);
            }
            break;
        }
        case PythonOverload::Kind::DtypeDispatcher: {
            emit_dtype_dispatcher(os, f, resolve(ov.instantiation_indices), b, mod_var);
            break;
        }
        case PythonOverload::Kind::TemplateKwargsDispatcher: {
            emit_template_kwargs_dispatcher(os, f, resolve(ov.instantiation_indices), b, mod_var);
            break;
        }
        }
    }
}

// Walk the full class tree to collect every nested enum (which must be
// bound after its parent class — pybind11 needs the class type to exist
// before py::enum_ on a nested type). Recursion is intentional for
// arbitrarily deep nested-class chains.
// NOLINTNEXTLINE(misc-no-recursion)
void collect_nested_enums(BoundClass const &cls, std::vector<BoundEnum const *> &out) {
    for (auto const &e : cls.nested_enums) {
        out.push_back(&e);
    }
    for (auto const &n : cls.nested_classes) {
        collect_nested_enums(n, out);
    }
}

// Walk the class tree and collect every nested CLASS so the emitter can
// produce ``py::class_<>`` blocks for them after the parent. Without
// this, methods returning a nested type fail to bind ("unregistered
// type") at import time.
// NOLINTNEXTLINE(misc-no-recursion)
void collect_nested_classes(BoundClass const &cls, std::vector<BoundClass const *> &out) {
    for (auto const &n : cls.nested_classes) {
        out.push_back(&n);
        collect_nested_classes(n, out);
    }
}

// ---- clang-format integration --------------------------------------------

std::string format_cpp(std::string const &source, llvm::StringRef path_hint) {
    auto style = clang::format::getStyle("file", path_hint, "LLVM", source);
    if (!style) {
        // Style lookup failed — emit unformatted rather than dropping output.
        llvm::consumeError(style.takeError());
        return source;
    }
    clang::tooling::Replacements const replacements =
        clang::format::reformat(*style, source, {{0, static_cast<unsigned>(source.size())}}, path_hint);
    auto formatted = clang::tooling::applyAllReplacements(source, replacements);
    if (!formatted) {
        llvm::consumeError(formatted.takeError());
        return source;
    }
    return *formatted;
}

} // namespace

std::string emit(Module const &module_, EmitOptions const &opts) {
    std::string              buffer;
    llvm::raw_string_ostream os(buffer);

    os << "// This file is generated by einsums-pybind. Do not edit.\n"
       << "//\n"
       << "// Source IR contains " << module_.classes.size() << " class(es), " << module_.functions.size() << " function(s), "
       << module_.enums.size() << " enum(s).\n\n";

    // Source headers the bindings refer to.
    for (auto const &inc : opts.source_includes) {
        os << "#include \"" << inc << "\"\n";
    }
    if (!opts.source_includes.empty()) {
        os << "\n";
    }

    Backend const b = make_backend(opts.target);
    for (std::string const &h : b.headers) {
        os << "#include " << h << "\n";
    }
    os << "\nnamespace " << b.ns << " = " << b.ns_full << ";\n\n";

    // We deliberately do NOT emit ``template class Class<args>;`` here.
    // Explicit instantiation in the codegen TU duplicates the one already
    // provided by the owning library (e.g. TensorDefs.cpp's instantiations
    // of GeneralRuntimeTensor<T, std::allocator<T>>) and forces the
    // codegen TU's instantiation to use the pybind11_add_module visibility
    // (hidden) which collides with EINSUMS_EXPORT (default) on the library
    // side. Implicit instantiation of just the bound methods produces weak
    // symbols that the linker resolves against the library's strong ones.
    // For instantiations not provided by any library — purely binding-only
    // types — the implicit weak symbols stand on their own.

    bool const use_register_form = !opts.register_function_name.empty();
    if (use_register_form) {
        os << "void " << opts.register_function_name << "(" << b.ns << "::module_ &m) {\n";
    } else {
        os << b.module_macro << "(" << opts.module_name << ", m) {\n";
    }

    // Build the set of bound class qualified names so emit_class_body can
    // filter cls.bases to only those that are themselves being bound.
    // Includes nested classes since they're emitted as top-level
    // py::class_<> too.
    std::unordered_set<std::string> bound_class_names;
    std::vector<BoundClass const *> all_nested_classes;
    for (auto const &c : module_.classes) {
        bound_class_names.insert(c.qualified_name);
        collect_nested_classes(c, all_nested_classes);
    }
    for (BoundClass const *n : all_nested_classes) {
        bound_class_names.insert(n->qualified_name);
    }

    // Collect every submodule path referenced by any annotated entity
    // (and its parents — "tensor.algebra" requires "tensor" too).
    // Emit def_submodule calls in dependency order (parent before child)
    // so each ``_sub_x = m.def_submodule("x")`` already has its parent
    // module variable defined.
    std::set<std::string> submodule_paths;
    auto                  add_path = [&](std::string const &p) {
        if (p.empty()) {
            return;
        }
        std::string accumulated;
        for (char const c : p) {
            if (c == '.') {
                submodule_paths.insert(accumulated);
            }
            accumulated += c;
        }
        submodule_paths.insert(accumulated);
    };
    for (auto const &c : module_.classes) {
        add_path(submodule_path_for(c));
    }
    for (BoundClass const *n : all_nested_classes) {
        add_path(submodule_path_for(*n));
    }
    for (auto const &e : module_.enums) {
        add_path(submodule_path_for(e));
    }
    for (auto const &f : module_.functions) {
        add_path(submodule_path_for(f));
    }
    if (!submodule_paths.empty()) {
        os << "    // Submodules declared via @module directives.\n";
        // ``std::set<std::string>`` orders lexicographically, so
        // "tensor" comes before "tensor.algebra" — parents first.
        for (std::string const &path : submodule_paths) {
            std::string parent_var = "m";
            std::string leaf       = path;
            if (auto dot = path.rfind('.'); dot != std::string::npos) {
                parent_var = submodule_var_name(path.substr(0, dot));
                leaf       = path.substr(dot + 1);
            }
            os << "    auto " << submodule_var_name(path) << " = " << parent_var << ".def_submodule(\"" << leaf << "\");\n";
        }
        os << "\n";
    }

    // Enums are emitted before classes so that any class method/ctor
    // that references an enum value as a default argument can resolve
    // the type at registration time. pybind11 raises
    // "could not convert default argument into a Python object" if the
    // enum binding hasn't run yet when the default is evaluated.
    for (auto const &e : module_.enums) {
        emit_enum(os, e, b, submodule_var_name(submodule_path_for(e)));
    }
    std::vector<BoundEnum const *> nested;
    for (auto const &c : module_.classes) {
        collect_nested_enums(c, nested);
    }
    for (BoundEnum const *e : nested) {
        emit_enum(os, *e, b, submodule_var_name(submodule_path_for(*e)));
    }

    for (auto const &c : module_.classes) {
        emit_class(os, c, bound_class_names, b, submodule_var_name(submodule_path_for(c)));
    }

    // Nested classes after their parents — pybind11 needs every type
    // referenced by a method signature to already be registered. They
    // inherit their parent's submodule placement unless they have their
    // own @module directive (the visitor records them with the parent's
    // qualified name as a prefix; their own directives apply directly).
    for (BoundClass const *n : all_nested_classes) {
        emit_class(os, *n, bound_class_names, b, submodule_var_name(submodule_path_for(*n)));
    }

    for (auto const &f : module_.functions) {
        emit_function(os, f, b, submodule_var_name(submodule_path_for(f)));
    }

    os << "}\n";

    return format_cpp(buffer, opts.source_path_for_format);
}

} // namespace einsums::pybind
