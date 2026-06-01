//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "DocsJson.hpp"

#include <string>

#include "DocComment.hpp"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace einsums::pybind {

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

// A std::optional<std::string> becomes a JSON string when present and JSON
// null when absent — renderers can switch on null without ambiguity
// against the empty string (which carries meaning, e.g. an empty doc).
// Resolved Python identifier for an entity: the argument of a `rename`
// directive, or the Python dunder of an `operator` directive (e.g.
// ``operator+`` bound as ``__add__``), else the C++ name. Without the
// `operator` case, annotated operators render with their C++ spelling
// (``operator+``) and read as missing Python dunders.
std::string resolved_py_name(BoundEntityCommon const &e) {
    for (auto const &d : e.directives) {
        if ((d.name == "rename" || d.name == "operator") && !d.args.empty()) {
            return d.args.front();
        }
    }
    return e.name;
}

// True when the entity carries a `hide` directive — bound in C++ but
// intentionally omitted from the Python-facing surface. The renderer
// skips these. Mirrors PyiEmitter's is_hidden.
bool is_hidden(BoundEntityCommon const &e) {
    for (auto const &d : e.directives) {
        if (d.name == "hide") {
            return true;
        }
    }
    return false;
}

Value json_doc_entries(std::vector<DocEntry> const &entries) {
    Array out;
    for (auto const &e : entries) {
        out.push_back(Object{
            {"name", e.name},
            {"description", e.description},
        });
    }
    return out;
}

// Parse a raw Doxygen doc body into the structured, reST-ready form the
// docs renderer consumes (brief/detail/params/tparams/returns/throws).
Value json_doc_structured(std::string const &raw) {
    DocComment const dc = parse_doc_comment(raw);
    return Object{
        {"brief", dc.brief},
        {"detail", dc.detail},
        {"params", json_doc_entries(dc.params)},
        {"tparams", json_doc_entries(dc.tparams)},
        {"returns", dc.returns},
        {"throws", json_doc_entries(dc.throws_)},
    };
}

Value opt_string(std::optional<std::string> const &s) {
    // NB: do not use a braced return here — `{*s}` / `{nullptr}` select
    // json::Value's Array constructor and wrap the scalar in `[ ... ]`.
    Value v = s ? Value(*s) : Value(nullptr);
    return v;
}

Value json_location(SourceLocation const &loc) {
    return Object{
        {"file", loc.file},
        {"line", static_cast<int64_t>(loc.line)},
        {"column", static_cast<int64_t>(loc.column)},
    };
}

Value json_directives(DirectiveList const &dirs) {
    Array out;
    for (auto const &d : dirs) {
        Array args;
        for (auto const &a : d.args) {
            args.push_back(a);
        }
        out.push_back(Object{
            {"name", d.name},
            {"args", std::move(args)},
        });
    }
    return out;
}

Value json_param(BoundParam const &p) {
    return Object{
        {"name", p.name},
        {"type", p.type},
        {"type_canonical", p.type_canonical},
        {"py_type", p.py_type},
        {"default", opt_string(p.default_value)},
        {"default_py", opt_string(p.default_value_py)},
    };
}

Value json_params(std::vector<BoundParam> const &params) {
    Array out;
    for (auto const &p : params) {
        out.push_back(json_param(p));
    }
    return out;
}

Value json_string_list(std::vector<std::string> const &items); // fwd

Value json_method(BoundMethod const &m) {
    return Object{
        {"name", m.name},
        {"py_name", resolved_py_name(m)},
        {"hidden", is_hidden(m)},
        {"qualified_name", m.qualified_name},
        {"doc", m.doc},
        {"doc_structured", json_doc_structured(m.doc)},
        {"location", json_location(m.location)},
        {"submodule", opt_string(m.submodule)},
        {"return_type", m.return_type},
        {"return_type_canonical", m.return_type_canonical},
        {"return_py_type", m.return_py_type},
        {"params", json_params(m.params)},
        {"is_const", m.is_const},
        {"is_static", m.is_static},
        {"is_virtual", m.is_virtual},
        {"is_pure_virtual", m.is_pure_virtual},
        {"is_constructor", m.is_constructor},
        {"is_destructor", m.is_destructor},
        {"is_operator", m.is_operator},
        {"is_deleted", m.is_deleted},
        {"template_params", json_string_list(m.template_param_names)},
        {"directives", json_directives(m.directives)},
    };
}

Value json_methods(std::vector<BoundMethod> const &methods) {
    Array out;
    for (auto const &m : methods) {
        out.push_back(json_method(m));
    }
    return out;
}

Value json_field(BoundField const &f) {
    return Object{
        {"name", f.name},
        {"py_name", resolved_py_name(f)},
        {"hidden", is_hidden(f)},
        {"qualified_name", f.qualified_name},
        {"doc", f.doc},
        {"location", json_location(f.location)},
        {"type", f.type},
        {"py_type", f.py_type},
        {"is_static", f.is_static},
        {"doc_structured", json_doc_structured(f.doc)},
        {"directives", json_directives(f.directives)},
    };
}

Value json_enum(BoundEnum const &e) {
    Array enumerators;
    for (auto const &v : e.enumerators) {
        enumerators.push_back(Object{
            {"name", v.name},
            {"value", v.value},
            {"doc", v.doc},
            {"doc_structured", json_doc_structured(v.doc)},
        });
    }
    return Object{
        {"name", e.name},
        {"py_name", resolved_py_name(e)},
        {"hidden", is_hidden(e)},
        {"qualified_name", e.qualified_name},
        {"doc", e.doc},
        {"doc_structured", json_doc_structured(e.doc)},
        {"location", json_location(e.location)},
        {"submodule", opt_string(e.submodule)},
        {"is_scoped", e.is_scoped},
        {"underlying_type", e.underlying_type},
        {"underlying_py_type", e.underlying_py_type},
        {"enumerators", std::move(enumerators)},
        {"directives", json_directives(e.directives)},
    };
}

Value json_property(BoundProperty const &p) {
    return Object{
        {"py_name", p.py_name},
        {"type", p.type},
        {"py_type", p.py_type},
        {"doc", p.doc},
        {"doc_structured", json_doc_structured(p.doc)},
        {"writable", p.has_setter},
    };
}

Value json_instantiation(BoundInstantiation const &inst) {
    return Object{
        {"py_name", inst.py_name},
        {"type_args", inst.type_args},
    };
}

Value json_instantiations(std::vector<BoundInstantiation> const &insts) {
    Array out;
    for (auto const &inst : insts) {
        out.push_back(json_instantiation(inst));
    }
    return out;
}

char const *overload_kind_name(PythonOverload::Kind kind) {
    switch (kind) {
    case PythonOverload::Kind::NonTemplate:
        return "non_template";
    case PythonOverload::Kind::SingleInstantiation:
        return "single_instantiation";
    case PythonOverload::Kind::DtypeDispatcher:
        return "dtype_dispatcher";
    case PythonOverload::Kind::TemplateKwargsDispatcher:
        return "template_kwargs_dispatcher";
    case PythonOverload::Kind::OverloadSet:
        return "overload_set";
    }
    return "unknown";
}

Value json_python_overloads(std::vector<PythonOverload> const &overloads) {
    Array out;
    for (auto const &o : overloads) {
        Array indices;
        for (auto i : o.instantiation_indices) {
            indices.push_back(static_cast<int64_t>(i));
        }
        Array dtypes;
        for (auto const &d : o.dtype_values) {
            dtypes.push_back(d);
        }
        Array kwargs;
        for (auto const &k : o.kwarg_names) {
            kwargs.push_back(k);
        }
        out.push_back(Object{
            {"kind", overload_kind_name(o.kind)},
            {"py_name", o.py_name},
            {"instantiation_indices", std::move(indices)},
            {"dtype_values", std::move(dtypes)},
            {"default_dtype", o.default_dtype},
            {"kwarg_names", std::move(kwargs)},
        });
    }
    return out;
}

Value json_string_list(std::vector<std::string> const &items) {
    Array out;
    for (auto const &s : items) {
        out.push_back(s);
    }
    return out;
}

bool class_has_directive(BoundClass const &c, char const *name) {
    for (auto const &d : c.directives) {
        if (d.name == name) {
            return true;
        }
    }
    return false;
}

// Mirror the Python dunder methods the C++ emitter SYNTHESIZES from a
// class's protocol directives (see Emitter.cpp: index_protocol_std →
// __getitem__/__setitem__, iterator_std → __iter__). These methods are
// real on the bound class but have no annotated C++ declaration, so they
// are absent from the IR; without this the generated docs would omit them.
//
// NB: this covers only the *codegen*-synthesized protocol dunders. The
// numpy-ergonomics arithmetic/__len__/copy dunders are monkey-patched in
// pure Python (libs/Einsums/Python/python/einsums/__init__.py) and are
// outside the C++ AST entirely — they need a Python-side doc path.
void add_protocol_dunders(BoundClass const &c, Array &methods) {
    auto py_param = [](char const *nm, char const *ty) -> Value {
        return Object{
            {"name", nm},
            {"type", ""},
            {"type_canonical", ""},
            {"py_type", ty},
            {"default", Value(nullptr)},
            {"default_py", Value(nullptr)},
        };
    };
    auto emit = [&](std::string const &dunder, Array params, char const *ret, std::string const &brief) {
        methods.push_back(Object{
            {"name", dunder},
            {"py_name", dunder},
            {"hidden", false},
            {"qualified_name", c.qualified_name + "::" + dunder},
            {"doc", brief},
            {"doc_structured",
             Object{{"brief", brief}, {"detail", ""}, {"params", Array{}}, {"tparams", Array{}}, {"returns", ""}, {"throws", Array{}}}},
            {"location", json_location(c.location)},
            {"submodule", opt_string(c.submodule)},
            {"return_type", ""},
            {"return_type_canonical", ""},
            {"return_py_type", ret},
            {"params", std::move(params)},
            {"is_const", false},
            {"is_static", false},
            {"is_virtual", false},
            {"is_pure_virtual", false},
            {"is_constructor", false},
            {"is_destructor", false},
            {"is_operator", false},
            {"is_deleted", false},
            {"directives", Array{}},
        });
    };

    if (class_has_directive(c, "index_protocol_std")) {
        Array get_params;
        get_params.push_back(py_param("key", "int | slice | tuple"));
        emit("__getitem__", std::move(get_params), "object",
             "Subscript access (NumPy-style). A full integer index returns a scalar; partial, "
             "slice, or ellipsis indices return a view. Synthesized from the index protocol.");
        Array set_params;
        set_params.push_back(py_param("key", "int | slice | tuple"));
        set_params.push_back(py_param("value", "object"));
        emit("__setitem__", std::move(set_params), "None", "Assign to a scalar element or sub-view. Synthesized from the index protocol.");
    }
    if (class_has_directive(c, "iterator_std")) {
        emit("__iter__", Array{}, "Iterator",
             "Iterate over the tensor's elements in storage order. Synthesized from the iterator protocol.");
    }
}

// Recursion is intended: nested classes are themselves BoundClass.
// NOLINTNEXTLINE(misc-no-recursion)
Value json_class(BoundClass const &c) {
    Array ctors;
    for (auto const &m : c.ctors) {
        ctors.push_back(json_method(m));
    }
    Array properties;
    for (auto const &p : c.properties) {
        properties.push_back(json_property(p));
    }
    Array fields;
    for (auto const &f : c.fields) {
        fields.push_back(json_field(f));
    }
    Array nested_enums;
    for (auto const &e : c.nested_enums) {
        nested_enums.push_back(json_enum(e));
    }
    Array nested_classes;
    for (auto const &n : c.nested_classes) {
        nested_classes.push_back(json_class(n));
    }
    Array methods;
    for (auto const &m : c.methods) {
        methods.push_back(json_method(m));
    }
    add_protocol_dunders(c, methods);
    return Object{
        {"name", c.name},
        {"py_name", resolved_py_name(c)},
        {"hidden", is_hidden(c)},
        {"qualified_name", c.qualified_name},
        {"doc", c.doc},
        {"doc_structured", json_doc_structured(c.doc)},
        {"location", json_location(c.location)},
        {"submodule", opt_string(c.submodule)},
        {"is_template", c.is_template},
        {"is_external", c.is_external},
        {"template_params", json_string_list(c.template_param_names)},
        {"bases", json_string_list(c.bases)},
        {"instantiations", json_instantiations(c.instantiations)},
        {"constructors", std::move(ctors)},
        {"methods", std::move(methods)},
        {"properties", std::move(properties)},
        {"fields", std::move(fields)},
        {"enums", std::move(nested_enums)},
        {"nested_classes", std::move(nested_classes)},
        {"directives", json_directives(c.directives)},
    };
}

Value json_function(BoundFunction const &f) {
    return Object{
        {"name", f.name},
        {"py_name", resolved_py_name(f)},
        {"hidden", is_hidden(f)},
        {"qualified_name", f.qualified_name},
        {"doc", f.doc},
        {"doc_structured", json_doc_structured(f.doc)},
        {"location", json_location(f.location)},
        {"submodule", opt_string(f.submodule)},
        {"return_type", f.return_type},
        {"return_type_canonical", f.return_type_canonical},
        {"return_py_type", f.return_py_type},
        {"params", json_params(f.params)},
        {"is_template", f.is_template},
        {"template_params", json_string_list(f.template_param_names)},
        {"template_kwargs", json_string_list(f.template_kwargs)},
        {"instantiations", json_instantiations(f.instantiations)},
        {"python_overloads", json_python_overloads(f.python_overloads)},
        {"directives", json_directives(f.directives)},
    };
}

} // namespace

std::string emit_docs_json(Module const &module_, std::string const &module_name) {
    Array classes;
    for (auto const &c : module_.classes) {
        classes.push_back(json_class(c));
    }
    Array functions;
    for (auto const &f : module_.functions) {
        functions.push_back(json_function(f));
    }
    Array enums;
    for (auto const &e : module_.enums) {
        enums.push_back(json_enum(e));
    }
    Array typedefs;
    for (auto const &t : module_.typedefs) {
        typedefs.push_back(Object{
            {"name", t.name},
            {"qualified_name", t.qualified_name},
            {"doc", t.doc},
            {"doc_structured", json_doc_structured(t.doc)},
            {"location", json_location(t.location)},
            {"underlying_type", t.underlying_type},
            {"template_params", json_string_list(t.template_param_names)},
        });
    }
    Array concepts;
    for (auto const &c : module_.concepts) {
        concepts.push_back(Object{
            {"name", c.name},
            {"qualified_name", c.qualified_name},
            {"doc", c.doc},
            {"doc_structured", json_doc_structured(c.doc)},
            {"location", json_location(c.location)},
            {"template_params", json_string_list(c.template_param_names)},
        });
    }

    Value root = Object{
        {"schema_version", k_docs_json_schema_version},
        {"module", module_name},
        {"classes", std::move(classes)},
        {"functions", std::move(functions)},
        {"enums", std::move(enums)},
        {"typedefs", std::move(typedefs)},
        {"concepts", std::move(concepts)},
    };

    std::string              buffer;
    llvm::raw_string_ostream os(buffer);
    os << llvm::formatv("{0:2}", root); // pretty-print with 2-space indent
    os << "\n";
    return buffer;
}

} // namespace einsums::pybind
