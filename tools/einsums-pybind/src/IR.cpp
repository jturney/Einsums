//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "IR.hpp"

#include <string>

#include "llvm/Support/raw_ostream.h"

namespace einsums::pybind {

namespace {

void indent(llvm::raw_string_ostream &os, int depth) {
    for (int i = 0; i < depth; ++i) {
        os << "  ";
    }
}

void dump_directives(llvm::raw_string_ostream &os, DirectiveList const &dirs, int depth) {
    for (auto const &d : dirs) {
        indent(os, depth);
        os << "@" << d.name;
        if (!d.args.empty()) {
            os << "(";
            for (std::size_t i = 0; i < d.args.size(); ++i) {
                if (i != 0) {
                    os << ", ";
                }
                os << d.args[i];
            }
            os << ")";
        }
        os << "\n";
    }
}

void dump_param(llvm::raw_string_ostream &os, BoundParam const &p) {
    os << p.type;
    if (!p.name.empty()) {
        os << " " << p.name;
    }
    if (p.default_value) {
        os << " = " << *p.default_value;
    }
}

void dump_param_py(llvm::raw_string_ostream &os, BoundParam const &p) {
    if (!p.name.empty()) {
        os << p.name << ": ";
    }
    os << p.py_type;
    if (p.default_value_py) {
        os << " = " << *p.default_value_py;
    }
}

void dump_method(llvm::raw_string_ostream &os, BoundMethod const &m, int depth) {
    indent(os, depth);
    os << "method " << m.name;
    if (m.is_constructor) {
        os << " [ctor]";
    }
    if (m.is_destructor) {
        os << " [dtor]";
    }
    if (m.is_operator) {
        os << " [op]";
    }
    if (m.is_static) {
        os << " [static]";
    }
    if (m.is_virtual) {
        os << " [virtual]";
    }
    if (m.is_pure_virtual) {
        os << " [pure]";
    }
    if (m.is_const) {
        os << " [const]";
    }
    if (m.is_deleted) {
        os << " [deleted]";
    }
    os << ": " << m.return_type << "(";
    for (std::size_t i = 0; i < m.params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        dump_param(os, m.params[i]);
    }
    os << ")\n";
    indent(os, depth + 1);
    os << "py: (";
    for (std::size_t i = 0; i < m.params.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        dump_param_py(os, m.params[i]);
    }
    os << ") -> " << (m.return_py_type.empty() ? std::string{"None"} : m.return_py_type) << "\n";
    dump_directives(os, m.directives, depth + 1);
    if (!m.doc.empty()) {
        indent(os, depth + 1);
        os << "doc: " << m.doc.size() << " chars\n";
    }
}

void dump_field(llvm::raw_string_ostream &os, BoundField const &f, int depth) {
    indent(os, depth);
    os << "field " << f.name;
    if (f.is_static) {
        os << " [static]";
    }
    os << ": " << f.type << "\n";
    indent(os, depth + 1);
    os << "py: " << f.py_type << "\n";
    dump_directives(os, f.directives, depth + 1);
}

void dump_enum(llvm::raw_string_ostream &os, BoundEnum const &e, int depth) {
    indent(os, depth);
    os << "enum " << (e.is_scoped ? "class " : "") << e.qualified_name;
    if (!e.underlying_type.empty()) {
        os << ": " << e.underlying_type;
    }
    os << "\n";
    if (!e.underlying_py_type.empty()) {
        indent(os, depth + 1);
        os << "py: " << e.underlying_py_type << "\n";
    }
    if (e.submodule) {
        indent(os, depth + 1);
        os << "submodule: " << *e.submodule << "\n";
    }
    dump_directives(os, e.directives, depth + 1);
    for (auto const &v : e.enumerators) {
        indent(os, depth + 1);
        os << v.name << " = " << v.value << "\n";
    }
}

// Recursion is intended: nested classes are themselves BoundClass.
// NOLINTNEXTLINE(misc-no-recursion)
void dump_class(llvm::raw_string_ostream &os, BoundClass const &c, int depth) {
    indent(os, depth);
    os << "class " << c.qualified_name;
    if (c.is_template) {
        os << " [template]";
    }
    for (auto const &inst : c.instantiations) {
        os << "\n";
        indent(os, depth + 1);
        os << "instantiate " << c.qualified_name << "<" << inst.type_args << "> as " << inst.py_name;
    }
    if (!c.bases.empty()) {
        os << " : ";
        for (std::size_t i = 0; i < c.bases.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << c.bases[i];
        }
    }
    os << "\n";
    if (c.submodule) {
        indent(os, depth + 1);
        os << "submodule: " << *c.submodule << "\n";
    }
    dump_directives(os, c.directives, depth + 1);
    for (auto const &m : c.ctors) {
        dump_method(os, m, depth + 1);
    }
    for (auto const &m : c.methods) {
        dump_method(os, m, depth + 1);
    }
    for (auto const &p : c.properties) {
        indent(os, depth + 1);
        os << "property " << p.py_name << ": " << p.py_type << (p.has_setter ? " [rw]" : " [ro]") << "\n";
    }
    for (auto const &f : c.fields) {
        dump_field(os, f, depth + 1);
    }
    for (auto const &e : c.nested_enums) {
        dump_enum(os, e, depth + 1);
    }
    for (auto const &n : c.nested_classes) {
        dump_class(os, n, depth + 1);
    }
}

} // namespace

std::string dump(Module const &module_) {
    std::string              buffer;
    llvm::raw_string_ostream os(buffer);
    for (auto const &c : module_.classes) {
        dump_class(os, c, 0);
    }
    for (auto const &f : module_.functions) {
        os << "function " << f.qualified_name << ": " << f.return_type << "(";
        for (std::size_t i = 0; i < f.params.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            dump_param(os, f.params[i]);
        }
        os << ")";
        if (f.is_template) {
            os << " [template]";
        }
        os << "\n";
        if (!f.return_type_canonical.empty() && f.return_type_canonical != f.return_type) {
            indent(os, 1);
            os << "ret_canonical: " << f.return_type_canonical << "\n";
        }
        indent(os, 1);
        os << "py: (";
        for (std::size_t i = 0; i < f.params.size(); ++i) {
            if (i != 0) {
                os << ", ";
            }
            dump_param_py(os, f.params[i]);
        }
        os << ") -> " << (f.return_py_type.empty() ? std::string{"None"} : f.return_py_type) << "\n";
        if (f.submodule) {
            indent(os, 1);
            os << "submodule: " << *f.submodule << "\n";
        }
        if (!f.doc.empty()) {
            indent(os, 1);
            os << "doc: " << f.doc.size() << " chars\n";
        }
        dump_directives(os, f.directives, 1);
    }
    for (auto const &e : module_.enums) {
        dump_enum(os, e, 0);
    }
    return buffer;
}

} // namespace einsums::pybind
