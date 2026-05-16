//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Properties.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace einsums::pybind {

namespace {

// Find the first directive matching `name` and return its leading arg
// (the convention for both @getter("X") and @setter("X")).
std::optional<std::string> first_directive_arg(BoundMethod const &m, std::string const &name) {
    for (auto const &d : m.directives) {
        if (d.name == name && !d.args.empty()) {
            return d.args.front();
        }
    }
    return std::nullopt;
}

// Recursion is intended: nested classes are themselves BoundClass.
// NOLINTNEXTLINE(misc-no-recursion)
void compute_properties_for(BoundClass &cls) {
    cls.properties.clear();

    for (std::size_t i = 0; i < cls.methods.size(); ++i) {
        auto const &m         = cls.methods[i];
        auto const  prop_name = first_directive_arg(m, "getter");
        if (!prop_name) {
            continue;
        }

        BoundProperty prop;
        prop.py_name      = *prop_name;
        prop.type         = m.return_type;
        prop.py_type      = m.return_py_type;
        prop.doc          = m.doc;
        prop.getter_index = i;

        for (std::size_t k = 0; k < cls.methods.size(); ++k) {
            if (k == i) {
                continue;
            }
            auto const setter_name = first_directive_arg(cls.methods[k], "setter");
            if (setter_name && *setter_name == *prop_name) {
                prop.has_setter   = true;
                prop.setter_index = k;
                break;
            }
        }

        cls.properties.push_back(std::move(prop));
    }

    for (auto &nested : cls.nested_classes) {
        compute_properties_for(nested);
    }
}

} // namespace

void compute_properties(BoundClass &cls) {
    compute_properties_for(cls);
}

void compute_properties(Module &module_) {
    for (auto &cls : module_.classes) {
        compute_properties_for(cls);
    }
}

} // namespace einsums::pybind
