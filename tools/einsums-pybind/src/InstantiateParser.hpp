//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace einsums::pybind {

// Parses the payload of an @instantiate directive. The raw text after the
// macro expands (and after AnnotationParser strips the prefix) looks like:
//
//     Tensor, T(float, double, std::complex<float>), Rank(1, 2, 3)
//
// We split it into:
//   - class_name = "Tensor"
//   - groups = [{keyword:"T",    values:["float","double","std::complex<float>"]},
//               {keyword:"Rank", values:["1","2","3"]}]
//
// The parser respects nested ``<>`` and ``()`` so commas inside template
// arguments don't split the wrong list. The keyword on each group is
// load-bearing — Visitor::collect_instantiations matches each keyword
// against the class's actual template-parameter names and refuses to emit
// bindings on mismatch. That gives a clean diagnostic if a stray macro
// (e.g. ``#define Element ...``) mangled the payload.
struct ParamGroup {
    std::string              keyword;
    std::vector<std::string> values;
};

struct InstantiateSpec {
    std::string             class_name;
    std::vector<ParamGroup> groups;
};

InstantiateSpec parse_instantiate(std::string const &payload);

// Parses an @instantiate_as directive, which already arrives with two
// well-defined args from AnnotationParser:
//   args[0] = "Tensor2d"
//   args[1] = "Tensor<double, 2>"   (full concrete C++ type expression)
//
// We split the type expression into the angle-bracket payload (the
// "type_args" the codegen emits between ``<`` and ``>``) and the leading
// class name.
struct InstantiateAsSpec {
    std::string py_name;
    std::string class_name;
    std::string type_args;
};

InstantiateAsSpec parse_instantiate_as(std::string const &py_name, std::string const &type_expr);

// Cross-product expansion: given an ordered list of value lists, produce
// every combination as a comma-joined string ready to paste between ``<``
// and ``>``.  E.g. ([float,double], [1,2]) -> ["float, 1", "float, 2",
// "double, 1", "double, 2"].
std::vector<std::string> cross_product(std::vector<std::vector<std::string>> const &lists);

// Build a Python identifier from a class base name and a comma-joined
// argument string. Non-identifier characters collapse to underscores so
// ``Tensor`` + ``std::complex<float>, 2`` -> ``Tensor_std_complex_float_2``.
std::string sanitize_python_name(std::string const &base, std::string const &type_args);

} // namespace einsums::pybind
