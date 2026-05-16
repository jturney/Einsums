//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "AnnotationParser.hpp"

#include "llvm/ADT/StringSwitch.h"

namespace einsums::pybind {

namespace {

// Directives whose argument is free-form text that may itself contain `:`
// or commas. For these we keep the entire tail in a single arg so the
// downstream parser (codegen tool) can interpret it without our colon
// splitter chopping the payload. ``holder:std::shared_ptr`` is the
// canonical motivating case — without this, the inner ``::`` would split
// the holder type into ``std`` and ``shared_ptr``.
bool directive_takes_free_form_tail(llvm::StringRef name) {
    return llvm::StringSwitch<bool>(name)
        .Case("doc", true)
        .Case("instantiate", true)
        .Case("instantiate_member", true)
        .Case("holder", true)
        .Case("buffer_from", true)
        .Case("buffer_protocol_std", true)
        .Case("iterator_std", true)
        .Case("index_protocol_std", true)
        .Case("implicit_from", true)
        .Case("template_kwargs", true)
        .Default(false);
}

// instantiate_as / instantiate_template / instantiate_bools are hybrids:
// first ':'-delimited field after the directive name is the Python
// identifier (or name template); the rest is the C++ payload and may
// contain colons (e.g., for nested templates with std::ratio default args).
bool directive_takes_one_then_tail(llvm::StringRef name) {
    return name == "instantiate_as" || name == "instantiate_template" || name == "instantiate_bools" || name == "instantiate_member_as";
}

} // namespace

std::optional<Directive> parse_annotation(llvm::StringRef payload) {
    if (!payload.starts_with(k_annotation_prefix)) {
        return std::nullopt;
    }
    payload = payload.drop_front(k_annotation_prefix.size());

    auto [name, rest] = payload.split(':');
    Directive result;
    result.name = name.str();

    if (directive_takes_free_form_tail(name)) {
        if (!rest.empty() || payload.contains(':')) {
            result.args.push_back(rest.str());
        }
        return result;
    }

    if (directive_takes_one_then_tail(name)) {
        auto [first, tail] = rest.split(':');
        result.args.push_back(first.str());
        result.args.push_back(tail.str());
        return result;
    }

    while (!rest.empty()) {
        auto [head, more] = rest.split(':');
        result.args.push_back(head.str());
        rest = more;
    }
    return result;
}

} // namespace einsums::pybind
