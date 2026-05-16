//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <optional>

#include "IR.hpp"
#include "llvm/ADT/StringRef.h"

namespace einsums::pybind {

// Parses an annotation payload of the form
// "einsums_pybind:<name>[:<arg>[:<arg>...]]" into a Directive. Returns
// nullopt if the prefix doesn't match.
//
// Special cases:
//   - "doc:<text>" — the text contains arbitrary characters including ':',
//     so the directive name is split off and the remainder is one args[0].
//   - "instantiate:<list>" / "instantiate_as:<py>:<type>" — same: the rest
//     after the directive name (or the first ':' for instantiate_as) is
//     left intact as a single arg, since the inner payload contains commas
//     and parentheses that the codegen tool will parse later.
std::optional<Directive> parse_annotation(llvm::StringRef payload);

// Project-wide annotation prefix laid down by EINSUMS_PYBIND_DETAIL_ANNOTATE.
inline constexpr llvm::StringLiteral k_annotation_prefix = "einsums_pybind:";

} // namespace einsums::pybind
