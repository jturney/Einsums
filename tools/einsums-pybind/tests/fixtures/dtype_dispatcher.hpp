//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Dtype-dispatcher fixture: same-Python-name + same-argument-signature
// + recognized scalar dtypes auto-collapse into a single Python def
// taking ``dtype="..."``. The codegen detects the pattern from the
// INSTANTIATE_AS directives below and emits one runtime if-chain
// (no manual ``EINSUMS_PYBIND_TEMPLATE_KWARGS`` needed for this case
// since there are no bool template parameters).

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

/// Allocate-and-zero a scalar value, returned as the dtype's value
/// type. The four INSTANTIATE_AS lines all share the same (int) arg
/// signature and differ only in the return type, so the codegen
/// auto-detects the dtype-dispatcher pattern.
template <typename T>
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_INSTANTIATE_AS("zero_value", float) EINSUMS_PYBIND_INSTANTIATE_AS("zero_value", double) T
    zero_value(int seed);

} // namespace einsums::fixture
