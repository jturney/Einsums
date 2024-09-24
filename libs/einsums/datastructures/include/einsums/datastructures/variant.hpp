//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_CXX17_COPY_ELISION)
#    include <variant>

namespace einsums::detail {
using std::get;
using std::holds_alternative;
using std::monostate;
using std::variant;
using std::visit;
} // namespace einsums::detail

#else
#    error C++17 with copy elision support in variant is required
#endif
