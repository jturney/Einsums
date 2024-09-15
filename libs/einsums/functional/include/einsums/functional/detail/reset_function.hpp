//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/functional/function.hpp>
#include <einsums/functional/unique_function.hpp>

namespace einsums::util::detail {
template <typename Sig>
inline void reset_function(einsums::util::detail::function<Sig> &f) {
    f.reset();
}

template <typename Sig>
inline void reset_function(einsums::util::detail::unique_function<Sig> &f) {
    f.reset();
}

template <typename Function>
inline void reset_function(Function &f) {
    f = Function();
}
} // namespace einsums::util::detail
