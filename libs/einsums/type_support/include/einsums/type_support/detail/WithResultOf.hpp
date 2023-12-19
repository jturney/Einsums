//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/Forward.hpp>

#include <utility>

namespace einsums::detail {

template <typename F>
struct WithResultOf {
  private:
    F &&f;

  public:
    using type = decltype(std::declval<F &&>()());

    explicit WithResultOf(F &&f) : f(EINSUMS_FORWARD(F, f)) {}

    operator type() { return EINSUMS_FORWARD(F, f)(); }
};

template <typename F>
inline WithResultOf<F> with_result_of(F &&f) {
    return WithResultOf<F>(EINSUMS_FORWARD(F, f));
}

} // namespace einsums::detail