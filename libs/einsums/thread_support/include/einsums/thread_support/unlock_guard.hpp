//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::detail {

template <typename Mutex>
struct unlock_guard {
    EINSUMS_NON_COPYABLE(unlock_guard);

    using mutex_type = Mutex;

    explicit unlock_guard(Mutex &m) : _m(m) { _m.unlock(); }

    ~unlock_guard() { _m.lock(); }

  private:
    Mutex &_m;
};

} // namespace einsums::detail