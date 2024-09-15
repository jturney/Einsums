//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::detail {
///////////////////////////////////////////////////////////////////////////
// This is a helper structure to make sure a lock gets unlocked and locked
// again in a scope.
template <typename Mutex>
class unlock_guard {
  public:
    EINSUMS_NON_COPYABLE(unlock_guard);

  public:
    using mutex_type = Mutex;

    explicit unlock_guard(Mutex &m) : m_(m) { m_.unlock(); }

    ~unlock_guard() { m_.lock(); }

  private:
    Mutex &m_;
};
} // namespace einsums::detail
