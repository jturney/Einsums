//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config/compiler_fence.hpp>
#include <einsums/thread_support/spinlock.hpp>

#include <chrono>
#include <thread>

namespace einsums::detail {

void spinlock::yield_k(unsigned k) noexcept {
    // Experiments on Windows and Fedora 32 show that a single pause,
    // followed by an immediate sleep, is best.

    if (k == 0) {
        EINSUMS_SMT_PAUSE;
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

} // namespace einsums::detail
