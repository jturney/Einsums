//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

// The implementation is based on the tree barrier from libc++ with the license below. See header
// file for differences to the original.

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <einsums/synchronization/barrier.hpp>
#include <einsums/threading_base/thread_data.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>

namespace einsums::detail {
    barrier_algorithm_base::barrier_algorithm_base(std::ptrdiff_t expected)
    {
        std::size_t const count = (expected + 1) >> 1;
        state = std::unique_ptr<state_t[]>(new state_t[count]);
    }

    bool barrier_algorithm_base::arrive(std::ptrdiff_t expected, detail::barrier_phase_t old_phase)
    {
        detail::barrier_phase_t const half_step = old_phase + 1, full_step = old_phase + 2;
        std::size_t current_expected = expected;
        auto einsums_thread_id = einsums::threads::detail::get_self_id();

        // The original libc++ implementation uses only the id of the current std::thread as the
        // input for the hash. This implementation prefers to use the einsums thread id if available,
        // and otherwise uses the std::thread id.
        std::size_t current = einsums_thread_id == einsums::threads::detail::invalid_thread_id ?
            std::hash<einsums::threads::detail::thread_id_type>()(
                einsums::threads::detail::get_self_id()) :
            std::hash<std::thread::id>()(std::this_thread::get_id()) % ((expected + 1) >> 1);
        for (int round = 0;; ++round)
        {
            if (current_expected <= 1) { return true; }

            std::size_t const end_node = ((current_expected + 1) >> 1), last_node = end_node - 1;

            while (true)
            {
                if (current == end_node) current = 0;
                detail::barrier_phase_t expect = old_phase;
                if (current == last_node && (current_expected & 1))
                {
                    if (state[current].tickets[round].phase.compare_exchange_strong(
                            expect, full_step, std::memory_order_acq_rel))
                        break;    // I'm 1 in 1, go to next round
                }
                else if (state[current].tickets[round].phase.compare_exchange_strong(
                             expect, half_step, std::memory_order_acq_rel))
                {
                    return false;    // I'm 1 in 2, done with arrival
                }
                else if (expect == half_step)
                {
                    if (state[current].tickets[round].phase.compare_exchange_strong(
                            expect, full_step, std::memory_order_acq_rel))
                        break;    // I'm 2 in 2, go to next round
                }

                ++current;
            }

            current_expected = last_node + 1;
            current >>= 1;
        }
    }
}    // namespace einsums::detail
