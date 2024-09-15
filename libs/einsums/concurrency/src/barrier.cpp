//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/concurrency/barrier.hpp>

namespace einsums::concurrency::detail {

barrier::barrier(std::size_t number_of_threads) : _number_of_threads(number_of_threads), _total(barrier_flag), _mtx(), _cond() {
}

barrier::~barrier() {
    std::unique_lock<mutex_type> l(_mtx);

    // Wait until everyone exists the barrier
    _cond.wait(l, [&] { return _total <= barrier_flag; });
}

void barrier::wait() {
    std::unique_lock<mutex_type> l(_mtx);

    // Wait until everyone exists the barrier
    _cond.wait(l, [&] { return _total <= barrier_flag; });

    // Are we the first to enter?
    if (_total == barrier_flag)
        _total = 0;

    ++_total;

    if (_total == _number_of_threads) {
        _total += barrier_flag - 1;
        _cond.notify_all();
    } else {
        // Wait until enough threads enter the barrier
        _cond.wait(l, [&] { return _total >= barrier_flag; });

        --_total;

        // get entering threads to wake up
        if (_total == barrier_flag) {
            _cond.notify_all();
        }
    }
}

} // namespace einsums::concurrency::detail