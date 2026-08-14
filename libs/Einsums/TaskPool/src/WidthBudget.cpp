//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>

#include <algorithm>
#include <memory>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(task_pool)

EINSUMS_SINGLETON_IMPL(WidthBudget)

namespace {

/// The width this thread is currently admitted for, 0 when it is not running an
/// admitted task. Lives in one translation unit so every scope object agrees on
/// it whichever module constructs them.
unsigned &tls_hold() {
    static thread_local unsigned held = 0;
    return held;
}

/// The machine's thread count, asked of a thread that can answer.
///
/// Pool workers pin their own thread count to 1 at startup, so reading the
/// ceiling there reports the pin rather than the machine. The pool sized itself
/// from the machine before pinning anything, so its worker count is the same
/// number the calling thread would have reported.
unsigned machine_width() {
    if (TaskPool::current_worker_id() >= 0) {
        return std::max<unsigned>(1, static_cast<unsigned>(TaskPool::get_singleton().num_workers()));
    }
    int const threads = hardware::get_max_threads();
    return threads > 0 ? static_cast<unsigned>(threads) : 1U;
}

} // namespace

WidthBudget::WidthBudget() = default;

bool WidthBudget::less_urgent(Pending const &a, Pending const &b) {
    if (a.rank != b.rank) {
        return a.rank < b.rank;
    }
    if (a.tiebreak != b.tiebreak) {
        return a.tiebreak > b.tiebreak;
    }
    return a.seq > b.seq;
}

void WidthBudget::charge_locked(unsigned width) {
    _in_use += width;
    _peak = std::max(_peak, _in_use);
}

void WidthBudget::drain_locked(std::vector<Pending> &ready) {
    while (!_parked.empty()) {
        // Head of the queue only. Passing over it to admit something narrower
        // is exactly the starvation this policy exists to prevent.
        if (_in_use + _parked.front().width > _total) {
            break;
        }
        std::pop_heap(_parked.begin(), _parked.end(), less_urgent);
        charge_locked(_parked.back().width);
        ready.push_back(std::move(_parked.back()));
        _parked.pop_back();
    }
}

unsigned WidthBudget::acquire(unsigned width, Priority priority, Continuation resume) {
    if (width == 0) {
        return 0;
    }

    std::vector<Pending> ready;
    {
        std::scoped_lock const lk(_mutex);
        if (_total == 0) {
            _total = machine_width();
        }
        // Clamped, not refused: a width nobody can satisfy would park forever,
        // and the plan asking for it is a tuning artifact, not a requirement.
        unsigned const want = std::min(width, _total);

        // Head-of-line: while anything is parked, a newcomer queues behind it
        // rather than taking the room the head is waiting for, however well it
        // would fit.
        if (_parked.empty() && _in_use + want <= _total) {
            charge_locked(want);
            return want;
        }

        _parked.push_back(
            Pending{.rank = priority.rank, .tiebreak = priority.tiebreak, .seq = _seq++, .width = want, .resume = std::move(resume)});
        std::push_heap(_parked.begin(), _parked.end(), less_urgent);

        // The newcomer may itself be the most urgent thing waiting, so the
        // queue is drained rather than assumed blocked. It resumes through its
        // continuation either way, which keeps one path for both cases.
        drain_locked(ready);
    }

    for (auto &entry : ready) {
        entry.resume(entry.width);
    }
    return 0;
}

void WidthBudget::release(unsigned width) {
    if (width == 0) {
        return;
    }

    std::vector<Pending> ready;
    {
        std::scoped_lock const lk(_mutex);
        if (width > _in_use) {
            EINSUMS_LOG_WARN("WidthBudget: returning {} units with only {} charged; the counter is being clamped", width, _in_use);
            _in_use = 0;
        } else {
            _in_use -= width;
        }
        drain_locked(ready);
    }

    for (auto &entry : ready) {
        entry.resume(entry.width);
    }
}

void WidthBudget::recharge(unsigned width) {
    if (width == 0) {
        return;
    }
    std::scoped_lock const lk(_mutex);
    _in_use += width;
}

void WidthBudget::sync_machine_width() {
    if (TaskPool::current_worker_id() >= 0) {
        // A nested run starts on a worker, which cannot see the machine and
        // must not overwrite what the outer run recorded.
        return;
    }
    unsigned const observed = machine_width();

    std::scoped_lock const lk(_mutex);
    if (_in_use == 0 && _parked.empty()) {
        _total = observed;
    }
}

unsigned WidthBudget::total() const {
    std::scoped_lock const lk(_mutex);
    return _total;
}

unsigned WidthBudget::in_use() const {
    std::scoped_lock const lk(_mutex);
    return _in_use;
}

unsigned WidthBudget::peak_in_use() const {
    std::scoped_lock const lk(_mutex);
    return _peak;
}

std::size_t WidthBudget::parked() const {
    std::scoped_lock const lk(_mutex);
    return _parked.size();
}

void WidthBudget::reset_peak() {
    std::scoped_lock const lk(_mutex);
    _peak = _in_use;
}

WidthBudget::HoldScope::HoldScope(unsigned width) : _width(width) {
    if (_width > 0) {
        _prev      = tls_hold();
        tls_hold() = _width;
    }
}

WidthBudget::HoldScope::~HoldScope() {
    if (_width > 0) {
        tls_hold() = _prev;
    }
}

WidthBudget::BlockedScope::BlockedScope() : _width(tls_hold()) {
    if (_width > 0) {
        // Cleared before the width goes back, so a nested run that blocks again
        // deeper in does not lend the same unit twice.
        tls_hold() = 0;
        WidthBudget::get_singleton().release(_width);
    }
}

WidthBudget::BlockedScope::~BlockedScope() {
    if (_width > 0) {
        WidthBudget::get_singleton().recharge(_width);
        tls_hold() = _width;
    }
}

EINSUMS_NAMESPACE_END(task_pool)
