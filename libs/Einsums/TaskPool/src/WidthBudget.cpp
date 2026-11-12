//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

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

/// The width this thread is admitted for, 0 outside an admitted task. In one TU so every module agrees.
unsigned &tls_hold() {
    static thread_local unsigned held = 0;
    return held;
}

/// The machine's thread count. A pool worker reports its pin of 1, so there the pool's worker count
/// (sized from the machine) is used instead.
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

void WidthBudget::note_peak(unsigned in_use) {
    unsigned seen = _peak.load(std::memory_order_relaxed);
    while (in_use > seen && !_peak.compare_exchange_weak(seen, in_use, std::memory_order_relaxed)) {
    }
}

bool WidthBudget::try_charge(unsigned want, unsigned total, std::uint64_t blockers) {
    std::uint64_t state = _state.load(std::memory_order_relaxed);
    while ((state & blockers) == 0 && (state & in_use_mask) + want <= total) {
        if (_state.compare_exchange_weak(state, state + want, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            note_peak(static_cast<unsigned>((state & in_use_mask) + want));
            return true;
        }
    }
    return false;
}

void WidthBudget::drain_locked(std::vector<Pending> &ready) {
    unsigned const total = _total.load(std::memory_order_relaxed);
    while (!_parked.empty()) {
        // Head of the queue only. Passing over it to admit something narrower
        // is exactly the starvation this policy exists to prevent.
        if (!try_charge(_parked.front().width, total, 0)) {
            break;
        }
        std::pop_heap(_parked.begin(), _parked.end(), less_urgent);
        ready.push_back(std::move(_parked.back()));
        _parked.pop_back();
    }
    if (_parked.empty()) {
        _state.fetch_and(~parked_bit, std::memory_order_acq_rel);
    }
}

unsigned WidthBudget::acquire(unsigned width, Priority priority, Continuation resume) {
    if (width == 0) {
        return 0;
    }

    // Nothing parked and room for it: one compare-and-swap. Either flag bit sends the request to
    // the lock, which is what keeps the head-of-line rule exact.
    if (unsigned const total = _total.load(std::memory_order_acquire); total != 0) {
        unsigned const want = std::min(width, total);
        if (try_charge(want, total, parked_bit | resizing_bit)) {
            return want;
        }
    }

    std::vector<Pending> ready;
    {
        std::scoped_lock const lk(_mutex);
        if (_total.load(std::memory_order_relaxed) == 0) {
            _total.store(machine_width(), std::memory_order_release);
        }
        unsigned const total = _total.load(std::memory_order_relaxed);
        // Clamped, not refused: a width nobody can satisfy would park forever,
        // and the plan asking for it is a tuning artifact, not a requirement.
        unsigned const want = std::min(width, total);

        // Head-of-line: while anything is parked, a newcomer queues behind it.
        if (_parked.empty() && try_charge(want, total, 0)) {
            return want;
        }

        _parked.push_back(
            Pending{.rank = priority.rank, .tiebreak = priority.tiebreak, .seq = _seq++, .width = want, .resume = std::move(resume)});
        std::push_heap(_parked.begin(), _parked.end(), less_urgent);

        // The flag goes up before the drain: a release that slipped in without the lock before
        // it is in the charge the drain reads, and every release after it takes the lock and
        // drains itself, so no unit freed while this request was parking is missed.
        _state.fetch_or(parked_bit, std::memory_order_acq_rel);

        // The newcomer may be the most urgent, so drain rather than assume it blocked.
        drain_locked(ready);
    }

    run_ready(ready);
    return 0;
}

void WidthBudget::release(unsigned width) {
    if (width == 0) {
        return;
    }

    // Nothing parked to admit: one compare-and-swap.
    std::uint64_t state = _state.load(std::memory_order_relaxed);
    while ((state & (parked_bit | resizing_bit)) == 0 && (state & in_use_mask) >= width) {
        if (_state.compare_exchange_weak(state, state - width, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return;
        }
    }

    std::vector<Pending> ready;
    {
        std::scoped_lock const lk(_mutex);
        // The units come out in a loop because a request can still charge without the lock
        // until it sees the parked bit.
        state                = _state.load(std::memory_order_relaxed);
        std::uint64_t next   = 0;
        std::uint64_t in_use = 0;
        do {
            in_use = state & in_use_mask;
            next   = (state & ~in_use_mask) | (width > in_use ? 0 : in_use - width);
        } while (!_state.compare_exchange_weak(state, next, std::memory_order_acq_rel, std::memory_order_relaxed));
        if (width > in_use) {
            EINSUMS_LOG_WARN("WidthBudget: returning {} units with only {} charged; the counter is being clamped", width, in_use);
        }
        drain_locked(ready);
    }

    run_ready(ready);
}

void WidthBudget::run_ready(std::vector<Pending> &ready) {
    // Continuations admitted while this thread is already running one. A continuation that frees
    // its units at once - a task that fails before it runs, or a caller that releases inline -
    // drains the queue again from inside the one before it, and with a few hundred parked tasks on
    // a small machine that recursion ran a thread off the end of its stack. Queued here, the
    // outermost call runs them one after another on the same thread.
    struct Resumes {
        bool                 running{false};
        std::vector<Pending> queued;
    };
    static thread_local Resumes resumes;

    if (resumes.running) {
        for (auto &entry : ready) {
            resumes.queued.push_back(std::move(entry));
        }
        return;
    }

    struct Running {
        Resumes &r;
        explicit Running(Resumes &r) : r(r) { r.running = true; }
        ~Running() { r.running = false; }
        Running(Running const &)            = delete;
        Running &operator=(Running const &) = delete;
    } const running(resumes);

    for (auto &entry : ready) {
        entry.resume(entry.width);
    }
    // In admission order: each entry was queued after the ones before it were admitted. Moved out
    // before it runs, since running it may queue more and grow the vector.
    for (std::size_t i = 0; i < resumes.queued.size(); ++i) {
        Pending entry = std::move(resumes.queued[i]);
        entry.resume(entry.width);
    }
    resumes.queued.clear();
}

void WidthBudget::recharge(unsigned width) {
    if (width == 0) {
        return;
    }
    _state.fetch_add(width, std::memory_order_acq_rel);
}

void WidthBudget::sync_machine_width() {
    if (TaskPool::current_worker_id() >= 0) {
        // A nested run starts on a worker, which cannot see the machine and
        // must not overwrite what the outer run recorded.
        return;
    }
    unsigned const observed = machine_width();

    std::scoped_lock const lk(_mutex);
    // Only while idle: nothing charged and nothing parked, claimed in one step so no request
    // can be admitted against the old total while the new one goes in.
    std::uint64_t idle = 0;
    if (_state.compare_exchange_strong(idle, resizing_bit, std::memory_order_acq_rel)) {
        _total.store(observed, std::memory_order_release);
        _state.fetch_and(~resizing_bit, std::memory_order_acq_rel);
    }
}

unsigned WidthBudget::total() const {
    return _total.load(std::memory_order_acquire);
}

unsigned WidthBudget::in_use() const {
    return static_cast<unsigned>(_state.load(std::memory_order_acquire) & in_use_mask);
}

unsigned WidthBudget::peak_in_use() const {
    return _peak.load(std::memory_order_relaxed);
}

std::size_t WidthBudget::parked() const {
    std::scoped_lock const lk(_mutex);
    return _parked.size();
}

void WidthBudget::reset_peak() {
    _peak.store(in_use(), std::memory_order_relaxed);
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
