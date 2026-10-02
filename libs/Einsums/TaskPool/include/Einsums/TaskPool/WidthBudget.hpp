//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(task_pool)

/**
 * @brief Process-wide admission control for the threads a task is allowed to fork.
 *
 * A moldable task declares a width (its kernel's thread count) and must be admitted before it
 * runs; admitted widths never sum past the machine's thread count. Process-wide, since nested and
 * concurrent replays share one machine.
 *
 * @par Admission order
 * Strict priority with head-of-line blocking: while the highest-priority parked task does not
 * fit, nothing is admitted ahead of it. No backfilling, which would need duration estimates.
 *
 * @par Why it cannot deadlock
 * Requests are clamped to the total, and a task holding width either runs to completion or lends
 * its width back while it waits (@ref BlockedScope). So the charge drains to zero and the head fits.
 *
 * @par Threading
 * Every member is thread-safe. Continuations run on the releasing thread, never under the lock.
 */
class EINSUMS_EXPORT WidthBudget {
    EINSUMS_SINGLETON_DEF(WidthBudget)

  public:
    /// @brief What a parked task does once its width is granted.
    ///
    /// Called exactly once, with the granted width (the request clamped to the total).
    using Continuation = std::function<void(unsigned)>;

    /// @brief Admission order key.
    ///
    /// Larger @c rank (longest remaining path to a sink) first, then smaller @c tiebreak, so the order
    /// depends on the graph, not on thread timing.
    struct Priority {
        std::int64_t rank{0};
        std::size_t  tiebreak{0};
    };

    /**
     * @brief Ask for @p width units on behalf of a task.
     *
     * @return The width charged, to be handed back through @ref release; 0 means the task was parked
     *         and @p resume (which may already have run) owns it.
     *
     * Never blocks. A request wider than the budget is clamped, not refused.
     */
    [[nodiscard]] unsigned acquire(unsigned width, Priority priority, Continuation resume);

    /// @brief Hand back @p width units and admit whatever parked task now fits.
    void release(unsigned width);

    /**
     * @brief Adopt the machine's current thread count as the budget total.
     *
     * Only while idle, so admitted tasks never see the total change. Call from the thread starting
     * a run: a pool worker is pinned to one thread.
     */
    void sync_machine_width();

    /// @brief Threads the budget is currently rationing.
    [[nodiscard]] unsigned total() const;

    /// @brief Width charged right now.
    [[nodiscard]] unsigned in_use() const;

    /// @brief Largest width ever charged by admission, since the last @ref reset_peak.
    ///
    /// Admissions only (not widths reclaimed after a @ref BlockedScope), so it must never exceed @ref total.
    [[nodiscard]] unsigned peak_in_use() const;

    /// @brief Number of tasks parked waiting for width.
    [[nodiscard]] std::size_t parked() const;

    /// @brief Forget the recorded peak. For tests and diagnostics.
    void reset_peak();

    /**
     * @brief Marks the calling thread as running an admitted task of @p width.
     *
     * Read by @ref BlockedScope. A width of 0 does nothing.
     */
    class EINSUMS_EXPORT HoldScope {
      public:
        explicit HoldScope(unsigned width);
        ~HoldScope();

        HoldScope(HoldScope const &)            = delete;
        HoldScope &operator=(HoldScope const &) = delete;
        HoldScope(HoldScope &&)                 = delete;
        HoldScope &operator=(HoldScope &&)      = delete;

      private:
        unsigned _width{0};
        unsigned _prev{0};
    };

    /**
     * @brief Lends the calling task's width to the work it is about to wait on.
     *
     * Constructed where a nested run begins: the waiting task is not computing, and its body's tasks
     * draw on the same budget, so holding the width could wedge the run. Does nothing outside an
     * admitted task.
     */
    class EINSUMS_EXPORT BlockedScope {
      public:
        BlockedScope();
        ~BlockedScope();

        BlockedScope(BlockedScope const &)            = delete;
        BlockedScope &operator=(BlockedScope const &) = delete;
        BlockedScope(BlockedScope &&)                 = delete;
        BlockedScope &operator=(BlockedScope &&)      = delete;

      private:
        unsigned _width{0};
    };

  private:
    WidthBudget();

    /// One parked request, ordered by Priority for the head-of-line rule.
    struct Pending {
        std::int64_t  rank{0};
        std::size_t   tiebreak{0};
        std::uint64_t seq{0}; ///< arrival order, the last tiebreak so ordering is total
        unsigned      width{0};
        Continuation  resume;
    };

    /// Ordering for the max-heap in @c _parked: the entry that comes LAST here
    /// is the one admitted first.
    static bool less_urgent(Pending const &a, Pending const &b);

    /// Charge @p width and update the peak. Caller holds @c _mutex.
    void charge_locked(unsigned width);

    /// Move parked tasks that fit, in priority order, into @p ready, stopping at the first that does
    /// not (head-of-line). Caller holds @c _mutex.
    void drain_locked(std::vector<Pending> &ready);

    /// Take back @p width without an admission check, for a task resuming after lending its width.
    /// May briefly push the charge past the total.
    void recharge(unsigned width);

    mutable std::mutex   _mutex;
    unsigned             _total{0};
    unsigned             _in_use{0};
    unsigned             _peak{0};
    std::uint64_t        _seq{0};
    std::vector<Pending> _parked;
};

EINSUMS_NAMESPACE_END(task_pool)
