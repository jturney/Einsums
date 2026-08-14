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
 * A moldable task declares a width - the number of threads its kernel will run
 * with - and must be admitted before it runs. The budget guarantees the widths
 * admitted at any instant sum to no more than the machine's thread count, which
 * is the only thing standing between per-task threading and unbounded
 * oversubscription.
 *
 * It is process-wide rather than per-scheduler because there is one machine.
 * One executor instance can be installed on several loop bodies, a control-flow
 * node nests one graph replay inside another, and independent replays run
 * concurrently; a budget per scheduler would hand each of them the whole
 * machine.
 *
 * @par Admission order
 * Strict priority with head-of-line blocking. Parked tasks are ordered by
 * @ref Priority, and while the highest-priority parked task does not fit,
 * nothing narrower is admitted ahead of it, so the machine drains until it
 * does. This bounds the wait of a wide task at the cost of some utilization;
 * backfilling a narrow task into the drain bubble needs honest duration
 * estimates and is deliberately not done here.
 *
 * @par Why it cannot deadlock
 * Every request is clamped to the total, so the head of the parked queue is
 * always admissible once the budget drains. It drains because a task that holds
 * width either computes to completion without waiting on another task, or - if
 * it waits, which is what a control-flow node does while its body replays -
 * hands its width back for the duration through @ref BlockedScope. So with no
 * new admissions the charged width strictly decreases to zero, at which point
 * the head of the queue fits by construction.
 *
 * @par Threading
 * Every public member is safe to call from any thread. Continuations are
 * invoked by whichever thread released the width that admitted them, and never
 * with the budget's lock held, so a continuation may take any lock and may
 * submit work.
 */
class EINSUMS_EXPORT WidthBudget {
    EINSUMS_SINGLETON_DEF(WidthBudget)

  public:
    /// @brief What a parked task does once its width is granted.
    ///
    /// Called with the width actually granted, which is the requested width
    /// clamped to the budget total. Called exactly once per parked request.
    using Continuation = std::function<void(unsigned)>;

    /// @brief Admission order key.
    ///
    /// @c rank is the task's urgency - the longest remaining path to a sink -
    /// and larger goes first. Equal ranks break by @c tiebreak, smaller first,
    /// so admission order is a function of the graph rather than of which
    /// thread happened to get there.
    struct Priority {
        std::int64_t rank{0};
        std::size_t  tiebreak{0};
    };

    /**
     * @brief Ask for @p width units on behalf of a task.
     *
     * @return The width charged, which the caller must later hand back through
     *         @ref release. A return of 0 means the task was parked and
     *         @p resume owns it from here; note that @p resume may already have
     *         run by the time this returns, when parking it made it the head of
     *         a queue that fits.
     *
     * Never blocks the calling thread. A request wider than the whole budget is
     * clamped to it rather than refused, because refusing would strand a task
     * no one can widen the machine for; the plan gets less parallelism than it
     * asked for, and the kernel still computes the right answer.
     */
    [[nodiscard]] unsigned acquire(unsigned width, Priority priority, Continuation resume);

    /// @brief Hand back @p width units and admit whatever parked task now fits.
    void release(unsigned width);

    /**
     * @brief Adopt the machine's current thread count as the budget total.
     *
     * Takes effect only while the budget is idle, so a total never changes
     * under tasks that were admitted against it. Call it from the thread that
     * starts a run, before any admission: a pool worker is pinned to one thread
     * and would report the pin rather than the machine.
     */
    void sync_machine_width();

    /// @brief Threads the budget is currently rationing.
    [[nodiscard]] unsigned total() const;

    /// @brief Width charged right now.
    [[nodiscard]] unsigned in_use() const;

    /// @brief Largest width ever charged by admission, since the last @ref reset_peak.
    ///
    /// Counts admissions only. The unit a blocked task reclaims when its nested
    /// run returns (@ref BlockedScope) is not an admission and is excluded, so
    /// this is exactly the invariant the gate is responsible for: the width the
    /// budget ever let compute at once, which must never exceed @ref total.
    [[nodiscard]] unsigned peak_in_use() const;

    /// @brief Number of tasks parked waiting for width.
    [[nodiscard]] std::size_t parked() const;

    /// @brief Forget the recorded peak. For tests and diagnostics.
    void reset_peak();

    /**
     * @brief Marks the calling thread as running an admitted task of @p width.
     *
     * What @ref BlockedScope reads to know how much to hand back. Constructing
     * one with a width of 0 does nothing at all, so an unplanned task pays
     * nothing for it.
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
     * A control-flow node's task replays its body through a nested run and then
     * does nothing until that run finishes. Holding its width across the wait
     * would be both untruthful - it is not computing - and unsafe: the body's
     * own tasks acquire from this same budget, so a body node planned at the
     * full machine width could never be admitted while an ancestor held a unit
     * of it, and the run would wedge.
     *
     * Constructed where a nested run begins. Outside an admitted task (the
     * thread has no hold) it does nothing.
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

    /// Move every parked task that fits, in priority order, into @p ready.
    /// Stops at the first one that does not fit - that is the head-of-line
    /// rule. Caller holds @c _mutex.
    void drain_locked(std::vector<Pending> &ready);

    /// Take back @p width without an admission check, for a task that lent its
    /// own width out and is resuming. Cannot fail; can momentarily push the
    /// charge past the total by the width of the resuming tasks, which is the
    /// honest accounting for a control-flow node that is briefly running again.
    void recharge(unsigned width);

    mutable std::mutex   _mutex;
    unsigned             _total{0};
    unsigned             _in_use{0};
    unsigned             _peak{0};
    std::uint64_t        _seq{0};
    std::vector<Pending> _parked;
};

EINSUMS_NAMESPACE_END(task_pool)
