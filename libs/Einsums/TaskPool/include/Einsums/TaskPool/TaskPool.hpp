//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#if defined(EINSUMS_HAVE_PROFILER)
#    include <Einsums/Profile/Profile.hpp>
#endif
#include <Einsums/TaskPool/ExecutionPolicy.hpp>
#include <Einsums/TaskPool/TaskGroup.hpp>
#include <Einsums/TaskPool/TaskHandle.hpp>
#include <Einsums/TaskPool/WorkStealingDeque.hpp>
#include <Einsums/TaskPool/WorkerContext.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::task_pool {

/// @brief Work-stealing thread pool with HPX-inspired continuations and dataflow.
///
/// Provides fine-grained task parallelism for data-parallel workloads like
/// quantum chemistry integral generation. Key features:
///
/// - **Work-stealing**: Lock-free Chase-Lev deques per worker for load balancing
/// - **Continuations**: TaskHandle::then() chains, when_all() fan-in
/// - **Dataflow**: Tasks declare input futures and run only when all are ready
/// - **parallel_for / parallel_reduce**: Data-parallel patterns with chunking
/// - **Profiler integration**: Each task is a named profiler region
///
/// @par Example
/// @code
/// auto& pool = TaskPool::get_singleton();
///
/// auto a = pool.submit("compute_a", []() { return 42; });
/// auto b = pool.submit("compute_b", []() { return 3.14; });
/// auto c = pool.dataflow("combine", [](int x, double y) {
///     return x + y;
/// }, a, b);
///
/// double result = c.get();  // 45.14
/// @endcode
class EINSUMS_EXPORT TaskPool {
    EINSUMS_SINGLETON_DEF(TaskPool)

  public:
    /// @brief Number of worker threads.
    [[nodiscard]] size_t num_workers() const { return _workers.size(); }

    /// @brief Submit a named task. Returns a TaskHandle for the result.
    template <typename F>
    auto submit(std::string name, F &&callable) -> TaskHandle<std::invoke_result_t<F>> {
        using R    = std::invoke_result_t<F>;
        auto state = std::make_shared<detail::SharedState<R>>();

        auto wrapped = make_wrapped_task(std::move(name), std::forward<F>(callable), state);

        enqueue(std::move(wrapped));
        return TaskHandle<R>(std::move(state));
    }

    /// @brief Submit an unnamed task.
    template <typename F>
    auto submit(F &&callable) -> TaskHandle<std::invoke_result_t<F>> {
        return submit("task", std::forward<F>(callable));
    }

    /// @brief Fire-and-forget submission: no TaskHandle, no SharedState.
    ///
    /// For callers that track completion and failure themselves (the
    /// counter-based DataflowExecutor), submit() wastes a SharedState
    /// allocation per task on a handle that is immediately dropped. This
    /// path skips the handle machinery entirely; the task still gets a
    /// profiler zone under its name.
    ///
    /// The callable should not throw: with no handle to carry an exception,
    /// anything that escapes is logged and dropped, never propagated into
    /// the worker loop.
    template <typename F>
    void submit_detached(std::string name, F &&callable) {
        enqueue([name = std::move(name), task = std::forward<F>(callable)]() mutable {
#if defined(EINSUMS_HAVE_PROFILER)
            profile::Profiler::instance().push(name);
#endif
            try {
                task();
            } catch (...) { // NOLINT
                EINSUMS_LOG_ERROR("TaskPool: detached task '{}' threw; the exception is dropped", name);
            }
#if defined(EINSUMS_HAVE_PROFILER)
            profile::Profiler::instance().pop();
#endif
        });
    }

    /// @brief Submit a group of tasks with a collective barrier.
    TaskGroup submit_group(std::string name, std::vector<std::function<void()>> tasks);

    /// @brief Data-parallel for with execution policy.
    template <typename F>
    void parallel_for(execution::Policy policy, std::string const &name, size_t begin, size_t end, F &&callable) {
        if (begin >= end)
            return;

        if (std::holds_alternative<execution::SequentialPolicy>(policy)) {
            for (size_t i = begin; i < end; i++) {
                callable(i);
            }
            return;
        }

        auto const  &par   = std::get<execution::ParallelPolicy>(policy);
        size_t const total = end - begin;
        size_t const nw    = num_workers() + 1;
        size_t const chunk = par.chunk_size > 0 ? par.chunk_size : std::max(size_t{1}, total / (nw * 4));

        auto remaining = std::make_shared<std::atomic<size_t>>(0);

        for (size_t i = begin; i < end; i += chunk) {
            size_t const chunk_end = std::min(i + chunk, end);
            remaining->fetch_add(1);

            enqueue([name, i, chunk_end, &callable, remaining]() {
                for (size_t j = i; j < chunk_end; j++) {
                    callable(j);
                }
                remaining->fetch_sub(1);
            });
        }

        // Calling thread helps by stealing work while waiting
        help_until([&] { return remaining->load() == 0; });
    }

    /// @brief Data-parallel for with default parallel policy.
    template <typename F>
    void parallel_for(std::string name, size_t begin, size_t end, F &&callable) {
        parallel_for(execution::par, std::move(name), begin, end, std::forward<F>(callable));
    }

    /// @brief Parallel reduce with thread-local accumulation and sequential merge.
    template <typename Acc, typename InitFactory, typename Body, typename Combiner>
    Acc parallel_reduce(std::string name, size_t begin, size_t end, InitFactory &&init_factory, Body &&body, Combiner &&combiner) {
        size_t nw = num_workers() + 1;

        // Cache-line aligned accumulators to prevent false sharing
        struct alignas(64) PaddedAcc {
            Acc acc;
        };
        std::vector<PaddedAcc> accs(nw);
        for (auto &pa : accs) {
            pa.acc = init_factory();
        }

        // Each worker gets an accumulator slot based on worker_id.
        // The calling thread (worker_id = -1) uses slot 0.
        parallel_for(std::move(name), begin, end, [&](size_t idx) {
            int const wid  = current_worker_id();
            size_t    slot = (wid >= 0) ? (static_cast<size_t>(wid) % nw) : 0;
            body(idx, accs[slot].acc);
        });

        // Sequential merge
        Acc result = std::move(accs[0].acc);
        for (size_t i = 1; i < nw; i++) {
            combiner(result, accs[i].acc);
        }
        return result;
    }

    /// @brief Dataflow: submit a task that runs when all typed input handles are ready.
    ///
    /// The callable receives the results of the input handles as arguments.
    /// Fully asynchronous, with no blocking. The callable is submitted to the pool
    /// when all inputs complete, via the variadic when_all continuation mechanism.
    template <typename F, typename... Ts>
    auto dataflow(std::string name, F &&callable, TaskHandle<Ts>... inputs) -> TaskHandle<std::invoke_result_t<F, Ts...>> {
        using R         = std::invoke_result_t<F, Ts...>;
        using TupleType = std::tuple<Ts...>;

        auto result_state = std::make_shared<detail::SharedState<R>>();

        // Async when_all: returns TaskHandle<tuple<Ts...>>
        auto combined = when_all(inputs...);

        // When all inputs are ready, submit the callable with the results.
        // If any input failed, propagate its exception instead of running f.
        combined.state()->on_complete(
            [this, name = std::move(name), f = std::forward<F>(callable), result_state](TupleType const &results) mutable {
                enqueue([name = std::move(name), f = std::move(f), result_state, results]() mutable {
                    try {
                        if constexpr (std::is_void_v<R>) {
                            std::apply(f, results);
                            result_state->set_value();
                        } else {
                            result_state->set_value(std::apply(f, results));
                        }
                    } catch (...) {
                        result_state->set_exception(std::current_exception());
                    }
                });
            },
            [result_state](std::exception_ptr ep) { result_state->set_exception(ep); });

        return TaskHandle<R>(std::move(result_state));
    }

    /// @brief Dataflow specialization for void inputs.
    template <typename F>
    auto dataflow(std::string name, F &&callable, TaskHandle<void> const &input) -> TaskHandle<std::invoke_result_t<F>> {
        using R    = std::invoke_result_t<F>;
        auto state = std::make_shared<detail::SharedState<R>>();

        input.state()->on_complete(
            [this, name = std::move(name), f = std::forward<F>(callable), state]() mutable {
                enqueue([name = std::move(name), f = std::move(f), state]() mutable {
                    try {
                        if constexpr (std::is_void_v<R>) {
                            f();
                            state->set_value();
                        } else {
                            state->set_value(f());
                        }
                    } catch (...) {
                        state->set_exception(std::current_exception());
                    }
                });
            },
            [state](std::exception_ptr ep) { state->set_exception(ep); });

        return TaskHandle<R>(std::move(state));
    }

    /// @brief Shutdown the pool, joining all worker threads.
    void shutdown();

    /// @brief Check if the calling thread is a TaskPool worker.
    [[nodiscard]] bool is_worker_thread() const;

    /// @brief Get the current worker ID (or -1 if not a worker thread).
    static int current_worker_id();

    /// @brief Wake all parked workers. Called from handle wait loops.
    void wake_workers() { _notify_cv.notify_all(); }

    /// @brief Aggregate metrics for profiling.
    struct Metrics {
        size_t              total_submitted{0};
        size_t              total_completed{0};
        size_t              total_steals{0};
        size_t              active_workers{0};
        std::vector<size_t> per_worker_executed;
        std::vector<size_t> per_worker_stolen;
    };

    [[nodiscard]] Metrics snapshot_metrics() const;

    ~TaskPool();

    /// Participate in work-stealing on the calling thread until the
    /// predicate returns true. Used by TaskHandle::wait and by executors
    /// with their own completion tracking (e.g. the counter-based
    /// DataflowExecutor), so a waiting thread contributes instead of parking.
    void help_until(std::function<bool()> const &predicate);

  private:
    TaskPool();

    struct Worker {
        std::thread                              thread;
        WorkStealingDeque<std::function<void()>> deque;
        WorkerContext                            context;
        std::atomic<size_t>                      tasks_executed{0};
        std::atomic<size_t>                      tasks_stolen{0};
    };

    void worker_loop(size_t worker_id);
    template <typename U>
    friend class TaskHandle;

    void enqueue(std::function<void()> task);

    /// Create a wrapped task with profiler instrumentation.
    template <typename F, typename R>
    std::function<void()> make_wrapped_task(std::string name, F &&callable, std::shared_ptr<detail::SharedState<R>> state) {
        return [name = std::move(name), task = std::forward<F>(callable), state = std::move(state)]() mutable {
#if defined(EINSUMS_HAVE_PROFILER)
            profile::Profiler::instance().push(fmt::format("task:{}", name));
#endif
            try {
                if constexpr (std::is_void_v<R>) {
                    task();
                    state->set_value();
                } else {
                    state->set_value(task());
                }
            } catch (...) {
                state->set_exception(std::current_exception());
            }
#if defined(EINSUMS_HAVE_PROFILER)
            profile::Profiler::instance().pop();
#endif
        };
    }

    std::vector<std::unique_ptr<Worker>> _workers;
    std::atomic<bool>                    _shutdown_flag{false};
    std::atomic<size_t>                  _total_submitted{0};
    std::atomic<size_t>                  _total_completed{0};

    // Notification for parked workers
    std::mutex              _notify_mutex;
    std::condition_variable _notify_cv;
    std::atomic<size_t>     _parked_count{0};

    // Shared external submission queue (thread-safe for multi-producer).
    // External threads (non-workers) push here; workers pop from here.
    std::mutex                        _external_mutex;
    std::queue<std::function<void()>> _external_queue;
};

// ── Deferred .then() implementations (need TaskPool to submit continuations) ──

template <typename T>
template <typename F>
auto TaskHandle<T>::then(F &&next_fn) -> TaskHandle<std::invoke_result_t<F, T>> {
    return then("continuation", std::forward<F>(next_fn));
}

template <typename T>
template <typename F>
auto TaskHandle<T>::then(std::string name, F &&next_fn) -> TaskHandle<std::invoke_result_t<F, T>> {
    using R    = std::invoke_result_t<F, T>;
    auto state = std::make_shared<detail::SharedState<R>>();

    _state->on_complete(
        [name = std::move(name), f = std::forward<F>(next_fn), state](T const &val) mutable {
            // Submit continuation to the pool
            TaskPool::get_singleton().enqueue([f = std::move(f), state, val]() mutable {
                try {
                    if constexpr (std::is_void_v<R>) {
                        f(val);
                        state->set_value();
                    } else {
                        state->set_value(f(val));
                    }
                } catch (...) {
                    state->set_exception(std::current_exception());
                }
            });
        },
        [state](std::exception_ptr ep) { state->set_exception(ep); });

    return TaskHandle<R>(std::move(state));
}

template <typename F>
auto TaskHandle<void>::then(F &&next_fn) -> TaskHandle<std::invoke_result_t<F>> {
    return then("continuation", std::forward<F>(next_fn));
}

template <typename F>
auto TaskHandle<void>::then(std::string name, F &&next_fn) -> TaskHandle<std::invoke_result_t<F>> {
    using R    = std::invoke_result_t<F>;
    auto state = std::make_shared<detail::SharedState<R>>();

    _state->on_complete(
        [name = std::move(name), f = std::forward<F>(next_fn), state]() mutable {
            TaskPool::get_singleton().enqueue([f = std::move(f), state]() mutable {
                try {
                    if constexpr (std::is_void_v<R>) {
                        f();
                        state->set_value();
                    } else {
                        state->set_value(f());
                    }
                } catch (...) {
                    state->set_exception(std::current_exception());
                }
            });
        },
        [state](std::exception_ptr ep) { state->set_exception(ep); });

    return TaskHandle<R>(std::move(state));
}

} // namespace einsums::task_pool
