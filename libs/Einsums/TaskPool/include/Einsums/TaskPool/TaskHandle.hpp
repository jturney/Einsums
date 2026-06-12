//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::task_pool {

// Forward declaration
class TaskPool;

/// @brief Wake all TaskPool workers. Called from handle wait loops to
/// ensure enqueued tasks get processed even if the enqueue notification was missed.
void wake_pool_workers();

namespace detail {

/// @brief Shared state for a TaskHandle. Holds the result, exception, and continuations.
template <typename T>
struct SharedState {
    std::mutex                                           mutex;
    std::condition_variable                              cv;
    std::optional<T>                                     result;
    std::exception_ptr                                   exception;
    bool                                                 ready{false};
    std::vector<std::function<void(T const &)>>          continuations;
    std::vector<std::function<void(std::exception_ptr)>> exc_continuations;

    void set_value(T value) {
        std::vector<std::function<void(T const &)>> conts;
        {
            std::scoped_lock lock(mutex);
            result = std::move(value);
            ready  = true;
            conts  = std::move(continuations);
            exc_continuations.clear(); // success: failure handlers will never run
        }
        cv.notify_all();
        for (auto &c : conts) {
            c(*result);
        }
    }

    void set_exception(std::exception_ptr ep) {
        std::vector<std::function<void(std::exception_ptr)>> econts;
        {
            std::scoped_lock lock(mutex);
            exception = std::move(ep);
            ready     = true;
            econts    = std::move(exc_continuations);
            continuations.clear(); // failure: success handlers will never run
        }
        cv.notify_all();
        // Propagate the failure downstream so dependents complete (with this
        // exception) instead of being orphaned and hanging forever.
        for (auto &c : econts) {
            c(exception);
        }
    }

    T get() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return ready; });
        if (exception) {
            std::rethrow_exception(exception);
        }
        return *result;
    }

    void wait() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return ready; });
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

    [[nodiscard]] bool is_ready() const {
        // Relaxed is fine for a polling check
        return ready;
    }

    /// Register completion handlers. @p on_ok runs with the value on success;
    /// @p on_err runs with the exception on failure. Exactly one fires. If the
    /// state is already complete the matching handler runs immediately, but only
    /// after the lock is released. User code never runs while holding the mutex,
    /// because it can re-enter the pool or acquire other locks and deadlock.
    void on_complete(std::function<void(T const &)> on_ok, std::function<void(std::exception_ptr)> on_err) {
        bool run_ok  = false;
        bool run_err = false;
        {
            std::scoped_lock lock(mutex);
            if (ready) {
                if (exception) {
                    run_err = true;
                } else {
                    run_ok = true;
                }
            } else {
                continuations.push_back(std::move(on_ok));
                exc_continuations.push_back(std::move(on_err));
            }
        }
        if (run_ok) {
            on_ok(*result); // result/exception are stable once ready
        } else if (run_err) {
            on_err(exception);
        }
    }
};

/// @brief Specialization for void.
template <>
struct SharedState<void> {
    std::mutex                                           mutex;
    std::condition_variable                              cv;
    std::exception_ptr                                   exception;
    bool                                                 ready{false};
    std::vector<std::function<void()>>                   continuations;
    std::vector<std::function<void(std::exception_ptr)>> exc_continuations;

    void set_value() {
        std::vector<std::function<void()>> conts;
        {
            std::scoped_lock const lock(mutex);
            ready = true;
            conts = std::move(continuations);
            exc_continuations.clear(); // success: failure handlers will never run
        }
        cv.notify_all();
        for (auto &c : conts) {
            c();
        }
    }

    void set_exception(std::exception_ptr ep) {
        std::vector<std::function<void(std::exception_ptr)>> econts;
        {
            std::scoped_lock const lock(mutex);
            exception = std::move(ep);
            ready     = true;
            econts    = std::move(exc_continuations);
            continuations.clear(); // failure: success handlers will never run
        }
        cv.notify_all();
        // Propagate the failure downstream so dependents complete (with this
        // exception) instead of being orphaned and hanging forever.
        for (auto &c : econts) {
            c(exception);
        }
    }

    void get() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return ready; });
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

    void wait() { get(); }

    [[nodiscard]] bool is_ready() const { return ready; }

    /// See SharedState<T>::on_complete. Exactly one of @p on_ok / @p on_err
    /// fires; if already complete it runs after the lock is released.
    void on_complete(std::function<void()> on_ok, std::function<void(std::exception_ptr)> on_err) {
        bool run_ok  = false;
        bool run_err = false;
        {
            std::scoped_lock const lock(mutex);
            if (ready) {
                if (exception) {
                    run_err = true;
                } else {
                    run_ok = true;
                }
            } else {
                continuations.push_back(std::move(on_ok));
                exc_continuations.push_back(std::move(on_err));
            }
        }
        if (run_ok) {
            on_ok();
        } else if (run_err) {
            on_err(exception);
        }
    }
};

} // namespace detail

/// @brief Future-like handle with continuation support.
///
/// Inspired by HPX's hpx::future. Supports:
/// - get() / wait() / ready() for synchronization
/// - then() for continuation chaining
/// - when_all() for fan-in
template <typename T>
class TaskHandle {
  public:
    TaskHandle() = default;
    explicit TaskHandle(std::shared_ptr<detail::SharedState<T>> state) : _state(std::move(state)) {}

    /// @brief Block until the task completes and return the result.
    /// @throws Rethrows any exception from the task.
    T get() { return _state->get(); }

    /// @brief Block until the task completes.
    void wait() { _state->wait(); }

    /// @brief Non-blocking check if the task has completed.
    [[nodiscard]] bool ready() const { return _state && _state->is_ready(); }

    /// @brief Check if this handle is valid (has shared state).
    [[nodiscard]] bool valid() const { return _state != nullptr; }

    /// @brief Register a continuation that runs when this task completes.
    ///
    /// The continuation receives the result of this task and returns a new value.
    /// Returns a TaskHandle for the continuation's result.
    ///
    /// NOTE: The continuation is submitted to the TaskPool for execution,
    /// not run inline. This requires TaskPool to be available (implemented in TaskPool.hpp).
    template <typename F>
    auto then(F &&next_fn) -> TaskHandle<std::invoke_result_t<F, T>>;

    /// @brief Named continuation (for profiler visibility).
    template <typename F>
    auto then(std::string name, F &&next_fn) -> TaskHandle<std::invoke_result_t<F, T>>;

    /// Access shared state (internal use by TaskPool).
    std::shared_ptr<detail::SharedState<T>> const &state() const { return _state; }

  private:
    std::shared_ptr<detail::SharedState<T>> _state;
};

/// @brief Specialization for void.
template <>
class TaskHandle<void> {
  public:
    TaskHandle() = default;
    explicit TaskHandle(std::shared_ptr<detail::SharedState<void>> state) : _state(std::move(state)) {}

    void               get() { _state->get(); }
    void               wait() { _state->wait(); }
    [[nodiscard]] bool ready() const { return _state && _state->is_ready(); }
    [[nodiscard]] bool valid() const { return _state != nullptr; }

    template <typename F>
    auto then(F &&next_fn) -> TaskHandle<std::invoke_result_t<F>>;

    template <typename F>
    auto then(std::string name, F &&next_fn) -> TaskHandle<std::invoke_result_t<F>>;

    [[nodiscard]] std::shared_ptr<detail::SharedState<void>> const &state() const { return _state; }

  private:
    std::shared_ptr<detail::SharedState<void>> _state;
};

/// @brief Create an immediately-ready TaskHandle.
template <typename T>
TaskHandle<T> make_ready_handle(T value) {
    auto state = std::make_shared<detail::SharedState<T>>();
    state->set_value(std::move(value));
    return TaskHandle<T>(std::move(state));
}

/// @brief Create an immediately-ready void TaskHandle.
inline TaskHandle<void> make_ready_handle() {
    auto state = std::make_shared<detail::SharedState<void>>();
    state->set_value();
    return TaskHandle<void>(std::move(state));
}

/// @brief Fan-in: returns a handle that completes when all input handles complete.
template <typename T>
TaskHandle<std::vector<T>> when_all(std::vector<TaskHandle<T>> handles) {
    auto   combined_state = std::make_shared<detail::SharedState<std::vector<T>>>();
    size_t n              = handles.size();

    if (n == 0) {
        combined_state->set_value({});
        return TaskHandle<std::vector<T>>(combined_state);
    }

    auto results   = std::make_shared<std::vector<T>>(n);
    auto remaining = std::make_shared<std::atomic<size_t>>(n);
    auto failed    = std::make_shared<std::atomic<bool>>(false);

    for (size_t i = 0; i < n; i++) {
        handles[i].state()->on_complete(
            [i, results, remaining, combined_state, failed](T const &val) {
                (*results)[i] = val;
                if (remaining->fetch_sub(1) == 1 && !failed->load()) {
                    combined_state->set_value(std::move(*results));
                }
            },
            [combined_state, failed](std::exception_ptr ep) {
                // First input failure propagates to the combined handle.
                if (!failed->exchange(true)) {
                    combined_state->set_exception(ep);
                }
            });
    }

    return TaskHandle<std::vector<T>>(combined_state);
}

/// @brief Fan-in for void handles.
inline TaskHandle<void> when_all(std::vector<TaskHandle<void>> handles) {
    auto         combined_state = std::make_shared<detail::SharedState<void>>();
    size_t const n              = handles.size();

    if (n == 0) {
        combined_state->set_value();
        return TaskHandle<void>(combined_state);
    }

    auto remaining = std::make_shared<std::atomic<size_t>>(n);
    auto failed    = std::make_shared<std::atomic<bool>>(false);

    for (size_t i = 0; i < n; i++) {
        handles[i].state()->on_complete(
            [remaining, combined_state, failed]() {
                if (remaining->fetch_sub(1) == 1 && !failed->load()) {
                    combined_state->set_value();
                }
            },
            [combined_state, failed](std::exception_ptr ep) {
                if (!failed->exchange(true)) {
                    combined_state->set_exception(ep);
                }
            });
    }

    return TaskHandle<void>(combined_state);
}

// ── Variadic when_all: returns TaskHandle<std::tuple<T1, T2, ...>> ──────────

namespace detail {

/// Helper: register a continuation on handle[I] that stores its result into
/// slot I of the shared tuple, and decrements the remaining counter.
template <size_t I, typename Tuple, typename T>
void register_tuple_continuation(TaskHandle<T> &handle, std::shared_ptr<Tuple> results,
                                 std::shared_ptr<std::atomic<size_t>> const &remaining, std::shared_ptr<SharedState<Tuple>> combined_state,
                                 std::shared_ptr<std::atomic<bool>> const &failed) {
    handle.state()->on_complete(
        [results, remaining, combined_state, failed](T const &val) {
            std::get<I>(*results) = val;
            if (remaining->fetch_sub(1) == 1 && !failed->load()) {
                combined_state->set_value(std::move(*results));
            }
        },
        [combined_state, failed](std::exception_ptr ep) {
            if (!failed->exchange(true)) {
                combined_state->set_exception(ep);
            }
        });
}

/// Recursively register continuations for each handle in the tuple.
template <size_t I = 0, typename TupleState, typename Combined, typename... Ts>
void register_all_continuations(std::tuple<TaskHandle<Ts>...> &handles, std::shared_ptr<TupleState> results,
                                std::shared_ptr<std::atomic<size_t>> remaining, std::shared_ptr<Combined> combined_state,
                                std::shared_ptr<std::atomic<bool>> failed) {
    if constexpr (I < sizeof...(Ts)) {
        register_tuple_continuation<I>(std::get<I>(handles), results, remaining, combined_state, failed);
        register_all_continuations<I + 1>(handles, results, remaining, combined_state, failed);
    }
}

} // namespace detail

/// @brief Variadic when_all: returns TaskHandle<std::tuple<T1, T2, ...>>.
///
/// Completes when ALL input handles are ready. The result tuple contains
/// the values in the same order as the input handles.
///
/// @code
/// auto a = pool.submit("a", []() { return 42; });
/// auto b = pool.submit("b", []() { return 3.14; });
/// auto combined = when_all(a, b);
/// auto [x, y] = combined.get();  // x=42, y=3.14
/// @endcode
template <typename... Ts>
auto when_all(TaskHandle<Ts>... handles) -> TaskHandle<std::tuple<Ts...>> {
    using TupleType    = std::tuple<Ts...>;
    constexpr size_t N = sizeof...(Ts);

    auto combined_state = std::make_shared<detail::SharedState<TupleType>>();

    if constexpr (N == 0) {
        combined_state->set_value(TupleType{});
        return TaskHandle<TupleType>(combined_state);
    } else {
        auto results   = std::make_shared<TupleType>();
        auto remaining = std::make_shared<std::atomic<size_t>>(N);
        auto failed    = std::make_shared<std::atomic<bool>>(false);

        auto handle_tuple = std::make_tuple(handles...);
        detail::register_all_continuations(handle_tuple, results, remaining, combined_state, failed);

        return TaskHandle<TupleType>(std::move(combined_state));
    }
}

} // namespace einsums::task_pool
