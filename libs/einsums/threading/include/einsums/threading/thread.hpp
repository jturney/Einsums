//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/functional/deferred_call.hpp>
#include <einsums/functional/function.hpp>
#include <einsums/functional/unique_function.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_pool_base.hpp>
#include <einsums/timing/steady_clock.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <type_traits>
#include <utility>

#include <einsums/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace einsums {
///////////////////////////////////////////////////////////////////////////
using thread_termination_handler_type = util::detail::function<void(std::exception_ptr const &e)>;
EINSUMS_EXPORT void set_thread_termination_handler(thread_termination_handler_type f);

class EINSUMS_EXPORT thread {
    using mutex_type = einsums::concurrency::detail::spinlock;
    void terminate(const char *function, const char *reason) const;

  public:
    class id;
    using native_handle_type = threads::detail::thread_id_type;

    thread() noexcept;

    template <typename F, typename Enable = typename std::enable_if<!std::is_same<std::decay_t<F>, thread>::value>::type>
    explicit thread(F &&f) {
        auto thrd_data = einsums::threads::detail::get_self_id_data();
        EINSUMS_ASSERT(thrd_data);
        start_thread(thrd_data->get_scheduler_base()->get_parent_pool(), util::detail::deferred_call(std::forward<F>(f)));
    }

    template <typename F, typename... Ts>
    explicit thread(F &&f, Ts &&...vs) {
        auto thrd_data = einsums::threads::detail::get_self_id_data();
        EINSUMS_ASSERT(thrd_data);
        start_thread(thrd_data->get_scheduler_base()->get_parent_pool(),
                     util::detail::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...));
    }

    template <typename F>
    thread(threads::detail::thread_pool_base *pool, F &&f) {
        start_thread(pool, util::detail::deferred_call(std::forward<F>(f)));
    }

    template <typename F, typename... Ts>
    thread(threads::detail::thread_pool_base *pool, F &&f, Ts &&...vs) {
        start_thread(pool, util::detail::deferred_call(std::forward<F>(f), std::forward<Ts>(vs)...));
    }

    ~thread();

  public:
            thread(thread &&) noexcept;
    thread &operator=(thread &&) noexcept;

    void swap(thread &) noexcept;
    bool joinable() const noexcept {
        std::lock_guard<mutex_type> l(mtx_);
        return joinable_locked();
    }

    void join();
    void detach() {
        std::lock_guard<mutex_type> l(mtx_);
        detach_locked();
    }

    id get_id() const noexcept;

    native_handle_type native_handle() const //-V659
    {
        std::lock_guard<mutex_type> l(mtx_);
        return id_.noref();
    }

    [[nodiscard]] static unsigned int hardware_concurrency() noexcept;

    // extensions
    void interrupt(bool flag = true);
    bool interruption_requested() const;

    static void interrupt(id, bool flag = true);

    std::size_t get_thread_data() const;
    std::size_t set_thread_data(std::size_t);

  private:
    bool joinable_locked() const noexcept { return threads::detail::invalid_thread_id != id_; }
    void detach_locked() { id_ = threads::detail::invalid_thread_id; }
    void start_thread(threads::detail::thread_pool_base *pool, util::detail::unique_function<void()> &&func);
    static threads::detail::thread_result_type thread_function_nullary(util::detail::unique_function<void()> const &func);

    mutable mutex_type                  mtx_;
    threads::detail::thread_id_ref_type id_;
};

inline void swap(thread &x, thread &y) noexcept {
    x.swap(y);
}

///////////////////////////////////////////////////////////////////////////
class thread::id {
  private:
    threads::detail::thread_id_type id_;

    friend bool operator==(thread::id const &x, thread::id const &y) noexcept;
    friend bool operator!=(thread::id const &x, thread::id const &y) noexcept;
    friend bool operator<(thread::id const &x, thread::id const &y) noexcept;
    friend bool operator>(thread::id const &x, thread::id const &y) noexcept;
    friend bool operator<=(thread::id const &x, thread::id const &y) noexcept;
    friend bool operator>=(thread::id const &x, thread::id const &y) noexcept;

    template <typename Char, typename Traits>
    friend std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &, thread::id const &);

    friend class thread;

  public:
    id() noexcept = default;

    explicit id(threads::detail::thread_id_type const &i) noexcept : id_(i) {}
    explicit id(threads::detail::thread_id_type &&i) noexcept : id_(std::move(i)) {}

    explicit id(threads::detail::thread_id_ref_type const &i) noexcept : id_(i.get().get()) {}
    explicit id(threads::detail::thread_id_ref_type &&i) noexcept : id_(std::move(i).get().get()) {}

    threads::detail::thread_id_type const &native_handle() const { return id_; }
};

inline bool operator==(thread::id const &x, thread::id const &y) noexcept {
    return x.id_ == y.id_;
}

inline bool operator!=(thread::id const &x, thread::id const &y) noexcept {
    return !(x == y);
}

inline bool operator<(thread::id const &x, thread::id const &y) noexcept {
    return x.id_ < y.id_;
}

inline bool operator>(thread::id const &x, thread::id const &y) noexcept {
    return y < x;
}

inline bool operator<=(thread::id const &x, thread::id const &y) noexcept {
    return !(x > y);
}

inline bool operator>=(thread::id const &x, thread::id const &y) noexcept {
    return !(x < y);
}

template <typename Char, typename Traits>
std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &out, thread::id const &id) {
    out << id.id_;
    return out;
}

//     template <class T> struct hash;
//     template <> struct hash<thread::id>;

///////////////////////////////////////////////////////////////////////////
namespace this_thread {
EINSUMS_EXPORT thread::id get_id() noexcept;

EINSUMS_EXPORT void yield() noexcept;
EINSUMS_EXPORT void yield_to(thread::id) noexcept;

// extensions
EINSUMS_EXPORT execution::thread_priority get_priority();
EINSUMS_EXPORT std::ptrdiff_t get_stack_size();

EINSUMS_EXPORT void interruption_point();
EINSUMS_EXPORT bool interruption_enabled();
EINSUMS_EXPORT bool interruption_requested();

EINSUMS_EXPORT void interrupt();

EINSUMS_EXPORT void sleep_until(einsums::chrono::steady_time_point const &abs_time);

inline void sleep_for(einsums::chrono::steady_duration const &rel_time) {
    sleep_until(rel_time.from_now());
}

EINSUMS_EXPORT std::size_t get_thread_data();
EINSUMS_EXPORT std::size_t set_thread_data(std::size_t);

class EINSUMS_EXPORT disable_interruption {
  private:
                          disable_interruption(disable_interruption const &);
    disable_interruption &operator=(disable_interruption const &);

    bool interruption_was_enabled_;
    friend class restore_interruption;

  public:
     disable_interruption();
    ~disable_interruption();
};

class EINSUMS_EXPORT restore_interruption {
  private:
                          restore_interruption(restore_interruption const &);
    restore_interruption &operator=(restore_interruption const &);

    bool interruption_was_enabled_;

  public:
    explicit restore_interruption(disable_interruption &d);
    ~        restore_interruption();
};
} // namespace this_thread
} // namespace einsums

template <>
struct fmt::formatter<einsums::thread::id> : fmt::formatter<einsums::threads::detail::thread_id> {
    template <typename FormatContext>
    auto format(einsums::thread::id const &id, FormatContext &ctx) const {
        return fmt::formatter<einsums::threads::detail::thread_id>::format(id.native_handle(), ctx);
    }
};

namespace std {

// specialize std::hash for einsums::thread::id
template <>
struct hash<::einsums::thread::id> {
    std::size_t einsums_STATIC_CALL_OPERATOR(::einsums::thread::id const &id) {
        std::hash<::einsums::threads::detail::thread_id_ref_type> hasher_;
        return hasher_(id.native_handle());
    }
};
} // namespace std

#include <einsums/config/warnings_suffix.hpp>
