//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/concepts/has_member_xxx.hpp>
#include <einsums/functional/function.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <map>
#include <memory>
#ifdef EINSUMS_HAVE_VERIFY_LOCKS_BACKTRACE
#    include <string>
#endif
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::util {

struct register_lock_data {};

// Always provide function exports, which guarantees ABI compatibility of
// Debug and Release builds.

#if defined(EINSUMS_HAVE_VERIFY_LOCKS) || defined(EINSUMS_EXPORTS)

namespace detail {

struct EINSUMS_EXPORT lock_data {
#    ifdef EINSUMS_HAVE_VERIFY_LOCKS
    lock_data(std::size_t trace_depth);
    lock_data(register_lock_data *data, std::size_t trace_depth);

    ~lock_data();

    bool                ignore_;
    register_lock_data *user_data_;
#        ifdef EINSUMS_HAVE_VERIFY_LOCKS_BACKTRACE
    std::string backtrace_;
#        endif
#    endif
};
} // namespace detail

struct held_locks_data {
    using held_locks_map = std::map<void const *, detail::lock_data>;

    held_locks_data() : enabled_(true), ignore_all_locks_(false) {}

    held_locks_map map_;
    bool           enabled_;
    bool           ignore_all_locks_;
};

///////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT bool register_lock(void const *lock, register_lock_data *data = nullptr);
EINSUMS_EXPORT bool unregister_lock(void const *lock);
EINSUMS_EXPORT void verify_no_locks();
EINSUMS_EXPORT void force_error_on_lock();
EINSUMS_EXPORT void enable_lock_detection();
EINSUMS_EXPORT void disable_lock_detection();
EINSUMS_EXPORT void trace_depth_lock_detection(std::size_t value);
EINSUMS_EXPORT void ignore_lock(void const *lock);
EINSUMS_EXPORT void reset_ignored(void const *lock);
EINSUMS_EXPORT void ignore_all_locks();
EINSUMS_EXPORT void reset_ignored_all();

using registered_locks_error_handler_type = util::detail::function<void()>;

/// Sets a handler which gets called when verifying that no locks are held
/// fails. Can be used to print information at the point of failure such as
/// a backtrace.
EINSUMS_EXPORT void set_registered_locks_error_handler(registered_locks_error_handler_type);

using register_locks_predicate_type = util::detail::function<bool()>;

/// Sets a predicate which gets called each time a lock is registered,
/// unregistered, or when locks are verified. If the predicate returns
/// false, the corresponding function will not register, unregister, or
/// verify locks. If it returns true the corresponding function may
/// register, unregister, or verify locks, depending on other factors (such
/// as if lock detection is enabled globally). The predicate may return
/// different values depending on context.
EINSUMS_EXPORT void set_register_locks_predicate(register_locks_predicate_type);

///////////////////////////////////////////////////////////////////////////
struct ignore_all_while_checking {
    ignore_all_while_checking() { ignore_all_locks(); }

    ~ignore_all_while_checking() { reset_ignored_all(); }
};

namespace detail {
EINSUMS_HAS_MEMBER_XXX_TRAIT_DEF(mutex)
}

template <typename Lock, typename Enable = std::enable_if_t<detail::has_mutex_v<Lock>>>
struct ignore_while_checking {
    explicit ignore_while_checking(Lock const *lock) : mtx_(lock->mutex()) { ignore_lock(mtx_); }

    ~ignore_while_checking() { reset_ignored(mtx_); }

    void const *mtx_;
};

// The following functions are used to store the held locks information
// during thread suspension. The data is stored on a thread_local basis,
// so we must make sure that locks then are being ignored are restored
// after suspension even if the thread is being resumed on a different core.

// retrieve the current thread_local data about held locks
EINSUMS_EXPORT std::unique_ptr<held_locks_data> get_held_locks_data();

// set the current thread_local data about held locks
EINSUMS_EXPORT void set_held_locks_data(std::unique_ptr<held_locks_data> &&data);

#else

template <typename Lock, typename Enable = void>
struct ignore_while_checking {
    explicit constexpr ignore_while_checking(Lock const * /*lock*/) noexcept {}
};

struct ignore_all_while_checking {
    constexpr ignore_all_while_checking() noexcept {}
};

constexpr inline bool register_lock(void const *, util::register_lock_data * = nullptr) noexcept {
    return true;
}
constexpr inline bool unregister_lock(void const *) noexcept {
    return true;
}
constexpr inline void verify_no_locks() noexcept {
}
constexpr inline void force_error_on_lock() noexcept {
}
constexpr inline void enable_lock_detection() noexcept {
}
constexpr inline void disable_lock_detection() noexcept {
}
constexpr inline void trace_depth_lock_detection(std::size_t /*value*/) noexcept {
}
constexpr inline void ignore_lock(void const * /*lock*/) noexcept {
}
constexpr inline void reset_ignored(void const * /*lock*/) noexcept {
}

constexpr inline void ignore_all_locks() noexcept {
}
constexpr inline void reset_ignored_all() noexcept {
}

struct held_locks_data {};

inline std::unique_ptr<held_locks_data> get_held_locks_data() {
    return std::unique_ptr<held_locks_data>();
}

constexpr inline void set_held_locks_data(std::unique_ptr<held_locks_data> && /*data*/) noexcept {
}

#endif
} // namespace einsums::util
