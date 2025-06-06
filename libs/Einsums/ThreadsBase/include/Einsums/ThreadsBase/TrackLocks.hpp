//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace einsums::threads {

struct RegisterLockData {};

namespace detail {

struct EINSUMS_EXPORT LockData {
    LockData(std::size_t trace_depth);
    LockData(RegisterLockData *data, std::size_t trace_depth);

    ~LockData();

    bool              _ignore;
    RegisterLockData *_data;
    std::string       _backtrace;
};

} // namespace detail

struct HeldLocksData {
    using HeldLocksMap = std::map<void const *, detail::LockData>;

    HeldLocksData() : _enabled(true), _ignore_all_locks(false) {}

    HeldLocksMap _map;
    bool         _enabled;
    bool         _ignore_all_locks;
};

EINSUMS_EXPORT bool register_lock(void const *lock, RegisterLockData *data = nullptr);
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

using RegisteredLocksErrorHandlerType = std::function<void()>;

/// Sets a handler which gets called when verifying that no locks are held fails.
/// Can be used to print information at the point of failure such as a backtrace
EINSUMS_EXPORT void set_registered_locks_error_handler(RegisteredLocksErrorHandlerType);

using RegisterLocksPredicateType = std::function<bool()>;

/// Sets a predicate which gets called each time a lock is registered,
/// unregistered, or when locks are verified. If the predicate returns
/// false, the corresponding function will not register, unregister, or
/// verify locks. If it returns true the corresponding function may
/// register, unregister, or verify locks, depending on other factors (such
/// as if lock detection is enabled globally). The predicate may return
/// different values depending on context.
EINSUMS_EXPORT void set_register_locks_predicate(RegisterLocksPredicateType);

struct IgnoreAllWhileChecking {
    IgnoreAllWhileChecking() { ignore_all_locks(); }
    ~IgnoreAllWhileChecking() { reset_ignored_all(); }
};

template <typename Lock>
struct IgnoreWhileChecking {
    explicit IgnoreWhileChecking(Lock const &lock) : _lock(lock) { ignore_lock(_lock); }

    ~IgnoreWhileChecking() { reset_ignored(_lock); }

    void const *_lock;
};

// The following functions are used to store the held locks information
// during thread suspension. The data is stored on a thread_local basis,
// so we must make sure that locks that are being ignored are restored
// after suspension even if the thread is being resumed on a different core.

// retrieve the current thread_local data about held locks
EINSUMS_EXPORT std::unique_ptr<HeldLocksData> get_held_locks_data();

// set the current thread_local data about held locks
EINSUMS_EXPORT void set_held_locks_data(std::unique_ptr<HeldLocksData> &&data);

} // namespace einsums::threads