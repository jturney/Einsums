//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/Threads/detail/TrackLocks.hpp>

namespace einsums::threads {

namespace detail {

LockData::LockData(std::size_t trace_depth) : _ignore(false), _data(nullptr), _backtrace(util::backtrace(trace_depth)) {
}

LockData::LockData(RegisterLockData *data, std::size_t trace_depth)
    : _ignore(false), _data(data), _backtrace(util::backtrace(trace_depth)) {
}

LockData::~LockData() {
    delete _data;
}

struct HeldLocksDataPtr {
    HeldLocksDataPtr() : _data(new HeldLocksData) {}

    void reinitialize() { _data.reset(new HeldLocksData()); }

    std::unique_ptr<HeldLocksData> release() {
        EINSUMS_ASSERT(!!_data);
        return std::move(_data);
    }

    void set(std::unique_ptr<HeldLocksData> &&data) { _data = std::move(data); }

    std::unique_ptr<HeldLocksData> _data;
};

struct RegisterLocks {
    using HeldLocksMap = HeldLocksData::HeldLocksMap;

    static HeldLocksDataPtr &get_held_locks() {
        static thread_local HeldLocksDataPtr held_locks;
        if (!held_locks._data) {
            held_locks.reinitialize();
        }
        return held_locks;
    }

    static bool        _lock_detection_enabled;
    static std::size_t _lock_detection_trace_depth;

    static HeldLocksMap &get_lock_map() { return get_held_locks()._data->_map; }

    static bool get_lock_enabled() { return get_held_locks()._data->_enabled; }

    static void set_lock_enabled(bool enable) { get_held_locks()._data->_enabled = enable; }

    static bool get_ignore_all_locks() { return !get_held_locks()._data->_ignore_all_locks; }

    static void set_ignore_all_locks(bool enable) { get_held_locks()._data->_ignore_all_locks = enable; }
};

bool        RegisterLocks::_lock_detection_enabled     = false;
std::size_t RegisterLocks::_lock_detection_trace_depth = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH;

struct ResetLockEnabledOnExit {
    ResetLockEnabledOnExit() : _old_value(RegisterLocks::get_lock_enabled()) { RegisterLocks::set_lock_enabled(false); }
    ~ResetLockEnabledOnExit() { RegisterLocks::set_lock_enabled(_old_value); }

    bool _old_value;
};

} // namespace detail

// retrieve the current thread_local data about held locks
std::unique_ptr<HeldLocksData> get_held_locks_data() {
    return detail::RegisterLocks::get_held_locks().release();
}

// set the current thread_local data about held locks
void set_held_locks_data(std::unique_ptr<HeldLocksData> &&data) {
    detail::RegisterLocks::get_held_locks().set(std::move(data));
}

///////////////////////////////////////////////////////////////////////////
void enable_lock_detection() {
    detail::RegisterLocks::_lock_detection_enabled = true;
}

void disable_lock_detection() {
    detail::RegisterLocks::_lock_detection_enabled = false;
}

void trace_depth_lock_detection(std::size_t value) {
    detail::RegisterLocks::_lock_detection_trace_depth = value;
}

static RegisteredLocksErrorHandlerType registered_locks_error_handler;

void set_registered_locks_error_handler(RegisteredLocksErrorHandlerType f) {
    registered_locks_error_handler = f;
}

static RegisterLocksPredicateType RegisterLocks_predicate;

void set_RegisterLocks_predicate(RegisterLocksPredicateType f) {
    RegisterLocks_predicate = f;
}

///////////////////////////////////////////////////////////////////////////
bool register_lock(void const *lock, RegisterLockData *data) {
    using detail::RegisterLocks;

    if (RegisterLocks::_lock_detection_enabled && (!RegisterLocks_predicate || RegisterLocks_predicate())) {
        RegisterLocks::HeldLocksMap &held_locks = RegisterLocks::get_lock_map();

        auto it = held_locks.find(lock);
        if (it != held_locks.end())
            return false; // this lock is already registered

        std::pair<RegisterLocks::HeldLocksMap::iterator, bool> p;
        if (!data) {
            p = held_locks.insert(std::make_pair(lock, detail::LockData(RegisterLocks::_lock_detection_trace_depth)));
        } else {
            p = held_locks.insert(std::make_pair(lock, detail::LockData(data, RegisterLocks::_lock_detection_trace_depth)));
        }
        return p.second;
    }
    return true;
}

// unregister the given lock from this pika-thread
bool unregister_lock(void const *lock) {
    using detail::RegisterLocks;

    if (RegisterLocks::_lock_detection_enabled && (!RegisterLocks_predicate || RegisterLocks_predicate())) {
        RegisterLocks::HeldLocksMap &held_locks = RegisterLocks::get_lock_map();

        auto it = held_locks.find(lock);
        if (it == held_locks.end())
            return false; // this lock is not registered

        held_locks.erase(lock);
    }
    return true;
}

// verify that no locks are held by this pika-thread
namespace detail {

inline bool some_locks_are_not_ignored(RegisterLocks::HeldLocksMap const &held_locks) {
    auto end = held_locks.end();
    for (auto it = held_locks.begin(); it != end; ++it) {
        if (!it->second._ignore)
            return true;
    }

    return false;
}
} // namespace detail

void verify_no_locks() {
    using detail::RegisterLocks;

    bool enabled = RegisterLocks::get_ignore_all_locks() && RegisterLocks::get_lock_enabled();

    if (enabled && RegisterLocks::_lock_detection_enabled && (!RegisterLocks_predicate || RegisterLocks_predicate())) {
        RegisterLocks::HeldLocksMap &held_locks = RegisterLocks::get_lock_map();

        // we create a log message if there are still registered locks for
        // this OS-thread
        if (!held_locks.empty()) {
            // Temporarily disable verifying locks in case verify_no_locks
            // gets called recursively.
            detail::ResetLockEnabledOnExit e;

            if (detail::some_locks_are_not_ignored(held_locks)) {
                if (registered_locks_error_handler) {
                    registered_locks_error_handler();
                } else {
                    EINSUMS_THROW_EXCEPTION(bad_logic, "suspending thread while at least one lock is being held (default "
                                                       "handler)");
                }
            }
        }
    }
}

void force_error_on_lock() {
    // For now just do the same as during suspension. We can't reliably
    // tell whether there are still locks held as those could have been
    // acquired in a different OS thread.
    verify_no_locks();

    //{
    //    RegisterLocks::HeldLocksMap const& held_locks =
    //       RegisterLocks::get_lock_map();
    //
    //    // we throw an error if there are still registered locks for
    //    // this OS-thread
    //    if (!held_locks.empty()) {
    //        PIKA_THROW_EXCEPTION(pika::error::invalid_status, "force_error_on_lock",
    //            "At least one lock is held while thread is being "
    //            terminated or interrupted.");
    //    }
    //}
}

namespace detail {

void set_ignore_status(void const *lock, bool status) {
    if (RegisterLocks::_lock_detection_enabled && (!RegisterLocks_predicate || RegisterLocks_predicate())) {
        RegisterLocks::HeldLocksMap &held_locks = RegisterLocks::get_lock_map();

        auto it = held_locks.find(lock);
        if (it == held_locks.end()) {
            // this can happen if the lock was registered to be ignored
            // on a different OS thread
            // EINSUMS_THROW_EXCEPTION(
            //     invalid_status,
            //     "The given lock has not been registered.");
            return;
        }

        it->second._ignore = status;
    }
}
} // namespace detail

void ignore_lock(void const *lock) {
    detail::set_ignore_status(lock, true);
}

void reset_ignored(void const *lock) {
    detail::set_ignore_status(lock, false);
}

void ignore_all_locks() {
    detail::RegisterLocks::set_ignore_all_locks(true);
}

void reset_ignored_all() {
    detail::RegisterLocks::set_ignore_all_locks(false);
}

} // namespace einsums::threads