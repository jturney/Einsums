//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>

namespace einsums::design_pats {

/**
 * @struct Observable
 *
 * @brief Implementation of the Subject part of the Observable pattern from the Gang of Four.
 *
 * Here is an example of the use of this class.
 *
 * @code
 *  void thread_observer(Observable<int> &observable) {
 *      while (true) {
 *          observable.wait_for_change();
 *          int new_value = observable; // Use casting operator to get the value
 *          std::cout << "Thread observer detected change: " << new_value << '\n';
 *      }
 *  }
 *
 *  int main() {
 *      Observable<int> observable(0); // Initialize with 0
 *
 *      // Register a callback observer
 *      observable.add_observer([](const int& value) {
 *          std::cout << "Callback observer detected change: " << value << '\n';
 *      });
 *
 *      // Launch a thread-based observer
 *      std::thread observer_thread(thread_observer, std::ref(observable));
 *
 *      // Simulate changes in the variable
 *      for (int i = 1; i <= 5; ++i) {
 *          std::this_thread::sleep_for(std::chrono::seconds(1));
 *          observable = i; // Use assignment operator
 *      }
 *
 *      observer_thread.join();
 *      return 0;
 *  }
 *  @endcode
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{1.0.1}
 *      The observable can now be locked and unlocked. When unlocked, it can be told to update its observers or not.
 * @endversion
 */
template <typename T>
struct Observable {
  public:
    /**
     * @brief Constructor.
     *
     * @param initial_value The initial value of the observable.
     *
     * @versionadded{1.0.0}
     */
    Observable(T initial_value = T{}) noexcept : _state(std::move(initial_value)) {}

    /**
     * @brief Assignment operator for setting the value
     *
     * @param value The value to assign to the observable.
     *
     * @versionadded{1.0.0}
     */
    Observable &operator=(T const &value) {
        _state = value;
        return *this;
    }

    /**
     * @brief Casting operator for retrievign the value.
     *
     * @versionadded{1.0.0}
     */
    operator T() const { return get_value(); }

    /**
     * @brief Explicit getter.
     *
     * @versionadded{1.0.0}
     */
    T const &get_value() const {
        std::scoped_lock lock(_mutex);
        return _state;
    }

    /**
     * @brief Get a lock that updates internal values as well as the state.
     *
     * @versionadded{1.0.0}
     */
    T &get_value() { return _state; }

    /**
     * @brief Lock the state.
     *
     * @versionadded{1.0.1}
     */
    void lock() { _mutex.lock(); }

    /**
     * @brief Unlock the state.
     *
     * This will automatically notify observers when done if no arguments are passed.
     * If false is passed, then it will not notify the observers.
     *
     * @versionadded{1.0.1}
     */
    void unlock(bool notify = true) {
        _mutex.unlock();

        if (notify) {
            notify_observers();
        }
    }

    /**
     * Attempt to lock the object.
     *
     * @return True if the lock could be acquired. False if otherwise.
     *
     * @versionadded{1.0.1}
     */
    bool try_lock() { return _mutex.try_lock(); }

    /**
     * @brief Wait for the value to change.
     *
     * @versionadded{1.0.0}
     */
    void wait_for_change() {
        size_t                       check_changed = _value_changed;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this, check_changed]() { return _value_changed == check_changed; });
    }

    /**
     * @brief Register an observer callback.
     *
     * @versionadded{1.0.0}
     */
    void attach(std::function<void(T const &)> observer) {
        std::scoped_lock lock(_observer_mutex);
        _observers.push_back(std::move(observer));
    }

    /**
     * @brief Remove an observer callback.
     *
     * @versionadded{1.0.0}
     */
    void detach(std::function<void(T const &)> const &observer) {
        std::scoped_lock lock(_observer_mutex);

        _observers.remove_if([&observer](auto const &elem) { return observer.target() == elem.target(); });
    }

    /**
     * @brief Check to see if the observable has recently changed.
     *
     * @versionadded{1.0.0}
     */
    bool changed() const { return _value_changed; }

    /**
     * @brief Public alias for ``notify_observers()``.
     *
     * Allows wrapper types that aggregate several Observables (e.g.
     * ``GlobalConfigMap``) to drive notification *after* releasing every
     * sub-mutex, so an observer's implicit conversion-to-T doesn't
     * re-acquire a mutex while sibling locks are still held. Without this
     * decoupling, the unlock-then-notify-each pattern hits a TSan
     * lock-order inversion as the observer callback's
     * ``operator T() const`` calls ``get_value()`` and re-locks ``_mutex``
     * while sibling Observables in the same aggregate are still locked.
     */
    void notify() { notify_observers(); }

  protected:
    /**
     * @brief Notify all of the observers that observe this observable.
     *
     * @versionadded{1.0.0}
     */
    void notify_observers() {
        // Bump the change counter without re-acquiring _mutex. The earlier
        // version of this method locked _mutex around the increment, but
        // notify_observers is called from unlock(bool) AFTER _mutex was just
        // released — re-acquiring it there created a lock-order risk when the
        // Observable was one element of a `std::scoped_lock` over several
        // Observables (the other siblings' mutexes are still held during the
        // destructor walk, so re-locking this one builds an acquisition order
        // that conflicts with the order `std::lock` used during construction
        // for deadlock avoidance). TSan caught this as a lock-order inversion
        // in RuntimeConfiguration::pre_initialize, which scoped-locks four
        // ConfigMaps at once. Making _value_changed atomic removes the need
        // for the re-acquire entirely.
        _value_changed.fetch_add(1, std::memory_order_release);

        std::scoped_lock lock(_observer_mutex);
        for (auto const &observer : _observers) {
            observer(*this); // Call each registered observer with the new values
        }
    }

    /**
     * @property _state
     *
     * @brief The internal state of the observable.
     *
     * @versionadded{1.0.0}
     */
    T                       _state;
    mutable std::mutex      _mutex{}; ///< For thread-safe value access
    std::condition_variable _cv{};    ///< For thread synchronization
    /// Counter indicating how many times the value has changed. Atomic so
    /// ``notify_observers`` can bump it without re-acquiring ``_mutex`` — the
    /// re-acquire previously created a lock-order risk when the Observable
    /// participated in a multi-mutex ``std::scoped_lock`` (see the comment in
    /// ``notify_observers``).
    std::atomic<size_t> _value_changed{0};

    std::list<std::function<void(T const &)>> _observers{};      ///< List of observers
    std::mutex                                _observer_mutex{}; ///< Protects the observer list
};
} // namespace einsums::design_pats
