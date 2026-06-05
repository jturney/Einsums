//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::design_pats {

/**
 * @class Lockable
 *
 * @brief Base class that enables mutexes on an object, making thread safety a breeze.
 *
 * Simply inherit this class and the new class will have everything it needs to satisfy
 * the Lockable requirement. You can even specify what kind of mutex to use to handle
 * the locks.
 *
 * @versionadded{1.0.0}
 */
template <typename Mutex>
class Lockable {
  public:
    /**
     * Default constructor.
     *
     * @versionadded{1.0.0}
     */
    Lockable() = default;

    /**
     * "Copy constructor." It doesn't perform any copies and is only
     * necessary for subclasses so that they can define copy constructors.
     *
     * @versionadded{1.0.0}
     */
    Lockable(Lockable<Mutex> const &) : lock_{} {} // NOLINT(modernize-use-default-member-init)

    /**
     * @brief Lock the object.
     *
     * @versionadded{1.0.0}
     */
    void lock() const { lock_.lock(); }

    /**
     * @brief Try to lock the object. Returns true if successful.
     *
     * @versionadded{1.0.0}
     */
    bool try_lock() const { return lock_.try_lock(); }

    /**
     * @brief Unlock the object.
     */
    void unlock() const { lock_.unlock(); }

    /**
     * @brief Get the underlying mutex.
     *
     * @versionadded{1.0.0}
     */
    Mutex &get_mutex() { return lock_; }

  protected:
    /**
     * @property lock_
     *
     * @brief The underlying locking object. Usually some sort of mutex.
     *
     * @versionadded{1.0.0}
     */
    mutable Mutex lock_{};
};

} // namespace einsums::design_pats
