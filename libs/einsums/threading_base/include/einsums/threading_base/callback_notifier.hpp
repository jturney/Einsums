//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <deque>
#include <exception>
#include <functional>

namespace einsums::threads {

struct EINSUMS_EXPORT callback_notifier {
    using on_startstop_type = std::function<void(std::size_t, std::size_t, char const *, char const *)>;
    using on_error_type     = std::function<bool(std::size_t, std::exception_ptr const &)>;

    callback_notifier() = default;

    void on_start_thread(std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name, char const *postfix) const {
        for (auto &callback : on_start_thread_callbacks) {
            if (callback) {
                callback(local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    void on_stop_thread(std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name, char const *postfix) const {
        for (auto &callback : on_stop_thread_callbacks) {
            if (callback) {
                callback(local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    bool on_error(std::size_t global_thread_num, std::exception_ptr const &e) const {
        if (on_error_callback) {
            return on_error_callback(global_thread_num, e);
        }
        return true;
    }

    void add_on_start_thread_callback(on_startstop_type const &callback) { on_start_thread_callbacks.push_back(callback); }

    void add_on_stop_thread_callback(on_startstop_type const &callback) { on_stop_thread_callbacks.push_front(callback); }

    void set_on_error_callback(on_error_type const &callback) { on_error_callback = callback; }

    // Functions to call for each created thread
    std::deque<on_startstop_type> on_start_thread_callbacks;
    // Functions to call in case of unexpected stop
    std::deque<on_startstop_type> on_stop_thread_callbacks;
    // Function to call in case of error
    on_error_type on_error_callback;
};

} // namespace einsums::threads
