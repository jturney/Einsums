//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <deque>
#include <exception>
#include <functional>

namespace einsums::threads {

struct EINSUMS_EXPORT CallbackNotifier {
    using OnStartStopType = std::function<void(std::size_t, std::size_t, char const *, char const *)>;
    using OnErrorType     = std::function<bool(std::size_t, std::exception_ptr const &)>;

    CallbackNotifier() = default;

    void on_start_thread(std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name, char const *postfix) const {
        for (auto &callback : _on_start_thread_callbacks) {
            if (callback) {
                callback(local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    void on_stop_thread(std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name, char const *postfix) const {
        for (auto &callback : _on_stop_thread_callbacks) {
            if (callback) {
                callback(local_thread_num, global_thread_num, pool_name, postfix);
            }
        }
    }

    bool on_error(std::size_t global_thread_num, std::exception_ptr const &e) const {
        if (_on_error) {
            return _on_error(global_thread_num, e);
        }
        return true;
    }

    void add_on_start_thread_callback(OnStartStopType callback) { _on_start_thread_callbacks.emplace_back(std::move(callback)); }

    void add_on_stop_thread_callback(OnStartStopType callback) { _on_stop_thread_callbacks.emplace_back(std::move(callback)); }

    void set_on_error_callback(OnErrorType callback) { _on_error = callback; }

    /// Functions to call for each created thread
    std::deque<OnStartStopType> _on_start_thread_callbacks;
    /// Functions to call in case of unexpected stop
    std::deque<OnStartStopType> _on_stop_thread_callbacks;
    /// Functions to call in case of error
    OnErrorType _on_error;
};

} // namespace einsums::threads