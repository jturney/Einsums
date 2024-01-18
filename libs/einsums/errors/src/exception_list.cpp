//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/errors/exception.hpp>
#include <einsums/errors/exception_list.hpp>
#include <einsums/thread_support/unlock_guard.hpp>

#include <exception>
#include <mutex>
#include <string>
#include <system_error>

namespace einsums {
namespace detail {

auto indent_message(std::string const &msg_) -> std::string {
    std::string            result;
    std::string const     &msg(msg_);
    std::string::size_type pos          = msg.find_first_of('\n');
    std::string::size_type first_non_ws = msg.find_first_not_of(" \n");
    std::string::size_type pos1         = 0;

    while (std::string::npos != pos) {
        if (pos > first_non_ws) {
            result += msg.substr(pos1, pos - pos1 + 1);
            pos = msg.find_first_of('\n', pos1 = pos + 1);
            if (std::string::npos != pos) {
                result += "  ";
            }
        } else {
            pos = msg.find_first_of('\n', pos1 = pos + 1);
        }
    }

    result += msg.substr(pos1);
    return result;
}

} // namespace detail

error_code throws; // "throw on error" special error_code

exception_list::exception_list() : einsums::exception(einsums::error::success), _mtx() {
}

exception_list::exception_list(std::exception_ptr const &e)
    : einsums::exception(einsums::get_error(e), einsums::get_error_what(e)), _mtx() {
    add_no_lock(e);
}

exception_list::exception_list(exception_list_type &&l)
    : einsums::exception(!l.empty() ? einsums::get_error(l.front()) : einsums::error::success),
      _exceptions(EINSUMS_MOVE(l)), _mtx() {
}

exception_list::exception_list(exception_list const &l)
    : einsums::exception(static_cast<einsums::exception const &>(l)), _exceptions(l._exceptions), _mtx() {
}

exception_list::exception_list(exception_list &&l) noexcept
    : einsums::exception(EINSUMS_MOVE(static_cast<einsums::exception &>(l))), _exceptions(EINSUMS_MOVE(l._exceptions)),
      _mtx() {
}

auto exception_list::operator=(exception_list const &l) -> exception_list & {
    if (this != &l) {
        *static_cast<einsums::exception *>(this) = static_cast<einsums::exception const &>(l);
        _exceptions                              = l._exceptions;
    }
    return *this;
}

auto exception_list::operator=(exception_list &&l) noexcept -> exception_list & {
    if (this != &l) {
        static_cast<einsums::exception &>(*this) = EINSUMS_MOVE(static_cast<einsums::exception &>(l));
        _exceptions                              = EINSUMS_MOVE(l._exceptions);
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////
auto exception_list::get_error() const -> std::error_code {
    std::lock_guard<mutex_type> l(_mtx);
    if (_exceptions.empty())
        return einsums::error::no_success;
    return einsums::get_error(_exceptions.front());
}

auto exception_list::get_message() const -> std::string {
    std::lock_guard<mutex_type> l(_mtx);
    if (_exceptions.empty())
        return "";

    if (1 == _exceptions.size())
        return einsums::get_error_what(_exceptions.front());

    std::string result("\n");

    auto end = _exceptions.end();
    auto it  = _exceptions.begin();
    for (/**/; it != end; ++it) {
        result += "  ";
        result += detail::indent_message(einsums::get_error_what(*it));
        if (result.find_last_of('\n') < result.size() - 1)
            result += "\n";
    }
    return result;
}

void exception_list::add(std::exception_ptr const &e) {
    std::unique_lock<mutex_type> l(_mtx);
    if (_exceptions.empty()) {
        einsums::exception ex;
        {
            detail::unlock_guard<std::unique_lock<mutex_type>> ul(l);
            ex = einsums::exception(einsums::get_error(e));
        }

        // set the error code for our base class
        static_cast<einsums::exception &>(*this) = ex;
    }
    _exceptions.push_back(e);
}

void exception_list::add_no_lock(std::exception_ptr const &e) {
    if (_exceptions.empty()) {
        // set the error code for our base class
        static_cast<einsums::exception &>(*this) = einsums::exception(einsums::get_error(e));
    }
    _exceptions.push_back(e);
}

} // namespace einsums