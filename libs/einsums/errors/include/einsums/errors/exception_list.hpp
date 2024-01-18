//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/exception.hpp>
#include <einsums/thread_support/spinlock.hpp>

#include <cstddef>
#include <exception>
#include <list>
#include <mutex>
#include <string>
#include <system_error>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

namespace einsums {

/// The class exception_list is a container of exception_ptr objects
/// parallel algorithms may use to communicate uncaught exceptions
/// encountered during parallel execution to the caller of the algorithm
///
/// The type exception_list::const_iterator fulfills the requirements of
/// a forward iterator.
///
class EINSUMS_EXPORT exception_list : public einsums::exception {
  private:
    /// \cond NOINTERNAL
    using mutex_type = einsums::detail::spinlock;

    using exception_list_type = std::list<std::exception_ptr>;
    exception_list_type _exceptions;
    mutable mutex_type  _mtx;

    void add_no_lock(std::exception_ptr const &e);
    /// \endcond

  public:
    /// bidirectional iterator
    using iterator = exception_list_type::const_iterator;

    /// \cond NOINTERNAL
    // \throws nothing
    ~exception_list() noexcept = default;

    exception_list();
    explicit exception_list(std::exception_ptr const &e);
    explicit exception_list(exception_list_type &&l);

    exception_list(exception_list const &l);
    exception_list(exception_list &&l) noexcept;

    auto operator=(exception_list const &l) -> exception_list &;
    auto operator=(exception_list &&l) noexcept -> exception_list &;

    ///
    void add(std::exception_ptr const &e);
    /// \endcond

    /// The number of exception_ptr objects contained within the
    /// exception_list.
    ///
    /// \note Complexity: Constant time.
    auto size() const noexcept -> std::size_t {
        std::lock_guard<mutex_type> l(_mtx);
        return _exceptions.size();
    }

    /// An iterator referring to the first exception_ptr object contained
    /// within the exception_list.
    auto begin() const noexcept -> exception_list_type::const_iterator {
        std::lock_guard<mutex_type> l(_mtx);
        return _exceptions.begin();
    }

    /// An iterator which is the past-the-end value for the exception_list.
    auto end() const noexcept -> exception_list_type::const_iterator {
        std::lock_guard<mutex_type> l(_mtx);
        return _exceptions.end();
    }

    /// \cond NOINTERNAL
    auto get_error() const -> std::error_code;

    auto get_message() const -> std::string;
    /// \endcond
};

} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>
