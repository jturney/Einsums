//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/coroutine_accessor.hpp>
#include <einsums/coroutines/detail/coroutine_impl.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/functional/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace einsums::threads::coroutines::detail {
class coroutine_self {
  public:
    EINSUMS_NON_COPYABLE(coroutine_self);

  protected:
    // store the current this and write it to the TSS on exit
    struct reset_self_on_exit {
        reset_self_on_exit(coroutine_self *self) : self_(self) { set_self(self->_next_self); }

        ~reset_self_on_exit() { set_self(self_); }

        coroutine_self *self_;
    };

  public:
    using thread_id_type = einsums::threads::detail::thread_id;

    using result_type = std::pair<threads::detail::thread_schedule_state, thread_id_type>;
    using arg_type    = einsums::threads::detail::thread_restart_state;

    using yield_decorator_type = util::detail::function<arg_type(result_type)>;

    explicit coroutine_self(coroutine_self *next_self) : _next_self(next_self) {}

    arg_type yield(result_type arg = result_type()) {
        return !_yield_decorator.empty() ? _yield_decorator(std::move(arg)) : yield_impl(std::move(arg));
    }

    template <typename F>
    yield_decorator_type decorate_yield(F &&f) {
        yield_decorator_type tmp(std::forward<F>(f));
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    yield_decorator_type decorate_yield(yield_decorator_type const &f) {
        yield_decorator_type tmp(f);
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    yield_decorator_type decorate_yield(yield_decorator_type &&f) {
        std::swap(f, _yield_decorator);
        return std::move(f);
    }

    yield_decorator_type undecorate_yield() {
        yield_decorator_type tmp;
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    virtual ~coroutine_self() = default;

    virtual arg_type yield_impl(result_type arg) = 0;

    virtual thread_id_type get_thread_id() const = 0;

    virtual std::size_t get_thread_phase() const = 0;

    virtual std::ptrdiff_t get_available_stack_space() = 0;

    virtual std::size_t get_thread_data() const           = 0;
    virtual std::size_t set_thread_data(std::size_t data) = 0;

    virtual tss_storage *get_thread_tss_data()           = 0;
    virtual tss_storage *get_or_create_thread_tss_data() = 0;

    virtual std::size_t &get_continuation_recursion_count() = 0;

    // access coroutines context object
    using impl_type = coroutine_impl;
    using impl_ptr  = impl_type *;

  private:
    friend struct coroutine_accessor;
    virtual impl_ptr get_impl() { return nullptr; }

  public:
    static EINSUMS_EXPORT coroutine_self *&local_self();

    static void            set_self(coroutine_self *self) { local_self() = self; }
    static coroutine_self *get_self() { return local_self(); }

  private:
    yield_decorator_type _yield_decorator;
    coroutine_self      *_next_self;
};

////////////////////////////////////////////////////////////////////////////
struct reset_self_on_exit {
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    reset_self_on_exit(coroutine_self *val, coroutine_self *old_val = nullptr)
        // NOLINTEND(bugprone-easily-swappable-parameters)
        : old_self(old_val) {
        coroutine_self::set_self(val);
    }

    ~reset_self_on_exit() { coroutine_self::set_self(old_self); }

    coroutine_self *old_self;
};

} // namespace einsums::threads::coroutines::detail
