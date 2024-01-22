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
    using arg_type    = threads::detail::thread_restart_state;

    using yield_decorator_type = util::detail::function<arg_type(result_type)>;

    explicit coroutine_self(coroutine_self *next_self) : _next_self(next_self) {}

    auto yield(result_type arg = result_type()) -> arg_type {
        return !_yield_decorator.empty() ? _yield_decorator(EINSUMS_MOVE(arg)) : yield_impl(EINSUMS_MOVE(arg));
    }

    template <typename F>
    auto decorate_yield(F &&f) -> yield_decorator_type {
        yield_decorator_type tmp(EINSUMS_FORWARD(F, f));
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    auto decorate_yield(yield_decorator_type const &f) -> yield_decorator_type {
        yield_decorator_type tmp(f);
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    auto decorate_yield(yield_decorator_type &&f) -> yield_decorator_type {
        std::swap(f, _yield_decorator);
        return EINSUMS_MOVE(f);
    }

    auto undecorate_yield() -> yield_decorator_type {
        yield_decorator_type tmp;
        std::swap(tmp, _yield_decorator);
        return tmp;
    }

    virtual ~coroutine_self() = default;

    virtual auto yield_impl(result_type arg) -> arg_type = 0;

    virtual auto get_thread_id() const -> thread_id_type = 0;

    virtual auto get_thread_phase() const -> std::size_t = 0;

    virtual auto get_available_stack_space() -> std::ptrdiff_t = 0;

    virtual auto get_thread_data() const -> std::size_t           = 0;
    virtual auto set_thread_data(std::size_t data) -> std::size_t = 0;

    virtual auto get_thread_tss_data() -> tss_storage           * = 0;
    virtual auto get_or_create_thread_tss_data() -> tss_storage * = 0;

    virtual auto get_continuation_recursion_count() -> std::size_t & = 0;

    // access coroutines context object
    using impl_type = coroutine_impl;
    using impl_ptr  = impl_type *;

  private:
    friend struct coroutine_accessor;
    virtual auto get_impl() -> impl_ptr { return nullptr; }

  public:
    static EINSUMS_EXPORT coroutine_self *&local_self();

    static void set_self(coroutine_self *self) { local_self() = self; }
    static auto get_self() -> coroutine_self * { return local_self(); }

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
