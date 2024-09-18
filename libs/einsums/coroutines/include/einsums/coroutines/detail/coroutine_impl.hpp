//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(push)
#    pragma warning(disable : 4355) // this used in base member initializer
#endif

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/coroutine_fwd.hpp>
#include <einsums/coroutines/detail/context_base.hpp>
#include <einsums/coroutines/detail/coroutine_accessor.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/functional/unique_function.hpp>

#include <cstddef>
#include <utility>

namespace einsums::threads::coroutines::detail {

///////////////////////////////////////////////////////////////////////////
// This type augments the context_base type with the type of the stored
// functor.
class coroutine_impl : public context_base<coroutine_impl> {
  public:
    EINSUMS_NON_COPYABLE(coroutine_impl);

  public:
    using super_type     = context_base;
    using thread_id_type = context_base::thread_id_type;

    using result_type = std::pair<threads::detail::thread_schedule_state, thread_id_type>;
    using arg_type    = einsums::threads::detail::thread_restart_state;

    using functor_type = util::detail::unique_function<result_type(arg_type)>;

    coroutine_impl(functor_type &&f, thread_id_type id, std::ptrdiff_t stack_size)
        : context_base(stack_size, id), _result(threads::detail::thread_schedule_state::unknown, threads::detail::invalid_thread_id),
          _arg(nullptr), _fun(std::move(f)) {}

#if defined(EINSUMS_DEBUG)
    EINSUMS_EXPORT ~coroutine_impl();
#endif

    // execute the coroutine using normal context switching
    EINSUMS_EXPORT void operator()() noexcept;

  public:
    void bind_result(result_type res) {
        EINSUMS_ASSERT(_result.first != threads::detail::thread_schedule_state::terminated);
        _result = res;
    }

    result_type result() const { return _result; }
    arg_type   *args() noexcept {
        EINSUMS_ASSERT(_arg);
        return _arg;
    };

    void bind_args(arg_type *arg) noexcept { _arg = arg; }

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t get_thread_phase() const { return this->phase(); }
#endif

    void init() { this->super_type::init(); }

    void reset() {
        // First reset the function and arguments
        _arg = nullptr;
        _fun.reset();

        // Then reset the id and stack as they may be used by the
        // destructors of the thread function above
        this->super_type::reset();
        this->reset_stack();
    }

    void rebind(functor_type &&f, thread_id_type id) {
        EINSUMS_ASSERT(_result.first == threads::detail::thread_schedule_state::unknown ||
                       _result.first == threads::detail::thread_schedule_state::terminated);
        this->rebind_stack(); // count how often a coroutines object was reused
        _result = result_type(threads::detail::thread_schedule_state::unknown, threads::detail::invalid_thread_id);
        _arg    = nullptr;
        _fun    = std::move(f);
        this->super_type::rebind_base(id);
    }

  private:
    result_type  _result;
    arg_type    *_arg;
    functor_type _fun;
};
} // namespace einsums::threads::coroutines::detail

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(pop)
#endif
