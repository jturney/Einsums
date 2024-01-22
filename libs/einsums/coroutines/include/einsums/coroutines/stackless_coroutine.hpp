//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/coroutine.hpp>
#include <einsums/coroutines/detail/coroutine_self.hpp>
#include <einsums/coroutines/detail/tss.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/functional/detail/reset_function.hpp>
#include <einsums/functional/unique_function.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <utility>

namespace einsums::threads::coroutines::detail {
class stackless_coroutine {
  private:
    enum context_state {
        ctx_running, // context running.
        ctx_ready,   // context at yield point.
        ctx_exited   // context is finished.
    };

    static constexpr std::ptrdiff_t default_stack_size = -1;

    auto running() const -> bool { return _state == ctx_running; }

    auto exited() const -> bool { return _state == ctx_exited; }

  public:
    friend struct coroutine_accessor;

    using thread_id_type = einsums::threads::detail::thread_id;

    using result_type = std::pair<threads::detail::thread_schedule_state, thread_id_type>;
    using arg_type    = threads::detail::thread_restart_state;

    using functor_type = util::detail::unique_function<result_type(arg_type)>;

    stackless_coroutine(functor_type &&f, thread_id_type id, std::ptrdiff_t /*stack_size*/ = default_stack_size)
        : _f(EINSUMS_MOVE(f)), _state(ctx_ready), _id(id)
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
          ,
          _phase(0)
#endif
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
          ,
          _thread_data(nullptr)
#else
          ,
          _thread_data(0)
#endif
          ,
          _continuation_recursion_count(0) {
    }

    ~stackless_coroutine() {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        delete_tss_storage(_thread_data);
#else
        _thread_data = 0;
#endif
    }

    stackless_coroutine(stackless_coroutine const &src)                     = delete;
    auto operator=(stackless_coroutine const &src) -> stackless_coroutine & = delete;
    stackless_coroutine(stackless_coroutine &&src)                          = delete;
    auto operator=(stackless_coroutine &&src) -> stackless_coroutine      & = delete;

    auto get_thread_id() const -> thread_id_type { return _id; }

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t get_thread_phase() const { return _phase; }
#endif
    auto get_thread_data() const -> std::size_t {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        if (!_thread_data)
            return 0;
        return get_tss_thread_data(_thread_data);
#else
        return _thread_data;
#endif
    }

    auto set_thread_data(std::size_t data) -> std::size_t {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        return set_tss_thread_data(_thread_data, data);
#else
        std::size_t olddata = _thread_data;
        _thread_data        = data;
        return olddata;
#endif
    }

#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
    tss_storage *get_thread_tss_data(bool create_if_needed) const {
        if (!_thread_data && create_if_needed)
            _thread_data = create_tss_storage();
        return _thread_data;
    }
#endif

    void rebind(functor_type &&f, thread_id_type id) {
        EINSUMS_ASSERT(exited());

        _f  = EINSUMS_MOVE(f);
        _id = id;

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        _phase = 0;
#endif
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_thread_data == nullptr);
#else
        EINSUMS_ASSERT(_thread_data == 0);
#endif
        _state = stackless_coroutine::ctx_ready;
    }

    void reset_tss() {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        delete_tss_storage(_thread_data);
#else
        _thread_data = 0;
#endif
    }

    void reset() {
        EINSUMS_ASSERT(exited());

        util::detail::reset_function(_f);

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        phase_ = 0;
#endif
        _id.reset();
    }

  private:
    struct reset_on_exit {
        reset_on_exit(stackless_coroutine &self) : this_(self) { this_._state = stackless_coroutine::ctx_running; }

        ~reset_on_exit() { this_._state = stackless_coroutine::ctx_exited; }
        stackless_coroutine &this_;
    };
    friend struct reset_on_exit;

  public:
    EINSUMS_FORCEINLINE result_type operator()(arg_type arg = arg_type());

    explicit operator bool() const { return !exited(); }

    auto is_ready() const -> bool { return _state == ctx_ready; }

    auto get_available_stack_space() -> std::ptrdiff_t { return (std::numeric_limits<std::ptrdiff_t>::max)(); }

    auto get_continuation_recursion_count() -> std::size_t & { return _continuation_recursion_count; }

  protected:
    functor_type   _f;
    context_state  _state;
    thread_id_type _id;

#ifdef EINSUMS_HAVE_THREAD_PHASE_INFORMATION
    std::size_t _phase;
#endif
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
    mutable tss_storage *_thread_data;
#else
    mutable std::size_t _thread_data;
#endif
    std::size_t _continuation_recursion_count;
};
} // namespace einsums::threads::coroutines::detail

////////////////////////////////////////////////////////////////////////////////
#include <einsums/coroutines/detail/coroutine_stackless_self.hpp>

namespace einsums::threads::coroutines::detail {
EINSUMS_FORCEINLINE stackless_coroutine::result_type stackless_coroutine::operator()(arg_type arg) {
    EINSUMS_ASSERT(is_ready());

    result_type result(threads::detail::thread_schedule_state::terminated, threads::detail::invalid_thread_id);

    {
        coroutine_stackless_self self(this);
        reset_self_on_exit       on_self_exit(&self, nullptr);

        {
            [[maybe_unused]] reset_on_exit on_exit{*this};

            result = _f(arg); // invoke wrapped function

            // we always have to run to completion
            EINSUMS_ASSERT(result.first == threads::detail::thread_schedule_state::terminated);
        }

        reset_tss();
        reset();
    }

    return result;
}
} // namespace einsums::threads::coroutines::detail
