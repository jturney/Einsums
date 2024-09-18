//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

// clang-format off
#include <einsums/coroutines/detail/context_impl.hpp>
// clang-format on

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/swap_context.hpp>
#include <einsums/coroutines/detail/tss.hpp>
#include <einsums/coroutines/thread_id_type.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#define EINSUMS_COROUTINE_NU_ALL_HEAPS                                                                                                     \
    (EINSUMS_COROUTINE_NU_HEAPS + EINSUMS_COROUTINE_NU_HEAPS / 2 + EINSUMS_COROUTINE_NU_HEAPS / 4 + EINSUMS_COROUTINE_NU_HEAPS / 4) /**/

namespace einsums::threads::coroutines::detail {

constexpr std::ptrdiff_t const default_stack_size = -1;

template <typename CoroutineImpl>
struct context_base : public default_context_impl<CoroutineImpl> {
    using base_type = default_context_impl<CoroutineImpl>;

  public:
    using deleter_type   = void(context_base const *);
    using thread_id_type = einsums::threads::detail::thread_id;

    context_base(std::ptrdiff_t stack_size, thread_id_type id)
        : base_type(stack_size), _caller(), _state(ctx_ready), _exit_state(ctx_exit_not_requested), _exit_status(ctx_not_exited)
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
          _type_info(), _thread_id(id), _continuation_recursion_count(0) {
    }

    void reset_tss() {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        delete_tss_storage(_thread_data);
#else
        _thread_data = 0;
#endif
    }

    void reset() {
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        _phase = 0;
#endif
        _thread_id.reset();
    }

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t phase() const { return _phase; }
#endif

    thread_id_type get_thread_id() const { return _thread_id; }

    /*
     * Returns true if the context is runnable.
     */
    bool is_ready() const { return _state == ctx_ready; }

    bool running() const { return _state == ctx_running; }

    bool exited() const { return _state == ctx_exited; }

    void init() { base_type::init(); }

    // Resume coroutine.
    // Pre:  The coroutine must be ready.
    // Post: The coroutine relinquished control. It might be ready
    //       or exited.
    // Throws:- 'abnormal_exit' if the coroutine was exited by another
    //          uncaught exception.
    // Note, it guarantees that the coroutine is resumed. Can throw only
    // on return.
    void invoke() {
        base_type::init();
        EINSUMS_ASSERT(is_ready());
        do_invoke();

        if (_exit_status != ctx_not_exited) {
            if (_exit_status == ctx_exited_return)
                return;
            if (_exit_status == ctx_exited_abnormally) {
                EINSUMS_ASSERT(_type_info);
                std::rethrow_exception(_type_info);
            }
            EINSUMS_ASSERT_MSG(false, "unknown exit status");
        }
    }

    // Put coroutine in ready state and relinquish control
    // to caller until resumed again.
    // Pre:  Coroutine is running.
    //       Exit not pending.
    //       Operations not pending.
    // Post: Coroutine is running.
    // Throws: exit_exception, if exit is pending *after* it has been
    //         resumed.
    void yield() {
        // Misbehaved threads may try to yield while handling an exception.
        // This is dangerous if the thread can migrate to other worker
        // threads since the count for std::uncaught_exceptions may become
        // inconsistent (including negative). If at any point in the future
        // there is a legitimate use case for yielding with uncaught
        // exceptions this assertion can be revisited, but until then we
        // prefer to be strict about it.
        EINSUMS_ASSERT(std::uncaught_exceptions() == 0);

        // prevent infinite loops
        EINSUMS_ASSERT(_exit_state < ctx_exit_signaled);
        EINSUMS_ASSERT(running());

        _state = ctx_ready;
#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->start_yield_fiber(&this->asan_fake_stack, _caller);
#endif
        do_yield();

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->finish_yield_fiber(this->asan_fake_stack);
#endif
        _exit_status = ctx_not_exited;

        EINSUMS_ASSERT(running());
    }

    // Nothrow.
    ~context_base() noexcept {
        EINSUMS_ASSERT(!running());
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        EINSUMS_ASSERT(exited() || (is_ready() && _phase == 0));
#else
        EINSUMS_ASSERT(exited() || is_ready());
#endif
        _thread_id.reset();
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        delete_tss_storage(_thread_data);
#else
        _thread_data = 0;
#endif
    }

    std::size_t get_thread_data() const {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        if (!_thread_data)
            return 0;
        return get_tss_thread_data(_thread_data);
#else
        return _thread_data;
#endif
    }

    std::size_t set_thread_data(std::size_t data) {
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

    std::size_t &get_continuation_recursion_count() { return _continuation_recursion_count; }

    enum context_state {
        ctx_running = 0, // context running
        ctx_ready,       // context at yield point
        ctx_exited       // context is finishe
    };

  protected:
    enum context_exit_state {
        ctx_exit_not_requested = 0, // exit not requested
        ctx_exit_pending,           // exit requested
        ctx_exit_signaled           // exit request delivered
    };

    enum context_exit_status {
        ctx_not_exited,
        ctx_exited_return,    // process exited by return.
        ctx_exited_abnormally // process exited uncleanly.
    };

    void rebind_base(thread_id_type id) {
        EINSUMS_ASSERT(!running());

        _thread_id   = id;
        _state       = ctx_ready;
        _exit_state  = ctx_exit_not_requested;
        _exit_status = ctx_not_exited;
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        EINSUMS_ASSERT(_phase == 0);
#endif
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_thread_data == nullptr);
#else
        EINSUMS_ASSERT(_thread_data == 0);
#endif
        // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
        _type_info = std::exception_ptr();
    }

    // Nothrow.
    void do_return(context_exit_status status, std::exception_ptr &&info) noexcept {
        EINSUMS_ASSERT(status != ctx_not_exited);
        EINSUMS_ASSERT(_state == ctx_running);
        _type_info   = std::move(info);
        _state       = ctx_exited;
        _exit_status = status;
#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->start_yield_fiber(&this->asan_fake_stack, _caller);
#endif

        do_yield();
    }

  protected:
    void do_yield() noexcept { swap_context(*this, _caller, detail::yield_hint()); }

    void do_invoke() noexcept {
        EINSUMS_ASSERT(is_ready());
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        ++_phase;
#endif
        _state = ctx_running;

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->start_switch_fiber(&this->asan_fake_stack);
#endif

        swap_context(_caller, *this, detail::invoke_hint());

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->finish_switch_fiber(this->asan_fake_stack, _caller);
#endif
    }

    using ctx_type = typename base_type::context_impl_base;
    ctx_type _caller;

    context_state       _state;
    context_exit_state  _exit_state;
    context_exit_status _exit_status;
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t _phase;
#endif
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
    mutable detail::tss_storage *_thread_data;
#else
    mutable std::size_t _thread_data;
#endif

    // This is used to generate a meaningful exception trace.
    std::exception_ptr _type_info;
    thread_id_type     _thread_id;

    std::size_t _continuation_recursion_count;
};

} // namespace einsums::threads::coroutines::detail