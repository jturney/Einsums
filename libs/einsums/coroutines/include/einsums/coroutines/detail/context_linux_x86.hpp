//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
#    include <einsums/config.hpp>

#    include <einsums/assert.hpp>
#    include <einsums/core/get_and_reset_value.hpp>
#    include <einsums/coroutines/detail/get_stack_pointer.hpp>
#    include <einsums/coroutines/detail/posix_utility.hpp>
#    include <einsums/coroutines/detail/swap_context.hpp>

#    include <fmt/format.h>

#    include <atomic>
#    include <cstddef>
#    include <cstdint>
#    include <cstdlib>
#    include <stdexcept>
#    include <sys/param.h>

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION)

#        include <cstring>
#        include <signal.h>
#        include <stdlib.h>
#        include <strings.h>

#        if !defined(SEGV_STACK_SIZE)
#            define SEGV_STACK_SIZE MINSIGSTKSZ + 4096
#        endif

#    endif

#    include <iomanip>
#    include <iostream>

#    if defined(EINSUMS_HAVE_VALGRIND)
#        if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#            if defined(EINSUMS_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#                pragma GCC diagnostic push
#            endif
#            pragma GCC diagnostic ignored "-Wpointer-arith"
#        endif
#        include <valgrind/valgrind.h>
#    endif

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
#        include <sanitizer/asan_interface.h>
#    endif

/*
 * Defining EINSUMS_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate
 * invoke, and yield swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

#    if defined(__x86_64__)
extern "C" void swapcontext_stack(void ***, void **) noexcept;
extern "C" void swapcontext_stack2(void ***, void **) noexcept;
#    else
extern "C" void swapcontext_stack(void ***, void **) noexcept __attribute((regparm(2)));
extern "C" void swapcontext_stack2(void ***, void **) noexcept __attribute((regparm(2)));
#    endif

///////////////////////////////////////////////////////////////////////////////
namespace einsums::threads::coroutines {
namespace detail {
// some platforms need special preparation of the main thread
struct prepare_main_thread {
    constexpr prepare_main_thread() = default;
};
} // namespace detail

namespace detail::lx {
template <typename T>
EINSUMS_FORCEINLINE void trampoline(void *fun) {
    (*static_cast<T *>(fun))();
    std::abort();
}

template <typename CoroutineImpl>
class x86_linux_context_impl;

class x86_linux_context_impl_base : detail::context_impl_base {
  public:
    x86_linux_context_impl_base()
        : _sp(nullptr)
#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
          ,
          asan_fake_stack(nullptr), asan_stack_bottom(nullptr), asan_stack_size(0)
#    endif
    {
    }

    void prefetch() const {
#    if defined(__x86_64__)
        EINSUMS_ASSERT(sizeof(void *) == 8);
#    else
        EINSUMS_ASSERT(sizeof(void *) == 4);
#    endif

        __builtin_prefetch(_sp, 1, 3);
        __builtin_prefetch(_sp, 0, 3);
        __builtin_prefetch(static_cast<void **>(_sp) + 64 / sizeof(void *), 1, 3);
        __builtin_prefetch(static_cast<void **>(_sp) + 64 / sizeof(void *), 0, 3);
#    if !defined(__x86_64__)
        __builtin_prefetch(static_cast<void **>(m_sp) + 32 / sizeof(void *), 1, 3);
        __builtin_prefetch(static_cast<void **>(m_sp) + 32 / sizeof(void *), 0, 3);
        __builtin_prefetch(static_cast<void **>(m_sp) - 32 / sizeof(void *), 1, 3);
        __builtin_prefetch(static_cast<void **>(m_sp) - 32 / sizeof(void *), 0, 3);
#    endif
        __builtin_prefetch(static_cast<void **>(_sp) - 64 / sizeof(void *), 1, 3);
        __builtin_prefetch(static_cast<void **>(_sp) - 64 / sizeof(void *), 0, 3);
    }

    /**
     * Free function. Saves the current context in @p from
     * and restores the context in @p to.
     * @note This function is found by ADL.
     */
    friend void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, default_hint);

    friend void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, yield_hint);

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
    void start_switch_fiber(void **fake_stack) {
        __sanitizer_start_switch_fiber(fake_stack, asan_stack_bottom, asan_stack_size);
    }
    void start_yield_fiber(void **fake_stack, x86_linux_context_impl_base &caller) {
        __sanitizer_start_switch_fiber(fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
    }
    void finish_yield_fiber(void *fake_stack) {
        __sanitizer_finish_switch_fiber(fake_stack, &asan_stack_bottom, &asan_stack_size);
    }
    void finish_switch_fiber(void *fake_stack, x86_linux_context_impl_base &caller) {
        __sanitizer_finish_switch_fiber(fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
    }
#    endif

  protected:
    void **_sp;

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
  public:
    void       *asan_fake_stack;
    void const *asan_stack_bottom;
    std::size_t asan_stack_size;
#    endif
};

template <typename CoroutineImpl>
class x86_linux_context_impl : public x86_linux_context_impl_base {
  public:
    enum { default_stack_size = 4 * EXEC_PAGESIZE };

    using context_impl_base = x86_linux_context_impl_base;

    /**
     * Create a context that on restore invokes Functor on
     *  a new stack. The stack size can be optionally specified.
     */
    explicit x86_linux_context_impl(std::ptrdiff_t stack_size = -1)
        : _stack_size(stack_size == -1 ? static_cast<std::ptrdiff_t>(default_stack_size) : stack_size),
          _stack(nullptr) {}

    void init() {
        if (_stack != nullptr)
            return;

        if (0 != (_stack_size % EXEC_PAGESIZE)) {
            throw std::runtime_error(
                fmt::format("stack size of {} is not page aligned, page size is {}", _stack_size, EXEC_PAGESIZE));
        }

        if (0 >= _stack_size) {
            throw std::runtime_error(fmt::format("stack size of {} is invalid", _stack_size));
        }

        _stack = posix::alloc_stack(static_cast<std::size_t>(_stack_size));
        if (_stack == nullptr) {
            throw std::runtime_error("could not allocate memory for stack");
        }

        posix::watermark_stack(_stack, static_cast<std::size_t>(_stack_size));

        using fun = void(void *);
        fun *funp = trampoline<CoroutineImpl>;

        _sp = (static_cast<void **>(_stack) + static_cast<std::size_t>(_stack_size) / sizeof(void *)) - context_size;

        _sp[cb_idx]   = this;
        _sp[funp_idx] = reinterpret_cast<void *>(funp);

#    if defined(EINSUMS_HAVE_VALGRIND) && !defined(NVALGRIND)
        {
            void *eos             = static_cast<char *>(m_stack) + m_stack_size;
            m_sp[valgrind_id_idx] = reinterpret_cast<void *>(VALGRIND_STACK_REGISTER(m_stack, eos));
        }
#    endif
#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        asan_stack_size   = m_stack_size;
        asan_stack_bottom = const_cast<void const *>(m_stack);
#    endif

        set_sigsegv_handler();
    }

    ~x86_linux_context_impl() {
        if (_stack) {
#    if defined(EINSUMS_HAVE_VALGRIND) && !defined(NVALGRIND)
            VALGRIND_STACK_DEREGISTER(reinterpret_cast<std::size_t>(m_sp[valgrind_id_idx]));
#    endif
            posix::free_stack(_stack, static_cast<std::size_t>(_stack_size));
        }

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        free(segv_stack.ss_sp);
#    endif
    }

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)

// heuristic value 1 kilobyte
#        define COROUTINE_STACKOVERFLOW_ADDR_EPSILON 1000UL

    static void check_coroutine_stack_overflow(siginfo_t *infoptr, void *ctxptr) {
        ucontext_t *uc_ctx      = static_cast<ucontext_t *>(ctxptr);
        char       *sigsegv_ptr = static_cast<char *>(infoptr->si_addr);

        // https://www.gnu.org/software/libc/manual/html_node/Signal-Stack.html
        //
        char *stk_ptr = static_cast<char *>(uc_ctx->uc_stack.ss_sp);

        std::ptrdiff_t addr_delta = (sigsegv_ptr > stk_ptr) ? (sigsegv_ptr - stk_ptr) : (stk_ptr - sigsegv_ptr);

        // check the stack addresses, if they're < 10 apart, terminate
        // program should filter segmentation faults caused by
        // coroutine stack overflows from 'genuine' stack overflows
        //
        if (static_cast<size_t>(addr_delta) < COROUTINE_STACKOVERFLOW_ADDR_EPSILON) {
            std::cerr << "Stack overflow in coroutine at address " << std::internal << std::hex
                      << std::setw(sizeof(sigsegv_ptr) * 2 + 2) << std::setfill('0') << sigsegv_ptr << ".\n\n";

            std::cerr << "Configure the einsums runtime to allocate a larger coroutine stack "
                         "size.\n Use the einsums.stacks.small_size, einsums.stacks.medium_size,\n "
                         "einsums.stacks.large_size, or einsums.stacks.huge_size configuration\nflags "
                         "to configure coroutine stack sizes.\n"
                      << std::endl;
        }
    }

    static void sigsegv_handler(int signum, siginfo_t *infoptr, void *ctxptr) {
        char *reason = strsignal(signum);
        std::cerr << "{what}: " << (reason ? reason : "Unknown signal") << std::endl;

        check_coroutine_stack_overflow(infoptr, ctxptr);

        std::abort();
    }
#    endif

    // Return the size of the reserved stack address space.
    auto get_stacksize() const -> std::ptrdiff_t { return _stack_size; }

    void reset_stack() {
        EINSUMS_ASSERT(m_stack);
        if (posix::reset_stack(_stack, static_cast<std::size_t>(_stack_size))) {
#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
            increment_stack_unbind_count();
#    endif
        }
    }

    void rebind_stack() {
        EINSUMS_ASSERT(m_stack);
#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
        increment_stack_recycle_count();
#    endif

        // On rebind, we initialize our stack to ensure a virgin stack
        _sp = (static_cast<void **>(_stack) + static_cast<std::size_t>(_stack_size) / sizeof(void *)) - context_size;

        using fun     = void(void *);
        fun *funp     = trampoline<CoroutineImpl>;
        _sp[cb_idx]   = this;
        _sp[funp_idx] = reinterpret_cast<void *>(funp);
#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        asan_stack_size   = m_stack_size;
        asan_stack_bottom = const_cast<void const *>(m_stack);
#    endif
    }

    auto get_available_stack_space() -> std::ptrdiff_t {
        return get_stack_ptr() - reinterpret_cast<std::size_t>(_stack) - context_size;
    }

    using counter_type = std::atomic<std::int64_t>;

#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
  private:
    static counter_type &get_stack_unbind_counter() {
        static counter_type counter(0);
        return counter;
    }

    static counter_type &get_stack_recycle_counter() {
        static counter_type counter(0);
        return counter;
    }

    static std::uint64_t increment_stack_unbind_count() { return ++get_stack_unbind_counter(); }

    static std::uint64_t increment_stack_recycle_count() { return ++get_stack_recycle_counter(); }

  public:
    static std::uint64_t get_stack_unbind_count(bool reset) {
        return ::einsums::detail::get_and_reset_value(get_stack_unbind_counter(), reset);
    }

    static std::uint64_t get_stack_recycle_count(bool reset) {
        return ::einsums::detail::get_and_reset_value(get_stack_recycle_counter(), reset);
    }
#    endif

    friend void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, default_hint);

    friend void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, yield_hint);

  private:
    void set_sigsegv_handler() {
#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        // concept inspired by the following links:
        //
        // https://rethinkdb.com/blog/handling-stack-overflow-on-custom-stacks/
        // http://www.evanjones.ca/software/threading.html
        //
        segv_stack.ss_sp    = valloc(SEGV_STACK_SIZE);
        segv_stack.ss_flags = 0;
        segv_stack.ss_size  = SEGV_STACK_SIZE;

        std::memset(&action, '\0', sizeof(action));
        action.sa_flags     = SA_SIGINFO | SA_ONSTACK;
        action.sa_sigaction = &x86_linux_context_impl::sigsegv_handler;

        sigaltstack(&segv_stack, nullptr);
        sigemptyset(&action.sa_mask);
        sigaddset(&action.sa_mask, SIGSEGV);
        sigaction(SIGSEGV, &action, nullptr);
#    endif
    }

#    if defined(__x86_64__)
    /** structure of context_data:
     * 11: additional alignment (or valgrind_id if enabled)
     * 10: parm 0 of trampoline
     * 9:  dummy return address for trampoline
     * 8:  return addr (here: start addr)
     * 7:  rbp
     * 6:  rbx
     * 5:  rsi
     * 4:  rdi
     * 3:  r12
     * 2:  r13
     * 1:  r14
     * 0:  r15
     **/
#        if defined(EINSUMS_HAVE_VALGRIND) && !defined(NVALGRIND)
    static std::size_t const valgrind_id_idx = 11;
#        endif

    static std::size_t const context_size = 12;
    static std::size_t const cb_idx       = 10;
    static std::size_t const funp_idx     = 8;
#    else
    /** structure of context_data:
     * 7: valgrind_id (if enabled)
     * 6: parm 0 of trampoline
     * 5: dummy return address for trampoline
     * 4: return addr (here: start addr)
     * 3: ebp
     * 2: ebx
     * 1: esi
     * 0: edi
     **/
#        if defined(EINSUMS_HAVE_VALGRIND) && !defined(NVALGRIND)
    static std::size_t const context_size    = 8;
    static std::size_t const valgrind_id_idx = 7;
#        else
    static std::size_t const context_size = 7;
#        endif

    static std::size_t const cb_idx   = 6;
    static std::size_t const funp_idx = 4;
#    endif

    std::ptrdiff_t _stack_size;
    void          *_stack;

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
    struct sigaction action;
    stack_t          segv_stack;
#    endif
};

/**
 * Free function. Saves the current context in @p from
 * and restores the context in @p to.
 * @note This function is found by ADL.
 */
inline void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, default_hint) {
    //        EINSUMS_ASSERT(*(void**)to.m_stack == (void*)~0);
    to.prefetch();
    swapcontext_stack(&from._sp, to._sp);
}

inline void swap_context(x86_linux_context_impl_base &from, x86_linux_context_impl_base const &to, yield_hint) {
    //        EINSUMS_ASSERT(*(void**)from.m_stack == (void*)~0);
    to.prefetch();
#    if !defined(EINSUMS_COROUTINE_NO_SEPARATE_CALL_SITES)
    swapcontext_stack2(&from._sp, to._sp);
#    else
    swapcontext_stack(&from.m_sp, to.m_sp);
#    endif
}
} // namespace detail::lx
} // namespace einsums::threads::coroutines

#    if defined(EINSUMS_HAVE_VALGRIND)
#        if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#            if defined(EINSUMS_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#                pragma GCC diagnostic pop
#            endif
#        endif
#    endif

#else

#    error This header can only be included when compiling for linux systems.

#endif