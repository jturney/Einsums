//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

// NOTE (per http://lists.apple.com/archives/darwin-dev/2008/Jan/msg00232.html):
// > Why the bus error? What am I doing wrong?
// This is a known issue where getcontext(3) is writing past the end of the
// ucontext_t struct when _XOPEN_SOURCE is not defined (rdar://problem/5578699
// ). As a workaround, define _XOPEN_SOURCE before including ucontext.h.
#if defined(__APPLE__) && !defined(_XOPEN_SOURCE)
#    define _XOPEN_SOURCE
// However, the above #define will only affect <ucontext.h> if it has not yet
// been #included by something else!
#    if defined(_STRUCT_UCONTEXT)
// #        error You must #include coroutine headers before anything that #includes <ucontext.h>
#    endif
#endif

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/type_support/unused.hpp>
#include <einsums/util/get_and_reset_value.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have
// the unistd header...
#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
#    include <sanitizer/asan_interface.h>
#endif

#if defined(__FreeBSD__) || (defined(_XOPEN_UNIX) && defined(_XOPEN_VERSION) && _XOPEN_VERSION >= 500) || defined(__bgq__) ||              \
    defined(__powerpc__) || defined(__s390x__)

// OS X 10.4 -- despite passing the test above -- doesn't support
// swapcontext() et al. Use GNU Pth workalike functions.
#    if defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 1050)

#        include <cerrno>
#        include <cstddef>
#        include <cstdint>
#        include <exception>
#        include <limits>

#        include "pth/pth.h"

namespace einsums::threads::coroutines::detail::posix::pth {
inline int check_(int rc) {
    // The makecontext() functions return zero for success, nonzero for
    // error. The Pth library returns TRUE for success, FALSE for error,
    // with errno set to the nonzero error in the latter case. Map the Pth
    // returns to ucontext style.
    return rc ? 0 : errno;
}
} // namespace einsums::threads::coroutines::detail::posix::pth

#        define EINSUMS_COROUTINE_POSIX_IMPL            "Pth implementation"
#        define EINSUMS_COROUTINE_DECLARE_CONTEXT(name) pth_uctx_t name
#        define EINSUMS_COROUTINE_CREATE_CONTEXT(ctx)   einsums::threads::coroutines::detail::posix::pth::check_(pth_uctx_create(&(ctx)))
#        define EINSUMS_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto)                                              \
            /* const sigset_t* sigmask = nullptr: we don't expect per-context signal                                                       \
             * masks */                                                                                                                    \
            einsums::threads::coroutines::detail::posix::pth::check_(                                                                      \
                pth_uctx_make(*(ctx), static_cast<char *>(stack), (size), nullptr, (startfunc), (startarg), (exitto)))
#        define EINSUMS_COROUTINE_SWAP_CONTEXT(from, to)                                                                                   \
            einsums::threads::coroutines::detail::posix::pth::check_(                                                                      \
                pth_uctx_switch(*(from), *(to))) #define EINSUMS_COROUTINE_DESTROY_CONTEXT(ctx)                                            \
                einsums::threads::coroutines::detail::posix::pth::check_(pth_uctx_destroy(ctx))

#    else                  // generic Posix platform (e.g. OS X >= 10.5)

/*
 * makecontext based context implementation. Should be available on all
 * SuSv2 compliant UNIX systems.
 * NOTE: this implementation is not
 * optimal as the makecontext API saves and restore the signal mask.
 * This requires a system call for every context switch that really kills
 * performance. Still is very portable and guaranteed to work.
 * NOTE2: makecontext and friends are declared obsolescent in SuSv3, but
 * it is unlikely that they will be removed any time soon.
 */
#        include <cstddef> // ptrdiff_t
#        include <ucontext.h>

#        if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION)

#            include <cstring>
#            include <signal.h>
#            include <stdlib.h>
#            include <strings.h>

#            if !defined(SEGV_STACK_SIZE)
#                define SEGV_STACK_SIZE MINSIGSTKSZ + 4096
#            endif

#        endif

#        include <iomanip>
#        include <iostream>

namespace einsums::threads::coroutines::detail::posix::ucontext {
inline int make_context(::ucontext_t *ctx, void *stack, std::ptrdiff_t size, void (*startfunc)(void *), void *startarg,
                        ::ucontext_t *exitto = nullptr) {
    int error = ::getcontext(ctx);
    if (error)
        return error;

    ctx->uc_stack.ss_sp   = (char *)stack;
    ctx->uc_stack.ss_size = size;
    ctx->uc_link          = exitto;

    using ctx_main = void (*)();
    // makecontext can't fail.
    ::makecontext(ctx, (ctx_main)(startfunc), 1, startarg);
    return 0;
}
} // namespace einsums::threads::coroutines::detail::posix::ucontext

#        define EINSUMS_COROUTINE_POSIX_IMPL            "ucontext implementation"
#        define EINSUMS_COROUTINE_DECLARE_CONTEXT(name) ::ucontext_t name
#        define EINSUMS_COROUTINE_CREATE_CONTEXT(ctx)   /* nop */
#        define EINSUMS_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto)                                              \
            einsums::threads::coroutines::detail::posix::ucontext::make_context(ctx, stack, size, startfunc, startarg, exitto)
#        define EINSUMS_COROUTINE_SWAP_CONTEXT(pfrom, pto) ::swapcontext((pfrom), (pto))
#        define EINSUMS_COROUTINE_DESTROY_CONTEXT(ctx)     /* nop */

#    endif // generic Posix platform

#    include <einsums/coroutines/detail/get_stack_pointer.hpp>
#    include <einsums/coroutines/detail/posix_utility.hpp>
#    include <einsums/coroutines/detail/swap_context.hpp>

#    include <atomic>
#    include <signal.h> // SIGSTKSZ

namespace einsums::threads::coroutines {
namespace detail {
// some platforms need special preparation of the main thread
struct prepare_main_thread {
    constexpr prepare_main_thread() {}
};
} // namespace detail

namespace detail::posix {
/// Posix implementation for the context_impl_base class.
/// @note context_impl is not required to be consistent
/// If not initialized it can only be swapped out, not in
/// (at that point it will be initialized).
class ucontext_context_impl_base : detail::context_impl_base {
  public:
    // on some platforms SIGSTKSZ resolves to a syscall, we can't make
    // this constexpr
    EINSUMS_EXPORT static std::ptrdiff_t default_stack_size;

    ucontext_context_impl_base() { EINSUMS_COROUTINE_CREATE_CONTEXT(_ctx); }
    ~ucontext_context_impl_base() { EINSUMS_COROUTINE_DESTROY_CONTEXT(_ctx); }

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
    void start_switch_fiber(void **fake_stack) { __sanitizer_start_switch_fiber(fake_stack, asan_stack_bottom, asan_stack_size); }
    void start_yield_fiber(void **fake_stack, ucontext_context_impl_base &caller) {
        __sanitizer_start_switch_fiber(fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
    }
    void finish_yield_fiber(void *fake_stack) { __sanitizer_finish_switch_fiber(fake_stack, &asan_stack_bottom, &asan_stack_size); }
    void finish_switch_fiber(void *fake_stack, ucontext_context_impl_base &caller) {
        __sanitizer_finish_switch_fiber(fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
    }
#    endif

  private:
    /// Free function. Saves the current context in @p from
    /// and restores the context in @p to.
    friend void swap_context(ucontext_context_impl_base &from, const ucontext_context_impl_base &to, default_hint) {
        [[maybe_unused]] int error = EINSUMS_COROUTINE_SWAP_CONTEXT(&from._ctx, &to._ctx);
        EINSUMS_ASSERT(error == 0);
    }

  protected:
    EINSUMS_COROUTINE_DECLARE_CONTEXT(_ctx);

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
  public:
    void       *asan_fake_stack   = nullptr;
    void const *asan_stack_bottom = nullptr;
    std::size_t asan_stack_size   = 0;
#    endif
};

template <typename CoroutineImpl>
class ucontext_context_impl : public ucontext_context_impl_base {
  public:
    EINSUMS_NON_COPYABLE(ucontext_context_impl);

  public:
    using context_impl_base = ucontext_context_impl_base;

    /// Create a context that on restore invokes Functor on
    ///  a new stack. The stack size can be optionally specified.
    explicit ucontext_context_impl(std::ptrdiff_t stack_size = -1)
        : _stack_size(stack_size == -1 ? this->default_stack_size : stack_size), _stack(nullptr), funp_(&trampoline<CoroutineImpl>) {}

    void init() {
        if (_stack != nullptr)
            return;

        _stack = alloc_stack(static_cast<std::size_t>(_stack_size));
        if (_stack == nullptr) {
            throw std::runtime_error("could not allocate memory for stack");
        }

        [[maybe_unused]] int error = EINSUMS_COROUTINE_MAKE_CONTEXT(&_ctx, _stack, _stack_size, funp_, this, nullptr);

        EINSUMS_ASSERT(error == 0);

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        asan_stack_size   = _stack_size;
        asan_stack_bottom = const_cast<const void *>(_stack);
#    endif
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
        action.sa_sigaction = &ucontext_context_impl::sigsegv_handler;

        sigaltstack(&segv_stack, nullptr);
        sigemptyset(&action.sa_mask);
        sigaddset(&action.sa_mask, SIGSEGV);
        sigaction(SIGSEGV, &action, nullptr);
#    endif
    }

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)

    // heuristic value 1 kilobyte
    //

#        define COROUTINE_STACKOVERFLOW_ADDR_EPSILON 1000UL

    static void sigsegv_handler(int, siginfo_t *infoptr, void *ctxptr) {
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
            std::cerr << "Stack overflow in coroutine at address " << std::internal << std::hex << std::setw(sizeof(sigsegv_ptr) * 2 + 2)
                      << std::setfill('0') << sigsegv_ptr << ".\n\n";

            std::cerr << "Configure the einsums runtime to allocate a larger coroutine stack "
                         "size.\n Use the einsums.stacks.small_size, "
                         "einsums.stacks.mediu_size,\n "
                         "einsums.stacks.large_size, or einsums.stacks.huge_size "
                         "configuration\nflags "
                         "to configure coroutine stack sizes.\n"
                      << std::endl;

            std::terminate();
        }
    }
#    endif

    ~ucontext_context_impl() {
        if (_stack)
            free_stack(_stack, _stack_size);

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        free(segv_stack.ss_sp);
#    endif
    }

    // Return the size of the reserved stack address space.
    std::ptrdiff_t get_stacksize() const { return _stack_size; }

    std::ptrdiff_t get_available_stack_space() {
#    if defined(EINSUMS_HAVE_THREADS_GET_STACK_POINTER)
        return get_stack_ptr() - reinterpret_cast<std::size_t>(_stack);
#    else
        return (std::numeric_limits<std::ptrdiff_t>::max)();
#    endif
    }

    void reset_stack() {
        if (_stack) {
            if (posix::reset_stack(_stack, static_cast<std::size_t>(_stack_size))) {
#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
                increment_stack_unbind_count();
#    endif
            }
        }
    }

    void rebind_stack() {
        if (_stack) {
            // just reset the context stack pointer to its initial value at
            // the stack start
#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
            increment_stack_recycle_count();
#    endif
            [[maybe_unused]] int error = EINSUMS_COROUTINE_MAKE_CONTEXT(&_ctx, _stack, _stack_size, funp_, this, nullptr);
            EINSUMS_ASSERT(error == 0);

#    if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
            asan_stack_size   = _stack_size;
            asan_stack_bottom = const_cast<const void *>(_stack);
#    endif
        }
    }

#    if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
    using counter_type = std::atomic<std::int64_t>;

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
    static std::uint64_t get_stack_unbind_count(bool reset) { return detail::get_and_reset_value(get_stack_unbind_counter(), reset); }

    static std::uint64_t get_stack_recycle_count(bool reset) { return detail::get_and_reset_value(get_stack_recycle_counter(), reset); }
#    endif

  private:
    // declare _stack_size first so we can use it to initialize _stack
    std::ptrdiff_t _stack_size;
    void          *_stack;
    void (*funp_)(void *);

#    if defined(EINSUMS_HAVE_STACKOVERFLOW_DETECTION) && !defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
    struct sigaction action;
    stack_t          segv_stack;
#    endif
};
} // namespace detail::posix
} // namespace einsums::threads::coroutines

#else

/**
 * This #else clause is essentially unchanged from the original Google Summer
 * of Code version of Boost.Coroutine, which comments:
 * "Context swapping can be implemented on most posix systems lacking *context
 * using the signaltstack+longjmp trick."
 * This is in fact what the (highly portable) Pth library does, so if you
 * encounter such a system, perhaps the best approach would be to twiddle the
 * #if logic in this header to use the pth.h implementation above.
 */
#    error No context implementation for this POSIX system.

#endif
