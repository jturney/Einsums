//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/core/get_and_reset_value.hpp>
#include <einsums/coroutines/config/defines.hpp>
#include <einsums/coroutines/detail/get_stack_pointer.hpp>
#include <einsums/coroutines/detail/swap_context.hpp>
#include <einsums/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <windows.h>
#include <winnt.h>

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
#    include <processthreadsapi.h>
#    include <sanitizer/asan_interface.h>
#endif

#if defined(EINSUMS_HAVE_SWAP_CONTEXT_EMULATION)
extern "C" void switch_to_fiber(void *lpFiber) noexcept;
#endif

namespace einsums::threads::coroutines {
namespace detail {
// On Windows we need a special preparation for the main coroutines thread
struct prepare_main_thread {
    prepare_main_thread() noexcept {
        [[maybe_unused]] LPVOID result = ConvertThreadToFiber(nullptr);
        EINSUMS_ASSERT(nullptr != result);
    }

    ~prepare_main_thread() noexcept {
        [[maybe_unused]] BOOL result = ConvertFiberToThread();
        EINSUMS_ASSERT(FALSE != result);
    }
};
} // namespace detail

namespace detail::windows {
using fiber_ptr = LPVOID;

#if _WIN32_WINNT < 0x0600
/*
 * This number (0x1E00) has been sighted in the wild (at least on
 * windows XP systems) as return value from GetCurrentFiber() on non
 * fibrous threads.
 * This is somehow related to OS/2 where the current fiber pointer is
 * overloaded as a version field.
 * On non-NT systems, 0 is returned.
 */
fiber_ptr const fiber_magic = reinterpret_cast<fiber_ptr>(0x1E00);
#endif

/*
 * Return true if current thread is a fiber.
 */
inline bool is_fiber() noexcept {
#if _WIN32_WINNT >= 0x0600
    return IsThreadAFiber() ? true : false;
#else
    fiber_ptr current = GetCurrentFiber();
    return current != nullptr && current != fiber_magic;
#endif
}

/*
 * Windows implementation for the context_impl_base class.
 * @note context_impl is not required to be consistent
 * If not initialized it can only be swapped out, not in
 * (at that point it will be initialized).
 */
class fibers_context_impl_base : detail::context_impl_base {
  public:
    /**
     * Create an empty context.
     * An empty context cannot be restored from,
     * but can be saved in.
     */
    fibers_context_impl_base() noexcept
        : m_ctx(nullptr)
#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
          ,
          asan_fake_stack(nullptr), asan_stack_bottom(nullptr), asan_stack_size(0)
#endif
    {
    }

    /*
     * Free function. Saves the current context in @p from
     * and restores the context in @p to. On windows the from
     * parameter is ignored. The current context is saved on the
     * current fiber.
     * Note that if the current thread is not a fiber, it will be
     * converted to fiber on the fly on call and unconverted before
     * return. This is expensive. The user should convert the
     * current thread to a fiber once on thread creation for better performance.
     * Note that we can't leave the thread unconverted on return or else we
     * will leak resources on thread destruction. Do the right thing by
     * default.
     */
    friend void swap_context(fibers_context_impl_base &from, fibers_context_impl_base const &to,
                             default_hint) noexcept {
        if (!is_fiber()) {
            EINSUMS_ASSERT(from.m_ctx == nullptr);
            from.m_ctx = ConvertThreadToFiber(nullptr);
            EINSUMS_ASSERT(from.m_ctx != nullptr);

#if defined(EINSUMS_HAVE_SWAP_CONTEXT_EMULATION)
            switch_to_fiber(to.m_ctx);
#else
            SwitchToFiber(to.m_ctx);
#endif
            [[maybe_unused]] BOOL result = ConvertFiberToThread();
            EINSUMS_ASSERT(result);
            from.m_ctx = nullptr;
        } else {
            bool call_from_main = from.m_ctx == nullptr;
            if (call_from_main)
                from.m_ctx = GetCurrentFiber();
#if defined(EINSUMS_HAVE_SWAP_CONTEXT_EMULATION)
            switch_to_fiber(to.m_ctx);
#else
            SwitchToFiber(to.m_ctx);
#endif
            if (call_from_main)
                from.m_ctx = nullptr;
        }
    }

    ~fibers_context_impl_base() = default;

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
    void start_switch_fiber(void **fake_stack) {
        if (asan_stack_bottom == nullptr) {
            void const *dummy = nullptr;
            GetCurrentThreadStackLimits((PULONG_PTR)&dummy, (PULONG_PTR)&asan_stack_bottom);
        }
        __sanitizer_start_switch_fiber(fake_stack, asan_stack_bottom, asan_stack_size);
    }
    void start_yield_fiber(void **fake_stack, fibers_context_impl_base &caller) {
        __sanitizer_start_switch_fiber(fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
    }
    void finish_yield_fiber(void *fake_stack) {
        __sanitizer_finish_switch_fiber(fake_stack, &asan_stack_bottom, &asan_stack_size);
    }
    void finish_switch_fiber(void *fake_stack, fibers_context_impl_base &caller) {
        __sanitizer_finish_switch_fiber(fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
    }
#endif

  protected:
    explicit fibers_context_impl_base(fiber_ptr ctx) noexcept : m_ctx(ctx) {}

    fiber_ptr m_ctx;

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
  public:
    void       *asan_fake_stack;
    void const *asan_stack_bottom;
    std::size_t asan_stack_size;
#endif
};

template <typename T>
EINSUMS_FORCEINLINE VOID CALLBACK trampoline(LPVOID pv) {
    T *fun = static_cast<T *>(pv);
    EINSUMS_ASSERT(fun);
    (*fun)();
}

// initial stack size (grows as needed)
static std::size_t const stack_size = sizeof(void *) >= 8 ? 2048 : 1024;

template <typename CoroutineImpl>
class fibers_context_impl : public fibers_context_impl_base {
  public:
    EINSUMS_NON_COPYABLE(fibers_context_impl);

  public:
    using context_impl_base = fibers_context_impl_base;

    enum { default_stack_size = stack_size };

    /**
     * Create a context that on restore invokes Functor on
     *  a new stack. The stack size can be optionally specified.
     */
    explicit fibers_context_impl(std::ptrdiff_t stacksize)
        : stacksize_(stacksize == -1 ? std::ptrdiff_t(default_stack_size) : stacksize) {}

    void init() {
        if (m_ctx != nullptr)
            return;

        m_ctx = CreateFiberEx(stacksize_, stacksize_, 0, static_cast<LPFIBER_START_ROUTINE>(&trampoline<CoroutineImpl>),
                              static_cast<LPVOID>(this));
        if (nullptr == m_ctx) {
            throw std::system_error(GetLastError(), std::system_category());
        }

#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        this->asan_stack_size   = stacksize_;
        this->asan_stack_bottom = nullptr;
#endif
    }

    ~fibers_context_impl() {
        if (m_ctx != nullptr)
            DeleteFiber(m_ctx);
    }

    // Return the size of the reserved stack address space.
    std::ptrdiff_t get_stacksize() const noexcept { return stacksize_; }

    constexpr void reset_stack() noexcept {}

    void rebind_stack() noexcept {
#if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
        increment_stack_recycle_count();
#endif
    }

    // Detect remaining stack space (approximate), taken from here:
    // https://stackoverflow.com/a/20930496/269943
    std::ptrdiff_t get_available_stack_space() {
        MEMORY_BASIC_INFORMATION mbi;                 // page range
        VirtualQuery((PVOID)&mbi, &mbi, sizeof(mbi)); // get range
        return (std::ptrdiff_t)&mbi - (std::ptrdiff_t)mbi.AllocationBase;
    }

#if defined(EINSUMS_HAVE_COROUTINE_COUNTERS)
    using counter_type = std::atomic<std::int64_t>;

  private:
    static counter_type &get_stack_recycle_counter() noexcept {
        static counter_type counter(0);
        return counter;
    }

    static std::uint64_t increment_stack_recycle_count() noexcept { return ++get_stack_recycle_counter(); }

  public:
    static std::uint64_t get_stack_recycle_count(bool reset) noexcept {
        return detail::get_and_reset_value(get_stack_recycle_counter(), reset);
    }
#endif

  private:
    std::ptrdiff_t stacksize_;
};
} // namespace detail::windows
} // namespace einsums::threads::coroutines