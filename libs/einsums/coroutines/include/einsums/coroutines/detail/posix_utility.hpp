//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/type_support/unused.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
#    include <cerrno>
#    include <cstddef>
#    include <cstdlib>
#    include <cstring>
#    include <stdexcept>
#    include <string>

#    if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
#        include <errno.h>
#        include <stdexcept>
#        include <sys/mman.h>
#        include <sys/param.h>
#    endif

#    if defined(__FreeBSD__)
#        include <sys/param.h>
#        define EXEC_PAGESIZE PAGE_SIZE
#    endif

#    if defined(__APPLE__)
#        include <unistd.h>
#        define EXEC_PAGESIZE static_cast<std::size_t>(sysconf(_SC_PAGESIZE))
#    endif

namespace einsums::threads::coroutines::detail::posix {

EINSUMS_EXPORT extern bool use_guard_pages;

#    if defined(EINSUMS_HAVE_THREAD_STACK_MMAP) && defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
inline auto to_stack_with_guard_page(void *stack) -> void * {
    if (use_guard_pages) {
        return static_cast<void *>(static_cast<void **>(stack) - (EXEC_PAGESIZE / sizeof(void *)));
    }
    return stack;
}

inline auto to_stack_without_guard_page(void *stack) -> void * {
    if (use_guard_pages) {
        return static_cast<void *>(static_cast<void **>(stack) + (EXEC_PAGESIZE / sizeof(void *)));
    }

    return stack;
}

inline void add_guard_page(void *stack) {
    if (use_guard_pages) {
        int r = ::mprotect(stack, EXEC_PAGESIZE, PROT_NONE);
        if (r != 0) {
            std::string error_message = "mprotect on a stack allocation failed with errno " + std::to_string(errno) +
                                        " (" + std::strerror(errno) + ")";
            throw std::runtime_error(error_message);
        }
    }
}

inline auto stack_size_with_guard_page(std::size_t size) -> std::size_t {
    if (use_guard_pages) {
        return size + EXEC_PAGESIZE;
    }

    return size;
}

inline auto alloc_stack(std::size_t size) -> void * {
    void *real_stack = ::mmap(nullptr, stack_size_with_guard_page(size), PROT_READ | PROT_WRITE,
#        if defined(__APPLE__)
                              MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
#        elif defined(__FreeBSD__)
                              MAP_PRIVATE | MAP_ANON,
#        else
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
#        endif
                              -1, 0);

    if (real_stack == MAP_FAILED) {
        if (ENOMEM == errno && use_guard_pages) {
            char const *error_message = "mmap failed to allocate thread stack due to insufficient resources. "
                                        "Increasing /proc/sys/vm/max_map_count or disabling guard pages with the "
                                        "configuration option pika.stacks.use_guard_pages=0 may reduce memory "
                                        "consumption.";
            throw std::runtime_error(error_message);
        }

        std::string error_message = "mmap failed to allocate thread stack with errno " + std::to_string(errno) + " (" +
                                    std::strerror(errno) + ")";
        throw std::runtime_error(error_message);
    }

    add_guard_page(real_stack);
    return to_stack_without_guard_page(real_stack);
}

inline void watermark_stack(void *stack, std::size_t size) {
    EINSUMS_ASSERT(size > EXEC_PAGESIZE);

    // Fill the bottom 8 bytes of the first page with 1s.
    void **watermark = static_cast<void **>(stack) + ((size - EXEC_PAGESIZE) / sizeof(void *));
    *watermark       = reinterpret_cast<void *>(0xDEADBEEFDEADBEEFull);
}

inline auto reset_stack(void *stack, std::size_t size) -> bool {
    void **watermark = static_cast<void **>(stack) + ((size - EXEC_PAGESIZE) / sizeof(void *));

    // If the watermark has been overwritten, then we've gone past the first
    // page.
    if ((reinterpret_cast<void *>(0xDEADBEEFDEADBEEFull)) != *watermark) {
        // We never free up the first page, as it's initialized only when the
        // stack is created.
        int r = ::madvise(stack, size - EXEC_PAGESIZE, MADV_DONTNEED);
        if (r != 0) {
            std::string error_message = "madvise on a stack allocation failed with errno " + std::to_string(errno) +
                                        " (" + std::strerror(errno) + ")";
            throw std::runtime_error(error_message);
        }
        return true;
    }

    return false;
}

inline void free_stack(void *stack, std::size_t size) {
    int r = ::munmap(to_stack_with_guard_page(stack), stack_size_with_guard_page(size));
    if (r != 0) {
        std::string error_message = "munmap failed to deallocate thread stack with errno " + std::to_string(errno) +
                                    " (" + std::strerror(errno) + ")";
        throw std::runtime_error(error_message);
    }
}

#    else // non-mmap()

// this should be a fine default.
static std::size_t const stack_alignment = sizeof(void *) > 16 ? sizeof(void *) : 16;

struct stack_aligner {
    alignas(stack_alignment) char dummy[stack_alignment];
};

/**
 * Stack allocator and deleter functions.
 * Better implementations are possible using
 * mmap (might be required on some systems) and/or
 * using a pooling allocator.
 * NOTE: the SuSv3 documentation explicitly allows
 * the use of malloc to allocate stacks for makectx.
 * We use new/delete for guaranteed alignment.
 */
inline void *alloc_stack(std::size_t size) {
    return new stack_aligner[size / sizeof(stack_aligner)];
}

inline void watermark_stack(void *stack, std::size_t size) {
} // no-op

inline bool reset_stack(void *stack, std::size_t size) {
    return false;
}

inline void free_stack(void *stack, std::size_t size) {
    delete[] static_cast<stack_aligner *>(stack);
}

#    endif // non-mmap() implementation of alloc_stack()/free_stack()

/**
 * The splitter is needed for 64 bit systems.
 * @note The current implementation does NOT use
 * (for debug reasons).
 * Thus it is not 64 bit clean.
 * Use it for 64 bits systems.
 */
template <typename T>
union splitter {
    int int_[2];
    T  *ptr;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    splitter(int first_, int second_) {
        int_[0] = first_;
        int_[1] = second_;
    }

    auto first() -> int { return int_[0]; }

    auto second() -> int { return int_[1]; }

    splitter(T *ptr_) : ptr(ptr_) {}

    void operator()() { (*ptr)(); }
};

template <typename T>
inline void trampoline_split(int first, int second) {
    splitter<T> split(first, second);
    split();
}

template <typename T>
inline void trampoline(void *fun) {
    (*static_cast<T *>(fun))();
}
} // namespace einsums::threads::coroutines::detail::posix

#else
#    error This header can only be included when compiling for posix systems.
#endif