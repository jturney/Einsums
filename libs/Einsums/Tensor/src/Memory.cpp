//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Print.hpp>

namespace einsums::detail {

auto allocate_aligned_memory(size_t align, size_t size) -> void * {
    EINSUMS_ASSERT(align >= sizeof(void *));
    EINSUMS_ASSERT(align && !(align & (align - 1))); // Align should be a power of 2 but disallow 0.

    if (size == 0) {
        return nullptr;
    }

    // Determine padding to round up to align boundary.
    size_t extra = align - (size % align);
    size += extra;

    void *ptr{nullptr};
#if defined(_WIN32) || defined(_WIN64)
    ptr = malloc(size);
#else
    int rc = posix_memalign(&ptr, align, size);

    if (rc != 0) {
        println("posix_memalign returned non-zero!");
        return nullptr;
    }
#endif

    return ptr;
}

void deallocate_aligned_memory(void *ptr) noexcept {
    return free(ptr);
}

} // namespace einsums::detail