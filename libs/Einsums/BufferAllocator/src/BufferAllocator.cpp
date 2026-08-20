//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <mimalloc.h>

EINSUMS_NAMESPACE_BEGIN(memory)

void *aligned_alloc(size_t bytes) {
    return mi_malloc_aligned(bytes, 64);
}

void aligned_free(void *ptr) {
    mi_free(ptr);
}

EINSUMS_NAMESPACE_END(memory)

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(EINSUMS_WINDOWS)
#    include <windows.h>
#else
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN(detail)

std::string max_memory_provider() {
    size_t physical = 0;
#if defined(__APPLE__)
    size_t len = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &len, nullptr, 0) != 0) {
        physical = 0;
    }
#elif defined(EINSUMS_WINDOWS)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        physical = static_cast<size_t>(status.ullTotalPhys);
    }
#else
    long const pages = sysconf(_SC_PHYS_PAGES);
    long const page  = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) {
        physical = static_cast<size_t>(pages) * static_cast<size_t>(page);
    }
#endif
    if (physical == 0) {
        return "8GB"; // probe failed; a conservative ceiling beats an absent one
    }
    return std::to_string((physical / 100) * 80 / (1024 * 1024)) + "MB";
}

void *allocate(size_t n) {
    return memory::aligned_alloc(n);
}

void deallocate(void *p) {
    memory::aligned_free(p);
}

EINSUMS_NAMESPACE_END(detail)
