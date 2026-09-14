//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/Options.hpp>
#include <Einsums/Tensor/StorageBlock.hpp>

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN(detail)

size_t huge_page_advice_threshold() noexcept {
    // Two whole 2 MB pages can lie inside an unaligned 4 MB buffer, and the
    // kernel only backs whole aligned pages; below this the advice buys a VMA
    // split and nothing else.
    return size_t{4} << 20;
}

void advise_huge_pages(void *begin, size_t bytes) noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (begin == nullptr || bytes < huge_page_advice_threshold()) {
        return;
    }
    if (!config::get(option::TensorHugePages)) {
        return;
    }
    static auto const page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    auto const        lo   = reinterpret_cast<uintptr_t>(begin);
    auto const        hi   = lo + bytes;
    // Round the start DOWN: the page holding `begin` is mapped (the allocator's
    // header precedes it in the same mapping), and for an mmap'd chunk that page
    // is the mapping's first, which is what lets the leading 2 MB be huge.
    // Round the end DOWN: past the buffer the mapping may end, and madvise over
    // a hole reports ENOMEM.
    auto const start = lo & ~(page - 1);
    auto const stop  = hi & ~(page - 1);
    if (stop <= start) {
        return;
    }
    // Advice only: EINVAL when the kernel has THP compiled out, ENOMEM on a
    // hole, and either way the buffer stays a working 4 KB-paged buffer.
    (void)madvise(reinterpret_cast<void *>(start), static_cast<size_t>(stop - start), MADV_HUGEPAGE);
#else
    (void)begin;
    (void)bytes;
#endif
}

EINSUMS_NAMESPACE_END(detail)
