//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file StreamingScale.cpp
/// @brief dst = alpha * src over a large array, with prefetches and streaming stores.
///
/// When a loop writes an array it will not read again soon, an ordinary store first reads each
/// destination cache line into the cache, a read the program never asked for. stream_store writes
/// around the cache instead, so the line is never fetched. Three rules come with it:
///
///   - dst must be aligned to a whole Vec, since streaming stores take no unaligned form;
///   - streaming stores are weakly ordered, so stream_fence() must follow the last one before
///     anything, this thread or another, reads the data;
///   - it only pays for arrays much larger than the cache. A small array that is read back soon
///     is better off in the cache, and there an ordinary store is faster.
///
/// prefetch hints the hardware to start loading a later part of src while this part is computed.
/// Modern cores prefetch sequential reads well on their own; the explicit form matters more for
/// the irregular patterns the hardware cannot predict.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Prefetch.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>

using namespace einsums::simd;

namespace {

/// dst[i] = alpha * src[i]. @p dst must be aligned to native_alignment and @p n a multiple of lanes.
void scale_streaming(std::size_t n, double alpha, double const *src, double *dst) {
    constexpr std::size_t L        = lanes<double>;
    constexpr std::size_t distance = 16 * L; // how far ahead to prefetch, in elements
    Vec<double> const     va       = broadcast(alpha);
    for (std::size_t i = 0; i < n; i += L) {
        // The instruction would not fault past the end, but forming that pointer is undefined in C++.
        if (i + distance < n) {
            prefetch<PrefetchHint::T0>(src + i + distance);
        }
        stream_store(dst + i, va * loadu(src + i));
    }
    stream_fence(); // make every streamed store visible before the caller reads dst
}

} // namespace

int einsums_main() {
    std::size_t const n = std::size_t{1} << 22; // 32 MiB per array: larger than any cache

    // Aligned operator new is the portable way to get a Vec-aligned array (MSVC has no std::aligned_alloc).
    struct AlignedDelete {
        void operator()(double *p) const { ::operator delete[](p, std::align_val_t{native_alignment}); }
    };
    auto aligned_array = [](std::size_t count) {
        return std::unique_ptr<double[], AlignedDelete>(
            static_cast<double *>(::operator new[](count * sizeof(double), std::align_val_t{native_alignment})));
    };
    auto src = aligned_array(n);
    auto dst = aligned_array(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<double>(i % 1000) * 0.5;
    }

    scale_streaming(n, 3.0, src.get(), dst.get());

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) {
        wrong += dst[i] != 3.0 * src[i];
    }
    std::cout << "scaled " << n << " doubles with streaming stores: " << (wrong ? "FAILED" : "ok") << "\n";
    return wrong ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
