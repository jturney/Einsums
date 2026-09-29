//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// stream_store_span at every destination misalignment and every length around the lane count: the
// head and tail partial stores, the streamed middle, and nothing written past the span. The
// exact-size cases put the span at the end of its allocation, so a store past it is a heap overflow
// the sanitizer legs report.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Prefetch.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

namespace {

// A scaled copy, the producer HPTT's contiguous paths use.
template <typename T>
auto scaled_copy(T const *src, T alpha) {
    return [src, alpha](std::size_t i, std::size_t count) {
        Vec<T> const v = count == static_cast<std::size_t>(Vec<T>::lanes) ? loadu(src + i) : loadu_partial(src + i, count);
        return mul(broadcast(alpha), v);
    };
}

} // namespace

TEMPLATE_TEST_CASE("stream_store_span writes the span and nothing around it", "[simd][stream]", float, double) {
    constexpr std::size_t L = static_cast<std::size_t>(Vec<TestType>::lanes);
    for (std::size_t offset = 0; offset <= L; ++offset) {
        for (std::size_t n = 0; n <= 4 * L + 3; ++n) {
            INFO("offset " << offset << ", n " << n);
            std::vector<TestType> src(n + 1);
            for (std::size_t i = 0; i < src.size(); ++i) {
                src[i] = static_cast<TestType>(i + 1);
            }
            std::vector<TestType> dst(offset + n + L, static_cast<TestType>(-7));
            stream_store_span(dst.data() + offset, n, scaled_copy(src.data(), TestType{2}));
            stream_fence();
            for (std::size_t i = 0; i < dst.size(); ++i) {
                bool const inside = i >= offset && i < offset + n;
                CHECK(dst[i] == (inside ? TestType{2} * src[i - offset] : static_cast<TestType>(-7)));
            }
        }
    }
}

TEMPLATE_TEST_CASE("stream_store_span stays inside an exact-size allocation", "[simd][stream]", float, double) {
    constexpr std::size_t L = static_cast<std::size_t>(Vec<TestType>::lanes);
    for (std::size_t offset = 0; offset < L; ++offset) {
        for (std::size_t n = 1; n <= 3 * L; ++n) {
            auto const src = std::make_unique<TestType[]>(n);
            auto const buf = std::make_unique<TestType[]>(offset + n);
            for (std::size_t i = 0; i < n; ++i) {
                src[i] = static_cast<TestType>(i) - TestType{3};
            }
            stream_store_span(buf.get() + offset, n, scaled_copy(src.get(), TestType{1}));
            stream_fence();
            for (std::size_t i = 0; i < n; ++i) {
                CHECK(buf[offset + i] == src[i]);
            }
        }
    }
}
