//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Partial loads and stores at every tail length. The exact-size cases allocate exactly n elements
// on the heap, so an access past them is an out-of-bounds read or write that the sanitizer legs
// report, rather than a quiet read of whatever sits next in a larger buffer.

#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <catch2/catch_all.hpp>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

using namespace einsums::simd;

TEMPLATE_TEST_CASE("loadu_partial reads n lanes and zeroes the rest", "[simd][partial]", float, double, int32_t, uint32_t, int64_t,
                   uint64_t) {
    constexpr int         L = Vec<TestType>::lanes;
    std::vector<TestType> src(L);
    for (int i = 0; i < L; ++i) {
        src[i] = TestType(i) + TestType(1);
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L) + 1; ++n) {
        TestType out[L];
        storeu(out, loadu_partial(src.data(), n));
        for (int i = 0; i < L; ++i) {
            INFO("n = " << n << ", lane " << i);
            CHECK(out[i] == (static_cast<std::size_t>(i) < n ? src[i] : TestType(0)));
        }
    }
}

TEMPLATE_TEST_CASE("storeu_partial writes n lanes and nothing past them", "[simd][partial]", float, double, int32_t, uint32_t, int64_t,
                   uint64_t) {
    constexpr int L = Vec<TestType>::lanes;
    TestType      src[L];
    for (int i = 0; i < L; ++i) {
        src[i] = TestType(i) + TestType(1);
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L) + 1; ++n) {
        std::vector<TestType> dst(L + 2, TestType(99));
        storeu_partial(dst.data(), loadu(src), n);
        for (int i = 0; i < L + 2; ++i) {
            INFO("n = " << n << ", element " << i);
            CHECK(dst[i] == (static_cast<std::size_t>(i) < std::min(n, static_cast<std::size_t>(L)) ? src[i] : TestType(99)));
        }
    }
}

TEMPLATE_TEST_CASE("partial load and store stay inside an exact-size allocation", "[simd][partial]", float, double, int32_t, int64_t) {
    constexpr int L = Vec<TestType>::lanes;
    for (std::size_t n = 1; n < static_cast<std::size_t>(L); ++n) {
        auto const src = std::make_unique<TestType[]>(n);
        auto const dst = std::make_unique<TestType[]>(n);
        for (std::size_t i = 0; i < n; ++i) {
            src[i] = TestType(i) + TestType(1);
        }
        storeu_partial(dst.get(), loadu_partial(src.get(), n), n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(dst[i] == src[i]);
        }
    }
}

TEST_CASE("native_masked_memory is true exactly where a masked load or store is one instruction", "[simd][partial]") {
#if (defined(__AVX512F__) && defined(__AVX512VL__)) || defined(__AVX__)
    // AVX-512 masks every type; AVX2 has VPMASKMOV for the integers, and AVX without it moves them
    // through VMASKMOV's float form.
    STATIC_CHECK(native_masked_memory<float>);
    STATIC_CHECK(native_masked_memory<double>);
    STATIC_CHECK(native_masked_memory<int32_t>);
    STATIC_CHECK(native_masked_memory<int64_t>);
#else
    STATIC_CHECK_FALSE(native_masked_memory<float>);
    STATIC_CHECK_FALSE(native_masked_memory<double>);
    STATIC_CHECK_FALSE(native_masked_memory<int32_t>);
#endif
    // native_partial is the earlier name.
    STATIC_CHECK(native_partial<float> == native_masked_memory<float>);
    STATIC_CHECK(native_partial<int64_t> == native_masked_memory<int64_t>);
}

#if defined(__linux__)
namespace {

/// Two pages, the second inaccessible: an element run ending at the end of the first page faults if
/// anything reads or writes past it.
struct GuardPage {
    std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void       *base = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    GuardPage() {
        REQUIRE(base != MAP_FAILED);
        REQUIRE(mprotect(static_cast<char *>(base) + page, page, PROT_NONE) == 0);
    }
    ~GuardPage() { munmap(base, 2 * page); }
    /// The first of the last @p n elements before the guard.
    template <typename T>
    T *last(std::size_t n) const {
        return reinterpret_cast<T *>(static_cast<char *>(base) + page) - n;
    }
};

} // namespace

TEMPLATE_TEST_CASE("masked load, store and gather never touch an inactive lane's memory", "[simd][partial][mask]", float, double, int32_t,
                   uint32_t, int64_t, uint64_t) {
    // The active lanes sit against an inaccessible page, so a read or write of any inactive lane past
    // them faults; inactive lanes' gather indices point into it.
    using T           = TestType;
    constexpr int   L = Vec<T>::lanes;
    GuardPage const guard;
    for (std::size_t n = 0; n <= static_cast<std::size_t>(L); ++n) {
        INFO("n = " << n);
        T *const p = guard.last<T>(n);
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = static_cast<T>(i + 1);
        }
        Mask<T> const m = first_n<T>(n);

        T out[L];
        storeu(out, loadu(p, m));
        for (int i = 0; i < L; ++i) {
            CHECK(out[i] == (static_cast<std::size_t>(i) < n ? static_cast<T>(i + 1) : T(0)));
        }

        storeu(p, broadcast(T(9)), m);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(p[i] == T(9));
        }

        storeu(out, loadu_partial(p, n));
        storeu_partial(p, loadu(out), n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(p[i] == T(9));
        }

        if constexpr (std::is_floating_point_v<T>) {
            // Active lanes read p[n - 1 - i]; inactive lanes' indices are far past the guard.
            using I = gather_index_t<T>;
            I at[L];
            for (int i = 0; i < L; ++i) {
                at[i] = static_cast<std::size_t>(i) < n ? static_cast<I>(n - 1 - static_cast<std::size_t>(i)) : I{1} << 20;
            }
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = static_cast<T>(10 + i);
            }
            storeu(out, gather(p, loadu(at), m));
            for (int i = 0; i < L; ++i) {
                CHECK(out[i] == (static_cast<std::size_t>(i) < n ? static_cast<T>(10 + n - 1 - static_cast<std::size_t>(i)) : T(0)));
            }
        }
    }
}
#endif
