//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/GPU/MPSBackend.hpp>
#include <Einsums/GPU/Platform.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>

#include <Einsums/Testing.hpp>

#if defined(EINSUMS_HAVE_MPS)

#    include <unistd.h>

using namespace einsums::gpu;

namespace {

/// A page-aligned float buffer, which MPS wraps in place rather than copying.
struct PageBuffer {
    explicit PageBuffer(size_t count) {
        auto const page  = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        auto const bytes = (count * sizeof(float) + page - 1) / page * page;
        data.reset(static_cast<float *>(std::aligned_alloc(page, bytes)));
        REQUIRE(data != nullptr);
        REQUIRE(reinterpret_cast<std::uintptr_t>(data.get()) % page == 0);
        for (size_t i = 0; i < count; i++) {
            data.get()[i] = 1.0f;
        }
    }

    struct Free {
        void operator()(float *p) const { std::free(p); }
    };
    std::unique_ptr<float, Free> data;
};

} // namespace

// A BLAS call wraps the caller's page-aligned operands in MTLBuffers without copying them, and
// newBufferWithBytesNoCopy requires that memory to outlive its buffer. The command buffer comes
// back autoreleased and holds every buffer it encoded, and no autorelease pool drains on a thread
// Einsums does not own: every wrapper used to live as long as the process, still mapping memory a
// graph had freed and reused between replays, and they accumulated with every call. Each call now
// releases its wrappers before it returns, on whatever thread it runs.
TEST_CASE("MPS BLAS calls release the buffers wrapping caller memory", "[GPU][MPS]") {
    if (mps::device_name() == "Unknown MPS Device") {
        SKIP("no Metal device");
    }
    int constexpr n     = 64;
    size_t constexpr nn = size_t{n} * n;
    PageBuffer a(nn), b(nn), c(nn), x(n), y(n);

    auto const gemm = [&] { return mps::sgemm('n', 'n', n, n, n, 1.0f, a.data.get(), n, b.data.get(), n, 0.0f, c.data.get(), n); };
    auto const gemv = [&] { return mps::sgemv('n', n, n, 1.0f, a.data.get(), n, x.data.get(), 1, 0.0f, y.data.get(), 1); };

    SECTION("on the calling thread") {
        REQUIRE(gemm());
        CHECK(mps::live_host_wrappers() == 0);
        REQUIRE(gemv());
        CHECK(mps::live_host_wrappers() == 0);
    }

    SECTION("on a thread with no autorelease pool of its own") {
        bool   gemm_ok = false, gemv_ok = false;
        size_t after_gemm = 1, after_gemv = 1;
        std::thread([&] {
            gemm_ok    = gemm();
            after_gemm = mps::live_host_wrappers();
            gemv_ok    = gemv();
            after_gemv = mps::live_host_wrappers();
        }).join();
        REQUIRE(gemm_ok);
        REQUIRE(gemv_ok);
        CHECK(after_gemm == 0);
        CHECK(after_gemv == 0);
    }

    // The wrapped memory holds the results: every operand is all ones, so each entry is n.
    CHECK(c.data.get()[0] == static_cast<float>(n));
    CHECK(y.data.get()[n - 1] == static_cast<float>(n));
}

#endif
