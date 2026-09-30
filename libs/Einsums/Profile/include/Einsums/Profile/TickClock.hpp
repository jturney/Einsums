//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <chrono>
#    include <cstdint>

#    if (defined(__x86_64__) || defined(_M_X64))
#        if defined(_MSC_VER) && !defined(__clang__)
#            include <intrin.h>
#        else
#            include <x86intrin.h>
#        endif
#    endif

EINSUMS_NAMESPACE_BEGIN(profile)

/**
 * @brief The clock profiling events are stamped with: the CPU's own counter, read raw.
 *
 * A zone is two timestamps, and ``std::chrono::steady_clock::now()`` costs about 14 ns on macOS:
 * a library call that reads the counter and converts it to nanoseconds. The producer needs only
 * the counter. So the event carries raw ticks and the consumer, off the hot path, converts them
 * to time points with @ref to_time_point.
 *
 * - arm64 (not MSVC): the generic timer's virtual count, ``cntvct_el0``, readable from user space
 *   on macOS and Linux, at the frequency ``cntfrq_el0`` reports (1 GHz on an M4, 24 MHz on M1-M3).
 * - x86-64 with an invariant TSC: ``rdtsc``, whose frequency is measured once against
 *   ``steady_clock``, as the TSC has no architectural register that states it.
 * - anything else, or an x86 TSC that is not invariant: ``steady_clock`` itself, in nanoseconds.
 *
 * Ticks are comparable only with ticks from the same process.
 */
struct EINSUMS_EXPORT TickClock {
    /// The counter as it stands, in ticks. Monotonic; a few nanoseconds or less to read.
    static inline std::uint64_t now() noexcept {
#    if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(_MSC_VER)
        std::uint64_t v;
        asm volatile("mrs %0, cntvct_el0" : "=r"(v));
        return v;
#    elif defined(__x86_64__) || defined(_M_X64)
        if (instance().uses_tsc) {
            return __rdtsc();
        }
        return fallback_now();
#    else
        return fallback_now();
#    endif
    }

    /// The steady_clock instant @p ticks corresponds to.
    [[nodiscard]] std::chrono::steady_clock::time_point to_time_point(std::uint64_t ticks) const noexcept {
        auto const delta = static_cast<double>(static_cast<std::int64_t>(ticks - anchor_ticks)) * ns_per_tick;
        return anchor_time + std::chrono::nanoseconds(static_cast<std::int64_t>(delta));
    }

    /// The process's clock, anchored and (on x86) calibrated on first use.
    static TickClock const &instance();

    /// Nanoseconds per tick.
    double ns_per_tick{1.0};

    /// A tick count and the steady_clock instant it was read at, together.
    std::uint64_t                         anchor_ticks{0};
    std::chrono::steady_clock::time_point anchor_time{};

    /// Whether @ref now reads the TSC; false where it falls back to steady_clock.
    bool uses_tsc{false};

    /// The name of the counter @ref now reads, for the report.
    char const *source{"steady_clock"};

  private:
    static std::uint64_t fallback_now() noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    TickClock();
};

EINSUMS_NAMESPACE_END(profile)

#endif
