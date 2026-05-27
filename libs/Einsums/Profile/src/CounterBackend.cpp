//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/CounterBackend.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    ifdef __linux__
#        include <cstring>
#        include <linux/perf_event.h>
#        include <sys/ioctl.h>
#        include <sys/syscall.h>
#        include <unistd.h>
#    endif

namespace einsums::profile {

#    ifdef __linux__

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

PerfCounterBackend::PerfCounterBackend()  = default;
PerfCounterBackend::~PerfCounterBackend() = default;

auto PerfCounterBackend::thread_counters() -> ThreadCounters & {
    thread_local ThreadCounters tc;
    return tc;
}

void PerfCounterBackend::open_thread_counters() {
    auto &tc = thread_counters();
    if (tc.open)
        return;

    // perf event types for our 4 slots
    static const uint32_t event_types[kNumCounterSlots] = {
        PERF_COUNT_HW_CPU_CYCLES,
        PERF_COUNT_HW_INSTRUCTIONS,
        PERF_COUNT_HW_CACHE_MISSES,
        PERF_COUNT_HW_BRANCH_MISSES,
    };

    int group_fd = -1;
    for (int i = 0; i < kNumCounterSlots; ++i) {
        struct perf_event_attr pe {};
        pe.type           = PERF_TYPE_HARDWARE;
        pe.size           = sizeof(pe);
        pe.config         = event_types[i];
        pe.disabled       = (i == 0) ? 1 : 0; // group leader starts disabled
        pe.exclude_kernel = 1;
        pe.exclude_hv     = 1;

        int fd = static_cast<int>(perf_event_open(&pe, 0, -1, group_fd, 0));
        if (fd < 0) {
            // Counter not available, close any already opened
            for (int j = 0; j < i; ++j) {
                if (tc.fds[j] >= 0) {
                    close(tc.fds[j]);
                    tc.fds[j] = -1;
                }
            }
            return;
        }
        tc.fds[i] = fd;
        if (i == 0)
            group_fd = fd;
    }

    // Enable the group
    ioctl(tc.fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(tc.fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    tc.open = true;
}

void PerfCounterBackend::close_thread_counters() {
    auto &tc = thread_counters();
    if (!tc.open)
        return;
    for (int i = 0; i < kNumCounterSlots; ++i) {
        if (tc.fds[i] >= 0) {
            close(tc.fds[i]);
            tc.fds[i] = -1;
        }
    }
    tc.open = false;
}

void PerfCounterBackend::read(std::array<uint64_t, kNumCounterSlots> &values) {
    auto &tc = thread_counters();
    if (!tc.open) {
        values.fill(0);
        return;
    }
    for (int i = 0; i < kNumCounterSlots; ++i) {
        uint64_t val = 0;
        if (tc.fds[i] >= 0) {
            ::read(tc.fds[i], &val, sizeof(val));
        }
        values[i] = val;
    }
}

auto PerfCounterBackend::slot_name(int slot) const -> std::string {
    static const std::array<std::string, kNumCounterSlots> names = {{"cycles", "instructions", "cache-misses", "branch-misses"}};
    return names[slot];
}

auto PerfCounterBackend::available() const -> bool {
    return true;
}

auto get_counter_backend() -> CounterBackend & {
    static PerfCounterBackend backend;
    return backend;
}

#    else // not __linux__

auto get_counter_backend() -> CounterBackend & {
    static NoopCounterBackend backend;
    return backend;
}

#    endif

} // namespace einsums::profile

#endif
