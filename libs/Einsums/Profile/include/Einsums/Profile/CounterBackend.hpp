//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <array>
#    include <cstdint>
#    include <string>

namespace einsums::profile {

/// Number of hardware counter slots per event.
static constexpr int kNumCounterSlots = 4;

/// Abstract interface for hardware performance counter backends.
class CounterBackend {
  public:
    virtual ~CounterBackend() = default;

    /// Open hardware counters for the calling thread. Called once per thread.
    virtual void open_thread_counters() = 0;

    /// Close hardware counters for the calling thread. Called on thread exit.
    virtual void close_thread_counters() = 0;

    /// Read current counter values into the provided array.
    virtual void read(std::array<uint64_t, kNumCounterSlots> &values) = 0;

    /// Name of counter slot i.
    [[nodiscard]] virtual auto slot_name(int slot) const -> std::string = 0;

    /// Whether this backend has real hardware counters.
    [[nodiscard]] virtual auto available() const -> bool = 0;
};

/// No-op counter backend (used on macOS, Windows, or when counters are disabled).
class NoopCounterBackend : public CounterBackend {
  public:
    void open_thread_counters() override {}
    void close_thread_counters() override {}

    void read(std::array<uint64_t, kNumCounterSlots> &values) override { values.fill(0); }

    [[nodiscard]] auto slot_name(int slot) const -> std::string override {
        static std::array<std::string, kNumCounterSlots> const names = {{"cycles", "instructions", "cache-misses", "branch-misses"}};
        return names[slot];
    }

    [[nodiscard]] auto available() const -> bool override { return false; }
};

#    ifdef __linux__

/// Linux perf_event_open counter backend.
class PerfCounterBackend : public CounterBackend {
  public:
    PerfCounterBackend();
    ~PerfCounterBackend() override;

    void open_thread_counters() override;
    void close_thread_counters() override;
    void read(std::array<uint64_t, kNumCounterSlots> &values) override;
    auto slot_name(int slot) const -> std::string override;
    auto available() const -> bool override;

  private:
    struct ThreadCounters {
        int  fds[kNumCounterSlots] = {-1, -1, -1, -1}; // NOLINT(modernize-avoid-c-arrays)
        bool open                  = false;
    };

    static auto thread_counters() -> ThreadCounters &;
};

#    endif // __linux__

/// Get the global counter backend instance.
EINSUMS_EXPORT auto get_counter_backend() -> CounterBackend &;

} // namespace einsums::profile

#endif
