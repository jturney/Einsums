//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <chrono>
#    include <cstdint>

namespace einsums::profile {

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using ns        = std::chrono::nanoseconds; // NOLINT

enum class EventType : uint8_t {
    Push,
    Pop,
    Annotate,
    SetThreadName,
    MemAlloc,
    MemFree,
};

enum class AnnotateValueType : uint8_t {
    String,
    Int64,
    Float64,
};

struct Event {
    EventType type;
    TimePoint timestamp;

    // For Push/Pop: interned string IDs
    uint32_t name_id;
    uint32_t file_id;
    uint32_t func_id;
    int      line;

    // Hardware counter slots (Phase 2; zeros for now)
    uint64_t counters[4];

    // For Annotate events
    uint32_t          key_id;
    AnnotateValueType value_type;
    union {
        uint32_t string_id;
        int64_t  int_val;
        double   float_val;
    };

    // For MemAlloc/MemFree events
    int64_t mem_bytes{0};
};

} // namespace einsums::profile

#endif
