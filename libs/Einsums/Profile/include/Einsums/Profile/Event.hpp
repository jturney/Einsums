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

EINSUMS_NAMESPACE_BEGIN(profile)

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

    /// For Push/Pop: how deeply the PRODUCER was nested, 1 for an outermost
    /// zone. On a Push it is the level of the zone being opened; on a Pop, the
    /// level of the zone being closed.
    ///
    /// The consumer cannot infer this. Events are dropped when a thread's ring
    /// buffer fills, and a Push and its Pop are dropped independently, so a
    /// reconstruction that trusts its own stack drifts: one lost Pop leaves a
    /// frame open forever and nests every later zone under it, a level deeper
    /// each time. That turned a three-level tree into a chain 100k nodes long
    /// on a run that dropped a million events, which no report can be read
    /// through and which overflowed the stack when the tree was freed. Stamping
    /// the producer's own depth makes every event enough to resynchronize
    /// against, so drops cost the zones they hit and nothing else.
    uint32_t depth{0};

    // Hardware counter slots (future work; zeros for now)
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

EINSUMS_NAMESPACE_END(profile)

#endif
