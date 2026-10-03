//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <chrono>
#include <cstdint>

WAGGLE_NAMESPACE_BEGIN

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using ns        = std::chrono::nanoseconds; // NOLINT

/// What an @ref Event records: a zone opening or closing, an annotation, a thread's name, or memory.
enum class EventType : uint8_t {
    Push,
    Pop,
    Annotate,
    SetThreadName,
    MemAlloc,
    MemFree,
};

/// Which member of an @ref AnnotationPayload's value an annotation filled.
enum class AnnotateValueType : uint8_t {
    String,
    Int64,
    Float64,
};

/// The payload of an Annotate event: a key and a typed value.
struct AnnotationPayload {
    uint32_t          key_id;
    AnnotateValueType value_type;
    union {
        uint32_t string_id;
        int64_t  int_val;
        double   float_val;
    };
};

/**
 * @brief One record in a thread's ring buffer, exactly one cache line.
 *
 * Each event type uses one arm of the trailing union: Push and Pop the hardware counters, Annotate
 * the annotation, MemAlloc and MemFree the byte count.
 */
struct alignas(64) Event {
    /// Raw @ref TickClock ticks; the consumer converts them with TickClock::to_time_point.
    uint64_t  ticks;
    EventType type;

    /// For Push: the zone's call site (see SiteTable), which supplies its file, line, function
    /// and domain.
    uint32_t site_id;
    /// For Push: the zone's name when built at run time, a string-table id; 0 for the site's own.
    uint32_t name_id;

    /// For Push/Pop: the producer's nesting level of the zone opened or closed, 1 for outermost.
    ///
    /// The consumer cannot infer it, because a Push and its Pop are dropped independently when a
    /// ring fills; one lost Pop would nest every later zone a level deeper. With the producer's
    /// depth on every event, a drop costs only the zones it hit.
    uint32_t depth{0};

    union {
        /// For Push/Pop: hardware counter values, left zero unless a counter backend is active.
        uint64_t counters[4]; // NOLINT(modernize-avoid-c-arrays)
        /// For Annotate.
        AnnotationPayload annotation;
        /// For MemAlloc/MemFree: the allocation and its size.
        struct {
            uint64_t address;
            int64_t  bytes;
        } mem;
    };
};

static_assert(sizeof(Event) == 64, "an Event is meant to fill one cache line exactly");

WAGGLE_NAMESPACE_END
