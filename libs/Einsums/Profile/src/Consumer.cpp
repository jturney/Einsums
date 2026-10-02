//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Consumer.hpp>
#include <Einsums/Profile/TickClock.hpp>

#include <algorithm>
#include <string>

#if defined(EINSUMS_HAVE_PROFILER)

EINSUMS_NAMESPACE_BEGIN(profile)

Consumer::Consumer(StringTable &strings) : _strings(strings), _other_id(strings.intern("(other)")) {
    _running.store(true, std::memory_order_relaxed);
    _thread = std::thread([this] { consumer_loop(); });
}

Consumer::~Consumer() {
    shutdown();
}

auto Consumer::timeline_events() const -> std::vector<TimelineEvent> {
    std::vector<TimelineEvent> out;
    out.reserve(_timeline.size());
    // Oldest first: once the ring has wrapped, the oldest record is the one about to be overwritten.
    size_t const start = _timeline.size() < kMaxTimelineEvents ? 0 : _timeline_next;
    for (size_t k = 0; k < _timeline.size(); ++k) {
        auto const &rec = _timeline[(start + k) % _timeline.size()];
        out.push_back({.thread_id = rec.thread_id, .name = _strings.get(rec.name_id), .start_ms = rec.start_ms, .end_ms = rec.end_ms});
    }
    return out;
}

void Consumer::register_thread(uint32_t thread_id, std::shared_ptr<EventRingBuffer> rb) {
    std::scoped_lock const lock(_reg_mutex);
    _registrations.push_back({.thread_id = thread_id, .ring_buffer = std::move(rb)});
}

auto Consumer::dropped_count() const -> uint64_t {
    std::scoped_lock const lock(_reg_mutex);
    uint64_t               total = 0;
    for (auto const &reg : _registrations) {
        total += reg.ring_buffer->refused();
    }
    return total;
}

void Consumer::set_thread_name(uint32_t thread_id, std::string name) {
    std::unique_lock const lock(_tree_mutex);
    _threads[thread_id].name = name;
    _thread_names[thread_id] = std::move(name);
}

void Consumer::shutdown() {
    if (!_running.exchange(false, std::memory_order_acq_rel))
        return;            // already shut down
    _wake_cv.notify_one(); // Wake consumer thread immediately
    if (_thread.joinable())
        _thread.join();
    // Final drain under exclusive lock
    drain_all();
}

void Consumer::flush() {
    _wake_cv.notify_one(); // Wake consumer thread to drain immediately
    drain_all();
}

void Consumer::consumer_loop() {
    auto last_tick = std::chrono::steady_clock::now();
    // The nap doubles, up to kMaxNap, each time a look finds nothing, and resets to kMinNap when one
    // finds events, so an idle process is not woken a thousand times a second. A ring passing half
    // full, flush() and shutdown() wake it early. The cap stays well under the 50 ms some callers
    // wait for the tree without flushing.
    constexpr auto kMinNap = std::chrono::milliseconds(1);
    constexpr auto kMaxNap = std::chrono::milliseconds(10);
    auto           nap     = kMinNap;
    while (_running.load(std::memory_order_relaxed)) {
        nap = drain_all() > 0 ? kMinNap : std::min(2 * nap, kMaxNap);
        // Tick every ~500 ms: copy the callback under its lock, call it outside.
        auto now = std::chrono::steady_clock::now();
        if ((now - last_tick) >= std::chrono::milliseconds(500)) {
            std::function<void()> tick;
            {
                std::scoped_lock const lock(_tick_mutex);
                tick = _tick_callback;
            }
            last_tick = now;
            if (tick) {
                tick();
            }
        }
        // Nap until notified (by a filling ring, flush() or shutdown()) or the nap ends.
        std::unique_lock lock(_wake_mutex);
        _wake_cv.wait_for(lock, nap);
    }
}

size_t Consumer::drain_all() {
    // Return early if every ring is empty, so an idle process skips the snapshot and the tree lock.
    std::vector<ThreadRegistration> regs;
    {
        std::scoped_lock const lock(_reg_mutex);
        if (std::ranges::all_of(_registrations, [](ThreadRegistration const &r) { return r.ring_buffer->empty(); })) {
            return 0;
        }
        regs = _registrations;
    }

    if (regs.empty())
        return 0;

    // Drain events from all ring buffers under a single exclusive lock
    std::unique_lock const lock(_tree_mutex);
    size_t                 drained = 0;
    for (auto &reg : regs) {
        drained += reg.ring_buffer->drain([&](Event const &evt) { process_event(reg.thread_id, evt); });
    }
    return drained;
}

void Consumer::process_event(uint32_t thread_id, Event const &evt) {
    auto &ts = _threads[thread_id];

    switch (evt.type) {
    case EventType::Push:
        process_push(ts, evt);
        break;
    case EventType::Pop:
        process_pop(ts, evt, thread_id);
        break;
    case EventType::Annotate:
        process_annotate(ts, evt);
        break;
    case EventType::SetThreadName:
        // Handled via set_thread_name() API, not through ring buffer events
        break;
    case EventType::MemAlloc:
    case EventType::MemFree:
        process_mem(ts, evt);
        break;
    }
}

void Consumer::unwind_stale_frames(ThreadState &ts, size_t depth) {
    if (ts.stack.size() <= depth) {
        return;
    }
    _unmatched_zones.fetch_add(ts.stack.size() - depth, std::memory_order_relaxed);
    // No time is recorded for these zones or charged to their parents: it would be invented.
    ts.stack.resize(depth);
}

void Consumer::process_push(ThreadState &ts, Event const &evt) {
    // Anything open at or below the level this zone opens at lost its Pop.
    if (evt.depth > 0) {
        unwind_stale_frames(ts, evt.depth - 1);
    }

    ThreadState::StackFrame frame{};
    frame.name_id    = evt.name_id;
    frame.file_id    = evt.file_id;
    frame.func_id    = evt.func_id;
    frame.line       = evt.line;
    frame.child_time = ns{0};
    frame.start      = TickClock::instance().to_time_point(evt.ticks);
    for (int i = 0; i < kNumCounterSlots; ++i)
        frame.counters[i] = evt.counters[i];

    // One step down from the parent's node, resolved when the parent was pushed.
    AggNode *parent = ts.stack.empty() ? &ts.root : ts.stack.back().node;
    auto     it     = parent->children.find(evt.name_id);
    if (it == parent->children.end()) {
        // Names built at run time could grow the tree without bound. Past the cap a new name joins
        // the parent's "(other)" node, which still times it.
        std::int64_t const cap   = _max_distinct_children.load(std::memory_order_relaxed);
        auto               other = parent->children.find(_other_id);
        size_t const       named = parent->children.size() - (other != parent->children.end() ? 1 : 0);
        if (cap > 0 && named >= static_cast<size_t>(cap)) {
            if (other == parent->children.end()) {
                auto node      = std::make_unique<AggNode>(_strings.get(_other_id));
                node->file     = _strings.get(evt.file_id);
                node->line     = evt.line;
                node->function = _strings.get(evt.func_id);

                parent->children[_other_id] = std::move(node);
                other                       = parent->children.find(_other_id);
            }
            AggNode &folded = *other->second;
            if (folded.folded_names.insert(evt.name_id).second) {
                folded.annotations["distinct"] = std::to_string(folded.folded_names.size());
            }
            it = other;
        }
    }
    if (it == parent->children.end()) {
        // The only place ids are resolved to strings: once per distinct call path, not per event.
        auto node      = std::make_unique<AggNode>(_strings.get(evt.name_id));
        node->file     = _strings.get(evt.file_id);
        node->line     = evt.line;
        node->function = _strings.get(evt.func_id);

        parent->children[evt.name_id] = std::move(node);
        it                            = parent->children.find(evt.name_id);
    }
    frame.node = it->second.get();

    ts.stack.push_back(frame);
}

void AggNode::record_exclusive(ns exclusive) {
    call_count += 1;
    total_exclusive += exclusive;

    // Welford's online variance, in double (see the field declarations).
    double const sample = static_cast<double>(exclusive.count());
    double const delta  = sample - total_exclusive_mean;
    total_exclusive_mean += delta / static_cast<double>(call_count);
    double const delta2 = sample - total_exclusive_mean;
    total_exclusive_M2 += delta * delta2;

    if (exclusive < exclusive_min)
        exclusive_min = exclusive;
    if (exclusive > exclusive_max)
        exclusive_max = exclusive;

    // Per-call log2 histogram (buckets in microseconds: bucket i = [2^i, 2^(i+1)) us)
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(exclusive).count();
    if (us < 1)
        us = 1; // clamp to 1us minimum
    int  bucket = 0;
    auto v      = us;
    while (v > 1 && bucket < AggNode::kHistogramBuckets - 1) {
        v >>= 1;
        ++bucket;
    }
    histogram[bucket]++;
}

void Consumer::process_pop(ThreadState &ts, Event const &evt, uint32_t thread_id) {
    // Anything open below the level being closed lost its Pop. The common case: a burst's pops
    // land after the burst has overrun the ring.
    if (evt.depth > 0) {
        unwind_stale_frames(ts, evt.depth);
    }

    if (ts.stack.empty())
        return;

    auto frame = ts.stack.back();
    ts.stack.pop_back();

    TimePoint const end       = TickClock::instance().to_time_point(evt.ticks);
    ns const        duration  = std::chrono::duration_cast<ns>(end - frame.start);
    ns const        exclusive = duration - frame.child_time;

    // The node was resolved at push.
    AggNode *cur = frame.node;
    if (cur == nullptr) {
        return; // push was never processed for this frame
    }

    cur->record_exclusive(exclusive);

    // Merge counter deltas only when a counter backend is active; otherwise they are all zero.
    if (!_counters_checked) {
        auto &backend    = get_counter_backend();
        _counters_active = backend.available();
        for (int i = 0; i < kNumCounterSlots; ++i) {
            _counter_names[i] = backend.slot_name(i);
        }
        _counters_checked = true;
    }
    for (int i = 0; _counters_active && i < kNumCounterSlots; ++i) {
        uint64_t const     counter_delta = evt.counters[i] - frame.counters[i];
        std::string const &cname         = _counter_names[i];
        cur->counters_total[cname] += counter_delta;
        auto itmin = cur->counters_min.find(cname);
        if (itmin == cur->counters_min.end()) {
            cur->counters_min[cname] = counter_delta;
            cur->counters_max[cname] = counter_delta;
        } else {
            if (counter_delta < cur->counters_min[cname])
                cur->counters_min[cname] = counter_delta;
            if (counter_delta > cur->counters_max[cname])
                cur->counters_max[cname] = counter_delta;
        }
    }

    // Record timeline event for Gantt chart
    {
        using ms = std::chrono::duration<double, std::milli>;
        TimelineRecord const rec{.thread_id = thread_id,
                                 .name_id   = frame.name_id,
                                 .start_ms  = std::chrono::duration_cast<ms>(frame.start - _program_start).count(),
                                 .end_ms    = std::chrono::duration_cast<ms>(end - _program_start).count()};
        if (_timeline.size() < kMaxTimelineEvents) {
            _timeline.push_back(rec);
        } else {
            _timeline[_timeline_next] = rec;
        }
        _timeline_next = (_timeline_next + 1) % kMaxTimelineEvents;
    }

    // Add duration to parent's child_time
    if (!ts.stack.empty())
        ts.stack.back().child_time += duration;
}

void Consumer::process_annotate(ThreadState &ts, Event const &evt) {
    if (ts.stack.empty())
        return;

    // The innermost frame already knows its node; no walk, no string lookups.
    AggNode *cur = ts.stack.back().node;
    if (cur == nullptr)
        return; // push was never processed for this frame

    std::string const &key = _strings.get(evt.annotation.key_id);

    switch (evt.annotation.value_type) {
    case AnnotateValueType::String: {
        cur->annotations[key] = _strings.get(evt.annotation.string_id);
        break;
    }
    case AnnotateValueType::Int64: {
        cur->annotations[key] = std::to_string(evt.annotation.int_val);
        auto &na              = cur->numeric_annotations[key];
        na.total += static_cast<double>(evt.annotation.int_val);
        na.count += 1;
        auto val = static_cast<double>(evt.annotation.int_val);
        if (val < na.min_val)
            na.min_val = val;
        if (val > na.max_val)
            na.max_val = val;
        break;
    }
    case AnnotateValueType::Float64: {
        cur->annotations[key] = std::to_string(evt.annotation.float_val);
        auto &na              = cur->numeric_annotations[key];
        na.total += evt.annotation.float_val;
        na.count += 1;
        if (evt.annotation.float_val < na.min_val)
            na.min_val = evt.annotation.float_val;
        if (evt.annotation.float_val > na.max_val)
            na.max_val = evt.annotation.float_val;
        break;
    }
    }
}

void Consumer::process_mem(ThreadState &ts, Event const &evt) {
    if (ts.stack.empty())
        return;

    // The innermost frame already knows its node; no walk, no string lookups.
    AggNode *cur = ts.stack.back().node;
    if (cur == nullptr)
        return; // push was never processed for this frame

    if (evt.type == EventType::MemAlloc) {
        cur->mem_alloc_count += 1;
        cur->mem_alloc_bytes += evt.mem_bytes;
        cur->mem_current_bytes += evt.mem_bytes;
        if (cur->mem_current_bytes > cur->mem_peak_bytes)
            cur->mem_peak_bytes = cur->mem_current_bytes;
    } else {
        cur->mem_free_count += 1;
        cur->mem_free_bytes += evt.mem_bytes;
        cur->mem_current_bytes -= evt.mem_bytes;
    }
}

EINSUMS_NAMESPACE_END(profile)

#endif
