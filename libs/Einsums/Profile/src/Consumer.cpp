//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Consumer.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

namespace einsums::profile {

Consumer::Consumer(StringTable &strings) : _strings(strings) {
    _running.store(true, std::memory_order_relaxed);
    _thread = std::thread([this] { consumer_loop(); });
}

Consumer::~Consumer() {
    shutdown();
}

void Consumer::register_thread(uint32_t thread_id, std::shared_ptr<EventRingBuffer> rb) {
    std::scoped_lock const lock(_reg_mutex);
    _registrations.push_back({.thread_id = thread_id, .ring_buffer = std::move(rb)});
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
    while (_running.load(std::memory_order_relaxed)) {
        drain_all();
        // Call tick callback every ~500ms. Snapshot it under the lock (the main
        // thread may install it after this loop has started) and invoke outside.
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
        // Wait for notification from producers or timeout after 1ms.
        // If notified (e.g., by flush() or shutdown()), wake immediately.
        std::unique_lock lock(_wake_mutex);
        _wake_cv.wait_for(lock, std::chrono::milliseconds(1));
    }
}

void Consumer::drain_all() {
    // Snapshot registrations
    std::vector<ThreadRegistration> regs;
    {
        std::scoped_lock const lock(_reg_mutex);
        regs = _registrations;
    }

    if (regs.empty())
        return;

    // Drain events from all ring buffers under a single exclusive lock
    std::unique_lock const lock(_tree_mutex);
    for (auto &reg : regs) {
        Event evt;
        while (reg.ring_buffer->try_pop(evt)) {
            process_event(reg.thread_id, evt);
        }
    }
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

void Consumer::process_push(ThreadState &ts, Event const &evt) {
    ThreadState::StackFrame frame{};
    frame.name_id    = evt.name_id;
    frame.file_id    = evt.file_id;
    frame.func_id    = evt.func_id;
    frame.line       = evt.line;
    frame.child_time = ns{0};
    frame.start      = evt.timestamp;
    for (int i = 0; i < kNumCounterSlots; ++i)
        frame.counters[i] = evt.counters[i];
    ts.stack.push_back(frame);

    // Eagerly create the tree path so annotations can find the node
    AggNode *cur = &ts.root;
    for (auto &f : ts.stack) {
        auto const &name = _strings.get(f.name_id);
        auto        it   = cur->children.find(name);
        if (it == cur->children.end()) {
            cur->children[name] = std::make_unique<AggNode>(name);
            it                  = cur->children.find(name);
        }
        cur = it->second.get();
    }
}

void Consumer::process_pop(ThreadState &ts, Event const &evt, uint32_t thread_id) {
    if (ts.stack.empty())
        return;

    auto frame = ts.stack.back();
    ts.stack.pop_back();

    ns const duration  = std::chrono::duration_cast<ns>(evt.timestamp - frame.start);
    ns const exclusive = duration - frame.child_time;

    // Build path from root to this node
    std::vector<std::string const *> path;
    path.reserve(ts.stack.size());
    for (auto &f : ts.stack) {
        path.push_back(&_strings.get(f.name_id));
    }
    path.push_back(&_strings.get(frame.name_id));

    // Walk tree, creating nodes as needed
    AggNode *cur = &ts.root;
    for (auto *name_ptr : path) {
        auto &name = *name_ptr;
        auto  it   = cur->children.find(name);
        if (it == cur->children.end()) {
            cur->children[name] = std::make_unique<AggNode>(name);
            it                  = cur->children.find(name);
        }
        cur = it->second.get();
    }

    cur->file     = _strings.get(frame.file_id);
    cur->line     = frame.line;
    cur->function = _strings.get(frame.func_id);

    cur->call_count += 1;
    cur->total_exclusive += exclusive;

    // Welford's online variance
    int64_t const delta = exclusive.count() - cur->total_exclusive_mean;
    cur->total_exclusive_mean += delta / static_cast<int64_t>(cur->call_count);
    int64_t const delta2 = exclusive.count() - cur->total_exclusive_mean;
    cur->total_exclusive_M2 += delta * delta2;

    if (exclusive < cur->exclusive_min)
        cur->exclusive_min = exclusive;
    if (exclusive > cur->exclusive_max)
        cur->exclusive_max = exclusive;

    // Per-call log2 histogram (buckets in microseconds: bucket i = [2^i, 2^(i+1)) us)
    {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(exclusive).count();
        if (us < 1)
            us = 1; // clamp to 1us minimum
        int  bucket = 0;
        auto v      = us;
        while (v > 1 && bucket < AggNode::kHistogramBuckets - 1) {
            v >>= 1;
            ++bucket;
        }
        cur->histogram[bucket]++;
    }

    // Merge hardware counter deltas
    auto &counter_backend = get_counter_backend();
    for (int i = 0; i < kNumCounterSlots; ++i) {
        uint64_t const    counter_delta = evt.counters[i] - frame.counters[i];
        std::string const cname         = counter_backend.slot_name(i);
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
        auto start_since_program = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame.start - _program_start);
        auto end_since_program   = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(evt.timestamp - _program_start);
        TimelineEvent te;
        te.thread_id = thread_id;
        te.name      = _strings.get(frame.name_id);
        te.start_ms  = start_since_program.count();
        te.end_ms    = end_since_program.count();
        if (_timeline_events.size() >= kMaxTimelineEvents) {
            _timeline_events.erase(_timeline_events.begin());
        }
        _timeline_events.push_back(std::move(te));
    }

    // Add duration to parent's child_time
    if (!ts.stack.empty())
        ts.stack.back().child_time += duration;
}

void Consumer::process_annotate(ThreadState &ts, Event const &evt) {
    if (ts.stack.empty())
        return;

    // Find the current node in the tree by walking the stack path
    AggNode *cur = &ts.root;
    for (auto &f : ts.stack) {
        auto &name = _strings.get(f.name_id);
        auto  it   = cur->children.find(name);
        if (it == cur->children.end())
            return; // node doesn't exist yet (shouldn't happen if push was processed)
        cur = it->second.get();
    }

    std::string const &key = _strings.get(evt.key_id);

    switch (evt.value_type) {
    case AnnotateValueType::String: {
        cur->annotations[key] = _strings.get(evt.string_id);
        break;
    }
    case AnnotateValueType::Int64: {
        cur->annotations[key] = std::to_string(evt.int_val);
        auto &na              = cur->numeric_annotations[key];
        na.total += static_cast<double>(evt.int_val);
        na.count += 1;
        auto val = static_cast<double>(evt.int_val);
        if (val < na.min_val)
            na.min_val = val;
        if (val > na.max_val)
            na.max_val = val;
        break;
    }
    case AnnotateValueType::Float64: {
        cur->annotations[key] = std::to_string(evt.float_val);
        auto &na              = cur->numeric_annotations[key];
        na.total += evt.float_val;
        na.count += 1;
        if (evt.float_val < na.min_val)
            na.min_val = evt.float_val;
        if (evt.float_val > na.max_val)
            na.max_val = evt.float_val;
        break;
    }
    }
}

void Consumer::process_mem(ThreadState &ts, Event const &evt) {
    if (ts.stack.empty())
        return;

    // Find the current node in the tree by walking the stack path
    AggNode *cur = &ts.root;
    for (auto &f : ts.stack) {
        auto &name = _strings.get(f.name_id);
        auto  it   = cur->children.find(name);
        if (it == cur->children.end())
            return;
        cur = it->second.get();
    }

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

} // namespace einsums::profile

#endif
