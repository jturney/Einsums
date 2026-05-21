//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <Einsums/Profile/CounterBackend.hpp>
#    include <Einsums/Profile/Event.hpp>
#    include <Einsums/Profile/RingBuffer.hpp>
#    include <Einsums/Profile/StringTable.hpp>
#    include <Einsums/TypeSupport/InsertionOrderedMap.hpp>

#    include <atomic>
#    include <chrono>
#    include <condition_variable>
#    include <cstdint>
#    include <functional>
#    include <limits>
#    include <map>
#    include <memory>
#    include <mutex>
#    include <shared_mutex>
#    include <string>
#    include <thread>
#    include <unordered_map>
#    include <vector>

namespace einsums::profile {

/// Event ring buffer capacity per thread (64K entries).
static constexpr size_t kRingBufferCapacity = 65536;

using EventRingBuffer = RingBuffer<Event, kRingBufferCapacity>;

// ---------------------- Aggregation node ----------------------
struct AggNode {
    std::string name;
    std::string file;
    int         line = 0;
    std::string function;

    // counts and times (ns)
    uint64_t call_count = 0;
    ns       total_exclusive{0};

    // Welford's running variance (values in nanoseconds)
    int64_t total_exclusive_mean{0};
    int64_t total_exclusive_M2{0};

    // min/max for exclusive time
    ns exclusive_min{std::numeric_limits<int64_t>::max()};
    ns exclusive_max{0};

    // counters aggregate: name -> total/min/max
    std::map<std::string, uint64_t> counters_total;
    std::map<std::string, uint64_t> counters_min;
    std::map<std::string, uint64_t> counters_max;

    // Structured annotations (insertion-ordered so display matches source order)
    InsertionOrderedMap<std::string, std::string> annotations;
    struct NumericAnnotation {
        double   total{0};
        double   min_val{std::numeric_limits<double>::max()};
        double   max_val{std::numeric_limits<double>::lowest()};
        uint64_t count{0};
    };
    InsertionOrderedMap<std::string, NumericAnnotation> numeric_annotations;

    // Memory tracking
    uint64_t mem_alloc_count{0};
    uint64_t mem_free_count{0};
    int64_t  mem_alloc_bytes{0};
    int64_t  mem_free_bytes{0};
    int64_t  mem_current_bytes{0}; // alloc - free (net live bytes within zone)
    int64_t  mem_peak_bytes{0};    // high-water mark of mem_current_bytes

    // Per-call log2 histogram: 21 buckets from 1us to ~2s (bucket i = [2^i us, 2^(i+1) us))
    static constexpr int kHistogramBuckets = 21;
    uint64_t             histogram[kHistogramBuckets]{}; // NOLINT(modernize-avoid-c-arrays)

    InsertionOrderedMap<std::string, std::unique_ptr<AggNode>> children;

    AggNode() = default;
    explicit AggNode(std::string n) : name(std::move(n)) {}
};

// ---------------------- Timeline event for Gantt chart ----------------------
struct TimelineEvent {
    uint32_t    thread_id;
    std::string name;
    double      start_ms; // relative to program start
    double      end_ms;
};

// ---------------------- Per-thread state reconstructed by consumer ----------------------
struct ThreadState {
    struct StackFrame {
        uint32_t  name_id;
        uint32_t  file_id;
        uint32_t  func_id;
        int       line;
        ns        child_time{0};
        TimePoint start;
        uint64_t  counters[kNumCounterSlots]{}; // NOLINT(modernize-avoid-c-arrays)
    };

    std::string             name;
    std::vector<StackFrame> stack;
    AggNode                 root;
};

// ---------------------- Thread registration info ----------------------
struct ThreadRegistration {
    uint32_t                         thread_id;
    std::shared_ptr<EventRingBuffer> ring_buffer; ///< Shared so it outlives a transient producer thread.
};

// ---------------------- Consumer thread ----------------------
class EINSUMS_EXPORT Consumer {
  public:
    Consumer(StringTable &strings);
    ~Consumer();

    // Non-copyable
    Consumer(Consumer const &)            = delete;
    Consumer &operator=(Consumer const &) = delete;

    /// Register a thread's ring buffer. Called once per thread on first push().
    void register_thread(uint32_t thread_id, std::shared_ptr<EventRingBuffer> rb);

    /// Set a human-readable name for a thread.
    void set_thread_name(uint32_t thread_id, std::string name);

    /// Get the human-readable name for a thread (empty string if not set).
    /// Caller must hold shared lock on tree.
    auto thread_name(uint32_t thread_id) const -> std::string {
        auto it = _thread_names.find(thread_id);
        return (it != _thread_names.end()) ? it->second : std::string{};
    }

    /// Stop the consumer thread and drain remaining events.
    void shutdown();

    /// Force an immediate drain of all ring buffers. Blocks until complete.
    /// Use this before reading the tree to ensure all pending events are processed.
    void flush();

    /// Access the aggregated tree (under shared lock for concurrent readers).
    auto lock_shared() -> std::shared_lock<std::shared_mutex> { return std::shared_lock<std::shared_mutex>(_tree_mutex); }

    /// Get thread data map (caller must hold shared lock).
    auto thread_data() const -> std::unordered_map<uint32_t, ThreadState> const & { return _threads; }

    /// Number of events dropped across all threads.
    auto dropped_count() const -> uint64_t { return _dropped.load(std::memory_order_relaxed); }

    /// Increment dropped counter (called by producer when ring buffer is full).
    void increment_dropped() { _dropped.fetch_add(1, std::memory_order_relaxed); }

    /// Notify the consumer that new events are available (called by producer after push).
    void notify() { _wake_cv.notify_one(); }

    /// Set a callback to be invoked after each drain cycle (e.g., for server tick).
    void set_tick_callback(std::function<void()> cb) { _tick_callback = std::move(cb); }

    /// Get recent timeline events for Gantt chart. Caller must hold shared lock.
    auto timeline_events() const -> std::vector<TimelineEvent> const & { return _timeline_events; }

    /// Maximum number of timeline events to keep.
    static constexpr size_t kMaxTimelineEvents = 1000;

    /// Collect all annotations from the current zone and all ancestor zones for the given thread.
    /// Child annotations override parent annotations with the same key.
    /// Caller must hold shared lock.
    auto collect_zone_annotations(uint32_t thread_id) const -> InsertionOrderedMap<std::string, std::string> {
        InsertionOrderedMap<std::string, std::string> merged;
        auto                                          it = _threads.find(thread_id);
        if (it == _threads.end())
            return merged;

        auto const &ts = it->second;
        if (ts.stack.empty())
            return merged;

        // Walk from root to current node, collecting annotations at each level
        AggNode const *cur = &ts.root;
        for (auto const &frame : ts.stack) {
            auto const &name     = _strings.get(frame.name_id);
            auto        child_it = cur->children.find(name);
            if (child_it == cur->children.end())
                break;
            cur = child_it->second.get();
            // Merge this node's annotations (child overrides parent)
            for (auto const &[key, val] : cur->annotations) {
                merged[key] = val;
            }
        }
        return merged;
    }

  private:
    void consumer_loop();
    void drain_all();
    void process_event(uint32_t thread_id, Event const &evt);
    void process_push(ThreadState &ts, Event const &evt);
    void process_pop(ThreadState &ts, Event const &evt, uint32_t thread_id);
    void process_annotate(ThreadState &ts, Event const &evt);
    void process_mem(ThreadState &ts, Event const &evt);

    StringTable &_strings;

    // Registered ring buffers (protected by reg_mutex_)
    std::mutex                      _reg_mutex;
    std::vector<ThreadRegistration> _registrations;

    // Aggregated tree (protected by tree_mutex_)
    std::shared_mutex                         _tree_mutex;
    std::unordered_map<uint32_t, ThreadState> _threads;
    std::unordered_map<uint32_t, std::string> _thread_names;

    // Consumer thread
    std::atomic<bool>       _running{false};
    std::thread             _thread;
    std::mutex              _wake_mutex;
    std::condition_variable _wake_cv;

    // Dropped event counter
    std::atomic<uint64_t> _dropped{0};

    // Tick callback (e.g., for server)
    std::function<void()> _tick_callback;

    // Timeline events for Gantt chart (circular buffer, protected by tree_mutex_)
    std::vector<TimelineEvent> _timeline_events;
    TimePoint                  _program_start{std::chrono::steady_clock::now()};
};

} // namespace einsums::profile

#endif
