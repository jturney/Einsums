//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CounterBackend.hpp"
#include "Detail/InsertionOrderedMap.hpp"
#include "Event.hpp"
#include "RingBuffer.hpp"
#include "Sites.hpp"
#include "StringTable.hpp"

WAGGLE_NAMESPACE_BEGIN

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

    /// Welford's running mean and sum of squared deviations, in nanoseconds. Double, not integer: an
    /// int64 sum of squares overflows once a zone's spread reaches seconds, and an integer mean
    /// truncates every update.
    double total_exclusive_mean{0.0};
    double total_exclusive_M2{0.0};

    // min/max for exclusive time
    ns exclusive_min{std::numeric_limits<int64_t>::max()};
    ns exclusive_max{0};

    // counters aggregate: name -> total/min/max
    std::map<std::string, uint64_t> counters_total;
    std::map<std::string, uint64_t> counters_min;
    std::map<std::string, uint64_t> counters_max;

    // Structured annotations (insertion-ordered so display matches source order)
    detail::InsertionOrderedMap<std::string, std::string> annotations;
    struct NumericAnnotation {
        double   total{0};
        double   min_val{std::numeric_limits<double>::max()};
        double   max_val{std::numeric_limits<double>::lowest()};
        uint64_t count{0};
    };
    detail::InsertionOrderedMap<std::string, NumericAnnotation> numeric_annotations;

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

    /// Children keyed by interned name id, so aggregation never resolves or hashes a string.
    detail::InsertionOrderedMap<uint32_t, std::unique_ptr<AggNode>> children;

    /// Set only on a parent's "(other)" node: the ids of the names folded into it once the parent
    /// held Settings::max_distinct_children named children. Its size is the node's "distinct"
    /// annotation.
    std::unordered_set<uint32_t> folded_names;

    AggNode() = default;
    explicit AggNode(std::string n) : name(std::move(n)) {}

    /// Frees the subtree with a worklist: the implicit destructor recurses once per level, and a
    /// deep enough tree overflows the stack at exit.
    ~AggNode() {
        std::vector<std::unique_ptr<AggNode>> pending;
        // Moved out, not erased: the emptied map then frees nothing recursively.
        auto const detach = [&pending](AggNode &node) {
            for (auto &child : node.children) {
                if (child.second) {
                    pending.push_back(std::move(child.second));
                }
            }
        };

        detach(*this);
        while (!pending.empty()) {
            // Detached first, so the implicit destructor finds no children.
            std::unique_ptr<AggNode> const node = std::move(pending.back());
            pending.pop_back();
            detach(*node);
        }
    }

    AggNode(AggNode const &)            = delete;
    AggNode &operator=(AggNode const &) = delete;
    AggNode(AggNode &&)                 = delete;
    AggNode &operator=(AggNode &&)      = delete;

    /// Fold one exclusive duration into this node's statistics: count, total, mean and variance,
    /// min/max and the log2 histogram. Exported so tests can feed it durations directly.
    WAGGLE_EXPORT void record_exclusive(ns exclusive);
};

/// The time spent in @p node and everything below it: its exclusive time plus its descendants'.
/// Walks with an explicit stack, as recursion overflowed the stack under TSan on deep trees.
WAGGLE_EXPORT auto inclusive_time(AggNode const &node) -> ns;

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
        uint32_t  name_id; ///< the zone's name: the event's run-time name, else its site's
        ns        child_time{0};
        TimePoint start;
        uint64_t  counters[kNumCounterSlots]{}; // NOLINT(modernize-avoid-c-arrays)

        /// The node this frame accumulates into, resolved at push. Stays valid because nodes are
        /// never erased.
        AggNode *node{nullptr};
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
class WAGGLE_EXPORT Consumer {
  public:
    Consumer(StringTable &strings, SiteTable const &sites);
    ~Consumer();

    // Non-copyable
    Consumer(Consumer const &)            = delete;
    Consumer &operator=(Consumer const &) = delete;

    /// Register a thread's ring buffer. Called once per thread on first push().
    void register_thread(uint32_t thread_id, std::shared_ptr<EventRingBuffer> rb);

    /// Set a human-readable name for a thread.
    void set_thread_name(uint32_t thread_id, std::string name);

    /// Name @p thread_id @p name unless something already named it: an automatic name must not
    /// replace one a program gave the thread before its first zone.
    void name_thread_if_unnamed(uint32_t thread_id, std::string name);

    /// Get the human-readable name for a thread (empty string if not set).
    /// Caller must hold shared lock on tree.
    auto thread_name(uint32_t thread_id) const -> std::string {
        auto it = _thread_names.find(thread_id);
        return (it != _thread_names.end()) ? it->second : std::string{};
    }

    /// Stop the consumer thread and drain remaining events.
    void shutdown();

    /// Drain all ring buffers now, blocking until done. Call before reading the tree.
    void flush();

    /// Access the aggregated tree (under shared lock for concurrent readers).
    auto lock_shared() -> std::shared_lock<std::shared_mutex> { return std::shared_lock<std::shared_mutex>(_tree_mutex); }

    /// Get thread data map (caller must hold shared lock).
    auto thread_data() const -> std::unordered_map<uint32_t, ThreadState> const & { return _threads; }

    /// Number of events dropped across all threads: the pushes each thread's ring refused, summed.
    auto dropped_count() const -> uint64_t;

    /// Zones abandoned because their closing events were dropped.
    auto unmatched_zone_count() const -> uint64_t { return _unmatched_zones.load(std::memory_order_relaxed); }

    /// Notify the consumer that new events are available (called by producer after push).
    void notify() { _wake_cv.notify_one(); }

    /// Distinct child names a node keeps before the rest fold into its "(other)" node; 0 for no
    /// limit. Safe while the consumer runs.
    void set_max_distinct_children(std::int64_t cap) { _max_distinct_children.store(cap, std::memory_order_relaxed); }

    /// Set a callback run after each drain cycle (the server's tick). Safe while the consumer runs.
    void set_tick_callback(std::function<void()> cb) {
        std::scoped_lock const lock(_tick_mutex);
        _tick_callback = std::move(cb);
    }

    /// The most recent zones for the Gantt chart, oldest first, with their names resolved. Caller must
    /// hold the shared lock.
    auto timeline_events() const -> std::vector<TimelineEvent>;

    /// Maximum number of timeline events to keep.
    static constexpr size_t kMaxTimelineEvents = 1000;

    /// Collect all annotations from the current zone and all ancestor zones for the given thread.
    /// Child annotations override parent annotations with the same key.
    /// Caller must hold shared lock.
    auto collect_zone_annotations(uint32_t thread_id) const -> detail::InsertionOrderedMap<std::string, std::string> {
        detail::InsertionOrderedMap<std::string, std::string> merged;
        auto                                                  it = _threads.find(thread_id);
        if (it == _threads.end())
            return merged;

        auto const &ts = it->second;
        if (ts.stack.empty())
            return merged;

        // Root to current node; the frames carry their nodes, so no tree lookups are needed.
        for (auto const &frame : ts.stack) {
            if (frame.node == nullptr) {
                break;
            }
            // Merge this node's annotations (child overrides parent)
            for (auto const &[key, val] : frame.node->annotations) {
                merged[key] = val;
            }
        }
        return merged;
    }

  private:
    void consumer_loop();
    /// Drain every ring into the tree; returns how many events it processed.
    size_t drain_all();
    void   process_event(uint32_t thread_id, Event const &evt);
    void   process_push(ThreadState &ts, Event const &evt);
    void   process_pop(ThreadState &ts, Event const &evt, uint32_t thread_id);
    void   process_annotate(ThreadState &ts, Event const &evt);
    void   process_mem(ThreadState &ts, Event const &evt);

    /// Close, unrecorded, the frames deeper than @p depth: their Pops were dropped and will never come.
    void unwind_stale_frames(ThreadState &ts, size_t depth);

    StringTable     &_strings;
    SiteTable const &_sites;

    /// The id of "(other)", the node a parent's names fold into past its distinct-name cap.
    uint32_t _other_id;

    // Registered ring buffers (protected by reg_mutex_)
    mutable std::mutex              _reg_mutex;
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

    // Zones whose Pop was among the dropped events (see unwind_stale_frames).
    std::atomic<uint64_t> _unmatched_zones{0};

    // Tick callback (the server's), installed after the consumer thread starts.
    std::mutex            _tick_mutex;
    std::function<void()> _tick_callback;

    std::atomic<std::int64_t> _max_distinct_children{256};

    /// The last kMaxTimelineEvents closed zones, for the Gantt chart: a ring written at
    /// _timeline_next, under _tree_mutex.
    struct TimelineRecord {
        uint32_t thread_id;
        uint32_t name_id;
        double   start_ms;
        double   end_ms;
    };
    std::vector<TimelineRecord> _timeline;
    size_t                      _timeline_next{0};

    /// The counter backend's slot names, read once, and whether it is active at all.
    bool                                      _counters_checked{false};
    bool                                      _counters_active{false};
    std::array<std::string, kNumCounterSlots> _counter_names;
    TimePoint                                 _program_start{std::chrono::steady_clock::now()};
};

WAGGLE_NAMESPACE_END
