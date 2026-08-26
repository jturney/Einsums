//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

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

EINSUMS_NAMESPACE_BEGIN(profile)

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

    /// Welford's running mean and sum of squared deviations, in nanoseconds.
    ///
    /// Floating point, not integer, for two reasons. The sum of squares
    /// overflows a signed 64-bit integer once a zone's spread reaches a few
    /// seconds: a sample 108 s from the mean squares to ~5.8e21 against an
    /// int64 ceiling of 9.2e18, which is undefined behaviour and was reported
    /// as such by UBSan on the slow sanitizer runs where zones get that long.
    /// And an integer mean advanced by ``delta / call_count`` truncates every
    /// update, so it drifts low even when nothing overflows.
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

    /// Children keyed by INTERNED name id, not by name.
    ///
    /// The producer already interns each zone's name once per call site and
    /// ships a ``uint32_t`` (see ``Profiler::push_interned``, whose comment
    /// explains that interning per entry meant hashing and ``memcmp``-ing the
    /// same strings - including a long absolute ``__FILE__`` - under a shared
    /// mutex millions of times). Keying this map by ``std::string`` undid that
    /// at the consumer: every push and pop resolved the ids back to strings
    /// through the string table's ``shared_mutex`` and then hashed the full
    /// name at every level of the path. A 4-byte key hashes in constant time
    /// and needs no string at all; ``name`` below still carries it for display.
    InsertionOrderedMap<uint32_t, std::unique_ptr<AggNode>> children;

    AggNode() = default;
    explicit AggNode(std::string n) : name(std::move(n)) {}

    /// Frees the subtree with a worklist rather than by recursion.
    ///
    /// The compiler-generated destructor recurses once per level through the
    /// child ``unique_ptr`` members, so freeing the tree costs a stack frame per level
    /// of nesting - and a profiler is exactly the thing that must not turn a
    /// deeply nested program into a crash at exit. One did: a run that dropped
    /// events built a chain over 100k nodes long (see @ref Event::depth) and
    /// died of stack overflow while tearing it down, after the work it was
    /// measuring had finished and passed.
    ~AggNode() {
        std::vector<std::unique_ptr<AggNode>> pending;
        // Moved out rather than erased: the emptied entries own nothing, so
        // destroying the map they sit in frees no node and recurses nowhere.
        auto const detach = [&pending](AggNode &node) {
            for (auto &child : node.children) {
                if (child.second) {
                    pending.push_back(std::move(child.second));
                }
            }
        };

        detach(*this);
        while (!pending.empty()) {
            // Detached before it goes out of scope, so the implicit destructor
            // that runs here always finds its children already gone.
            std::unique_ptr<AggNode> const node = std::move(pending.back());
            pending.pop_back();
            detach(*node);
        }
    }

    AggNode(AggNode const &)            = delete;
    AggNode &operator=(AggNode const &) = delete;
    AggNode(AggNode &&)                 = delete;
    AggNode &operator=(AggNode &&)      = delete;

    /// Fold one measured exclusive duration into this node's statistics:
    /// call count, total, running mean and variance, min/max, and the
    /// per-call log2 histogram.
    ///
    /// Split out of the consumer's pop handler so the arithmetic can be
    /// exercised directly, with durations no test would want to spend real
    /// time producing. Exported for that reason: AggNode is otherwise a plain
    /// aggregate whose members are all inline.
    EINSUMS_EXPORT void record_exclusive(ns exclusive);
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

        /// The aggregation node this frame accumulates into, resolved once when
        /// the zone was pushed.
        ///
        /// Every handler used to rediscover it by walking the tree from the
        /// root, so a zone at depth d cost d map lookups on push, d again on
        /// pop, and d more on every annotation - all of it repeating work the
        /// push had already done. Caching the pointer makes each O(1). It stays
        /// valid because the node is owned by a ``unique_ptr`` in its parent's
        /// map and nothing erases nodes; the tree only grows.
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

    /// Zones abandoned because the events that would have closed them were
    /// dropped. Reported alongside @ref dropped_count so a thinned-out tree is
    /// visibly thinned rather than quietly wrong.
    auto unmatched_zone_count() const -> uint64_t { return _unmatched_zones.load(std::memory_order_relaxed); }

    /// Notify the consumer that new events are available (called by producer after push).
    void notify() { _wake_cv.notify_one(); }

    /// Set a callback to be invoked after each drain cycle (e.g., for server tick).
    /// Guarded by _tick_mutex: the consumer thread is started in the constructor
    /// and reads/calls _tick_callback concurrently with this setter.
    void set_tick_callback(std::function<void()> cb) {
        std::scoped_lock const lock(_tick_mutex);
        _tick_callback = std::move(cb);
    }

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

        // Root to current node, collecting annotations at each level. The
        // frames already carry their nodes, so this needs no tree lookups and
        // no string-table traffic at all.
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
    void drain_all();
    void process_event(uint32_t thread_id, Event const &evt);
    void process_push(ThreadState &ts, Event const &evt);
    void process_pop(ThreadState &ts, Event const &evt, uint32_t thread_id);
    void process_annotate(ThreadState &ts, Event const &evt);
    void process_mem(ThreadState &ts, Event const &evt);

    /// Close frames the producer is no longer inside, without recording them.
    ///
    /// A frame is only ever removed by its own Pop, so a Pop that the ring
    /// buffer dropped strands its frame here for the rest of the run. The depth
    /// each event carries says where the producer actually is, and every frame
    /// deeper than that belongs to a zone whose Pop is never coming: dropping
    /// them keeps this stack the shape of the code being measured instead of
    /// growing one level per lost event.
    void unwind_stale_frames(ThreadState &ts, size_t depth);

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

    // Zones whose Pop was among the dropped events (see unwind_stale_frames).
    std::atomic<uint64_t> _unmatched_zones{0};

    // Tick callback (e.g., for server). Set on the main thread after the consumer
    // thread is already running, so access is guarded by _tick_mutex.
    std::mutex            _tick_mutex;
    std::function<void()> _tick_callback;

    // Timeline events for Gantt chart (circular buffer, protected by tree_mutex_)
    std::vector<TimelineEvent> _timeline_events;
    TimePoint                  _program_start{std::chrono::steady_clock::now()};
};

EINSUMS_NAMESPACE_END(profile)

#endif
