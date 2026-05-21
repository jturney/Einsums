//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Profile/Consumer.hpp>
#include <Einsums/Profile/CounterBackend.hpp>
#include <Einsums/Profile/Event.hpp>
#include <Einsums/Profile/RingBuffer.hpp>
#include <Einsums/Profile/Server.hpp>
#include <Einsums/Profile/StringTable.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TypeSupport/InsertionOrderedMap.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>

#ifdef EINSUMS_HAVE_TRACY
#    include <tracy/Tracy.hpp>
#endif

#if defined _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <malloc.h>
#    include <windows.h>

#    include "TracyWinFamily.hpp"
#else
#    include <cstring>
#    include <pthread.h>
#    include <unistd.h>
#endif

#ifdef __linux__
#    ifdef __ANDROID__
#        include <sys/types.h>
#    else
#        include <sys/syscall.h>
#    endif
#    include <fcntl.h>
#elif defined __FreeBSD__
#    include <sys/thr.h>
#elif defined __NetBSD__
#    include <lwp.h>
#elif defined __DragonFly__
#    include <sys/lwp.h>
#elif defined __QNX__
#    include <process.h>
#    include <sys/neutrino.h>
#endif

namespace einsums::profile {

#if defined(EINSUMS_HAVE_PROFILER)

// ---------------------- Profiler class ----------------------
struct EINSUMS_EXPORT Profiler {
    static auto instance() -> Profiler &;

    // Start a timer region. Optionally provide file/line/func (if available).
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        auto overhead_start = Clock::now();
        auto now            = overhead_start;

#    ifdef EINSUMS_HAVE_TRACY
        auto z =
            std::make_unique<tracy::ScopedZone>(line, file.c_str(), file.size(), func.c_str(), func.size(), name.c_str(), name.size(), 1);
        thread_tracy_zones().push_back(std::move(z));
#    endif

        // Intern strings
        uint32_t const name_id = _strings.intern(name);
        uint32_t const file_id = _strings.intern(file);
        uint32_t const func_id = _strings.intern(func);

        // Ensure thread is registered with consumer
        auto &rb = thread_ring_buffer();

        // Write event to ring buffer
        Event evt{};
        evt.type      = EventType::Push;
        evt.timestamp = now;
        evt.name_id   = name_id;
        evt.file_id   = file_id;
        evt.func_id   = func_id;
        evt.line      = line;

        // Read hardware counters
        auto                                  &counters = get_counter_backend();
        std::array<uint64_t, kNumCounterSlots> cvals;
        counters.read(cvals);
        for (int i = 0; i < kNumCounterSlots; ++i)
            evt.counters[i] = cvals[i];

        if (!rb->try_push(evt)) {
            _consumer->increment_dropped();
        }

        auto overhead_end = Clock::now();
        _push_overhead_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(overhead_end - overhead_start).count()),
            std::memory_order_relaxed);
        _push_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Stop timer region
    void pop() {
        auto overhead_start = Clock::now();
        auto now            = overhead_start;

#    ifdef EINSUMS_HAVE_TRACY
        if (!thread_tracy_zones().empty())
            thread_tracy_zones().pop_back();
#    endif

        auto &rb = thread_ring_buffer();

        Event evt{};
        evt.type      = EventType::Pop;
        evt.timestamp = now;

        // Read hardware counters
        auto                                  &counters = get_counter_backend();
        std::array<uint64_t, kNumCounterSlots> cvals;
        counters.read(cvals);
        for (int i = 0; i < kNumCounterSlots; ++i)
            evt.counters[i] = cvals[i];

        if (!rb->try_push(evt)) {
            _consumer->increment_dropped();
        }

        auto overhead_end = Clock::now();
        _pop_overhead_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(overhead_end - overhead_start).count()),
            std::memory_order_relaxed);
        _pop_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Print default compact report (exclusive time, percent, name, file:line clickable, func)
    // detailed -> show min/max/avg and counters
    void print(bool detailed = false, std::ostream &os = std::cout);

    // JSON & CSV exporters (optional)
    auto export_json(std::string const &path = "einsums_profile.json") -> std::optional<std::string>;

    // Shutdown the consumer thread, server, and do final drain.
    void shutdown() {
        // Clear print output sink before shutting down server to avoid use-after-free on the queue pointer
        einsums::print::clear_output_sink();
        if (_consumer)
            _consumer->shutdown();
        if (_server)
            _server->shutdown();
    }

    // Flush all pending events from ring buffers into the aggregated tree.
    void flush() {
        if (_consumer)
            _consumer->flush();
    }

    // Overhead measurement accessors
    auto avg_push_overhead_ns() const -> double {
        auto c = _push_count.load(std::memory_order_relaxed);
        return c > 0 ? static_cast<double>(_push_overhead_ns.load(std::memory_order_relaxed)) / static_cast<double>(c) : 0.0;
    }
    auto avg_pop_overhead_ns() const -> double {
        auto c = _pop_count.load(std::memory_order_relaxed);
        return c > 0 ? static_cast<double>(_pop_overhead_ns.load(std::memory_order_relaxed)) / static_cast<double>(c) : 0.0;
    }
    auto total_push_count() const -> uint64_t { return _push_count.load(std::memory_order_relaxed); }
    auto total_pop_count() const -> uint64_t { return _pop_count.load(std::memory_order_relaxed); }

    // Access string table (for interning annotation keys/values)
    auto string_table() -> StringTable & { return _strings; }

    // Access consumer (for annotations, shared lock on tree, etc.)
    auto consumer() -> Consumer * { return _consumer.get(); }

    // Access server (for registering request handlers from other modules).
    auto server() -> Server * { return _server.get(); }

    // Get the profiler's thread ID for the calling thread (platform-specific, matches Consumer keys).
    static auto current_thread_id() -> uint32_t { return thread_key(); }

    // Set a human-readable name for the calling thread.
    void set_thread_name(std::string const &name) { _consumer->set_thread_name(thread_key(), name); }

    // Emit an event to the thread-local ring buffer. Used by annotation API.
    void emit_event(Event const &evt) {
        auto &rb = thread_ring_buffer();
        if (!rb->try_push(evt)) {
            _consumer->increment_dropped();
        }
    }

  private:
    Profiler() : _consumer(std::make_unique<Consumer>(_strings)) {
        // Read server port from config (default 19216)
        uint16_t port = 19216;
        try {
            auto &gc = GlobalConfigMap::get_singleton();
            port     = static_cast<uint16_t>(gc.get_int("profiler-port", 19216));
        } catch (...) { // NOLINT
        }
        _server = std::make_unique<Server>(*_consumer, _strings, "127.0.0.1", port);
        _consumer->set_tick_callback([this] { _server->tick(); });
        // Signal handlers are NOT installed here to avoid conflicting with
        // the Runtime module's signal handlers (set_signal_handlers in Runtime.cpp).
        // Profiler shutdown is handled by einsums::finalize() in Finalize.cpp,
        // which calls prof.shutdown() + prof.print() during the shutdown phase.
    }

    // Stop the background consumer thread before members are destroyed. The
    // consumer's periodic tick callback calls into ``_server``; members destruct
    // in reverse declaration order, so ``_server`` would otherwise be torn down
    // while the thread is still ticking, and the thread would dereference a
    // destroyed Server (an intermittent shutdown SIGSEGV). This is the fallback
    // for interpreter/static shutdown when einsums::finalize() — which already
    // calls shutdown() — was not invoked (e.g. a Python process exiting). Both
    // calls are idempotent with finalize()'s.
    ~Profiler() {
        if (_consumer) {
            _consumer->shutdown();
        }
        if (_server) {
            _server->shutdown();
        }
    }

    void write_node_json(std::ostream &ofs, AggNode const &n, int indent);
    void print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed);

    void print_node_recursive(std::ostream &os, AggNode *n, double thread_total_ms, int depth, bool detailed) {
        print_node_recursive(os, static_cast<AggNode const *>(n), thread_total_ms, depth, detailed);
    }

    // ------------------ thread-local ring buffer ------------------
    static auto thread_ring_buffer() -> std::unique_ptr<EventRingBuffer> & {
        thread_local auto rb = [] {
            auto ptr = std::make_unique<EventRingBuffer>();
            auto tid = thread_key();
            // Register with consumer and open hardware counters for this thread
            Profiler::instance()._consumer->register_thread(tid, ptr.get());
            get_counter_backend().open_thread_counters();
            // Auto-name the thread: the first thread to initialize is "main"
            static std::atomic<bool> first_thread{true};
            if (first_thread.exchange(false, std::memory_order_acq_rel)) {
                Profiler::instance()._consumer->set_thread_name(tid, "main");
            } else {
                Profiler::instance()._consumer->set_thread_name(tid, "thread-" + std::to_string(tid));
            }
            return ptr;
        }();
        return rb;
    }

#    ifdef EINSUMS_HAVE_TRACY
    static auto thread_tracy_zones() -> std::vector<std::unique_ptr<tracy::ScopedZone>> & {
        thread_local std::vector<std::unique_ptr<tracy::ScopedZone>> v;
        return v;
    }
#    endif

    // Platform-specific thread ID
    static auto thread_key() -> uint32_t {
#    if defined _WIN32
        static_assert(sizeof(decltype(GetCurrentThreadId())) <= sizeof(uint32_t), "Thread handle too big to fit in protocol");
        return uint32_t(GetCurrentThreadId());
#    elif defined __APPLE__
        uint64_t id;
        pthread_threadid_np(pthread_self(), &id);
        return static_cast<uint32_t>(id);
#    elif defined __ANDROID__
        return (uint32_t)gettid();
#    elif defined __linux__
        return static_cast<uint32_t>(syscall(SYS_gettid));
#    elif defined __FreeBSD__
        long id;
        thr_self(&id);
        return id;
#    elif defined __NetBSD__
        return _lwp_self();
#    elif defined __DragonFly__
        return lwp_gettid();
#    elif defined __OpenBSD__
        return getthrid();
#    elif defined __QNX__
        return (uint32_t)gettid();
#    elif defined __EMSCRIPTEN__
        return 0;
#    else
#        error "Unsupported platform!"
#    endif
    }

    StringTable               _strings;
    std::unique_ptr<Consumer> _consumer;
    std::unique_ptr<Server>   _server;

    // Overhead measurement counters
    std::atomic<uint64_t> _push_overhead_ns{0};
    std::atomic<uint64_t> _pop_overhead_ns{0};
    std::atomic<uint64_t> _push_count{0};
    std::atomic<uint64_t> _pop_count{0};
};

// ---------------------- Scoped helper ----------------------
struct ScopedZone {
    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        Profiler::instance().push(name, file, line, func);
    }
    ~ScopedZone() { Profiler::instance().pop(); }
};

// ---------------------- Annotation API ----------------------

/// Attach a string annotation to the current profiling zone.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void annotate(std::string_view key, std::string_view value) {
    auto &prof = Profiler::instance();
    auto &st   = prof.string_table();

    Event evt{};
    evt.type       = EventType::Annotate;
    evt.timestamp  = Clock::now();
    evt.key_id     = st.intern(key);
    evt.value_type = AnnotateValueType::String;
    evt.string_id  = st.intern(value);

    prof.emit_event(evt);
}

/// Attach an integer annotation to the current profiling zone.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void annotate(std::string_view key, int64_t value) {
    auto &prof = Profiler::instance();
    auto &st   = prof.string_table();

    Event evt{};
    evt.type       = EventType::Annotate;
    evt.timestamp  = Clock::now();
    evt.key_id     = st.intern(key);
    evt.value_type = AnnotateValueType::Int64;
    evt.int_val    = value;

    prof.emit_event(evt);
}

/// Attach a floating-point annotation to the current profiling zone.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void annotate(std::string_view key, double value) {
    auto &prof = Profiler::instance();
    auto &st   = prof.string_table();

    Event evt{};
    evt.type       = EventType::Annotate;
    evt.timestamp  = Clock::now();
    evt.key_id     = st.intern(key);
    evt.value_type = AnnotateValueType::Float64;
    evt.float_val  = value;

    prof.emit_event(evt);
}

/// Attach a vector of dimension sizes as annotations (dim.0, dim.1, ...).
inline void annotate_dims(std::string_view key, std::span<int64_t const> dims) {
    for (size_t i = 0; i < dims.size(); ++i) {
        annotate(fmt::format("{}.{}", key, i), dims[i]);
    }
}

/// Record a memory allocation in the current profiling zone.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void mem_alloc(int64_t bytes) {
    Event evt{};
    evt.type      = EventType::MemAlloc;
    evt.timestamp = Clock::now();
    evt.mem_bytes = bytes;
    Profiler::instance().emit_event(evt);
}

/// Record a memory deallocation in the current profiling zone.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void mem_free(int64_t bytes) {
    Event evt{};
    evt.type      = EventType::MemFree;
    evt.timestamp = Clock::now();
    evt.mem_bytes = bytes;
    Profiler::instance().emit_event(evt);
}

// ---------------------- Python bindings ----------------------
// Thin free-function wrappers around the Profiler singleton so the
// einsums.profile Python submodule can drive push/pop/flush/print without
// having to bind the Profiler class itself (which holds non-copyable
// unique_ptrs and exposes an ostream& on print()).

/// Begin a profile region. Pair with ``pop()`` — usually via the
/// ``einsums.profile.section(name)`` context manager.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void push(std::string const &name, std::string const &file = "", int line = 0,
                                                                        std::string const &func = "") {
    Profiler::instance().push(name, file, line, func);
}

/// End the innermost profile region.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void pop() {
    Profiler::instance().pop();
}

/// Drain all per-thread ring buffers into the aggregated tree. Call before
/// ``print_report`` / ``export_json`` to make sure recent events are visible.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void flush() {
    Profiler::instance().flush();
}

/// Print the compact (or detailed) report to standard output.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void print_report(bool detailed = false) {
    Profiler::instance().print(detailed);
    // Flush std::cout so pytest's capfd (and any non-tty stdout) sees the
    // output before the caller returns. Profiler::print otherwise leaves
    // the data in C++ stdio's userspace buffer.
    std::cout.flush();
}

/// Write the aggregated profile to JSON. Returns the resolved path on
/// success or ``None`` on failure.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline std::optional<std::string> export_json(
    std::string const &path = "einsums_profile.json") {
    return Profiler::instance().export_json(path);
}

/// Set a human-readable name for the calling thread.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline void set_thread_name(std::string const &name) {
    Profiler::instance().set_thread_name(name);
}

/// Return the profiler's thread id for the calling thread.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline uint32_t current_thread_id() {
    return Profiler::current_thread_id();
}

/// Average per-call overhead of ``push`` in nanoseconds.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline double avg_push_overhead_ns() {
    return Profiler::instance().avg_push_overhead_ns();
}

/// Average per-call overhead of ``pop`` in nanoseconds.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline double avg_pop_overhead_ns() {
    return Profiler::instance().avg_pop_overhead_ns();
}

/// Total number of ``push`` calls observed since process start.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline uint64_t total_push_count() {
    return Profiler::instance().total_push_count();
}

/// Total number of ``pop`` calls observed since process start.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("profile") inline uint64_t total_pop_count() {
    return Profiler::instance().total_pop_count();
}

#    define LabeledSection(name_format, ...)                                                                                               \
        ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(fmt::format(name_format __VA_OPT__(, ) __VA_ARGS__),  \
                                                                                     __FILE__, __LINE__, __func__)
#    define LabeledSection0() LabeledSection(__func__)
#    if defined(EINSUMS_WITH_PROFILER_INTERNAL)
#        define LabeledSectionInternal(name_format, ...)                                                                                   \
            ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(                                                  \
                fmt::format(name_format __VA_OPT__(, ) __VA_ARGS__), __FILE__, __LINE__, __func__)
#        define LabeledSectionInternal0() LabeledSectionInternal(__func__)
#    else
#        define LabeledSectionInternal(...)
#        define LabeledSectionInternal0()
#    endif

#    define ProfileAnnotate(key, value)    ::einsums::profile::annotate(key, value)
#    define ProfileAnnotateDims(key, dims) ::einsums::profile::annotate_dims(key, dims)
#    define ProfileMemAlloc(bytes)         ::einsums::profile::mem_alloc(static_cast<int64_t>(bytes))
#    define ProfileMemFree(bytes)          ::einsums::profile::mem_free(static_cast<int64_t>(bytes))

#else
#    define LabeledSection(...)
#    define LabeledSection0()
#    define LabeledSectionInternal(...)
#    define LabeledSectionInternal0()
#    define ProfileAnnotate(key, value)
#    define ProfileAnnotateDims(key, dims)
#    define ProfileMemAlloc(bytes)
#    define ProfileMemFree(bytes)
#endif

} // namespace einsums::profile
