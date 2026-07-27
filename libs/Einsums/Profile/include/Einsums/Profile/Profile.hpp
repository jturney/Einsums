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

    /// Whether zones and annotations are recorded. Checked first in every
    /// instrumentation entry point so a disabled profiler costs one relaxed load.
    [[nodiscard]] bool enabled() const { return _enabled.load(std::memory_order_relaxed); }
    void               set_enabled(bool on) { _enabled.store(on, std::memory_order_relaxed); }

    // Start a timer region. Optionally provide file/line/func (if available).
    // Interns on every call; prefer the pre-interned overload below, which is what
    // LabeledSection uses.
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        if (!enabled()) {
            return;
        }
        push_interned(_strings.intern(name), _strings.intern(file), _strings.intern(func), line, name, file, func);
    }

    /// Start a timer region from ALREADY INTERNED ids.
    ///
    /// A zone's name, file and function are compile-time constants at the call
    /// site, so interning them per entry means hashing and ``memcmp``-ing the same
    /// strings (including a long absolute ``__FILE__`` path) under a shared mutex
    /// millions of times. @ref ZoneSite interns once per site and hands the ids
    /// here. The trailing string views are only read by the Tracy backend.
    void push_interned(uint32_t name_id, uint32_t file_id, uint32_t func_id, int line, std::string_view name = {},
                       std::string_view file = {}, std::string_view func = {}) {
        if (!enabled()) {
            return;
        }
        auto overhead_start = Clock::now();
        auto now            = overhead_start;

#    ifdef EINSUMS_HAVE_TRACY
        auto z = std::make_unique<tracy::ScopedZone>(line, file.data(), file.size(), func.data(), func.size(), name.data(), name.size(), 1);
        thread_tracy_zones().push_back(std::move(z));
#    else
        (void)name;
        (void)file;
        (void)func;
#    endif

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
        if (!enabled()) {
            return;
        }
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
            // --einsums:profile:disable. Recording every zone and annotation is not
            // free: on small operations it dominates, so a run that does not want a
            // profile should be able to say so and pay one relaxed load per zone.
            _enabled.store(!gc.get_bool("profile-disable", false), std::memory_order_relaxed);
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
    // for interpreter/static shutdown when einsums::finalize(), which already
    // calls shutdown(), was not invoked, such as a Python process exiting. Both
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
    // The ring buffer is shared with the Consumer: a producer thread (e.g. a
    // transient TaskPool worker) can exit while the Consumer's drain thread is
    // still popping residual events, so ownership must outlive the thread. With
    // a thread_local unique_ptr the buffer was freed on thread exit while the
    // Consumer held a raw pointer to it, a heap-use-after-free (caught by ASan
    // via the DataflowExecutor). Shared ownership lets the buffer live until the
    // last of {producer thread, Consumer} releases it.
    static auto thread_ring_buffer() -> std::shared_ptr<EventRingBuffer> & {
        thread_local auto rb = [] {
            auto ptr = std::make_shared<EventRingBuffer>();
            auto tid = thread_key();
            // Register with consumer (shares ownership) and open hardware counters.
            Profiler::instance()._consumer->register_thread(tid, ptr);
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
    /// Recording switch. On by default so the default report keeps working;
    /// --einsums:profile:disable turns it off, which reduces every zone and
    /// annotation to one relaxed load.
    std::atomic<bool>     _enabled{true};
    std::atomic<uint64_t> _push_overhead_ns{0};
    std::atomic<uint64_t> _pop_overhead_ns{0};
    std::atomic<uint64_t> _push_count{0};
    std::atomic<uint64_t> _pop_count{0};
};

// ---------------------- Scoped helper ----------------------
/**
 * @brief The interned identity of one instrumentation site's location.
 *
 * ``__FILE__`` and ``__func__`` are compile-time constants where a zone is
 * written, so interning them on every entry re-hashes and re-``memcmp``s the same
 * strings under a shared mutex - and ``__FILE__`` is a long absolute path.
 * @ref LabeledSection declares one of these as a function-local static, paying
 * that once per site for the life of the process.
 *
 * The name is cached too, which is what makes a plain zone entry take no locks at
 * all: @ref StringTable::intern is the only mutex on this path (the event ring
 * buffer itself never blocks a producer), so interning nothing means locking
 * nothing. That holds only because @ref LabeledSection's name is a literal. A
 * name built per call keeps its own path: with format arguments the formatted
 * string is interned per entry, and a name computed at runtime uses
 * @ref LabeledSectionRuntime.
 *
 * The views are kept for the Tracy backend, which wants the characters; they
 * point at the literals the macro passes, which outlive the site.
 */
struct ZoneSite {
    ZoneSite(std::string_view name_, char const *file_, int line_, char const *func_) : name{name_}, file{file_}, func{func_}, line{line_} {
        auto &st = Profiler::instance().string_table();
        name_id  = st.intern(name);
        file_id  = st.intern(file);
        func_id  = st.intern(func);
    }

    std::string_view name;
    std::string_view file;
    std::string_view func;
    int              line{0};
    uint32_t         name_id{0};
    uint32_t         file_id{0};
    uint32_t         func_id{0};
};

struct ScopedZone {
    /**
     * @brief Enter a zone at @p site named by @p name (plus any format arguments).
     *
     * One constructor covers the three ways a name arrives, chosen at compile
     * time so the common case does no work it does not need:
     * - a literal with no arguments is interned straight from its ``string_view``,
     *   with no ``fmt::format`` call and so no allocation;
     * - a literal with arguments is formatted per call, as it must be;
     * - anything else (notably ``fmt::runtime``) is formatted per call too, which
     *   is what keeps a runtime-named zone correctly labelled.
     */
    /// Enter a zone whose name is fixed at the call site. Nothing is interned, so
    /// nothing is locked: the site already holds every id, and the event ring
    /// buffer is lock-free.
    explicit ScopedZone(ZoneSite const &site) {
        Profiler::instance().push_interned(site.name_id, site.file_id, site.func_id, site.line, site.name, site.file, site.func);
    }

    /// Enter a zone whose name is built per call (format arguments, or a name
    /// computed at runtime). Only the name is interned; the location comes from
    /// the site.
    ///
    /// The name arrives as a CALLABLE so that building it is skipped entirely when
    /// recording is off. Passing the string directly would evaluate it as an
    /// argument, i.e. before this constructor could check: PackedGemm's zone name
    /// formats three fmt::join views on every contraction, pure waste in a run that
    /// is not profiling.
    template <typename MakeName>
        requires std::invocable<MakeName>
    ScopedZone(ZoneSite const &site, MakeName &&make_name) {
        auto &prof = Profiler::instance();
        if (!prof.enabled()) {
            return;
        }
        std::string const name = make_name();
        prof.push_interned(prof.string_table().intern(name), site.file_id, site.func_id, site.line, name, site.file, site.func);
    }

    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        Profiler::instance().push(name, file, line, func);
    }
    ~ScopedZone() { Profiler::instance().pop(); }
};

// ---------------------- Annotation API ----------------------

/// Attach a string annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate(std::string_view key, std::string_view value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    auto &st = prof.string_table();

    Event evt{};
    evt.type       = EventType::Annotate;
    evt.timestamp  = Clock::now();
    evt.key_id     = st.intern(key);
    evt.value_type = AnnotateValueType::String;
    evt.string_id  = st.intern(value);

    prof.emit_event(evt);
}

/// Attach an integer annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate(std::string_view key, int64_t value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    auto &st = prof.string_table();

    Event evt{};
    evt.type       = EventType::Annotate;
    evt.timestamp  = Clock::now();
    evt.key_id     = st.intern(key);
    evt.value_type = AnnotateValueType::Int64;
    evt.int_val    = value;

    prof.emit_event(evt);
}

/// Attach a floating-point annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate(std::string_view key, double value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    auto &st = prof.string_table();

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
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_alloc(int64_t bytes) {
    Event evt{};
    evt.type      = EventType::MemAlloc;
    evt.timestamp = Clock::now();
    evt.mem_bytes = bytes;
    Profiler::instance().emit_event(evt);
}

/// Record a memory deallocation in the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_free(int64_t bytes) {
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

/// Begin a profile region. Pair it with ``pop()``, usually through the
/// ``einsums.profile.section(name)`` context manager.
APIARY_EXPOSE APIARY_MODULE("profile") inline void push(std::string const &name, std::string const &file = "", int line = 0,
                                                        std::string const &func = "") {
    Profiler::instance().push(name, file, line, func);
}

/// End the innermost profile region.
APIARY_EXPOSE APIARY_MODULE("profile") inline void pop() {
    Profiler::instance().pop();
}

/// Drain all per-thread ring buffers into the aggregated tree. Call before
/// ``print_report`` / ``export_json`` to make sure recent events are visible.
APIARY_EXPOSE APIARY_MODULE("profile") inline void flush() {
    Profiler::instance().flush();
}

/// Print the compact (or detailed) report to standard output.
APIARY_EXPOSE APIARY_MODULE("profile") inline void print_report(bool detailed = false) {
    Profiler::instance().print(detailed);
    // Flush std::cout so pytest's capfd (and any non-tty stdout) sees the
    // output before the caller returns. Profiler::print otherwise leaves
    // the data in C++ stdio's userspace buffer.
    std::cout.flush();
}

/// Write the aggregated profile to JSON. Returns the resolved path on
/// success or ``None`` on failure.
APIARY_EXPOSE APIARY_MODULE("profile") inline std::optional<std::string> export_json(std::string const &path = "einsums_profile.json") {
    return Profiler::instance().export_json(path);
}

/// Set a human-readable name for the calling thread.
APIARY_EXPOSE APIARY_MODULE("profile") inline void set_thread_name(std::string const &name) {
    Profiler::instance().set_thread_name(name);
}

/// Return the profiler's thread id for the calling thread.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint32_t current_thread_id() {
    return Profiler::current_thread_id();
}

/// Average per-call overhead of ``push`` in nanoseconds.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_push_overhead_ns() {
    return Profiler::instance().avg_push_overhead_ns();
}

/// Average per-call overhead of ``pop`` in nanoseconds.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_pop_overhead_ns() {
    return Profiler::instance().avg_pop_overhead_ns();
}

/// Total number of ``push`` calls observed since process start.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_push_count() {
    return Profiler::instance().total_push_count();
}

/// Total number of ``pop`` calls observed since process start.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_pop_count() {
    return Profiler::instance().total_pop_count();
}

// The site is a function-local static, so name/file/func are interned once per
// call site rather than on every entry. With no format arguments the name is the
// literal itself and nothing is formatted; with arguments the name is built per
// call and only it is interned. Expands to TWO declarations, so it must be used
// at statement scope (as every call site does).
// The site interns name, file and func once per call site (a function-local
// static). Without format arguments, entry then interns NOTHING and so takes no
// lock. With arguments the name must be built per call, so only it is interned.
// The fmt::format call stays here rather than inside ScopedZone because fmt
// validates format strings with a consteval constructor, and a format string
// forwarded through a template parameter is no longer a constant expression.
//
// @p name_format must be a compile-time literal. For a name computed at runtime
// use @ref LabeledSectionRuntime, which cannot cache it.
//
// Expands to TWO declarations, so use it at statement scope.
#    define LabeledSection(name_format, ...)                                                                                               \
        static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};        \
        ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(                                                 \
            EINSUMS_PP_CAT(_zone_site_, __LINE__) __VA_OPT__(, [&] { return fmt::format(name_format, __VA_ARGS__); }))

/// A zone whose name is only known at runtime. The name is interned on every
/// entry, which is one lock; prefer @ref LabeledSection wherever the label can be
/// a literal.
#    define LabeledSectionRuntime(name_expr)                                                                                               \
        static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){"", __FILE__, __LINE__, __func__};                 \
        ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(EINSUMS_PP_CAT(_zone_site_, __LINE__),           \
                                                                                     [&] { return fmt::format("{}", name_expr); })
#    define LabeledSection0() LabeledSection(__func__)
#    if defined(EINSUMS_WITH_PROFILER_INTERNAL)
#        define LabeledSectionInternal(name_format, ...)                                                                                   \
            static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};    \
            ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(                                             \
                EINSUMS_PP_CAT(_zone_site_, __LINE__) __VA_OPT__(, [&] { return fmt::format(name_format, __VA_ARGS__); }))
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
#    define LabeledSectionRuntime(...)
#    define LabeledSectionInternal(...)
#    define LabeledSectionInternal0()
#    define ProfileAnnotate(key, value)
#    define ProfileAnnotateDims(key, dims)
#    define ProfileMemAlloc(bytes)
#    define ProfileMemFree(bytes)
#endif

} // namespace einsums::profile
