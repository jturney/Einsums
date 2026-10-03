//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Clock.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Consumer.hpp"
#include "CounterBackend.hpp"
#include "Detail/InsertionOrderedMap.hpp"
#include "Diagnostics.hpp"
#include "Event.hpp"
#include "RequestHandlers.hpp"
#include "RingBuffer.hpp"
#include "Server.hpp"
#include "Settings.hpp"
#include "Sites.hpp"
#include "StringTable.hpp"

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

WAGGLE_NAMESPACE_BEGIN

// ---------------------- Profiler class ----------------------
struct WAGGLE_EXPORT Profiler {
    static auto instance() -> Profiler &;

    /// Whether zones and annotations are recorded. When off, each entry point costs one relaxed load.
    [[nodiscard]] bool enabled() const { return std::atomic_ref<int32_t>(_enabled).load(std::memory_order_relaxed) != 0; }
    void               set_enabled(bool on) { std::atomic_ref<int32_t>(_enabled).store(on ? 1 : 0, std::memory_order_relaxed); }

    /// The recording switch, for callers that check it before calling in (waggle_enabled_flag).
    [[nodiscard]] auto enabled_flag() const -> int32_t const * { return &_enabled; }

    /// Drain, then clear what the trees and the timeline hold (Consumer::reset).
    void reset() {
        if (_consumer) {
            _consumer->reset();
        }
    }

    // Start a zone, registering its site on every call. WAGGLE_ZONE uses push_interned instead.
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        if (!enabled()) {
            return;
        }
        push_interned(register_site(name, file, line, func), 0);
    }

    /// Start a zone at site @p site_id (see @ref ZoneSite), named @p name_id, or by its site when
    /// @p name_id is 0. Returns whether it opened one: only then is there a zone for @ref pop to
    /// close.
    auto push_interned(uint32_t site_id, uint32_t name_id) -> bool {
        if (!enabled()) {
            return false;
        }
        write_push(thread_channel(), site_id, name_id);
        return true;
    }

    /// The id of the call site at @p file : @p line in @p func, named @p name, of the library
    /// @p domain names; the same description always gives the same id. Thread-safe.
    auto register_site(std::string_view name, std::string_view file, int line, std::string_view func, std::string_view domain = {})
        -> uint32_t {
        return _sites.add({.name_id = _strings.intern(name),
                           .file_id = _strings.intern(file),
                           .func_id = _strings.intern(func),
                           .line    = line,
                           .domain  = _domains.add(domain)});
    }

    /// As @ref register_site, for a domain already registered.
    auto register_site(std::string_view name, std::string_view file, int line, std::string_view func, uint32_t domain) -> uint32_t {
        return _sites.add({.name_id = _strings.intern(name),
                           .file_id = _strings.intern(file),
                           .func_id = _strings.intern(func),
                           .line    = line,
                           .domain  = domain});
    }

    /// The id of the library named @p name, registering it if new. Thread-safe.
    auto register_domain(std::string_view name) -> uint32_t { return _domains.add(name); }

    /// The call sites registered so far.
    auto sites() const -> SiteTable const & { return _sites; }

    // Stop timer region
    /// Close the zone a @ref push_interned that returned true opened, whatever the switch says
    /// now.
    void pop() { write_pop(thread_channel()); }

    // Print the report: exclusive time, percent, name, file:line and function. @p detailed adds
    // min/max/avg and counters.
    void print(bool detailed = false, std::ostream &os = std::cout);

    // Write the aggregated profile as JSON.
    auto export_json(std::string const &path = "einsums_profile.json") -> std::optional<std::string>;

    // Stop the consumer (with a final drain) and the server.
    void shutdown() {
        if (_consumer)
            _consumer->shutdown();
        if (_server)
            _server->shutdown(settings().wait_for_viewer);
    }

    /// Apply @p update. A setting another library already set to a different value keeps that
    /// value, and the refusal is printed: libraries share this profiler, and none should have its
    /// choice changed under it. Effects are immediate: recording switches, a server starts.
    /// Returns how many settings were refused.
    auto configure(SettingsUpdate const &update) -> size_t;

    /// Apply @p update whoever set those settings before. For tests and tools that must put a value
    /// back; libraries use @ref configure.
    void override_settings(SettingsUpdate const &update);

    /// The settings in force.
    [[nodiscard]] auto settings() const -> Settings;

    /// The current value of setting @p key as text, or nothing for an unknown name.
    [[nodiscard]] auto setting_text(std::string_view key) const -> std::optional<std::string> {
        std::scoped_lock const lock(_settings_mutex);
        return _settings.text(key);
    }

    /// Count @p client as using the profiler until its matching @ref finalize. Viewers and session
    /// files list every client.
    void init(ClientInfo client);

    /// Release the @ref init of the client named @p client. The last release writes the session
    /// file and the report the settings ask for, then stops the consumer and the server; recording
    /// ends there for the whole process.
    void finalize(std::string const &client);

    /// The libraries using the profiler, in the order they arrived.
    [[nodiscard]] auto clients() const -> std::vector<ClientInfo> { return _handlers.clients(); }

    /// Hold the calling thread until a viewer connects, if the settings ask for that and a server
    /// is listening. Returns at once otherwise.
    void wait_for_viewer();

    // Flush all pending events from ring buffers into the aggregated tree.
    void flush() {
        if (_consumer)
            _consumer->flush();
    }

    /// What one recorded push and one recorded pop cost, in nanoseconds. Measured once, on first
    /// request, by running the real path into a scratch ring.
    auto avg_push_overhead_ns() -> double { return calibrated_overhead().push_ns; }
    auto avg_pop_overhead_ns() -> double { return calibrated_overhead().pop_ns; }

    /// Zones opened and closed so far, on every thread, whether or not their events were dropped.
    auto total_push_count() const -> uint64_t;
    auto total_pop_count() const -> uint64_t;

    // Access string table (for interning annotation keys/values)
    auto string_table() -> StringTable & { return _strings; }

    // Access consumer (for annotations, shared lock on tree, etc.)
    auto consumer() -> Consumer * { return _consumer.get(); }

    /// The live-viewing server, or null until @ref start_server runs. Safe from any thread.
    auto server() -> Server * { return _server_ptr.load(std::memory_order_acquire); }

    /// Start the live-viewing server on @p port unless one already runs. Safe at any time and from
    /// any thread: zones recorded before it starts are in the first snapshot it sends.
    void start_server(uint16_t port);

    /// Register @p handler for viewer requests named @p method. Works before any server exists: the
    /// profiler keeps the table, and a server started later answers from it.
    void register_handler(std::string method, RequestHandler handler) { _handlers.add(std::move(method), std::move(handler)); }

    /// Remove the handler for @p method, waiting for any call of it in progress. An owner whose
    /// handler captures it calls this from its destructor.
    void unregister_handler(std::string const &method) { _handlers.remove(method); }

    /// Embed @p section's JSON under @p key in every session file.
    void register_session_section(std::string key, SessionSection section) {
        _handlers.add_session_section(std::move(key), std::move(section));
    }

    /// Stream a log message to connected viewers' log panel. @p level runs 0 (trace) to 5
    /// (critical), spdlog's numbering. Returns at once when no server runs, before formatting.
    void log(int level, std::chrono::system_clock::time_point when, std::string_view file, int line, std::string_view function,
             std::string_view message);

    /// Stream one line the program printed to connected viewers. Returns at once without a server.
    void output(std::string_view message);

    /// Send @p json_object (a JSON object) to every connected viewer as a message of type @p type.
    /// Dropped when no server runs: nothing would ever read it.
    void publish(std::string_view type, std::string_view json_object) {
        if (auto *srv = server()) {
            srv->publish(type, json_object);
        }
    }

    // Get the profiler's thread ID for the calling thread (platform-specific, matches Consumer keys).
    static auto current_thread_id() -> uint32_t { return thread_key(); }

    // Set a human-readable name for the calling thread.
    void set_thread_name(std::string const &name) { _consumer->set_thread_name(thread_key(), name); }

    /// Annotate the calling thread's innermost zone: @p fill sets the value in the payload.
    template <typename Fill>
    void annotate(uint32_t key_id, AnnotateValueType type, Fill &&fill) {
        if (!enabled()) {
            return;
        }
        Event evt{};
        evt.type                  = EventType::Annotate;
        evt.ticks                 = TickClock::now();
        evt.annotation.key_id     = key_id;
        evt.annotation.value_type = type;
        fill(evt.annotation);
        emit_event(evt);
    }

    /// Record an allocation or a free (@p type) in the calling thread's innermost zone. An empty
    /// one records nothing.
    void memory(EventType type, void const *address, int64_t bytes) {
        if (bytes == 0 || !enabled()) {
            return;
        }
        Event evt{};
        evt.type        = type;
        evt.ticks       = TickClock::now();
        evt.mem.address = reinterpret_cast<uintptr_t>(address);
        evt.mem.bytes   = bytes;
        emit_event(evt);
    }

    // Emit an event to the thread-local ring buffer. Used by annotation API.
    void emit_event(Event const &evt) {
        auto &ch = thread_channel();
        (void)ch.ring.try_push(evt); // a refused push is counted by the ring
        wake_consumer_if_filling(ch);
    }

  private:
    /// Takes the ``WAGGLE_*`` environment, below whatever libraries configure later. No signal
    /// handlers here: the host program owns those.
    Profiler();

    /// Make the profiler match @p s: the recording switch, the consumer's child cap, the server.
    void apply(Settings const &s);

    // Fallback for programs that never call finalize. The consumer must stop
    // before members are destroyed: its tick calls into _server, which is destroyed first.
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

    // ------------------ per-thread channel ------------------

    /// How many events a thread records between looks at how full its ring is.
    static constexpr uint32_t kFillCheckEvery = 512;

    /**
     * @brief One thread's ring buffer, nesting depth and zone counts.
     *
     * Only the owning thread writes it. The counts are atomic so other threads can read them, and
     * the owner bumps them with a relaxed load and store, not a read-modify-write.
     *
     * Channels live for the whole process, shared with the Consumer, so a thread's unread events
     * and its counts outlive the thread.
     */
    struct ThreadChannel {
        EventRingBuffer ring;
        alignas(64) std::atomic<uint64_t> pushes{0};
        std::atomic<uint64_t> pops{0};
        /// Zones open on this thread, stamped into every Push and Pop (see @ref Event::depth).
        uint32_t depth{0};
        /// Whether a hardware counter backend is active, read once when the thread registers.
        bool counters{false};
        /// Whether this thread has woken the consumer since its ring last passed half full.
        bool woke_consumer{false};
        /// Events left before @ref wake_consumer_if_filling looks at the ring again.
        uint32_t until_fill_check{kFillCheckEvery};
    };

    /// The calling thread's channel, registered on first use.
    ///
    /// Out of line so there is one per thread: under -fvisibility-inlines-hidden each shared object
    /// gets its own copy of an inline function's thread_local, which would split a thread's zones
    /// between the library and, say, the Python bindings.
    static auto thread_channel() -> ThreadChannel &;

    // Platform-specific thread ID
    static auto thread_key() -> uint32_t {
#if defined _WIN32
        static_assert(sizeof(decltype(GetCurrentThreadId())) <= sizeof(uint32_t), "Thread handle too big to fit in protocol");
        return uint32_t(GetCurrentThreadId());
#elif defined __APPLE__
        uint64_t id;
        pthread_threadid_np(pthread_self(), &id);
        return static_cast<uint32_t>(id);
#elif defined __ANDROID__
        return (uint32_t)gettid();
#elif defined __linux__
        return static_cast<uint32_t>(syscall(SYS_gettid));
#elif defined __FreeBSD__
        long id;
        thr_self(&id);
        return id;
#elif defined __NetBSD__
        return _lwp_self();
#elif defined __DragonFly__
        return lwp_gettid();
#elif defined __OpenBSD__
        return getthrid();
#elif defined __QNX__
        return (uint32_t)gettid();
#elif defined __EMSCRIPTEN__
        return 0;
#else
#    error "Unsupported platform!"
#endif
    }

    StringTable               _strings;
    SiteTable                 _sites;
    DomainTable               _domains;
    std::unique_ptr<Consumer> _consumer;

    /// What libraries registered for the server and session files. Declared before the server,
    /// which reads it, so it is destroyed after it.
    RequestHandlers _handlers;

    /// Owned by @ref _server; @ref _server_ptr publishes it to other threads once it exists.
    std::mutex              _server_mutex;
    std::unique_ptr<Server> _server;
    std::atomic<Server *>   _server_ptr{nullptr};

    /// Recording switch, from Settings::record.
    /// Read and written only through std::atomic_ref, here and by every library that checks it
    /// inline: a plain int32_t is the one type the C interface can hand out. Mutable, as
    /// std::atomic_ref takes no const object, even to load.
    alignas(std::atomic_ref<int32_t>::required_alignment) mutable int32_t _enabled{1};

    /// The settings and who set them; under @ref _settings_mutex.
    mutable std::mutex _settings_mutex;
    SettingsStore      _settings;

    /// Serializes @ref init and @ref finalize; the clients themselves are in @ref _handlers.
    std::mutex _lifecycle_mutex;
    bool       _finalized{false};

    /// Every thread's channel, for the life of the process; see @ref ThreadChannel.
    mutable std::mutex                          _channels_mutex;
    std::vector<std::shared_ptr<ThreadChannel>> _channels;

    struct Overhead {
        double push_ns{0.0};
        double pop_ns{0.0};
    };
    std::once_flag _calibration_once;
    Overhead       _calibration;

    /// Create, register and return the calling thread's channel. The cold half of @ref thread_channel.
    auto register_thread() -> ThreadChannel &;

    /// Run the push and pop paths into a scratch channel and time them, once.
    auto calibrated_overhead() -> Overhead const &;

    /// Record a zone opening on @p ch: one clock read and one event written in place. A full ring
    /// skips the clock and counter reads.
    void write_push(ThreadChannel &ch, uint32_t site_id, uint32_t name_id) {
        // Counted even when the event is dropped: the consumer resynchronizes on it.
        uint32_t const depth = ++ch.depth;
        if (Event *evt = ch.ring.try_claim()) {
            *evt = Event{.ticks = TickClock::now(), .type = EventType::Push, .site_id = site_id, .name_id = name_id, .depth = depth};
            if (ch.counters) {
                read_counters(*evt);
            }
            ch.ring.commit();
        }
        wake_consumer_if_filling(ch);
        ch.pushes.store(ch.pushes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    /// Record a zone's closing on @p ch.
    void write_pop(ThreadChannel &ch) {
        // Nothing open: skip it, so a Pop's depth always names an open zone.
        if (ch.depth == 0) {
            return;
        }
        uint32_t const depth = ch.depth--;
        if (Event *evt = ch.ring.try_claim()) {
            *evt = Event{.ticks = TickClock::now(), .type = EventType::Pop, .depth = depth};
            if (ch.counters) {
                read_counters(*evt);
            }
            ch.ring.commit();
        }
        wake_consumer_if_filling(ch);
        ch.pops.store(ch.pops.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    /// Wake the consumer once when @p ch's ring passes half full, so a burst that starts during its
    /// nap does not overflow. Checked every kFillCheckEvery events, since past half each check reads
    /// the consumer's tail.
    void wake_consumer_if_filling(ThreadChannel &ch) {
        if (--ch.until_fill_check != 0) {
            return;
        }
        ch.until_fill_check = kFillCheckEvery;
        if (ch.ring.past_half()) {
            if (!ch.woke_consumer) {
                ch.woke_consumer = true;
                _consumer->notify();
            }
        } else {
            ch.woke_consumer = false;
        }
    }

    static void read_counters(Event &evt) {
        std::array<uint64_t, kNumCounterSlots> values;
        get_counter_backend().read(values);
        for (int i = 0; i < kNumCounterSlots; ++i) {
            evt.counters[i] = values[i];
        }
    }
};

WAGGLE_NAMESPACE_END
