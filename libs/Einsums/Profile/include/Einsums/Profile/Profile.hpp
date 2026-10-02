//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Consumer.hpp>
#include <Einsums/Profile/CounterBackend.hpp>
#include <Einsums/Profile/Event.hpp>
#include <Einsums/Profile/Options.hpp>
#include <Einsums/Profile/RingBuffer.hpp>
#include <Einsums/Profile/Server.hpp>
#include <Einsums/Profile/StringTable.hpp>
#include <Einsums/Profile/TickClock.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TypeSupport/InsertionOrderedMap.hpp>

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

EINSUMS_NAMESPACE_BEGIN(profile)

#if defined(EINSUMS_HAVE_PROFILER)

// ---------------------- Profiler class ----------------------
struct EINSUMS_EXPORT Profiler {
    static auto instance() -> Profiler &;

    /// Whether zones and annotations are recorded. When off, each entry point costs one relaxed load.
    [[nodiscard]] bool enabled() const { return _enabled.load(std::memory_order_relaxed); }
    void               set_enabled(bool on) { _enabled.store(on, std::memory_order_relaxed); }

    // Start a zone, interning its strings on every call. LabeledSection uses push_interned instead.
    void push(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        if (!enabled()) {
            return;
        }
        push_interned(_strings.intern(name), _strings.intern(file), _strings.intern(func), line);
    }

    /// Start a zone from already-interned ids (see @ref ZoneSite).
    void push_interned(uint32_t name_id, uint32_t file_id, uint32_t func_id, int line) {
        if (!enabled()) {
            return;
        }
        write_push(thread_channel(), name_id, file_id, func_id, line);
    }

    // Stop timer region
    void pop() {
        if (!enabled()) {
            return;
        }
        write_pop(thread_channel());
    }

    // Print the report: exclusive time, percent, name, file:line and function. @p detailed adds
    // min/max/avg and counters.
    void print(bool detailed = false, std::ostream &os = std::cout);

    // Write the aggregated profile as JSON.
    auto export_json(std::string const &path = "einsums_profile.json") -> std::optional<std::string>;

    // Stop the consumer (with a final drain) and the server.
    void shutdown() {
        // The print sink points into the server's queue, so detach it first.
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

    // Access server (for registering request handlers from other modules).
    auto server() -> Server * { return _server.get(); }

    // Get the profiler's thread ID for the calling thread (platform-specific, matches Consumer keys).
    static auto current_thread_id() -> uint32_t { return thread_key(); }

    // Set a human-readable name for the calling thread.
    void set_thread_name(std::string const &name) { _consumer->set_thread_name(thread_key(), name); }

    // Emit an event to the thread-local ring buffer. Used by annotation API.
    void emit_event(Event const &evt) {
        auto &ch = thread_channel();
        (void)ch.ring.try_push(evt); // a refused push is counted by the ring
        wake_consumer_if_filling(ch);
    }

  private:
    Profiler() : _consumer(std::make_unique<Consumer>(_strings)) {
        // Read server port from config (default 19216)
        uint16_t port = 19216;
        try {
            port = static_cast<uint16_t>(profile_server_port());
            // --einsums:profile:disable: recording dominates small operations, so runs can opt out.
            _enabled.store(!profile_recording_disabled(), std::memory_order_relaxed);
        } catch (...) { // NOLINT
        }
        // The callback dereferences _server on every consumer tick, so it goes inside the guard.
        if (profile_server_enabled()) {
            _server = std::make_unique<Server>(*_consumer, _strings, "127.0.0.1", port);
            _consumer->set_tick_callback([this] { _server->tick(); });
        }
        // No signal handlers here: Runtime owns those, and einsums::finalize() shuts the profiler down.
    }

    // Fallback for exits that skip einsums::finalize(), such as Python's. The consumer must stop
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

    /// Recording switch: on by default, off with --einsums:profile:disable.
    std::atomic<bool> _enabled{true};

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
    void write_push(ThreadChannel &ch, uint32_t name_id, uint32_t file_id, uint32_t func_id, int line) {
        // Counted even when the event is dropped: the consumer resynchronizes on it.
        uint32_t const depth = ++ch.depth;
        if (Event *evt = ch.ring.try_claim()) {
            *evt = Event{.ticks   = TickClock::now(),
                         .type    = EventType::Push,
                         .name_id = name_id,
                         .file_id = file_id,
                         .func_id = func_id,
                         .line    = line,
                         .depth   = depth};
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

// ---------------------- Scoped helper ----------------------
/**
 * @brief The interned name, file and function of one zone call site.
 *
 * @ref LabeledSection makes one a function-local static, so these strings are interned once per
 * site rather than on every entry, under the string table's lock. With a literal name, entering a
 * zone takes no lock at all.
 */
struct ZoneSite {
    ZoneSite(std::string_view name, char const *file, int line_, char const *func) : line{line_} {
        auto &st = Profiler::instance().string_table();
        name_id  = st.intern(name);
        file_id  = st.intern(file);
        func_id  = st.intern(func);
    }

    int      line{0};
    uint32_t name_id{0};
    uint32_t file_id{0};
    uint32_t func_id{0};
};

namespace site_cache {

template <typename T>
struct is_join_view : std::false_type {};
template <typename It, typename Sentinel>
struct is_join_view<fmt::join_view<It, Sentinel, char>> : std::true_type {};

/// Whether a zone-name argument of type @p T has a value that can stand for it in a cache key.
template <typename T>
concept NameKeyScalar = std::is_arithmetic_v<std::remove_cvref_t<T>> || std::is_enum_v<std::remove_cvref_t<T>>;
template <typename T>
concept NameKeyString = std::is_convertible_v<T const &, std::string_view>;

template <typename T>
constexpr bool name_keyable() {
    using U = std::remove_cvref_t<T>;
    if constexpr (NameKeyScalar<U> || NameKeyString<U>) {
        return true;
    } else if constexpr (is_join_view<U>::value) {
        // A join view is read twice on a miss (key, then format), so it must be a forward range.
        using It = decltype(std::declval<U const &>().begin);
        using E  = std::remove_cvref_t<std::iter_reference_t<It>>;
        return std::forward_iterator<It> && (NameKeyScalar<E> || NameKeyString<E>);
    } else {
        return false;
    }
}

/// Writes a cache key into a fixed buffer: scalars as bytes, strings length-prefixed, so distinct
/// argument lists never collide. A key that does not fit is flagged and its zone named uncached.
/// (A plain memcpy: appending the pieces to a std::string cost 74 ns per zone.)
struct KeyWriter {
    char *pos;
    char *end;
    bool  fits{true};

    void put(void const *src, std::size_t n) noexcept {
        if (static_cast<std::size_t>(end - pos) < n) {
            fits = false;
            return;
        }
        std::memcpy(pos, src, n);
        pos += n;
    }

    void put_string(std::string_view s) noexcept {
        auto const n = s.size();
        put(&n, sizeof(n));
        put(s.data(), n);
    }
};

template <typename T>
void write_key(KeyWriter &w, T const &v) noexcept {
    using U = std::remove_cvref_t<T>;
    if constexpr (NameKeyScalar<U>) {
        w.put(&v, sizeof(U));
    } else if constexpr (NameKeyString<U>) {
        w.put_string(std::string_view(v));
    } else {
        w.put_string(std::string_view(v.sep.data(), v.sep.size()));
        std::size_t count = 0;
        for (auto it = v.begin; it != v.end; ++it) {
            write_key(w, *it);
            ++count;
        }
        w.put(&count, sizeof(count));
    }
}

struct StringKeyHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

/// Interned ids by string, for one call site on one thread. No lock: only its thread touches it.
using IdCache = std::unordered_map<std::string, uint32_t, StringKeyHash, std::equal_to<>>;

/// The interned id of @p s through @p cache, interning (one lock) only the first time.
inline uint32_t cached_id(IdCache &cache, std::string_view s) {
    if (auto it = cache.find(s); it != cache.end()) {
        return it->second;
    }
    uint32_t const id = Profiler::instance().string_table().intern(s);
    cache.emplace(std::string(s), id);
    return id;
}

/// Room for one zone-name key; see KeyWriter.
inline constexpr std::size_t kNameKeyCapacity = 1024;

/// One call site's zone names on one thread: the key being built, the most recent key and its id,
/// and every id the site has named.
struct NameSiteCache {
    char        key[kNameKeyCapacity]; // NOLINT(modernize-avoid-c-arrays)
    std::string last_key;
    uint32_t    last_id{std::numeric_limits<uint32_t>::max()};
    IdCache     ids;
};

/// The interned id of the zone name @p format_name builds from @p args, cached per call site (each
/// site's lambda is its own type) and per thread. The last key is compared first, so a loop that
/// repeats one name skips the hash.
template <typename FormatName, typename... Args>
uint32_t zone_name_id(FormatName const &format_name, Args &&...args) {
    // Forwarded because fmt rejects an lvalue join view. Building the key only copies its iterators.
    auto const intern_formatted = [&] { return Profiler::instance().string_table().intern(format_name(std::forward<Args>(args)...)); };
    if constexpr ((name_keyable<Args>() && ...)) {
        thread_local NameSiteCache cache;
        KeyWriter                  w{.pos = cache.key, .end = cache.key + kNameKeyCapacity};
        (write_key(w, args), ...);
        if (!w.fits) [[unlikely]] {
            return intern_formatted();
        }
        std::string_view const key(cache.key, static_cast<std::size_t>(w.pos - cache.key));
        if (key == cache.last_key && cache.last_id != std::numeric_limits<uint32_t>::max()) {
            return cache.last_id;
        }
        uint32_t id;
        if (auto it = cache.ids.find(key); it != cache.ids.end()) {
            id = it->second;
        } else {
            id = intern_formatted();
            cache.ids.emplace(std::string(key), id);
        }
        cache.last_key.assign(key);
        cache.last_id = id;
        return id;
    } else {
        // An argument with no value to key on: format every time.
        return intern_formatted();
    }
}

/// Sentinel for an id not interned yet.
inline constexpr uint32_t kNotInterned = std::numeric_limits<uint32_t>::max();

/// One ProfileAnnotate call site's key id, interned on first use; racing threads store the same id.
///
/// Values are not held here: ``c ? "T" : "N"`` has a literal's type but not a fixed value.
struct AnnotateSite {
    std::atomic<uint32_t> key_id{kNotInterned};

    static uint32_t fill(std::atomic<uint32_t> &slot, std::string_view s) {
        uint32_t id = slot.load(std::memory_order_relaxed);
        if (id == kNotInterned) [[unlikely]] {
            id = Profiler::instance().string_table().intern(s);
            slot.store(id, std::memory_order_relaxed);
        }
        return id;
    }
};

} // namespace site_cache

struct ScopedZone {
    /// Enter a zone with a fixed name. Takes no lock: the site holds every id.
    explicit ScopedZone(ZoneSite const &site) { Profiler::instance().push_interned(site.name_id, site.file_id, site.func_id, site.line); }

    /// Enter a zone whose name is built per call; only the name is interned. It arrives as a callable
    /// so nothing is built when recording is off.
    template <typename MakeName>
        requires std::invocable<MakeName>
    ScopedZone(ZoneSite const &site, MakeName &&make_name) {
        auto &prof = Profiler::instance();
        if (!prof.enabled()) {
            return;
        }
        prof.push_interned(prof.string_table().intern(make_name()), site.file_id, site.func_id, site.line);
    }

    /**
     * @brief Enter a zone named by formatting arguments, cached per call site and thread.
     *
     * @p apply_args supplies the arguments, so they are evaluated only when recording. @p format_name
     * runs only for argument values this thread has not seen at this site, so formatting and the
     * intern lock are paid once per distinct name. Numbers, strings and fmt::join views over forward
     * ranges of them can key the cache; any other argument is formatted every time.
     */
    template <typename ApplyArgs, typename FormatName>
        requires std::is_class_v<std::remove_cvref_t<ApplyArgs>> && std::is_class_v<FormatName>
    ScopedZone(ZoneSite const &site, ApplyArgs &&apply_args, FormatName const &format_name) {
        auto &prof = Profiler::instance();
        if (!prof.enabled()) {
            return;
        }
        std::forward<ApplyArgs>(apply_args)([&](auto &&...args) {
            uint32_t const id = site_cache::zone_name_id(format_name, std::forward<decltype(args)>(args)...);
            prof.push_interned(id, site.file_id, site.func_id, site.line);
        });
    }

    /// Enter a zone with a name the caller interned (@ref intern_string), for callers with a stable
    /// set of runtime names, such as graph replay. Takes no lock.
    ScopedZone(ZoneSite const &site, uint32_t name_id) {
        Profiler::instance().push_interned(name_id, site.file_id, site.func_id, site.line);
    }

    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        Profiler::instance().push(name, file, line, func);
    }
    ~ScopedZone() { Profiler::instance().pop(); }
};

/// Intern @p s, for callers that cache their own ids. Ids are stable for the life of the process.
inline uint32_t intern_string(std::string_view s) {
    return Profiler::instance().string_table().intern(s);
}

// ---------------------- Annotation API ----------------------

/// Attach a string annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate(std::string_view key, std::string_view value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    auto &st = prof.string_table();

    Event evt{};
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = st.intern(key);
    evt.annotation.value_type = AnnotateValueType::String;
    evt.annotation.string_id  = st.intern(value);

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
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = st.intern(key);
    evt.annotation.value_type = AnnotateValueType::Int64;
    evt.annotation.int_val    = value;

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
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = st.intern(key);
    evt.annotation.value_type = AnnotateValueType::Float64;
    evt.annotation.float_val  = value;

    prof.emit_event(evt);
}

/// Attach a string annotation with a pre-interned key and value, skipping the per-call interning
/// (and lock) of @ref annotate.
inline void annotate_interned(uint32_t key_id, uint32_t value_id) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    Event evt{};
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = key_id;
    evt.annotation.value_type = AnnotateValueType::String;
    evt.annotation.string_id  = value_id;

    prof.emit_event(evt);
}

/// Attach an integer annotation under a pre-interned key.
inline void annotate_interned(uint32_t key_id, int64_t value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    Event evt{};
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = key_id;
    evt.annotation.value_type = AnnotateValueType::Int64;
    evt.annotation.int_val    = value;

    prof.emit_event(evt);
}

/// Attach a floating-point annotation under a pre-interned key.
inline void annotate_interned(uint32_t key_id, double value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    Event evt{};
    evt.type                  = EventType::Annotate;
    evt.ticks                 = TickClock::now();
    evt.annotation.key_id     = key_id;
    evt.annotation.value_type = AnnotateValueType::Float64;
    evt.annotation.float_val  = value;

    prof.emit_event(evt);
}

namespace site_cache {

/**
 * @brief The body of @ref ProfileAnnotate.
 *
 * The literal key is interned once per site, string values once per distinct value through a
 * per-site, per-thread cache. @p get_value runs only when recording.
 */
template <std::size_t N, typename GetValue>
void annotate_at(AnnotateSite &site, char const (&key)[N], GetValue &&get_value) {
    auto &prof = Profiler::instance();
    if (!prof.enabled()) {
        return;
    }
    uint32_t const key_id = AnnotateSite::fill(site.key_id, key);
    using V               = decltype(std::forward<GetValue>(get_value)());
    if constexpr (std::is_convertible_v<V, std::string_view>) {
        thread_local IdCache cache;
        annotate_interned(key_id, cached_id(cache, std::string_view(get_value())));
    } else if constexpr (std::is_floating_point_v<std::remove_cvref_t<V>>) {
        annotate_interned(key_id, static_cast<double>(get_value()));
    } else {
        annotate_interned(key_id, static_cast<int64_t>(get_value()));
    }
}

} // namespace site_cache

/// Attach a vector of dimension sizes as annotations (dim.0, dim.1, ...).
inline void annotate_dims(std::string_view key, std::span<int64_t const> dims) {
    for (size_t i = 0; i < dims.size(); ++i) {
        annotate(fmt::format("{}.{}", key, i), dims[i]);
    }
}

/// Record a memory allocation in the current profiling zone. An empty one records nothing: resizing
/// an empty tensor reports freeing its old, zero-byte storage.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_alloc(int64_t bytes) {
    auto &prof = Profiler::instance();
    if (bytes == 0 || !prof.enabled()) {
        return;
    }
    Event evt{};
    evt.type      = EventType::MemAlloc;
    evt.ticks     = TickClock::now();
    evt.mem_bytes = bytes;
    prof.emit_event(evt);
}

/// Record a memory deallocation in the current profiling zone. An empty one records nothing.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_free(int64_t bytes) {
    auto &prof = Profiler::instance();
    if (bytes == 0 || !prof.enabled()) {
        return;
    }
    Event evt{};
    evt.type      = EventType::MemFree;
    evt.ticks     = TickClock::now();
    evt.mem_bytes = bytes;
    prof.emit_event(evt);
}

// ---------------------- Python bindings ----------------------
// Free functions, so einsums.profile need not bind the non-copyable Profiler.

/// Whether this build records anything. ``False`` means einsums was built with
/// ``EINSUMS_WITH_PROFILER=OFF``: the API still exists but does nothing.
APIARY_EXPOSE APIARY_MODULE("profile") constexpr bool available() {
    return true;
}

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

/// Drain the per-thread ring buffers into the aggregated tree, so ``print_report``
/// and ``export_json`` see recent events.
APIARY_EXPOSE APIARY_MODULE("profile") inline void flush() {
    Profiler::instance().flush();
}

/// Print the compact (or detailed) report to standard output.
APIARY_EXPOSE APIARY_MODULE("profile") inline void print_report(bool detailed = false) {
    Profiler::instance().print(detailed);
    // Flush so pytest's capfd and non-tty stdout see the report before returning.
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

// Open a zone for the rest of the scope. Name, file and function are interned once per site; with
// format arguments the name is cached per distinct value (see ScopedZone).
//
// fmt::format is called here, not in ScopedZone, because fmt checks format strings at compile time
// and a format string forwarded through a template parameter is no longer a constant expression.
//
// @p name_format must be a literal; use LabeledSectionRuntime otherwise. Expands to two
// declarations, so use it at statement scope.
#    define LabeledSection(name_format, ...)                                                                                                \
        static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};         \
        ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(EINSUMS_PP_CAT(_zone_site_, __LINE__) __VA_OPT__( \
            , [&](auto &&_zone_f) { return _zone_f(__VA_ARGS__); },                                                                         \
            [](auto &&..._zone_a) { return fmt::format(name_format, std::forward<decltype(_zone_a)>(_zone_a)...); }))

/// A zone named at runtime. Interns, under a lock, on every entry; prefer @ref LabeledSection.
#    define LabeledSectionRuntime(name_expr)                                                                                               \
        static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){"", __FILE__, __LINE__, __func__};                 \
        ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(EINSUMS_PP_CAT(_zone_site_, __LINE__),           \
                                                                                          [&] { return fmt::format("{}", name_expr); })
#    define LabeledSection0() LabeledSection(__func__)
#    if defined(EINSUMS_HAVE_PROFILER_INTERNAL)
#        define LabeledSectionInternal(name_format, ...)                                                                                   \
            static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};    \
            ::einsums::profile::ScopedZone const EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(EINSUMS_PP_CAT(_zone_site_, __LINE__) __VA_OPT__( \
                , [&](auto &&_zone_f) { return _zone_f(__VA_ARGS__); },                                                                    \
                [](auto &&..._zone_a) { return fmt::format(name_format, std::forward<decltype(_zone_a)>(_zone_a)...); }))
#        define LabeledSectionInternal0() LabeledSectionInternal(__func__)
#    else
#        define LabeledSectionInternal(...)
#        define LabeledSectionInternal0()
#    endif

/// Annotate the open zone. @p key must be a string literal, as its id is cached per site; @p value
/// is evaluated only when recording.
#    define ProfileAnnotate(key, value)                                                                                                    \
        [&]() {                                                                                                                            \
            static ::einsums::profile::site_cache::AnnotateSite _annotate_site;                                                            \
            ::einsums::profile::site_cache::annotate_at(_annotate_site, key, [&]() -> decltype(auto) { return (value); });                 \
        }()
#    define ProfileAnnotateDims(key, dims) ::einsums::profile::annotate_dims(key, dims)
#    define ProfileMemAlloc(bytes)         ::einsums::profile::mem_alloc(static_cast<int64_t>(bytes))
#    define ProfileMemFree(bytes)          ::einsums::profile::mem_free(static_cast<int64_t>(bytes))

#else

// ---------------------- Disabled-profiler shims ----------------------
//
// With EINSUMS_WITH_PROFILER=OFF the instrumentation API remains as empty inlines, so call sites
// need no guards. The machinery (rings, consumer, server) has no stand-in: code that uses
// Profiler::server, a Consumer or a BenchmarkResultEntry stays behind EINSUMS_HAVE_PROFILER.

/// Stand-in for the recording profiler: its instrumentation and lifecycle entry points only.
struct Profiler {
    static Profiler &instance() {
        static Profiler p;
        return p;
    }

    [[nodiscard]] bool enabled() const { return false; }
    void               set_enabled(bool /*on*/) {}

    void push(std::string const & /*name*/, std::string const & /*file*/ = "", int /*line*/ = 0, std::string const & /*func*/ = "") {}
    void push_interned(uint32_t /*name_id*/, uint32_t /*file_id*/, uint32_t /*func_id*/, int /*line*/) {}
    void pop() {}

    void set_thread_name(std::string const & /*name*/) {}
    void flush() {}
    void shutdown() {}
    void print(bool /*detailed*/ = false, std::ostream & /*os*/ = std::cout) {}

    static uint32_t current_thread_id() { return 0; }
};

/// Stand-in for an interned call site. Interns nothing, so it holds nothing.
struct ZoneSite {
    constexpr ZoneSite(std::string_view /*name*/, char const * /*file*/, int /*line*/, char const * /*func*/) {}
};

/// Stand-in for a zone. A name the caller builds is still built; @ref LabeledSection drops it at
/// preprocessing.
struct ScopedZone {
    explicit ScopedZone(ZoneSite const & /*site*/) {}

    template <typename MakeName>
        requires std::invocable<MakeName>
    ScopedZone(ZoneSite const & /*site*/, MakeName && /*make_name*/) {}

    ScopedZone(ZoneSite const & /*site*/, uint32_t /*name_id*/) {}

    explicit ScopedZone(std::string const & /*name*/, std::string const & /*file*/ = "", int /*line*/ = 0,
                        std::string const & /*func*/ = "") {}
};

/// Stand-in for interning. There is no string table, so every id is 0.
inline uint32_t intern_string([[maybe_unused]] std::string_view s) {
    return 0;
}

// The Python surface stays, so instrumented scripts run on this build too. Parameters keep their
// names: they are the bindings' keyword arguments.

/// Whether this build records anything at all. Always ``False`` here.
APIARY_EXPOSE APIARY_MODULE("profile") constexpr bool available() {
    return false;
}

/// Discard a string annotation. This build records nothing to attach it to.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key,
                                                            [[maybe_unused]] std::string_view value) {
}

/// Discard an integer annotation.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key, [[maybe_unused]] int64_t value) {
}

/// Discard a floating-point annotation.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key, [[maybe_unused]] double value) {
}

/// Discard a pre-interned string annotation.
inline void annotate_interned([[maybe_unused]] uint32_t key_id, [[maybe_unused]] uint32_t value_id) {
}

/// Discard a pre-interned integer annotation.
inline void annotate_interned([[maybe_unused]] uint32_t key_id, [[maybe_unused]] int64_t value) {
}

/// Discard a pre-interned floating-point annotation.
inline void annotate_interned([[maybe_unused]] uint32_t key_id, [[maybe_unused]] double value) {
}

/// Discard a vector of dimension sizes.
inline void annotate_dims([[maybe_unused]] std::string_view key, [[maybe_unused]] std::span<int64_t const> dims) {
}

/// Discard a memory allocation record.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_alloc([[maybe_unused]] int64_t bytes) {
}

/// Discard a memory deallocation record.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_free([[maybe_unused]] int64_t bytes) {
}

/// Begin a profile region that is not recorded.
APIARY_EXPOSE APIARY_MODULE("profile") inline void push([[maybe_unused]] std::string const &name,
                                                        [[maybe_unused]] std::string const &file = "", [[maybe_unused]] int line = 0,
                                                        [[maybe_unused]] std::string const &func = "") {
}

/// End the innermost profile region.
APIARY_EXPOSE APIARY_MODULE("profile") inline void pop() {
}

/// Drain the per-thread ring buffers, of which there are none.
APIARY_EXPOSE APIARY_MODULE("profile") inline void flush() {
}

/// Print nothing: this build aggregates no report.
APIARY_EXPOSE APIARY_MODULE("profile") inline void print_report([[maybe_unused]] bool detailed = false) {
}

/// Always ``None``: this build has no aggregated profile to write.
APIARY_EXPOSE APIARY_MODULE("profile") inline std::optional<std::string>
export_json([[maybe_unused]] std::string const &path = "einsums_profile.json") {
    return std::nullopt;
}

/// Discard a thread name. No report names the calling thread.
APIARY_EXPOSE APIARY_MODULE("profile") inline void set_thread_name([[maybe_unused]] std::string const &name) {
}

/// Always ``0``: threads are never registered with a profiler that is not there.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint32_t current_thread_id() {
    return Profiler::current_thread_id();
}

/// Always ``0``: recording nothing costs nothing.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_push_overhead_ns() {
    return 0.0;
}

/// Always ``0``: recording nothing costs nothing.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_pop_overhead_ns() {
    return 0.0;
}

/// Always ``0``: pushes are not counted.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_push_count() {
    return 0;
}

/// Always ``0``: pops are not counted.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_pop_count() {
    return 0;
}

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

EINSUMS_NAMESPACE_END(profile)
