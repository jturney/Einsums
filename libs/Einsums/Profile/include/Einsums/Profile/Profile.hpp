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

EINSUMS_NAMESPACE_BEGIN(profile)

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
#    ifdef EINSUMS_HAVE_TRACY
        auto z = std::make_unique<tracy::ScopedZone>(line, file.data(), file.size(), func.data(), func.size(), name.data(), name.size(), 1);
        thread_tracy_zones().push_back(std::move(z));
#    else
        (void)name;
        (void)file;
        (void)func;
#    endif
        write_push(thread_channel(), name_id, file_id, func_id, line);
    }

    // Stop timer region
    void pop() {
        if (!enabled()) {
            return;
        }
#    ifdef EINSUMS_HAVE_TRACY
        if (!thread_tracy_zones().empty())
            thread_tracy_zones().pop_back();
#    endif
        write_pop(thread_channel());
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

    /// What one recorded push and one recorded pop cost, in nanoseconds.
    ///
    /// Measured once, on first request, by running the same code path into a scratch ring. A zone
    /// used to time itself: two extra clock reads and four fetch_adds on counters every thread
    /// shared, which made the overhead it reported a large part of the overhead it had, and the
    /// shared counters grew that part with every thread added.
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
        if (!ch.ring.try_push(evt)) {
            _consumer->increment_dropped();
        }
        wake_consumer_if_filling(ch);
    }

  private:
    Profiler() : _consumer(std::make_unique<Consumer>(_strings)) {
        // Read server port from config (default 19216)
        uint16_t port = 19216;
        try {
            port = static_cast<uint16_t>(profile_server_port());
            // --einsums:profile:disable. Recording every zone and annotation is not
            // free: on small operations it dominates, so a run that does not want a
            // profile should be able to say so and pay one relaxed load per zone.
            _enabled.store(!profile_recording_disabled(), std::memory_order_relaxed);
        } catch (...) { // NOLINT
        }
        // The callback dereferences _server on every consumer tick, so it goes inside the guard.
        if (profile_server_enabled()) {
            _server = std::make_unique<Server>(*_consumer, _strings, "127.0.0.1", port);
            _consumer->set_tick_callback([this] { _server->tick(); });
        }
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

    // ------------------ per-thread channel ------------------

    /**
     * @brief One thread's side of the profiler: its ring buffer, nesting depth and zone counts.
     *
     * Only the owning thread writes any of it. The counts are atomics only so another thread can
     * read them: the owner bumps them with a relaxed load and store, never a read-modify-write, so
     * no producer ever touches a cache line another producer writes.
     *
     * The ring buffer is shared with the Consumer: a producer thread (e.g. a transient TaskPool
     * worker) can exit while the Consumer's drain thread is still popping residual events, so
     * ownership must outlive the thread. The profiler keeps every channel for the life of the
     * process, which is also what keeps an exited thread's zones in the counts.
     */
    struct ThreadChannel {
        EventRingBuffer ring;
        alignas(64) std::atomic<uint64_t> pushes{0};
        std::atomic<uint64_t> pops{0};
        /// How many zones this thread has open, stamped into every Push and Pop (see
        /// @ref Event::depth) so the consumer can tell a nesting level from a lost event.
        uint32_t depth{0};
        /// Whether a hardware counter backend is active, read once when the thread registers.
        bool counters{false};
        /// Whether this thread has woken the consumer since its ring last passed half full.
        bool woke_consumer{false};
    };

    /// The calling thread's channel, registered on first use.
    ///
    /// A plain pointer in a constant-initialized thread_local, so the hot path reads it with no
    /// initialization guard; registration is the cold path.
    static auto thread_channel() -> ThreadChannel & {
        static thread_local ThreadChannel *channel = nullptr;
        if (channel == nullptr) [[unlikely]] {
            channel = &instance().register_thread();
        }
        return *channel;
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

    /// Recording switch. On by default so the default report keeps working;
    /// --einsums:profile:disable turns it off, which reduces every zone and
    /// annotation to one relaxed load.
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

    /// Record a zone's opening on @p ch. The whole hot path of a recorded zone: one raw clock read,
    /// one event written into a ring only this thread writes, and the thread's own count.
    void write_push(ThreadChannel &ch, uint32_t name_id, uint32_t file_id, uint32_t func_id, int line) {
        Event evt{};
        evt.ticks   = TickClock::now();
        evt.type    = EventType::Push;
        evt.name_id = name_id;
        evt.file_id = file_id;
        evt.func_id = func_id;
        evt.line    = line;
        // Counted whether or not the event makes it into the buffer: this is
        // where the thread actually is, and the consumer resynchronizes against
        // it precisely when the events between have been dropped.
        evt.depth = ++ch.depth;
        if (ch.counters) {
            read_counters(evt);
        }
        if (!ch.ring.try_push(evt)) {
            _consumer->increment_dropped();
        }
        wake_consumer_if_filling(ch);
        ch.pushes.store(ch.pushes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    /// Record a zone's closing on @p ch.
    void write_pop(ThreadChannel &ch) {
        // A pop with nothing open closes nothing. It used to be sent anyway and
        // dropped at the far end; keeping the count here means the depth a Pop
        // carries is always the level of a zone that is really open.
        if (ch.depth == 0) {
            return;
        }
        Event evt{};
        evt.ticks = TickClock::now();
        evt.type  = EventType::Pop;
        evt.depth = ch.depth--;
        if (ch.counters) {
            read_counters(evt);
        }
        if (!ch.ring.try_push(evt)) {
            _consumer->increment_dropped();
        }
        wake_consumer_if_filling(ch);
        ch.pops.store(ch.pops.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    /// Wake the consumer once when @p ch's ring passes half full. The consumer naps longer the longer
    /// nothing arrives, so a burst that starts during a nap would otherwise fill the ring and drop
    /// events before it looked again.
    void wake_consumer_if_filling(ThreadChannel &ch) {
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
 * @brief The interned identity of one instrumentation site's location.
 *
 * ``__FILE__`` and ``__func__`` are compile-time constants where a zone is
 * written, so interning them on every entry re-hashes and re-compares the same
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
        // A fmt::join view is read twice on a miss, once for the key and once to format, so only
        // one over a forward range can stand in the key; a single-pass range is formatted every time.
        using It = decltype(std::declval<U const &>().begin);
        using E  = std::remove_cvref_t<std::iter_reference_t<It>>;
        return std::forward_iterator<It> && (NameKeyScalar<E> || NameKeyString<E>);
    } else {
        return false;
    }
}

/// Writes a cache key into a fixed buffer: scalars as their bytes, strings length-prefixed, so no
/// two different argument lists write the same key. A key the buffer cannot hold is marked, and its
/// zone is named the uncached way.
///
/// A plain buffer and memcpy, not std::string::append: three joined index lists are about twenty
/// pieces, and appending them to a string one call at a time cost 74 ns, most of a zone's entry.
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

/// The interned id of the zone name @p format_name makes from @p args, cached per call site and
/// thread. Each call site's formatting lambda is its own type, so each gets its own cache.
///
/// The most recent key is checked first with one comparison: a loop running the same contraction
/// takes that path every time, and skips the hash and the map.
template <typename FormatName, typename... Args>
uint32_t zone_name_id(FormatName const &format_name, Args &&...args) {
    // Forwarded, because fmt refuses a view (a fmt::join) passed as an lvalue. A view read for the key
    // is only iterators, so it can still be formatted afterwards.
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

/// One ProfileAnnotate call site: its literal key, interned on first use and then read with one
/// relaxed load. Two threads racing to fill the slot intern the same string and store the same id.
///
/// Only the key is held here. A value that looks like a literal need not be one: a conditional
/// between two literals of the same length, ``ta == 't' ? "T" : "N"``, has the same type as a
/// single literal, and caching it here would pin the site to whichever value came first.
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

    /**
     * @brief Enter a zone whose name is formatted from arguments, with the name cached per call site.
     *
     * @p apply_args calls what it is given with the zone's arguments, so they are evaluated only
     * when recording is on. @p format_name formats them; it is called only when this thread has not
     * seen these argument values at this site before, so a zone named after its operands pays for
     * fmt::format and the string table's lock once per distinct name instead of on every entry.
     * Numbers, strings and fmt::join views over forward ranges of either can key the cache; a zone
     * with any other argument is formatted every time.
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
#    ifdef EINSUMS_HAVE_TRACY
            std::string const name = prof.string_table().get(id);
#    else
            std::string_view const name{};
#    endif
            prof.push_interned(id, site.file_id, site.func_id, site.line, name, site.file, site.func);
        });
    }

    /// Enter a zone whose name the CALLER interned, at a fixed call site.
    ///
    /// For a caller that has its own stable set of runtime names and can cache
    /// their ids - a graph replaying the same nodes, say - this is the plain
    /// site path with a name the site could not know: nothing is interned, so
    /// nothing is locked. Get the id from @ref intern_string.
    ScopedZone(ZoneSite const &site, uint32_t name_id, std::string_view name = {}) {
        Profiler::instance().push_interned(name_id, site.file_id, site.func_id, site.line, name, site.file, site.func);
    }

    explicit ScopedZone(std::string const &name, std::string const &file = "", int line = 0, std::string const &func = "") {
        Profiler::instance().push(name, file, line, func);
    }
    ~ScopedZone() { Profiler::instance().pop(); }
};

/// Intern @p s and return its id, for callers that cache annotation keys,
/// values or zone names of their own. Ids are stable for the life of the
/// process: the string table only ever grows.
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

/// Attach a string annotation whose key AND value were interned ahead of time.
///
/// The @ref annotate overloads above intern on every call, under the string
/// table's lock. A caller whose annotations are invariant across repetitions -
/// a graph node's shapes and index lists, say - can intern once and come
/// through here instead, which costs the enabled() check and the event write.
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
 * @brief The body of @ref ProfileAnnotate: annotate the open zone from call site @p site.
 *
 * The key is a literal, so its id is the site's, interned once. A string value is looked up in a
 * cache of this site's own on this thread, so it takes the string table's lock once per distinct
 * value. Numbers need no interning. @p get_value produces the value, and is called only when
 * recording is on.
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

/// Record a memory allocation in the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_alloc(int64_t bytes) {
    Event evt{};
    evt.type      = EventType::MemAlloc;
    evt.ticks     = TickClock::now();
    evt.mem_bytes = bytes;
    Profiler::instance().emit_event(evt);
}

/// Record a memory deallocation in the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_free(int64_t bytes) {
    Event evt{};
    evt.type      = EventType::MemFree;
    evt.ticks     = TickClock::now();
    evt.mem_bytes = bytes;
    Profiler::instance().emit_event(evt);
}

// ---------------------- Python bindings ----------------------
// Thin free-function wrappers around the Profiler singleton so the
// einsums.profile Python submodule can drive push/pop/flush/print without
// having to bind the Profiler class itself (which holds non-copyable
// unique_ptrs and exposes an ostream& on print()).

/// Whether this build records anything at all. ``False`` means einsums was
/// compiled with ``EINSUMS_WITH_PROFILER=OFF``: the whole API below is still
/// callable and still does nothing, so instrumented code needs no branch of its
/// own, but no report, session or counter will ever be non-empty.
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
#    define LabeledSection(name_format, ...)                                                                                                \
        static ::einsums::profile::ZoneSite const EINSUMS_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};         \
        ::einsums::profile::ScopedZone const      EINSUMS_PP_CAT(_scoped_zone_, __LINE__)(EINSUMS_PP_CAT(_zone_site_, __LINE__) __VA_OPT__( \
            , [&](auto &&_zone_f) { return _zone_f(__VA_ARGS__); },                                                                         \
            [](auto &&..._zone_a) { return fmt::format(name_format, std::forward<decltype(_zone_a)>(_zone_a)...); }))

/// A zone whose name is only known at runtime. The name is interned on every
/// entry, which is one lock; prefer @ref LabeledSection wherever the label can be
/// a literal.
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

/// Annotate the open zone. The key must be a plain string literal, not an expression that picks
/// one: its id is interned once per call site and reused (see site_cache::annotate_at). The value may
/// be anything the annotation overloads take, and is evaluated only when recording.
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
// With EINSUMS_WITH_PROFILER=OFF the instrumentation API still exists and does
// nothing, so an instrumented call site needs no preprocessor guard of its own.
// Every entry point below is an empty inline function, which an optimizing build
// erases along with the call.
//
// What does not survive is the profiler's machinery: the ring buffers, the
// aggregating consumer and the TCP server have no stand-in, because a caller
// that wants a Server wants to talk to something. The handful of sites that
// reach for @ref Profiler::server, a Consumer or a BenchmarkResultEntry stay
// guarded by EINSUMS_HAVE_PROFILER, and so does any surrounding work (reading
// profiler-* options, opening a report file) that only exists to feed them.

/// Stand-in for the recording profiler. Mirrors the recording type's
/// instrumentation and lifecycle entry points; the reporting and transport ones
/// (``server``, ``consumer``, ``export_json``, the overhead counters) are absent
/// on purpose.
struct Profiler {
    static Profiler &instance() {
        static Profiler p;
        return p;
    }

    [[nodiscard]] bool enabled() const { return false; }
    void               set_enabled(bool /*on*/) {}

    void push(std::string const & /*name*/, std::string const & /*file*/ = "", int /*line*/ = 0, std::string const & /*func*/ = "") {}
    void push_interned(uint32_t /*name_id*/, uint32_t /*file_id*/, uint32_t /*func_id*/, int /*line*/, std::string_view /*name*/ = {},
                       std::string_view /*file*/ = {}, std::string_view /*func*/ = {}) {}
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

/// Stand-in for a zone. Note that the name still has to be *built* by the caller
/// unless it arrives through @ref LabeledSection, which drops the whole
/// expression at preprocessing time.
struct ScopedZone {
    explicit ScopedZone(ZoneSite const & /*site*/) {}

    template <typename MakeName>
        requires std::invocable<MakeName>
    ScopedZone(ZoneSite const & /*site*/, MakeName && /*make_name*/) {}

    ScopedZone(ZoneSite const & /*site*/, uint32_t /*name_id*/, std::string_view /*name*/ = {}) {}

    explicit ScopedZone(std::string const & /*name*/, std::string const & /*file*/ = "", int /*line*/ = 0,
                        std::string const & /*func*/ = "") {}
};

/// Stand-in for interning. There is no string table, so every id is 0.
inline uint32_t intern_string([[maybe_unused]] std::string_view s) {
    return 0;
}

// The einsums.profile Python surface stays intact so that an instrumented script
// still runs against a build with the profiler compiled out, the same way an
// instrumented translation unit still compiles. Parameters keep their names
// because those names are the keyword arguments the bindings expose.

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
