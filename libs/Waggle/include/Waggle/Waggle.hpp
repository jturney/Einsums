//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// Waggle for C++: the instrumentation macros, the call-site caches behind them, and the host
/// interface, all over the C interface in <Waggle/Waggle.h>. Nothing here depends on how the
/// collector stores what it records, so a library built against this header runs with any later
/// collector of the same major version.

#include <Waggle/Config.hpp>

#include <Waggle/Types.hpp>
#include <Waggle/Waggle.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

WAGGLE_NAMESPACE_BEGIN

namespace detail {

/// The id of @p s.
inline uint32_t intern(std::string_view s) {
    return waggle_intern(s.data(), s.size());
}

/// Hands a std::function to the C interface: the user pointer owns a copy, released by @ref release.
template <typename F>
void release(void *user) {
    delete static_cast<F *>(user);
}

/// A handler's answer, or a JSON error when it throws: no exception may cross the C interface.
inline void answer(waggle_reply *reply, RequestHandler const &handler, std::string const &params) {
    std::string out;
    try {
        out = handler(params);
    } catch (std::exception const &e) {
        out = fmt::format(R"({{"error":"{}"}})", e.what());
    } catch (...) {
        out = R"({"error":"the handler threw"})";
    }
    waggle_reply_set(reply, out.data(), out.size());
}

/// @p value as a setting's text.
inline std::string setting_text(bool value) {
    return value ? "true" : "false";
}
inline std::string setting_text(std::int64_t value) {
    return std::to_string(value);
}
inline std::string setting_text(std::string const &value) {
    return value;
}

/// The setting named @p key as text, or empty for an unknown name.
inline std::optional<std::string> setting(char const *key) {
    std::int64_t const length = waggle_config_get(key, nullptr, 0);
    if (length < 0) {
        return std::nullopt;
    }
    std::string value(static_cast<size_t>(length) + 1, '\0');
    waggle_config_get(key, value.data(), value.size());
    value.resize(static_cast<size_t>(length));
    return value;
}

/// Call @p set (waggle_config_set or waggle_config_override) once with every member @p update
/// holds, so they apply as one change.
template <typename Set>
auto apply_settings(SettingsUpdate const &update, Set set) -> int {
    std::vector<char const *> keys;
    std::vector<std::string>  texts;
    auto const                one = [&](char const *key, auto const &member) {
        if (member) {
            keys.push_back(key);
            texts.push_back(setting_text(*member));
        }
    };
    one("record", update.record);
    one("report", update.report);
    one("report_file", update.report_file);
    one("report_append", update.report_append);
    one("report_detailed", update.report_detailed);
    one("save", update.save);
    one("server", update.server);
    one("port", update.port);
    one("wait_for_viewer", update.wait_for_viewer);
    one("max_distinct_children", update.max_distinct_children);
    std::vector<char const *> values;
    values.reserve(texts.size());
    for (auto const &text : texts) {
        values.push_back(text.c_str());
    }
    return set(keys.data(), values.data(), keys.size());
}

/**
 * @brief Whether recording is on, read inline from the collector's switch.
 *
 * A zone or annotation checks this before calling into the collector, so with recording off it
 * costs a load and a branch rather than a call. The switch's address is looked up once.
 */
inline bool recording() {
    static std::int32_t *const flag = const_cast<std::int32_t *>(waggle_enabled_flag()); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    return std::atomic_ref<std::int32_t>(*flag).load(std::memory_order_relaxed) != 0;
}

} // namespace detail

// ---------------------- Recording ----------------------

/// Whether zones and annotations are recorded now.
inline bool enabled() {
    return detail::recording();
}

/// Turn recording on or off for the whole process.
inline void set_enabled(bool on) {
    waggle_set_enabled(on ? 1 : 0);
}

/// The id of the call site at @p file : @p line in @p func, named @p name; the same description
/// always gives the same id.
inline uint32_t register_site(std::string_view name, char const *file, int line, char const *func, uint32_t domain = 0) {
    return waggle_register_site(name.data(), name.size(), file, line, func, domain);
}

/// The id of the library named @p name.
inline uint32_t register_domain(std::string_view name) {
    return waggle_register_domain(name.data(), name.size());
}

/// Open a zone described at run time; prefer @ref WAGGLE_ZONE, which registers its site once.
inline void push(std::string_view name, char const *file = "", int line = 0, char const *func = "") {
    if (enabled()) {
        waggle_zone_begin(register_site(name, file, line, func), 0);
    }
}

/// Close the calling thread's innermost zone, if recording is on: a @ref push made while it was
/// off opened none. Prefer @ref ScopedZone, which pairs them whatever the switch does between.
inline void pop() {
    if (enabled()) {
        waggle_zone_end();
    }
}

/// Name the calling thread in reports and viewers.
inline void set_thread_name(std::string_view name) {
    waggle_set_thread_name(name.data(), name.size());
}

/// The calling thread's id, as reports and viewers show it.
inline uint32_t current_thread_id() {
    return waggle_current_thread_id();
}

// ---------------------- Settings and lifecycle ----------------------

/// Apply @p update. A setting another library already set to a different value keeps that value,
/// and the refusal is reported as a diagnostic.
inline void configure(SettingsUpdate const &update) {
    detail::apply_settings(update, waggle_config_set);
}

/// Apply @p update whoever set those settings before; for tests and tools that put a value back.
inline void override_settings(SettingsUpdate const &update) {
    detail::apply_settings(update, waggle_config_override);
}

/// The settings in force.
inline Settings settings() {
    Settings   s;
    auto const flag = [](char const *key, bool &into) {
        if (auto v = detail::setting(key)) {
            into = *v == "true";
        }
    };
    auto const number = [](char const *key, std::int64_t &into) {
        if (auto v = detail::setting(key)) {
            into = std::stoll(*v);
        }
    };
    auto const text = [](char const *key, std::string &into) {
        if (auto v = detail::setting(key)) {
            into = *v;
        }
    };
    flag("record", s.record);
    flag("report", s.report);
    text("report_file", s.report_file);
    flag("report_append", s.report_append);
    flag("report_detailed", s.report_detailed);
    text("save", s.save);
    flag("server", s.server);
    number("port", s.port);
    flag("wait_for_viewer", s.wait_for_viewer);
    number("max_distinct_children", s.max_distinct_children);
    return s;
}

/// Count @p client as using the profiler until its matching @ref finalize.
inline void init(ClientInfo const &client) {
    waggle_init(client.name.c_str(), client.version.c_str(), client.git_commit.c_str(), client.git_branch.c_str(), client.git_dirty ? 1 : 0,
                client.build_type.c_str());
}

/// Release the @ref init of the client named @p client; the last release writes the outputs.
inline void finalize(std::string const &client) {
    waggle_finalize(client.c_str());
}

/// Drain every thread's recorded events into the aggregated trees.
inline void flush() {
    waggle_flush();
}

/// Hold the calling thread until a viewer connects, if the settings ask for that.
inline void wait_for_viewer() {
    waggle_wait_for_viewer();
}

// ---------------------- The live server ----------------------

inline void start_server(uint16_t port) {
    waggle_server_start(port);
}
inline bool server_running() {
    return waggle_server_running() != 0;
}
inline uint16_t server_port() {
    return waggle_server_port();
}
inline bool viewer_connected() {
    return waggle_viewer_connected() != 0;
}

/// Answer viewer requests named @p method with @p handler, which may be registered before any
/// server runs. A handler must not register or remove handlers itself.
inline void register_handler(std::string const &method, RequestHandler handler) {
    waggle_register_handler(
        method.c_str(),
        [](void *user, char const *params, size_t length, waggle_reply *reply) {
            detail::answer(reply, *static_cast<RequestHandler *>(user), std::string(params, length));
        },
        new RequestHandler(std::move(handler)), &detail::release<RequestHandler>);
}

/// Remove the handler for @p method, waiting for any call of it in progress; an owner whose handler
/// captures it calls this from its destructor.
inline void unregister_handler(std::string const &method) {
    waggle_unregister_handler(method.c_str());
}

/// Embed @p section's JSON under @p key in every session file.
inline void register_session_section(std::string const &key, SessionSection section) {
    waggle_register_session_section(
        key.c_str(),
        [](void *user, char const *, size_t, waggle_reply *reply) {
            auto const &call = *static_cast<SessionSection *>(user);
            detail::answer(reply, [&call](std::string const &) { return call(); }, {});
        },
        new SessionSection(std::move(section)), &detail::release<SessionSection>);
}

/// Send @p json_object to every connected viewer as a message of type @p type.
inline void publish(std::string const &type, std::string_view json_object) {
    waggle_publish(type.c_str(), json_object.data(), json_object.size());
}

/// Stream a log message to connected viewers; @p level runs 0 (trace) to 5 (critical).
inline void log(int level, std::chrono::system_clock::time_point when, char const *file, int line, char const *function,
                std::string_view message) {
    auto const ns = std::chrono::duration_cast<std::chrono::nanoseconds>(when.time_since_epoch()).count();
    waggle_log(level, static_cast<int64_t>(ns), file != nullptr ? file : "", line, function != nullptr ? function : "", message.data(),
               message.size());
}

/// Stream a line the program printed to connected viewers.
inline void output(std::string_view message) {
    waggle_output(message.data(), message.size());
}

// ---------------------- Diagnostics ----------------------

/// Send the profiler's own messages to @p handler; an empty handler restores stderr.
inline void set_diagnostic_handler(DiagnosticHandler handler) {
    if (!handler) {
        waggle_set_diagnostic_handler(nullptr, nullptr, nullptr);
        return;
    }
    waggle_set_diagnostic_handler(
        [](void *user, int level, char const *message, size_t length) {
            try {
                (*static_cast<DiagnosticHandler *>(user))(static_cast<DiagnosticLevel>(level), std::string_view(message, length));
            } catch (...) { // NOLINT(bugprone-empty-catch): a diagnostic must never throw across C
            }
        },
        new DiagnosticHandler(std::move(handler)), &detail::release<DiagnosticHandler>);
}

// ---------------------- Reports ----------------------

/// Print the text report to standard output.
inline void print_report(bool detailed = false) {
    waggle_print_report(detailed ? 1 : 0);
}

/// Write the aggregated trees as JSON to @p path; the path on success.
inline std::optional<std::string> export_json(std::string const &path) {
    return waggle_export_json(path.c_str()) == 0 ? std::optional<std::string>(path) : std::nullopt;
}

inline uint64_t total_push_count() {
    return waggle_total_push_count();
}
inline uint64_t total_pop_count() {
    return waggle_total_pop_count();
}
inline double push_overhead_ns() {
    return waggle_push_overhead_ns();
}
inline double pop_overhead_ns() {
    return waggle_pop_overhead_ns();
}

/// Each string annotation on the calling thread's open zones, outermost first.
inline std::vector<std::pair<std::string, std::string>> open_zone_annotations() {
    std::vector<std::pair<std::string, std::string>> out;
    waggle_open_zone_annotations(
        [](void *user, char const *key, size_t key_length, char const *value, size_t value_length) {
            static_cast<std::vector<std::pair<std::string, std::string>> *>(user)->emplace_back(std::string(key, key_length),
                                                                                                std::string(value, value_length));
        },
        &out);
    return out;
}

// ---------------------- Call sites and zones ----------------------
/**
 * @brief One zone call site, registered once.
 *
 * @ref WAGGLE_ZONE makes one a function-local static, so the site's strings are interned and the
 * site registered once rather than on every entry, under the tables' locks. With a literal name,
 * entering a zone takes no lock at all.
 */
struct ZoneSite {
    ZoneSite(std::string_view name, char const *file, int line, char const *func)
        : site_id{waggle_register_site(name.data(), name.size(), file, line, func, 0)} {}

    uint32_t site_id{0};
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
    uint32_t const id = detail::intern(s);
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
    auto const intern_formatted = [&] { return detail::intern(format_name(std::forward<Args>(args)...)); };
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

/// One WAGGLE_ANNOTATE call site's key id, interned on first use; racing threads store the same id.
///
/// Values are not held here: ``c ? "T" : "N"`` has a literal's type but not a fixed value.
struct AnnotateSite {
    std::atomic<uint32_t> key_id{kNotInterned};

    static uint32_t fill(std::atomic<uint32_t> &slot, std::string_view s) {
        uint32_t id = slot.load(std::memory_order_relaxed);
        if (id == kNotInterned) [[unlikely]] {
            id = detail::intern(s);
            slot.store(id, std::memory_order_relaxed);
        }
        return id;
    }
};

} // namespace site_cache

struct ScopedZone {
    /// Enter a zone with a fixed name. Takes no lock: the site holds every id.
    explicit ScopedZone(ZoneSite const &site) : _open(detail::recording() && waggle_zone_begin(site.site_id, 0) != 0) {}

    /// Enter a zone whose name is built per call; only the name is interned. It arrives as a callable
    /// so nothing is built when recording is off.
    template <typename MakeName>
        requires std::invocable<MakeName>
    ScopedZone(ZoneSite const &site, MakeName &&make_name) {
        if (!detail::recording()) {
            return;
        }
        _open = waggle_zone_begin(site.site_id, detail::intern(make_name())) != 0;
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
        if (!detail::recording()) {
            return;
        }
        std::forward<ApplyArgs>(apply_args)([&](auto &&...args) {
            _open = waggle_zone_begin(site.site_id, site_cache::zone_name_id(format_name, std::forward<decltype(args)>(args)...)) != 0;
        });
    }

    /// Enter a zone with a name the caller interned (@ref intern_string), for callers with a stable
    /// set of runtime names, such as graph replay. Takes no lock.
    ScopedZone(ZoneSite const &site, uint32_t name_id) : _open(detail::recording() && waggle_zone_begin(site.site_id, name_id) != 0) {}

    /// Enter a zone at a site described at run time, registered on every entry.
    explicit ScopedZone(std::string_view name, char const *file = "", int line = 0, char const *func = "") {
        if (detail::recording()) {
            _open = waggle_zone_begin(waggle_register_site(name.data(), name.size(), file, line, func, 0), 0) != 0;
        }
    }

    /// Leave the zone, if entering opened one: switching recording while it is open changes
    /// nothing about which zones close.
    ~ScopedZone() {
        if (_open) {
            waggle_zone_end();
        }
    }

    ScopedZone(ScopedZone const &)            = delete;
    ScopedZone &operator=(ScopedZone const &) = delete;

  private:
    bool _open = false;
};

/// Intern @p s, for callers that cache their own ids. Ids are stable for the life of the process.
inline uint32_t intern_string(std::string_view s) {
    return detail::intern(s);
}

// ---------------------- Annotation API ----------------------

/// Attach a string annotation to the current profiling zone.
inline void annotate(std::string_view key, std::string_view value) {
    if (detail::recording()) {
        waggle_annotate_str(detail::intern(key), detail::intern(value));
    }
}

/// Attach an integer annotation to the current profiling zone.
inline void annotate(std::string_view key, int64_t value) {
    if (detail::recording()) {
        waggle_annotate_i64(detail::intern(key), value);
    }
}

/// Attach a floating-point annotation to the current profiling zone.
inline void annotate(std::string_view key, double value) {
    if (detail::recording()) {
        waggle_annotate_f64(detail::intern(key), value);
    }
}

/// Attach a string annotation with a pre-interned key and value, skipping the per-call interning
/// (and lock) of @ref annotate.
inline void annotate_interned(uint32_t key_id, uint32_t value_id) {
    if (detail::recording()) {
        waggle_annotate_str(key_id, value_id);
    }
}

/// Attach an integer annotation under a pre-interned key.
inline void annotate_interned(uint32_t key_id, int64_t value) {
    if (detail::recording()) {
        waggle_annotate_i64(key_id, value);
    }
}

/// Attach a floating-point annotation under a pre-interned key.
inline void annotate_interned(uint32_t key_id, double value) {
    if (detail::recording()) {
        waggle_annotate_f64(key_id, value);
    }
}

namespace site_cache {

/**
 * @brief The body of @ref WAGGLE_ANNOTATE.
 *
 * The literal key is interned once per site, string values once per distinct value through a
 * per-site, per-thread cache. @p get_value runs only when recording.
 */
template <std::size_t N, typename GetValue>
void annotate_at(AnnotateSite &site, char const (&key)[N], GetValue &&get_value) {
    if (!detail::recording()) {
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
/// an empty tensor reports freeing its old, zero-byte storage. @p address, when given, is the
/// allocation's, for matching it to its free.
inline void mem_alloc(int64_t bytes, void const *address = nullptr) {
    if (bytes != 0 && detail::recording()) {
        waggle_mem_alloc(address, bytes);
    }
}

/// Record a memory deallocation in the current profiling zone. An empty one records nothing.
/// The size is the caller's to give: a buffer freed on another thread than the one that allocated
/// it can be processed before its allocation, so the collector cannot look it up.
inline void mem_free(int64_t bytes, void const *address = nullptr) {
    if (bytes != 0 && detail::recording()) {
        waggle_mem_free(address, bytes);
    }
}

// ---------------------- Snapshots and reset ----------------------

/// A zone's statistics, as waggle_node_stats holds them.
using NodeStats = waggle_node_stats;

/// One numeric annotation of a zone: the total, smallest, largest and count of its values.
struct NumericAnnotation {
    std::string key;
    double      total = 0.0;
    double      min   = 0.0;
    double      max   = 0.0;
    uint64_t    count = 0;
};

/**
 * @brief One zone of a @ref Snapshot, valid while the snapshot lives.
 */
class SnapshotNode {
  public:
    explicit SnapshotNode(waggle_node const *node) : _node(node) {}

    [[nodiscard]] std::string_view name() const { return text(waggle_node_name); }
    [[nodiscard]] std::string_view file() const { return text(waggle_node_file); }
    [[nodiscard]] int              line() const { return waggle_node_line(_node); }
    [[nodiscard]] std::string_view function() const { return text(waggle_node_function); }

    [[nodiscard]] NodeStats stats() const {
        NodeStats s{};
        s.size = sizeof(NodeStats);
        waggle_node_stats_get(_node, &s);
        return s;
    }

    [[nodiscard]] std::vector<SnapshotNode> children() const {
        std::vector<SnapshotNode> out;
        size_t const              n = waggle_node_child_count(_node);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.emplace_back(waggle_node_child(_node, i));
        }
        return out;
    }

    /// Every annotation's latest value as text, numeric ones included, in the order they were
    /// first made.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> annotations() const {
        std::vector<std::pair<std::string, std::string>> out;
        waggle_node_annotations(
            _node,
            [](void *user, char const *key, size_t key_length, char const *value, size_t value_length) {
                static_cast<std::vector<std::pair<std::string, std::string>> *>(user)->emplace_back(std::string(key, key_length),
                                                                                                    std::string(value, value_length));
            },
            &out);
        return out;
    }

    /// The numeric annotations, in the order they were first made.
    [[nodiscard]] std::vector<NumericAnnotation> numeric_annotations() const {
        std::vector<NumericAnnotation> out;
        waggle_node_numeric_annotations(
            _node,
            [](void *user, char const *key, size_t key_length, double total, double min, double max, uint64_t count) {
                static_cast<std::vector<NumericAnnotation> *>(user)->push_back(
                    {.key = std::string(key, key_length), .total = total, .min = min, .max = max, .count = count});
            },
            &out);
        return out;
    }

  private:
    [[nodiscard]] std::string_view text(char const *(*get)(waggle_node const *, size_t *)) const {
        size_t      length = 0;
        char const *s      = get(_node, &length);
        return {s, length};
    }

    waggle_node const *_node;
};

/**
 * @brief A copy of the aggregated trees, read while recording continues.
 *
 * Every zone opened so far on every thread, with its statistics and annotations. Taking one
 * flushes the threads' events first, so it holds everything recorded before the call.
 */
class Snapshot {
  public:
    struct Thread {
        uint32_t     id;
        std::string  name;
        SnapshotNode root; ///< unnamed, above the thread's outermost zones
    };

    /// Take a snapshot: one tree per thread, or with @p merge_threads one tree for the process,
    /// zones matched by their name path.
    static Snapshot take(bool merge_threads = false) {
        return Snapshot(waggle_snapshot_take(merge_threads ? WAGGLE_SNAPSHOT_MERGE_THREADS : 0U));
    }

    [[nodiscard]] std::vector<Thread> threads() const {
        std::vector<Thread> out;
        size_t const        n = waggle_snapshot_thread_count(_snapshot.get());
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            size_t      length = 0;
            char const *name   = waggle_snapshot_thread_name(_snapshot.get(), i, &length);
            out.push_back({.id   = waggle_snapshot_thread_id(_snapshot.get(), i),
                           .name = std::string(name, length),
                           .root = SnapshotNode(waggle_snapshot_root(_snapshot.get(), i))});
        }
        return out;
    }

    /// The zone at @p path ("solve/iterate/gemm") below thread @p thread's root (an index into
    /// @ref threads), if there is one.
    [[nodiscard]] std::optional<SnapshotNode> find(size_t thread, std::string_view path) const {
        auto const *node = waggle_snapshot_find(_snapshot.get(), thread, path.data(), path.size());
        return node == nullptr ? std::nullopt : std::optional<SnapshotNode>(SnapshotNode(node));
    }

    /// The zone at @p path on the first thread that has one.
    [[nodiscard]] std::optional<SnapshotNode> find(std::string_view path) const {
        size_t const n = waggle_snapshot_thread_count(_snapshot.get());
        for (size_t i = 0; i < n; ++i) {
            if (auto node = find(i, path)) {
                return node;
            }
        }
        return std::nullopt;
    }

  private:
    explicit Snapshot(waggle_snapshot *snapshot) : _snapshot(snapshot) {}

    struct Release {
        void operator()(waggle_snapshot *snapshot) const { waggle_snapshot_release(snapshot); }
    };
    std::unique_ptr<waggle_snapshot, Release> _snapshot;
};

/// Clear every statistic, annotation and timeline entry, keeping the zones open now, which are
/// timed from here. Process-wide: every library's data goes.
inline void reset() {
    waggle_reset();
}

WAGGLE_NAMESPACE_END

// ---------------------- Instrumentation macros ----------------------
//
// WAGGLE_DISABLE, defined before this header, makes every macro below expand to nothing; the API
// above is unchanged, so a library that disables its own zones still shares the process's one
// profiler with libraries that do not. WAGGLE_DETAIL turns on WAGGLE_ZONE_DETAIL, the zones a
// library keeps for profiling its own internals.
#if defined(WAGGLE_DISABLE)
#    define WAGGLE_ZONE(...)
#    define WAGGLE_ZONE_FUNC()
#    define WAGGLE_ZONE_DYNAMIC(...)
#    define WAGGLE_ZONE_DETAIL(...)
#    define WAGGLE_ZONE_DETAIL_FUNC()
#    define WAGGLE_ANNOTATE(key, value)
#    define WAGGLE_ANNOTATE_DIMS(key, dims)
#    define WAGGLE_MEM_ALLOC(bytes)
#    define WAGGLE_MEM_FREE(bytes)
#else
// Open a zone for the rest of the scope. Name, file and function are interned once per site; with
// format arguments the name is cached per distinct value (see ScopedZone).
//
// fmt::format is called here, not in ScopedZone, because fmt checks format strings at compile time
// and a format string forwarded through a template parameter is no longer a constant expression.
//
// @p name_format must be a literal; use WAGGLE_ZONE_DYNAMIC otherwise. Expands to two
// declarations, so use it at statement scope.
#    define WAGGLE_ZONE(name_format, ...)                                                                                                  \
        static ::waggle::ZoneSite const WAGGLE_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};                   \
        ::waggle::ScopedZone const      WAGGLE_PP_CAT(_scoped_zone_, __LINE__)(WAGGLE_PP_CAT(_zone_site_, __LINE__) __VA_OPT__(            \
            , [&](auto &&_zone_f) { return _zone_f(__VA_ARGS__); },                                                                        \
            [](auto &&..._zone_a) { return fmt::format(name_format, std::forward<decltype(_zone_a)>(_zone_a)...); }))

/// A zone named at runtime. Interns, under a lock, on every entry; prefer @ref WAGGLE_ZONE.
#    define WAGGLE_ZONE_DYNAMIC(name_expr)                                                                                                 \
        static ::waggle::ZoneSite const WAGGLE_PP_CAT(_zone_site_, __LINE__){"", __FILE__, __LINE__, __func__};                            \
        ::waggle::ScopedZone const      WAGGLE_PP_CAT(_scoped_zone_, __LINE__)(WAGGLE_PP_CAT(_zone_site_, __LINE__),                       \
                                                                               [&] { return fmt::format("{}", name_expr); })
#    define WAGGLE_ZONE_FUNC() WAGGLE_ZONE(__func__)
#    if defined(WAGGLE_DETAIL)
#        define WAGGLE_ZONE_DETAIL(name_format, ...)                                                                                       \
            static ::waggle::ZoneSite const WAGGLE_PP_CAT(_zone_site_, __LINE__){name_format, __FILE__, __LINE__, __func__};               \
            ::waggle::ScopedZone const      WAGGLE_PP_CAT(_scoped_zone_, __LINE__)(WAGGLE_PP_CAT(_zone_site_, __LINE__) __VA_OPT__(        \
                , [&](auto &&_zone_f) { return _zone_f(__VA_ARGS__); },                                                                    \
                [](auto &&..._zone_a) { return fmt::format(name_format, std::forward<decltype(_zone_a)>(_zone_a)...); }))
#        define WAGGLE_ZONE_DETAIL_FUNC() WAGGLE_ZONE_DETAIL(__func__)
#    else
#        define WAGGLE_ZONE_DETAIL(...)
#        define WAGGLE_ZONE_DETAIL_FUNC()
#    endif

/// Annotate the open zone. @p key must be a string literal, as its id is cached per site; @p value
/// is evaluated only when recording.
#    define WAGGLE_ANNOTATE(key, value)                                                                                                    \
        [&]() {                                                                                                                            \
            static ::waggle::site_cache::AnnotateSite _annotate_site;                                                                      \
            ::waggle::site_cache::annotate_at(_annotate_site, key, [&]() -> decltype(auto) { return (value); });                           \
        }()
#    define WAGGLE_ANNOTATE_DIMS(key, dims) ::waggle::annotate_dims(key, dims)
#    define WAGGLE_MEM_ALLOC(bytes)         ::waggle::mem_alloc(static_cast<int64_t>(bytes))
#    define WAGGLE_MEM_FREE(bytes)          ::waggle::mem_free(static_cast<int64_t>(bytes))
#endif
