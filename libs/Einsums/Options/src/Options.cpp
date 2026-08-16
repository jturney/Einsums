//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Options.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * Where a descriptor becomes a registered option, and where a typed read
 * becomes a value.
 *
 * The registry is the store. Every option owns one slot and every read and
 * write is an atomic operation on it: no hash, no map, no shared_ptr
 * indirection, and no lock. That is what makes the option system safe under
 * concurrent read and write at any time rather than safe only because the
 * writing happens to be confined to a single-threaded startup window.
 *
 * One mutex remains. It guards registration and the table of keys no
 * descriptor claims, and nothing ever holds it while taking another, so there
 * is no lock order to maintain and no lock-order inversion to chase.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

namespace detail {

/// Which of an entry's slots have ever been written.
enum struct ValueKind : std::uint8_t { Bool = 1, Int = 2, Double = 4, String = 8 };

/**
 * @brief One option's storage.
 *
 * All four slots are present rather than one per type, because a key reached
 * through the dynamic API carries no declared type and the four parallel maps
 * this replaced let one key hold a value of each kind at once. @c assigned
 * records which slots have ever been written, so a read of a slot nobody set
 * still answers with the caller's default rather than a zero.
 *
 * @c primary is whichever generated option carries the value's provenance: for
 * a flag pair that is the positively named half, since both publish into this
 * one slot but only one of them is the option the descriptor names.
 */
struct OptionEntry {
    std::string key;
    OptionKind  kind    = OptionKind::Value;
    OptionBase *primary = nullptr;

    std::atomic<std::uint8_t> assigned{0};

    std::atomic<bool>         bool_value{false};
    std::atomic<std::int64_t> int_value{0};
    std::atomic<double>       double_value{0.0};

    /*
     * A string cannot live in an atomic, so the slot publishes a pointer to an
     * immutable snapshot instead: a reader loads it and copies, and never sees
     * a string being rewritten underneath it. Superseded snapshots are kept
     * rather than freed, because a reader may still be copying one and there
     * is nothing to tell us when it has finished. That costs one string per
     * WRITE to this option, and an option is written at parse time and
     * essentially never again.
     */
    std::atomic<std::string const *>                string_value{nullptr};
    std::vector<std::unique_ptr<std::string const>> string_snapshots;
    std::mutex                                      snapshot_mutex;

    /// Called after a store, holding no lock. Appended at registration and
    /// never removed, so copying them out under the mutex is all a
    /// notification needs.
    std::vector<std::function<void()>> observers;
    std::mutex                         observer_mutex;
};

namespace {

/// Keys compare without regard to case, and a dash and an underscore are the
/// same character, so `BUFFER_SIZE` and `buffer-size` name one option.
std::string normalize_key(std::string_view key) {
    std::string out;
    out.reserve(key.size());
    for (char const c : key) {
        auto const uc = static_cast<unsigned char>(c);
        out.push_back(c == '_' ? '-' : static_cast<char>(std::tolower(uc)));
    }
    return out;
}

/**
 * @brief Everything registration allocates, kept alive until the process ends.
 *
 * The parser holds raw pointers into these and an option deregisters itself
 * when it dies, so the owning containers must outlive the parser. Entries live
 * in a deque because descriptors hold pointers to them.
 */
struct OwnedRegistrations {
    std::deque<OptionEntry>                         entries;
    std::unordered_map<std::string, OptionEntry *>  by_key;
    std::vector<std::unique_ptr<OptionBase>>        options;
    std::vector<std::unique_ptr<OptionCategory>>    categories;
    std::vector<std::shared_ptr<ExclusiveCategory>> exclusions;
    std::mutex                                      mutex;
    bool                                            frozen = false;
};

OwnedRegistrations &owned() {
    static OwnedRegistrations r;
    return r;
}

/// The category with this heading, created on first mention.
OptionCategory &category_named(std::string_view name) {
    auto &r = owned();
    for (auto const &c : r.categories) {
        if (c->name == name) {
            return *c;
        }
    }
    r.categories.push_back(std::make_unique<OptionCategory>(name));
    return *r.categories.back();
}

/// The entry for a key, created if absent. The caller holds the registry mutex.
OptionEntry &entry_locked(std::string_view key) {
    auto      &r          = owned();
    auto const normalized = normalize_key(key);
    if (auto const it = r.by_key.find(normalized); it != r.by_key.end()) {
        return *it->second;
    }
    r.entries.emplace_back();
    auto &entry = r.entries.back();
    entry.key   = normalized;
    r.by_key.emplace(normalized, &entry);
    return entry;
}

/// Apply the fields every generated option takes from its descriptor.
template <typename OptionType, typename Descriptor>
void apply_common(OptionType &opt, Descriptor const &desc) {
    if (desc.hidden) {
        opt.visibility = Visibility::Hidden;
    }
}

} // namespace

OptionEntry *entry_of(std::string_view key) {
    auto                  &r = owned();
    std::scoped_lock const lock(r.mutex);
    auto const             it = r.by_key.find(normalize_key(key));
    return it == r.by_key.end() ? nullptr : it->second;
}

OptionEntry &entry_for_write(std::string_view key) {
    auto                  &r = owned();
    std::scoped_lock const lock(r.mutex);
    return entry_locked(key);
}

bool has(OptionEntry const &entry, ValueKind kind) noexcept {
    return (entry.assigned.load(std::memory_order_acquire) & static_cast<std::uint8_t>(kind)) != 0;
}

/// Run an entry's observers holding no lock, which is the point of copying
/// them out first: an observer is free to read or write any option it likes.
void notify(OptionEntry &entry) {
    std::vector<std::function<void()>> callbacks;
    {
        std::scoped_lock const lock(entry.observer_mutex);
        callbacks = entry.observers;
    }
    for (auto const &fn : callbacks) {
        fn();
    }
}

void store_bool(OptionEntry &entry, bool value) {
    entry.bool_value.store(value, std::memory_order_release);
    entry.assigned.fetch_or(static_cast<std::uint8_t>(ValueKind::Bool), std::memory_order_release);
    notify(entry);
}

void store_int(OptionEntry &entry, std::int64_t value) {
    entry.int_value.store(value, std::memory_order_release);
    entry.assigned.fetch_or(static_cast<std::uint8_t>(ValueKind::Int), std::memory_order_release);
    notify(entry);
}

void store_double(OptionEntry &entry, double value) {
    entry.double_value.store(value, std::memory_order_release);
    entry.assigned.fetch_or(static_cast<std::uint8_t>(ValueKind::Double), std::memory_order_release);
    notify(entry);
}

void store_string(OptionEntry &entry, std::string value) {
    auto        snapshot  = std::make_unique<std::string const>(std::move(value));
    auto const *published = snapshot.get();
    {
        std::scoped_lock const lock(entry.snapshot_mutex);
        entry.string_snapshots.push_back(std::move(snapshot));
    }
    entry.string_value.store(published, std::memory_order_release);
    entry.assigned.fetch_or(static_cast<std::uint8_t>(ValueKind::String), std::memory_order_release);
    notify(entry);
}

} // namespace detail

std::string derive_key(std::string_view long_name) {
    constexpr std::string_view prefix = "einsums:";

    std::string_view rest = long_name;
    if (rest.starts_with(prefix)) {
        rest.remove_prefix(prefix.size());
    }

    std::string out(rest);
    std::ranges::replace(out, ':', '-');
    return out;
}

std::string derive_negated_name(std::string_view long_name) {
    auto const pos = long_name.rfind(':');
    if (pos == std::string_view::npos) {
        return "no-" + std::string(long_name);
    }
    return std::string(long_name.substr(0, pos + 1)) + "no-" + std::string(long_name.substr(pos + 1));
}

void register_option(ConfigOption<bool> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);

    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    assert(!r.frozen && "an option must be registered before the command line is parsed");

    auto &cat   = detail::category_named(opt.category);
    auto &entry = detail::entry_locked(derive_key(opt.name));
    entry.kind  = OptionKind::Flag;

    // The positive spelling is the one the descriptor declares and the one
    // --help shows; the negation is generated so that a default-true option
    // can be turned off without anyone hand-writing an inverted name.
    auto const negated = derive_negated_name(opt.name);

    // Publishing through the setter rather than a Location pointer is what
    // lets the value live in an atomic: there is no bool to take the address
    // of any more.
    auto publish = [&entry](bool const &v, Source) { detail::store_bool(entry, v); };

    auto yes                   = std::make_unique<Flag>(opt.name, std::initializer_list<char>{}, opt.help, cat);
    yes->setter                = publish;
    yes->value                 = opt.default_value;
    yes->default_value         = opt.default_value;
    yes->implicit_on           = true;
    yes->has_implicit_override = true;
    yes->extra_annotation      = fmt::format("[negate: --{}]", negated);
    detail::apply_common(*yes, opt);

    auto no                   = std::make_unique<Flag>(negated, std::initializer_list<char>{}, opt.help, cat);
    no->setter                = publish;
    no->value                 = opt.default_value;
    no->default_value         = opt.default_value;
    no->implicit_on           = false;
    no->has_implicit_override = true;
    // Listing both halves of every pair would double the help table for no
    // information; the positive entry names its twin instead.
    no->visibility = Visibility::Hidden;
    // Only one of the pair may re-apply the shared default at the start of a
    // parse, or the second would overwrite what the first just decided.
    no->set_on_unseen = false;

    // Naming both at the same precedence level is a contradiction, and the
    // parser already knows how to report one.
    auto exclusion = std::make_shared<ExclusiveCategory>();
    exclusion->options.push_back(yes.get());
    exclusion->options.push_back(no.get());
    yes->exclusions = exclusion.get();
    no->exclusions  = exclusion.get();

    entry.primary = yes.get();

    r.options.push_back(std::move(yes));
    r.options.push_back(std::move(no));
    r.exclusions.push_back(std::move(exclusion));

    // The slot answers with the declared default from the moment it exists,
    // not merely from the moment the parser gets around to it.
    detail::store_bool(entry, opt.default_value);

    opt.entry.store(&entry, std::memory_order_release);
}

namespace {

/**
 * @brief The shared body of the three value-option registrations.
 *
 * @param store The slot writer for T. The generated option calls it on every
 *              assignment, whatever the source.
 */
template <typename T>
detail::OptionEntry &register_value_option(ConfigOption<T> &opt, void (*store)(detail::OptionEntry &, T)) {
    auto &r = detail::owned();

    auto &cat   = detail::category_named(opt.category);
    auto &entry = detail::entry_locked(derive_key(opt.name));

    auto option = std::make_unique<Opt<T>>(opt.name, std::initializer_list<char>{}, opt.help, cat);
    // A default that could not be written down at compile time is asked for
    // here, once, at the moment the option gains a value.
    option->value         = opt.default_provider != nullptr ? opt.default_provider() : T(opt.default_value);
    option->default_value = option->value;
    option->has_default   = true;
    option->setter        = [&entry, store](T const &v, Source) { store(entry, v); };
    if (opt.range.has_value()) {
        option->range = *opt.range;
    }
    if (!opt.value_name.empty()) {
        option->value_name = std::string(opt.value_name);
    }
    detail::apply_common(*option, opt);

    entry.primary = option.get();
    store(entry, option->value);
    r.options.push_back(std::move(option));
    return entry;
}

} // namespace

void register_option(ConfigOption<std::int64_t> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    assert(!r.frozen && "an option must be registered before the command line is parsed");

    auto &entry = register_value_option<std::int64_t>(opt, &detail::store_int);
    opt.entry.store(&entry, std::memory_order_release);
}

void register_option(ConfigOption<double> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    assert(!r.frozen && "an option must be registered before the command line is parsed");

    auto &entry = register_value_option<double>(opt, &detail::store_double);
    opt.entry.store(&entry, std::memory_order_release);
}

void register_option(ConfigOption<std::string> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    assert(!r.frozen && "an option must be registered before the command line is parsed");

    auto &entry = register_value_option<std::string>(opt, &detail::store_string);
    opt.entry.store(&entry, std::memory_order_release);
}

namespace {

/// Attach an observer to a descriptor that has already been registered.
template <typename T>
void attach_observer(ConfigOption<T> &opt, std::function<void()> fn) {
    auto *entry = opt.entry.load(std::memory_order_acquire);
    assert(entry != nullptr && "register_option must run before on_change");
    if (entry == nullptr) {
        return;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto                  &mutable_entry = *const_cast<detail::OptionEntry *>(entry);
    std::scoped_lock const lock(mutable_entry.observer_mutex);
    mutable_entry.observers.push_back(std::move(fn));
}

} // namespace

void on_change(ConfigOption<bool> &opt, std::function<void()> fn) {
    attach_observer(opt, std::move(fn));
}

void on_change(ConfigOption<std::int64_t> &opt, std::function<void()> fn) {
    attach_observer(opt, std::move(fn));
}

void on_change(ConfigOption<double> &opt, std::function<void()> fn) {
    attach_observer(opt, std::move(fn));
}

void on_change(ConfigOption<std::string> &opt, std::function<void()> fn) {
    attach_observer(opt, std::move(fn));
}

void freeze_registry() {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    r.frozen = true;
}

void unfreeze_registry_for_tests() {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    r.frozen = false;
}

std::vector<std::pair<std::string, std::string>> registered_option_values() {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);

    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(r.entries.size());
    for (auto const &entry : r.entries) {
        std::string rendered;
        if (detail::has(entry, detail::ValueKind::String)) {
            auto const *value = entry.string_value.load(std::memory_order_acquire);
            rendered          = value != nullptr ? fmt::format("\"{}\"", *value) : std::string{"\"\""};
        } else if (detail::has(entry, detail::ValueKind::Bool)) {
            rendered = entry.bool_value.load(std::memory_order_acquire) ? "true" : "false";
        } else if (detail::has(entry, detail::ValueKind::Int)) {
            rendered = fmt::format("{}", entry.int_value.load(std::memory_order_acquire));
        } else if (detail::has(entry, detail::ValueKind::Double)) {
            rendered = fmt::format("{}", entry.double_value.load(std::memory_order_acquire));
        } else {
            continue;
        }
        out.emplace_back(entry.key, std::move(rendered));
    }
    return out;
}

void verify_registered_options() {
#if defined(EINSUMS_DEBUG)
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);

    // by_key is keyed on the normalized spelling, so two descriptors resolving
    // to one key would have collided into a single entry and the second would
    // have overwritten the first's primary. Checking that every entry's key
    // still derives from its primary's name is what catches that.
    for (auto const &entry : r.entries) {
        if (entry.primary == nullptr) {
            continue; // an overflow key, which claims no name
        }
        assert(derive_key(entry.primary->long_name) == entry.key && "a config key must derive from the name it was registered under");
    }

    std::set<std::string> keys;
    for (auto const *option : Registry::instance().options) {
        if (option->is_positional || option->long_name.starts_with("no-") || option->long_name.contains(":no-")) {
            continue;
        }
        [[maybe_unused]] auto const inserted = keys.insert(derive_key(option->long_name)).second;
        assert(inserted && "two options resolve to the same config key");
    }
#endif
}

EINSUMS_NAMESPACE_END(cl)

EINSUMS_NAMESPACE_BEGIN(config)

namespace detail {

namespace {

/**
 * @brief Report a read that beat its own registration.
 *
 * Legal, and answered with the descriptor's declared default because there is
 * no other honest answer, but worth seeing while chasing an option that
 * appears not to take effect. Deliberately not routed through the logging
 * module: this module sits below it.
 */
void note_unregistered([[maybe_unused]] std::string_view name) noexcept {
#if defined(EINSUMS_DEBUG)
    static bool const trace = std::getenv("EINSUMS_OPTIONS_TRACE") != nullptr;
    if (trace) {
        std::fprintf(stderr, "einsums: option '%.*s' read before it was registered; using its declared default\n",
                     static_cast<int>(name.size()), name.data());
    }
#endif
}

} // namespace

bool read_bool(cl::detail::OptionEntry const *entry, std::string_view name, bool default_value) noexcept {
    if (entry == nullptr) {
        note_unregistered(name);
        return default_value;
    }
    return entry->bool_value.load(std::memory_order_acquire);
}

std::int64_t read_int(cl::detail::OptionEntry const *entry, std::string_view name, std::int64_t default_value) noexcept {
    if (entry == nullptr) {
        note_unregistered(name);
        return default_value;
    }
    return entry->int_value.load(std::memory_order_acquire);
}

double read_double(cl::detail::OptionEntry const *entry, std::string_view name, double default_value) noexcept {
    if (entry == nullptr) {
        note_unregistered(name);
        return default_value;
    }
    return entry->double_value.load(std::memory_order_acquire);
}

std::string read_string(cl::detail::OptionEntry const *entry, std::string_view name, std::string_view default_value) {
    if (entry == nullptr) {
        note_unregistered(name);
        return std::string(default_value);
    }
    auto const *value = entry->string_value.load(std::memory_order_acquire);
    return value != nullptr ? *value : std::string(default_value);
}

bool was_specified(cl::detail::OptionEntry const *entry) noexcept {
    return entry != nullptr && entry->primary != nullptr && entry->primary->was_specified();
}

bool dynamic_bool(std::string const &key, bool default_value) noexcept {
    auto const *entry = cl::detail::entry_of(key);
    if (entry == nullptr || !cl::detail::has(*entry, cl::detail::ValueKind::Bool)) {
        return default_value;
    }
    return entry->bool_value.load(std::memory_order_acquire);
}

std::int64_t dynamic_int(std::string const &key, std::int64_t default_value) noexcept {
    auto const *entry = cl::detail::entry_of(key);
    if (entry == nullptr || !cl::detail::has(*entry, cl::detail::ValueKind::Int)) {
        return default_value;
    }
    return entry->int_value.load(std::memory_order_acquire);
}

double dynamic_double(std::string const &key, double default_value) noexcept {
    auto const *entry = cl::detail::entry_of(key);
    if (entry == nullptr || !cl::detail::has(*entry, cl::detail::ValueKind::Double)) {
        return default_value;
    }
    return entry->double_value.load(std::memory_order_acquire);
}

std::string dynamic_string(std::string const &key, std::string const &default_value) {
    auto const *entry = cl::detail::entry_of(key);
    if (entry == nullptr || !cl::detail::has(*entry, cl::detail::ValueKind::String)) {
        return default_value;
    }
    auto const *value = entry->string_value.load(std::memory_order_acquire);
    return value != nullptr ? *value : default_value;
}

/*
 * A write goes to the slot and then, holding nothing, to whoever asked to hear
 * about it. A key no descriptor claims gets an entry of its own: that is all
 * that is left of the four string-keyed maps.
 */
void set_dynamic_bool(std::string const &key, bool value) {
    cl::detail::store_bool(cl::detail::entry_for_write(key), value);
}

void set_dynamic_int(std::string const &key, std::int64_t value) {
    cl::detail::store_int(cl::detail::entry_for_write(key), value);
}

void set_dynamic_double(std::string const &key, double value) {
    cl::detail::store_double(cl::detail::entry_for_write(key), value);
}

void set_dynamic_string(std::string const &key, std::string const &value) {
    cl::detail::store_string(cl::detail::entry_for_write(key), value);
}

} // namespace detail

EINSUMS_NAMESPACE_END(config)
