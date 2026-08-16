//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Types.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

/*
 * Where a descriptor becomes a registered option, and where a typed read
 * becomes a value.
 *
 * The store is still the GlobalConfigMap: registration binds each generated
 * option to the map slot its derived key names, exactly as the hand-written
 * registrations did, so a descriptor read and a legacy string read see the
 * same value. Only the spellings have collapsed into one place.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

namespace detail {

/**
 * @brief What a descriptor points at once it has been registered.
 *
 * @c primary is whichever generated option carries the value's provenance:
 * for a flag pair that is the positively named half, since both write the same
 * slot but only one of them is the option the descriptor names.
 */
struct OptionEntry {
    std::string key;
    OptionKind  kind    = OptionKind::Value;
    OptionBase *primary = nullptr;
};

namespace {

/**
 * @brief Everything registration allocates, kept alive until the process ends.
 *
 * The parser holds raw pointers into these, and an option deregisters itself
 * when it dies, so the owning containers must outlive the parser. Entries live
 * in a deque because descriptors hold pointers to them.
 */
struct OwnedRegistrations {
    std::deque<OptionEntry>                         entries;
    std::vector<std::unique_ptr<OptionBase>>        options;
    std::vector<std::unique_ptr<OptionCategory>>    categories;
    std::vector<std::shared_ptr<ExclusiveCategory>> exclusions;
    std::mutex                                      mutex;
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

/// Apply the fields every generated option takes from its descriptor.
template <typename OptionType, typename Descriptor>
void apply_common(OptionType &opt, Descriptor const &desc) {
    if (desc.hidden) {
        opt.visibility = Visibility::Hidden;
    }
}

} // namespace
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

    auto       &cat  = detail::category_named(opt.category);
    auto const  key  = derive_key(opt.name);
    auto       &map  = GlobalConfigMap::get_singleton().get_bool_map()->get_value();
    bool *const slot = &map[key];

    // The positive spelling is the one the descriptor declares and the one
    // --help shows; the negation is generated so that a default-true option
    // can be turned off without anyone hand-writing an inverted name.
    auto const negated = derive_negated_name(opt.name);

    auto yes                   = std::make_unique<Flag>(opt.name, std::initializer_list<char>{}, opt.help, cat);
    yes->bound                 = slot;
    yes->value                 = opt.default_value;
    yes->default_value         = opt.default_value;
    yes->implicit_on           = true;
    yes->has_implicit_override = true;
    yes->extra_annotation      = fmt::format("[negate: --{}]", negated);
    detail::apply_common(*yes, opt);

    auto no                   = std::make_unique<Flag>(negated, std::initializer_list<char>{}, opt.help, cat);
    no->bound                 = slot;
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

    r.entries.push_back(detail::OptionEntry{.key = key, .kind = OptionKind::Flag, .primary = yes.get()});
    r.options.push_back(std::move(yes));
    r.options.push_back(std::move(no));
    r.exclusions.push_back(std::move(exclusion));

    opt.entry.store(&r.entries.back(), std::memory_order_release);
}

namespace {

/// The shared body of the three value-option registrations.
template <typename T, typename Map>
detail::OptionEntry *register_value_option(ConfigOption<T> &opt, Map &map) {
    auto &r = detail::owned();

    auto      &cat  = detail::category_named(opt.category);
    auto const key  = derive_key(opt.name);
    T *const   slot = &map[key];

    auto option           = std::make_unique<Opt<T>>(opt.name, std::initializer_list<char>{}, opt.help, cat);
    option->bound         = slot;
    option->value         = T(opt.default_value);
    option->default_value = T(opt.default_value);
    option->has_default   = true;
    if (opt.range.has_value()) {
        option->range = *opt.range;
    }
    if (!opt.value_name.empty()) {
        option->value_name = std::string(opt.value_name);
    }
    detail::apply_common(*option, opt);

    r.entries.push_back(detail::OptionEntry{.key = key, .kind = OptionKind::Value, .primary = option.get()});
    r.options.push_back(std::move(option));
    return &r.entries.back();
}

} // namespace

void register_option(ConfigOption<std::int64_t> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    auto *entry = register_value_option(opt, GlobalConfigMap::get_singleton().get_int_map()->get_value());
    opt.entry.store(entry, std::memory_order_release);
}

void register_option(ConfigOption<double> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    auto *entry = register_value_option(opt, GlobalConfigMap::get_singleton().get_double_map()->get_value());
    opt.entry.store(entry, std::memory_order_release);
}

void register_option(ConfigOption<std::string> &opt) {
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);
    if (opt.entry.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    auto *entry = register_value_option(opt, GlobalConfigMap::get_singleton().get_string_map()->get_value());
    opt.entry.store(entry, std::memory_order_release);
}

void verify_registered_options() {
#if defined(EINSUMS_DEBUG)
    auto                  &r = detail::owned();
    std::scoped_lock const lock(r.mutex);

    std::set<std::string, std::less<>> keys;
    for (auto const &entry : r.entries) {
        assert(entry.primary != nullptr && "a registered option must own the option carrying its value");
        assert(derive_key(entry.primary->long_name) == entry.key && "a config key must derive from the name it was registered under");
        [[maybe_unused]] auto const inserted = keys.insert(entry.key).second;
        assert(inserted && "two option descriptors resolve to the same config key");
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
 * Legal, and answered with the descriptor's declared default (there is no
 * other honest answer), but worth seeing while chasing an option that appears
 * not to take effect. Deliberately not routed through the logging module: this
 * module sits below it.
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
    return GlobalConfigMap::get_singleton().get_bool(entry->key, default_value);
}

std::int64_t read_int(cl::detail::OptionEntry const *entry, std::string_view name, std::int64_t default_value) noexcept {
    if (entry == nullptr) {
        note_unregistered(name);
        return default_value;
    }
    return GlobalConfigMap::get_singleton().get_int(entry->key, default_value);
}

double read_double(cl::detail::OptionEntry const *entry, std::string_view name, double default_value) noexcept {
    if (entry == nullptr) {
        note_unregistered(name);
        return default_value;
    }
    return GlobalConfigMap::get_singleton().get_double(entry->key, default_value);
}

std::string read_string(cl::detail::OptionEntry const *entry, std::string_view name, std::string_view default_value) {
    if (entry == nullptr) {
        note_unregistered(name);
        return std::string(default_value);
    }
    return GlobalConfigMap::get_singleton().get_string(entry->key, std::string(default_value));
}

bool was_specified(cl::detail::OptionEntry const *entry) noexcept {
    return entry != nullptr && entry->primary != nullptr && entry->primary->was_specified();
}

bool dynamic_bool(std::string const &key, bool default_value) noexcept {
    return GlobalConfigMap::get_singleton().get_bool(key, default_value);
}

std::int64_t dynamic_int(std::string const &key, std::int64_t default_value) noexcept {
    return GlobalConfigMap::get_singleton().get_int(key, default_value);
}

double dynamic_double(std::string const &key, double default_value) noexcept {
    return GlobalConfigMap::get_singleton().get_double(key, default_value);
}

std::string dynamic_string(std::string const &key, std::string const &default_value) {
    return GlobalConfigMap::get_singleton().get_string(key, default_value);
}

void set_dynamic_bool(std::string const &key, bool value) {
    GlobalConfigMap::get_singleton().set_bool(key, value);
}

void set_dynamic_int(std::string const &key, std::int64_t value) {
    GlobalConfigMap::get_singleton().set_int(key, value);
}

void set_dynamic_double(std::string const &key, double value) {
    GlobalConfigMap::get_singleton().set_double(key, value);
}

void set_dynamic_string(std::string const &key, std::string const &value) {
    GlobalConfigMap::get_singleton().set_string(key, value);
}

} // namespace detail

EINSUMS_NAMESPACE_END(config)
