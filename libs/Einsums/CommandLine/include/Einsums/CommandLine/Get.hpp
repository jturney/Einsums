//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Source.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

/*
 * The reader's half of the option system, and the only part of it a module
 * that merely CONSUMES a value has to include. It declares the descriptor
 * type, the typed accessors, and the string-keyed escape hatch; it pulls in no
 * parser, no help renderer, and no fmt, because it is on its way to being
 * included by hundreds of translation units.
 *
 * A descriptor declares one option completely - name, help, category, type,
 * default - and is the single place any of those facts is spelled. The reader
 * names the descriptor rather than a string, so a typo is a compile error and
 * the type cannot be mismatched.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

namespace detail {

/**
 * @brief The registry's record for one registered option.
 *
 * Opaque on purpose: a reader only ever holds a pointer to one and hands it
 * back to the accessors below. Its layout is the store's business and changes
 * when the store does.
 */
struct OptionEntry;

/// The value type a descriptor stores its default in. Strings are held as
/// views so a descriptor stays constant-initializable.
template <typename T>
struct StorageOf {
    using type = T;
};

template <>
struct StorageOf<std::string> {
    using type = std::string_view;
};

template <typename T>
using storage_t = typename StorageOf<T>::type;

} // namespace detail

/// How an option is spelled on the command line.
enum struct OptionKind : std::uint8_t {
    Value, ///< Takes a value: `--einsums:log:level 3`.
    Flag   ///< Presence carries the meaning; a `--no-` twin is generated for it.
};

/**
 * @brief An option declared completely, in one place.
 *
 * Descriptors are aggregates of literal types plus one atomic back-pointer, so
 * a namespace-scope descriptor is constant-initialized: there is no static
 * initialization order to reason about, and a read that happens before the
 * option is registered simply sees @ref default_value.
 *
 * @tparam T The option's value type: `bool`, `std::int64_t`, `double`, or
 *           `std::string`.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
struct ConfigOption {
    using value_type   = T;
    using storage_type = detail::storage_t<T>;

    /// The command-line long name, e.g. `einsums:log:level`. Every other
    /// spelling - the config key, the environment variable, the help entry -
    /// is derived from it.
    std::string_view name;
    /// The `--help` description.
    std::string_view help;
    /// The `--help` heading this option groups under.
    std::string_view category;
    /// The value in effect when nothing else supplies one.
    storage_type default_value{};
    /// Flag or value option.
    OptionKind kind = OptionKind::Value;
    /// The placeholder shown in help for a value option, e.g. `N`.
    std::string_view value_name{};
    /// Inclusive bounds enforced after parsing, for numeric options.
    std::optional<Range> range{};
    /// Keep the option out of `--help`.
    bool hidden = false;
    /// A default that cannot be written down at compile time - a temp
    /// directory, a name carrying the process id. When set it wins over
    /// @ref default_value, which stays the fallback for a read that beats
    /// registration.
    value_type (*default_provider)() = nullptr;

    /// Filled in by registration; read by the accessors. Null until the owning
    /// module registers the option.
    std::atomic<detail::OptionEntry const *> entry{nullptr};
};

/// Declare a boolean option. Registration also generates the `--no-` spelling,
/// so a default-true flag needs no hand-written negation.
constexpr ConfigOption<bool> config_flag(std::string_view name, std::string_view help, std::string_view category,
                                         bool default_value = false) {
    return ConfigOption<bool>{.name = name, .help = help, .category = category, .default_value = default_value, .kind = OptionKind::Flag};
}

/// Declare a value option.
template <typename T>
constexpr ConfigOption<T> config_opt(std::string_view name, std::string_view help, std::string_view category,
                                     detail::storage_t<T> default_value, std::string_view value_name = {},
                                     std::optional<Range> range = std::nullopt) {
    return ConfigOption<T>{.name          = name,
                           .help          = help,
                           .category      = category,
                           .default_value = default_value,
                           .kind          = OptionKind::Value,
                           .value_name    = value_name,
                           .range         = range};
}

/// Declare a value option whose default is only knowable at run time, such as
/// the system temporary directory or a name built from the process id.
template <typename T>
constexpr ConfigOption<T> config_opt_computed(std::string_view name, std::string_view help, std::string_view category, T (*provider)(),
                                              std::string_view value_name = {}) {
    return ConfigOption<T>{.name             = name,
                           .help             = help,
                           .category         = category,
                           .kind             = OptionKind::Value,
                           .value_name       = value_name,
                           .default_provider = provider};
}

/**
 * @brief The config key derived from a command-line long name.
 *
 * The rule: drop a leading `einsums:`, then turn every remaining `:` into `-`.
 * So `einsums:log:level` keys on `log-level` and `einsums:buffer-size` on
 * `buffer-size`. It is the only relationship between the two spellings, which
 * is what keeps them from drifting.
 */
EINSUMS_EXPORT std::string derive_key(std::string_view long_name);

/**
 * @brief The `--no-` spelling generated for a flag.
 *
 * `einsums:debug:attach-debugger` pairs with `einsums:debug:no-attach-debugger`:
 * the negation goes on the last segment, where a reader looks for it.
 */
EINSUMS_EXPORT std::string derive_negated_name(std::string_view long_name);

EINSUMS_NAMESPACE_END(cl)

EINSUMS_NAMESPACE_BEGIN(config)

namespace detail {

/*
 * The accessors are thin wrappers over these, so the store can change shape
 * without recompiling every reader.
 */
EINSUMS_EXPORT bool read_bool(cl::detail::OptionEntry const *entry, std::string_view name, bool default_value) noexcept;
EINSUMS_EXPORT std::int64_t read_int(cl::detail::OptionEntry const *entry, std::string_view name, std::int64_t default_value) noexcept;
EINSUMS_EXPORT double       read_double(cl::detail::OptionEntry const *entry, std::string_view name, double default_value) noexcept;
EINSUMS_EXPORT std::string read_string(cl::detail::OptionEntry const *entry, std::string_view name, std::string_view default_value);

/// True when something other than the declared default supplied the value.
EINSUMS_EXPORT bool was_specified(cl::detail::OptionEntry const *entry) noexcept;

EINSUMS_EXPORT bool dynamic_bool(std::string const &key, bool default_value) noexcept;
EINSUMS_EXPORT std::int64_t dynamic_int(std::string const &key, std::int64_t default_value) noexcept;
EINSUMS_EXPORT double       dynamic_double(std::string const &key, double default_value) noexcept;
EINSUMS_EXPORT std::string dynamic_string(std::string const &key, std::string const &default_value);

EINSUMS_EXPORT void set_dynamic_bool(std::string const &key, bool value);
EINSUMS_EXPORT void set_dynamic_int(std::string const &key, std::int64_t value);
EINSUMS_EXPORT void set_dynamic_double(std::string const &key, double value);
EINSUMS_EXPORT void set_dynamic_string(std::string const &key, std::string const &value);

} // namespace detail

/**
 * @brief The value in effect for an option.
 *
 * Reading an option that has not been registered yet - before
 * `einsums::initialize`, say - yields the descriptor's declared default rather
 * than a zero, so there is no second default to keep in step at the call site.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
T get(cl::ConfigOption<T> const &opt) {
    auto const *entry = opt.entry.load(std::memory_order_acquire);
    if constexpr (std::is_same_v<T, bool>) {
        return detail::read_bool(entry, opt.name, opt.default_value);
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return detail::read_int(entry, opt.name, opt.default_value);
    } else if constexpr (std::is_same_v<T, double>) {
        return detail::read_double(entry, opt.name, opt.default_value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (entry == nullptr && opt.default_provider != nullptr) {
            return opt.default_provider();
        }
        return detail::read_string(entry, opt.name, opt.default_value);
    } else {
        static_assert(cl::detail::always_false<T>, "config::get supports bool, std::int64_t, double, and std::string options.");
    }
}

/**
 * @brief The value, but only when something explicitly supplied it.
 *
 * Distinguishes "left at its default" from "set to the same value the default
 * happens to be", which the plain @ref get cannot.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
std::optional<T> try_get(cl::ConfigOption<T> const &opt) {
    auto const *entry = opt.entry.load(std::memory_order_acquire);
    if (!detail::was_specified(entry)) {
        return std::nullopt;
    }
    return get(opt);
}

/**
 * @brief Read an option by key, for keys built at run time.
 *
 * The escape hatch for genuinely dynamic names - the per-pass flags the
 * optimizer reads, and whatever a config file carries that no descriptor
 * claims. Anything known at compile time belongs in a descriptor instead.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
T get_dynamic(std::string const &key, T default_value) {
    if constexpr (std::is_same_v<T, bool>) {
        return detail::dynamic_bool(key, default_value);
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return detail::dynamic_int(key, default_value);
    } else if constexpr (std::is_same_v<T, double>) {
        return detail::dynamic_double(key, default_value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return detail::dynamic_string(key, default_value);
    } else {
        static_assert(cl::detail::always_false<T>, "config::get_dynamic supports bool, std::int64_t, double, and std::string.");
    }
}

/**
 * @brief Write an option by key.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void set_dynamic(std::string const &key, T value) {
    if constexpr (std::is_same_v<T, bool>) {
        detail::set_dynamic_bool(key, value);
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        detail::set_dynamic_int(key, value);
    } else if constexpr (std::is_same_v<T, double>) {
        detail::set_dynamic_double(key, value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        detail::set_dynamic_string(key, value);
    } else {
        static_assert(cl::detail::always_false<T>, "config::set_dynamic supports bool, std::int64_t, double, and std::string.");
    }
}

/**
 * @brief Write an option through its descriptor.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void set(cl::ConfigOption<T> const &opt, T value) {
    set_dynamic<T>(std::string(cl::derive_key(opt.name)), std::move(value));
}

EINSUMS_NAMESPACE_END(config)
