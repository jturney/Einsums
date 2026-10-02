//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Source.hpp>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

/*
 * The reader's half: descriptors, typed accessors and the dynamic escape hatch. No parser or fmt,
 * as hundreds of TUs include it. Readers name a descriptor, so a typo fails to compile.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

namespace detail {

/**
 * @brief The registry's record for one registered option.
 *
 * Opaque to readers.
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

/**
 * @brief An option declared completely, in one place.
 *
 * Constant-initialized, so a read before registration sees @ref default_value.
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

    /// The command-line long name, e.g. `einsums:log:level`, from which every other spelling derives.
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
    /// A default known only at run time. Wins over @ref default_value, which serves reads before registration.
    value_type (*default_provider)() = nullptr;

    /// A cache of the registry entry. Each binary has its own copy of the descriptor, which
    /// registration never fills, so a null entry is looked up by key and cached here.
    mutable std::atomic<detail::OptionEntry const *> entry{nullptr};
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
 * Drop a leading `einsums:` and turn `:` into `-`: `einsums:log:level` keys on `log-level`.
 */
EINSUMS_EXPORT std::string derive_key(std::string_view long_name);

/**
 * @brief The `--no-` spelling generated for a flag.
 *
 * The negation goes on the last segment: `einsums:debug:no-attach-debugger`.
 */
EINSUMS_EXPORT std::string derive_negated_name(std::string_view long_name);

EINSUMS_NAMESPACE_END(cl)

EINSUMS_NAMESPACE_BEGIN(config)

namespace detail {

// Out of line, so the store can change without recompiling readers.
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

/// The registered entry for the option with this long name, or null if no copy
/// of its descriptor has been registered yet.
EINSUMS_EXPORT cl::detail::OptionEntry const *registered_entry(std::string_view name);

/// The entry @p opt reads, looked up once per copy. Null (and not cached) before registration.
template <typename T>
cl::detail::OptionEntry const *entry_for(cl::ConfigOption<T> const &opt) {
    if (auto const *entry = opt.entry.load(std::memory_order_acquire); entry != nullptr) {
        return entry;
    }
    auto const *entry = registered_entry(opt.name);
    if (entry != nullptr) {
        opt.entry.store(entry, std::memory_order_release);
    }
    return entry;
}

} // namespace detail

/**
 * @brief The value in effect for an option.
 *
 * Before registration, the descriptor's declared default.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
T get(cl::ConfigOption<T> const &opt) {
    auto const *entry = detail::entry_for(opt);
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
 * Distinguishes "left at the default" from "set to the default's value".
 *
 * @versionadded{2.0.0}
 */
template <typename T>
std::optional<T> try_get(cl::ConfigOption<T> const &opt) {
    auto const *entry = detail::entry_for(opt);
    if (!detail::was_specified(entry)) {
        return std::nullopt;
    }
    return get(opt);
}

/**
 * @brief Read an option by key, for keys built at run time.
 *
 * Only for names built at run time, such as the optimizer's per-pass flags.
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
