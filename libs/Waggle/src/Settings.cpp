//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Settings.hpp"

#include <Waggle/Config.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <type_traits>

WAGGLE_NAMESPACE_BEGIN

namespace {

/// Call @p f(index, name, the update's member, the settings' member) for every setting, in
/// @ref SettingsUpdate's member order. The one list of settings the rest of this file walks.
template <typename Update, typename F>
void for_each_setting(Update &update, Settings &settings, F &&f) {
    f(0, "record", update.record, settings.record);
    f(1, "report", update.report, settings.report);
    f(2, "report_file", update.report_file, settings.report_file);
    f(3, "report_append", update.report_append, settings.report_append);
    f(4, "report_detailed", update.report_detailed, settings.report_detailed);
    f(5, "save", update.save, settings.save);
    f(6, "server", update.server, settings.server);
    f(7, "port", update.port, settings.port);
    f(8, "wait_for_viewer", update.wait_for_viewer, settings.wait_for_viewer);
    f(9, "max_distinct_children", update.max_distinct_children, settings.max_distinct_children);
}

template <typename T>
auto show(T const &value) -> std::string {
    if constexpr (std::is_same_v<T, std::string>) {
        return fmt::format("\"{}\"", value);
    } else {
        return fmt::format("{}", value);
    }
}

auto parse_bool(std::string value) -> std::optional<bool> {
    std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no") {
        return false;
    }
    return std::nullopt;
}

auto parse_int(std::string const &value) -> std::optional<std::int64_t> {
    std::int64_t out{};
    auto const [end, ec] = std::from_chars(value.data(), value.data() + value.size(), out);
    if (ec != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return out;
}

} // namespace

auto SettingsStore::add_from_text(SettingsUpdate &update, std::string_view key, std::string const &value) -> bool {
    Settings unused;
    bool     known = false;
    bool     good  = false;
    for_each_setting(update, unused, [&](int, char const *name, auto &wanted, auto const &) {
        if (known || key != name) {
            return;
        }
        known      = true;
        using Type = std::remove_cvref_t<decltype(*wanted)>;
        if constexpr (std::is_same_v<Type, bool>) {
            if (auto const v = parse_bool(value)) {
                wanted = *v;
                good   = true;
            }
        } else if constexpr (std::is_same_v<Type, std::int64_t>) {
            if (auto const v = parse_int(value)) {
                wanted = *v;
                good   = true;
            }
        } else {
            wanted = value;
            good   = true;
        }
    });
    return known && good;
}

auto SettingsStore::text(std::string_view key) const -> std::optional<std::string> {
    std::optional<std::string> out;
    SettingsUpdate             unused;
    Settings                   copy = _settings;
    for_each_setting(unused, copy, [&](int, char const *name, auto const &, auto const &current) {
        if (key != name) {
            return;
        }
        using Type = std::remove_cvref_t<decltype(current)>;
        if constexpr (std::is_same_v<Type, bool>) {
            out = current ? "true" : "false";
        } else if constexpr (std::is_same_v<Type, std::int64_t>) {
            out = std::to_string(current);
        } else {
            out = current;
        }
    });
    return out;
}

auto SettingsStore::process_environment() -> EnvironmentReader {
    return [](char const *name) -> std::optional<std::string> {
        char const *value = std::getenv(name); // NOLINT(concurrency-mt-unsafe): read at construction only
        return value == nullptr ? std::nullopt : std::optional<std::string>(value);
    };
}

auto SettingsStore::apply_environment(EnvironmentReader const &getenv) -> std::vector<std::string> {
    std::vector<std::string> problems;
    SettingsUpdate           update;

    auto const take_bool = [&](char const *name, std::optional<bool> &into, bool invert = false) {
        if (auto const raw = getenv(name)) {
            if (auto const value = parse_bool(*raw)) {
                into = invert ? !*value : *value;
            } else {
                problems.push_back(fmt::format("ignoring {}=\"{}\": expected 1/0, true/false, on/off or yes/no", name, *raw));
            }
        }
    };
    auto const take_int = [&](char const *name, std::optional<std::int64_t> &into) {
        if (auto const raw = getenv(name)) {
            if (auto const value = parse_int(*raw)) {
                into = *value;
            } else {
                problems.push_back(fmt::format("ignoring {}=\"{}\": expected an integer", name, *raw));
            }
        }
    };
    auto const take_string = [&](char const *name, std::optional<std::string> &into) {
        if (auto const raw = getenv(name)) {
            into = *raw;
        }
    };

    take_bool("WAGGLE_DISABLE", update.record, /*invert=*/true);
    take_bool("WAGGLE_REPORT", update.report);
    take_string("WAGGLE_REPORT_FILE", update.report_file);
    take_bool("WAGGLE_REPORT_APPEND", update.report_append);
    take_bool("WAGGLE_REPORT_DETAILED", update.report_detailed);
    take_string("WAGGLE_SAVE", update.save);
    take_bool("WAGGLE_SERVER", update.server);
    take_int("WAGGLE_PORT", update.port);
    take_bool("WAGGLE_WAIT_FOR_VIEWER", update.wait_for_viewer);
    take_int("WAGGLE_MAX_DISTINCT_CHILDREN", update.max_distinct_children);

    // Below explicit settings: fill only what no library set.
    for_each_setting(update, _settings, [this](int index, char const * /*name*/, auto const &wanted, auto &current) {
        if (wanted && !_explicit[index]) {
            current = *wanted;
        }
    });
    return problems;
}

auto SettingsStore::configure(SettingsUpdate const &update) -> std::vector<std::string> {
    std::vector<std::string> refused;
    for_each_setting(update, _settings, [&](int index, char const *name, auto const &wanted, auto &current) {
        if (!wanted) {
            return;
        }
        if (!_explicit[index]) {
            current          = *wanted;
            _explicit[index] = true;
        } else if (current != *wanted) {
            refused.push_back(fmt::format("{} was already set to {}; keeping it, not {}", name, show(current), show(*wanted)));
        }
    });
    return refused;
}

void SettingsStore::override_settings(SettingsUpdate const &update) {
    for_each_setting(update, _settings, [this](int index, char const * /*name*/, auto const &wanted, auto &current) {
        if (wanted) {
            current          = *wanted;
            _explicit[index] = true;
        }
    });
}

WAGGLE_NAMESPACE_END
