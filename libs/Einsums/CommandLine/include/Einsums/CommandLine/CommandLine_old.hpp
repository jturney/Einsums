//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/Casting.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::cl {

// -------------------------- Small utilities ------------------------------- //

struct StringRef {
    std::string_view s;
    constexpr StringRef() = default;
    constexpr StringRef(char const *c) : s(c) {}
    constexpr StringRef(std::string_view v) : s(v) {}
    constexpr operator std::string_view() const { return s; }
};

struct Range {
    long long min_v = (std::numeric_limits<long long>::min)();
    long long max_v = (std::numeric_limits<long long>::max)();
};

inline Range RangeBetween(long long min_v, long long max_v) {
    return Range{min_v, max_v};
}

struct ParseResult {
    bool ok        = true;
    int  exit_code = 0;
};

// Forward decls
struct OptionBase;
struct OptionCategory;

// -------------------------- Registry ------------------------------------- //

struct Registry {
    std::vector<OptionBase *>     options;
    std::vector<OptionCategory *> categories;

    static Registry &instance() {
        static Registry R;
        return R;
    }

    void add_option(OptionBase *o) { options.push_back(o); }
    void add_category(OptionCategory *c) { categories.push_back(c); }

    void clear_for_tests() {
        options.clear();
        categories.clear();
    }
};

// -------------------------- Categories ----------------------------------- //

struct OptionCategory {
    std::string name;
    explicit OptionCategory(StringRef n) : name(n.s) { Registry::instance().add_category(this); }
};

// -------------------------- Parsing helpers ------------------------------ //

template <typename T>
static bool parse_value(std::string_view, T &out, std::string &err);

inline bool parse_value(std::string_view sv, std::string &out, std::string &) {
    out.assign(sv.begin(), sv.end());
    return true;
}
inline bool parse_value(std::string_view sv, bool &out, std::string &err) {
    if (sv == "1" || sv == "true" || sv == "on" || sv == "yes") {
        out = true;
        return true;
    }
    if (sv == "0" || sv == "false" || sv == "off" || sv == "no") {
        out = false;
        return true;
    }
    err = fmt::format("invalid boolean '{}', expected true/false", sv);
    return false;
}
inline bool parse_value(std::string_view sv, int &out, std::string &err) {
    auto *b = sv.data();
    auto *e = b + sv.size();
    int   v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    err = fmt::format("invalid integer '{}'", sv);
    return false;
}
inline bool parse_value(std::string_view sv, long &out, std::string &err) {
    auto     *b = sv.data();
    auto     *e = b + sv.size();
    long long v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    err = fmt::format("invalid integer '{}'", sv);
    return false;
}
inline bool parse_value(std::string_view sv, long long &out, std::string &err) {
    auto     *b = sv.data();
    auto     *e = b + sv.size();
    long long v{};
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc() && p == e) {
        out = v;
        return true;
    }
    err = fmt::format("invalid integer '{}'", sv);
    return false;
}
inline bool parse_value(std::string_view sv, double &out, std::string &err) {
    try {
        size_t      idx = 0;
        std::string tmp(sv);
        double      v = std::stod(tmp, &idx);
        if (idx == tmp.size()) {
            out = v;
            return true;
        }
    } catch (...) {
    }
    err = fmt::format("invalid real '{}'", sv);
    return false;
}

// -------------------------- Core types ----------------------------------- //

enum struct Visibility : uint8_t { Normal, Hidden };
enum struct Occurrence : uint8_t { Optional, Required, ZeroOrMore, OneOrMore };
enum struct ValueExpected : uint8_t { ValueDisallowed, ValueOptional, ValueRequired };

struct Positional {};

// Base option (no subcommand affinity)
struct OptionBase {
    std::string       long_name;   // "--long"
    std::vector<char> short_names; // {'v'}
    std::string       help;
    OptionCategory   *category       = nullptr;
    Visibility        visibility     = Visibility::Normal;
    Occurrence        occurrence     = Occurrence::Optional;
    ValueExpected     value_expected = ValueExpected::ValueOptional;
    bool              is_positional  = false;
    bool              seen_cli       = false;
    bool              seen_config    = false;
    int               occurrences    = 0;

    std::function<void()> on_seen;

    OptionBase(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, OptionCategory *cat)
        : long_name(longName.s), short_names(shorts), help(helpText.s), category(cat) {
        Registry::instance().add_option(this);
    }

    OptionBase(StringRef positional_name, Positional, StringRef helpText)
        : long_name(positional_name.s), help(helpText.s), is_positional(true) {
        Registry::instance().add_option(this);
    }

    virtual ~OptionBase() = default;

    virtual bool parse_token(std::string_view key, std::optional<std::string_view> val, std::string &error, bool from_config = false) = 0;

    virtual void print_help_line(std::string_view prog, size_t pad_long, size_t pad_short) const = 0;

    virtual bool validate(std::string &error) const {
        (void)error;
        return true;
    }

    virtual void finalize_default() {}
};

// -------------------------- Location & Setter ---------------------------- //

template <typename T>
struct Location {
    T *ptr = nullptr;
    explicit Location(T &r) : ptr(&r) {}
};

template <typename T>
struct Setter {
    std::function<void(T const &)> fn;
    Setter() = default;
    template <typename F>
    explicit Setter(F &&f) : fn(std::forward<F>(f)) {}
};

// -------------------------- Named-arg tags ------------------------------- //

template <typename T>
struct DefaultTag {
    T v;
};
template <typename T>
struct ImplicitValueTag {
    T v;
};

// template <typename T> DefaultTag(std::decay_t<T>)->DefaultTag<std::decay_t<T>>;
// template <typename T> ImplicitValueTag(std::decay_t<T>)->ImplicitValueTag<std::decay_t<T>>;

template <typename T>
inline auto Default(T &&v) -> DefaultTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

template <typename T>
inline auto ImplicitValue(T &&v) -> ImplicitValueTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

struct ValueNameTag {
    std::string name;
};
inline ValueNameTag ValueName(std::string n) {
    return {std::move(n)};
}

// -------------------------- Flag ---------------------------------------- //

struct Flag : OptionBase {
    bool                              value = false;
    bool                             *bound = nullptr;
    std::function<void(bool const &)> setter;
    bool                              implicit_on           = true;
    bool                              has_implicit_override = false;

    template <class... Args>
    Flag(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueOptional;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    Flag &OnSet(std::function<void(bool const &)> f) {
        setter = std::move(f);
        return *this;
    }

    void finalize_default() override {
        if (bound)
            *bound = value;
        if (setter)
            setter(value);
    }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(Location<bool> loc) { bound = loc.ptr; }
    void apply_arg(std::function<void(bool const &)> f) { setter = std::move(f); }
    void apply_arg(Setter<bool> s) { setter = s.fn; }
    void apply_arg(DefaultTag<bool> d) {
        value = d.v;
        if (bound)
            *bound = value;
    }
    void apply_arg(ImplicitValueTag<bool> d) {
        implicit_on           = d.v;
        has_implicit_override = true;
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Flag");
    }

  public:
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        bool tmp;
        if (!val.has_value()) {
            tmp = has_implicit_override ? implicit_on : true; // presence => true
        } else if (!parse_value(*val, tmp, error)) {
            return false;
        }
        value = tmp;
        if (bound)
            *bound = value;
        if (setter)
            setter(value);
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help);
    }

    bool get() const { return bound ? *bound : value; }
};

// -------------------------- Opt<T> -------------------------------------- //

template <typename T>
struct Opt : OptionBase {
    T                              value{};
    T                             *bound       = nullptr;
    bool                           has_default = false;
    std::optional<Range>           range;
    std::optional<T>               implicit_value;
    std::string                    value_name = "value";
    std::function<void(T const &)> setter;

    // With positional default value
    template <class... Args>
    Opt(StringRef longName, std::initializer_list<char> shorts, T defaultValue, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(defaultValue) {
        has_default    = true;
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Without positional default value
    template <class... Args>
    Opt(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Fluent
    Opt &Implicit(T v) {
        implicit_value = std::move(v);
        return *this;
    }
    Opt &ValueName(StringRef n) {
        value_name = std::string(n.s);
        return *this;
    }
    Opt &OnSet(std::function<void(T const &)> f) {
        setter = std::move(f);
        return *this;
    }

    void finalize_default() override {
        if (bound)
            *bound = value;
        if (setter)
            setter(value);
    }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(ValueExpected ve) { value_expected = ve; }
    void apply_arg(Range r) { range = r; }
    void apply_arg(Location<T> loc) { bound = loc.ptr; }
    void apply_arg(std::function<void(T const &)> f) { setter = std::move(f); }
    void apply_arg(Setter<T> s) { setter = s.fn; }
    void apply_arg(ValueNameTag t) { value_name = std::move(t.name); }
    template <class U>
    void apply_arg(DefaultTag<U> d) {
        static_assert(std::is_same_v<std::decay_t<U>, T>, "Default(value) type must match Opt<T>");
        value       = d.v;
        has_default = true;
    }
    template <class U>
    void apply_arg(ImplicitValueTag<U> d) {
        static_assert(std::is_same_v<std::decay_t<U>, T>, "ImplicitValue(value) type must match Opt<T>");
        implicit_value = d.v;
    }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Opt<T> constructor");
    }

  public:
    template <typename U = T>
    bool assign_checked(T const &tmp, std::string &error, bool from_config) {
        if constexpr (std::is_arithmetic_v<U>) {
            if (range.has_value()) {
                long long vll = static_cast<long long>(tmp);
                if (vll < range->min_v || vll > range->max_v) {
                    error = fmt::format("value for '--{}' out of range [{}, {}]", long_name, range->min_v, range->max_v);
                    return false;
                }
            }
        }
        value = tmp;
        if (bound)
            *bound = value;
        if (setter)
            setter(value);
        return true;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            if (value_expected == ValueExpected::ValueRequired) {
                if (implicit_value.has_value()) {
                    if (!assign_checked(*implicit_value, error, from_config))
                        return false;
                } else {
                    error = fmt::format("option '--{}' requires a value", long_name);
                    return false;
                }
            }
            if (from_config) {
                seen_config = true;
                return true;
            }
            seen_cli = true;
            ++occurrences;
            return true;
        }
        T tmp{};
        if (!parse_value(*val, tmp, error))
            return false;
        if (!assign_checked(tmp, error, from_config))
            return false;
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        std::string def;
        if (has_default && !bound)
            def = fmt::format(" (default: {})", value);
        fmt::print("  {:<{}}  {:<{}}  {}{}\n", fmt::format("--{} <{}>", long_name, value_name), pad_long, shorts, pad_short, help, def);
    }

    T const &get() const { return bound ? *bound : value; }
};

// -------------------------- List<T> ------------------------------------- //

template <typename T>
struct List : OptionBase {
    std::vector<T> vals;

    // Named list: --include a --include b  OR  --include=a,b
    template <class... Args>
    List(StringRef longName, std::initializer_list<char> shorts, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Positional list (captures remaining tokens)
    List(StringRef positional_name, Positional, StringRef helpText) : OptionBase(positional_name, Positional{}, helpText) {
        is_positional  = true;
        value_expected = ValueExpected::ValueRequired;
    }

    static bool classof(OptionBase const *) { return true; }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to List");
    }

  public:
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            error = fmt::format("option '--{}' requires a value", long_name);
            return false;
        }
        std::string_view s     = *val;
        size_t           start = 0;
        while (start <= s.size()) {
            size_t           comma = s.find(',', start);
            std::string_view item  = (comma == std::string_view::npos) ? s.substr(start) : s.substr(start, comma - start);
            if (!item.empty()) {
                T tmp{};
                if (!parse_value(item, tmp, error))
                    return false;
                vals.push_back(std::move(tmp));
            }
            if (comma == std::string_view::npos)
                break;
            start = comma + 1;
        }
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        fmt::print("  {:<{}}  {:<{}}  {}\n", fmt::format("--{} <v1,v2,...>", long_name), pad_long, shorts, pad_short, help);
    }

    std::vector<T> const &values() const { return vals; }
};

// -------------------------- OptEnum ------------------------------------- //

template <typename Enum>
struct OptEnum : OptionBase {
    Enum                                     value{};
    Enum                                    *bound       = nullptr;
    bool                                     has_default = false;
    std::map<std::string, Enum, std::less<>> mapping;
    std::function<void(Enum const &, bool)>  setter;

    template <class... Args>
    OptEnum(StringRef longName, std::initializer_list<char> shorts, Enum defaultValue,
            std::initializer_list<std::pair<std::string, Enum>> map, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(defaultValue), has_default(true), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    template <class... Args>
    OptEnum(StringRef longName, std::initializer_list<char> shorts, std::initializer_list<std::pair<std::string, Enum>> map,
            StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), mapping(map) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(Location<Enum> loc) { bound = loc.ptr; }
    void apply_arg(std::function<void(Enum const &, bool)> f) { setter = std::move(f); }
    void apply_arg(Setter<Enum> s) { setter = s.fn; }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to OptEnum");
    }

  public:
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        if (!val.has_value()) {
            error = fmt::format("option '--{}' requires a value", long_name);
            return false;
        }
        auto it = mapping.find(std::string(*val));
        if (it == mapping.end()) {
            std::string keys;
            size_t      i = 0;
            for (auto &kv : mapping) {
                keys += kv.first;
                if (++i < mapping.size())
                    keys += ", ";
            }
            error = fmt::format("invalid value '{}' for '--{}' (choices: {})", *val, long_name, keys);
            return false;
        }
        Enum newv = it->second;
        if (bound)
            *bound = newv;
        else
            value = newv;
        if (setter)
            setter(bound ? *bound : value, from_config);
        if (from_config) {
            seen_config = true;
            return true;
        }
        seen_cli = true;
        ++occurrences;
        return true;
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        std::string keys;
        size_t      i = 0;
        for (auto &kv : mapping) {
            keys += kv.first;
            if (++i < mapping.size())
                keys += "|";
        }
        fmt::print("  {:<{}}  {:<{}}  {} (one of: {})\n", fmt::format("--{} <{}>", long_name, keys), pad_long, shorts, pad_short, help,
                   keys);
    }

    Enum const &get() const { return bound ? *bound : value; }
    std::string to_string() const {
        for (auto &kv : mapping)
            if ((bound ? *bound : value) == kv.second)
                return kv.first;
        return {};
    }
};

// -------------------------- Alias --------------------------------------- //

struct Alias : OptionBase {
    OptionBase                *target = nullptr;
    std::optional<std::string> preset_value;

    template <class... Args>
    Alias(StringRef longName, std::initializer_list<char> shorts, OptionBase &tgt, StringRef helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), target(&tgt) {
        (apply_arg(std::forward<Args>(args)), ...);
    }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(std::string v) { preset_value = std::move(v); }
    template <class U>
    void apply_arg(U &&) {
        static_assert(sizeof(U) == 0, "Unsupported argument to Alias");
    }

  public:
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, bool from_config = false) override {
        seen_cli                          = !from_config;
        seen_config                       = from_config;
        std::optional<std::string_view> v = preset_value ? std::optional{std::string_view(*preset_value)} : val;
        return target->parse_token(target->long_name, v, error, from_config);
    }

    void print_help_line(std::string_view, size_t pad_long, size_t pad_short) const override {
        if (visibility == Visibility::Hidden)
            return;
        std::string shorts;
        for (char c : short_names)
            shorts += fmt::format("-{}, ", c);
        if (!shorts.empty())
            shorts.erase(shorts.end() - 2, shorts.end());
        fmt::print("  {:<{}}  {:<{}}  {} (alias for --{})\n", fmt::format("--{}", long_name), pad_long, shorts, pad_short, help,
                   target ? target->long_name : "?");
    }
};

// -------------------------- Built-ins ----------------------------------- //

struct Builtins {
    OptionCategory cat{"Help"};
    Flag           help{"help", {'h'}, "Show this help message and exit", cat};
    Flag           version{"version", {}, "Show version and exit", cat};
    Builtins() {
        help.value_expected    = ValueExpected::ValueDisallowed;
        version.value_expected = ValueExpected::ValueDisallowed;
    }
};
inline Builtins &builtins() {
    static Builtins B;
    return B;
}

// -------------------------- Config reader -------------------------------- //

inline std::map<std::string, std::string, std::less<>> read_config(std::string_view path) {
    std::map<std::string, std::string, std::less<>> kv;
    if (path.empty())
        return kv;
    FILE *f = fopen(std::string(path).c_str(), "rb");
    if (!f)
        return kv;
    std::string buf;
    char        tmp[4096];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0)
        buf.append(tmp, tmp + n);
    fclose(f);

    auto trim = [](std::string &s) {
        size_t a = 0;
        while (a < s.size() && std::isspace((unsigned char)s[a]))
            ++a;
        size_t b = s.size();
        while (b > a && std::isspace((unsigned char)s[b - 1]))
            --b;
        s = s.substr(a, b - a);
    };
    auto lower = [](std::string s) {
        for (auto &c : s)
            c = (char)std::tolower((unsigned char)c);
        return s;
    };
    auto looks_json = [](std::string const &s) {
        for (char c : s) {
            if (!std::isspace((unsigned char)c))
                return c == '{';
        }
        return false;
    };

    if (!looks_json(buf)) {
        size_t start = 0;
        while (start <= buf.size()) {
            size_t      end  = buf.find_first_of("\r\n", start);
            std::string line = (end == std::string::npos) ? buf.substr(start) : buf.substr(start, end - start);
            start            = (end == std::string::npos) ? buf.size() + 1 : end + 1;
            if (line.empty() || line[0] == '#')
                continue;
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            trim(k);
            trim(v);
            kv[lower(k)] = v;
        }
        return kv;
    }

    // Minimal flat JSON object: { "k": value }
    size_t i    = 0;
    auto   s    = buf;
    auto   skip = [&] {
        while (i < s.size() && std::isspace((unsigned char)s[i]))
            ++i;
    };
    i = 0;
    skip();
    if (i >= s.size() || s[i] != '{')
        return kv;
    ++i;
    while (true) {
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            break;
        }
        if (i >= s.size() || s[i] != '\"')
            break;
        ++i;
        std::string key;
        while (i < s.size() && s[i] != '\"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                key.push_back(s[i + 1]);
                i += 2;
            } else {
                key.push_back(s[i++]);
            }
        }
        if (i < s.size())
            ++i;
        skip();
        if (i >= s.size() || s[i] != ':')
            break;
        ++i;
        skip();
        std::string val;
        if (i < s.size() && s[i] == '\"') {
            ++i;
            while (i < s.size() && s[i] != '\"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    val.push_back(s[i + 1]);
                    i += 2;
                } else {
                    val.push_back(s[i++]);
                }
            }
            if (i < s.size())
                ++i;
        } else if (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+')) {
            size_t j = i;
            while (j < s.size() &&
                   (std::isdigit((unsigned char)s[j]) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' || s[j] == '+' || s[j] == '-'))
                ++j;
            val = s.substr(i, j - i);
            i   = j;
        } else if (s.compare(i, 4, "true") == 0) {
            val = "true";
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            val = "false";
            i += 5;
        } else {
            while (i < s.size() && s[i] != ',' && s[i] != '}')
                ++i;
        }
        std::string lk;
        for (char c : key)
            lk.push_back((char)std::tolower((unsigned char)c));
        kv[lk] = val;
        skip();
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            break;
        }
    }
    return kv;
}

// -------------------------- Help / Version ------------------------------- //

inline void print_version(std::string_view prog, std::string_view ver) {
    if (!ver.empty())
        fmt::print("{} {}\n", prog, ver);
}

namespace detail {
inline OptionBase *find_long(std::string_view name) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional && o->long_name == name)
            return o;
    return nullptr;
}
inline OptionBase *find_short(char c) {
    for (auto *o : Registry::instance().options)
        if (!o->is_positional)
            for (char s : o->short_names)
                if (s == c)
                    return o;
    return nullptr;
}
inline std::vector<OptionBase *> positional_options() {
    std::vector<OptionBase *> v;
    for (auto *o : Registry::instance().options)
        if (o->is_positional)
            v.push_back(o);
    return v;
}
} // namespace detail

inline void print_help(std::string_view prog) {
    auto &R = Registry::instance();

    size_t pad_long = 0, pad_short = 0;
    for (auto *o : R.options)
        if (!o->is_positional && o->visibility == Visibility::Normal) {
            pad_long = std::max(pad_long, std::string("--" + o->long_name).size());
            std::string shorts;
            for (char c : o->short_names)
                shorts += fmt::format("-{}, ", c);
            if (!shorts.empty())
                shorts.erase(shorts.end() - 2, shorts.end());
            pad_short = std::max(pad_short, shorts.size());
        }

    fmt::print("Usage: {} [options]", prog);
    auto pos = detail::positional_options();
    for (auto *p : pos)
        fmt::print(" <{}>", p->long_name);
    fmt::print("\n\n");

    std::map<std::string, std::vector<OptionBase *>> groups;
    for (auto *o : R.options)
        if (!o->is_positional)
            groups[o->category ? o->category->name : std::string{}].push_back(o);

    for (auto &[cat, opts] : groups) {
        if (!cat.empty())
            fmt::print("{}:\n", cat);
        for (auto *o : opts)
            o->print_help_line(prog, pad_long + 2, pad_short + 2);
        fmt::print("\n");
    }

    if (!pos.empty()) {
        fmt::print("Positional arguments:\n");
        for (auto *p : pos)
            p->print_help_line(prog, pad_long + 2, 0);
        fmt::print("\n");
    }
}

// -------------------------- Parser (no subcommands) ---------------------- //

inline ParseResult parse_internal(std::vector<std::string> const &args, char const *programName, std::string_view version,
                                  std::map<std::string, std::string, std::less<>> *config,
                                  std::vector<std::string>                        *unknown_args = nullptr) {
    Builtins                 _;
    GlobalConfigMapLockScope __;
    std::string              prog = programName ? programName : (!args.empty() ? args[0] : "Einsums");

    for (auto *o : Registry::instance().options) {
        o->finalize_default();
    }

    // Apply config first (defaults < config < CLI)
    if (config && !config->empty()) {
        for (auto *o : Registry::instance().options) {
            if (o->is_positional)
                continue;
            auto it = config->find(o->long_name);
            if (it == config->end())
                continue;
            std::string                     err;
            std::optional<std::string_view> v;
            if (!it->second.empty())
                v = std::string_view(it->second);
            if (!o->parse_token(o->long_name, v, err, /*from_config=*/true)) {
                fmt::print(stderr, "config error for '{}': {}\n", o->long_name, err);
                return {false, 1};
            }
        }
    }

    auto looks_like_option_token = [](std::string_view sv) -> bool {
        if (sv.size() >= 1 && sv[0] == '-') {
            // Treat numeric-looking tokens like "-5" or "-3.14" as values, not options
            if (sv.size() >= 2 && std::isdigit(static_cast<unsigned char>(sv[1])))
                return false;
            return true;
        }
        return false;
    };

    size_t pos_index = 0;

    auto consume_positional = [&](std::string_view token, std::string &err) -> bool {
        auto pos = detail::positional_options();
        if (pos_index >= pos.size()) {
            // No positional to consume -> treat as unknown (per your policy)
            if (unknown_args)
                unknown_args->push_back(std::string(token));
            err.clear();
            return true;
        }

        OptionBase *p  = pos[pos_index];
        bool        ok = p->parse_token(p->long_name, token, err);
        if (!ok)
            return false;

        p->seen_cli = true;
        ++p->occurrences;

        // Stay on the same positional if it's a List<std::string>
        // so it can keep capturing subsequent tokens.
        if (dyn_cast<List<std::string> *>(p) == nullptr) {
            ++pos_index;
        }

        return true;
    };

    // Parse CLI
    for (size_t i = 1; i < args.size(); ++i) {
        std::string_view tok(args[i]);

        // Everything after "--" -> unknown_args
        if (tok == "--") {
            while (++i < args.size()) {
                if (unknown_args)
                    unknown_args->push_back(args[i]);
            }
            break;
        }

        // Long options: --name or --name=value
        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-') {
            auto             eq   = tok.find('=');
            std::string_view name = tok.substr(2, eq == std::string_view::npos ? tok.size() - 2 : eq - 2);
            OptionBase      *o    = detail::find_long(name);
            if (!o) {
                if (unknown_args)
                    unknown_args->push_back(std::string(tok));
                continue;
            }

            std::optional<std::string_view> val;
            if (eq != std::string_view::npos) {
                val = tok.substr(eq + 1);
            } else if (o->value_expected == ValueExpected::ValueRequired) {
                // Look ahead; only consume if it doesn't look like another option
                if (i + 1 < args.size()) {
                    std::string_view next = args[i + 1];
                    if (!looks_like_option_token(next)) {
                        val = std::string_view(args[++i]); // consume as value
                    } // else leave val = nullopt to allow ImplicitValue(...)
                } // else leave val = nullopt
            }

            std::string err;
            if (!o->parse_token(name, val, err)) {
                fmt::print(stderr, "error: {}\n", err);
                return {false, 1};
            }
            if (o->on_seen)
                o->on_seen();
            if (o->long_name == "help") {
                print_help(prog);
                return {false, 0};
            }
            if (o->long_name == "version") {
                print_version(prog, version);
                return {false, 0};
            }
            continue;
        }

        // Short options (possibly bundled): -abc, -o value, -ovalue
        if (tok.size() >= 2 && tok[0] == '-') {
            for (size_t j = 1; j < tok.size(); ++j) {
                char        c = tok[j];
                OptionBase *o = detail::find_short(c);
                if (!o) {
                    if (unknown_args)
                        unknown_args->push_back(fmt::format("-{}", c));
                    continue;
                }

                std::optional<std::string_view> val;
                bool                            last_in_bundle = (j + 1 == tok.size());
                if (o->value_expected == ValueExpected::ValueRequired) {
                    if (!last_in_bundle) {
                        // remainder of bundle is the value: -ovalue
                        val = tok.substr(j + 1);
                        j   = tok.size();
                    } else {
                        // last in bundle; optionally consume next token if it's a value
                        if (i + 1 < args.size()) {
                            std::string_view next = args[i + 1];
                            if (!looks_like_option_token(next)) {
                                val = std::string_view(args[++i]); // consume as value
                            } // else leave nullopt to allow ImplicitValue(...)
                        } // else leave nullopt
                    }
                }

                std::string err;
                if (!o->parse_token(std::string_view(&c, 1), val, err)) {
                    fmt::print(stderr, "error: {}\n", err);
                    return {false, 1};
                }
                if (o->on_seen)
                    o->on_seen();
                if (o->long_name == "help") {
                    print_help(prog);
                    return {false, 0};
                }
                if (o->long_name == "version") {
                    print_version(prog, version);
                    return {false, 0};
                }
            }
            continue;
        }

        // Bare token -> positional or unknown
        std::string err;
        if (!consume_positional(tok, err)) {
            fmt::print(stderr, "error: {}\n", err);
            return {false, 1};
        }
    }

    // Validate required/occurrence
    for (auto *o : Registry::instance().options) {
        if ((o->occurrence == Occurrence::Required || o->occurrence == Occurrence::OneOrMore) && o->occurrences == 0) {
            fmt::print(stderr, "error: missing required option '--{}'\n", o->long_name);
            return {false, 1};
        }
        std::string err;
        if (!o->validate(err)) {
            fmt::print(stderr, "error: {}\n", err);
            return {false, 1};
        }
    }

    return {true, 0};
}

/**
 * Parses command-line arguments storing their presence into previously registered Opt/Flag/OpenEnum option.
 *
 * @param args command-line arguments converted to a std::vector<std::string>
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse(std::vector<std::string> const &args, char const *programName = nullptr, std::string_view version = {},
                         std::vector<std::string> *unknown_args = nullptr) {
    return parse_internal(args, programName, version, nullptr, unknown_args);
}

/**
 * Parses command-line arguments storing their presence into previously registered Opt/Flag/OpenEnum option.
 *
 * @param args command-line arguments converted to a std::vector<std::string>
 * @param programName the program name to display in help printing
 * @param version the program version to display in version printing
 * @param config_path key=value or simple json config file that you want to be read in before command line processing
 * @param unknown_args arguments not understood by our parser are placed here
 * @return if ParseResult.ok is true then parsing completed successfully
 */
inline ParseResult parse_with_config(std::vector<std::string> const &args, char const *programName = nullptr, std::string_view version = {},
                                     std::string_view config_path = {}, std::vector<std::string> *unknown_args = nullptr) {
    auto kv = read_config(config_path);
    return parse_internal(args, programName, version, &kv, unknown_args);
}

} // namespace einsums::cl
