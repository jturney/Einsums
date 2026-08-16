//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Source.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <charconv>
#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/*
 * The declaration side of the option system: the option types themselves, the
 * named-argument tags they are built from, and the registry they add
 * themselves to. Only the handful of translation units that DECLARE options
 * need this header; a translation unit that merely reads a value wants
 * Options/Get.hpp instead, which is deliberately free of fmt and of everything
 * below.
 */

EINSUMS_NAMESPACE_BEGIN(cl)

// -------------------------- Value parsing --------------------------------- //

/*
 * parse_value is an ADL customization point. The overloads below cover strings,
 * bools, and every integral and floating point type; a user-defined T becomes
 * usable with Opt<T>/List<T> simply by declaring
 *
 *     bool parse_value(std::string_view, T &out, std::string &err);
 *
 * in T's own namespace, where argument-dependent lookup will find it.
 */

namespace detail {
/// Error text builders, out of line so the parse_value templates do not drag
/// fmt into every declaration site's instantiation.
EINSUMS_EXPORT std::string invalid_bool_message(std::string_view sv);
EINSUMS_EXPORT std::string invalid_integer_message(std::string_view sv);
EINSUMS_EXPORT std::string invalid_real_message(std::string_view sv);
EINSUMS_EXPORT std::string invalid_real_message(std::string_view sv, std::string_view what);
EINSUMS_EXPORT std::string requires_value_message(std::string_view long_name);
EINSUMS_EXPORT std::string out_of_range_int_message(std::string_view long_name, long long lo, long long hi);
EINSUMS_EXPORT std::string out_of_range_real_message(std::string_view long_name, double lo, double hi);
EINSUMS_EXPORT std::string invalid_choice_message(std::string_view value, std::string_view long_name, std::string_view choices);
EINSUMS_EXPORT std::string long_invocation(std::string_view long_name);
EINSUMS_EXPORT std::string value_invocation(std::string_view long_name, std::string_view value_name);
EINSUMS_EXPORT std::string list_invocation(std::string_view long_name, bool positional);
EINSUMS_EXPORT std::string alias_annotation(std::string_view target_name);
} // namespace detail

EINSUMS_EXPORT bool parse_value(std::string_view sv, std::string &out, std::string &err);
EINSUMS_EXPORT bool parse_value(std::string_view sv, bool &out, std::string &err);

template <std::integral T>
    requires(!std::same_as<T, bool>)
bool parse_value(std::string_view sv, T &out, std::string &err) {
    T          v{};
    auto const first   = sv.data();
    auto const last    = first + sv.size();
    auto const [p, ec] = std::from_chars(first, last, v);
    if (ec == std::errc{} && p == last) {
        out = v;
        return true;
    }
    err = detail::invalid_integer_message(sv);
    return false;
}

template <std::floating_point T>
bool parse_value(std::string_view sv, T &out, std::string &err) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    // from_chars is locale-independent and non-throwing. It is stricter than
    // stod in that it rejects a leading '+' and leading whitespace, which is
    // the behaviour we want out of a command line.
    T          v{};
    auto const first   = sv.data();
    auto const last    = first + sv.size();
    auto const [p, ec] = std::from_chars(first, last, v);
    if (ec == std::errc{} && p == last) {
        out = v;
        return true;
    }
#else
    // Floating point from_chars is not universally available; fall back to the
    // throwing conversion and normalize it to the same bool contract.
    try {
        std::string const tmp(sv);
        std::size_t       pos = 0;
        long double const v   = std::stold(tmp, &pos);
        if (pos == tmp.size()) {
            out = static_cast<T>(v);
            return true;
        }
    } catch (std::exception const &e) {
        err = detail::invalid_real_message(sv, e.what());
        return false;
    }
#endif
    err = detail::invalid_real_message(sv);
    return false;
}

/**
 * @brief Types usable as the value type of Opt<T>, List<T>, and friends.
 *
 * Satisfied by anything for which a `parse_value(std::string_view, T&,
 * std::string&) -> bool` overload is visible, whether one of the built-in
 * overloads or one found by ADL in the user's own namespace. Constraining on
 * this turns what used to be an undefined-symbol link error into a readable
 * diagnostic at the point of declaration.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
concept Parsable = std::default_initializable<T> && requires(std::string_view sv, T &out, std::string &err) {
    { parse_value(sv, out, err) } -> std::same_as<bool>;
};

// Forward decls
struct OptionBase;
struct OptionCategory;
struct ExclusiveCategory;

// -------------------------- Registry -------------------------------------- //

/**
 * @brief Process-wide list of every declared option.
 *
 * Options add themselves on construction and remove themselves on destruction,
 * so an option with automatic storage duration is safe to declare inside a
 * scope (a unit test, say) without leaving a dangling entry behind.
 *
 * @warning Neither the registry nor the option state it points at is
 *          synchronized. Declare options and call @ref parse from one thread.
 */
struct EINSUMS_EXPORT Registry {
    std::vector<OptionBase *>        options;
    std::vector<OptionCategory *>    categories;
    std::vector<ExclusiveCategory *> exclusions;

    static Registry &instance();

    void add_option(OptionBase *o);
    void add_category(OptionCategory *c);
    void add_exclusion(ExclusiveCategory *c);

    /// Register only if absent. The built-ins are a singleton, so they are
    /// constructed once but may need re-adding after clear_for_tests().
    void ensure_option(OptionBase *o);
    void ensure_category(OptionCategory *c);

    void remove_option(OptionBase *o);
    void remove_category(OptionCategory *c);
    void remove_exclusion(ExclusiveCategory *c);

    /**
     * @brief Turn on automatic environment variable names for every option.
     *
     * With a prefix set, an option that names no variable of its own derives
     * one from its long name: separators become underscores, letters are
     * upper-cased, and the prefix is prepended unless the name already carries
     * it. So with prefix `EINSUMS`, `einsums:log:level` reads
     * `EINSUMS_LOG_LEVEL` and a bare `buffer-size` reads
     * `EINSUMS_BUFFER_SIZE`.
     *
     * An empty prefix (the default) disables derivation, leaving only options
     * annotated with @ref Env environment-sensitive.
     *
     * @param prefix The variable-name prefix, without a trailing underscore.
     */
    void set_env_prefix(std::string prefix);

    [[nodiscard]] std::string const &get_env_prefix() const noexcept;

    void clear_for_tests();

  private:
    std::string _env_prefix;
};

/**
 * @brief Derive an environment variable name from a prefix and an option name.
 *
 * Separators (`:`, `-`, `.`, whitespace) become underscores, letters are
 * upper-cased, and anything else is dropped. The prefix is prepended unless
 * the normalized name already starts with it, so an option that is already
 * namespaced does not end up with the prefix twice.
 *
 * Returns an empty string when @p prefix is empty.
 */
EINSUMS_EXPORT std::string derive_env_name(std::string_view prefix, std::string_view long_name);

// -------------------------- Categories ------------------------------------ //

/**
 * @brief A heading that groups related options in `--help`.
 *
 * An option joins a category by naming one in its constructor; options with
 * no category print before the first heading.
 */
struct EINSUMS_EXPORT OptionCategory {
    std::string name;

    explicit OptionCategory(std::string_view n);

    OptionCategory(OptionCategory const &)            = delete;
    OptionCategory &operator=(OptionCategory const &) = delete;

    ~OptionCategory();
};

// -------------------------- Location & Setter ----------------------------- //

/// Binds an option to storage that outlives parsing.
template <typename T>
struct Location {
    T *ptr = nullptr;
    explicit Location(T &r) : ptr(&r) {}
};

/**
 * @brief A callback invoked every time an option receives a value.
 *
 * The callable may take either `(T const &)` or `(T const &, Source)`; the
 * two-argument form additionally learns whether the value arrived from the
 * default, a config file, the environment, or the command line.
 */
template <typename T>
struct Setter {
    std::function<void(T const &, Source)> fn;

    Setter() = default;

    template <typename F>
        requires std::invocable<F, T const &, Source>
    explicit Setter(F &&f) : fn(std::forward<F>(f)) {}

    template <typename F>
        requires(std::invocable<F, T const &> && !std::invocable<F, T const &, Source>)
    explicit Setter(F &&f) : fn([g = std::forward<F>(f)](T const &v, Source) { g(v); }) {}
};

/// Anything acceptable as a Setter<T> callback.
template <typename F, typename T>
concept SetterCallable = std::invocable<F, T const &> || std::invocable<F, T const &, Source>;

// -------------------------- Named-arg tags -------------------------------- //

template <typename T>
struct DefaultTag {
    T v;
};

template <typename T>
struct ImplicitValueTag {
    T v;
};

template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
auto Default(T &&v) -> DefaultTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
auto ImplicitValue(T &&v) -> ImplicitValueTag<std::decay_t<T>> {
    return {std::forward<T>(v)};
}

struct ValueNameTag {
    std::string name;
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline ValueNameTag ValueName(std::string n) {
    return {std::move(n)};
}

/// The environment variable name carried by @ref Env.
struct EnvTag {
    std::string name;
};

/**
 * @brief Read this option from a named environment variable.
 *
 * Overrides any name the registry's prefix policy would have derived. Use it
 * for variables whose spelling predates the convention.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
inline EnvTag Env(std::string n) {
    return {std::move(n)};
}

/// Exempt this option from the registry's automatic environment name derivation.
struct NoEnvTag {};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr NoEnvTag NoEnv{};

// -------------------------- Help rendering -------------------------------- //

/**
 * @brief One option's contribution to `--help`, before layout.
 *
 * Options describe themselves; @ref print_help owns column widths and wrapping,
 * so every option type lines up without repeating the formatting logic.
 */
struct HelpEntry {
    std::string invocation;  ///< e.g. `--threads <N>`
    std::string shorts;      ///< e.g. `-t, -T`
    std::string description; ///< The help text.
    /// Trailing notes, e.g. `(default: 4)` and `[env: EINSUMS_THREADS]`. Each
    /// is laid out as a unit so a variable name never breaks across lines.
    std::vector<std::string> annotations;
};

// -------------------------- OptionBase ------------------------------------ //

/**
 * @brief The behaviour every option shares, whatever its value type.
 *
 * Holds the names, help text, and grouping an option is declared with, plus
 * the state a parse leaves behind: how many times it appeared and which
 * @ref Source supplied the value now in effect. Derived types add the value
 * itself and say how a token becomes one.
 *
 * Construction registers the option with the process-wide @ref Registry and
 * destruction removes it again, so an option declared in a scope does not
 * outlive its entry.
 */
struct EINSUMS_EXPORT OptionBase {
    std::string        long_name;   // "--long"
    std::vector<char>  short_names; // {'v'}
    std::string        help;
    OptionCategory    *category       = nullptr;
    ExclusiveCategory *exclusions     = nullptr;
    Visibility         visibility     = Visibility::Normal;
    Occurrence         occurrence     = Occurrence::Optional;
    ValueExpected      value_expected = ValueExpected::ValueOptional;
    bool               is_positional  = false;
    Source             value_source   = Source::None;
    int                occurrences    = 0;

    /// Explicit environment variable name from Env(...); empty if unset.
    std::string env_name;
    /// True when NoEnv was passed, which suppresses derivation from the prefix.
    bool env_opt_out = false;

    std::function<void()> on_seen;

    OptionBase(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, OptionCategory *cat);

    OptionBase(std::string_view positional_name, Positional, std::string_view helpText);

    OptionBase(OptionBase const &)            = delete;
    OptionBase &operator=(OptionBase const &) = delete;

    virtual ~OptionBase();

    virtual bool parse_token(std::string_view key, std::optional<std::string_view> val, std::string &error,
                             Source src = Source::CommandLine) = 0;

    [[nodiscard]] virtual HelpEntry help_entry() const = 0;

    [[nodiscard]] virtual bool validate(std::string &error) const;

    virtual void finalize_default();

    /**
     * @brief Forget everything a previous parse recorded.
     *
     * Options are long-lived (usually function-local statics), so a second
     * @ref parse in the same process must not inherit the first one's
     * occurrence counts, value source, or accumulated list items.
     */
    virtual void reset();

    /// True when something other than the programmer's default supplied the value.
    [[nodiscard]] bool was_specified() const noexcept { return value_source > Source::Default; }

    /**
     * @brief The environment variable this option reads, or empty for none.
     *
     * Resolved lazily rather than at construction because the registry's
     * prefix policy is usually installed after the options themselves.
     */
    [[nodiscard]] virtual std::string effective_env_name() const;

  protected:
    /// Common bookkeeping every parse_token override performs on success.
    bool record_source(Source src);

    /// `-a, -b` for the help table.
    [[nodiscard]] std::string format_shorts() const;

    /// The `(default: ...)` / `[env: ...]` notes shared by every option type.
    [[nodiscard]] std::vector<std::string> format_annotations(std::string const &default_text) const;
};

/**
 * @brief A set of options of which at most one may be given.
 *
 * Membership is declared by passing the category to an option's constructor.
 * Conflicts are judged per precedence level, so two options set from the same
 * @ref Source contradict one another while one from the environment losing to
 * one from the command line does not.
 */
struct EINSUMS_EXPORT ExclusiveCategory {
    std::vector<OptionBase *> options;

    ExclusiveCategory();

    ExclusiveCategory(ExclusiveCategory const &)            = delete;
    ExclusiveCategory &operator=(ExclusiveCategory const &) = delete;

    ~ExclusiveCategory();

    /**
     * @brief Options in this group that carry an explicitly supplied value.
     *
     * @param src Restrict to options whose value came from this source.
     */
    [[nodiscard]] std::vector<OptionBase *> found_options(Source src) const;

    /**
     * @brief The source at which two options in this group conflict, if any.
     *
     * Only options resolved at the *same* precedence level conflict. Two
     * mutually exclusive flags both named on the command line is a genuine
     * contradiction; one set in the environment and the other on the command
     * line is not, because the command line simply outranks the environment.
     */
    [[nodiscard]] std::optional<Source> conflicting_source() const;
};

// -------------------------- Flag ------------------------------------------ //

/**
 * @brief A boolean option whose meaning is carried by its presence.
 *
 * `--verbose` sets the flag and `--verbose=false` clears it explicitly. An
 * ImplicitValue(false) inverts what presence means, which is how the negated
 * options in this library are spelled: passing
 * `--einsums:debug:no-attach-debugger` sets the bound `attach-debugger` to
 * false.
 *
 * A config file or environment variable has no presence to observe, so its
 * value answers that question instead - see @ref Flag::parse_token.
 */
struct EINSUMS_EXPORT Flag : OptionBase {
    bool                                      value = false;
    bool                                     *bound = nullptr;
    std::function<void(bool const &, Source)> setter;
    bool                                      implicit_on           = true;
    bool                                      has_implicit_override = false;
    bool                                      set_on_unseen         = true;
    /// The declared default, kept apart from `value` so that help text reports
    /// what the option defaults to rather than what it currently holds.
    bool default_value = false;

    template <typename... Args>
    Flag(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueOptional;
        (apply_arg(std::forward<Args>(args)), ...);
        default_value = value;
    }

    template <typename F>
        requires SetterCallable<F, bool>
    // NOLINTNEXTLINE(readability-identifier-naming)
    Flag &OnSet(F &&f) {
        setter = Setter<bool>(std::forward<F>(f)).fn;
        return *this;
    }

    void finalize_default() override;

    /**
     * @brief Apply a flag from any source.
     *
     * On the command line a flag carries its meaning in its presence, so
     * `--verbose` sets the flag and `--verbose=false` clears it explicitly.
     *
     * A config file or environment variable has no notion of presence, so its
     * value answers that question instead: a truthy value applies the flag
     * exactly as if it had been named on the command line, and a falsey one
     * leaves the default in place. That distinction matters for the negated
     * flags this library is full of - `EINSUMS_DEBUG_NO_ATTACH_DEBUGGER=1`
     * has to mean "yes, do not attach", which is the flag's ImplicitValue of
     * false, not the literal true it parsed.
     */
    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, Source src = Source::CommandLine) override;

    [[nodiscard]] HelpEntry help_entry() const override;

    [[nodiscard]] bool get() const;

  private:
    void assign(bool v, Source src);

    void apply_arg(OptionCategory &c);
    void apply_arg(Visibility v);
    void apply_arg(Occurrence o);
    void apply_arg(ValueExpected ve);
    void apply_arg(Location<bool> loc);
    void apply_arg(Setter<bool> const &s);
    void apply_arg(ValueNameTag const &); // flags take no value; accepted and ignored
    void apply_arg(EnvTag t);
    void apply_arg(NoEnvTag);
    void apply_arg(DefaultTag<bool> d);
    void apply_arg(ImplicitValueTag<bool> d);
    void apply_arg(ExclusiveCategory &cat);

    template <typename F>
        requires SetterCallable<F, bool>
    void apply_arg(F &&f) {
        setter = Setter<bool>(std::forward<F>(f)).fn;
    }

    template <typename U>
    void apply_arg(U &&) {
        static_assert(detail::always_false<U>, "Unsupported argument to Flag. Accepted: OptionCategory&, ExclusiveCategory&, Visibility, "
                                               "Occurrence, ValueExpected, Location<bool>, Setter<bool> (or a callable), Default(bool), "
                                               "ImplicitValue(bool), Env(\"NAME\"), NoEnv.");
    }
};

EINSUMS_EXPORT std::shared_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value = false);

// -------------------------- Opt<T> ---------------------------------------- //

template <Parsable T>
struct Opt : OptionBase {
    T value{};
    /// The declared default, so help text is unaffected by parsing order.
    T                                      default_value{};
    T                                     *bound       = nullptr;
    bool                                   has_default = false;
    std::optional<Range>                   range;
    std::optional<T>                       implicit_value;
    std::string                            value_name = "value";
    std::function<void(T const &, Source)> setter;

    // With positional default value
    template <typename... Args>
    Opt(std::string_view longName, std::initializer_list<char> shorts, T defaultValue, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(std::move(defaultValue)) {
        has_default    = true;
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
        default_value = value;
    }

    // Without positional default value
    template <typename... Args>
    Opt(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
        default_value = value;
    }

    // Fluent
    // NOLINTNEXTLINE(readability-identifier-naming)
    Opt &Implicit(T v) {
        implicit_value = std::move(v);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Opt &ValueName(std::string_view n) {
        value_name = std::string(n);
        return *this;
    }

    template <typename F>
        requires SetterCallable<F, T>
    // NOLINTNEXTLINE(readability-identifier-naming)
    Opt &OnSet(F &&f) {
        setter = Setter<T>(std::forward<F>(f)).fn;
        return *this;
    }

    void finalize_default() override {
        assign(default_value, Source::Default);
        value_source = Source::Default;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, Source src = Source::CommandLine) override {
        if (!val.has_value()) {
            if (value_expected == ValueExpected::ValueRequired) {
                if (!implicit_value.has_value()) {
                    error = detail::requires_value_message(long_name);
                    return false;
                }
                if (!assign_checked(*implicit_value, error, src)) {
                    return false;
                }
            }
            return record_source(src);
        }
        T tmp{};
        if (!parse_value(*val, tmp, error)) {
            return false;
        }
        if (!assign_checked(tmp, error, src)) {
            return false;
        }
        return record_source(src);
    }

    bool assign_checked(T const &tmp, std::string &error, Source src) {
        if (range.has_value()) {
            if constexpr (std::integral<T> && !std::same_as<T, bool>) {
                if (tmp < static_cast<T>(range->int_min) || tmp > static_cast<T>(range->int_max)) {
                    error = detail::out_of_range_int_message(long_name, range->int_min, range->int_max);
                    return false;
                }
            } else if constexpr (std::floating_point<T>) {
                auto const v = static_cast<long double>(tmp);
                if (v < range->real_min || v > range->real_max) {
                    error = detail::out_of_range_real_message(long_name, static_cast<double>(range->real_min),
                                                              static_cast<double>(range->real_max));
                    return false;
                }
            }
        }
        assign(tmp, src);
        return true;
    }

    [[nodiscard]] HelpEntry help_entry() const override {
        std::string shown;
        if (has_default) {
            if constexpr (fmt::is_formattable<T>::value) {
                shown = fmt::format("{}", default_value);
            }
        }
        return HelpEntry{.invocation  = detail::value_invocation(long_name, value_name),
                         .shorts      = format_shorts(),
                         .description = help,
                         .annotations = format_annotations(shown)};
    }

    T const &get() const { return bound ? *bound : value; }

  private:
    void assign(T const &v, Source src) {
        value = v;
        if (bound) {
            *bound = v;
        }
        if (setter) {
            setter(v, src);
        }
    }

    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(ValueExpected ve) { value_expected = ve; }
    void apply_arg(Range r) { range = r; }
    void apply_arg(Location<T> loc) { bound = loc.ptr; }
    void apply_arg(Setter<T> const &s) { setter = s.fn; }
    void apply_arg(ValueNameTag t) { value_name = std::move(t.name); }
    void apply_arg(EnvTag t) { env_name = std::move(t.name); }
    void apply_arg(NoEnvTag) { env_opt_out = true; }

    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }

    template <typename U>
        requires(!std::same_as<U, T>)
    void apply_arg(Location<U>) {
        static_assert(detail::always_false<U>, "Location(ref) must refer to storage of the same type as Opt<T>.");
    }

    template <typename U>
    void apply_arg(DefaultTag<U> d) {
        static_assert(std::same_as<std::decay_t<U>, T>, "Default(value) type must match Opt<T>");
        value       = std::move(d.v);
        has_default = true;
    }

    template <typename U>
    void apply_arg(ImplicitValueTag<U> d) {
        static_assert(std::same_as<std::decay_t<U>, T>, "ImplicitValue(value) type must match Opt<T>");
        implicit_value = std::move(d.v);
    }

    template <typename F>
        requires SetterCallable<F, T>
    void apply_arg(F &&f) {
        setter = Setter<T>(std::forward<F>(f)).fn;
    }

    template <typename U>
    void apply_arg(U &&) {
        static_assert(detail::always_false<U>, "Unsupported argument to Opt<T>. Accepted: OptionCategory&, ExclusiveCategory&, Visibility, "
                                               "Occurrence, ValueExpected, Range, Location<T>, Setter<T> (or a callable), Default(T), "
                                               "ImplicitValue(T), ValueName(\"NAME\"), Env(\"NAME\"), NoEnv.");
    }
};

// -------------------------- List<T> --------------------------------------- //

template <Parsable T>
struct List : OptionBase {
    std::vector<T> vals;

    // Named list: --include a --include b  OR  --include=a,b
    template <typename... Args>
    List(std::string_view longName, std::initializer_list<char> shorts, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        (apply_arg(std::forward<Args>(args)), ...);
    }

    // Positional list (captures remaining tokens)
    List(std::string_view positional_name, Positional, std::string_view helpText) : OptionBase(positional_name, Positional{}, helpText) {
        is_positional  = true;
        value_expected = ValueExpected::ValueRequired;
    }

    void reset() override {
        OptionBase::reset();
        vals.clear();
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, Source src = Source::CommandLine) override {
        if (!val.has_value()) {
            error = detail::requires_value_message(long_name);
            return false;
        }
        // A single token may carry several comma-separated items.
        std::string_view const s     = *val;
        std::size_t            start = 0;
        while (start <= s.size()) {
            std::size_t const      comma = s.find(',', start);
            std::string_view const item  = (comma == std::string_view::npos) ? s.substr(start) : s.substr(start, comma - start);
            if (!item.empty()) {
                T tmp{};
                if (!parse_value(item, tmp, error)) {
                    return false;
                }
                vals.push_back(std::move(tmp));
            }
            if (comma == std::string_view::npos) {
                break;
            }
            start = comma + 1;
        }
        return record_source(src);
    }

    [[nodiscard]] HelpEntry help_entry() const override {
        return HelpEntry{.invocation  = detail::list_invocation(long_name, is_positional),
                         .shorts      = format_shorts(),
                         .description = help,
                         .annotations = format_annotations({})};
    }

    [[nodiscard]] std::vector<T> const &values() const { return vals; }

  private:
    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(ValueNameTag const &) {}
    void apply_arg(EnvTag t) { env_name = std::move(t.name); }
    void apply_arg(NoEnvTag) { env_opt_out = true; }

    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }

    template <typename U>
    void apply_arg(U &&) {
        static_assert(detail::always_false<U>, "Unsupported argument to List<T>. Accepted: OptionCategory&, ExclusiveCategory&, "
                                               "Visibility, Occurrence, Env(\"NAME\"), NoEnv.");
    }
};

/*
 * Opt<T> and List<T> are instantiated over a small closed set across the whole
 * tree. Declaring those instantiations extern keeps their bodies out of every
 * registration TU; the definitions live in CommandLine.cpp. A T outside the set
 * still works, it simply instantiates where it is used as before.
 */
extern template struct EINSUMS_EXTERN_TEMPLATE_EXPORT Opt<std::string>;
extern template struct EINSUMS_EXTERN_TEMPLATE_EXPORT Opt<std::int64_t>;
extern template struct EINSUMS_EXTERN_TEMPLATE_EXPORT Opt<double>;
extern template struct EINSUMS_EXTERN_TEMPLATE_EXPORT List<std::string>;

// -------------------------- OptEnum --------------------------------------- //

/**
 * @brief An option whose value is one of a fixed set of named choices.
 *
 * @code
 * enum struct Mode { Fast, Accurate };
 * OptEnum<Mode> mode{"mode", {'m'}, Mode::Fast,
 *                    {{"fast", Mode::Fast}, {"accurate", Mode::Accurate}},
 *                    "Execution mode"};
 * @endcode
 */
template <typename Enum>
struct OptEnum : OptionBase {
    Enum value{};
    /// The declared default, so help text is unaffected by parsing order.
    Enum                                      default_value{};
    Enum                                     *bound       = nullptr;
    bool                                      has_default = false;
    std::map<std::string, Enum, std::less<>>  mapping;
    std::function<void(Enum const &, Source)> setter;

    template <typename... Args>
    OptEnum(std::string_view longName, std::initializer_list<char> shorts, Enum defaultValue,
            std::initializer_list<std::pair<std::string_view, Enum>> choices, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), value(defaultValue), default_value(defaultValue), has_default(true) {
        value_expected = ValueExpected::ValueRequired;
        add_choices(choices);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    template <typename... Args>
    OptEnum(std::string_view longName, std::initializer_list<char> shorts, std::initializer_list<std::pair<std::string_view, Enum>> choices,
            std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr) {
        value_expected = ValueExpected::ValueRequired;
        add_choices(choices);
        (apply_arg(std::forward<Args>(args)), ...);
    }

    template <typename F>
        requires SetterCallable<F, Enum>
    // NOLINTNEXTLINE(readability-identifier-naming)
    OptEnum &OnSet(F &&f) {
        setter = Setter<Enum>(std::forward<F>(f)).fn;
        return *this;
    }

    void finalize_default() override {
        assign(default_value, Source::Default);
        value_source = Source::Default;
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, Source src = Source::CommandLine) override {
        if (!val.has_value()) {
            error = detail::requires_value_message(long_name);
            return false;
        }
        auto const it = mapping.find(*val);
        if (it == mapping.end()) {
            error = detail::invalid_choice_message(*val, long_name, choice_list(", "));
            return false;
        }
        assign(it->second, src);
        return record_source(src);
    }

    [[nodiscard]] HelpEntry help_entry() const override {
        return HelpEntry{.invocation  = detail::value_invocation(long_name, choice_list("|")),
                         .shorts      = format_shorts(),
                         .description = help,
                         .annotations = format_annotations(has_default ? name_of(default_value) : std::string{})};
    }

    Enum const &get() const { return bound ? *bound : value; }

    /// The choice name matching the current value, or empty if none matches.
    [[nodiscard]] std::string to_string() const { return name_of(get()); }

    /// The choice name mapped to @p v, or empty if none matches.
    [[nodiscard]] std::string name_of(Enum v) const {
        for (auto const &[name, choice] : mapping) {
            if (v == choice) {
                return name;
            }
        }
        return {};
    }

  private:
    void add_choices(std::initializer_list<std::pair<std::string_view, Enum>> choices) {
        for (auto const &[name, choice] : choices) {
            mapping.emplace(std::string(name), choice);
        }
    }

    [[nodiscard]] std::string choice_list(std::string_view sep) const {
        std::string out;
        for (auto const &[name, choice] : mapping) {
            if (!out.empty()) {
                out += sep;
            }
            out += name;
        }
        return out;
    }

    void assign(Enum v, Source src) {
        value = v;
        if (bound) {
            *bound = v;
        }
        if (setter) {
            setter(v, src);
        }
    }

    void apply_arg(OptionCategory &c) { category = &c; }
    void apply_arg(Visibility v) { visibility = v; }
    void apply_arg(Occurrence o) { occurrence = o; }
    void apply_arg(Location<Enum> loc) { bound = loc.ptr; }
    void apply_arg(Setter<Enum> const &s) { setter = s.fn; }
    void apply_arg(ValueNameTag const &) {}
    void apply_arg(EnvTag t) { env_name = std::move(t.name); }
    void apply_arg(NoEnvTag) { env_opt_out = true; }

    void apply_arg(ExclusiveCategory &cat) {
        cat.options.push_back(this);
        exclusions = &cat;
    }

    template <typename F>
        requires SetterCallable<F, Enum>
    void apply_arg(F &&f) {
        setter = Setter<Enum>(std::forward<F>(f)).fn;
    }

    template <typename U>
    void apply_arg(U &&) {
        static_assert(detail::always_false<U>,
                      "Unsupported argument to OptEnum<Enum>. Accepted: OptionCategory&, ExclusiveCategory&, "
                      "Visibility, Occurrence, Location<Enum>, Setter<Enum> (or a callable), Env(\"NAME\"), NoEnv.");
    }
};

// -------------------------- Alias ----------------------------------------- //

struct EINSUMS_EXPORT Alias : OptionBase {
    OptionBase                *target = nullptr;
    std::optional<std::string> preset_value;

    template <typename... Args>
    Alias(std::string_view longName, std::initializer_list<char> shorts, OptionBase &tgt, std::string_view helpText, Args &&...args)
        : OptionBase(longName, shorts, helpText, /*cat*/ nullptr), target(&tgt) {
        value_expected = tgt.value_expected;
        (apply_arg(std::forward<Args>(args)), ...);
        if (preset_value.has_value()) {
            // The alias supplies the value itself, so it must not consume the
            // next token on the target's behalf and then discard it.
            value_expected = ValueExpected::ValueDisallowed;
        }
    }

    bool parse_token(std::string_view, std::optional<std::string_view> val, std::string &error, Source src = Source::CommandLine) override;

    [[nodiscard]] HelpEntry help_entry() const override;

    /**
     * @brief Aliases are never read from the environment.
     *
     * An alias carries no value of its own, so a variable could only say "act
     * as though this alias appeared", and any value at all - `=0` included -
     * would trigger it. Set the target's variable instead, where the value
     * means what it says.
     */
    [[nodiscard]] std::string effective_env_name() const override;

  private:
    void apply_arg(OptionCategory &c);
    void apply_arg(Visibility v);
    void apply_arg(Occurrence o);
    void apply_arg(std::string v);
    void apply_arg(std::string_view v);
    void apply_arg(char const *v);

    template <typename U>
    void apply_arg(U &&) {
        static_assert(detail::always_false<U>,
                      "Unsupported argument to Alias. Accepted: OptionCategory&, Visibility, Occurrence, a preset value string.");
    }
};

// -------------------------- Built-ins ------------------------------------- //

/**
 * @brief The always-available `--help` and `--version` options.
 *
 * A singleton, because these register themselves with the process-wide
 * registry and so must outlive every call to @ref parse.
 */
struct EINSUMS_EXPORT Builtins {
    OptionCategory cat{"Help"};
    Flag           help{"help", {'h'}, "Show this help message and exit", cat, NoEnv};
    Flag           version{"version", {}, "Show version and exit", cat, NoEnv};

    Builtins();
};

EINSUMS_EXPORT Builtins &builtins();

EINSUMS_NAMESPACE_END(cl)
