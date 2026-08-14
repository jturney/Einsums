//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#if defined(EINSUMS_WINDOWS)
#    include <io.h>
#    include <windows.h>
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN(cl)

Registry &Registry::instance() {
    static Registry R;
    return R;
}

// -------------------------- Environment names ----------------------------- //

std::string derive_env_name(std::string_view prefix, std::string_view long_name) {
    if (prefix.empty() || long_name.empty()) {
        return {};
    }

    // Normalize the option name: separators become underscores, letters are
    // upper-cased, and anything that cannot appear in a variable name is
    // dropped. Runs of separators collapse so "a--b" does not become "A__B".
    std::string normalized;
    normalized.reserve(prefix.size() + 1 + long_name.size());
    for (char const c : long_name) {
        auto const uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) != 0) {
            normalized.push_back(static_cast<char>(std::toupper(uc)));
        } else if (!normalized.empty() && normalized.back() != '_') {
            normalized.push_back('_');
        }
    }
    while (!normalized.empty() && normalized.back() == '_') {
        normalized.pop_back();
    }
    if (normalized.empty()) {
        return {};
    }

    std::string upper_prefix;
    upper_prefix.reserve(prefix.size());
    for (char const c : prefix) {
        auto const uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) != 0) {
            upper_prefix.push_back(static_cast<char>(std::toupper(uc)));
        } else if (!upper_prefix.empty() && upper_prefix.back() != '_') {
            upper_prefix.push_back('_');
        }
    }
    while (!upper_prefix.empty() && upper_prefix.back() == '_') {
        upper_prefix.pop_back();
    }
    if (upper_prefix.empty()) {
        return normalized;
    }

    // An option whose long name is already namespaced (einsums:log:level) must
    // not pick up the prefix twice.
    if (normalized.starts_with(upper_prefix) && (normalized.size() == upper_prefix.size() || normalized[upper_prefix.size()] == '_')) {
        return normalized;
    }
    return upper_prefix + "_" + normalized;
}

// -------------------------- Yes/no pairs ---------------------------------- //

std::shared_ptr<ExclusiveCategory> make_yes_no(Flag &yes_flag, Flag &no_flag, bool default_value) {
    auto out = std::make_shared<ExclusiveCategory>();

    yes_flag.exclusions = out.get();
    no_flag.exclusions  = out.get();

    out->options.push_back(&yes_flag);
    out->options.push_back(&no_flag);

    // finalize_default() must not clobber the shared binding on the flag that
    // was not named; the default below is applied once, here.
    yes_flag.set_on_unseen = false;
    no_flag.set_on_unseen  = false;

    yes_flag.value = true;
    no_flag.value  = false;

    Flag const &carrier = default_value ? yes_flag : no_flag;
    if (carrier.bound) {
        *carrier.bound = default_value;
    }
    if (carrier.setter) {
        carrier.setter(default_value, Source::Default);
    }

    return out;
}

// -------------------------- Config reader --------------------------------- //

namespace {

void trim(std::string &s) {
    constexpr std::string_view whitespace = " \t\v\f\r\n";
    auto const                 first      = s.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        s.clear();
        return;
    }
    s = s.substr(first, s.find_last_not_of(whitespace) - first + 1);
}

std::string to_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char const c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

bool looks_like_json(std::string_view s) {
    for (char const c : s) {
        if (std::isspace(static_cast<unsigned char>(c)) == 0) {
            return c == '{';
        }
    }
    return false;
}

/// Parse a flat `{"key": value}` object. Nested objects and arrays are skipped.
void parse_flat_json(std::string_view s, std::map<std::string, std::string, std::less<>> &kv) {
    std::size_t i = 0;

    auto skip_space = [&] {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) {
            ++i;
        }
    };
    auto starts_with_at = [&](std::string_view lit) { return s.substr(i).starts_with(lit); };
    auto read_quoted    = [&] {
        std::string out;
        ++i; // opening quote
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                out.push_back(s[i + 1]);
                i += 2;
            } else {
                out.push_back(s[i++]);
            }
        }
        if (i < s.size()) {
            ++i; // closing quote
        }
        return out;
    };

    skip_space();
    if (i >= s.size() || s[i] != '{') {
        return;
    }
    ++i;

    while (true) {
        skip_space();
        if (i >= s.size() || s[i] == '}') {
            break;
        }
        if (s[i] != '"') {
            break;
        }
        std::string const key = read_quoted();
        skip_space();
        if (i >= s.size() || s[i] != ':') {
            break;
        }
        ++i;
        skip_space();

        std::string val;
        if (i < s.size() && s[i] == '"') {
            val = read_quoted();
        } else if (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) != 0 || s[i] == '-' || s[i] == '+')) {
            std::size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) != 0 || s[j] == '.' || s[j] == 'e' || s[j] == 'E' ||
                                    s[j] == '+' || s[j] == '-')) {
                ++j;
            }
            val = std::string(s.substr(i, j - i));
            i   = j;
        } else if (starts_with_at("true")) {
            val = "true";
            i += 4;
        } else if (starts_with_at("false")) {
            val = "false";
            i += 5;
        } else if (starts_with_at("null")) {
            i += 4;
        } else {
            // Unsupported value shape (nested object or array): skip it.
            while (i < s.size() && s[i] != ',' && s[i] != '}') {
                ++i;
            }
        }

        kv[to_lower(key)] = val;

        skip_space();
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
}

} // namespace

std::map<std::string, std::string, std::less<>> read_config(std::string_view path) {
    std::map<std::string, std::string, std::less<>> kv;
    if (path.empty()) {
        return kv;
    }

    std::ifstream const in{std::string(path), std::ios::binary};
    if (!in) {
        return kv;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string const buf = ss.str();

    if (looks_like_json(buf)) {
        parse_flat_json(buf, kv);
        return kv;
    }

    std::istringstream lines{buf};
    std::string        line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto const eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        trim(key);
        trim(val);
        kv[to_lower(key)] = val;
    }
    return kv;
}

// -------------------------- Help rendering -------------------------------- //

namespace {

constexpr std::size_t k_min_width      = 40;
constexpr std::size_t k_max_width      = 200;
constexpr std::size_t k_default_width  = 100;
constexpr std::size_t k_max_inv_column = 44;

/**
 * @brief Append `text` to `out`, wrapping at `width` with a hanging indent.
 *
 * @param col The column already occupied on the current line, measured from
 *            the hanging indent rather than from the left margin.
 * @return The column occupied after the appended text.
 */
std::size_t wrap_into(std::string &out, std::string_view text, std::size_t indent, std::size_t width, std::size_t col) {
    std::size_t const avail = width > indent + 20 ? width - indent : 20;

    std::size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && text[pos] == ' ') {
            ++pos;
        }
        std::size_t end = text.find(' ', pos);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        std::string_view const word = text.substr(pos, end - pos);
        if (word.empty()) {
            break;
        }
        if (col > 0 && col + 1 + word.size() > avail) {
            out += '\n';
            out.append(indent, ' ');
            col = 0;
        } else if (col > 0) {
            out += ' ';
            ++col;
        }
        out += word;
        col += word.size();
        pos = end;
    }
    return col;
}

} // namespace

std::size_t terminal_width() {
    // COLUMNS wins so that a caller (or a test) can pin the layout.
    if (char const *cols = std::getenv("COLUMNS"); cols != nullptr && *cols != '\0') {
        int        v{};
        auto const last    = cols + std::strlen(cols);
        auto const [p, ec] = std::from_chars(cols, last, v);
        if (ec == std::errc{} && p == last && v >= static_cast<int>(k_min_width)) {
            return (std::min)(static_cast<std::size_t>(v), k_max_width);
        }
    }

#if defined(EINSUMS_WINDOWS)
    CONSOLE_SCREEN_BUFFER_INFO csbi{};
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi) != 0) {
        auto const w = static_cast<std::size_t>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
        if (w >= k_min_width) {
            return (std::min)(w, k_max_width);
        }
    }
#else
    struct winsize ws {};
    if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && static_cast<std::size_t>(ws.ws_col) >= k_min_width) {
        return (std::min)(static_cast<std::size_t>(ws.ws_col), k_max_width);
    }
#endif
    return k_default_width;
}

std::string format_help(std::string_view prog) {
    auto &R = Registry::instance();

    std::vector<OptionBase *> visible;
    for (auto *o : R.options) {
        if (!o->is_positional && o->visibility == Visibility::Normal) {
            visible.push_back(o);
        }
    }
    auto const positionals = detail::positional_options();

    // Column widths come from the entries themselves, capped so that one very
    // long option name cannot push every description off the right edge.
    std::map<OptionBase const *, HelpEntry> entries;
    std::size_t                             inv_w    = 0;
    std::size_t                             shorts_w = 0;
    for (auto *o : visible) {
        auto e   = o->help_entry();
        inv_w    = (std::max)(inv_w, (std::min)(e.invocation.size(), k_max_inv_column));
        shorts_w = (std::max)(shorts_w, e.shorts.size());
        entries.emplace(o, std::move(e));
    }

    std::size_t const width  = terminal_width();
    std::size_t const indent = 2 + inv_w + 2 + shorts_w + 2;

    std::string out;

    out += fmt::format("Usage: {} [options]", prog);
    for (auto *p : positionals) {
        out += fmt::format(" <{}>", p->long_name);
    }
    out += "\n\n";

    auto render = [&](HelpEntry const &e) {
        if (e.invocation.size() > inv_w) {
            // Too long for its column: give the description a line of its own.
            out += fmt::format("  {}\n", e.invocation);
            out.append(indent, ' ');
        } else {
            out += fmt::format("  {:<{}}  {:<{}}  ", e.invocation, inv_w, e.shorts, shorts_w);
        }
        std::size_t const avail = width > indent + 20 ? width - indent : 20;
        std::size_t       col   = wrap_into(out, e.description, indent, width, 0);
        for (auto const &note : e.annotations) {
            // Each note is placed whole: breaking "[env: NAME]" across lines
            // makes the variable name unreadable.
            if (col > 0 && col + 1 + note.size() > avail) {
                out += '\n';
                out.append(indent, ' ');
                col = 0;
            } else if (col > 0) {
                out += ' ';
                ++col;
            }
            out += note;
            col += note.size();
        }
        out += '\n';
    };

    std::map<std::string, std::vector<OptionBase *>> groups;
    for (auto *o : visible) {
        groups[o->category ? o->category->name : std::string{}].push_back(o);
    }

    for (auto const &[cat, opts] : groups) {
        if (!cat.empty()) {
            out += fmt::format("{}:\n", cat);
        }
        for (auto *o : opts) {
            render(entries.at(o));
        }
        out += '\n';
    }

    if (!positionals.empty()) {
        out += "Positional arguments:\n";
        for (auto *p : positionals) {
            render(p->help_entry());
        }
        out += '\n';
    }

    return out;
}

// -------------------------- Diagnostics ----------------------------------- //

namespace detail {

std::optional<std::string> duplicate_long_name() {
    std::set<std::string, std::less<>> seen;
    for (auto const *o : Registry::instance().options) {
        if (o->is_positional) {
            continue;
        }
        if (!seen.insert(o->long_name).second) {
            return o->long_name;
        }
    }
    return std::nullopt;
}

bool apply_environment(std::string &error) {
    for (auto *o : Registry::instance().options) {
        std::string const name = o->effective_env_name();
        if (name.empty()) {
            continue;
        }
        char const *raw = std::getenv(name.c_str());
        if (raw == nullptr || *raw == '\0') {
            // An unset or empty variable is "not specified", matching how the
            // rest of the tree reads EINSUMS_* variables.
            continue;
        }
        std::string err;
        if (!o->parse_token(o->long_name, std::string_view{raw}, err, Source::Environment)) {
            error = fmt::format("environment variable {}: {}", name, err);
            return false;
        }
    }
    return true;
}

} // namespace detail

// -------------------------- Parser ---------------------------------------- //

namespace {

/// True for a token that should be read as an option rather than a value.
bool looks_like_option_token(std::string_view sv) {
    if (sv.size() < 2 || sv[0] != '-') {
        return false;
    }
    // "-5" and "-3.14" are values, not options.
    return std::isdigit(static_cast<unsigned char>(sv[1])) == 0;
}

} // namespace

ParseResult parse_internal(std::span<std::string const> args, char const *programName, std::string_view version,
                           std::map<std::string, std::string, std::less<>> const *config, std::vector<std::string> *unknown_args) {
    // Force the built-ins into the registry. They must be a singleton: an
    // option registers itself on construction, so a Builtins with automatic
    // storage duration would leave the registry pointing at a destroyed frame.
    Builtins &built = builtins();
    Registry::instance().ensure_category(&built.cat);
    Registry::instance().ensure_option(&built.help);
    Registry::instance().ensure_option(&built.version);

    GlobalConfigMapLockScope const config_lock;

    std::string const prog = programName != nullptr ? programName : (!args.empty() ? args[0] : std::string{"Einsums"});

    if (auto const dup = detail::duplicate_long_name(); dup.has_value()) {
        fmt::print(stderr, "error: option '--{}' is declared more than once\n", *dup);
        return {.ok = false, .exit_code = 1};
    }

    auto const fail = [](std::string_view msg) {
        fmt::print(stderr, "error: {}\n", msg);
        return ParseResult{.ok = false, .exit_code = 1};
    };

    // 1. Defaults, on top of a clean slate so a repeated parse in one process
    //    does not inherit the previous parse's state.
    for (auto *o : Registry::instance().options) {
        o->reset();
        o->finalize_default();
    }

    // 2. Config file / map.
    if (config != nullptr && !config->empty()) {
        for (auto *o : Registry::instance().options) {
            if (o->is_positional) {
                continue;
            }
            auto const it = config->find(o->long_name);
            if (it == config->end()) {
                continue;
            }
            std::string                     err;
            std::optional<std::string_view> v;
            if (!it->second.empty()) {
                v = std::string_view(it->second);
            }
            if (!o->parse_token(o->long_name, v, err, Source::ConfigFile)) {
                fmt::print(stderr, "config error for '{}': {}\n", o->long_name, err);
                return {.ok = false, .exit_code = 1};
            }
        }
    }

    // 3. Environment.
    if (std::string env_error; !detail::apply_environment(env_error)) {
        return fail(env_error);
    }

    // 4. Command line.
    std::size_t pos_index = 0;

    auto consume_positional = [&](std::string_view token, std::string &err) -> bool {
        auto const pos = detail::positional_options();
        if (pos_index >= pos.size()) {
            // Nothing left to bind it to; hand it back to the caller.
            if (unknown_args != nullptr) {
                unknown_args->emplace_back(token);
            }
            err.clear();
            return true;
        }

        OptionBase *p = pos[pos_index];
        if (!p->parse_token(p->long_name, token, err, Source::CommandLine)) {
            return false;
        }

        // A positional List keeps gathering; anything else advances.
        if (dynamic_cast<List<std::string> *>(p) == nullptr) {
            ++pos_index;
        }
        return true;
    };

    // Returns true when parsing should stop (help or version was requested).
    auto handle_builtin = [&](OptionBase const *o, ParseResult &result) {
        if (o == &built.help) {
            print_help(prog);
            result = {.ok = false, .exit_code = 0};
            return true;
        }
        if (o == &built.version) {
            print_version(prog, version);
            result = {.ok = false, .exit_code = 0};
            return true;
        }
        return false;
    };

    for (std::size_t i = 1; i < args.size(); ++i) {
        std::string_view const tok(args[i]);

        // Everything after "--" is the caller's business.
        if (tok == "--") {
            while (++i < args.size()) {
                if (unknown_args != nullptr) {
                    unknown_args->push_back(args[i]);
                }
            }
            break;
        }

        // Long options: --name or --name=value
        if (tok.size() >= 2 && tok[0] == '-' && tok[1] == '-') {
            auto const             eq   = tok.find('=');
            std::string_view const name = tok.substr(2, eq == std::string_view::npos ? tok.size() - 2 : eq - 2);
            OptionBase            *o    = detail::find_long(name);
            if (o == nullptr) {
                if (unknown_args != nullptr) {
                    unknown_args->emplace_back(tok);
                }
                continue;
            }

            std::optional<std::string_view> val;
            if (eq != std::string_view::npos) {
                if (o->value_expected == ValueExpected::ValueDisallowed) {
                    return fail(fmt::format("option '--{}' does not take a value", o->long_name));
                }
                val = tok.substr(eq + 1);
            } else if (o->value_expected == ValueExpected::ValueRequired && i + 1 < args.size() && !looks_like_option_token(args[i + 1])) {
                // Only consume the next token when it cannot be an option, so
                // that ImplicitValue(...) still has a chance to apply.
                val = std::string_view(args[++i]);
            }

            std::string err;
            if (!o->parse_token(name, val, err, Source::CommandLine)) {
                return fail(err);
            }
            if (o->on_seen) {
                o->on_seen();
            }
            if (ParseResult r{}; handle_builtin(o, r)) {
                return r;
            }
            continue;
        }

        // Short options, possibly bundled: -abc, -o value, -ovalue
        if (tok.size() >= 2 && tok[0] == '-') {
            for (std::size_t j = 1; j < tok.size(); ++j) {
                char const  c = tok[j];
                OptionBase *o = detail::find_short(c);
                if (o == nullptr) {
                    if (unknown_args != nullptr) {
                        unknown_args->push_back(fmt::format("-{}", c));
                    }
                    continue;
                }

                std::optional<std::string_view> val;
                bool const                      last_in_bundle = (j + 1 == tok.size());
                if (o->value_expected == ValueExpected::ValueRequired) {
                    if (!last_in_bundle) {
                        // The rest of the bundle is the value: -ovalue
                        val = tok.substr(j + 1);
                        j   = tok.size();
                    } else if (i + 1 < args.size() && !looks_like_option_token(args[i + 1])) {
                        val = std::string_view(args[++i]);
                    }
                }

                std::string err;
                if (!o->parse_token(std::string_view(&c, 1), val, err, Source::CommandLine)) {
                    return fail(err);
                }
                if (o->on_seen) {
                    o->on_seen();
                }
                if (ParseResult r{}; handle_builtin(o, r)) {
                    return r;
                }
            }
            continue;
        }

        // Bare token -> positional or unknown
        std::string err;
        if (!consume_positional(tok, err)) {
            return fail(err);
        }
    }

    // Required options and per-option validation. A requirement is satisfied by
    // any explicit source, not just the command line: naming the option in the
    // environment or a config file is exactly as explicit, and `occurrences`
    // counts only command-line appearances.
    for (auto const *o : Registry::instance().options) {
        if ((o->occurrence == Occurrence::Required || o->occurrence == Occurrence::OneOrMore) && !o->was_specified()) {
            return fail(fmt::format("missing required option '--{}'", o->long_name));
        }
        std::string err;
        if (!o->validate(err)) {
            return fail(err);
        }
    }

    // Mutually exclusive groups. Only options resolved at the same precedence
    // level contradict one another; the environment losing to the command line
    // is precedence working as intended, not a conflict.
    for (auto const *exc : Registry::instance().exclusions) {
        auto const conflict = exc->conflicting_source();
        if (!conflict.has_value()) {
            continue;
        }
        auto const  found = exc->found_options(*conflict);
        std::string names;
        for (auto const *o : found) {
            if (!names.empty()) {
                names += ", ";
            }
            names += fmt::format("--{}", o->long_name);
        }
        return fail(fmt::format("incompatible arguments found on the {}: {}", to_string(*conflict), names));
    }

    return {.ok = true, .exit_code = 0};
}

EINSUMS_NAMESPACE_END(cl)
