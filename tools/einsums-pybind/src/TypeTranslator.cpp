//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "TypeTranslator.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "clang/AST/PrettyPrinter.h"

namespace einsums::pybind {

std::string translate_type(clang::QualType type, clang::ASTContext const &ctx) {
    clang::PrintingPolicy policy(ctx.getLangOpts());
    policy.SuppressTagKeyword     = true;  // drop "class "/"struct " prefixes
    policy.SuppressScope          = false; // keep ::ns:: qualifiers for clarity
    policy.FullyQualifiedName     = true;  // canonical names so the emitter can match
    policy.SuppressUnwrittenScope = true;
    return type.getAsString(policy);
}

namespace {

// Tiny C++17-compatible shims for the C++20 string-view membership
// helpers. The codegen tool itself still builds at C++17 to keep the
// LLVM/Clang library matrix it links against simple.
bool sv_starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}
bool sv_ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}
bool s_ends_with(std::string const &s, std::string_view suffix) {
    return sv_ends_with(std::string_view{s}, suffix);
}

// Strip leading/trailing whitespace.
std::string_view trim(std::string_view s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n')) {
        --e;
    }
    return s.substr(b, e - b);
}

// Split a single template-argument list (text between the outermost
// `<` and `>`) into top-level comma-separated entries, respecting `<>`
// and parenthesis nesting. Each result is trimmed.
std::vector<std::string> split_template_args(std::string_view body) {
    std::vector<std::string> out;
    std::string              cur;
    int                      angle = 0;
    int                      paren = 0;
    for (char const c : body) {
        if (c == '<') {
            ++angle;
        } else if (c == '>') {
            if (angle > 0) {
                --angle;
            }
        } else if (c == '(') {
            ++paren;
        } else if (c == ')') {
            if (paren > 0) {
                --paren;
            }
        }
        if (c == ',' && angle == 0 && paren == 0) {
            out.emplace_back(trim(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.emplace_back(trim(cur));
    }
    return out;
}

// Match `prefix<...>` and return the inner argument text on success.
// Comparison ignores leading `::` so we treat `::std::vector` and
// `std::vector` the same.
std::optional<std::string> peel_template(std::string_view canonical, std::string_view prefix) {
    std::string_view s = canonical;
    if (sv_starts_with(s, "::")) {
        s.remove_prefix(2);
    }
    if (!sv_starts_with(s, prefix)) {
        return std::nullopt;
    }
    s.remove_prefix(prefix.size());
    s = trim(s);
    if (s.empty() || s.front() != '<' || s.back() != '>') {
        return std::nullopt;
    }
    s.remove_prefix(1);
    s.remove_suffix(1);
    return std::string{trim(s)};
}

// Map a fundamental scalar C++ type to its Python equivalent. Returns
// empty string when no mapping applies.
std::string map_fundamental(std::string_view canonical) {
    using namespace std::string_view_literals;
    using Row                                  = std::pair<std::string_view, std::string_view>;
    static constexpr std::array<Row, 36> table = {{
        {"bool"sv, "bool"sv},
        {"void"sv, "None"sv},
        {"char"sv, "str"sv},
        {"signed char"sv, "int"sv},
        {"unsigned char"sv, "int"sv},
        {"short"sv, "int"sv},
        {"short int"sv, "int"sv},
        {"unsigned short"sv, "int"sv},
        {"unsigned short int"sv, "int"sv},
        {"int"sv, "int"sv},
        {"unsigned"sv, "int"sv},
        {"unsigned int"sv, "int"sv},
        {"long"sv, "int"sv},
        {"long int"sv, "int"sv},
        {"unsigned long"sv, "int"sv},
        {"unsigned long int"sv, "int"sv},
        {"long long"sv, "int"sv},
        {"long long int"sv, "int"sv},
        {"unsigned long long"sv, "int"sv},
        {"unsigned long long int"sv, "int"sv},
        {"size_t"sv, "int"sv},
        {"ssize_t"sv, "int"sv},
        {"ptrdiff_t"sv, "int"sv},
        {"intptr_t"sv, "int"sv},
        {"uintptr_t"sv, "int"sv},
        {"int8_t"sv, "int"sv},
        {"uint8_t"sv, "int"sv},
        {"int16_t"sv, "int"sv},
        {"uint16_t"sv, "int"sv},
        {"int32_t"sv, "int"sv},
        {"uint32_t"sv, "int"sv},
        {"int64_t"sv, "int"sv},
        {"uint64_t"sv, "int"sv},
        {"float"sv, "float"sv},
        {"double"sv, "float"sv},
        {"long double"sv, "float"sv},
    }};
    for (auto const &row : table) {
        if (canonical == row.first) {
            return std::string{row.second};
        }
    }
    return {};
}

// Strip leading `const`/`volatile` and trailing `&`/`&&`/`*`. A
// reference/pointer to `T` translates the same as `T` for stub
// purposes; pyright doesn't model C++ value categories.
std::string_view strip_qualifiers(std::string_view canonical) {
    canonical = trim(canonical);
    while (!canonical.empty() && (canonical.back() == '&' || canonical.back() == '*')) {
        canonical.remove_suffix(1);
        canonical = trim(canonical);
    }
    auto strip_prefix = [&](std::string_view p) {
        if (sv_starts_with(canonical, p)) {
            canonical.remove_prefix(p.size());
            canonical = trim(canonical);
            return true;
        }
        return false;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        if (strip_prefix("const ")) {
            changed = true;
        }
        if (strip_prefix("volatile ")) {
            changed = true;
        }
    }
    // Also trim trailing const that some printers emit (`int const`).
    auto strip_suffix = [&](std::string_view sfx) {
        if (canonical.size() >= sfx.size() && canonical.substr(canonical.size() - sfx.size()) == sfx) {
            canonical.remove_suffix(sfx.size());
            canonical = trim(canonical);
            return true;
        }
        return false;
    };
    changed = true;
    while (changed) {
        changed = false;
        if (strip_suffix(" const")) {
            changed = true;
        }
        if (strip_suffix(" volatile")) {
            changed = true;
        }
    }
    while (!canonical.empty() && (canonical.back() == '&' || canonical.back() == '*')) {
        canonical.remove_suffix(1);
        canonical = trim(canonical);
    }
    return canonical;
}

// Recursive translation: takes a canonical C++ type string, returns a
// Python-typed string. Falls through to `cpp_text` (without leading
// `::`) when no rule applies — the IR post-pass can then resolve it
// against bound classes.
// NOLINTNEXTLINE(misc-no-recursion)
std::string translate_canonical(std::string_view canonical) {
    canonical = strip_qualifiers(canonical);
    if (canonical.empty()) {
        return "None";
    }

    // Drop any leading `::`.
    if (sv_starts_with(canonical, "::")) {
        canonical.remove_prefix(2);
    }

    if (auto const py = map_fundamental(canonical); !py.empty()) {
        return py;
    }

    // `std::`-prefixed cstdint typedefs that clang didn't canonicalize to a
    // builtin (e.g. `std::int64_t`, `std::size_t`). map_fundamental keys on
    // the unprefixed spelling, so retry once with `std::` removed.
    if (sv_starts_with(canonical, "std::")) {
        if (auto const py = map_fundamental(canonical.substr(5)); !py.empty()) {
            return py;
        }
    }

    // Standard exception types map onto Python's exception hierarchy — this
    // is an accurate translation (pybind surfaces them as Python exceptions),
    // not a fall-through to `Any`.
    if (sv_starts_with(canonical, "std::") && s_ends_with(std::string{canonical}, "_error")) {
        return "Exception";
    }
    if (canonical == "std::exception" || canonical == "std::bad_alloc" || canonical == "std::bad_cast") {
        return "Exception";
    }

    // String-likes.
    if (canonical == "std::string" || canonical == "std::string_view" || canonical == "std::wstring" || canonical == "std::wstring_view" ||
        canonical == "char *" || canonical == "const char *" || canonical == "std::filesystem::path") {
        return "str";
    }

    // std::byte / std::nullptr_t.
    if (canonical == "std::byte") {
        return "int";
    }
    if (canonical == "std::nullptr_t" || canonical == "nullptr_t") {
        return "None";
    }

    // std::optional<T>  ->  T | None
    for (auto const prefix : {"std::optional", "std::experimental::optional"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            return translate_canonical(*inner) + " | None";
        }
    }

    // std::vector<T>, std::array<T, N>, std::span<T>, std::list<T>,
    // std::deque<T>, std::initializer_list<T> -> list[T_py]
    for (auto const prefix : {"std::vector", "std::list", "std::deque", "std::initializer_list", "std::span"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            auto args = split_template_args(*inner);
            if (!args.empty()) {
                return "list[" + translate_canonical(args.front()) + "]";
            }
        }
    }
    if (auto inner = peel_template(canonical, "std::array")) {
        auto args = split_template_args(*inner);
        if (!args.empty()) {
            return "list[" + translate_canonical(args.front()) + "]";
        }
    }

    // std::set<T>, std::unordered_set<T>, std::multiset<T> -> set[T_py]
    for (auto const prefix : {"std::set", "std::unordered_set", "std::multiset", "std::unordered_multiset"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            auto args = split_template_args(*inner);
            if (!args.empty()) {
                return "set[" + translate_canonical(args.front()) + "]";
            }
        }
    }

    // std::map<K, V>, std::unordered_map<K, V>, multimap variants
    //   -> dict[K_py, V_py]
    for (auto const prefix : {"std::map", "std::unordered_map", "std::multimap", "std::unordered_multimap"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            auto args = split_template_args(*inner);
            if (args.size() >= 2) {
                return "dict[" + translate_canonical(args[0]) + ", " + translate_canonical(args[1]) + "]";
            }
        }
    }

    // std::pair<A, B> -> tuple[A_py, B_py]
    if (auto inner = peel_template(canonical, "std::pair")) {
        auto args = split_template_args(*inner);
        if (args.size() == 2) {
            return "tuple[" + translate_canonical(args[0]) + ", " + translate_canonical(args[1]) + "]";
        }
    }

    // std::tuple<...> -> tuple[..., ...] (or `tuple[()]` for empty)
    if (auto inner = peel_template(canonical, "std::tuple")) {
        auto args = split_template_args(*inner);
        if (args.empty()) {
            return "tuple[()]";
        }
        std::string out = "tuple[";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i != 0) {
                out += ", ";
            }
            out += translate_canonical(args[i]);
        }
        out += "]";
        return out;
    }

    // std::variant<...> -> A | B | C
    if (auto inner = peel_template(canonical, "std::variant")) {
        auto args = split_template_args(*inner);
        if (!args.empty()) {
            std::string out;
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (i != 0) {
                    out += " | ";
                }
                out += translate_canonical(args[i]);
            }
            return out;
        }
    }

    // std::function<R(Args...)> -> Callable[[Args_py...], R_py]
    if (auto inner = peel_template(canonical, "std::function")) {
        // inner is like "void (int, double)" — find the first '(' at
        // depth 0 to split the return type from the arg list.
        std::string const &body  = *inner;
        int                angle = 0;
        std::size_t        i     = 0;
        for (; i < body.size(); ++i) {
            char const c = body[i];
            if (c == '<') {
                ++angle;
            } else if (c == '>') {
                if (angle > 0) {
                    --angle;
                }
            } else if (c == '(' && angle == 0) {
                break;
            }
        }
        if (i < body.size()) {
            std::string const ret_text  = std::string{trim(std::string_view{body}.substr(0, i))};
            std::string       args_text = body.substr(i + 1);
            // Drop trailing `)` from args_text.
            if (!args_text.empty() && args_text.back() == ')') {
                args_text.pop_back();
            }
            auto        args = split_template_args(args_text);
            std::string out  = "Callable[[";
            for (std::size_t k = 0; k < args.size(); ++k) {
                if (k != 0) {
                    out += ", ";
                }
                out += translate_canonical(args[k]);
            }
            out += "], ";
            out += translate_canonical(ret_text);
            out += "]";
            return out;
        }
    }

    // std::shared_ptr<T>, std::unique_ptr<T>, std::weak_ptr<T> -> T_py
    for (auto const prefix : {"std::shared_ptr", "std::unique_ptr", "std::weak_ptr"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            auto args = split_template_args(*inner);
            if (!args.empty()) {
                return translate_canonical(args.front());
            }
        }
    }

    // std::reference_wrapper<T> -> T_py
    if (auto inner = peel_template(canonical, "std::reference_wrapper")) {
        auto args = split_template_args(*inner);
        if (!args.empty()) {
            return translate_canonical(args.front());
        }
    }

    // std::complex<T> -> complex (Python builtin)
    if (peel_template(canonical, "std::complex")) {
        return "complex";
    }

    // py::array_t<T> / pybind11::array_t<T> -> numpy.typing.NDArray[<dtype>]
    for (auto const prefix : {"py::array_t", "pybind11::array_t"}) {
        if (auto inner = peel_template(canonical, prefix)) {
            auto args = split_template_args(*inner);
            if (!args.empty()) {
                return "numpy.typing.NDArray[numpy." + translate_canonical(args.front()) + "]";
            }
        }
    }
    if (canonical == "py::array" || canonical == "pybind11::array") {
        return "numpy.typing.NDArray[numpy.generic]";
    }

    // Fall-through: hand back the canonical name. The post-pass over the
    // IR will resolve this against bound classes; anything still
    // unresolved will end up as the raw qualified name (and pyright
    // will flag it). This is intentional — silent `Any` makes type
    // mistakes invisible.
    return std::string{canonical};
}

} // namespace

std::string translate_python_type_string(std::string const &cpp_type) {
    return translate_canonical(cpp_type);
}

std::string translate_python_type(clang::QualType type, clang::ASTContext const &ctx) {
    // First try the as-printed form, which preserves typedef names like
    // ``Dim<Rank>`` — useful when the typedef IS a bound type.
    std::string printed_py = translate_canonical(translate_type(type, ctx));

    // If that translation didn't reduce to a recognized Python form
    // (i.e. it still looks like an unresolved C++ identifier),
    // dereference the typedef and try again on the canonical type.
    // Catches `Dim` → `std::array<size_t, Rank>` → `list[int]`.
    auto looks_unresolved = [](std::string const &s) {
        if (s.empty()) {
            return false;
        }
        // Common Python forms we've already mapped to.
        static constexpr std::array<std::string_view, 11> ok = {
            "int", "float", "bool", "str", "bytes", "complex", "None", "Any", "list", "dict", "set",
        };
        for (auto const &k : ok) {
            if (s == k) {
                return false;
            }
            if (s.size() > k.size() && s.compare(0, k.size(), k) == 0 && (s[k.size()] == '[' || s[k.size()] == ' ' || s[k.size()] == '(')) {
                return false;
            }
        }
        if (s.find("::") != std::string::npos || s.find('<') != std::string::npos) {
            return true;
        }
        if (s[0] >= 'A' && s[0] <= 'Z') {
            return true;
        }
        return false;
    };

    if (!looks_unresolved(printed_py)) {
        return printed_py;
    }

    clang::QualType const canonical = type.getCanonicalType();
    if (canonical.getAsString() == type.getAsString()) {
        return printed_py;
    }
    std::string const canon_text = translate_type(canonical, ctx);
    // Skip the canonical form if clang reports unbound template
    // parameters as ``type-parameter-N-M`` — that's strictly worse
    // than the typedef name we already have.
    if (canon_text.find("type-parameter-") != std::string::npos) {
        return printed_py;
    }
    std::string const canon_py = translate_canonical(canon_text);
    return looks_unresolved(canon_py) ? printed_py : canon_py;
}

namespace {

// Strip a single integer/float suffix (case-insensitive). Returns true
// if a suffix was removed.
bool strip_numeric_suffix(std::string &s) {
    // Float suffixes
    static constexpr std::array<std::string_view, 8> float_suffixes = {"f", "F", "l", "L", "f16", "f32", "f64", "f128"};
    for (auto const sfx : float_suffixes) {
        if (s.size() > sfx.size() && s_ends_with(s, sfx)) {
            // Only strip if the prefix actually looks numeric (contains
            // a digit and optionally a `.`). Avoid eating identifiers
            // ending in `f` like `foof`.
            std::string_view const prefix{s.data(), s.size() - sfx.size()};
            bool                   has_digit = false;
            bool                   ok        = true;
            for (char const c : prefix) {
                if (c >= '0' && c <= '9') {
                    has_digit = true;
                } else if (c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E' && c != 'x' && c != 'X' && !(c >= 'a' && c <= 'f') &&
                           !(c >= 'A' && c <= 'F') && c != '\'') {
                    ok = false;
                    break;
                }
            }
            if (has_digit && ok) {
                s.resize(prefix.size());
                return true;
            }
        }
    }
    // Integer suffixes (longest first to avoid `u` eating `ull`).
    static constexpr std::array<std::string_view, 16> int_suffixes = {"ull", "ULL", "uLL", "Ull", "llu", "LLU", "ll", "LL",
                                                                      "ul",  "UL",  "lu",  "LU",  "u",   "U",   "l",  "L"};
    for (auto const sfx : int_suffixes) {
        if (s.size() > sfx.size() && s_ends_with(s, sfx)) {
            std::string_view const prefix{s.data(), s.size() - sfx.size()};
            bool                   has_digit = false;
            bool                   ok        = true;
            for (char const c : prefix) {
                if (c >= '0' && c <= '9') {
                    has_digit = true;
                } else if (c != 'x' && c != 'X' && c != '\'' && !(c >= 'a' && c <= 'f') && !(c >= 'A' && c <= 'F')) {
                    ok = false;
                    break;
                }
            }
            if (has_digit && ok) {
                s.resize(prefix.size());
                return true;
            }
        }
    }
    return false;
}

} // namespace

std::string translate_python_default(std::string const &cpp_default) {
    std::string s = cpp_default;
    // Trim surrounding whitespace.
    {
        std::size_t b = 0;
        std::size_t e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n')) {
            ++b;
        }
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n')) {
            --e;
        }
        s = s.substr(b, e - b);
    }

    if (s.empty()) {
        return s;
    }

    // Boolean / null literals.
    if (s == "true") {
        return "True";
    }
    if (s == "false") {
        return "False";
    }
    if (s == "nullptr" || s == "std::nullopt" || s == "{}" || s == "NULL") {
        return "None";
    }

    // String literal: pybind11 will pass through, just preserve it
    // (Python accepts the same `"..."` form).
    if (s.front() == '"' && s.back() == '"') {
        return s;
    }
    // Char literal `'x'` -> `"x"`.
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return "\"" + s.substr(1, s.size() - 2) + "\"";
    }

    // Numeric literal: strip the suffix.
    if ((s.front() >= '0' && s.front() <= '9') || s.front() == '-' || s.front() == '+' || s.front() == '.') {
        std::string copy = s;
        strip_numeric_suffix(copy);
        return copy;
    }

    // No rewrite — let the .pyi emitter decide what to do (typically
    // emit `...` placeholder for arbitrary expressions).
    return s;
}

} // namespace einsums::pybind
