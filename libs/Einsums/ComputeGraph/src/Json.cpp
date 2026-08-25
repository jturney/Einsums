//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::json)

std::string ParseError::to_string() const {
    return fmt::format("line {} column {}: {}", where.line, where.column, message);
}

// ── Object ─────────────────────────────────────────────────────────────────

void Object::set(std::string key, Value value) {
    if (contains(key)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "json::Object::set: key '{}' is already present", key);
    }
    _keys.push_back(std::move(key));
    _values.push_back(std::move(value));
    _consumed.push_back(0);
}

bool Object::contains(std::string_view key) const noexcept {
    return std::ranges::find(_keys, key) != _keys.end();
}

Value const *Object::peek(std::string_view key) const noexcept {
    auto const it = std::ranges::find(_keys, key);
    if (it == _keys.end()) {
        return nullptr;
    }
    return &_values[static_cast<std::size_t>(it - _keys.begin())];
}

Value const *Object::take(std::string_view key) const noexcept {
    auto const it = std::ranges::find(_keys, key);
    if (it == _keys.end()) {
        return nullptr;
    }
    auto const index = static_cast<std::size_t>(it - _keys.begin());
    _consumed[index] = 1;
    return &_values[index];
}

void Object::mark_consumed(std::string_view key) const noexcept {
    auto const it = std::ranges::find(_keys, key);
    if (it != _keys.end()) {
        _consumed[static_cast<std::size_t>(it - _keys.begin())] = 1;
    }
}

std::vector<std::string> Object::unconsumed_keys() const {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < _keys.size(); ++i) {
        if (_consumed[i] == 0) {
            out.push_back(_keys[i]);
        }
    }
    return out;
}

std::vector<std::string> Object::keys() const {
    return _keys;
}

// ── Value ──────────────────────────────────────────────────────────────────

std::string_view Value::type_name(Type type) noexcept {
    switch (type) {
    case Type::Null:
        return "null";
    case Type::Bool:
        return "bool";
    case Type::Int:
        return "integer";
    case Type::Double:
        return "number";
    case Type::String:
        return "string";
    case Type::Array:
        return "array";
    case Type::Object:
        return "object";
    }
    return "unknown";
}

// ── Emission ───────────────────────────────────────────────────────────────

namespace {

/// Append @p text as a JSON string literal, escaping what the grammar requires.
///
/// Only the mandatory escapes plus the two conventional ones (``\n``, ``\t``
/// and friends); everything else printable goes through verbatim, and a control
/// character below 0x20 becomes ``\u00XX``. Bytes at or above 0x80 are passed
/// through, so a UTF-8 document survives a round trip unchanged rather than
/// being re-encoded into escapes.
void write_string(std::string &out, std::string_view text) {
    out.push_back('"');
    for (char const ch : text) {
        auto const byte = static_cast<unsigned char>(ch);
        switch (ch) {
        case '"':
            out += "\\\"";
            continue;
        case '\\':
            out += "\\\\";
            continue;
        case '\b':
            out += "\\b";
            continue;
        case '\f':
            out += "\\f";
            continue;
        case '\n':
            out += "\\n";
            continue;
        case '\r':
            out += "\\r";
            continue;
        case '\t':
            out += "\\t";
            continue;
        default:
            break;
        }
        if (byte < 0x20) {
            out += fmt::format("\\u{:04x}", static_cast<unsigned>(byte));
            continue;
        }
        out.push_back(ch);
    }
    out.push_back('"');
}

/// Render @p value in its shortest form that reads back bit-exactly.
///
/// ``std::to_chars``'s shortest round-trip form is the whole mechanism; the
/// only work left is making sure the result still LOOKS like a JSON number to a
/// reader that has to tell an integer from a double, so a value that comes out
/// as bare digits gets a ``.0``. Without that, a double 2.0 would be written
/// ``2`` and read back as an integer, which is exactly the type confusion the
/// distinct Int alternative exists to prevent.
void write_double(std::string &out, double value) {
    if (!std::isfinite(value)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "json::emit: {} cannot be written as a JSON number; a schema that carries specials must tag them as "
                                "strings (see Json.hpp)",
                                std::isnan(value) ? "NaN" : (value > 0 ? "+infinity" : "-infinity"));
    }
    std::array<char, 40> buffer{};
    auto const [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (ec != std::errc{}) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "json::emit: could not format a double");
    }
    std::string_view const text{buffer.data(), static_cast<std::size_t>(end - buffer.data())};
    out += text;
    if (text.find_first_of(".eE") == std::string_view::npos) {
        out += ".0";
    }
}

void write_canonical(std::string &out, Value const &value);

void write_canonical(std::string &out, Value const &value) { // NOLINT(misc-no-recursion): JSON nests.
    switch (value.type()) {
    case Value::Type::Null:
        out += "null";
        return;
    case Value::Type::Bool:
        out += value.as_bool() ? "true" : "false";
        return;
    case Value::Type::Int:
        out += std::to_string(value.as_int());
        return;
    case Value::Type::Double:
        write_double(out, value.as_double());
        return;
    case Value::Type::String:
        write_string(out, value.as_string());
        return;
    case Value::Type::Array: {
        out.push_back('[');
        bool first = true;
        for (auto const &item : value.as_array()) {
            if (!first) {
                out.push_back(',');
            }
            first = false;
            write_canonical(out, item);
        }
        out.push_back(']');
        return;
    }
    case Value::Type::Object: {
        auto const &object = value.as_object();
        out.push_back('{');
        for (std::size_t i = 0; i < object.size(); ++i) {
            if (i != 0) {
                out.push_back(',');
            }
            write_string(out, object.key_at(i));
            out.push_back(':');
            write_canonical(out, object.value_at(i));
        }
        out.push_back('}');
        return;
    }
    }
}

/// Pretty emission. Containers short enough to fit are rendered canonically on
/// one line; anything longer expands one entry per line.
void write_pretty(std::string &out, Value const &value, std::size_t indent, EmitOptions const &options);

void write_pretty(std::string &out, Value const &value, std::size_t indent, // NOLINT(misc-no-recursion): JSON nests.
                  EmitOptions const &options) {
    if (!value.is_array() && !value.is_object()) {
        write_canonical(out, value);
        return;
    }

    std::string inlined;
    write_canonical(inlined, value);
    if (inlined.size() <= options.inline_width) {
        out += inlined;
        return;
    }

    std::string const pad(2 * (indent + 1), ' ');
    std::string const closing_pad(2 * indent, ' ');

    if (value.is_array()) {
        out += "[\n";
        auto const &items = value.as_array();
        for (std::size_t i = 0; i < items.size(); ++i) {
            out += pad;
            write_pretty(out, items[i], indent + 1, options);
            out += (i + 1 < items.size()) ? ",\n" : "\n";
        }
        out += closing_pad;
        out.push_back(']');
        return;
    }

    auto const &object = value.as_object();
    out += "{\n";
    for (std::size_t i = 0; i < object.size(); ++i) {
        out += pad;
        write_string(out, object.key_at(i));
        out += ": ";
        write_pretty(out, object.value_at(i), indent + 1, options);
        out += (i + 1 < object.size()) ? ",\n" : "\n";
    }
    out += closing_pad;
    out.push_back('}');
}

} // namespace

std::string emit(Value const &value, EmitOptions options) {
    std::string out;
    if (options.style == EmitStyle::Canonical) {
        write_canonical(out, value);
    } else {
        write_pretty(out, value, 0, options);
        out.push_back('\n');
    }
    return out;
}

// ── Parsing ────────────────────────────────────────────────────────────────

namespace {

/// A recursive-descent reader over the whole document.
///
/// Position is tracked as the cursor advances rather than recomputed, so every
/// diagnostic can name a line and a column without a second pass over the text.
class Parser {
  public:
    Parser(std::string_view text, ParseOptions options) : _text(text), _options(options) {}

    [[nodiscard]] expected<Value, ParseError> run() {
        skip_whitespace();
        auto value = parse_value(0);
        if (!value) {
            return unexpected(value.error());
        }
        skip_whitespace();
        if (_pos < _text.size()) {
            return unexpected(fail(fmt::format("unexpected trailing content '{}' after the top-level value", peek_char())));
        }
        return value;
    }

  private:
    [[nodiscard]] Position here() const { return Position{.line = _line, .column = _column}; }

    [[nodiscard]] ParseError fail(std::string message) const { return ParseError{.message = std::move(message), .where = here()}; }

    [[nodiscard]] ParseError fail_at(std::string message, Position where) const {
        return ParseError{.message = std::move(message), .where = where};
    }

    [[nodiscard]] bool at_end() const noexcept { return _pos >= _text.size(); }

    [[nodiscard]] char peek_char() const noexcept { return _text[_pos]; }

    char advance() {
        char const ch = _text[_pos++];
        if (ch == '\n') {
            ++_line;
            _column = 1;
        } else {
            ++_column;
        }
        return ch;
    }

    void skip_whitespace() {
        while (!at_end()) {
            char const ch = peek_char();
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                advance();
                continue;
            }
            return;
        }
    }

    /// Match a bare keyword such as ``true``; the caller has already seen its first letter.
    [[nodiscard]] bool match_keyword(std::string_view word) {
        if (_text.substr(_pos, word.size()) != word) {
            return false;
        }
        for (std::size_t i = 0; i < word.size(); ++i) {
            advance();
        }
        return true;
    }

    // NOLINTNEXTLINE(misc-no-recursion): JSON nests; the depth cap bounds this.
    [[nodiscard]] expected<Value, ParseError> parse_value(std::size_t depth) {
        if (at_end()) {
            return unexpected(fail("unexpected end of input where a value was expected"));
        }
        Position const start = here();
        char const     ch    = peek_char();
        // NOLINTNEXTLINE(misc-no-recursion): the dispatch is part of parse_value's own recursion.
        expected<Value, ParseError> result = [&]() -> expected<Value, ParseError> {
            switch (ch) {
            case '{':
                return parse_object(depth);
            case '[':
                return parse_array(depth);
            case '"': {
                auto text = parse_string();
                if (!text) {
                    return unexpected(text.error());
                }
                return Value{std::move(*text)};
            }
            case 't':
                if (match_keyword("true")) {
                    return Value{true};
                }
                return unexpected(fail("invalid literal; expected 'true'"));
            case 'f':
                if (match_keyword("false")) {
                    return Value{false};
                }
                return unexpected(fail("invalid literal; expected 'false'"));
            case 'n':
                if (match_keyword("null")) {
                    return Value{nullptr};
                }
                return unexpected(fail("invalid literal; expected 'null'"));
            default:
                return parse_number();
            }
        }();
        if (result) {
            result->position = start;
        }
        return result;
    }

    // NOLINTNEXTLINE(misc-no-recursion): see parse_value.
    [[nodiscard]] expected<Value, ParseError> parse_object(std::size_t depth) {
        if (depth + 1 > _options.max_depth) {
            return unexpected(fail(fmt::format("nesting deeper than the {}-level limit", _options.max_depth)));
        }
        advance(); // '{'
        Object object;
        skip_whitespace();
        if (at_end()) {
            return unexpected(fail("unexpected end of input inside an object"));
        }
        if (peek_char() == '}') {
            advance();
            return Value{std::move(object)};
        }
        while (true) {
            skip_whitespace();
            if (at_end()) {
                return unexpected(fail("unexpected end of input inside an object"));
            }
            if (peek_char() != '"') {
                return unexpected(fail(fmt::format("expected a quoted key, found '{}'", peek_char())));
            }
            Position const key_at = here();
            auto           key    = parse_string();
            if (!key) {
                return unexpected(key.error());
            }
            if (object.contains(*key)) {
                return unexpected(fail_at(fmt::format("duplicate key '{}' in one object", *key), key_at));
            }
            skip_whitespace();
            if (at_end() || peek_char() != ':') {
                return unexpected(fail(fmt::format("expected ':' after key '{}'", *key)));
            }
            advance();
            skip_whitespace();
            auto value = parse_value(depth + 1);
            if (!value) {
                return unexpected(value.error());
            }
            object.set(std::move(*key), std::move(*value));

            skip_whitespace();
            if (at_end()) {
                return unexpected(fail("unexpected end of input inside an object"));
            }
            char const sep = peek_char();
            if (sep == ',') {
                advance();
                skip_whitespace();
                if (!at_end() && peek_char() == '}') {
                    return unexpected(fail("trailing comma before '}'"));
                }
                continue;
            }
            if (sep == '}') {
                advance();
                return Value{std::move(object)};
            }
            return unexpected(fail(fmt::format("expected ',' or '}}' in an object, found '{}'", sep)));
        }
    }

    // NOLINTNEXTLINE(misc-no-recursion): see parse_value.
    [[nodiscard]] expected<Value, ParseError> parse_array(std::size_t depth) {
        if (depth + 1 > _options.max_depth) {
            return unexpected(fail(fmt::format("nesting deeper than the {}-level limit", _options.max_depth)));
        }
        advance(); // '['
        Value::Array items;
        skip_whitespace();
        if (at_end()) {
            return unexpected(fail("unexpected end of input inside an array"));
        }
        if (peek_char() == ']') {
            advance();
            return Value{std::move(items)};
        }
        while (true) {
            skip_whitespace();
            auto value = parse_value(depth + 1);
            if (!value) {
                return unexpected(value.error());
            }
            items.push_back(std::move(*value));
            skip_whitespace();
            if (at_end()) {
                return unexpected(fail("unexpected end of input inside an array"));
            }
            char const sep = peek_char();
            if (sep == ',') {
                advance();
                skip_whitespace();
                if (!at_end() && peek_char() == ']') {
                    return unexpected(fail("trailing comma before ']'"));
                }
                continue;
            }
            if (sep == ']') {
                advance();
                return Value{std::move(items)};
            }
            return unexpected(fail(fmt::format("expected ',' or ']' in an array, found '{}'", sep)));
        }
    }

    /// Append @p code_point to @p out as UTF-8.
    static void append_utf8(std::string &out, std::uint32_t code_point) {
        if (code_point < 0x80) {
            out.push_back(static_cast<char>(code_point));
        } else if (code_point < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
            out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        } else if (code_point < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        }
    }

    [[nodiscard]] expected<std::uint32_t, ParseError> parse_hex4() {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            if (at_end()) {
                return unexpected(fail("unexpected end of input inside a \\u escape"));
            }
            char const    ch    = advance();
            std::uint32_t digit = 0;
            if (ch >= '0' && ch <= '9') {
                digit = static_cast<std::uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                digit = static_cast<std::uint32_t>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                digit = static_cast<std::uint32_t>(ch - 'A' + 10);
            } else {
                return unexpected(fail(fmt::format("'{}' is not a hexadecimal digit in a \\u escape", ch)));
            }
            value = (value << 4) | digit;
        }
        return value;
    }

    [[nodiscard]] expected<std::string, ParseError> parse_string() {
        advance(); // opening quote
        std::string out;
        while (true) {
            if (at_end()) {
                return unexpected(fail("unexpected end of input inside a string"));
            }
            char const ch = advance();
            if (ch == '"') {
                return out;
            }
            if (static_cast<unsigned char>(ch) < 0x20) {
                return unexpected(fail(fmt::format("unescaped control character U+{:04X} in a string", static_cast<unsigned>(ch))));
            }
            if (ch != '\\') {
                out.push_back(ch);
                continue;
            }
            if (at_end()) {
                return unexpected(fail("unexpected end of input after '\\' in a string"));
            }
            char const escape = advance();
            switch (escape) {
            case '"':
                out.push_back('"');
                continue;
            case '\\':
                out.push_back('\\');
                continue;
            case '/':
                out.push_back('/');
                continue;
            case 'b':
                out.push_back('\b');
                continue;
            case 'f':
                out.push_back('\f');
                continue;
            case 'n':
                out.push_back('\n');
                continue;
            case 'r':
                out.push_back('\r');
                continue;
            case 't':
                out.push_back('\t');
                continue;
            case 'u':
                break;
            default:
                return unexpected(fail(fmt::format("unknown escape '\\{}' in a string", escape)));
            }

            auto first = parse_hex4();
            if (!first) {
                return unexpected(first.error());
            }
            std::uint32_t code_point = *first;
            if (code_point >= 0xD800 && code_point <= 0xDBFF) {
                if (_text.substr(_pos, 2) != "\\u") {
                    return unexpected(fail("a high surrogate must be followed by a \\u low surrogate"));
                }
                advance();
                advance();
                auto second = parse_hex4();
                if (!second) {
                    return unexpected(second.error());
                }
                if (*second < 0xDC00 || *second > 0xDFFF) {
                    return unexpected(fail("a high surrogate must be followed by a low surrogate"));
                }
                code_point = 0x10000 + ((code_point - 0xD800) << 10) + (*second - 0xDC00);
            } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
                return unexpected(fail("a lone low surrogate is not a valid code point"));
            }
            append_utf8(out, code_point);
        }
    }

    [[nodiscard]] expected<Value, ParseError> parse_number() {
        std::size_t const start = _pos;
        Position const    where = here();

        if (!at_end() && peek_char() == '-') {
            advance();
        }
        if (at_end() || peek_char() < '0' || peek_char() > '9') {
            return unexpected(fail_at(fmt::format("'{}' does not start a JSON value", at_end() ? '?' : _text[start]), where));
        }
        bool const  leading_zero = peek_char() == '0';
        std::size_t digits       = 0;
        while (!at_end() && peek_char() >= '0' && peek_char() <= '9') {
            advance();
            ++digits;
        }
        if (leading_zero && digits > 1) {
            return unexpected(fail_at("a number must not have a leading zero", where));
        }

        bool is_double = false;
        if (!at_end() && peek_char() == '.') {
            is_double = true;
            advance();
            std::size_t fraction_digits = 0;
            while (!at_end() && peek_char() >= '0' && peek_char() <= '9') {
                advance();
                ++fraction_digits;
            }
            if (fraction_digits == 0) {
                return unexpected(fail_at("a number needs at least one digit after '.'", where));
            }
        }
        if (!at_end() && (peek_char() == 'e' || peek_char() == 'E')) {
            is_double = true;
            advance();
            if (!at_end() && (peek_char() == '+' || peek_char() == '-')) {
                advance();
            }
            std::size_t exponent_digits = 0;
            while (!at_end() && peek_char() >= '0' && peek_char() <= '9') {
                advance();
                ++exponent_digits;
            }
            if (exponent_digits == 0) {
                return unexpected(fail_at("a number needs at least one digit in its exponent", where));
            }
        }

        std::string_view const text{_text.data() + start, _pos - start};
        if (!is_double) {
            std::int64_t value   = 0;
            auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                return unexpected(fail_at(fmt::format("integer '{}' does not fit in a 64-bit signed integer", text), where));
            }
            return Value{value};
        }
        double value         = 0.0;
        auto const [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec != std::errc{} || end != text.data() + text.size()) {
            return unexpected(fail_at(fmt::format("'{}' is not a representable number", text), where));
        }
        return Value{value};
    }

    std::string_view _text;
    ParseOptions     _options;
    std::size_t      _pos{0};
    std::size_t      _line{1};
    std::size_t      _column{1};
};

} // namespace

expected<Value, ParseError> parse(std::string_view text, ParseOptions options) {
    Parser parser(text, options);
    return parser.run();
}

// NOLINTNEXTLINE(misc-no-recursion): JSON nests.
void collect_unconsumed(Value const &value, std::string const &path, std::vector<std::string> &out) {
    if (value.is_array()) {
        auto const &items = value.as_array();
        for (std::size_t i = 0; i < items.size(); ++i) {
            collect_unconsumed(items[i], fmt::format("{}[{}]", path, i), out);
        }
        return;
    }
    if (!value.is_object()) {
        return;
    }
    auto const &object     = value.as_object();
    auto const  unconsumed = object.unconsumed_keys();
    for (std::size_t i = 0; i < object.size(); ++i) {
        std::string const child = fmt::format("{}.{}", path, object.key_at(i));
        if (std::ranges::find(unconsumed, object.key_at(i)) != unconsumed.end()) {
            out.push_back(child);
            continue; // an unconsumed subtree is reported once, at its root
        }
        collect_unconsumed(object.value_at(i), child, out);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::json)
