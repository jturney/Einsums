//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Json.hpp
 * @brief A strict, consumed-key JSON document model for the graph IR.
 *
 * @par Why a document model of our own
 * This module already had two JSON stories and neither is the one a graph IR
 * needs. ``Graph::to_json`` writes a debug view and never reads it back.
 * ``CostModel``'s hand parser scans for a field name and silently skips every
 * row it does not recognise, which is the same failure mode that got Glaze
 * rejected for this job: a file whose content the reader half-understood loads
 * as if it were fully understood, and the part that was dropped is discovered
 * as a wrong number much later.
 *
 * So the rule here is the opposite one, and it is a POLICY rather than a
 * courtesy: every object records which of its keys a reader consumed, and the
 * loader refuses a document that still has an unconsumed key, naming the key
 * and its path. A field this build does not know about is an error, not a
 * shrug. Additive schema evolution stays possible because the compatibility
 * policy makes the READER responsible for knowing the field (a newer build loads
 * an older IR, never the reverse), so an unconsumed key genuinely
 * means "this file was written by something this build does not understand".
 *
 * @par The value model
 * Deliberately small: null, bool, int64, double, string, array, object. Two
 * properties are load bearing.
 *
 * - ``int64`` is a DISTINCT type from ``double``. A tensor extent, a node id
 *   and an iteration count are integers, and a model that stores them as
 *   doubles cannot say so; keeping them apart is also what lets a future
 *   packed binary encoding map this model one-to-one without inventing a
 *   discriminator.
 * - Object keys keep their INSERTION ORDER. The canonical emission of a
 *   document is its hash domain (@ref emit with @ref EmitStyle::Canonical), so
 *   key order has to be a property of the writer rather than of a hash table's
 *   seed.
 *
 * @par Numbers
 * Doubles round-trip bit-exactly: @ref emit writes the shortest form
 * ``std::to_chars`` produces and the parser reads it back with
 * ``std::from_chars``, so ``parse(emit(x)) == x`` holds for every finite
 * double including denormals. JSON has no NaN and no infinity, and this model
 * does not invent them: @ref emit REFUSES a non-finite double rather than
 * writing ``null`` or a bare ``nan`` token. A schema that needs to carry a
 * special encodes it as a tagged STRING inside its own typed object -
 * ``{"dtype":"float64","re":"nan"}`` - which is what GraphIR.hpp does for
 * prefactors.
 *
 * @par Errors
 * Every parse failure carries a line and a column, and a truncated or
 * malformed document fails whole: @ref parse returns an error and never a
 * partially populated value. Nesting is capped (@ref ParseOptions::max_depth)
 * so a depth bomb is a diagnostic rather than a stack overflow.
 *
 * @see GraphIR.hpp for the schema this document model carries
 * @see Einsums/ComputeGraph/GraphIR.hpp for the schema written on top of this.
 */

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::json)

class Value;

/// An ordered list of values, which is what a JSON array holds.
///
/// Declared here rather than as a member of @ref Value because @ref Value's own
/// storage needs it: a ``std::vector`` of an incomplete type is legal, and the
/// element type completes before anything indexes one.
using Array = std::vector<Value>;

/**
 * @brief A 1-based source position, for a diagnostic that points at the text.
 * @versionadded{2.0.0}
 */
struct Position {
    std::size_t line{1};   ///< 1-based line number.
    std::size_t column{1}; ///< 1-based column, counted in bytes.
};

/**
 * @brief What went wrong while parsing, and where.
 * @versionadded{2.0.0}
 */
struct ParseError {
    std::string message; ///< Human-readable cause.
    Position    where;   ///< Where in the text it was detected.

    /// @brief The message with its position prefixed, as a loader would report it.
    /// @return ``"line L column C: message"``.
    [[nodiscard]] EINSUMS_EXPORT std::string to_string() const;
};

/**
 * @brief An ordered, consumption-tracking JSON object.
 *
 * Keys keep insertion order (see the file note on the hash domain) and a
 * duplicate key is a parse error rather than a last-one-wins overwrite.
 *
 * Reads come in two flavours, and the distinction is the point of the type:
 * @ref take marks the key consumed, @ref peek does not. A loader uses
 * @ref take for everything it understands and then asks @ref unconsumed_keys
 * what is left; anything left is a field this build does not know, which the
 * strict policy refuses.
 *
 * @note The export annotation sits on the individual members, NOT on the class.
 *       ``__declspec(dllexport)`` on a class instantiates its implicit members
 *       right at the class definition, and this one holds a
 *       ``std::vector<Value>`` while @ref Value is still incomplete: declaring
 *       the vector is legal there, destroying it is not. Exporting the class
 *       wholesale therefore fails to compile under MSVC and clang-cl even
 *       though it is fine on the Itanium ABI. @ref Value carries the same
 *       constraint, for the same reason.
 *
 * @versionadded{2.0.0}
 */
class Object {
  public:
    Object() = default;

    /**
     * @brief Append @p key with @p value.
     * @param[in] key   The key. Appending a key that is already present is a
     *            caller error and throws, mirroring the parser's duplicate-key rule.
     * @param[in] value The value.
     * @throws std::invalid_argument When @p key is already present.
     */
    EINSUMS_EXPORT void set(std::string key, Value value);

    /// @brief Whether @p key is present.
    /// @param[in] key The key to look for.
    /// @return True when present, consumed or not.
    [[nodiscard]] EINSUMS_EXPORT bool contains(std::string_view key) const noexcept;

    /**
     * @brief Read @p key WITHOUT marking it consumed.
     * @param[in] key The key to read.
     * @return The value, or nullptr when absent.
     */
    [[nodiscard]] EINSUMS_EXPORT Value const *peek(std::string_view key) const noexcept;

    /**
     * @brief Read @p key and mark it consumed.
     * @param[in] key The key to read.
     * @return The value, or nullptr when absent (nothing is marked then).
     */
    [[nodiscard]] EINSUMS_EXPORT Value const *take(std::string_view key) const noexcept;

    /// @brief Mark @p key consumed without reading it, for a key a reader
    ///        deliberately ignores.
    /// @param[in] key The key to mark.
    EINSUMS_EXPORT void mark_consumed(std::string_view key) const noexcept;

    /// @brief Keys still unconsumed, in insertion order.
    /// @return The names. Empty when the reader understood the whole object.
    [[nodiscard]] EINSUMS_EXPORT std::vector<std::string> unconsumed_keys() const;

    /// @brief Every key, in insertion order.
    /// @return The names.
    [[nodiscard]] EINSUMS_EXPORT std::vector<std::string> keys() const;

    /// How many keys the object holds.
    [[nodiscard]] std::size_t size() const noexcept { return _keys.size(); }

    /// Whether the object holds no keys.
    [[nodiscard]] bool empty() const noexcept { return _keys.empty(); }

    /// @brief The key at @p index, in insertion order.
    /// @param[in] index Position. Must be less than @ref size.
    /// @return The key.
    [[nodiscard]] std::string const &key_at(std::size_t index) const { return _keys[index]; }

    /// @brief The value at @p index, in insertion order.
    /// @param[in] index Position. Must be less than @ref size.
    /// @return The value.
    ///
    /// Defined below @ref Value, which is incomplete here: the object holds a
    /// ``std::vector`` of values, and a vector of an incomplete type is legal
    /// while indexing one is not.
    [[nodiscard]] Value const &value_at(std::size_t index) const;

  private:
    std::vector<std::string> _keys;
    std::vector<Value>       _values;

    /// One flag per key. Mutable because consumption is bookkeeping about the
    /// READER, not content of the document: a loader walks a ``Value const &``
    /// and must still be able to record what it understood.
    mutable std::vector<char> _consumed;
};

/**
 * @brief One JSON value: null, bool, int64, double, string, array or object.
 *
 * @note Exported per-member rather than per-class; see the note on @ref Object.
 *       @ref Array is a ``std::vector<Value>``, so this class holds itself
 *       incompletely too.
 *
 * @versionadded{2.0.0}
 */
class Value {
  public:
    /// The seven alternatives, in the order @ref Type enumerates them.
    using Storage = std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;

    /// Which alternative a value holds.
    enum class Type : std::uint8_t {
        Null,   ///< ``null``
        Bool,   ///< ``true`` / ``false``
        Int,    ///< an integer that fits in @c std::int64_t
        Double, ///< a number with a fraction or an exponent
        String, ///< a string
        Array,  ///< an ordered list
        Object, ///< an ordered, consumption-tracking map
    };

    Value() = default;
    Value(std::nullptr_t) : _storage(nullptr) {}                             // NOLINT(google-explicit-constructor)
    Value(bool value) : _storage(value) {}                                   // NOLINT(google-explicit-constructor)
    Value(std::int64_t value) : _storage(value) {}                           // NOLINT(google-explicit-constructor)
    Value(int value) : _storage(static_cast<std::int64_t>(value)) {}         // NOLINT(google-explicit-constructor)
    Value(std::size_t value) : _storage(static_cast<std::int64_t>(value)) {} // NOLINT(google-explicit-constructor)
    Value(double value) : _storage(value) {}                                 // NOLINT(google-explicit-constructor)
    Value(std::string value) : _storage(std::move(value)) {}                 // NOLINT(google-explicit-constructor)
    Value(char const *value) : _storage(std::string(value)) {}               // NOLINT(google-explicit-constructor)
    Value(std::string_view value) : _storage(std::string(value)) {}          // NOLINT(google-explicit-constructor)
    Value(Array value) : _storage(std::move(value)) {}                       // NOLINT(google-explicit-constructor)
    Value(json::Object value) : _storage(std::move(value)) {}                // NOLINT(google-explicit-constructor)

    /// The alternative this value holds.
    [[nodiscard]] Type type() const noexcept { return static_cast<Type>(_storage.index()); }

    [[nodiscard]] bool is_null() const noexcept { return type() == Type::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return type() == Type::Bool; }
    [[nodiscard]] bool is_int() const noexcept { return type() == Type::Int; }
    [[nodiscard]] bool is_double() const noexcept { return type() == Type::Double; }
    /// True for either numeric alternative, for a reader that accepts ``1`` where a double is meant.
    [[nodiscard]] bool is_number() const noexcept { return is_int() || is_double(); }
    [[nodiscard]] bool is_string() const noexcept { return type() == Type::String; }
    [[nodiscard]] bool is_array() const noexcept { return type() == Type::Array; }
    [[nodiscard]] bool is_object() const noexcept { return type() == Type::Object; }

    /// @brief The boolean this value holds.
    /// @return The value. Precondition: @ref is_bool.
    [[nodiscard]] bool as_bool() const { return std::get<bool>(_storage); }

    /// @brief The integer this value holds.
    /// @return The value. Precondition: @ref is_int.
    [[nodiscard]] std::int64_t as_int() const { return std::get<std::int64_t>(_storage); }

    /// @brief This value as a double, widening an integer if that is what it holds.
    /// @return The value. Precondition: @ref is_number.
    [[nodiscard]] double as_double() const {
        return is_int() ? static_cast<double>(std::get<std::int64_t>(_storage)) : std::get<double>(_storage);
    }

    /// @brief The string this value holds.
    /// @return The value. Precondition: @ref is_string.
    [[nodiscard]] std::string const &as_string() const { return std::get<std::string>(_storage); }

    /// @brief The array this value holds.
    /// @return The value. Precondition: @ref is_array.
    [[nodiscard]] Array const &as_array() const { return std::get<Array>(_storage); }

    /// @brief The object this value holds.
    /// @return The value. Precondition: @ref is_object.
    [[nodiscard]] json::Object const &as_object() const { return std::get<json::Object>(_storage); }

    /// @brief The name of @p type, for a diagnostic.
    /// @param[in] type The type to name.
    /// @return ``"null"``, ``"bool"``, ``"integer"``, ``"number"``, ``"string"``, ``"array"`` or ``"object"``.
    [[nodiscard]] EINSUMS_EXPORT static std::string_view type_name(Type type) noexcept;

    /// The name of this value's own type.
    [[nodiscard]] std::string_view type_name() const noexcept { return type_name(type()); }

    /// Where in the parsed text this value began. Default-constructed for a value a writer built.
    Position position;

  private:
    Storage _storage{nullptr};
};

inline Value const &Object::value_at(std::size_t index) const {
    return _values[index];
}

/// @brief How @ref emit lays a document out.
enum class EmitStyle : std::uint8_t {
    /// No whitespace at all, keys in insertion order, numbers in their shortest
    /// round-tripping form. This is the byte sequence a content hash is taken
    /// over, so it must be a pure function of the document.
    Canonical,

    /// Diff-tuned: one key per line, with a container that fits in
    /// @ref EmitOptions::inline_width rendered on a single line. That is what
    /// puts one node on one line, so a graph diff shows the nodes that changed
    /// rather than a reflowed file.
    Pretty,
};

/// @brief Knobs for @ref emit.
struct EmitOptions {
    EmitStyle style{EmitStyle::Canonical}; ///< Layout.

    /// Longest canonical rendering that @ref EmitStyle::Pretty still puts on one
    /// line. Ignored by @ref EmitStyle::Canonical.
    std::size_t inline_width{160};
};

/// @brief Knobs for @ref parse.
struct ParseOptions {
    /// Deepest nesting accepted, counting each array and object as one level.
    /// A document past it is refused with its position, which is what turns a
    /// depth bomb into a diagnostic instead of a stack overflow.
    std::size_t max_depth{64};
};

/**
 * @brief Parse @p text as a strict JSON document.
 *
 * @param[in] text    The whole document.
 * @param[in] options Depth limit; see @ref ParseOptions.
 * @return The value, or the first error with its position.
 *
 * Strict means: no comments, no trailing commas, no unquoted keys, no
 * duplicate keys, no ``NaN``/``Infinity`` literals, no leading ``+``, no
 * leading zeros, no unescaped control characters, no lone surrogates, and no
 * trailing content after the top-level value. Every one of those is a place a
 * lenient reader would guess, and guessing is what this model exists to stop.
 *
 * A number is @ref Value::Type::Int when it is written with no fraction and no
 * exponent AND fits in @c std::int64_t; one that is written as an integer but
 * does not fit is an error rather than a silent widening to double.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT expected<Value, ParseError> parse(std::string_view text, ParseOptions options = {});

/**
 * @brief Render @p value as JSON text.
 *
 * @param[in] value   The document.
 * @param[in] options Layout; see @ref EmitOptions.
 * @return The text.
 * @throws std::invalid_argument When the document holds a non-finite double.
 *         JSON cannot spell one, and writing ``null`` or ``nan`` in its place
 *         would be exactly the silent corruption this model refuses; a schema
 *         that needs specials tags them as strings.
 *
 * ``parse(emit(v, Canonical))`` reproduces @p value exactly, and
 * ``emit(parse(t), Canonical)`` reproduces @p t byte for byte when @p t is
 * itself canonical. Both directions are tested.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::string emit(Value const &value, EmitOptions options = {});

/**
 * @brief Collect the paths of every key no reader consumed.
 *
 * @param[in]     value The document, after a reader has walked it.
 * @param[in]     path  Prefix for the reported paths, e.g. ``"$"``.
 * @param[in,out] out   Destination; paths are appended in document order.
 *
 * Recurses into arrays and objects, so one call over the root reports the whole
 * document. A path looks like ``$.nodes[3].descriptor.alpha``.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void collect_unconsumed(Value const &value, std::string const &path, std::vector<std::string> &out);

EINSUMS_NAMESPACE_END(compute_graph::json)
