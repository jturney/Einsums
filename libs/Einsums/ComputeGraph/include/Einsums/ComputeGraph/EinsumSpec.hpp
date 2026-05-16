//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Error.hpp>

/**
 * @file EinsumSpec.hpp
 * @brief Parser for string-based einsum notation.
 *
 * Supports two notation styles:
 *
 * **Arrow notation** (output on left):
 * @code
 * "ij <- ik ; kj"              // single-char indices
 * "mu,nu <- mu,rho ; rho,nu"   // multi-char indices
 * @endcode
 *
 * **NumPy notation** (output on right):
 * @code
 * "ik;kj -> ij"                // single-char indices
 * "mu,rho;rho,nu -> mu,nu"     // multi-char indices
 * @endcode
 *
 * **Index parsing rules:**
 * - If ANY index group contains a comma → multi-char mode (commas separate indices)
 * - If NO commas → single-char mode (each character is one index)
 * - Whitespace is ignored everywhere
 * - `;` always separates operands
 * - `<-` or `->` separates output from inputs
 */

#include <string>
#include <string_view>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Result of parsing an einsum specification string.
 *
 * Contains the parsed index lists for the output tensor (C) and the two
 * input tensors (A and B). Also stores the original string for error messages.
 */
struct EINSUMS_EXPORT ParsedEinsumSpec {
    std::vector<std::string> c_indices; ///< Output (C) indices
    std::vector<std::string> a_indices; ///< First input (A) indices
    std::vector<std::string> b_indices; ///< Second input (B) indices
    std::string              raw;       ///< Original specification string

    /// Compute link indices (in both A and B, not in C).
    [[nodiscard]] std::vector<std::string> link_indices() const;

    /// Compute target indices (unique indices in C).
    [[nodiscard]] std::vector<std::string> target_indices() const;
};

/**
 * @brief Parse an einsum specification string.
 *
 * Supports both arrow (`<-`) and NumPy (`->`) notation. Auto-detects
 * single-char vs multi-char index mode based on presence of commas.
 *
 * @param[in] spec The specification string (e.g., "ij <- ik ; kj").
 * @return Parsed specification with separated index lists.
 * @throws std::invalid_argument If the string is malformed.
 *
 * @par Examples
 * @code
 * auto s1 = parse_einsum_spec("ij <- ik ; kj");
 * // s1.c_indices = {"i", "j"}, s1.a_indices = {"i", "k"}, s1.b_indices = {"k", "j"}
 *
 * auto s2 = parse_einsum_spec("mu,nu <- mu,rho ; rho,nu");
 * // s2.c_indices = {"mu", "nu"}, s2.a_indices = {"mu", "rho"}, s2.b_indices = {"rho", "nu"}
 *
 * auto s3 = parse_einsum_spec("ik;kj -> ij");
 * // s3.c_indices = {"i", "j"}, s3.a_indices = {"i", "k"}, s3.b_indices = {"k", "j"}
 * @endcode
 */
[[nodiscard]] EINSUMS_EXPORT expected<ParsedEinsumSpec, GraphError> parse_einsum_spec(std::string_view spec);

/**
 * @brief Validate an einsum specification string at compile time.
 *
 * Checks structural validity: presence of exactly one arrow (`<-` or `->`),
 * presence of exactly one semicolon for operand separation, valid characters.
 * Does NOT check tensor rank compatibility (that requires runtime info).
 *
 * @param[in] spec The specification string.
 * @return True if structurally valid.
 *
 * @code
 * static_assert(validate_einsum_spec("ij <- ik ; kj"));
 * static_assert(!validate_einsum_spec("ij <- ik"));        // Missing semicolon
 * static_assert(!validate_einsum_spec("ij ik ; kj"));      // Missing arrow
 * @endcode
 */
constexpr bool validate_einsum_spec(std::string_view spec);

// ─── Implementation of constexpr validator ──────────────────────────────────

namespace detail {

constexpr bool is_einsum_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ',' || c == ';' || c == '-' || c == '<' ||
           c == '>' || c == ' ' || c == '\t';
}

} // namespace detail

// ─── Compile-time index counter ─────────────────────────────────────────────
//
// Used by cg::einsum to validate that the rank of each typed tensor operand
// matches the number of indices the spec asks for. Runs at consteval time
// from EinsumFormatString's literal ctor so the resulting counts fold to
// compile-time constants at every well-typed callsite. Zeroed for runtime-
// constructed strings — the dispatcher then skips the rank check, matching
// the "if possible, compile-time check; otherwise silent" policy.

/// @brief Per-operand index counts parsed from an einsum spec.
struct IndexCounts {
    std::size_t c     = 0;
    std::size_t a     = 0;
    std::size_t b     = 0;
    bool        known = false; ///< true when populated by a consteval parse.
};

namespace detail {

// Count indices in one operand: comma-separated multi-char tokens if a
// comma is present, otherwise one index per non-whitespace character.
// Used by parse_index_counts below.
constexpr std::size_t count_operand_indices(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
        s.remove_suffix(1);
    if (s.empty())
        return 0;

    bool has_comma = false;
    for (char const ch : s) {
        if (ch == ',') {
            has_comma = true;
            break;
        }
    }

    if (has_comma) {
        std::size_t commas   = 0;
        bool        in_token = false;
        for (char const ch : s) {
            if (ch == ',') {
                ++commas;
                in_token = false;
            } else if (ch != ' ' && ch != '\t') {
                in_token = true;
            }
        }
        return commas + 1;
    }

    std::size_t n = 0;
    for (char const ch : s) {
        if (ch != ' ' && ch != '\t')
            ++n;
    }
    return n;
}

} // namespace detail

/// @brief Parse per-operand index counts from a (validated) einsum spec.
///
/// Accepts both arrow forms — ``"C <- A ; B"`` and ``"A ; B -> C"``.
/// Returns ``known = false`` for malformed input so the caller falls back
/// to runtime parsing (validate_einsum_spec is responsible for diagnostics).
constexpr IndexCounts parse_index_counts(std::string_view spec) {
    IndexCounts r;

    std::size_t arrow_pos = std::string_view::npos;
    bool        reverse   = false;
    for (std::size_t i = 0; i + 1 < spec.size(); ++i) {
        if (spec[i] == '<' && spec[i + 1] == '-') {
            arrow_pos = i;
            reverse   = false;
            break;
        }
        if (spec[i] == '-' && spec[i + 1] == '>') {
            arrow_pos = i;
            reverse   = true;
            break;
        }
    }
    if (arrow_pos == std::string_view::npos)
        return r;

    std::size_t const semi = spec.find(';');
    if (semi == std::string_view::npos)
        return r;

    std::string_view target_part;
    std::string_view a_part;
    std::string_view b_part;
    if (reverse) {
        // "A ; B -> C"
        if (semi >= arrow_pos)
            return r;
        a_part      = spec.substr(0, semi);
        b_part      = spec.substr(semi + 1, arrow_pos - semi - 1);
        target_part = spec.substr(arrow_pos + 2);
    } else {
        // "C <- A ; B"
        if (semi <= arrow_pos)
            return r;
        target_part = spec.substr(0, arrow_pos);
        a_part      = spec.substr(arrow_pos + 2, semi - arrow_pos - 2);
        b_part      = spec.substr(semi + 1);
    }

    r.c     = detail::count_operand_indices(target_part);
    r.a     = detail::count_operand_indices(a_part);
    r.b     = detail::count_operand_indices(b_part);
    r.known = true;
    return r;
}

constexpr bool validate_einsum_spec(std::string_view spec) {
    // Check all characters are valid
    for (char const c : spec) {
        if (!detail::is_einsum_char(c))
            return false;
    }

    // Count arrows
    int left_arrows  = 0; // <-
    int right_arrows = 0; // ->
    for (size_t i = 0; i + 1 < spec.size(); i++) {
        if (spec[i] == '<' && spec[i + 1] == '-')
            left_arrows++;
        if (spec[i] == '-' && spec[i + 1] == '>')
            right_arrows++;
    }

    // Exactly one arrow type
    if (left_arrows + right_arrows != 1)
        return false;

    // Count semicolons (operand separator)
    int semicolons = 0;
    for (char const c : spec) {
        if (c == ';')
            semicolons++;
    }

    // Need exactly one semicolon to separate the two operands
    if (semicolons != 1)
        return false;

    return true;
}

/**
 * @brief Compile-time validated einsum format string (fmtlib pattern).
 *
 * This type has a `consteval` constructor that validates the einsum string
 * at compile time. When used as a function parameter, string literals are
 * checked at compile time for structural validity:
 *
 * @code
 * cg::einsum(EinsumFormatString("ij <- ik ; kj"), &C, A, B);  // OK — validated at compile time
 * cg::einsum(EinsumFormatString("ij <- ik"),       &C, A, B);  // Compile error! Missing ';'
 * @endcode
 *
 * For runtime-constructed strings (e.g., from Python), use the `std::string_view`
 * overloads of einsum() directly.
 *
 * @note This follows the same pattern as `fmt::format_string<T...>` in fmtlib.
 */
struct EinsumFormatString {
    std::string_view str;
    /// Per-operand index counts. Populated at consteval time when the
    /// string is a literal; ``counts.known == false`` for runtime-built
    /// strings. Used by cg::einsum to validate operand ranks.
    IndexCounts counts;

    /**
     * @brief Construct from a string literal with compile-time validation.
     *
     * The `consteval` keyword ensures this constructor runs at compile time
     * for string literals. If the string is invalid, a compile error is produced.
     *
     * @param[in] s The einsum specification string literal.
     */
    template <size_t N>
    consteval EinsumFormatString(char const (&s)[N]) : str(s, N - 1), counts(parse_index_counts(str)) { // NOLINT
        if (!validate_einsum_spec(str)) {
            throw "Invalid einsum format string: must contain exactly one '<-' or '->' and exactly one ';'";
        }
    }

    /**
     * @brief Construct from a runtime string (no compile-time validation).
     *
     * Use this for strings constructed at runtime (e.g., from Python bindings
     * or user input). Validation happens at runtime via parse_einsum_spec().
     *
     * Non-explicit to allow Python-facing bindings to receive a Python ``str``
     * — pybind11 keeps the converted std::string alive for the duration of
     * the call, so the string_view remains valid throughout the einsum
     * dispatch. C++ callers passing a temporary string should still wrap
     * explicitly for clarity.
     *
     * @param[in] s The einsum specification string.
     */
    EinsumFormatString(std::string_view s) : str(s), counts{} {} // NOLINT(google-explicit-constructor)

    /// Implicit conversion to string_view for use with parse_einsum_spec().
    constexpr operator std::string_view() const { return str; }
};

// ═════════════════════════════════════════════════════════════════════════════
// Permute format string
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Result of parsing a permute specification string.
 *
 * Contains the parsed index lists for the output tensor (C) and the
 * input tensor (A). Also stores the original string for error messages.
 */
struct ParsedPermuteSpec {
    std::vector<std::string> c_indices; ///< Output (C) indices
    std::vector<std::string> a_indices; ///< Input (A) indices
    std::string              raw;       ///< Original specification string
};

/**
 * @brief Parse a permute specification string.
 *
 * Supports arrow notation: `"ijk <- kji"` or `"kji -> ijk"`.
 * No semicolon needed (only one input tensor).
 *
 * @par Examples
 * @code
 * auto s1 = parse_permute_spec("ji <- ij");
 * // s1.c_indices = {"j", "i"}, s1.a_indices = {"i", "j"}
 *
 * auto s2 = parse_permute_spec("mu,nu <- nu,mu");
 * // s2.c_indices = {"mu", "nu"}, s2.a_indices = {"nu", "mu"}
 * @endcode
 */
[[nodiscard]] EINSUMS_EXPORT expected<ParsedPermuteSpec, GraphError> parse_permute_spec(std::string_view spec);

/**
 * @brief Validate a permute specification string at compile time.
 *
 * Checks structural validity: presence of exactly one arrow, no semicolons.
 */
constexpr bool validate_permute_spec(std::string_view spec) {
    for (char const c : spec) {
        if (!detail::is_einsum_char(c))
            return false;
    }

    int left_arrows = 0, right_arrows = 0;
    for (size_t i = 0; i + 1 < spec.size(); i++) {
        if (spec[i] == '<' && spec[i + 1] == '-')
            left_arrows++;
        if (spec[i] == '-' && spec[i + 1] == '>')
            right_arrows++;
    }
    if (left_arrows + right_arrows != 1)
        return false;

    // Permute has NO semicolon (only one input tensor)
    for (char const c : spec) {
        if (c == ';')
            return false;
    }

    return true;
}

/**
 * @brief Compile-time validated permute format string.
 *
 * @code
 * cg::permute("ji <- ij", 1.0, &C, 0.0, A);  // Compile-time validated
 * @endcode
 */
struct PermuteFormatString {
    std::string_view str;

    template <size_t N>
    consteval PermuteFormatString(char const (&s)[N]) : str(s, N - 1) { // NOLINT
        if (!validate_permute_spec(str)) {
            throw "Invalid permute format string: must contain exactly one '<-' or '->' and no ';'";
        }
    }

    explicit PermuteFormatString(std::string_view s) : str(s) {}

    constexpr operator std::string_view() const { return str; }
};

} // namespace einsums::compute_graph
