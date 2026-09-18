//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/Config/Namespace.hpp>

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
 * - Decided PER OPERAND: an operand containing a comma is comma-split (so
 *   multi-char index names work), a comma-less operand is char-split, one index
 *   per character. Deciding it across the whole spec mis-tokenized a comma-less
 *   operand whenever a sibling used commas.
 * - Whitespace is ignored everywhere
 * - `;` always separates operands
 * - `<-` or `->` separates output from inputs
 *
 * **Permutation operators** (@ref PermutationOperator) may prefix the TERM, which
 * is the right-hand side under either arrow:
 * @code
 * "i,a,j,b <- P(i/j) P(a/b) i,k,a,c ; k,c,j,b"
 * "P(i/j) P(a/b) i,k,a,c ; k,c,j,b -> i,a,j,b"
 * @endcode
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief One permutation (antisymmetrizer) operator, as written `P(i/jk)`.
 *
 * The operator partitions a set of OUTPUT index letters into two or more
 * disjoint, non-empty, ordered groups, and expands to one term per coset of the
 * Young subgroup @f$S(g_0) \times \dots \times S(g_{k-1})@f$ in @f$S(L)@f$,
 * where @f$L@f$ is the concatenation of the groups. The term count is
 * @f$n! / \prod_t |g_t|!@f$ and each term carries the parity of its
 * representative as a sign.
 *
 * The expansions, which are the ones the coupled-cluster literature prints:
 *
 * - `P(i/j)`, 2 terms: `1 - (ij)`
 * - `P(i/jk)`, 3 terms: `1 - (ij) - (ik)`
 * - `P(ij/k)`, 3 terms: `1 - (ik) - (jk)`
 * - `P(i/j/k)`, 6 terms: the full antisymmetrizer over `ijk`
 * - `P(ij/kl)`, 6 terms: `1 - (ik) - (il) - (jk) - (jl) + (ik)(jl)`
 *
 * @warning A coset contains members of DIFFERENT parity, so which one is chosen
 * changes the value. The operator is well-defined only on a base term that is
 * already antisymmetric within each group, which is what makes the notation
 * legitimate in the coupled-cluster literature and holds wherever it is used
 * there. @ref expand_permutation_operators documents the representative this
 * implementation picks and why.
 *
 * Comma-vs-character mode is decided across the WHOLE operator, not per group:
 * `P(mu,nu/rho)` has to read `rho` as one index, which a per-group rule would
 * char-split into three. `P(ij)` is sugar for `P(i/j)` in character mode only,
 * since `P(mu,nu)` is otherwise ambiguous between one group of two and two
 * groups of one.
 *
 * @versionadded{2.0.0}
 */
struct PermutationOperator {
    /// The partition, in source order. `P(i/jk)` gives `{{"i"}, {"j","k"}}`.
    std::vector<std::vector<std::string>> groups;

    /// The canonical `P(i/j,k)` spelling: slash between groups, comma inside one.
    [[nodiscard]] EINSUMS_EXPORT std::string render() const;

    /// Every letter this operator names, in group-concatenation order.
    [[nodiscard]] EINSUMS_EXPORT std::vector<std::string> letters() const;
};

/**
 * @brief One term of an expanded permutation operator product.
 *
 * @ref c_indices is the output index list relabelled by this term's
 * permutation, which is exactly the output spec of the permuted accumulation
 * that realizes the term: `permute(term.c_indices <- base.c_indices)` scaled by
 * @ref sign reads the base result and adds it at the transposed position.
 *
 * @versionadded{2.0.0}
 */
struct PermutationTerm {
    std::vector<std::string> c_indices; ///< Output order for this term.
    double                   sign{1.0}; ///< Parity of the representative, +1 or -1.
};

/**
 * @brief Expand a product of permutation operators over an output index list.
 *
 * @param[in] c_indices The output index list the operators permute.
 * @param[in] operators The operators, which must name disjoint letter sets.
 * @return One @ref PermutationTerm per term, the identity term FIRST.
 *         An empty @p operators list yields exactly one term, the identity.
 *
 * @par Choice of representative
 * Within each coset the member that displaces the fewest letters is taken, ties
 * broken by pairing the displaced letters between groups in their original
 * order. That reproduces the spelling the literature prints, so
 * `P(i/jk) f = f(ijk) - f(jik) - f(kji)` rather than an equivalent-on-valid-input
 * set with different parities. It is a CONVENTION chosen to match what a user
 * checks against by hand, not a correctness property: see the warning on
 * @ref PermutationOperator.
 *
 * With several operators the terms are the Cartesian product of the per-operator
 * expansions, signs multiplied, with the FIRST operator varying slowest.
 *
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::vector<PermutationTerm> expand_permutation_operators(std::vector<std::string> const         &c_indices,
                                                                                       std::vector<PermutationOperator> const &operators);

/**
 * @brief Result of parsing an einsum specification string.
 *
 * Contains the parsed index lists for the output tensor (C) and the two
 * input tensors (A and B). Also stores the original string for error messages.
 */
struct EINSUMS_EXPORT ParsedEinsumSpec {
    std::vector<std::string> c_indices;     ///< Output (C) indices
    std::vector<std::string> a_indices;     ///< First input (A) indices
    std::vector<std::string> b_indices;     ///< Second input (B) indices
    std::string              raw;           ///< Original specification string
    bool                     conj_a{false}; ///< A wrapped in conj(...) in the spec
    bool                     conj_b{false}; ///< B wrapped in conj(...) in the spec

    /// Permutation operators prefixing the term, in source order. Empty for
    /// every spec that does not write one, which is every spec written before
    /// they existed.
    /// @see PermutationOperator, expand_permutation_operators
    /// @versionadded{2.0.0}
    std::vector<PermutationOperator> operators;

    /**
     * @brief The canonical `"c <- a ; b"` spelling of these index lists.
     *
     * @ref raw holds whatever string the spec was parsed from, which for a spec a
     * pass or the IR loader assembled from index lists is nothing at all. Every
     * such producer used to format the join by hand, and the spelling has to agree
     * across all of them: it is what an execute-time diagnostic quotes, and the IR
     * round-trip goldens compare byte for byte.
     *
     * Multi-character indices are comma-separated, which parses back to the same
     * lists under either notation.
     * @versionadded{2.0.0}
     */
    [[nodiscard]] std::string render() const;

    /// Compute link indices (in both A and B, not in C).
    [[nodiscard]] std::vector<std::string> link_indices() const;

    /// Compute target indices (unique indices in C).
    [[nodiscard]] std::vector<std::string> target_indices() const;
};

/**
 * @brief Where an operand's link (contracted) indices sit in its index list.
 *
 * A GEMM reads its operands as flat matrices, A as (M,K) and B as (K,N), so an
 * operand can be handed to BLAS without first being physically transposed only
 * when its link indices form a contiguous block at one END of its index list.
 * Which end tells you the transpose flag: an A whose links are a prefix is
 * stored (K,M) and needs `transA`, a B whose links are a suffix is stored (N,K)
 * and needs `transB`.
 *
 * Both flags are true when an operand is entirely link indices or entirely
 * target indices, and the caller picks whichever reading needs no transpose.
 * Both false means the links are interleaved with the targets, which no
 * transpose flag can fix - that operand needs a real permute first.
 */
struct LinkPlacement {
    bool prefix{false}; ///< The link indices occupy exactly the leading positions of the list.
    bool suffix{false}; ///< The link indices occupy exactly the trailing positions of the list.

    /// True when neither end holds the whole link block.
    [[nodiscard]] bool split() const { return !prefix && !suffix; }
};

/**
 * @brief Classify @p indices by where the members of @p link_indices sit.
 * @see LinkPlacement
 */
[[nodiscard]] EINSUMS_EXPORT LinkPlacement link_placement(std::vector<std::string> const &indices,
                                                          std::vector<std::string> const &link_indices);

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
           c == '>' || c == ' ' || c == '\t' || c == '(' || c == ')' || // ( ) for conj(...) and P(...)
           c == '/';                                                    // group separator inside P(...)
}

/// @brief Strip the leading run of ``P(...)`` operators off a term.
///
/// Shared by the @c consteval index counter and the structural validators, both
/// of which have to see the operand list the way the runtime parser will. The
/// input is NOT whitespace-stripped, so leading blanks are skipped between
/// operators the way the runtime parser's pre-stripped input has none.
///
/// Returns @p s unchanged when it does not start with an operator, and stops at
/// the first unterminated one so the validator can reject it separately.
constexpr std::string_view strip_permutation_operators(std::string_view s) {
    while (true) {
        std::size_t begin = 0;
        while (begin < s.size() && (s[begin] == ' ' || s[begin] == '\t')) {
            ++begin;
        }
        std::string_view const rest = s.substr(begin);
        if (rest.size() < 2 || rest[0] != 'P' || rest[1] != '(') {
            return s;
        }
        std::size_t const close = rest.find(')');
        if (close == std::string_view::npos) {
            return s; // unterminated; validate_* reports it
        }
        s = rest.substr(close + 1);
    }
}

} // namespace detail

// ─── Compile-time index counter ─────────────────────────────────────────────
//
// Used by cg::einsum to validate that the rank of each typed tensor operand
// matches the number of indices the spec asks for. Runs at consteval time
// from EinsumFormatString's literal ctor so the resulting counts fold to
// compile-time constants at every well-typed callsite. Zeroed for runtime-
// constructed strings, the dispatcher then skips the rank check, matching
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
    // A ``P(...)`` prefix names OUTPUT letters, not operand slots, so it must go
    // before anything is counted. Leaving it in counted its letters as operand
    // indices, which left ``counts.known`` true and made cg::einsum reject a
    // correctly-ranked operand at compile time, pointing at the wrong thing.
    s = strip_permutation_operators(s);

    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
        s.remove_suffix(1);
    if (s.empty())
        return 0;

    // A ``conj(...)`` wrapper counts as just the indices it encloses.
    if (s.size() >= 6 && s.starts_with("conj(") && s.back() == ')')
        s = s.substr(5, s.size() - 6);

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
/// Accepts both arrow forms: ``"C <- A ; B"`` and ``"A ; B -> C"``.
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

namespace detail {

/// @brief Structural check on the ``conj(...)`` and ``P(...)`` wrappers.
///
/// Character-level only: parentheses balance, never nest, and never enclose
/// nothing. Everything that needs the index lists (a letter that is not an
/// output index, overlapping groups, a group of one) is checked by the runtime
/// parser, which can name the offender. This one only has a string literal to
/// throw.
constexpr bool wrappers_well_formed(std::string_view spec) {
    int         depth = 0;
    std::size_t open  = 0;
    for (std::size_t i = 0; i < spec.size(); ++i) {
        if (spec[i] == '(') {
            if (depth != 0) {
                return false; // conj(P(...)) and friends: no nesting
            }
            depth = 1;
            open  = i;
        } else if (spec[i] == ')') {
            if (depth != 1 || i == open + 1) {
                return false; // unbalanced, or an empty P() / conj()
            }
            depth = 0;
        }
    }
    return depth == 0;
}

} // namespace detail

constexpr bool validate_einsum_spec(std::string_view spec) {
    // Check all characters are valid
    for (char const c : spec) {
        if (!detail::is_einsum_char(c))
            return false;
    }

    if (!detail::wrappers_well_formed(spec))
        return false;

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
 * cg::einsum(EinsumFormatString("ij <- ik ; kj"), &C, A, B);  // OK, validated at compile time
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
     * pybind11 keeps the converted std::string alive for the duration of
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

    /// Permutation operators prefixing the term, in source order.
    /// @see ParsedEinsumSpec::operators
    /// @versionadded{2.0.0}
    std::vector<PermutationOperator> operators;

    /// The canonical `"c <- a"` spelling of these index lists.
    /// @see ParsedEinsumSpec::render
    /// @versionadded{2.0.0}
    [[nodiscard]] EINSUMS_EXPORT std::string render() const;
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

    if (!detail::wrappers_well_formed(spec))
        return false;

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

EINSUMS_NAMESPACE_END(compute_graph)
