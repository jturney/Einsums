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
#include <cstdint>
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

    /// Compute target indices: C's unique indices, in the order C first names them. That is the
    /// order PackedGemm's target space takes them in, so a descriptor built from this list and the
    /// spec the dispatcher hands PackedGemm describe one contraction the same way.
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
 * @brief What one index letter does in a contraction ``C <- A ; B``.
 *
 * Every letter of a contraction plays exactly one of these parts, and the passes that reason in
 * GEMM terms (batch, M, N, K) differ only in what they do with each. They used to classify the
 * letters themselves, and differed on the lone letters: one counted them as reduced, one declined.
 */
enum class IndexRole : std::uint8_t {
    Batch,     ///< In A, B and C: the contraction repeats along it.
    Link,      ///< In A and B, not in C: summed over as a matrix product's K.
    AFree,     ///< In A and C only: the M side of a matrix product.
    BFree,     ///< In B and C only: the N side of a matrix product.
    ALone,     ///< In A only: summed within A before the product.
    BLone,     ///< In B only: summed within B before the product.
    OutputOnly ///< In C only. The parser rejects such a spec; listed so the classification is total.
};

/**
 * @brief The part @p letter plays in the contraction whose index lists are @p c, @p a and @p b.
 * @param[in] letter The index letter to classify.
 * @param[in] c The output's indices.
 * @param[in] a The first operand's indices.
 * @param[in] b The second operand's indices.
 * @return Its role. A letter in none of the lists is reported as @ref IndexRole::OutputOnly.
 */
[[nodiscard]] EINSUMS_EXPORT IndexRole index_role(std::string_view letter, std::vector<std::string> const &c,
                                                  std::vector<std::string> const &a, std::vector<std::string> const &b) noexcept;

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

/// The structural check both validators share: every character is one a spec may use, the wrappers
/// are well formed, there is exactly one arrow, and there are @p semicolons operand separators.
/// @p underscore admits ``_`` in index names, which a permute spec allows and an einsum spec does not.
constexpr bool validate_spec_structure(std::string_view spec, std::size_t semicolons, bool underscore = false) {
    if (!wrappers_well_formed(spec)) {
        return false;
    }
    std::size_t arrows = 0;
    std::size_t semis  = 0;
    for (std::size_t i = 0; i < spec.size(); ++i) {
        if (!is_einsum_char(spec[i]) && !(underscore && spec[i] == '_')) {
            return false;
        }
        std::string_view const rest = spec.substr(i);
        arrows += (rest.starts_with("<-") || rest.starts_with("->")) ? 1 : 0;
        semis += spec[i] == ';' ? 1 : 0;
    }
    return arrows == 1 && semis == semicolons;
}

} // namespace detail

constexpr bool validate_einsum_spec(std::string_view spec) {
    return detail::validate_spec_structure(spec, 1);
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

    /**
     * @brief Construct from a string literal with compile-time validation.
     *
     * The `consteval` keyword ensures this constructor runs at compile time
     * for string literals. If the string is invalid, a compile error is produced.
     *
     * @param[in] s The einsum specification string literal.
     */
    template <size_t N>
    consteval EinsumFormatString(char const (&s)[N]) : str(s, N - 1) { // NOLINT
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
    EinsumFormatString(std::string_view s) : str(s) {} // NOLINT(google-explicit-constructor)

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
 * Checks structural validity: presence of exactly one arrow, no semicolons. Index names may use
 * ``_`` (``mu_1,nu_1 <- nu_1,mu_1``), as tensor_permute's own specs may.
 */
constexpr bool validate_permute_spec(std::string_view spec) {
    return detail::validate_spec_structure(spec, 0, /*underscore=*/true);
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
