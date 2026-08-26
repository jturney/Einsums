//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file SymbolicCost.hpp
 * @brief Cost as a polynomial in index-space scales, and a total order on those polynomials.
 *
 * @ref einsums::compute_graph::CostModel answers "how many microseconds will this GEMM take on this
 * device" from measured data. A structural pass needs a different question answered: "which of these
 * two mathematically equivalent forms is cheaper for the whole family of problems", and it needs the
 * answer before any extent is bound. That is what this layer is: a multivariate polynomial in space
 * scales (2 o^2 v^2 x flops, o v^3 words of traffic) plus a comparison that always decides.
 *
 * The two models compose rather than compete. A structural pass decides WHAT to compute with
 * @ref einsums::compute_graph::SymbolicCost, then a tuning pass decides HOW to run it with
 * @ref einsums::compute_graph::CostModel.
 *
 * @par Why the comparison must be total
 * A search that cannot rank two candidates has to break the tie somehow, and a tie broken by
 * container iteration order makes the optimizer nondeterministic, which in turn makes every
 * measurement against it noise. The fallback chain here is therefore fixed and exhaustive, and its
 * last rung is a lexicographic comparison on the canonical form: arbitrary on purpose. What matters
 * is not which candidate it picks but that it picks the same one on every run.
 *
 * @par Determinism rules this file obeys
 * Nothing here iterates an unordered container. Every sequence is sorted explicitly, by a key
 * documented on the operation that uses it.
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

struct EinsumDescriptor;

/**
 * @brief One variable of a cost polynomial: either an annotated space or an unannotated letter.
 *
 * A variable stands for the EXTENT of something, never for a tensor. Two flavours exist because
 * annotation is partial in practice:
 *
 * - A @b space variable names a @ref SpaceId, so its scale symbol, its typical extent and its
 *   declared relations to other spaces are all recoverable from a @ref SpaceRegistry.
 * - An @b anonymous variable names an index LETTER that met no annotated slot. It supports no
 *   registry query at all, which is exactly why a partially annotated program still yields usable
 *   (if weaker) comparisons rather than no comparison.
 *
 * @par Anonymous-variable scope
 * An anonymous variable is derived from the letter name and nothing else, so it is the same variable
 * in every process, every node and every polynomial that uses that letter. That is deliberate: the
 * comparison's last rung must pick the same candidate in every process, and a variable interned
 * against a per-call counter would depend on call order. The consequence is that an anonymous
 * variable carries NO claim that two nodes using letter @c i range over the same set. It means
 * "the letter i", nothing more, and a caller must not read space identity into it. Callers that need
 * that claim must annotate the slots.
 */
class EINSUMS_EXPORT SymbolicVar {
  public:
    /// @brief Which flavour of variable this is.
    enum class Kind : std::uint8_t {
        Space,     ///< Names a registered index space.
        Anonymous, ///< Names an unannotated index letter.
    };

    /// @brief Construct an invalid variable: a space variable holding an invalid @ref SpaceId.
    SymbolicVar() = default;

    /**
     * @brief Build the variable standing for a registered space's extent.
     * @param[in] id The space. An invalid id yields an invalid variable.
     * @return The space variable.
     */
    [[nodiscard]] static SymbolicVar space(SpaceId id) { return SymbolicVar{Kind::Space, id, std::string{}}; }

    /**
     * @brief Build the anonymous variable standing for an unannotated letter's extent.
     * @param[in] letter The index letter. Used verbatim, including a multi-character index name.
     * @return The anonymous variable.
     */
    [[nodiscard]] static SymbolicVar anonymous(std::string_view letter) {
        return SymbolicVar{Kind::Anonymous, SpaceId{}, std::string{letter}};
    }

    /// @brief The flavour of this variable.
    /// @return @ref Kind::Space or @ref Kind::Anonymous.
    [[nodiscard]] Kind kind() const noexcept { return _kind; }

    /// @brief Whether this variable names a registered space.
    /// @return True for a space variable, including an invalid one.
    [[nodiscard]] bool is_space() const noexcept { return _kind == Kind::Space; }

    /// @brief Whether this variable names an unannotated letter.
    /// @return True for an anonymous variable.
    [[nodiscard]] bool is_anonymous() const noexcept { return _kind == Kind::Anonymous; }

    /// @brief Whether this variable names something.
    /// @return False only for a default-constructed variable and for an anonymous one with an empty
    ///         letter.
    [[nodiscard]] bool valid() const noexcept { return is_space() ? _space.valid() : !_letter.empty(); }

    /// @brief The space this variable names.
    /// @return The space id, or an invalid id for an anonymous variable.
    [[nodiscard]] SpaceId space_id() const noexcept { return _space; }

    /// @brief The letter this variable names.
    /// @return The letter, or an empty string for a space variable.
    [[nodiscard]] std::string const &letter() const noexcept { return _letter; }

    /// @brief Compare two variables for identity.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when both flavour and payload match.
    [[nodiscard]] friend bool operator==(SymbolicVar const &lhs, SymbolicVar const &rhs) = default;

    /**
     * @brief The canonical order on variables.
     * @param[in] rhs Right operand.
     * @return Space variables before anonymous ones; space variables by @ref SpaceId, which is
     *         registration order in their registry; anonymous variables lexicographically by letter.
     *
     * This order is what makes the canonical form of a polynomial deterministic. It is stable across
     * processes for anonymous variables unconditionally, and for space variables as long as the
     * spaces are registered in the same order, which is the normal case because registration is
     * startup code.
     *
     * A member rather than a hidden friend, which every other spaceship in the
     * tree already is: a variable converts from nothing, so the two forms behave
     * identically here, and the member form is the one the documentation
     * extractor renders without mangling the operator's name.
     */
    [[nodiscard]] std::strong_ordering operator<=>(SymbolicVar const &rhs) const = default;

    /// @brief Hash so a variable can key an unordered container.
    /// @return A hash of flavour and payload.
    [[nodiscard]] std::size_t hash() const noexcept;

  private:
    /**
     * @brief Construct a variable from its parts.
     * @param[in] kind The flavour.
     * @param[in] id The space, for a space variable.
     * @param[in] letter The letter, for an anonymous variable.
     */
    SymbolicVar(Kind kind, SpaceId id, std::string letter) : _kind{kind}, _space{id}, _letter{std::move(letter)} {}

    Kind        _kind{Kind::Space};
    SpaceId     _space;
    std::string _letter;
};

/**
 * @brief One variable raised to one power inside a monomial.
 *
 * The exponent is signed because the type does not forbid a reciprocal, but every polynomial this
 * module builds uses positive exponents, and the scale-order rung of @ref compare abstains when it
 * meets a negative one.
 */
struct SymbolicFactor {
    SymbolicVar variable;    ///< The variable being raised.
    int         exponent{1}; ///< The power. Zero factors are dropped by canonicalization.

    /// @brief Compare two factors field by field.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when variable and exponent both match.
    [[nodiscard]] friend bool operator==(SymbolicFactor const &lhs, SymbolicFactor const &rhs) = default;
};

/**
 * @brief One term of a cost polynomial: a coefficient times a monomial.
 *
 * A term is CANONICAL when its factors are sorted ascending by @ref SymbolicVar's order, no variable
 * appears twice, and no exponent is zero. @ref SymbolicPoly canonicalizes every term it stores, so a
 * term obtained from @ref SymbolicPoly::terms is always canonical.
 */
struct EINSUMS_EXPORT SymbolicTerm {
    double                      coefficient{1.0}; ///< The numeric coefficient.
    std::vector<SymbolicFactor> factors;          ///< The monomial, canonical once stored in a polynomial.

    /// @brief Sort the factors, merge repeats and drop zero exponents.
    void canonicalize();

    /// @brief The sum of the exponents.
    /// @return The total degree, zero for a constant term.
    [[nodiscard]] int total_degree() const noexcept;

    /// @brief The power one variable is raised to in this term.
    /// @param[in] variable The variable to look for.
    /// @return The exponent, or zero when the variable does not appear.
    [[nodiscard]] int degree_in(SymbolicVar const &variable) const noexcept;

    /// @brief Compare two terms field by field.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when coefficients and factor lists match exactly.
    [[nodiscard]] friend bool operator==(SymbolicTerm const &lhs, SymbolicTerm const &rhs) = default;
};

/// @brief Resolves a variable to a numeric extent, or reports that it cannot.
using ExtentLookup = std::function<std::optional<double>(SymbolicVar const &)>;

/**
 * @brief A multivariate polynomial in index-space scales, always in canonical form.
 *
 * @par Canonical form
 * Terms are sorted ascending by their MONOMIAL KEY, like terms are merged, and a term whose
 * coefficient is exactly zero is dropped. The monomial key compares two monomials by
 *
 * 1. their variable sequences, element by element under @ref SymbolicVar's order, a prefix being
 *    smaller than what extends it, then
 * 2. their exponent sequences, element by element.
 *
 * Because like terms are merged, no two stored terms share a monomial key, so the key alone orders
 * them. Canonical form is load-bearing: @ref operator== is structural equality on it, and the last
 * rung of @ref compare is a lexicographic walk over it.
 */
class EINSUMS_EXPORT SymbolicPoly {
  public:
    /// @brief Construct the zero polynomial, which has no terms.
    SymbolicPoly() = default;

    /// @brief The zero polynomial.
    /// @return A polynomial with no terms.
    [[nodiscard]] static SymbolicPoly zero() { return SymbolicPoly{}; }

    /**
     * @brief A constant polynomial.
     * @param[in] value The constant. Exactly zero yields the zero polynomial.
     * @return The polynomial.
     */
    [[nodiscard]] static SymbolicPoly constant(double value);

    /**
     * @brief A single power of a single variable, with coefficient one.
     * @param[in] variable The variable.
     * @param[in] exponent The power. Zero yields the constant one.
     * @return The polynomial.
     */
    [[nodiscard]] static SymbolicPoly variable(SymbolicVar variable, int exponent = 1);

    /**
     * @brief Build a polynomial from arbitrary terms, canonicalizing them.
     * @param[in] terms The terms, in any order, with any repeats.
     * @return The canonical polynomial.
     */
    [[nodiscard]] static SymbolicPoly from_terms(std::vector<SymbolicTerm> terms);

    /// @brief The terms, in canonical order.
    /// @return The stored terms. Empty for the zero polynomial.
    [[nodiscard]] std::vector<SymbolicTerm> const &terms() const noexcept { return _terms; }

    /// @brief Whether this is the zero polynomial.
    /// @return True when there are no terms.
    [[nodiscard]] bool is_zero() const noexcept { return _terms.empty(); }

    /// @brief The largest total degree over the terms.
    /// @return The degree, zero for the zero polynomial and for a constant.
    [[nodiscard]] int total_degree() const noexcept;

    /// @brief The largest exponent one variable carries in any term.
    /// @param[in] variable The variable to look for.
    /// @return The degree in that variable, zero when it never appears.
    [[nodiscard]] int degree_in(SymbolicVar const &variable) const noexcept;

    /// @brief Every variable appearing in any term.
    /// @return The variables, sorted ascending and deduplicated.
    [[nodiscard]] std::vector<SymbolicVar> variables() const;

    /**
     * @brief Substitute numeric extents and evaluate.
     * @param[in] extents Resolver for each variable.
     * @return The value, or an empty optional as soon as @p extents cannot resolve some variable.
     *         The zero polynomial evaluates to zero without consulting @p extents.
     */
    [[nodiscard]] std::optional<double> evaluate(ExtentLookup const &extents) const;

    /// @brief Add another polynomial in place.
    /// @param[in] other The addend.
    /// @return This polynomial.
    SymbolicPoly &operator+=(SymbolicPoly const &other);

    /// @brief Multiply by another polynomial in place.
    /// @param[in] other The multiplier.
    /// @return This polynomial.
    SymbolicPoly &operator*=(SymbolicPoly const &other);

    /// @brief Scale every coefficient in place.
    /// @param[in] factor The scalar. Exactly zero yields the zero polynomial.
    /// @return This polynomial.
    SymbolicPoly &operator*=(double factor);

    /// @brief Add two polynomials.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return The canonical sum.
    [[nodiscard]] friend SymbolicPoly operator+(SymbolicPoly lhs, SymbolicPoly const &rhs) {
        lhs += rhs;
        return lhs;
    }

    /// @brief Multiply two polynomials.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return The canonical product.
    [[nodiscard]] friend SymbolicPoly operator*(SymbolicPoly lhs, SymbolicPoly const &rhs) {
        lhs *= rhs;
        return lhs;
    }

    /// @brief Scale a polynomial.
    /// @param[in] lhs The polynomial.
    /// @param[in] rhs The scalar.
    /// @return The scaled polynomial.
    [[nodiscard]] friend SymbolicPoly operator*(SymbolicPoly lhs, double rhs) {
        lhs *= rhs;
        return lhs;
    }

    /// @brief Scale a polynomial.
    /// @param[in] lhs The scalar.
    /// @param[in] rhs The polynomial.
    /// @return The scaled polynomial.
    [[nodiscard]] friend SymbolicPoly operator*(double lhs, SymbolicPoly rhs) {
        rhs *= lhs;
        return rhs;
    }

    /// @brief Structural equality on the canonical form.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when the term sequences match exactly, coefficients included.
    [[nodiscard]] friend bool operator==(SymbolicPoly const &lhs, SymbolicPoly const &rhs) = default;

    /**
     * @brief Render the polynomial, for reports and test failures.
     * @param[in] registry Registry used to resolve a space variable to its scale symbol. May be
     *            null, in which case every space variable renders as its id.
     * @return Something like the string 2*o^2*v^2*?a. Terms appear in canonical order, joined by a
     *         plus sign; a coefficient of one is elided when the term has factors; an anonymous
     *         variable renders as a question mark followed by its letter; a space variable with no
     *         resolvable scale symbol renders as its name, or as the letter s followed by its id
     *         when even that is unavailable. The zero polynomial renders as a single zero.
     */
    [[nodiscard]] std::string to_string(SpaceRegistry const *registry = nullptr) const;

  private:
    /// Canonical terms: sorted by monomial key, merged, no zero coefficients.
    std::vector<SymbolicTerm> _terms;
};

/// @brief Which rung of the fallback chain decided a comparison.
enum class CompareRung : std::uint8_t {
    ScaleOrder,    ///< Asymptotic dominance under the registry's declared scale order.
    TypicalExtent, ///< Numeric, after substituting each space's advisory typical extent.
    BoundExtent,   ///< Numeric, after substituting extents supplied by the caller.
    Lexicographic, ///< The final arbitrary-but-deterministic walk over the canonical form.
};

/// @brief Name of a @ref CompareRung value.
/// @param[in] rung The rung to name.
/// @return "ScaleOrder", "TypicalExtent", "BoundExtent" or "Lexicographic".
[[nodiscard]] EINSUMS_EXPORT std::string_view compare_rung_name(CompareRung rung) noexcept;

/**
 * @brief Everything @ref compare is allowed to consult.
 *
 * An aggregate on purpose, so a caller writes only the rungs it can feed.
 */
struct ComparisonContext {
    /// Registry backing the scale-order and typical-extent rungs. Null disables both.
    SpaceRegistry const *registry{nullptr};

    /// Per-variable bound extents for the third rung. Empty disables it. It is consulted only when
    /// it resolves EVERY variable of both polynomials.
    ExtentLookup bound_extent;

    /// Relative tolerance below which two substituted values count as equal, which makes the numeric
    /// rung indecisive rather than deciding. Two values are equal when the magnitude of their
    /// difference is at most this tolerance times the largest of their two magnitudes and one.
    double relative_tolerance{1e-12};
};

/// @brief The verdict of @ref compare_explain, with the rung that produced it.
struct PolyComparison {
    std::strong_ordering order{std::strong_ordering::equal}; ///< The ordering.
    CompareRung          rung{CompareRung::Lexicographic};   ///< The rung that decided.
};

/**
 * @brief Order two cost polynomials, always, and report which rung decided.
 * @param[in] lhs Left operand.
 * @param[in] rhs Right operand.
 * @param[in] ctx What the comparison may consult.
 * @return The ordering and the deciding rung.
 *
 * @par The fallback chain
 * Each rung runs only when the previous one could not decide, and only a STRICT ordering counts as
 * a decision. A rung that finds the two equal hands the question down rather than answering it,
 * which is what makes @c equal come out of the last rung alone, and therefore makes
 * @c compare(a,b) == equal exactly equivalent to @c a == b.
 *
 * 1. @ref CompareRung::ScaleOrder, asymptotic dominance. See below.
 * 2. @ref CompareRung::TypicalExtent, numeric comparison after substituting @ref
 *    IndexSpace::typical_extent for every variable. A polynomial substitutes only when every
 *    variable in it is a space variable with a positive typical extent, so a single anonymous
 *    variable blocks it. When both substitute, their values are compared. When only ONE substitutes,
 *    that one is ranked below the other, which is a deliberate split into a numeric class and a
 *    structural one: interleaving a numeric verdict with the structural verdict the last rung gives
 *    an unsubstitutable polynomial is exactly what would let the chain contradict itself. When
 *    neither substitutes the rung passes.
 * 3. @ref CompareRung::BoundExtent, the same with @ref ComparisonContext::bound_extent, on the
 *    polynomials the previous rung left together.
 * 4. @ref CompareRung::Lexicographic, a walk over the canonical form: term by term, comparing the
 *    monomial key first and then the coefficient, with a shorter term list ordering before a longer
 *    one that starts the same way. This rung always decides, and returns @c equal only for
 *    identical canonical forms.
 *
 * @par The dominance rule of rung 1
 * Extents are assumed to be at least one, which is what lets extra factors only ever increase a
 * value. For monomials, @c m1 <= @c m2 when each variable of @c m1, repeated by its exponent, can be
 * matched INJECTIVELY to a distinct variable occurrence of @c m2 that is the same variable or is
 * declared smaller by @ref SpaceRegistry::is_less. The matching is a complete bipartite matching, so
 * the rule misses nothing expressible in those terms. For polynomials, @c a <= @c b when every term
 * of @c a is dominated by some term of @c b in that sense; @c a is then ranked below @c b when
 * @c a <= @c b holds and @c b <= @c a does not.
 *
 * The design's own example works out: with @c o declared smaller than @c v, the monomial o^3 v^3
 * sorts below o^2 v^4, because the occurrence list o,o,o,v,v,v matches into o,o,v,v,v,v but not
 * the reverse.
 *
 * @par Limits of the dominance rule
 * - It abstains when either polynomial carries a non-positive coefficient or a negative exponent,
 *   because a subtraction can cancel a leading term and the rule has no way to see that.
 * - It ignores coefficients entirely, so 1000 o ranks below @c v whenever @c o is declared below
 *   @c v, however the extents actually compare. That is what asymptotic means, and it is also
 *   why the numeric rungs sit below this one rather than above it.
 * - It uses only DECLARED relations. Two spaces the registry cannot relate make it indecisive, and
 *   an anonymous variable relates to nothing but itself.
 * - It does not consult @ref GrowthClass. A growth exponent describes how an extent scales with
 *   system size, which is a different question from which of two extents is larger.
 * - A space whose extent is exactly one is a degenerate case where the rule reports a strict
 *   ordering between quantities that are in fact equal. Harmless for ranking, worth knowing.
 * - It abstains on a monomial that expands to more than 64 variable occurrences, which keeps a
 *   pathological exponent from turning a comparison into a large matching problem.
 *
 * @par Order properties
 * The result is antisymmetric and complete by construction, and @c equal holds only for identical
 * canonical forms, so the comparison is usable directly as a strict weak ordering.
 *
 * Transitivity needs the rungs not to contradict each other, and the class split at rung 2 is what
 * buys most of that: a partially annotated program cannot produce a triple where a numeric verdict
 * and a structural one disagree, which is the failure a mixed set of polynomials would otherwise hit
 * immediately. Note that the dominance rung can never rank an unsubstitutable polynomial below a
 * substitutable one, because a variable matches only itself or a declared-smaller space, so
 * dominance agrees with the class split rather than fighting it. Three conditions on the inputs
 * close the rest, and all three are properties of the registry and the caller, not of the
 * comparison:
 *
 * - Typical extents respect the declared scale order.
 * - Either every space carries a typical extent, or none of the spaces a declared order relates
 *   does. A space that is declared small but carries no extent sits in the structural class while
 *   the space above it sits in the numeric one, and dominance then disagrees with the split.
 * - No coefficient is large enough to invert a declared scale relation at those extents, so that
 *   the asymptotic rung and the numeric one rank the same pairs the same way. Cost polynomials
 *   built by @ref symbolic_cost_for carry coefficients of one and two, which is nowhere near enough
 *   to matter.
 *
 * A @ref ComparisonContext::bound_extent table wants the same discipline: it should resolve whole
 * polynomials rather than an arbitrary subset of variables. Violating any of these is a
 * configuration bug, and the cure is to fix the configuration, because a rung cannot both honour a
 * declared order and contradict it.
 */
[[nodiscard]] EINSUMS_EXPORT PolyComparison compare_explain(SymbolicPoly const &lhs, SymbolicPoly const &rhs, ComparisonContext const &ctx);

/**
 * @brief Order two cost polynomials, always.
 * @param[in] lhs Left operand.
 * @param[in] rhs Right operand.
 * @param[in] ctx What the comparison may consult.
 * @return The ordering. See @ref compare_explain for the rules.
 */
[[nodiscard]] EINSUMS_EXPORT std::strong_ordering compare(SymbolicPoly const &lhs, SymbolicPoly const &rhs, ComparisonContext const &ctx);

/// @brief Which polynomial of a @ref SymbolicCost a comparison came down to.
enum class CostComponent : std::uint8_t {
    Flops,    ///< Arithmetic, the primary key.
    Traffic,  ///< Words moved, the first tie-break.
    Resident, ///< Words live, the last tie-break.
};

/// @brief Name of a @ref CostComponent value.
/// @param[in] component The component to name.
/// @return "Flops", "Traffic" or "Resident".
[[nodiscard]] EINSUMS_EXPORT std::string_view cost_component_name(CostComponent component) noexcept;

/**
 * @brief The symbolic cost of one operation: three polynomials in index-space scales.
 *
 * @par What "resident" approximates
 * The live footprint of an operation is really the MAXIMUM over its operands, and a maximum of
 * polynomials is not a polynomial. The first cut taken here is the SUM of the operand sizes, which
 * is the same quantity as @ref traffic for a node that touches each operand once. It is an upper
 * bound within a factor of the operand count, it orders the same way as the maximum whenever one
 * operand dominates the others (the usual case in a contraction), and it is wrong as an absolute
 * memory figure. Treat it as a ranking key, not as a byte count.
 */
struct EINSUMS_EXPORT SymbolicCost {
    SymbolicPoly flops;    ///< Arithmetic operations.
    SymbolicPoly traffic;  ///< Words moved between memory and the kernel.
    SymbolicPoly resident; ///< Words live at once. See the class documentation for the caveat.

    /// @brief Structural equality on all three polynomials.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when all three match exactly.
    [[nodiscard]] friend bool operator==(SymbolicCost const &lhs, SymbolicCost const &rhs) = default;

    /// @brief Render all three polynomials.
    /// @param[in] registry Registry used to resolve scale symbols. May be null.
    /// @return The three polynomials, each labelled with its name and separated by commas.
    [[nodiscard]] std::string to_string(SpaceRegistry const *registry = nullptr) const;
};

/// @brief The verdict of the @ref SymbolicCost overload of @ref compare_explain.
struct CostComparison {
    std::strong_ordering order{std::strong_ordering::equal}; ///< The ordering.
    CompareRung          rung{CompareRung::Lexicographic};   ///< The rung that decided.
    CostComponent        component{CostComponent::Flops};    ///< The polynomial that decided.
};

/**
 * @brief Order two cost bundles, always, and report what decided.
 * @param[in] lhs Left operand.
 * @param[in] rhs Right operand.
 * @param[in] ctx What the comparison may consult.
 * @return The ordering, the deciding rung and the deciding component.
 *
 * Flops are the primary key: the whole fallback chain of @ref compare_explain runs on @ref
 * SymbolicCost::flops first, and because that chain returns @c equal only for identical canonical
 * forms, traffic is consulted only when the two flop polynomials are literally the same polynomial.
 * @ref SymbolicCost::resident breaks the remaining tie on the same terms. Two bundles compare equal
 * only when all three polynomials are identical.
 */
[[nodiscard]] EINSUMS_EXPORT CostComparison compare_explain(SymbolicCost const &lhs, SymbolicCost const &rhs, ComparisonContext const &ctx);

/**
 * @brief Order two cost bundles, always.
 * @param[in] lhs Left operand.
 * @param[in] rhs Right operand.
 * @param[in] ctx What the comparison may consult.
 * @return The ordering. See the @ref SymbolicCost overload of @ref compare_explain for the rules.
 */
[[nodiscard]] EINSUMS_EXPORT std::strong_ordering compare(SymbolicCost const &lhs, SymbolicCost const &rhs, ComparisonContext const &ctx);

/**
 * @brief Resolves an index letter of one contraction to the variable standing for its extent.
 *
 * Built from a node's @ref EinsumDescriptor::letter_spaces, which is the per-node resolution of
 * letters to spaces, and which is partial or empty for a partially annotated or unannotated program.
 * An annotated letter yields its space variable; any other letter yields
 * @ref SymbolicVar::anonymous of the letter itself, with the scoping documented on @ref SymbolicVar.
 */
class EINSUMS_EXPORT LetterVars {
  public:
    /// @brief Construct a lookup that annotates nothing, so every letter is anonymous.
    LetterVars() = default;

    /**
     * @brief Construct from a node's letter-to-space map.
     * @param[in] letter_spaces The map, in any order. Stored sorted by letter, so lookup is a binary
     *            search and the result does not depend on the input order.
     */
    explicit LetterVars(std::vector<std::pair<std::string, SpaceId>> const &letter_spaces);

    /**
     * @brief The variable standing for one letter's extent.
     * @param[in] letter The index letter.
     * @return The letter's space variable when it is annotated, its anonymous variable otherwise.
     */
    [[nodiscard]] SymbolicVar var_for(std::string_view letter) const;

    /// @brief How many letters carry an annotation.
    /// @return The number of annotated letters.
    [[nodiscard]] std::size_t annotated_count() const noexcept { return _letter_spaces.size(); }

  private:
    /// Annotated letters, sorted ascending by letter.
    std::vector<std::pair<std::string, SpaceId>> _letter_spaces;
};

/**
 * @brief The letter lookup for one einsum node.
 * @param[in] desc The node's descriptor.
 * @return A lookup over its @ref EinsumDescriptor::letter_spaces.
 */
[[nodiscard]] EINSUMS_EXPORT LetterVars letter_vars_for(EinsumDescriptor const &desc);

/**
 * @brief The symbolic element count of a tensor with the given index list.
 * @param[in] indices The index letters, in tensor order, possibly with repeats.
 * @param[in] vars The letter lookup.
 * @return The product of the DISTINCT letters' variables, with coefficient one. An empty index list
 *         yields the constant one, which is right for a scalar.
 *
 * A repeated letter counts ONCE, so an operand indexed i,i contributes i rather than i^2. The quantity
 * measured is therefore the number of distinct elements the operation touches, which is what matches
 * the loop structure of a diagonal access, not the allocated footprint of the tensor holding them.
 * For an intermediate, whose letters are distinct by construction, the two agree.
 */
[[nodiscard]] EINSUMS_EXPORT SymbolicPoly symbolic_size_for(std::vector<std::string> const &indices, LetterVars const &vars);

/**
 * @brief The symbolic cost of one einsum node, derived mechanically from its contraction spec.
 * @param[in] desc The node's descriptor.
 * @return Its flops, traffic and resident polynomials.
 *
 * - @ref SymbolicCost::flops is @c 2 times the product of the distinct letters of the loop space
 *   (@c ContractionSpec::all_indices), the two counting the multiply and the add. A letter repeated
 *   within an operand appears once, matching the loop nest rather than the index list.
 * - @ref SymbolicCost::traffic is the sum of the three operand sizes, C first, then A and B, each
 *   from @ref symbolic_size_for. Every operand is counted once, which is what a streaming kernel
 *   costs and what a blocked one approaches.
 * - @ref SymbolicCost::resident equals the traffic polynomial. See @ref SymbolicCost for why that
 *   approximation is used and what it costs.
 *
 * Letters absent from @ref EinsumDescriptor::letter_spaces get anonymous variables, so an
 * unannotated node still yields a comparable cost, just one the scale-order rung cannot rank.
 */
[[nodiscard]] EINSUMS_EXPORT SymbolicCost symbolic_cost_for(EinsumDescriptor const &desc);

EINSUMS_NAMESPACE_END(compute_graph)

namespace std {

/// @brief Hash support so a @ref einsums::compute_graph::SymbolicVar can key an unordered container.
template <>
struct hash<::einsums::compute_graph::SymbolicVar> {
    /// @brief Hash a symbolic variable.
    /// @param[in] variable The variable to hash.
    /// @return Its hash.
    [[nodiscard]] std::size_t operator()(::einsums::compute_graph::SymbolicVar const &variable) const noexcept { return variable.hash(); }
};

} // namespace std
