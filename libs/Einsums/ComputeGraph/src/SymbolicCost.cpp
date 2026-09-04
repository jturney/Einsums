//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraphTypes/EnumNames.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Sentinel for "no match" in the dominance matching below.
constexpr std::size_t unmatched = static_cast<std::size_t>(-1);

/// Largest expanded monomial the dominance rung will try to match. Beyond this it abstains, which
/// keeps a pathological exponent from turning a comparison into a matching problem.
constexpr std::size_t max_dominance_factors = 64;

/**
 * @brief The canonical order on two monomials.
 * @param[in] lhs Left monomial, canonical.
 * @param[in] rhs Right monomial, canonical.
 * @return Variable sequences first, element by element with a prefix ordering before what extends
 *         it, then the exponent sequences.
 */
[[nodiscard]] std::strong_ordering compare_monomials(std::vector<SymbolicFactor> const &lhs, std::vector<SymbolicFactor> const &rhs) {
    std::size_t const shared = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < shared; ++i) {
        if (auto const order = lhs[i].variable <=> rhs[i].variable; order != 0) {
            return order;
        }
    }
    if (auto const order = lhs.size() <=> rhs.size(); order != 0) {
        return order;
    }
    for (std::size_t i = 0; i < shared; ++i) {
        if (auto const order = lhs[i].exponent <=> rhs[i].exponent; order != 0) {
            return order;
        }
    }
    return std::strong_ordering::equal;
}

/**
 * @brief Total order on two doubles, for the lexicographic rung.
 * @param[in] lhs Left operand.
 * @param[in] rhs Right operand.
 * @return The obvious ordering. Coefficients are never NaN in a polynomial this module builds.
 */
[[nodiscard]] std::strong_ordering compare_coefficients(double lhs, double rhs) {
    if (lhs < rhs) {
        return std::strong_ordering::less;
    }
    if (rhs < lhs) {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}

/**
 * @brief Sort, merge and prune a term list into canonical form.
 * @param[in,out] terms The terms to canonicalize in place.
 */
void canonicalize_terms(std::vector<SymbolicTerm> &terms) {
    for (auto &term : terms) {
        term.canonicalize();
    }
    std::ranges::sort(terms,
                      [](SymbolicTerm const &lhs, SymbolicTerm const &rhs) { return compare_monomials(lhs.factors, rhs.factors) < 0; });

    std::vector<SymbolicTerm> merged;
    merged.reserve(terms.size());
    for (auto &term : terms) {
        if (!merged.empty() && merged.back().factors == term.factors) {
            merged.back().coefficient += term.coefficient;
        } else {
            merged.push_back(std::move(term));
        }
    }
    std::erase_if(merged, [](SymbolicTerm const &term) { return term.coefficient == 0.0; });
    terms = std::move(merged);
}

/**
 * @brief Raise a base to a signed integer power without recursion.
 * @param[in] base The base.
 * @param[in] exponent The power, possibly negative.
 * @return @p base raised to @p exponent, by squaring.
 */
[[nodiscard]] double integer_power(double base, int exponent) {
    bool const   invert    = exponent < 0;
    std::int64_t remaining = invert ? -static_cast<std::int64_t>(exponent) : static_cast<std::int64_t>(exponent);
    double       factor    = base;
    double       result    = 1.0;
    while (remaining > 0) {
        if ((remaining & 1) != 0) {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1;
    }
    return invert ? 1.0 / result : result;
}

/**
 * @brief Whether one variable's extent is provably no larger than another's.
 * @param[in] small The candidate smaller variable.
 * @param[in] large The candidate larger variable.
 * @param[in] registry The registry holding the declared scale order. May be null.
 * @return True when the two are the same variable, or the registry declares @p small below
 *         @p large. Anonymous variables relate to nothing but themselves.
 */
[[nodiscard]] bool variable_dominated_by(SymbolicVar const &small, SymbolicVar const &large, SpaceRegistry const *registry) {
    if (small == large) {
        return true;
    }
    if (registry == nullptr || !small.is_space() || !large.is_space()) {
        return false;
    }
    SpaceId const lower = small.space_id();
    SpaceId const upper = large.space_id();
    if (!lower.valid() || !upper.valid()) {
        return false;
    }
    std::size_t const count = registry->size();
    if (lower.value() >= count || upper.value() >= count) {
        return false;
    }
    return registry->is_less(lower, upper) == Tristate::Yes;
}

/**
 * @brief Expand a monomial into one entry per factor occurrence.
 * @param[in] factors The canonical monomial.
 * @return The expanded variables, or an empty optional for a negative exponent or an expansion
 *         larger than @ref max_dominance_factors.
 */
[[nodiscard]] std::optional<std::vector<SymbolicVar>> expand_monomial(std::vector<SymbolicFactor> const &factors) {
    std::vector<SymbolicVar> expanded;
    for (auto const &factor : factors) {
        if (factor.exponent < 0) {
            return std::nullopt;
        }
        if (expanded.size() + static_cast<std::size_t>(factor.exponent) > max_dominance_factors) {
            return std::nullopt;
        }
        for (int i = 0; i < factor.exponent; ++i) {
            expanded.push_back(factor.variable);
        }
    }
    return expanded;
}

/**
 * @brief Expand every monomial of a polynomial.
 * @param[in] poly The polynomial.
 * @return One expanded monomial per term, or an empty optional when some term cannot be expanded.
 */
[[nodiscard]] std::optional<std::vector<std::vector<SymbolicVar>>> expand_poly(SymbolicPoly const &poly) {
    std::vector<std::vector<SymbolicVar>> expanded;
    expanded.reserve(poly.terms().size());
    for (auto const &term : poly.terms()) {
        auto monomial = expand_monomial(term.factors);
        if (!monomial.has_value()) {
            return std::nullopt;
        }
        expanded.push_back(std::move(*monomial));
    }
    return expanded;
}

/**
 * @brief Whether one expanded monomial is provably no larger than another.
 * @param[in] small The candidate smaller monomial.
 * @param[in] large The candidate larger monomial.
 * @param[in] registry The registry holding the declared scale order. May be null.
 * @return True when every occurrence in @p small matches injectively to a distinct occurrence in
 *         @p large that is the same variable or is declared smaller. Extents are assumed to be at
 *         least one, so unmatched occurrences of @p large only help.
 *
 * A complete bipartite matching by augmenting paths, written with an explicit stack because the
 * recursive spelling is what everyone writes and this file stays recursion free.
 */
[[nodiscard]] bool monomial_dominated_by(std::vector<SymbolicVar> const &small, std::vector<SymbolicVar> const &large,
                                         SpaceRegistry const *registry) {
    if (small.size() > large.size()) {
        return false;
    }
    if (small.empty()) {
        return true;
    }

    /// One level of the augmenting-path search: a left occurrence and the next right one to try.
    struct Frame {
        std::size_t left; ///< Index into @p small.
        std::size_t next; ///< Next index into @p large to consider.
    };

    std::vector<std::size_t> matched(large.size(), unmatched);
    std::vector<char>        visited(large.size(), 0);
    std::vector<Frame>       stack;

    for (std::size_t start = 0; start < small.size(); ++start) {
        std::ranges::fill(visited, 0);
        stack.clear();
        stack.push_back(Frame{.left = start, .next = 0});

        bool augmented = false;
        while (!stack.empty()) {
            std::size_t const depth = stack.size() - 1;
            if (stack[depth].next >= large.size()) {
                stack.pop_back();
                continue;
            }
            std::size_t const right = stack[depth].next++;
            std::size_t const left  = stack[depth].left;
            if (visited[right] != 0 || !variable_dominated_by(small[left], large[right], registry)) {
                continue;
            }
            visited[right] = 1;
            if (matched[right] == unmatched) {
                augmented = true;
                break;
            }
            stack.push_back(Frame{.left = matched[right], .next = 0});
        }

        if (!augmented) {
            return false;
        }
        // Every frame took the right occurrence it last tried; assigning them all shifts the path.
        for (auto const &frame : stack) {
            matched[frame.next - 1] = frame.left;
        }
    }
    return true;
}

/**
 * @brief Whether every term of one polynomial is dominated by some term of another.
 * @param[in] small Expanded monomials of the candidate smaller polynomial.
 * @param[in] large Expanded monomials of the candidate larger polynomial.
 * @param[in] registry The registry holding the declared scale order. May be null.
 * @return True when the domination holds term by term.
 */
[[nodiscard]] bool poly_dominated_by(std::vector<std::vector<SymbolicVar>> const &small, std::vector<std::vector<SymbolicVar>> const &large,
                                     SpaceRegistry const *registry) {
    return std::ranges::all_of(small, [&](std::vector<SymbolicVar> const &monomial) {
        return std::ranges::any_of(large,
                                   [&](std::vector<SymbolicVar> const &other) { return monomial_dominated_by(monomial, other, registry); });
    });
}

/**
 * @brief Whether every coefficient of a polynomial is strictly positive.
 * @param[in] poly The polynomial. The zero polynomial qualifies vacuously.
 * @return True when no term carries a non-positive coefficient.
 */
[[nodiscard]] bool all_coefficients_positive(SymbolicPoly const &poly) {
    return std::ranges::all_of(poly.terms(), [](SymbolicTerm const &term) { return term.coefficient > 0.0; });
}

/**
 * @brief The scale-order rung.
 * @param[in] lhs Left polynomial.
 * @param[in] rhs Right polynomial.
 * @param[in] registry The registry holding the declared scale order. May be null.
 * @return A strict ordering, or an empty optional when the rule cannot separate the two.
 */
[[nodiscard]] std::optional<std::strong_ordering> rung_scale_order(SymbolicPoly const &lhs, SymbolicPoly const &rhs,
                                                                   SpaceRegistry const *registry) {
    if (!all_coefficients_positive(lhs) || !all_coefficients_positive(rhs)) {
        return std::nullopt;
    }

    auto const left  = expand_poly(lhs);
    auto const right = expand_poly(rhs);
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }

    bool const left_below  = poly_dominated_by(*left, *right, registry);
    bool const right_below = poly_dominated_by(*right, *left, registry);
    if (left_below && !right_below) {
        return std::strong_ordering::less;
    }
    if (right_below && !left_below) {
        return std::strong_ordering::greater;
    }
    return std::nullopt;
}

/**
 * @brief Compare two substituted values, treating a near tie as no decision.
 * @param[in] lhs Left value.
 * @param[in] rhs Right value.
 * @param[in] tolerance Relative tolerance for equality.
 * @return A strict ordering, or an empty optional when the two are equal within @p tolerance.
 */
[[nodiscard]] std::optional<std::strong_ordering> numeric_order(double lhs, double rhs, double tolerance) {
    if (std::isnan(lhs) || std::isnan(rhs) || lhs == rhs) {
        return std::nullopt;
    }
    double const difference = std::abs(lhs - rhs);
    double const scale      = std::max({std::abs(lhs), std::abs(rhs), 1.0});
    if (std::isfinite(difference) && std::isfinite(scale) && difference <= tolerance * scale) {
        return std::nullopt;
    }
    return lhs < rhs ? std::strong_ordering::less : std::strong_ordering::greater;
}

/**
 * @brief A numeric rung: substitute and compare, and separate what substitutes from what does not.
 * @param[in] lhs Left polynomial.
 * @param[in] rhs Right polynomial.
 * @param[in] extents The substitution.
 * @param[in] tolerance Relative tolerance for equality.
 * @return A strict ordering, or an empty optional when neither polynomial substitutes, or when both
 *         do and the values tie.
 *
 * When exactly one polynomial substitutes, that one is ranked below the other. That is a deliberate
 * class split rather than a fudge: interleaving a numeric verdict with the structural verdict the
 * last rung gives an unsubstitutable polynomial is precisely what makes a comparison chain
 * contradict itself. Keeping the substitutable polynomials together, ordered among themselves and
 * ahead of the rest, is what keeps the chain transitive.
 */
[[nodiscard]] std::optional<std::strong_ordering> rung_numeric(SymbolicPoly const &lhs, SymbolicPoly const &rhs,
                                                               ExtentLookup const &extents, double tolerance) {
    auto const left  = lhs.evaluate(extents);
    auto const right = rhs.evaluate(extents);
    if (left.has_value() && right.has_value()) {
        return numeric_order(*left, *right, tolerance);
    }
    if (left.has_value()) {
        return std::strong_ordering::less;
    }
    if (right.has_value()) {
        return std::strong_ordering::greater;
    }
    return std::nullopt;
}

/**
 * @brief The substitution that reads each space's advisory typical extent.
 * @param[in] registry The registry. May be null, which yields an empty lookup.
 * @return The lookup, resolving only space variables with a positive typical extent.
 */
[[nodiscard]] ExtentLookup typical_extent_lookup(SpaceRegistry const *registry) {
    if (registry == nullptr) {
        return {};
    }
    return [registry](SymbolicVar const &variable) -> std::optional<double> {
        if (!variable.is_space()) {
            return std::nullopt;
        }
        SpaceId const id = variable.space_id();
        if (!id.valid() || id.value() >= registry->size()) {
            return std::nullopt;
        }
        double const extent = registry->space(id).typical_extent;
        if (!(extent > 0.0)) {
            return std::nullopt;
        }
        return extent;
    };
}

/**
 * @brief The final rung: a lexicographic walk over the canonical form.
 * @param[in] lhs Left polynomial.
 * @param[in] rhs Right polynomial.
 * @return The ordering. Equal exactly when the canonical forms are identical.
 */
[[nodiscard]] std::strong_ordering rung_lexicographic(SymbolicPoly const &lhs, SymbolicPoly const &rhs) {
    auto const       &left   = lhs.terms();
    auto const       &right  = rhs.terms();
    std::size_t const shared = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < shared; ++i) {
        if (auto const order = compare_monomials(left[i].factors, right[i].factors); order != 0) {
            return order;
        }
        if (auto const order = compare_coefficients(left[i].coefficient, right[i].coefficient); order != 0) {
            return order;
        }
    }
    return left.size() <=> right.size();
}

/**
 * @brief Render one variable.
 * @param[in] variable The variable.
 * @param[in] registry Registry used to resolve a space's scale symbol. May be null.
 * @return The scale symbol, the space name, an @c "s<id>" placeholder, or @c "?<letter>".
 */
[[nodiscard]] std::string symbol_for(SymbolicVar const &variable, SpaceRegistry const *registry) {
    if (variable.is_anonymous()) {
        return "?" + variable.letter();
    }
    SpaceId const id = variable.space_id();
    if (!id.valid()) {
        return "s?";
    }
    if (registry != nullptr && id.value() < registry->size()) {
        IndexSpace const &space = registry->space(id);
        if (!space.scale_symbol.empty()) {
            return space.scale_symbol;
        }
        if (!space.name.empty()) {
            return space.name;
        }
    }
    return "s" + std::to_string(id.value());
}

} // namespace

std::size_t SymbolicVar::hash() const noexcept {
    std::size_t seed = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(_kind));
    seed ^= std::hash<SpaceId>{}(_space) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::string>{}(_letter) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
}

void SymbolicTerm::canonicalize() {
    std::ranges::sort(factors, {}, &SymbolicFactor::variable);

    std::vector<SymbolicFactor> merged;
    merged.reserve(factors.size());
    for (auto const &factor : factors) {
        if (!merged.empty() && merged.back().variable == factor.variable) {
            merged.back().exponent += factor.exponent;
        } else {
            merged.push_back(factor);
        }
    }
    std::erase_if(merged, [](SymbolicFactor const &factor) { return factor.exponent == 0; });
    factors = std::move(merged);
}

int SymbolicTerm::total_degree() const noexcept {
    int degree = 0;
    for (auto const &factor : factors) {
        degree += factor.exponent;
    }
    return degree;
}

int SymbolicTerm::degree_in(SymbolicVar const &variable) const noexcept {
    auto const found = std::ranges::find(factors, variable, &SymbolicFactor::variable);
    return found == factors.end() ? 0 : found->exponent;
}

SymbolicPoly SymbolicPoly::constant(double value) {
    SymbolicPoly out;
    if (value != 0.0) {
        out._terms.push_back(SymbolicTerm{.coefficient = value, .factors = {}});
    }
    return out;
}

SymbolicPoly SymbolicPoly::variable(SymbolicVar variable, int exponent) {
    if (exponent == 0) {
        return constant(1.0);
    }
    SymbolicPoly out;
    out._terms.push_back(
        SymbolicTerm{.coefficient = 1.0, .factors = {SymbolicFactor{.variable = std::move(variable), .exponent = exponent}}});
    return out;
}

SymbolicPoly SymbolicPoly::from_terms(std::vector<SymbolicTerm> terms) {
    canonicalize_terms(terms);
    SymbolicPoly out;
    out._terms = std::move(terms);
    return out;
}

int SymbolicPoly::total_degree() const noexcept {
    int degree = 0;
    for (auto const &term : _terms) {
        degree = std::max(degree, term.total_degree());
    }
    return degree;
}

int SymbolicPoly::degree_in(SymbolicVar const &variable) const noexcept {
    int degree = 0;
    for (auto const &term : _terms) {
        degree = std::max(degree, term.degree_in(variable));
    }
    return degree;
}

std::vector<SymbolicVar> SymbolicPoly::variables() const {
    std::vector<SymbolicVar> out;
    for (auto const &term : _terms) {
        for (auto const &factor : term.factors) {
            out.push_back(factor.variable);
        }
    }
    std::ranges::sort(out);
    auto const duplicates = std::ranges::unique(out);
    out.erase(duplicates.begin(), duplicates.end());
    return out;
}

std::optional<double> SymbolicPoly::evaluate(ExtentLookup const &extents) const {
    double total = 0.0;
    for (auto const &term : _terms) {
        double value = term.coefficient;
        for (auto const &factor : term.factors) {
            if (!extents) {
                return std::nullopt;
            }
            auto const extent = extents(factor.variable);
            if (!extent.has_value()) {
                return std::nullopt;
            }
            value *= integer_power(*extent, factor.exponent);
        }
        total += value;
    }
    return total;
}

SymbolicPoly &SymbolicPoly::operator+=(SymbolicPoly const &other) {
    std::vector<SymbolicTerm> combined = _terms;
    combined.insert(combined.end(), other._terms.begin(), other._terms.end());
    canonicalize_terms(combined);
    _terms = std::move(combined);
    return *this;
}

SymbolicPoly &SymbolicPoly::operator*=(SymbolicPoly const &other) {
    std::vector<SymbolicTerm> product;
    product.reserve(_terms.size() * other._terms.size());
    for (auto const &left : _terms) {
        for (auto const &right : other._terms) {
            SymbolicTerm term{.coefficient = left.coefficient * right.coefficient, .factors = left.factors};
            term.factors.insert(term.factors.end(), right.factors.begin(), right.factors.end());
            product.push_back(std::move(term));
        }
    }
    canonicalize_terms(product);
    _terms = std::move(product);
    return *this;
}

SymbolicPoly &SymbolicPoly::operator*=(double factor) {
    if (factor == 0.0) {
        _terms.clear();
        return *this;
    }
    for (auto &term : _terms) {
        term.coefficient *= factor;
    }
    return *this;
}

std::string SymbolicPoly::to_string(SpaceRegistry const *registry) const {
    if (_terms.empty()) {
        return "0";
    }

    std::string out;
    for (auto const &term : _terms) {
        if (!out.empty()) {
            out += " + ";
        }

        std::string rendered;
        bool const  unit = term.coefficient == 1.0 && !term.factors.empty();
        if (!unit) {
            rendered = fmt::format("{}", term.coefficient);
        }
        for (auto const &factor : term.factors) {
            if (!rendered.empty()) {
                rendered += '*';
            }
            rendered += symbol_for(factor.variable, registry);
            if (factor.exponent != 1) {
                rendered += fmt::format("^{}", factor.exponent);
            }
        }
        out += rendered;
    }
    return out;
}

namespace {

/// One table, both directions. See EnumNames.hpp for why that is not two.
constexpr EnumNames kCompareRungNames{std::array<std::pair<CompareRung, std::string_view>, 4>{{
                                          {CompareRung::ScaleOrder, "ScaleOrder"},
                                          {CompareRung::TypicalExtent, "TypicalExtent"},
                                          {CompareRung::BoundExtent, "BoundExtent"},
                                          {CompareRung::Lexicographic, "Lexicographic"},
                                      }},
                                      "Lexicographic"};

constexpr EnumNames kCostComponentNames{std::array<std::pair<CostComponent, std::string_view>, 3>{{
                                            {CostComponent::Flops, "Flops"},
                                            {CostComponent::Traffic, "Traffic"},
                                            {CostComponent::Resident, "Resident"},
                                        }},
                                        "Flops"};

} // namespace

std::string_view compare_rung_name(CompareRung rung) noexcept {
    return kCompareRungNames.name(rung);
}

std::string_view cost_component_name(CostComponent component) noexcept {
    return kCostComponentNames.name(component);
}

PolyComparison compare_explain(SymbolicPoly const &lhs, SymbolicPoly const &rhs, ComparisonContext const &ctx) {
    if (auto const order = rung_scale_order(lhs, rhs, ctx.registry); order.has_value()) {
        return PolyComparison{.order = *order, .rung = CompareRung::ScaleOrder};
    }
    if (auto const order = rung_numeric(lhs, rhs, typical_extent_lookup(ctx.registry), ctx.relative_tolerance); order.has_value()) {
        return PolyComparison{.order = *order, .rung = CompareRung::TypicalExtent};
    }
    if (auto const order = rung_numeric(lhs, rhs, ctx.bound_extent, ctx.relative_tolerance); order.has_value()) {
        return PolyComparison{.order = *order, .rung = CompareRung::BoundExtent};
    }
    return PolyComparison{.order = rung_lexicographic(lhs, rhs), .rung = CompareRung::Lexicographic};
}

std::strong_ordering compare(SymbolicPoly const &lhs, SymbolicPoly const &rhs, ComparisonContext const &ctx) {
    return compare_explain(lhs, rhs, ctx).order;
}

std::string SymbolicCost::to_string(SpaceRegistry const *registry) const {
    return fmt::format("flops={}, traffic={}, resident={}", flops.to_string(registry), traffic.to_string(registry),
                       resident.to_string(registry));
}

CostComparison compare_explain(SymbolicCost const &lhs, SymbolicCost const &rhs, ComparisonContext const &ctx) {
    auto const on_flops = compare_explain(lhs.flops, rhs.flops, ctx);
    if (on_flops.order != 0) {
        return CostComparison{.order = on_flops.order, .rung = on_flops.rung, .component = CostComponent::Flops};
    }
    auto const on_traffic = compare_explain(lhs.traffic, rhs.traffic, ctx);
    if (on_traffic.order != 0) {
        return CostComparison{.order = on_traffic.order, .rung = on_traffic.rung, .component = CostComponent::Traffic};
    }
    auto const on_resident = compare_explain(lhs.resident, rhs.resident, ctx);
    return CostComparison{.order = on_resident.order, .rung = on_resident.rung, .component = CostComponent::Resident};
}

std::strong_ordering compare(SymbolicCost const &lhs, SymbolicCost const &rhs, ComparisonContext const &ctx) {
    return compare_explain(lhs, rhs, ctx).order;
}

LetterVars::LetterVars(std::vector<std::pair<std::string, SpaceId>> const &letter_spaces) : _letter_spaces{letter_spaces} {
    std::ranges::sort(_letter_spaces, {}, &std::pair<std::string, SpaceId>::first);
}

SymbolicVar LetterVars::var_for(std::string_view letter) const {
    std::string const key{letter};
    auto const        found = std::ranges::lower_bound(_letter_spaces, key, {}, &std::pair<std::string, SpaceId>::first);
    if (found != _letter_spaces.end() && found->first == key && found->second.valid()) {
        return SymbolicVar::space(found->second);
    }
    return SymbolicVar::anonymous(letter);
}

LetterVars letter_vars_for(EinsumDescriptor const &desc) {
    return LetterVars{desc.letter_spaces};
}

SymbolicPoly symbolic_size_for(std::vector<std::string> const &indices, LetterVars const &vars) {
    std::vector<std::string> distinct = indices;
    std::ranges::sort(distinct);
    auto const duplicates = std::ranges::unique(distinct);
    distinct.erase(duplicates.begin(), duplicates.end());

    SymbolicPoly size = SymbolicPoly::constant(1.0);
    for (auto const &letter : distinct) {
        size *= SymbolicPoly::variable(vars.var_for(letter));
    }
    return size;
}

SymbolicCost symbolic_cost_for(EinsumDescriptor const &desc) {
    LetterVars const vars{desc.letter_spaces};

    SymbolicCost cost;
    cost.flops    = symbolic_size_for(desc.spec.all_indices, vars) * 2.0;
    cost.traffic  = symbolic_size_for(desc.spec.c_indices, vars) + symbolic_size_for(desc.spec.a_indices, vars) +
                    symbolic_size_for(desc.spec.b_indices, vars);
    cost.resident = cost.traffic;
    return cost;
}

EINSUMS_NAMESPACE_END(compute_graph)
