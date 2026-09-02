//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/MultiTermFactorization.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// A factor set, as a bit per factor. Capped so the subset program stays bounded; see the cap on
/// the pass rather than a constant here, which only fixes the width of the mask.
using Mask = std::uint32_t;

/// @brief One leaf factor of a flattened product.
struct Factor {
    TensorId               tensor{};
    std::vector<ExprIndex> indices;
    bool                   conjugate{false};
};

/// @brief One statement, flattened into a product of leaf factors.
///
/// A statement whose value the pass cannot model keeps @ref searchable false and is re-emitted
/// exactly as it was raised, which is what lets one unmodellable statement cost its own rewrite
/// rather than the whole region's.
struct Term {
    std::size_t            statement{0};
    std::vector<Factor>    factors;
    std::vector<ExprIndex> output;
    PrefactorScalar        factor{double{1}};
    bool                   searchable{false};
};

/// @brief Everything the cost model needs to know about an index letter.
///
/// The extents are here because the graph is BOUND: a region raised from a live graph knows how
/// big every operand is, and throwing that away would leave the search ranking most of its
/// candidates by the comparison's last rung, which is a deterministic tie-break rather than an
/// answer. They are fed in as @ref ComparisonContext::bound_extent, which sits BELOW the
/// scale-order rung, so an annotated program's asymptotic verdict still decides and the extents
/// only settle what would otherwise be arbitrary.
///
/// That is a machine-independent input in the sense the phase rule cares about, since the extents
/// come from the problem rather than from the hardware; what it is not is size-independent, and a
/// plan chosen at one size stays CORRECT at another rather than necessarily optimal. That is the
/// same bargain @c LayoutAssignment strikes with its permute costs.
struct LetterTable {
    std::unordered_map<std::string, SymbolicVar> var;
    std::unordered_map<std::string, std::size_t> extent;
    std::unordered_map<std::string, double>      anonymous_extent;
    std::map<std::uint64_t, double>              space_extent;

    void observe(ExprIndex const &index, std::size_t extent_value) {
        if (auto const [it, fresh] = var.try_emplace(index.letter); fresh) {
            it->second = index.space.valid() ? SymbolicVar::space(index.space) : SymbolicVar::anonymous(index.letter);
        }
        extent.try_emplace(index.letter, extent_value);
        if (index.space.valid()) {
            space_extent.try_emplace(static_cast<std::uint64_t>(index.space.value()), static_cast<double>(extent_value));
        } else {
            anonymous_extent.try_emplace(index.letter, static_cast<double>(extent_value));
        }
    }

    /// @brief The lookup @ref ComparisonContext::bound_extent wants.
    /// @return A resolver over every letter this region mentions.
    [[nodiscard]] ExtentLookup lookup() const {
        return [this](SymbolicVar const &variable) -> std::optional<double> {
            if (variable.is_anonymous()) {
                auto const it = anonymous_extent.find(variable.letter());
                return it == anonymous_extent.end() ? std::nullopt : std::optional<double>{it->second};
            }
            if (variable.is_space()) {
                auto const it = space_extent.find(static_cast<std::uint64_t>(variable.space_id().value()));
                return it == space_extent.end() ? std::nullopt : std::optional<double>{it->second};
            }
            return std::nullopt;
        };
    }
};

/// @brief The product of the given letters' extent variables, coefficient one.
/// @param[in] letters Distinct letters, in any order; the polynomial is canonical either way.
/// @param[in] table   The letter lookup.
SymbolicPoly poly_over(std::set<std::string> const &letters, LetterTable const &table) {
    SymbolicPoly poly = SymbolicPoly::constant(1.0);
    for (auto const &letter : letters) {
        auto const it = table.var.find(letter);
        poly *= SymbolicPoly::variable(it == table.var.end() ? SymbolicVar::anonymous(letter) : it->second);
    }
    return poly;
}

/// @brief The cost of one binary contraction, in the same shape @ref symbolic_cost_for uses.
///
/// Deliberately the same conventions rather than a second opinion: flops is twice the loop space
/// for the multiply and the add, traffic is the three operand sizes, and resident equals traffic.
/// A search that ranked candidates by a different rule than the one @c ScalingAnalysis reports
/// would produce a plan nobody could check against the report.
SymbolicCost contraction_cost(std::set<std::string> const &left, std::set<std::string> const &right, std::set<std::string> const &out,
                              LetterTable const &table) {
    std::set<std::string> loop = left;
    loop.insert(right.begin(), right.end());

    SymbolicCost cost;
    cost.flops    = poly_over(loop, table) * 2.0;
    cost.traffic  = poly_over(out, table) + poly_over(left, table) + poly_over(right, table);
    cost.resident = cost.traffic;
    return cost;
}

SymbolicCost operator+(SymbolicCost const &lhs, SymbolicCost const &rhs) {
    return SymbolicCost{.flops = lhs.flops + rhs.flops, .traffic = lhs.traffic + rhs.traffic, .resident = lhs.resident + rhs.resident};
}

/// @brief The distinct letters of one factor.
std::set<std::string> letters_of(Factor const &factor) {
    std::set<std::string> out;
    for (auto const &index : factor.indices) {
        out.insert(index.letter);
    }
    return out;
}

/// @brief The result of the subset program over one term.
struct TreePlan {
    bool                      ok{false};
    SymbolicCost              cost;
    std::vector<Mask>         split;    ///< split[mask] = the left half chosen for that mask.
    std::vector<std::uint8_t> resolved; ///< Whether split[mask] is meaningful.
};

/// @brief The optimal binary contraction tree over @p term's factors.
///
/// The standard subset dynamic program: the best way to contract a set is the best split of it
/// into two sets, each contracted optimally. @c 3^N over the factor count, which is why the caller
/// caps that count rather than letting a pathological term decide how long the pass runs.
///
/// Masks and submasks are walked in a fixed integer order and an improvement must be STRICT, so
/// the first optimum encountered is the one kept and two runs over one term pick the same tree.
/// That is not a nicety: a search whose answer varies between runs makes every measurement
/// against it noise.
TreePlan solve_tree(Term const &term, LetterTable const &table, ComparisonContext const &ctx) {
    std::size_t const count = term.factors.size();
    Mask const        full  = static_cast<Mask>((Mask{1} << count) - 1);

    std::vector<std::set<std::string>> factor_letters;
    factor_letters.reserve(count);
    for (auto const &factor : term.factors) {
        factor_letters.push_back(letters_of(factor));
    }
    std::set<std::string> output_letters;
    for (auto const &index : term.output) {
        output_letters.insert(index.letter);
    }

    // The letters a subset must still expose: those it shares with a factor outside it, plus
    // those the term's own output asks for. Everything else is summed away when the subset is
    // formed, which is exactly what makes an early contraction cheap.
    auto external_of = [&](Mask mask) {
        std::set<std::string> inside;
        for (std::size_t f = 0; f < count; f++) {
            if ((mask & (Mask{1} << f)) != 0) {
                inside.insert(factor_letters[f].begin(), factor_letters[f].end());
            }
        }
        std::set<std::string> outside = output_letters;
        for (std::size_t f = 0; f < count; f++) {
            if ((mask & (Mask{1} << f)) == 0) {
                outside.insert(factor_letters[f].begin(), factor_letters[f].end());
            }
        }
        std::set<std::string> kept;
        for (auto const &letter : inside) {
            if (outside.count(letter) != 0) {
                kept.insert(letter);
            }
        }
        return kept;
    };

    std::vector<std::set<std::string>> external(static_cast<std::size_t>(full) + 1);
    for (Mask mask = 1; mask <= full; mask++) {
        external[mask] = external_of(mask);
    }

    TreePlan plan;
    plan.split.assign(static_cast<std::size_t>(full) + 1, 0);
    plan.resolved.assign(static_cast<std::size_t>(full) + 1, 0);
    std::vector<SymbolicCost> best(static_cast<std::size_t>(full) + 1);
    std::vector<std::uint8_t> have(static_cast<std::size_t>(full) + 1, 0);

    for (std::size_t f = 0; f < count; f++) {
        Mask const single = Mask{1} << f;
        best[single]      = SymbolicCost{};
        have[single]      = 1;
    }

    for (Mask mask = 1; mask <= full; mask++) {
        if (std::popcount(mask) < 2) {
            continue;
        }
        // Proper non-empty submasks, each pair visited once: taking only halves whose lowest set
        // bit is the mask's lowest set bit avoids visiting (S1,S2) and (S2,S1).
        Mask const anchor = mask & (~mask + 1);
        for (Mask left = (mask - 1) & mask; left != 0; left = (left - 1) & mask) {
            if ((left & anchor) == 0) {
                continue;
            }
            Mask const right = mask ^ left;
            if (right == 0 || have[left] == 0 || have[right] == 0) {
                continue;
            }
            SymbolicCost const combined =
                best[left] + best[right] + contraction_cost(external[left], external[right], external[mask], table);
            if (have[mask] == 0 || compare(combined, best[mask], ctx) < 0) {
                best[mask]          = combined;
                have[mask]          = 1;
                plan.split[mask]    = left;
                plan.resolved[mask] = 1;
            }
        }
    }

    if (have[full] == 0) {
        return plan;
    }
    plan.ok   = true;
    plan.cost = best[full];
    return plan;
}

/// @brief A pair of factors, named so two terms can recognize the same one.
///
/// The key is the pair's DEFINITION rather than its position: the two operands' tensors, their
/// index patterns rewritten into canonical letters, and the index pattern of the value they
/// produce. The last part matters and is easy to leave out: the same two tensors contracted in two
/// terms produce the same intermediate only when the same letters are summed away, and whether a
/// letter is summed depends on what else the term contains.
struct PairKey {
    std::string text;

    friend auto operator<=>(PairKey const &lhs, PairKey const &rhs) { return lhs.text <=> rhs.text; }
    friend bool operator==(PairKey const &lhs, PairKey const &rhs) = default;
};

/// @brief One occurrence of a candidate pair inside one term.
struct PairSite {
    std::size_t            term{0};
    std::size_t            left{0};  ///< Factor index of the canonically-first operand.
    std::size_t            right{0}; ///< Factor index of the canonically-second one.
    std::vector<ExprIndex> result;   ///< The intermediate's axes, in this term's own letters.
};

/// @brief The canonical description of contracting factors @p i and @p j of @p term.
/// @return The key and the result axes, or nothing when the pair contracts to a scalar.
std::optional<std::pair<PairKey, PairSite>> describe_pair(Term const &term, std::size_t index_i, std::size_t index_j,
                                                          std::size_t term_index) {
    // Canonical operand order, so a pair written A,B in one term and B,A in another is one
    // candidate. Contraction is commutative; the key must be too, or the sharing is invisible.
    std::size_t left = index_i, right = index_j;
    auto const &fi      = term.factors[index_i];
    auto const &fj      = term.factors[index_j];
    auto const  pattern = [](Factor const &factor) {
        std::vector<std::string> letters;
        letters.reserve(factor.indices.size());
        for (auto const &index : factor.indices) {
            letters.push_back(index.letter);
        }
        return letters;
    };
    if (std::make_tuple(fj.tensor, fj.conjugate, pattern(fj)) < std::make_tuple(fi.tensor, fi.conjugate, pattern(fi))) {
        std::swap(left, right);
    }

    Factor const &a = term.factors[left];
    Factor const &b = term.factors[right];

    // Which letters survive the pair: those appearing elsewhere in the term or in its output.
    std::set<std::string> outside;
    for (auto const &index : term.output) {
        outside.insert(index.letter);
    }
    for (std::size_t f = 0; f < term.factors.size(); f++) {
        if (f == left || f == right) {
            continue;
        }
        for (auto const &index : term.factors[f].indices) {
            outside.insert(index.letter);
        }
    }

    // Canonical letters, assigned by first appearance over the operands in canonical order, so
    // two occurrences that differ only in what their letters are called agree.
    std::unordered_map<std::string, std::string> canonical;
    std::vector<ExprIndex>                       result;
    std::string                                  a_pattern, b_pattern, out_pattern;
    auto const                                   render = [&](Factor const &factor, std::string &into) {
        for (auto const &index : factor.indices) {
            auto const [it, fresh] = canonical.try_emplace(index.letter, fmt::format("#{}", canonical.size()));
            into += it->second;
            into += ',';
        }
    };
    render(a, a_pattern);
    render(b, b_pattern);

    // The result axes, in canonical first-appearance order over a then b, which fixes one axis
    // order for the shared tensor whatever term reaches it first.
    std::set<std::string> seen;
    for (Factor const *factor : {&a, &b}) {
        for (auto const &index : factor->indices) {
            if (outside.count(index.letter) == 0 || !seen.insert(index.letter).second) {
                continue;
            }
            result.push_back(index);
            out_pattern += canonical.at(index.letter);
            out_pattern += ',';
        }
    }
    if (result.empty()) {
        return std::nullopt; // a scalar intermediate; nothing here emits one
    }

    PairKey key;
    key.text = fmt::format("{}{}:{}|{}{}:{}->{}", a.tensor, a.conjugate ? "*" : "", a_pattern, b.tensor, b.conjugate ? "*" : "", b_pattern,
                           out_pattern);
    return std::make_pair(key, PairSite{.term = term_index, .left = left, .right = right, .result = std::move(result)});
}

} // namespace

void MultiTermFactorization::reset_stats() {
    _num_rebracketed = 0;
    _num_shared      = 0;
    _num_inlined     = 0;
    _cut_off         = false;
}

bool MultiTermFactorization::search_enabled() const {
    return _search_explicit ? _search_enabled : config::get(option::GraphStructuralSearch);
}

bool MultiTermFactorization::applicable(Graph const &graph) const {
    if (!search_enabled()) {
        note_skip("structural search is switched off", "einsums:graph:structural-search is false and nothing overrode it");
        return false;
    }
    std::size_t contractions = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == OpKind::Einsum ? 1 : 0;
    }
    if (contractions < 2) {
        note_skip("fewer than two contractions to search over", fmt::format("{} contraction(s)", contractions));
        return false;
    }
    return true;
}

std::vector<std::string> MultiTermFactorization::describe() const {
    std::vector<std::string> lines;
    if (_num_inlined != 0 || _num_rebracketed != 0 || _num_shared != 0) {
        lines.push_back(fmt::format("MultiTermFactorization: dissolved {} captured intermediate(s), re-bracketed {} term(s), "
                                    "introduced {} shared intermediate(s)",
                                    _num_inlined, _num_rebracketed, _num_shared));
    }
    if (_cut_off) {
        // Said out loud, because a report that could not tell this apart from "already optimal"
        // would be silent in exactly the case the budget exists for.
        lines.push_back("MultiTermFactorization: the search was CUT OFF by its wall-clock budget; what it had found was applied");
    }
    return lines;
}

bool MultiTermFactorization::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    ComparisonContext ctx;
    ctx.registry = &graph.space_registry();

    std::unordered_set<TensorId> dissolvable(region.internal.begin(), region.internal.end());

    // ── Flatten ────────────────────────────────────────────────────────────────────────────
    //
    // A captured chain hides its products inside intermediates the author named. Those names are
    // an artifact of how the equations were written down, not of what has to be computed, so a
    // search that respected them would be searching the author's bracketing rather than the
    // problem's.
    std::unordered_map<TensorId, std::size_t> writer;  // tensor -> defining statement
    std::unordered_map<TensorId, std::size_t> readers; // tensor -> how many statements read it
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        auto const &statement = expr.statements[s];
        if (auto const [it, fresh] = writer.try_emplace(statement.target, s); !fresh) {
            writer[statement.target] = expr.statements.size(); // more than one writer: never inline
        }
        auto const &term = expr.at(statement.value);
        for (auto const operand : term.operands) {
            auto const &leaf = expr.at(operand);
            if (leaf.kind == TermKind::Leaf) {
                readers[leaf.tensor]++;
            }
        }
    }

    auto inlinable = [&](TensorId id, std::size_t consumer) -> std::optional<std::size_t> {
        if (dissolvable.count(id) == 0 || readers[id] != 1) {
            return std::nullopt;
        }
        auto const it = writer.find(id);
        if (it == writer.end() || it->second >= expr.statements.size() || it->second >= consumer) {
            // A region is in program order, so a producer always precedes its consumer; checking
            // rather than assuming costs one comparison and rules out folding a value away from
            // under a read that happens first.
            return std::nullopt;
        }
        auto const &statement = expr.statements[it->second];
        if (!is_zero(statement.target_prefactor)) {
            return std::nullopt; // an accumulation is more than one value; inlining would drop the rest
        }
        auto const &term = expr.at(statement.value);
        if (term.kind != TermKind::Contraction || term.operands.size() != 2 || !is_one(term.factor)) {
            // A prefactor other than one would have to be multiplied into the consumer's, and a
            // product of two PrefactorScalar variants is a conversion question this pass has no
            // reason to answer when declining costs one rewrite.
            return std::nullopt;
        }
        for (auto const operand : term.operands) {
            if (expr.at(operand).kind != TermKind::Leaf) {
                return std::nullopt;
            }
        }
        return it->second;
    };

    LetterTable                     table;
    std::unordered_set<std::size_t> consumed; // statements folded into a consumer
    std::size_t                     fresh_letter = 0;

    auto observe_factor = [&](Factor const &factor) -> bool {
        TensorHandle const *handle = graph.find_tensor(factor.tensor);
        if (handle == nullptr || handle->dims.size() != factor.indices.size()) {
            return false;
        }
        for (std::size_t axis = 0; axis < factor.indices.size(); axis++) {
            table.observe(factor.indices[axis], handle->dims[axis]);
        }
        return true;
    };

    // Expand one operand leaf into the factors of the definition behind it, renaming that
    // definition's summed letters so they cannot collide with the consumer's.
    std::function<bool(TermId, std::vector<ExprIndex> const &, bool, std::vector<Factor> &, std::size_t, std::size_t)> expand =
        [&](TermId leaf_id, std::vector<ExprIndex> const &as_seen, bool conjugate, std::vector<Factor> &into, std::size_t depth,
            std::size_t consumer) -> bool {
        auto const &leaf = expr.at(leaf_id);
        if (leaf.kind != TermKind::Leaf) {
            return false;
        }
        // A conjugated leaf is never folded. Conjugation does distribute over a product, but
        // carrying the flag onto every factor is a rewrite of its own and declining costs one
        // opportunity rather than risking a wrong sign.
        auto const definition = depth < 16 && !conjugate ? inlinable(leaf.tensor, consumer) : std::nullopt;
        if (!definition.has_value()) {
            Factor factor{.tensor = leaf.tensor, .indices = as_seen, .conjugate = conjugate};
            if (!observe_factor(factor)) {
                return false;
            }
            into.push_back(std::move(factor));
            return true;
        }

        auto const &statement = expr.statements[*definition];
        auto const &term      = expr.at(statement.value);
        if (statement.target_indices.size() != as_seen.size()) {
            return false;
        }
        // The definition's own letters, mapped onto the names the consumer uses for the same
        // axes; everything else it mentions is summed inside it and gets a name nothing else has.
        std::unordered_map<std::string, ExprIndex> substitution;
        for (std::size_t axis = 0; axis < as_seen.size(); axis++) {
            substitution.emplace(statement.target_indices[axis].letter, as_seen[axis]);
        }
        auto rename = [&](std::vector<ExprIndex> const &indices) {
            std::vector<ExprIndex> out;
            out.reserve(indices.size());
            for (auto const &index : indices) {
                auto const [it, fresh] = substitution.try_emplace(index.letter, index);
                if (fresh) {
                    it->second.letter = fmt::format("~{}", fresh_letter++);
                }
                out.push_back(it->second);
            }
            return out;
        };

        consumed.insert(*definition);
        for (std::size_t operand = 0; operand < term.operands.size(); operand++) {
            bool const operand_conj = operand < term.conjugate.size() && term.conjugate[operand];
            if (!expand(term.operands[operand], rename(term.operand_indices[operand]), operand_conj, into, depth + 1, *definition)) {
                return false;
            }
        }
        return true;
    };

    // One walk. A statement whose own flattening fails must not take its folded producers with
    // it, so the consumed set is snapshotted before each statement and restored when that
    // statement turns out to be unmodellable: those producers keep their own statements.
    std::vector<Term>               terms;
    std::unordered_set<std::size_t> folded;
    terms.reserve(expr.statements.size());
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        Term term;
        term.statement        = s;
        auto const &statement = expr.statements[s];
        auto const &value     = expr.at(statement.value);
        term.output           = statement.target_indices;
        term.factor           = value.factor;

        std::unordered_set<std::size_t> const before = consumed;
        if (value.kind == TermKind::Contraction && value.operands.size() == 2) {
            term.searchable = true;
            for (std::size_t operand = 0; operand < value.operands.size() && term.searchable; operand++) {
                bool const operand_conj = operand < value.conjugate.size() && value.conjugate[operand];
                term.searchable         = expand(value.operands[operand], value.operand_indices[operand], operand_conj, term.factors, 0, s);
            }
        }
        if (term.searchable) {
            // A repeated letter inside one operand is a diagonal access, which the loop-space cost
            // model above does not describe, and a term priced wrongly is worse than one declined.
            for (auto const &factor : term.factors) {
                std::set<std::string> seen;
                for (auto const &index : factor.indices) {
                    term.searchable = term.searchable && seen.insert(index.letter).second;
                }
            }
            std::set<std::string> target_seen;
            for (auto const &index : term.output) {
                term.searchable = term.searchable && target_seen.insert(index.letter).second;
            }
            if (term.factors.size() < 2 || term.factors.size() > _max_factors) {
                term.searchable = false;
            }
        }
        if (!term.searchable) {
            consumed = before;
            note_skip("statement is not a product this pass can model",
                      fmt::format("target '{}' has {} factor(s)", statement.target_name, term.factors.size()));
        }
        terms.push_back(std::move(term));
    }
    folded = consumed;
    for (auto const s : folded) {
        terms[s].searchable = false;
    }
    if (std::ranges::none_of(terms, [](Term const &term) { return term.searchable; })) {
        return false;
    }

    // ── Search ─────────────────────────────────────────────────────────────────────────────
    auto total_cost = [&](std::vector<TreePlan> const &plans) {
        SymbolicCost sum;
        for (auto const &plan : plans) {
            if (plan.ok) {
                sum = sum + plan.cost;
            }
        }
        return sum;
    };
    auto solve_all = [&]() {
        std::vector<TreePlan> plans(terms.size());
        for (std::size_t t = 0; t < terms.size(); t++) {
            if (terms[t].searchable && !folded.count(t)) {
                plans[t] = solve_tree(terms[t], table, ctx);
            }
        }
        return plans;
    };

    // Installed here rather than at construction: the table is only complete once every factor of
    // every term has been observed, and a lookup that resolved half a polynomial would make the
    // rung decide on a subset, which is the discipline SymbolicCost.hpp asks callers for.
    ctx.bound_extent = table.lookup();

    std::vector<TreePlan> plans    = solve_all();
    SymbolicCost const    original = total_cost(plans);

    /// One committed shared intermediate, in emission order.
    struct Shared {
        TensorId               tensor{};
        Factor                 left;
        Factor                 right;
        std::vector<ExprIndex> result;
    };
    std::vector<Shared> shared;

    for (;;) {
        if (budget().expired()) {
            _cut_off = true;
            note_skip("the wall-clock budget ran out mid-search", "the best assignment found so far was kept");
            break;
        }

        // Candidates, gathered in a std::map so the order they are tried in is the order of their
        // keys rather than of a hash table's buckets.
        std::map<PairKey, std::vector<PairSite>> candidates;
        for (std::size_t t = 0; t < terms.size(); t++) {
            if (!terms[t].searchable || terms[t].factors.size() < 3) {
                continue; // a two-factor term has nothing to share that is not the whole term
            }
            for (std::size_t i = 0; i + 1 < terms[t].factors.size(); i++) {
                for (std::size_t j = i + 1; j < terms[t].factors.size(); j++) {
                    if (auto described = describe_pair(terms[t], i, j, t); described.has_value()) {
                        candidates[described->first].push_back(std::move(described->second));
                    }
                }
            }
        }

        SymbolicCost const     baseline = total_cost(plans);
        std::optional<PairKey> best_key;
        SymbolicCost           best_cost;
        std::vector<TreePlan>  best_plans;

        for (auto const &[key, sites] : candidates) {
            if (budget().expired()) {
                _cut_off = true;
                break;
            }
            if (sites.size() < 2) {
                continue;
            }
            // Applying the candidate: each occurrence loses its two factors and gains one naming
            // the intermediate. A term left with a single factor would need a copy rather than a
            // contraction, which this pass does not emit.
            std::vector<Term> trial  = terms;
            bool              usable = true;
            for (auto const &site : sites) {
                Term &term = trial[site.term];
                if (term.factors.size() < 3) {
                    usable = false;
                    break;
                }
                Factor              placeholder{.tensor = TensorId{0}, .indices = site.result, .conjugate = false};
                std::vector<Factor> kept;
                kept.reserve(term.factors.size() - 1);
                for (std::size_t f = 0; f < term.factors.size(); f++) {
                    if (f != site.left && f != site.right) {
                        kept.push_back(term.factors[f]);
                    }
                }
                kept.push_back(std::move(placeholder));
                term.factors = std::move(kept);
            }
            if (!usable) {
                continue;
            }

            std::vector<TreePlan> trial_plans(trial.size());
            bool                  solved = true;
            for (std::size_t t = 0; t < trial.size() && solved; t++) {
                if (!trial[t].searchable || folded.count(t) != 0) {
                    continue;
                }
                trial_plans[t] = solve_tree(trial[t], table, ctx);
                solved         = trial_plans[t].ok;
            }
            if (!solved) {
                continue;
            }

            auto const           &first         = sites.front();
            std::set<std::string> left_letters  = letters_of(terms[first.term].factors[first.left]);
            std::set<std::string> right_letters = letters_of(terms[first.term].factors[first.right]);
            std::set<std::string> result_letters;
            for (auto const &index : first.result) {
                result_letters.insert(index.letter);
            }
            SymbolicCost const trial_cost = total_cost(trial_plans) + contraction_cost(left_letters, right_letters, result_letters, table);

            if (compare(trial_cost, baseline, ctx) >= 0) {
                continue;
            }
            if (!best_key.has_value() || compare(trial_cost, best_cost, ctx) < 0) {
                best_key   = key;
                best_cost  = trial_cost;
                best_plans = std::move(trial_plans);
            }
        }

        if (!best_key.has_value()) {
            break;
        }

        // Commit. The tensor is declared now, because a committed intermediate is one this pass
        // will emit, and a declaration for a candidate it merely considered would leave the graph
        // holding shells nothing writes.
        auto const  &sites = candidates.at(*best_key);
        auto const  &first = sites.front();
        Factor const left  = terms[first.term].factors[first.left];
        Factor const right = terms[first.term].factors[first.right];

        std::vector<std::size_t> dims;
        std::vector<SpaceId>     spaces;
        std::vector<std::string> symbols;
        bool                     every_axis_symbolic = true;
        dims.reserve(first.result.size());
        for (auto const &index : first.result) {
            auto const extent = table.extent.find(index.letter);
            if (extent == table.extent.end()) {
                every_axis_symbolic = false;
                break;
            }
            dims.push_back(extent->second);
            spaces.push_back(index.space);
            std::string symbol;
            if (index.space.valid() && index.space.value() < graph.space_registry().size()) {
                symbol = graph.space_registry().space(index.space).dim_symbol;
            }
            every_axis_symbolic = every_axis_symbolic && !symbol.empty();
            symbols.push_back(std::move(symbol));
        }
        if (dims.size() != first.result.size()) {
            note_skip("a shared candidate has an axis with no known extent", best_key->text);
            break;
        }

        TensorHandle const *model = graph.find_tensor(left.tensor);
        if (model == nullptr) {
            break;
        }
        TensorId const shared_id = detail::dispatch_scalar_type(model->dtype, [&]<typename T>(T /*tag*/) {
            auto &tensor = graph.declare_runtime_tensor<T>(fmt::format("mtf_shared{}", shared.size()), dims, /*intermediate=*/true);
            return graph.find_tensor_id_by_ptr(&tensor);
        });
        if (shared_id == 0) {
            break;
        }
        if (std::ranges::any_of(spaces, [](SpaceId id) { return id.valid(); })) {
            graph.annotate_spaces(shared_id, spaces);
        }
        if (every_axis_symbolic) {
            // Only when EVERY axis has a symbol: a partial annotation is what makes a bind move
            // some extents and not others, which is worse than none at all.
            graph.annotate_dims(shared_id, symbols);
        }

        for (auto const &site : sites) {
            Term               &term = terms[site.term];
            Factor              placeholder{.tensor = shared_id, .indices = site.result, .conjugate = false};
            std::vector<Factor> kept;
            kept.reserve(term.factors.size() - 1);
            for (std::size_t f = 0; f < term.factors.size(); f++) {
                if (f != site.left && f != site.right) {
                    kept.push_back(term.factors[f]);
                }
            }
            kept.push_back(std::move(placeholder));
            term.factors = std::move(kept);
        }
        shared.push_back(Shared{.tensor = shared_id, .left = left, .right = right, .result = first.result});
        _num_shared++;
        plans = std::move(best_plans);
        report(2, fmt::format("share {} across {} term(s)", best_key->text, sites.size()));
    }

    // ── Emit ───────────────────────────────────────────────────────────────────────────────
    //
    // Nothing above touched the expression, so a decision to leave it alone costs nothing.
    if (shared.empty() && compare(total_cost(plans), original, ctx) >= 0) {
        note_skip("no re-bracketing or sharing beats the captured form", fmt::format("{} term(s) examined", terms.size()));
        return false;
    }

    auto leaf_for = [&](TensorId id, std::string const &name) {
        ExprTerm leaf;
        leaf.kind   = TermKind::Leaf;
        leaf.tensor = id;
        leaf.name   = name;
        return expr.add(std::move(leaf));
    };

    std::vector<ExprStatement> emitted;
    emitted.reserve(expr.statements.size() + shared.size() + terms.size());
    std::size_t scratch_index = 0;

    auto emit_contraction = [&](Factor const &a, Factor const &b, TensorId target, std::string const &target_name,
                                std::vector<ExprIndex> const &target_indices, PrefactorScalar target_prefactor, PrefactorScalar factor,
                                std::string const &label) {
        ExprTerm term;
        term.kind            = TermKind::Contraction;
        term.indices         = target_indices;
        term.operands        = {leaf_for(a.tensor, {}), leaf_for(b.tensor, {})};
        term.operand_indices = {a.indices, b.indices};
        term.conjugate       = {a.conjugate, b.conjugate};
        term.factor          = factor;

        ExprStatement statement;
        statement.target           = target;
        statement.target_name      = target_name;
        statement.target_indices   = target_indices;
        statement.target_prefactor = target_prefactor;
        statement.value            = expr.add(std::move(term));
        statement.origin_kind      = OpKind::Einsum;
        statement.origin_label     = label;
        emitted.push_back(std::move(statement));
    };

    // The shared intermediates come first, in commit order, which is also dependency order: a
    // candidate over an already-shared factor could only be found after that one was committed.
    for (auto const &entry : shared) {
        emit_contraction(entry.left, entry.right, entry.tensor, {}, entry.result, PrefactorScalar{double{0}}, PrefactorScalar{double{1}},
                         fmt::format("mtf shared {}", entry.tensor));
    }

    bool ok = true;
    for (std::size_t t = 0; t < terms.size() && ok; t++) {
        auto const &statement = expr.statements[terms[t].statement];
        if (folded.count(t) != 0) {
            continue; // its value now lives inside a consumer
        }
        if (!terms[t].searchable || !plans[t].ok) {
            emitted.push_back(statement); // re-emitted exactly as raised
            continue;
        }

        Term const     &term = terms[t];
        TreePlan const &plan = plans[t];
        Mask const      full = static_cast<Mask>((Mask{1} << term.factors.size()) - 1);

        std::set<std::string> output_letters;
        for (auto const &index : term.output) {
            output_letters.insert(index.letter);
        }

        // Rebuild the tree bottom-up. A composite operand becomes a declared intermediate; the
        // outermost combine writes the statement's own target with its own prefactors.
        std::function<std::optional<Factor>(Mask)> build = [&](Mask mask) -> std::optional<Factor> {
            if (std::popcount(mask) == 1) {
                return term.factors[static_cast<std::size_t>(std::countr_zero(mask))];
            }
            if (plan.resolved[mask] == 0) {
                return std::nullopt;
            }
            auto const left  = build(plan.split[mask]);
            auto const right = build(mask ^ plan.split[mask]);
            if (!left.has_value() || !right.has_value()) {
                return std::nullopt;
            }

            // The axes this combine must expose, in first-appearance order over its two operands.
            std::set<std::string> outside = output_letters;
            for (std::size_t f = 0; f < term.factors.size(); f++) {
                if ((mask & (Mask{1} << f)) == 0) {
                    for (auto const &index : term.factors[f].indices) {
                        outside.insert(index.letter);
                    }
                }
            }
            std::vector<ExprIndex> axes;
            std::set<std::string>  seen;
            for (Factor const *operand : {&*left, &*right}) {
                for (auto const &index : operand->indices) {
                    if (outside.count(index.letter) != 0 && seen.insert(index.letter).second) {
                        axes.push_back(index);
                    }
                }
            }

            if (mask == full) {
                emit_contraction(*left, *right, statement.target, statement.target_name, statement.target_indices,
                                 statement.target_prefactor, term.factor, statement.origin_label);
                return Factor{.tensor = statement.target, .indices = statement.target_indices, .conjugate = false};
            }

            std::vector<std::size_t> dims;
            std::vector<SpaceId>     spaces;
            dims.reserve(axes.size());
            for (auto const &index : axes) {
                auto const extent = table.extent.find(index.letter);
                if (extent == table.extent.end()) {
                    return std::nullopt;
                }
                dims.push_back(extent->second);
                spaces.push_back(index.space);
            }
            TensorHandle const *model = graph.find_tensor(left->tensor);
            if (model == nullptr || dims.empty()) {
                return std::nullopt;
            }
            TensorId const scratch = detail::dispatch_scalar_type(model->dtype, [&]<typename T>(T /*tag*/) {
                auto &tensor = graph.declare_runtime_tensor<T>(fmt::format("mtf_t{}", scratch_index++), dims, /*intermediate=*/true);
                return graph.find_tensor_id_by_ptr(&tensor);
            });
            if (scratch == 0) {
                return std::nullopt;
            }
            if (std::ranges::any_of(spaces, [](SpaceId id) { return id.valid(); })) {
                graph.annotate_spaces(scratch, spaces);
            }
            emit_contraction(*left, *right, scratch, {}, axes, PrefactorScalar{double{0}}, PrefactorScalar{double{1}},
                             fmt::format("mtf {}", scratch));
            return Factor{.tensor = scratch, .indices = axes, .conjugate = false};
        };

        if (!build(full).has_value()) {
            ok = false;
            break;
        }
        _num_rebracketed++;
    }

    if (!ok) {
        note_skip("a term's tree could not be emitted", "the region is left as it was");
        return false;
    }

    expr.statements = std::move(emitted);
    // Counted here rather than where the folding happened: every return above leaves the
    // expression exactly as it was, and a counter that had already been raised would report
    // intermediates as dissolved on a run that dissolved nothing.
    _num_inlined += folded.size();
    EINSUMS_LOG_INFO("MultiTermFactorization: {} shared intermediate(s), {} term(s) re-bracketed, {} captured intermediate(s) dissolved",
                     _num_shared, _num_rebracketed, _num_inlined);
    report(1, fmt::format("{} shared intermediate(s), {} term(s) re-bracketed", _num_shared, _num_rebracketed));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
