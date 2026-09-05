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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ContractionTreeSearch.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The bracketing search, the letter table it prices against and the factor type they share
/// live in ContractionTreeSearch.hpp, because `FactorizationPass` asks the same question of the
/// chain a provider's factors leave behind.
using search::add_cost;
using search::contraction_cost;
using search::Factor;
using search::letters_of;
using search::LetterTable;
using search::Mask;
using search::solve_tree;
using search::TreePlan;

/// @brief One contraction of the CAPTURED bracketing, in the flattened term's own letters.
///
/// What the rewrite is compared against. A term's captured cost is the sum of these, priced
/// through the same @ref contraction_cost the search ranks its own trees with, so the two sides
/// of the comparison are one model rather than two opinions. The letters are the consumer's,
/// after the alpha-renaming the flattening performs, which is what lets one table price both.
struct CapturedStep {
    std::set<std::string> left;
    std::set<std::string> right;
    std::set<std::string> out;
};

/// @brief One statement, flattened into a product of leaf factors.
///
/// A statement whose value the pass cannot model keeps @ref searchable false and is re-emitted
/// exactly as it was raised, which is what lets one unmodellable statement cost its own rewrite
/// rather than the whole region's.
struct Term {
    std::size_t               statement{0};
    std::vector<Factor>       factors;
    std::vector<ExprIndex>    output;
    std::vector<CapturedStep> steps;
    PrefactorScalar           factor{double{1}};
    bool                      searchable{false};
};

/// @brief One statement seen as a two-operand product, whatever node kind wrote it.
///
/// Three kinds present one algebra. A contraction says which letters it sums directly. A
/// @ref OpKind::DirectProduct is the same product with NO summed letter, since its output carries
/// every axis both operands do. A @ref OpKind::Dot is the same product summed over EVERY letter,
/// since its destination is a scalar. Modelling the three in one place is what lets an amplitude
/// formed by a contraction, multiplied by a denominator and reduced by a dot, flatten into one
/// product instead of surviving as a stored leaf every candidate has to rebuild.
struct Modelled {
    std::vector<TermId>                 operands;
    std::vector<std::vector<ExprIndex>> operand_indices;
    std::vector<bool>                   conjugate;
    std::vector<ExprIndex>              output;
    PrefactorScalar                     factor{double{1}};
};

/// @brief Read @p statement as a two-operand product, or decline.
/// @param[in] expr      The arena the statement's terms live in.
/// @param[in] statement The statement.
/// @return The product, or nothing when this statement is not one.
std::optional<Modelled> model_statement(TensorExpr const &expr, ExprStatement const &statement) {
    if (statement.value == invalid_term || statement.value >= expr.terms.size()) {
        return std::nullopt;
    }
    ExprTerm const &value = expr.at(statement.value);
    if (value.operands.size() != 2) {
        return std::nullopt;
    }
    for (auto const operand : value.operands) {
        if (expr.at(operand).kind != TermKind::Leaf) {
            return std::nullopt;
        }
    }

    Modelled out;
    out.operands = value.operands;
    if (value.kind == TermKind::Contraction) {
        if (value.operand_indices.size() != 2) {
            return std::nullopt;
        }
        out.operand_indices = value.operand_indices;
        out.output          = statement.target_indices;
        out.factor          = value.factor;
        out.conjugate.assign({!value.conjugate.empty() && value.conjugate[0], value.conjugate.size() > 1 && value.conjugate[1]});
        return out;
    }
    if (value.kind != TermKind::Elementwise) {
        return std::nullopt;
    }

    if (value.element_kind == OpKind::DirectProduct) {
        auto const *scalars = std::get_if<ElementwiseBinaryDescriptor>(&value.descriptor);
        if (scalars == nullptr) {
            return std::nullopt;
        }
        // Every axis of the destination is an axis of both operands, which is what makes the
        // positional letters a raised elementwise term carries line up across the three.
        for (auto const operand : value.operands) {
            if (expr.at(operand).indices.size() != statement.target_indices.size()) {
                return std::nullopt;
            }
        }
        out.operand_indices.assign({statement.target_indices, statement.target_indices});
        out.output = statement.target_indices;
        out.factor = live_alpha(*scalars);
        out.conjugate.assign({false, false});
        return out;
    }

    if (value.element_kind == OpKind::Dot) {
        // The TILED dot shares the kind and reduces over a grid rather than over one buffer; the
        // descriptor is what tells the two apart.
        auto const *scalars = std::get_if<DotDescriptor>(&value.descriptor);
        if (scalars == nullptr) {
            return std::nullopt;
        }
        std::vector<ExprIndex> const &a = expr.at(value.operands[0]).indices;
        std::vector<ExprIndex> const &b = expr.at(value.operands[1]).indices;
        if (a.empty() || a.size() != b.size()) {
            return std::nullopt;
        }
        out.operand_indices.assign({a, b});
        // A scalar destination, whatever rank the tensor holding it has: the reduction sums every
        // letter, so the value has no free index and the emitted contraction says so with an empty
        // output list rather than with the destination's own positional axis, which would name a
        // letter the operands already use for something else.
        out.output = {};
        out.factor = PrefactorScalar{double{1}};
        out.conjugate.assign({scalars->conjugated, false});
        return out;
    }
    return std::nullopt;
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
    // The framework's counters too. Without this the region tallies were never zeroed and a pass
    // instance used for a second apply reported "rewrote 2 of 1 region(s)", which is the shape a
    // running total takes when nothing resets it.
    RegionRewrite::reset_stats();
    _num_rebracketed  = 0;
    _num_shared       = 0;
    _num_inlined      = 0;
    _num_cache_hits   = 0;
    _num_cache_misses = 0;
    _cut_off          = false;
    // The cache itself is NOT cleared here. It is the one piece of state whose whole value is
    // that it outlives an apply, and `clear_cache` is how a caller asks for it to go.
}

bool MultiTermFactorization::search_enabled() const {
    return _search_explicit ? _search_enabled : config::get(option::GraphStructuralSearch);
}

std::size_t MultiTermFactorization::max_factors() const {
    // The option is the process-wide statement and `set_max_factors` is the per-pipeline one,
    // which is the shape `search_enabled` already has. Clamped on the way out for the same reason
    // the setter clamps: a term of one factor is a copy and there is nothing to bracket.
    if (_max_factors_explicit) {
        return _max_factors;
    }
    auto const cap = config::get(option::GraphFactorizationMaxFactors);
    return cap < 2 ? std::size_t{2} : static_cast<std::size_t>(cap);
}

bool MultiTermFactorization::cache_enabled() const {
    return _cache_explicit ? _cache_enabled : config::get(option::GraphFactorizationCache);
}

bool MultiTermFactorization::applicable(Graph const &graph) const {
    if (!search_enabled()) {
        note_skip("structural search is switched off", "einsums:graph:structural-search is false and nothing overrode it");
        return false;
    }
    // Every kind the flattener reads as a product, not the contractions alone: an energy written
    // as one contraction, one direct product and one dot has a nine-factor product in it and a
    // gate counting einsums would have declined before looking.
    std::size_t products = 0;
    for (auto const &node : graph.nodes()) {
        products += node.kind == OpKind::Einsum || node.kind == OpKind::DirectProduct || node.kind == OpKind::Dot ? 1 : 0;
    }
    if (products < 2) {
        note_skip("fewer than two products to search over", fmt::format("{} product(s)", products));
        return false;
    }

    // Taken here because this is the one hook called exactly once per graph, and taken BEFORE any
    // region has been rewritten: the regions are visited back to front, so a hash read at the
    // second one would digest a graph the first had already moved.
    _graph_key_valid = false;
    if (cache_enabled()) {
        try {
            _graph_key       = graph.content_hash();
            _graph_key_valid = true;
        } catch (std::exception const &error) {
            // A graph with no canonical form has no hash, which is a reason to search rather than
            // an error: nothing is cached and everything else proceeds.
            note_skip("the graph has no canonical form, so no plan can be keyed on it", error.what());
        }
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
    if (_num_cache_hits != 0 || _num_cache_misses != 0) {
        lines.push_back(fmt::format("MultiTermFactorization: {} region(s) replayed a kept plan and {} searched; {} plan(s) held",
                                    _num_cache_hits, _num_cache_misses, _cache.size()));
    }
    if (_cut_off) {
        // Said out loud, because a report that could not tell this apart from "already optimal"
        // would be silent in exactly the case the budget exists for.
        lines.push_back("MultiTermFactorization: the search was CUT OFF by its wall-clock budget; what it had found was applied");
    }
    return lines;
}

bool MultiTermFactorization::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    // ── The kept plan, if this region has one ──────────────────────────────────────────────
    //
    // Asked before anything is computed, because the answer "nothing here is worth rewriting"
    // costs a whole search to reach and is worth keeping for exactly that reason.
    std::size_t const        cap = max_factors();
    PlanKey const            key{_graph_key, region.first, region.last, cap};
    FactorizationPlan const *cached = nullptr;
    if (_graph_key_valid) {
        if (auto const it = _cache.find(key); it != _cache.end()) {
            cached = &it->second;
            _num_cache_hits++;
            report(2, fmt::format("region [{},{}) replays a plan a structurally identical graph already found", region.first, region.last));
            if (!cached->rewrites) {
                note_skip("a structurally identical region was already searched and offered nothing",
                          fmt::format("region [{},{})", region.first, region.last));
                return false;
            }
        } else {
            _num_cache_misses++;
        }
    }
    bool const cut_off_on_entry = _cut_off;

    ComparisonContext ctx;
    ctx.registry = &graph.space_registry();

    std::unordered_set<TensorId> dissolvable(region.internal.begin(), region.internal.end());

    // ── Flatten ────────────────────────────────────────────────────────────────────────────
    //
    // A captured chain hides its products inside intermediates the author named. Those names are
    // an artifact of how the equations were written down, not of what has to be computed, so a
    // search that respected them would be searching the author's bracketing rather than the
    // problem's.
    std::unordered_map<TensorId, std::size_t>              writer;  // tensor -> defining statement
    std::unordered_map<TensorId, std::vector<std::size_t>> readers; // tensor -> statements reading it
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        auto const &statement = expr.statements[s];
        if (auto const [it, fresh] = writer.try_emplace(statement.target, s); !fresh) {
            writer[statement.target] = expr.statements.size(); // more than one writer: never inline
        }
        auto const &term = expr.at(statement.value);
        for (auto const operand : term.operands) {
            auto const &leaf = expr.at(operand);
            if (leaf.kind == TermKind::Leaf && (readers[leaf.tensor].empty() || readers[leaf.tensor].back() != s)) {
                readers[leaf.tensor].push_back(s);
            }
        }
    }

    // Whether one statement is a definition this pass could fold into a consumer, ignoring who
    // reads it.
    auto foldable = [&](std::size_t s) -> bool {
        auto const &statement = expr.statements[s];
        if (dissolvable.count(statement.target) == 0) {
            return false;
        }
        auto const own = writer.find(statement.target);
        if (own == writer.end() || own->second != s) {
            return false; // written more than once, or not by itself
        }
        if (!is_zero(statement.target_prefactor)) {
            return false; // an accumulation is more than one value; inlining would drop the rest
        }
        auto const product = model_statement(expr, statement);
        // A prefactor other than one would have to be multiplied into the consumer's, and a
        // product of two PrefactorScalar variants is a conversion question this pass has no reason
        // to answer when declining costs one rewrite.
        return product.has_value() && is_one(product->factor);
    };

    // Which statement, if any, ABSORBS each definition.
    //
    // The rule the first version had was "exactly one reader", and it is too narrow for the shape
    // this pass now flattens: an amplitude read both by the product that scales it and by the dot
    // that reduces it has two readers, and both of them end up inside one statement once the first
    // is folded into the second. So a definition is absorbed by the statement every one of its
    // readers resolves to, walking backwards so a reader's own owner is known before its
    // producer's is asked for. A definition whose readers resolve to two different statements
    // stays a statement of its own, which is what keeps a value two consumers need from being
    // computed twice.
    std::vector<std::optional<std::size_t>> owner(expr.statements.size());
    for (std::size_t back = expr.statements.size(); back-- > 0;) {
        if (!foldable(back)) {
            continue;
        }
        auto const it = readers.find(expr.statements[back].target);
        if (it == readers.end() || it->second.empty()) {
            continue;
        }
        std::optional<std::size_t> resolved;
        bool                       agreed = true;
        for (auto const reader : it->second) {
            if (reader <= back) {
                agreed = false; // a read before the write; a region is in program order, so decline
                break;
            }
            std::size_t const root = owner[reader].value_or(reader);
            if (!resolved.has_value()) {
                resolved = root;
            } else if (*resolved != root) {
                agreed = false;
                break;
            }
        }
        if (agreed && resolved.has_value() && *resolved > back) {
            owner[back] = resolved;
        }
    }

    auto inlinable = [&](TensorId id, std::size_t root) -> std::optional<std::size_t> {
        auto const it = writer.find(id);
        if (it == writer.end() || it->second >= expr.statements.size() || it->second >= root) {
            return std::nullopt;
        }
        return owner[it->second] == root ? std::optional<std::size_t>{it->second} : std::nullopt;
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
            ExprIndex index = factor.indices[axis];
            // The raised letter carries the space the descriptor froze AT CAPTURE, and that map
            // holds nothing for a program annotated afterwards, which is every program annotated
            // from Python. Re-derived from the operand's current handle, which is the argument
            // `LetterBindings.hpp` already makes for the diagnostic passes and what
            // `DeltaElimination` does for the same reason. It decides whether the comparison can
            // use the family's typical extents at all: one anonymous letter blocks that rung for
            // the whole polynomial, and the answer then rests on the extents this capture happened
            // to have.
            if (!index.space.valid() && axis < handle->spaces.size()) {
                index.space = handle->spaces[axis];
            }
            table.observe(index, handle->dims[axis]);
        }
        return true;
    };

    /// The letters of an index list, deduplicated, for the captured-cost record.
    auto step_letters = [](std::vector<ExprIndex> const &indices) {
        std::set<std::string> out;
        for (auto const &index : indices) {
            out.insert(index.letter);
        }
        return out;
    };

    // Expand one operand leaf into the factors of the definition behind it, renaming that
    // definition's summed letters so they cannot collide with the consumer's.
    std::function<bool(TermId, std::vector<ExprIndex> const &, bool, Term &, std::size_t, std::size_t)> expand =
        [&](TermId leaf_id, std::vector<ExprIndex> const &as_seen, bool conjugate, Term &term, std::size_t depth,
            std::size_t root) -> bool {
        auto const &leaf = expr.at(leaf_id);
        if (leaf.kind != TermKind::Leaf) {
            return false;
        }
        // A conjugated leaf is never folded. Conjugation does distribute over a product, but
        // carrying the flag onto every factor is a rewrite of its own and declining costs one
        // opportunity rather than risking a wrong sign.
        auto const definition = depth < 16 && !conjugate ? inlinable(leaf.tensor, root) : std::nullopt;
        if (!definition.has_value()) {
            Factor factor{.tensor = leaf.tensor, .indices = as_seen, .conjugate = conjugate};
            if (!observe_factor(factor)) {
                return false;
            }
            term.factors.push_back(std::move(factor));
            return true;
        }

        auto const &statement = expr.statements[*definition];
        auto const  product   = model_statement(expr, statement);
        if (!product.has_value() || statement.target_indices.size() != as_seen.size()) {
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
        std::vector<std::vector<ExprIndex>> renamed;
        renamed.reserve(product->operand_indices.size());
        for (auto const &indices : product->operand_indices) {
            renamed.push_back(rename(indices));
        }
        // What the captured form paid for this value, in the consumer's letters. A dissolved
        // definition is arithmetic the rewrite removes, so a comparison that ignored it would
        // decline a tree that pays for the whole flattening.
        term.steps.push_back(
            CapturedStep{.left = step_letters(renamed[0]), .right = step_letters(renamed[1]), .out = step_letters(as_seen)});
        for (std::size_t operand = 0; operand < product->operands.size(); operand++) {
            bool const operand_conj = operand < product->conjugate.size() && product->conjugate[operand];
            if (!expand(product->operands[operand], renamed[operand], operand_conj, term, depth + 1, root)) {
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
    // Which root statement's flattening consumed each definition, so one root can be reverted
    // without disturbing another's.
    std::unordered_map<std::size_t, std::size_t> consumed_by;
    terms.reserve(expr.statements.size());
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        Term term;
        term.statement        = s;
        auto const &statement = expr.statements[s];
        auto const  product   = model_statement(expr, statement);
        term.output           = product.has_value() ? product->output : statement.target_indices;
        term.factor           = product.has_value() ? product->factor : PrefactorScalar{double{1}};

        std::unordered_set<std::size_t> const before = consumed;
        if (product.has_value()) {
            term.searchable = true;
            term.steps.push_back(CapturedStep{.left  = step_letters(product->operand_indices[0]),
                                              .right = step_letters(product->operand_indices[1]),
                                              .out   = step_letters(product->output)});
            for (std::size_t operand = 0; operand < product->operands.size() && term.searchable; operand++) {
                bool const operand_conj = operand < product->conjugate.size() && product->conjugate[operand];
                term.searchable         = expand(product->operands[operand], product->operand_indices[operand], operand_conj, term, 0, s);
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
            if (term.factors.size() < 2 || term.factors.size() > cap) {
                term.searchable = false;
            }
        }
        if (!term.searchable) {
            consumed = before;
            note_skip("statement is not a product this pass can model",
                      fmt::format("target '{}' has {} factor(s)", statement.target_name, term.factors.size()));
        } else {
            for (auto const folded_here : consumed) {
                if (before.count(folded_here) == 0) {
                    consumed_by.emplace(folded_here, s);
                }
            }
        }
        terms.push_back(std::move(term));
    }
    folded = consumed;

    // A definition is absorbed only when every path into it folded. One path may decline where
    // another folds -- a conjugated use, or a nesting past the depth limit -- and the value would
    // then be read from a statement that is no longer emitted. So the flattening is CHECKED
    // against what survives it, and a root whose dissolution left a live reader behind is reverted
    // whole rather than patched.
    for (bool settled = false; !settled;) {
        settled = true;
        std::unordered_set<TensorId> still_read;
        for (std::size_t t = 0; t < terms.size(); t++) {
            if (folded.count(t) != 0) {
                continue;
            }
            if (terms[t].searchable) {
                for (auto const &factor : terms[t].factors) {
                    still_read.insert(factor.tensor);
                }
                continue;
            }
            for (auto const operand : expr.at(expr.statements[t].value).operands) {
                still_read.insert(expr.at(operand).tensor);
            }
        }
        for (auto const s : folded) {
            if (still_read.count(expr.statements[s].target) == 0) {
                continue;
            }
            std::size_t const root = consumed_by.at(s);
            terms[root].searchable = false;
            for (auto it = folded.begin(); it != folded.end();) {
                it = consumed_by.at(*it) == root ? folded.erase(it) : std::next(it);
            }
            note_skip("a flattening left one of its dissolved definitions with a reader",
                      fmt::format("target '{}'", expr.statements[root].target_name));
            settled = false;
            break;
        }
    }
    for (auto const s : folded) {
        terms[s].searchable = false;
    }

    // Every store goes through here, so the one rule that matters is stated once: a search cut
    // off by its budget is a partial answer and is never kept, which is also why the budget is
    // not part of the key.
    auto keep = [&](FactorizationPlan plan) {
        bool const cut_off_here = _cut_off && !cut_off_on_entry;
        if (_graph_key_valid && !cut_off_here && cached == nullptr) {
            _cache.insert_or_assign(key, std::move(plan));
        }
    };

    if (std::ranges::none_of(terms, [](Term const &term) { return term.searchable; })) {
        keep({});
        return false;
    }

    // ── Search ─────────────────────────────────────────────────────────────────────────────
    auto total_cost = [&](std::vector<TreePlan> const &plans) {
        SymbolicCost sum;
        for (auto const &plan : plans) {
            if (plan.ok) {
                sum = add_cost(sum, plan.cost);
            }
        }
        return sum;
    };
    auto solve_all = [&]() {
        std::vector<TreePlan> plans(terms.size());
        for (std::size_t t = 0; t < terms.size(); t++) {
            if (terms[t].searchable && !folded.count(t)) {
                plans[t] = solve_tree(terms[t].factors, terms[t].output, table, ctx);
            }
        }
        return plans;
    };

    // Installed here rather than at construction: the table is only complete once every factor of
    // every term has been observed, and a lookup that resolved half a polynomial would make the
    // rung decide on a subset, which is the discipline SymbolicCost.hpp asks callers for.
    ctx.bound_extent = table.lookup();

    // What the CAPTURED program pays for the statements this pass may rewrite: the bracketing the
    // author wrote, plus every definition the flattening dissolves, priced through the model the
    // search ranks its own trees with. Comparing against the searched cost instead would ask
    // whether a search improves on itself, which it never does, and would leave the pass unable to
    // fire on a re-bracketing that shares nothing.
    SymbolicCost captured;
    for (std::size_t t = 0; t < terms.size(); t++) {
        if (!terms[t].searchable || folded.count(t) != 0) {
            continue;
        }
        for (auto const &step : terms[t].steps) {
            captured = add_cost(captured, contraction_cost(step.left, step.right, step.out, table));
        }
    }

    /// One committed shared intermediate, in emission order.
    struct Shared {
        TensorId               tensor{};
        Factor                 left;
        Factor                 right;
        std::vector<ExprIndex> result;
    };
    std::vector<Shared> shared;
    /// What the committed shared intermediates themselves cost, which the per-term plans do not
    /// carry: a shared pair is one contraction outside every tree that reads it.
    SymbolicCost shared_total;

    /// The occurrences of each committed pair, in commit order, as the plan records them.
    std::vector<std::vector<std::array<std::size_t, 3>>> commit_log;

    // Declare the intermediate one committed pair needs and rewrite its occurrences onto it. The
    // search reaches this after picking a winner and a replay reaches it straight away, which is
    // what makes the two produce the same graph rather than two graphs that agree on a test.
    auto commit_pair = [&](std::vector<PairSite> const &sites, std::string_view label) -> bool {
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
            SpaceId const space = index.space.valid() ? index.space : table.space_for(index.letter);
            spaces.push_back(space);
            std::string symbol;
            if (space.valid() && space.value() < graph.space_registry().size()) {
                symbol = graph.space_registry().space(space).dim_symbol;
            }
            every_axis_symbolic = every_axis_symbolic && !symbol.empty();
            symbols.push_back(std::move(symbol));
        }
        if (dims.size() != first.result.size()) {
            note_skip("a shared candidate has an axis with no known extent", std::string{label});
            return false;
        }

        TensorHandle const *model = graph.find_tensor(left.tensor);
        if (model == nullptr) {
            return false;
        }
        // The tensor is declared now, because a committed intermediate is one this pass will
        // emit, and a declaration for a candidate it merely considered would leave the graph
        // holding shells nothing writes.
        TensorId const shared_id = detail::dispatch_scalar_type(model->dtype, [&]<typename T>(T /*tag*/) {
            auto &tensor = graph.declare_runtime_tensor<T>(fmt::format("mtf_shared{}", shared.size()), dims, /*intermediate=*/true);
            return graph.find_tensor_id_by_ptr(&tensor);
        });
        if (shared_id == 0) {
            return false;
        }
        // EVERY axis or none. `annotate_spaces` rightly refuses a hole in an annotation, so an
        // "any axis is valid" guard throws on the mixed case, which is what an intermediate over
        // one annotated space and one unannotated letter is; a decoupled energy has exactly that
        // shape as soon as its quadrature index is a space and something else is not.
        if (std::ranges::all_of(spaces, [](SpaceId id) { return id.valid(); })) {
            graph.annotate_spaces(shared_id, spaces);
        }
        if (every_axis_symbolic) {
            // Only when EVERY axis has a symbol: a partial annotation is what makes a bind move
            // some extents and not others, which is worse than none at all.
            graph.annotate_dims(shared_id, symbols);
        }

        std::vector<std::array<std::size_t, 3>> record;
        record.reserve(sites.size());
        for (auto const &site : sites) {
            record.push_back({site.term, site.left, site.right});
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
        commit_log.push_back(std::move(record));
        shared.push_back(Shared{.tensor = shared_id, .left = left, .right = right, .result = first.result});
        shared_total = add_cost(shared_total, contraction_cost(letters_of(left), letters_of(right), letters_of(first.result), table));
        _num_shared++;
        report(2, fmt::format("share {} across {} term(s)", label, sites.size()));
        return true;
    };

    std::vector<TreePlan> plans;

    // ── Replay, when a structurally identical region already answered this ─────────────────
    //
    // The commits are checked against this region's shapes before any of them is applied, so a
    // plan that does not fit is a miss rather than a half-applied rewrite. It cannot happen for a
    // plan keyed on this graph's content hash, and checking is cheaper than proving it cannot.
    bool replayed = false;
    if (cached != nullptr) {
        std::vector<std::size_t> factor_count(terms.size(), 0);
        for (std::size_t t = 0; t < terms.size(); t++) {
            factor_count[t] = terms[t].factors.size();
        }
        bool fits = cached->trees.size() == terms.size();
        for (auto const &commit : cached->commits) {
            fits = fits && !commit.empty();
            for (auto const &site : commit) {
                fits = fits && site[0] < terms.size() && terms[site[0]].searchable && factor_count[site[0]] >= 3 && site[1] < site[2] &&
                       site[2] < factor_count[site[0]];
                if (!fits) {
                    break;
                }
                factor_count[site[0]]--;
            }
            if (!fits) {
                break;
            }
        }

        if (!fits) {
            // Only reachable through a hash collision, and a wrong plan is worse than a slow one.
            note_skip("a kept plan does not describe this region, so it was dropped",
                      fmt::format("region [{},{})", region.first, region.last));
            _cache.erase(key);
            cached = nullptr;
        } else {
            replayed = true;
            for (auto const &commit : cached->commits) {
                std::vector<PairSite> sites;
                sites.reserve(commit.size());
                for (auto const &site : commit) {
                    auto described = describe_pair(terms[site[0]], site[1], site[2], site[0]);
                    if (!described.has_value()) {
                        replayed = false;
                        break;
                    }
                    sites.push_back(std::move(described->second));
                }
                if (!replayed || !commit_pair(sites, "a kept plan's shared pair")) {
                    replayed = false;
                    break;
                }
            }
            if (replayed) {
                plans.resize(terms.size());
                for (std::size_t t = 0; t < terms.size(); t++) {
                    plans[t].ok       = cached->trees[t].ok;
                    plans[t].split    = cached->trees[t].split;
                    plans[t].resolved = cached->trees[t].resolved;
                }
            } else {
                note_skip("a kept plan could not be replayed onto this region", fmt::format("region [{},{})", region.first, region.last));
                return false;
            }
        }
    }

    if (!replayed) {
        plans = solve_all();
    }

    while (!replayed) {
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
            // Two occurrences in ONE term are declined, and this is a defect the wider flattener
            // made reachable rather than a restriction of the idea. Applying a site rebuilds that
            // term's factor list, so the second occurrence's positions name factors that have
            // moved, and the two occurrences may share a factor outright, which would consume one
            // value twice. A same-term share is a real opportunity and reaching it needs the
            // occurrences applied together under a disjointness check, which is its own change.
            if (std::ranges::adjacent_find(sites, [](PairSite const &lhs, PairSite const &rhs) { return lhs.term == rhs.term; }) !=
                sites.end()) {
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
                trial_plans[t] = solve_tree(trial[t].factors, trial[t].output, table, ctx);
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
            SymbolicCost const trial_cost =
                add_cost(total_cost(trial_plans), contraction_cost(left_letters, right_letters, result_letters, table));

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

        if (!commit_pair(candidates.at(*best_key), best_key->text)) {
            break;
        }
        plans = std::move(best_plans);
    }

    // ── Emit ───────────────────────────────────────────────────────────────────────────────
    //
    // Nothing above touched the expression, so a decision to leave it alone costs nothing.
    if (!replayed && compare(add_cost(total_cost(plans), shared_total), captured, ctx) >= 0) {
        note_skip("no re-bracketing or sharing beats the captured form", fmt::format("{} term(s) examined", terms.size()));
        keep({});
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
    // Unique across the REGIONS of one graph as well as across graphs. The counter restarts per
    // region and this pass descends into loop bodies, so a program with two regions would
    // otherwise declare two different tensors under one name; the storage auditor keys its
    // duplicate check on the name and reads that as one tensor allocated twice. The region's
    // first node id is what separates them, and it is as deterministic as the node order is.
    std::string const scratch_stem =
        fmt::format("{}_r{}", graph.name(), region.nodes.empty() ? std::size_t{0} : static_cast<std::size_t>(region.nodes.front()));
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
        // Priced the way a raised term is, so the region's before-and-after compares like with
        // like. An emitted term with no cost reads as free, and the report then offers a rewrite
        // to nothing as evidence that the search was worth making.
        std::set<std::string> out_letters;
        for (auto const &index : target_indices) {
            out_letters.insert(index.letter);
        }
        term.cost = contraction_cost(letters_of(a), letters_of(b), out_letters, table);

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
                // The MODELLED output, not the statement's own index list: a dot writes a scalar
                // and the tensor holding it has an axis of its own, which is not a letter of this
                // product and would name one the operands already use for something else.
                emit_contraction(*left, *right, statement.target, statement.target_name, term.output, statement.target_prefactor,
                                 term.factor, statement.origin_label);
                return Factor{.tensor = statement.target, .indices = term.output, .conjugate = false};
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
                // From the table, which re-derived the space from the operands' handles where the
                // raised index carried none, so an intermediate of a program annotated after
                // capture is annotated too.
                spaces.push_back(index.space.valid() ? index.space : table.space_for(index.letter));
            }
            TensorHandle const *model = graph.find_tensor(left->tensor);
            if (model == nullptr || dims.empty()) {
                return std::nullopt;
            }
            TensorId const scratch = detail::dispatch_scalar_type(model->dtype, [&]<typename T>(T /*tag*/) {
                // Named after the graph it is declared in. This pass descends into loop bodies,
                // so one program holds one of these counters per graph and two graphs would
                // otherwise declare two different tensors under one name; the storage auditor
                // keys its duplicate check on the name and reads that as one tensor allocated
                // twice.
                auto &tensor = graph.declare_runtime_tensor<T>(fmt::format("{}_mtf_t{}", scratch_stem, scratch_index++), dims,
                                                               /*intermediate=*/true);
                return graph.find_tensor_id_by_ptr(&tensor);
            });
            if (scratch == 0) {
                return std::nullopt;
            }
            if (std::ranges::all_of(spaces, [](SpaceId id) { return id.valid(); })) {
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
        keep({});
        return false;
    }

    // The plan, in the vocabulary a replay needs: which pairs were committed, in order, and the
    // tree each term ended with. Nothing here names a tensor or a node, which is what lets it
    // apply to another graph that hashes the same.
    FactorizationPlan plan;
    plan.rewrites = true;
    plan.commits  = commit_log;
    plan.trees.reserve(plans.size());
    for (auto const &tree : plans) {
        plan.trees.push_back(FactorizationPlan::Tree{.ok = tree.ok, .split = tree.split, .resolved = tree.resolved});
    }
    keep(std::move(plan));

    expr.statements = std::move(emitted);
    // Counted here rather than where the folding happened: every return above leaves the
    // expression exactly as it was, and a counter that had already been raised would report
    // intermediates as dissolved on a run that dissolved nothing.
    _num_inlined += folded.size();
    EINSUMS_LOG_INFO("MultiTermFactorization: {} shared intermediate(s), {} term(s) re-bracketed, {} captured intermediate(s) dissolved",
                     _num_shared, _num_rebracketed, _num_inlined);
    report(1, fmt::format("{} shared intermediate(s), {} term(s) re-bracketed", _num_shared, _num_rebracketed));
    // The pass's OWN before-and-after, which is not the region framework's: that one prices the
    // raised statements and an elementwise statement claims no cost, so a region holding a direct
    // product or a dot has a before side that leaves them out. This line prices both sides through
    // one model and is what the decision above was actually taken on.
    report(2,
           fmt::format("the captured product(s) cost {} and the chosen tree(s) cost {}", captured.flops.to_string(&graph.space_registry()),
                       add_cost(total_cost(plans), shared_total).flops.to_string(&graph.space_registry())));
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
