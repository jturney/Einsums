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
#include "ExprHelpers.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The bracketing search, the letter table it prices against and the factor type they share
/// live in ContractionTreeSearch.hpp, because `FactorizationPass` asks the same question of the
/// chain a provider's factors leave behind.
using search::add_cost;
using search::contraction_cost;
using search::contraction_term;
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
    /// The statement this step prices. One definition inlined into two consumers appears in two
    /// terms and is ONE node of the captured program, so the sum counts each origin once.
    std::size_t origin{0};
};

/// @brief One definition spliced into a term, and where its factors landed.
///
/// The factors a splice contributes are contiguous, since the walk is depth first, so the term
/// the author wrote can be recovered from the flattened one by putting the leaf back over that
/// range. That is what lets the profit question be asked without flattening the term twice: the
/// two candidates are then over one set of letters rather than two alpha-renamings of it.
struct Splice {
    std::size_t            definition{0};
    std::size_t            first{0};
    std::size_t            count{0};
    std::vector<ExprIndex> as_seen; ///< The definition's axes in the consumer's letters.
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
    std::vector<Splice>       splices;
    PrefactorScalar           factor{double{1}};
    bool                      searchable{false};
};

/// @brief The term with one definition's factors put back as the leaf they came from.
/// @param[in] term       The flattened term.
/// @param[in] definition The statement whose splices are to be undone.
/// @return The factors and output the consumer would have had reading that definition, or nothing
///         when this term spliced it nowhere.
std::optional<std::vector<Factor>> without_splice(Term const &term, std::size_t definition) {
    std::vector<Splice> undone;
    for (auto const &splice : term.splices) {
        if (splice.definition == definition) {
            undone.push_back(splice);
        }
    }
    if (undone.empty()) {
        return std::nullopt;
    }
    // Highest first, so an earlier range's positions still mean what they said. Two splices of one
    // definition cannot nest, since a value is not inside itself.
    std::ranges::sort(undone, [](Splice const &lhs, Splice const &rhs) { return lhs.first > rhs.first; });

    std::vector<Factor> factors = term.factors;
    for (auto const &splice : undone) {
        if (splice.first + splice.count > factors.size()) {
            return std::nullopt;
        }
        Factor leaf{.tensor = TensorId{0}, .indices = splice.as_seen, .conjugate = false};
        factors.erase(factors.begin() + static_cast<std::ptrdiff_t>(splice.first),
                      factors.begin() + static_cast<std::ptrdiff_t>(splice.first + splice.count));
        factors.insert(factors.begin() + static_cast<std::ptrdiff_t>(splice.first), std::move(leaf));
    }
    if (factors.size() < 2) {
        return std::nullopt; // a product of one factor is a copy, which is not a shape to compare
    }
    return factors;
}

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

    // The GROUPED element-wise kinds are deliberately not modelled as products, which is where
    // the grouped family departs from the dense one. A dense direct product enters a term because
    // an einsum whose output carries every letter of both operands is exactly it, and the emitted
    // node is that einsum. There is no grouped einsum: a term carrying a member letter lowers onto
    // a grouped batched GEMM or a grouped reduction, and neither computes a Hadamard product. Its
    // per-member prefactors would have nowhere to go besides, since a grouped batch carries one
    // prefactor for the whole call.
    if (value.element_kind == OpKind::DirectProduct) {
        auto const *scalars = value.descriptor.get_if<ElementwiseBinaryDescriptor>();
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
        auto const *scalars = value.descriptor.get_if<DotDescriptor>();
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
    if (std::make_tuple(fj.key(), fj.conjugate, pattern(fj)) < std::make_tuple(fi.key(), fi.conjugate, pattern(fi))) {
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
            if (!outside.contains(index.letter) || !seen.insert(index.letter).second) {
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

    // The IDENTITY of each operand, which for a ragged factor is its member list
    // rather than its tensor id: every ragged factor's id is the same empty one,
    // so a key built from that alone would make two unrelated families' pairs
    // look like one candidate.
    PairKey key;
    key.text = fmt::format("{}{}:{}|{}{}:{}->{}", fmt::join(a.key(), "."), a.conjugate ? "*" : "", a_pattern, fmt::join(b.key(), "."),
                           b.conjugate ? "*" : "", b_pattern, out_pattern);
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
    _num_copies       = 0;
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

std::size_t MultiTermFactorization::max_readers() const {
    // The two-level shape every knob in this module has, clamped on the way out for the reason the
    // setter clamps: a definition no consumer may take is one this pass never inlines at all.
    if (_max_readers_explicit) {
        return _max_readers;
    }
    auto const cap = config::get(option::GraphFactorizationMaxReaders);
    return cap < 1 ? std::size_t{1} : static_cast<std::size_t>(cap);
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
    // The grouped batch and the grouped reduction count too, and for the same reason: a grouped
    // node is one operation over a family of members, which raises to a contraction carrying one
    // more free letter, and a pair body made entirely of them holds nothing else for this gate to
    // find.
    std::size_t products = 0;
    for (auto const &node : graph.nodes()) {
        products += node.kind == OpKind::Einsum || node.kind == OpKind::DirectProduct || node.kind == OpKind::Dot ||
                            node.kind == OpKind::GroupedBatchedGemm || node.kind == OpKind::GroupedDot ||
                            node.kind == OpKind::GroupedDirectProduct
                        ? 1
                        : 0;
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
    if (_num_inlined != 0 || _num_rebracketed != 0 || _num_shared != 0 || _num_copies != 0) {
        lines.push_back(fmt::format("MultiTermFactorization: dissolved {} captured intermediate(s), copied {} into a consumer that "
                                    "profits while keeping the definition, re-bracketed {} term(s), introduced {} shared intermediate(s)",
                                    _num_inlined, _num_copies, _num_rebracketed, _num_shared));
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
    std::size_t const        cap          = max_factors();
    std::size_t const        consumer_cap = max_readers();
    PlanKey const            key{_graph_key, region.first, region.last, cap, consumer_cap};
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

    // A GROUPED statement writes a member list, and it is dissolvable exactly when
    // every one of its members is: half a family dissolved would leave the other
    // half written by a node the rewrite had just removed.
    std::unordered_set<TensorId> internal(region.internal.begin(), region.internal.end());
    std::set<ValueKey>           dissolvable;
    for (auto const &statement : expr.statements) {
        auto const key = value_key(statement);
        if (std::ranges::all_of(key, [&internal](TensorId id) { return internal.contains(id); })) {
            dissolvable.insert(key);
        }
    }

    // ── Flatten ────────────────────────────────────────────────────────────────────────────
    //
    // A captured chain hides its products inside intermediates the author named. Those names are
    // an artifact of how the equations were written down, not of what has to be computed, so a
    // search that respected them would be searching the author's bracketing rather than the
    // problem's.
    std::map<ValueKey, std::size_t>              writer;  // value -> defining statement
    std::map<ValueKey, std::vector<std::size_t>> readers; // value -> statements reading it
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        auto const &statement = expr.statements[s];
        auto const  target    = value_key(statement);
        if (auto const [it, fresh] = writer.try_emplace(target, s); !fresh) {
            writer[target] = expr.statements.size(); // more than one writer: never inline
        }
        auto const &term = expr.at(statement.value);
        for (auto const operand : term.operands) {
            auto const &leaf = expr.at(operand);
            if (leaf.kind != TermKind::Leaf) {
                continue;
            }
            auto const source = value_key(leaf);
            if (readers[source].empty() || readers[source].back() != s) {
                readers[source].push_back(s);
            }
        }
    }

    // Whether one statement is a definition this pass could fold into a consumer, ignoring who
    // reads it.
    auto foldable = [&](std::size_t s) -> bool {
        auto const &statement = expr.statements[s];
        auto const  target    = value_key(statement);
        if (!dissolvable.contains(target)) {
            return false;
        }
        auto const own = writer.find(target);
        if (own == writer.end() || own->second != s) {
            return false; // written more than once, or not by itself
        }
        if (!is_zero(statement.target_prefactor)) {
            return false; // an accumulation is more than one value; inlining would drop the rest
        }
        if (!statement.operators.empty()) {
            return false; // inlined, the operators would act on one factor of the consumer's product
        }
        auto const product = model_statement(expr, statement);
        // A prefactor other than one would have to be multiplied into the consumer's, and a
        // product of two PrefactorScalar variants is a conversion question this pass has no reason
        // to answer when declining costs one rewrite.
        return product.has_value() && is_one(product->factor);
    };

    // What a statement's value READS, transitively through the definitions this pass may fold into
    // it. Inlining moves a read from where the author put it to where the value is used, so an
    // operand something in between overwrites would be read at its new value instead of the one
    // the captured bracketing saw. Program order is what a region is in, and this is what says
    // which of its reads may travel.
    std::vector<std::set<ValueKey>> cone_reads(expr.statements.size());
    for (std::size_t s = 0; s < expr.statements.size(); s++) {
        auto const &statement = expr.statements[s];
        if (statement.value == invalid_term || statement.value >= expr.terms.size()) {
            continue;
        }
        for (auto const operand : expr.at(statement.value).operands) {
            ValueKey const id = value_key(expr.at(operand));
            cone_reads[s].insert(id);
            auto const own = writer.find(id);
            if (own != writer.end() && own->second < s && foldable(own->second)) {
                cone_reads[s].insert(cone_reads[own->second].begin(), cone_reads[own->second].end());
            }
        }
    }

    // Which statements ABSORB each definition.
    //
    // The rule the first version had was "exactly one reader" and the second's was "every reader
    // resolves to one consumer". Both leave a value two consumers want as a stored leaf that pins
    // the algebra around it, and neither is what soundness asks for: what is unsound is
    // DISSOLVING a definition something still reads, not inlining it where it pays. So this is a
    // SET of consuming statements, walked backwards so a reader's own consumer is known before its
    // producer's is asked for, and the definition is kept for whichever of them does not profit.
    //
    // A reader that is itself absorbed contributes the statements it is absorbed into rather than
    // itself, which is what makes a chain of intermediates one product; a reader that is not
    // contributes itself.
    std::vector<std::set<std::size_t>> absorbers(expr.statements.size());
    for (std::size_t back = expr.statements.size(); back-- > 0;) {
        if (!foldable(back)) {
            continue;
        }
        auto const it = readers.find(value_key(expr.statements[back]));
        if (it == readers.end() || it->second.empty()) {
            continue;
        }
        std::set<std::size_t> sites;
        bool                  ordered = true;
        for (auto const reader : it->second) {
            if (reader <= back) {
                ordered = false; // a read before the write; a region is in program order, so decline
                break;
            }
            if (absorbers[reader].empty()) {
                sites.insert(reader);
            } else {
                sites.insert(absorbers[reader].begin(), absorbers[reader].end());
            }
        }
        if (!ordered) {
            continue;
        }
        // The growth bound, stated rather than discovered: every consumer that takes a copy is a
        // term of its own, so the search's size is the factor cap's program times the number of
        // copies. A value more consumers than this read is far more likely to be a quantity the
        // program genuinely shares than an artifact of the author's bracketing.
        if (sites.size() > consumer_cap) {
            note_skip("a definition has more consumers than the reader cap admits, so it is left whole",
                      fmt::format("target '{}' has {} consumer(s), einsums:graph:factorization-max-readers is {}",
                                  expr.statements[back].target_name, sites.size(), consumer_cap));
            continue;
        }
        bool travels = true;
        for (auto const site : sites) {
            for (std::size_t between = back + 1; between < site && travels; between++) {
                travels = cone_reads[back].count(value_key(expr.statements[between])) == 0;
            }
        }
        if (!travels) {
            note_skip("a definition reads an operand something rewrites before its consumer, so it cannot travel there",
                      fmt::format("target '{}'", expr.statements[back].target_name));
            continue;
        }
        absorbers[back] = std::move(sites);
    }

    // The consumers a definition is KEPT for, which the profit probe below fills in and a replayed
    // plan states outright. A pair in here is a consumer that reads the definition rather than
    // recomputing it.
    std::set<std::pair<std::size_t, std::size_t>> retained_for;
    if (cached != nullptr) {
        for (auto const &entry : cached->retained_for) {
            retained_for.emplace(entry[0], entry[1]);
        }
    }

    auto inlinable = [&](ValueKey const &id, std::size_t root) -> std::optional<std::size_t> {
        auto const it = writer.find(id);
        if (it == writer.end() || it->second >= expr.statements.size() || it->second >= root) {
            return std::nullopt;
        }
        if (absorbers[it->second].count(root) == 0 || retained_for.contains({it->second, root})) {
            return std::nullopt;
        }
        return it->second;
    };

    LetterTable                     table;
    std::unordered_set<std::size_t> consumed; // statements folded into a consumer
    std::size_t                     fresh_letter = 0;
    // Which letter each alpha-renamed one stands for. A renamed axis is the same axis, and a cost
    // model that did not know it would rank every re-bracketing below the form it came from: the
    // searched side would mention a variable the captured side never heard of, and the dominance
    // rung can only match a variable against itself.
    std::unordered_map<std::string, std::string> renamed_from;

    // What a RAGGED letter measures. The member letter's extent is the family's
    // member count; every other letter a family introduces has one extent per
    // member, so the table carries the whole column and the cost model reads the
    // typical one off it. Below the scale rung, exactly where a bound extent for
    // an ordinary letter sits.
    std::map<std::string, std::size_t>              ragged_typical;
    std::map<std::string, std::vector<std::size_t>> ragged_members;
    for (auto const &family : expr.families) {
        ragged_typical[family.letter] = family.members;
        for (auto const &[letter, values] : family.extents) {
            ragged_typical[letter] = family.typical_extent(letter);
            ragged_members[letter] = values;
        }
    }

    auto observe_factor = [&](Factor const &factor) -> bool {
        if (factor.ragged()) {
            for (auto const &index : factor.indices) {
                auto const        origin = renamed_from.find(index.letter);
                std::string const source = origin == renamed_from.end() ? index.letter : origin->second;
                auto const        hit    = ragged_typical.find(source);
                if (hit == ragged_typical.end()) {
                    return false;
                }
                if (origin == renamed_from.end()) {
                    table.observe(index, hit->second);
                } else {
                    table.observe_renamed(index, hit->second, origin->second);
                }
            }
            return true;
        }
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
            // One letter names one extent across the region, since intermediates are sized from the
            // table. Two statements that reuse a letter at different extents (`ij <- ik ; kj` over
            // 3x3 and then over 4x4) would otherwise size a term's intermediates by whichever came
            // first, so a term that disagrees is left as it was raised.
            if (auto const known = table.extent.find(index.letter); known != table.extent.end() && known->second != handle->dims[axis]) {
                return false;
            }
            auto const origin = renamed_from.find(index.letter);
            if (origin == renamed_from.end()) {
                table.observe(index, handle->dims[axis]);
            } else {
                table.observe_renamed(index, handle->dims[axis], origin->second);
            }
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
        auto const definition = depth < 16 && !conjugate ? inlinable(value_key(leaf), root) : std::nullopt;
        if (!definition.has_value()) {
            Factor factor{.tensor = leaf.tensor, .indices = as_seen, .conjugate = conjugate, .members = leaf.members};
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
                    it->second.letter    = fmt::format("~{}", fresh_letter++);
                    auto const inherited = renamed_from.find(index.letter);
                    renamed_from.emplace(it->second.letter, inherited == renamed_from.end() ? index.letter : inherited->second);
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
        // decline a tree that pays for the whole flattening. The origin is what keeps one
        // definition inlined into two consumers from being counted twice on the captured side.
        term.steps.push_back(CapturedStep{
            .left = step_letters(renamed[0]), .right = step_letters(renamed[1]), .out = step_letters(as_seen), .origin = *definition});
        std::size_t const first = term.factors.size();
        for (std::size_t operand = 0; operand < product->operands.size(); operand++) {
            bool const operand_conj = operand < product->conjugate.size() && product->conjugate[operand];
            if (!expand(product->operands[operand], renamed[operand], operand_conj, term, depth + 1, root)) {
                return false;
            }
        }
        term.splices.push_back(Splice{.definition = *definition, .first = first, .count = term.factors.size() - first, .as_seen = as_seen});
        return true;
    };

    // One walk per statement, repeated once when the profit probe below changes a decision. A
    // statement whose own flattening fails must not take its folded producers with it, so a
    // statement's consumed set reaches the run's only when that statement stands.
    std::vector<Term> terms;
    // Which statements each definition was spliced into. A definition every one of them took is
    // dissolved; one a consumer kept is copied into the others and stays a statement of its own.
    std::map<std::size_t, std::vector<std::size_t>> dissolved_into;

    auto build_all = [&](bool announce) {
        // From scratch, letters included. A walk that carried the previous one's counter would
        // name the same axis differently depending on how many walks ran, and a plan replayed in
        // one walk would then emit a different program than the search that found it in two.
        terms.clear();
        dissolved_into.clear();
        renamed_from.clear();
        table        = LetterTable{};
        fresh_letter = 0;
        terms.reserve(expr.statements.size());
        for (std::size_t s = 0; s < expr.statements.size(); s++) {
            Term term;
            term.statement        = s;
            auto const &statement = expr.statements[s];
            auto const  product   = model_statement(expr, statement);
            term.output           = product.has_value() ? product->output : statement.target_indices;
            term.factor           = product.has_value() ? product->factor : PrefactorScalar{double{1}};

            consumed.clear();
            if (product.has_value()) {
                term.searchable = true;
                term.steps.push_back(CapturedStep{.left   = step_letters(product->operand_indices[0]),
                                                  .right  = step_letters(product->operand_indices[1]),
                                                  .out    = step_letters(product->output),
                                                  .origin = s});
                for (std::size_t operand = 0; operand < product->operands.size() && term.searchable; operand++) {
                    bool const operand_conj = operand < product->conjugate.size() && product->conjugate[operand];
                    term.searchable = expand(product->operands[operand], product->operand_indices[operand], operand_conj, term, 0, s);
                }
            }
            if (term.searchable) {
                // A repeated letter inside one operand is a diagonal access, which the loop-space
                // cost model above does not describe, and a term priced wrongly is worse than one
                // declined.
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
                if (announce) {
                    note_skip("statement is not a product this pass can model",
                              fmt::format("target '{}' has {} factor(s)", statement.target_name, term.factors.size()));
                }
            } else {
                for (auto const folded_here : consumed) {
                    dissolved_into[folded_here].push_back(s);
                }
            }
            terms.push_back(std::move(term));
        }
    };

    // Installed before anything is ranked and after the first walk has observed every letter, so
    // no comparison ever resolves half a polynomial. The lookup reads the table as it stands, and
    // the second walk only adds letters to it.
    build_all(cached != nullptr);
    ctx.bound_extent = table.lookup();

    // ── Per-consumer inlining ──────────────────────────────────────────────────────────────
    //
    // A definition with one consumer goes there whole, which is the rule this pass has always
    // had. With several, each consumer decides for itself: inlining hands the search the factors
    // behind the value and it re-brackets around them, which is either cheaper than reading the
    // value or it is not, and the consumer that gains nothing reads the definition instead. What
    // is left over is priced once, because a kept definition is still one node.
    //
    // The probe is greedy across definitions, holding every other decision at "inlined": two
    // definitions inside one term interact, and asking the question jointly is a second subset
    // program over the decisions rather than a cheaper way to ask this one.
    if (cached == nullptr) {
        for (std::size_t definition = 0; definition < terms.size(); definition++) {
            if (absorbers[definition].size() < 2) {
                continue;
            }
            for (auto const site : absorbers[definition]) {
                if (budget().expired()) {
                    _cut_off = true;
                    break;
                }
                if (!terms[site].searchable) {
                    // Nothing was spliced there in any case, and saying so is what lets a
                    // consumer the inlining pushed past the factor cap read the definition and be
                    // searchable on the next walk.
                    retained_for.emplace(definition, site);
                    continue;
                }
                auto const reading = without_splice(terms[site], definition);
                if (!reading.has_value()) {
                    continue;
                }
                TreePlan const inlined = solve_tree(terms[site].factors, terms[site].output, table, ctx);
                TreePlan const kept    = solve_tree(*reading, terms[site].output, table, ctx);
                if (!inlined.ok || !kept.ok || compare(inlined.cost, kept.cost, ctx) >= 0) {
                    // A copy that only adds arithmetic. The bracketing that reads the definition
                    // is always available to the inlined term, by contracting the factors it was
                    // built from first, so this says the search found nothing better than that.
                    retained_for.emplace(definition, site);
                    note_skip("inlining a definition into one of its consumers buys that consumer nothing, so it reads it",
                              fmt::format("target '{}' into '{}': inlined it costs {} and reading it costs {}",
                                          expr.statements[definition].target_name, expr.statements[site].target_name,
                                          inlined.ok ? inlined.cost.flops.to_string(&graph.space_registry()) : "no tree",
                                          kept.ok ? kept.cost.flops.to_string(&graph.space_registry()) : "no tree"));
                }
            }
        }
        // The final walk, which is the one every decision below reads. Announced here rather
        // than on the probe's walk so a statement this pass cannot model is reported once.
        build_all(true);
    }

    // What survives the flattening.
    //
    // A definition is dissolved only when nothing still emitted reads it. That covers the consumer
    // that kept it, which is the ordinary outcome of the rule above, and it covers the path that
    // could not fold where another one did: a conjugated use, or a nesting past the depth limit.
    // The definition comes back as its own statement either way, which is always possible because
    // a term is a valid flattening of the statement it was built from, and the producers it reads
    // come back with it.
    std::vector<char> retained(terms.size(), 1);
    for (auto const &[definition, sites] : dissolved_into) {
        retained[definition] = 0;
    }
    for (bool settled = false; !settled;) {
        settled = true;
        std::unordered_set<TensorId> still_read;
        for (std::size_t t = 0; t < terms.size(); t++) {
            if (retained[t] == 0) {
                continue;
            }
            if (terms[t].searchable) {
                for (auto const &factor : terms[t].factors) {
                    still_read.insert(factor.tensor);
                }
                continue;
            }
            if (expr.statements[t].value == invalid_term || expr.statements[t].value >= expr.terms.size()) {
                continue;
            }
            for (auto const operand : expr.at(expr.statements[t].value).operands) {
                still_read.insert(expr.at(operand).tensor);
            }
        }
        for (auto const &[definition, sites] : dissolved_into) {
            if (retained[definition] != 0 || !still_read.contains(expr.statements[definition].target)) {
                continue;
            }
            retained[definition] = 1;
            settled              = false;
        }
    }

    std::unordered_set<std::size_t> folded;
    std::size_t                     copies = 0;
    for (auto const &[definition, sites] : dissolved_into) {
        if (retained[definition] == 0) {
            // Moved rather than copied: the definition is gone, so the consumer that took its
            // place is where the value is computed and only the others are copies.
            folded.insert(definition);
            copies += sites.size() - 1;
            continue;
        }
        copies += sites.size();
        report(2, fmt::format("'{}' is kept and copied into {} consumer(s) that profit", expr.statements[definition].target_name,
                              sites.size()));
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

    // Every value a statement of this region writes, for the sharing gate in the search.
    std::set<ValueKey> region_written;
    for (auto const &statement : expr.statements) {
        region_written.insert(value_key(statement));
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

    // What the CAPTURED program pays for the statements this pass may rewrite: the bracketing the
    // author wrote, plus every definition the flattening dissolves, priced through the model the
    // search ranks its own trees with. Comparing against the searched cost instead would ask
    // whether a search improves on itself, which it never does, and would leave the pass unable to
    // fire on a re-bracketing that shares nothing.
    //
    // Each origin ONCE. A definition copied into two consumers appears in both their step lists
    // and is one node of the captured program, so a sum that took it twice would credit the
    // rewrite with removing arithmetic that was only ever there once.
    SymbolicCost                    captured;
    std::unordered_set<std::size_t> priced;
    for (std::size_t t = 0; t < terms.size(); t++) {
        if (!terms[t].searchable || folded.contains(t)) {
            continue;
        }
        for (auto const &step : terms[t].steps) {
            if (!priced.insert(step.origin).second) {
                continue;
            }
            captured = add_cost(captured, contraction_cost(step.left, step.right, step.out, table));
        }
    }

    /// One committed shared intermediate, in emission order.
    struct Shared {
        TensorId               tensor{};
        Factor                 left;
        Factor                 right;
        std::vector<ExprIndex> result;
        /// The per-member destinations when the shared value is a grouped one,
        /// empty otherwise. A shared intermediate carrying a member letter IS a
        /// grouped tensor and is allocated per member from the extent table.
        std::vector<TensorId> members;
    };
    std::vector<Shared> shared;
    /// What the committed shared intermediates themselves cost, which the per-term plans do not
    /// carry: a shared pair is one contraction outside every tree that reads it.
    SymbolicCost shared_total;

    /// The occurrences of each committed pair, in commit order, as the plan records them.
    std::vector<std::vector<std::array<std::size_t, 3>>> commit_log;

    // Which occurrence of a candidate pair, if any, is a statement that ALREADY computes it.
    //
    // A definition kept for one consumer and copied into another is a term of two factors, and a
    // consumer that wants exactly that product does not need a second tensor for it: the
    // definition is the shared intermediate. Without this the search declares one beside it and
    // the program forms one value twice, which is what a full-axis energy does the moment its
    // exchange half keeps the integral its opposite-spin half went past.
    //
    // Four things have to hold, and each of them is a way for the substitution to name a value the
    // tensor does not hold: the term must be exactly the pair, in the axis order the definition's
    // own target has; nothing else may write that tensor; it must be written outright rather than
    // accumulated into and without a prefactor; and every consumer must come after it in program
    // order.
    auto provider_site = [&](std::vector<PairSite> const &sites) -> std::optional<std::size_t> {
        std::optional<std::size_t> found;
        for (std::size_t index = 0; index < sites.size(); index++) {
            Term const &term = terms[sites[index].term];
            if (term.factors.size() != 2) {
                continue;
            }
            if (found.has_value()) {
                return std::nullopt; // two of them; the second would be left holding a copy
            }
            found = index;
        }
        if (!found.has_value()) {
            return std::nullopt;
        }
        PairSite const &site      = sites[*found];
        Term const     &term      = terms[site.term];
        auto const     &statement = expr.statements[term.statement];
        if (retained[site.term] == 0 || !term.searchable) {
            return std::nullopt;
        }
        auto const own = writer.find(value_key(statement));
        if (own == writer.end() || own->second != term.statement) {
            return std::nullopt;
        }
        if (!is_zero(statement.target_prefactor) || !is_one(term.factor)) {
            return std::nullopt;
        }
        if (!statement.operators.empty()) {
            return std::nullopt; // its target holds the permuted product, not the product
        }
        if (term.output.size() != site.result.size()) {
            return std::nullopt;
        }
        for (std::size_t axis = 0; axis < term.output.size(); axis++) {
            if (term.output[axis].letter != site.result[axis].letter) {
                return std::nullopt; // the same axes in another order is a permute, not a read
            }
        }
        for (auto const &other : sites) {
            if (other.term != site.term && terms[other.term].statement <= term.statement) {
                return std::nullopt;
            }
        }
        return found;
    };

    // Declare the intermediate one committed pair needs and rewrite its occurrences onto it. The
    // search reaches this after picking a winner and a replay reaches it straight away, which is
    // what makes the two produce the same graph rather than two graphs that agree on a test.
    // ── Declaring a grouped intermediate ───────────────────────────────────────────────────
    //
    // An intermediate carrying a member letter is a grouped tensor: one buffer per
    // member, sized from the family's own per-instance extents rather than from the
    // typical one the cost model reads. The member letter is not an axis of any
    // member's buffer, so it is skipped; what is left is that member's shape.
    std::set<std::string> member_letters;
    for (auto const &family : expr.families) {
        member_letters.insert(family.letter);
    }
    auto origin_letter = [&](std::string const &letter) {
        auto const hit = renamed_from.find(letter);
        return hit == renamed_from.end() ? letter : hit->second;
    };
    auto is_member_letter = [&](std::string const &letter) { return member_letters.contains(origin_letter(letter)); };
    auto per_member       = [&](std::string const &letter) -> std::vector<std::size_t> const       *{
        auto const hit = ragged_members.find(origin_letter(letter));
        return hit == ragged_members.end() ? nullptr : &hit->second;
    };
    auto declare_ragged = [&](std::vector<ExprIndex> const &axes, std::string const &stem, packed_gemm::ScalarType dtype,
                              std::size_t members) -> std::optional<std::vector<TensorId>> {
        std::vector<TensorId> ids;
        ids.reserve(members);
        for (std::size_t member = 0; member < members; member++) {
            std::vector<std::size_t> dims;
            for (auto const &index : axes) {
                if (is_member_letter(index.letter)) {
                    continue;
                }
                auto const *values = per_member(index.letter);
                if (values == nullptr || member >= values->size()) {
                    return std::nullopt;
                }
                dims.push_back((*values)[member]);
            }
            if (dims.empty()) {
                return std::nullopt;
            }
            TensorId const id = expr::declare_scratch(graph, fmt::format("{}_m{}", stem, member), dtype, dims);
            if (id == 0) {
                return std::nullopt;
            }
            ids.push_back(id);
        }
        return ids;
    };

    auto commit_pair = [&](std::vector<PairSite> const &sites, std::string_view label) -> bool {
        auto const  &first = sites.front();
        Factor const left  = terms[first.term].factors[first.left];
        Factor const right = terms[first.term].factors[first.right];

        // A statement that already computes this product IS the shared intermediate.
        auto const provider = provider_site(sites);
        if (provider.has_value()) {
            std::vector<std::array<std::size_t, 3>> record;
            record.reserve(sites.size());
            TensorId const held = expr.statements[terms[sites[*provider].term].statement].target;
            for (std::size_t index = 0; index < sites.size(); index++) {
                auto const &site = sites[index];
                record.push_back({site.term, site.left, site.right});
                if (index == *provider) {
                    continue; // its own statement is what emits the value
                }
                Term               &term = terms[site.term];
                Factor              placeholder{.tensor = held, .indices = site.result, .conjugate = false};
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
            _num_shared++;
            report(2, fmt::format("share {} across {} term(s), through the definition that already computes it", label, sites.size()));
            return true;
        }

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

        TensorHandle const *model = graph.find_tensor(left.ragged() ? left.members.front() : left.tensor);
        if (model == nullptr) {
            return false;
        }
        // The tensor is declared now, because a committed intermediate is one this pass will
        // emit, and a declaration for a candidate it merely considered would leave the graph
        // holding shells nothing writes.
        //
        // A shared value carrying a member letter is a GROUPED tensor: one buffer per member,
        // sized from the family's per-instance extents, since the typical extent the cost model
        // ranks with is not a shape anything runs at.
        std::vector<TensorId> shared_members;
        if (left.ragged()) {
            auto declared = declare_ragged(first.result, fmt::format("mtf_shared{}", shared.size()), model->dtype, left.members.size());
            if (!declared) {
                note_skip("a shared grouped candidate has an axis with no per-member extent", std::string{label});
                return false;
            }
            shared_members = std::move(*declared);
        }
        TensorId const shared_id =
            left.ragged() ? TensorId{} : expr::declare_scratch(graph, fmt::format("mtf_shared{}", shared.size()), model->dtype, dims);
        if (!left.ragged() && shared_id == 0) {
            return false;
        }
        // EVERY axis or none. `annotate_spaces` rightly refuses a hole in an annotation, so an
        // "any axis is valid" guard throws on the mixed case, which is what an intermediate over
        // one annotated space and one unannotated letter is; a decoupled energy has exactly that
        // shape as soon as its quadrature index is a space and something else is not.
        if (!left.ragged() && std::ranges::all_of(spaces, [](SpaceId id) { return id.valid(); })) {
            graph.annotate_spaces(shared_id, spaces);
        }
        if (!left.ragged() && every_axis_symbolic) {
            // Only when EVERY axis has a symbol: a partial annotation is what makes a bind move
            // some extents and not others, which is worse than none at all.
            graph.annotate_dims(shared_id, symbols);
        }

        std::vector<std::array<std::size_t, 3>> record;
        record.reserve(sites.size());
        for (auto const &site : sites) {
            record.push_back({site.term, site.left, site.right});
            Term               &term = terms[site.term];
            Factor              placeholder{.tensor = shared_id, .indices = site.result, .conjugate = false, .members = shared_members};
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
        shared.push_back(Shared{.tensor = shared_id, .left = left, .right = right, .result = first.result, .members = shared_members});
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
        for (auto const &pair : cached->retained_for) {
            fits = fits && pair[0] < terms.size() && pair[1] < terms.size();
        }
        for (auto const &commit : cached->commits) {
            fits = fits && !commit.empty();
            for (auto const &site : commit) {
                // Two factors is the occurrence a statement already computes, which keeps its
                // factors and gives the others a tensor to read; anything else loses its two and
                // gains one, so it needs a third to be left with.
                std::size_t const needs = factor_count[site[0]] == 2 ? std::size_t{2} : std::size_t{3};
                fits = fits && site[0] < terms.size() && terms[site[0]].searchable && factor_count[site[0]] >= needs && site[1] < site[2] &&
                       site[2] < factor_count[site[0]];
                if (!fits) {
                    break;
                }
                factor_count[site[0]] -= needs == 2 ? 0 : 1;
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
            // A two-factor term offers its one pair, which is the whole term. That is not a
            // candidate to build a new intermediate for, since committing it would leave a copy
            // behind; it is a candidate to READ, because the statement already computes the
            // product something else wants. `provider_site` is what says whether it may.
            if (!terms[t].searchable || retained[t] == 0 || terms[t].factors.size() < 2) {
                continue;
            }
            for (std::size_t i = 0; i + 1 < terms[t].factors.size(); i++) {
                for (std::size_t j = i + 1; j < terms[t].factors.size(); j++) {
                    // A shared intermediate is emitted at the front of the region, ahead of every
                    // statement, so it may only read factors nothing in the region writes: one that
                    // a statement writes would be read before its value exists.
                    if (region_written.contains(terms[t].factors[i].key()) || region_written.contains(terms[t].factors[j].key())) {
                        continue;
                    }
                    if (auto described = describe_pair(terms[t], i, j, t); described.has_value()) {
                        candidates[described->first].push_back(std::move(described->second));
                    }
                }
            }
        }

        SymbolicCost const     baseline = total_cost(plans);
        std::optional<PairKey> best_key;
        std::vector<PairSite>  best_sites;
        SymbolicCost           best_cost;
        std::vector<TreePlan>  best_plans;

        for (auto const &[key, offered] : candidates) {
            if (budget().expired()) {
                _cut_off = true;
                break;
            }
            if (offered.size() < 2) {
                continue;
            }
            // Reading a value a statement already computes is decided PER CONSUMER, where building
            // a new intermediate is decided for all the occurrences at once. The difference is that
            // the value exists either way: a consumer that is cheaper reading it takes it and one
            // that is cheaper rebuilding the product goes past it, which is the same shape the
            // per-consumer inlining decision has and the reason an energy's two halves can want
            // opposite things about one integral.
            std::vector<PairSite> sites;
            if (auto const offers = provider_site(offered); offers.has_value()) {
                sites.push_back(offered[*offers]);
                for (std::size_t index = 0; index < offered.size(); index++) {
                    if (index == *offers || offered[index].term == offered[*offers].term) {
                        continue;
                    }
                    Term const &term = terms[offered[index].term];
                    if (term.factors.size() < 3 || !plans[offered[index].term].ok) {
                        continue;
                    }
                    std::vector<Factor> reading;
                    reading.reserve(term.factors.size() - 1);
                    for (std::size_t f = 0; f < term.factors.size(); f++) {
                        if (f != offered[index].left && f != offered[index].right) {
                            reading.push_back(term.factors[f]);
                        }
                    }
                    reading.push_back(Factor{.tensor = TensorId{0}, .indices = offered[index].result, .conjugate = false});
                    TreePlan const takes = solve_tree(reading, term.output, table, ctx);
                    if (takes.ok && compare(takes.cost, plans[offered[index].term].cost, ctx) < 0) {
                        sites.push_back(offered[index]);
                    }
                }
                if (sites.size() < 2) {
                    continue;
                }
                std::ranges::sort(sites, [](PairSite const &lhs, PairSite const &rhs) { return lhs.term < rhs.term; });
            } else {
                sites = offered;
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
            auto const        provider = provider_site(sites);
            std::vector<Term> trial    = terms;
            bool              usable   = true;
            for (std::size_t index = 0; index < sites.size(); index++) {
                auto const &site = sites[index];
                Term       &term = trial[site.term];
                if (provider.has_value() && index == *provider) {
                    continue; // unchanged: its own statement is what computes the value
                }
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
                if (!trial[t].searchable || folded.contains(t)) {
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
            // The pair costs nothing extra when a statement already computes it: that cost is
            // inside the provider's own tree, which is in the total above.
            SymbolicCost const trial_cost =
                provider.has_value()
                    ? total_cost(trial_plans)
                    : add_cost(total_cost(trial_plans), contraction_cost(left_letters, right_letters, result_letters, table));

            if (compare(trial_cost, baseline, ctx) >= 0) {
                continue;
            }
            if (!best_key.has_value() || compare(trial_cost, best_cost, ctx) < 0) {
                best_key   = key;
                best_sites = sites;
                best_cost  = trial_cost;
                best_plans = std::move(trial_plans);
            }
        }

        if (!best_key.has_value()) {
            break;
        }

        if (!commit_pair(best_sites, best_key->text)) {
            break;
        }
        plans = std::move(best_plans);
    }

    // ── Emit ───────────────────────────────────────────────────────────────────────────────
    //
    // Nothing above touched the expression, so a decision to leave it alone costs nothing.
    if (!replayed && compare(add_cost(total_cost(plans), shared_total), captured, ctx) >= 0) {
        // With the numbers, because a decline that only says "nothing beat it" is the reporting
        // gap this pass's own before-and-after line exists to close.
        note_skip("no re-bracketing or sharing beats the captured form",
                  fmt::format("{} term(s) examined; the captured product(s) cost {} and the best tree(s) {}", terms.size(),
                              captured.flops.to_string(&graph.space_registry()),
                              add_cost(total_cost(plans), shared_total).flops.to_string(&graph.space_registry())));
        keep({});
        return false;
    }

    auto leaf_for = [&](Factor const &factor, std::string const &name) {
        ExprTerm leaf;
        leaf.kind    = TermKind::Leaf;
        leaf.tensor  = factor.tensor;
        leaf.members = factor.members;
        leaf.name    = name;
        return expr.add(std::move(leaf));
    };

    // Which family a letter belongs to, so an emitted statement carrying a member letter names
    // the family whose extents describe it. The flattener maps a definition's output letters onto
    // the consumer's positionally, so a member letter is never alpha-renamed; resolving through
    // the rename table anyway costs nothing and makes that an observation rather than a premise.
    std::map<std::string, FamilyId> family_of_letter;
    for (FamilyId id = 0; id < expr.families.size(); id++) {
        family_of_letter.emplace(expr.families[id].letter, id);
    }
    auto family_for = [&](std::vector<ExprIndex> const &indices) {
        for (auto const &index : indices) {
            auto const hit = family_of_letter.find(origin_letter(index.letter));
            if (hit != family_of_letter.end()) {
                return hit->second;
            }
        }
        return invalid_family;
    };

    std::vector<ExprStatement> emitted;
    bool                       ok_to_emit = true;
    emitted.reserve(expr.statements.size() + shared.size() + terms.size());
    // Unique across the REGIONS of one graph as well as across graphs. The counter restarts per
    // region and this pass descends into loop bodies, so a program with two regions would
    // otherwise declare two different tensors under one name; the storage auditor keys its
    // duplicate check on the name and reads that as one tensor allocated twice. The region's
    // first node id is what separates them, and it is as deterministic as the node order is.
    std::string const scratch_stem =
        fmt::format("{}_r{}", graph.name(), region.nodes.empty() ? std::size_t{0} : static_cast<std::size_t>(region.nodes.front()));
    std::size_t scratch_index = 0;

    // The emitted terms are priced in the letters the NODES will carry, which is not what the
    // search ranks against. The table above resolves an alpha-renamed letter to the axis it was
    // renamed from, because a fresh anonymous variable on one side of a comparison and not the
    // other is enough to make the scale-order rung decide on the naming; the cost the framework
    // checks against the emitted nodes has to name what those nodes name, and a node carries the
    // fresh letter because that is the letter that cannot collide with the consumer's.
    LetterTable emit_table = table;
    for (auto const &[fresh, origin] : renamed_from) {
        auto const variable = emit_table.var.find(fresh);
        if (variable == emit_table.var.end() || !variable->second.is_anonymous()) {
            continue;
        }
        variable->second = SymbolicVar::anonymous(fresh);
        if (auto const extent = emit_table.extent.find(fresh); extent != emit_table.extent.end()) {
            emit_table.anonymous_extent[fresh] = static_cast<double>(extent->second);
        }
    }

    auto emit_contraction = [&](Factor const &a, Factor const &b, TensorId target, std::vector<TensorId> const &targets,
                                std::string const &target_name, std::vector<ExprIndex> const &target_indices,
                                PrefactorScalar target_prefactor, PrefactorScalar factor, std::string const &label) {
        ExprTerm term =
            contraction_term({.term = leaf_for(a, {}), .indices = a.indices, .conjugate = a.conjugate},
                             {.term = leaf_for(b, {}), .indices = b.indices, .conjugate = b.conjugate}, target_indices, factor, emit_table);

        ExprStatement statement;
        statement.target           = target;
        statement.targets          = targets;
        statement.family           = targets.empty() ? invalid_family : family_for(target_indices);
        statement.target_name      = target_name;
        statement.target_indices   = target_indices;
        statement.target_prefactor = target_prefactor;
        statement.value            = expr.add(std::move(term));
        auto const free_axes =
            std::ranges::count_if(target_indices, [&](ExprIndex const &index) { return !is_member_letter(index.letter); });
        statement.origin_kind  = targets.empty() ? OpKind::Einsum : (free_axes == 0 ? OpKind::GroupedDot : OpKind::GroupedBatchedGemm);
        statement.origin_label = label;
        emitted.push_back(std::move(statement));
    };

    // A grouped term lowers onto a grouped kind or it is declined; it is NEVER emitted as a
    // per-member loop of ordinary nodes, because that loop is what the grouped family exists to
    // avoid. Two axes beside the member letter is a batched matrix product and none is a batched
    // reduction; anything else maps onto no grouped kind and the tree that produced it is refused
    // here, where the reason can name the shape, rather than at the lowering.
    auto grouped_shape_lowers = [&](Factor const &a, Factor const &b, std::vector<ExprIndex> const &indices) {
        auto const free_of = [&](std::vector<ExprIndex> const &list) {
            std::vector<std::string> out;
            for (auto const &index : list) {
                if (!is_member_letter(index.letter)) {
                    out.push_back(index.letter);
                }
            }
            return out;
        };
        auto const out   = free_of(indices);
        auto const left  = free_of(a.indices);
        auto const right = free_of(b.indices);
        // A batched reduction sums everything but the member letter, and its two operands run
        // over one index list; a batched matrix product has one free axis from each operand and
        // one link they share.
        if (out.empty()) {
            return left == right && !left.empty();
        }
        if (out.size() != 2 || left.size() != 2 || right.size() != 2) {
            return false;
        }
        std::set<std::string> const in_left(left.begin(), left.end());
        std::set<std::string> const in_right(right.begin(), right.end());
        std::size_t                 link = 0;
        for (auto const &letter : left) {
            link += in_right.contains(letter) && std::ranges::find(out, letter) == out.end() ? 1 : 0;
        }
        // Either assignment of the two free axes to the two operands: the lowering reads the
        // roles off the letters and reverses the pair where it has to, so a search that settled
        // on the other order is a matrix product all the same.
        return link == 1 &&
               ((in_left.contains(out[0]) && in_right.contains(out[1])) || (in_right.contains(out[0]) && in_left.contains(out[1])));
    };

    // The shared intermediates come first, in commit order, which is also dependency order: a
    // candidate over an already-shared factor could only be found after that one was committed.
    for (auto const &entry : shared) {
        if (!entry.members.empty() && !grouped_shape_lowers(entry.left, entry.right, entry.result)) {
            note_skip("a shared grouped candidate has a shape that maps onto no grouped kind",
                      fmt::format("{} axes beside the member letter", entry.result.size() - 1));
            ok_to_emit = false;
            break;
        }
        emit_contraction(entry.left, entry.right, entry.tensor, entry.members, {}, entry.result, PrefactorScalar{double{0}},
                         PrefactorScalar{double{1}}, fmt::format("mtf shared {}", entry.tensor));
    }

    bool ok = ok_to_emit;
    for (std::size_t t = 0; t < terms.size() && ok; t++) {
        auto const &statement = expr.statements[terms[t].statement];
        if (folded.contains(t)) {
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
                    if (outside.contains(index.letter) && seen.insert(index.letter).second) {
                        axes.push_back(index);
                    }
                }
            }

            if (mask == full) {
                // The outermost combine of a grouped term is gated too: re-bracketing changes what
                // its two operands are, so a family that raised as a batched matrix product can
                // reach this having become a shape no grouped kind computes.
                if (!statement.targets.empty() && !grouped_shape_lowers(*left, *right, term.output)) {
                    return std::nullopt;
                }
                // The MODELLED output, not the statement's own index list: a dot writes a scalar
                // and the tensor holding it has an axis of its own, which is not a letter of this
                // product and would name one the operands already use for something else.
                emit_contraction(*left, *right, statement.target, statement.targets, statement.target_name, term.output,
                                 statement.target_prefactor, term.factor, statement.origin_label);
                // The operators act on the finished value, so they go on the combine that writes
                // the target and on none of the intermediates beneath it.
                emitted.back().operators = statement.operators;
                return Factor{.tensor = statement.target, .indices = term.output, .conjugate = false, .members = statement.targets};
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
            TensorHandle const *model = graph.find_tensor(left->ragged() ? left->members.front() : left->tensor);
            if (model == nullptr || dims.empty()) {
                return std::nullopt;
            }
            if (left->ragged()) {
                if (!grouped_shape_lowers(*left, *right, axes)) {
                    return std::nullopt;
                }
                auto declared =
                    declare_ragged(axes, fmt::format("{}_mtf_t{}", scratch_stem, scratch_index++), model->dtype, left->members.size());
                if (!declared) {
                    return std::nullopt;
                }
                emit_contraction(*left, *right, TensorId{}, *declared, {}, axes, PrefactorScalar{double{0}}, PrefactorScalar{double{1}},
                                 fmt::format("mtf grouped t{}", scratch_index));
                return Factor{.tensor = TensorId{}, .indices = axes, .conjugate = false, .members = std::move(*declared)};
            }
            // Named after the graph it is declared in. This pass descends into loop bodies, so one
            // program holds one of these counters per graph and two graphs would otherwise declare
            // two different tensors under one name; the storage auditor keys its duplicate check on
            // the name and reads that as one tensor allocated twice.
            TensorId const scratch =
                expr::declare_scratch(graph, fmt::format("{}_mtf_t{}", scratch_stem, scratch_index++), model->dtype, dims);
            if (scratch == 0) {
                return std::nullopt;
            }
            if (std::ranges::all_of(spaces, [](SpaceId id) { return id.valid(); })) {
                graph.annotate_spaces(scratch, spaces);
            }
            emit_contraction(*left, *right, scratch, {}, {}, axes, PrefactorScalar{double{0}}, PrefactorScalar{double{1}},
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
    plan.retained_for.reserve(retained_for.size());
    for (auto const &[definition, site] : retained_for) {
        plan.retained_for.push_back({definition, site});
    }
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
    _num_copies += copies;
    EINSUMS_LOG_INFO("MultiTermFactorization: {} shared intermediate(s), {} term(s) re-bracketed, {} captured intermediate(s) dissolved, "
                     "{} copied into a consumer that profits",
                     _num_shared, _num_rebracketed, _num_inlined, _num_copies);
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
