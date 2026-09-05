//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraph/SymbolicCost.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ContractionTreeSearch.hpp"
#include "LaplaceRewrite.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The letters of an index list, in order.
std::vector<std::string> letters_of(std::vector<ExprIndex> const &indices) {
    std::vector<std::string> out;
    out.reserve(indices.size());
    for (auto const &index : indices) {
        out.push_back(index.letter);
    }
    return out;
}

bool contains(std::vector<std::string> const &haystack, std::string const &needle) {
    return std::ranges::find(haystack, needle) != haystack.end();
}

/// Append @p from's letters to @p into, skipping ones already there.
void merge_letters(std::vector<std::string> &into, std::vector<std::string> const &from) {
    for (auto const &letter : from) {
        if (!contains(into, letter)) {
            into.push_back(letter);
        }
    }
}

/// The bracketing search and the letter table it prices against. Shared with
/// `MultiTermFactorization`, which asks the same question of a captured chain: two
/// implementations would rank the same candidates by two rules.
using search::add_cost;
using search::Mask;

/// The largest number of leaves the bracketing search is offered.
///
/// The subset program is @c 3^N in this number, and a provider's chain plus the operand it is
/// contracted with is what makes up the leaves. A plan above the cap is declined rather than
/// bracketed by something weaker, which is the same bargain `MultiTermFactorization` strikes
/// with its own factor cap.
constexpr std::size_t kMaxPieces = 12;

/// Whether any node of @p graph writes the buffer @p id names.
///
/// The gate on which tensors may be factorized at all. A tensor the graph PRODUCES would have
/// to have its factors refitted whenever it changed, and the setup body this pass emits runs
/// once per bound problem; so only an operand nothing here writes is safely replaceable.
bool written_anywhere(Graph const &graph, TensorId id) {
    auto const owner = graph.resolve_alias(id);
    for (auto const &node : graph.nodes()) {
        for (TensorId const out : node.outputs) {
            if (graph.resolve_alias(out) == owner) {
                return true;
            }
        }
    }
    return false;
}

/// A letter not already in @p used, seeded from @p wanted.
std::string fresh_letter(std::vector<std::string> const &used, std::string const &wanted) {
    if (!contains(used, wanted)) {
        return wanted;
    }
    for (int suffix = 1;; ++suffix) {
        std::string candidate = fmt::format("{}{}", wanted, suffix);
        if (!contains(used, candidate)) {
            return candidate;
        }
    }
}

/// One factor, renamed into the letters the consuming contraction uses.
struct RenamedFactor {
    std::vector<ExprIndex>   indices;
    std::vector<std::string> letters;
};

/// Declare a graph-owned deferred intermediate of the given shape.
///
/// Deferred rather than eager, and runtime-rank rather than typed, for the two reasons every
/// pass-created tensor has both: the memory passes can only manage storage they are allowed to
/// place, and a later bind can only move an extent whose storage has not been committed.
TensorId declare_scratch(Graph &graph, std::string name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
    return detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) -> TensorId {
        auto &tensor = graph.declare_runtime_tensor<T>(std::move(name), dims, /*intermediate=*/true);
        return graph.live_tensor_id_by_ptr(&tensor, {});
    });
}

/// What the tree emitter needs to build one product's statements.
struct EmitRequest {
    Graph const                       *graph{nullptr};
    TensorExpr                        *expr{nullptr};
    std::vector<search::Factor> const *pieces{nullptr};
    search::TreePlan const            *tree{nullptr};
    search::LetterTable const         *table{nullptr};
    std::string                        provider; ///< For the interior names and the labels.
    std::string                        stem;     ///< The tagged tensor's name, for the interior names.
    packed_gemm::ScalarType            dtype{packed_gemm::ScalarType::Float64};

    /// The root statement's destination, exactly as the statement it replaces had it.
    TensorId               root_target{};
    std::string            root_name;
    std::vector<ExprIndex> root_indices;
    PrefactorScalar        root_prefactor{double{0}};
    PrefactorScalar        root_factor{double{1}};
    NodeId                 origin{0};

    /// Make a tensor of the given name and shape. A live run declares it on the graph; a trial
    /// invents an id, so costing a rewrite that is then declined leaves no shells behind.
    std::function<TensorId(std::string const &, packed_gemm::ScalarType, std::vector<std::size_t> const &)> make;
};

/// Emit the chosen bracketing as binary contractions.
///
/// Every interior node becomes a declared intermediate and a binary contraction, which is the
/// same binarization `MultiTermFactorization` does and for the same reason: @ref lower_region
/// has no multi-operand contraction to lower one to. The statements come back in dependency
/// order with the root last, and nothing is spliced: the caller decides where they go.
std::optional<std::vector<ExprStatement>> emit_tree(EmitRequest const &request) {
    std::set<std::string> const output_letters = search::letters_of(request.root_indices);
    std::vector<ExprStatement>  emitted;
    std::size_t                 scratch_index = 0;
    Mask const                  full          = static_cast<Mask>((Mask{1} << request.pieces->size()) - 1);
    TensorExpr                 &expr          = *request.expr;

    auto make_leaf = [&](TensorId id, std::vector<ExprIndex> indices) {
        ExprTerm            leaf;
        TensorHandle const *held = request.graph->find_tensor(id);
        leaf.kind                = TermKind::Leaf;
        leaf.tensor              = id;
        leaf.name                = held != nullptr ? held->name : std::string{};
        leaf.indices             = std::move(indices);
        return expr.add(std::move(leaf));
    };

    std::function<std::optional<search::Factor>(Mask)> build = [&](Mask mask) -> std::optional<search::Factor> {
        if (std::popcount(mask) == 1) {
            return (*request.pieces)[static_cast<std::size_t>(std::countr_zero(mask))];
        }
        if (request.tree->resolved[mask] == 0) {
            return std::nullopt;
        }
        auto const left  = build(request.tree->split[mask]);
        auto const right = build(mask ^ request.tree->split[mask]);
        if (!left.has_value() || !right.has_value()) {
            return std::nullopt;
        }

        // The axes this combine must expose, in first-appearance order over its operands.
        std::set<std::string> outside = output_letters;
        for (std::size_t piece = 0; piece < request.pieces->size(); ++piece) {
            if ((mask & (Mask{1} << piece)) == 0) {
                for (auto const &index : (*request.pieces)[piece].indices) {
                    outside.insert(index.letter);
                }
            }
        }
        std::vector<ExprIndex> axes;
        std::set<std::string>  seen;
        for (search::Factor const *operand : {&*left, &*right}) {
            for (auto const &index : operand->indices) {
                if (outside.count(index.letter) != 0 && seen.insert(index.letter).second) {
                    axes.push_back(index);
                }
            }
        }

        bool const  root = mask == full;
        TensorId    target{};
        std::string target_name;
        if (root) {
            target      = request.root_target;
            target_name = request.root_name;
        } else {
            std::vector<std::size_t> dims;
            dims.reserve(axes.size());
            for (auto const &index : axes) {
                auto const extent = request.table->extent.find(index.letter);
                if (extent == request.table->extent.end()) {
                    return std::nullopt;
                }
                dims.push_back(extent->second);
            }
            if (dims.empty()) {
                return std::nullopt; // a scalar intermediate; nothing here emits one
            }
            target_name = fmt::format("{}_{}_x{}", request.provider, request.stem, scratch_index++);
            target      = request.make(target_name, request.dtype, dims);
        }

        ExprTerm value;
        value.kind    = TermKind::Contraction;
        value.indices = root ? request.root_indices : axes;
        value.operands.assign({make_leaf(left->tensor, left->indices), make_leaf(right->tensor, right->indices)});
        value.operand_indices.assign({left->indices, right->indices});
        value.conjugate.assign({left->conjugate, right->conjugate});
        value.factor = root ? request.root_factor : PrefactorScalar{double{1}};
        // Priced the way a raised term is, so the region's before-and-after compares like with
        // like. An emitted term with no cost reads as free, and the report then offers a
        // rewrite to nothing as evidence that the search was worth making.
        value.cost = search::contraction_cost(search::letters_of(left->indices), search::letters_of(right->indices),
                                              search::letters_of(value.indices), *request.table);

        ExprStatement statement;
        statement.target           = target;
        statement.target_name      = target_name;
        statement.target_indices   = value.indices;
        statement.target_prefactor = root ? request.root_prefactor : PrefactorScalar{double{0}};
        statement.value            = expr.add(std::move(value));
        statement.origin           = request.origin;
        statement.origin_kind      = OpKind::Einsum;
        statement.origin_label =
            fmt::format("{}: {}[{}]", request.provider, target_name, fmt::join(letters_of(statement.target_indices), ","));
        emitted.push_back(std::move(statement));
        return search::Factor{.tensor = target, .indices = root ? request.root_indices : axes, .conjugate = false};
    };

    if (!build(full).has_value()) {
        return std::nullopt;
    }
    return emitted;
}

/// The cost of every contraction in an expression, over one letter table.
///
/// Written here rather than read off @ref TensorExpr::total_cost because the two sides of a
/// JOINT comparison have to be priced by one rule: a raised term's cost comes from its node's
/// descriptor and an emitted term's from this file, and a sum of the two would be a sum of two
/// opinions about what a letter's extent variable is called.
SymbolicCost expression_cost(Graph const &graph, TensorExpr const &expr,
                             std::unordered_map<TensorId, std::vector<std::size_t>> const &invented,
                             std::vector<std::pair<std::string, std::size_t>> const       &constants) {
    search::LetterTable table;
    auto const          dims_of = [&](TensorId id) -> std::vector<std::size_t> const          *{
        if (auto const found = invented.find(id); found != invented.end()) {
            return &found->second;
        }
        TensorHandle const *handle = graph.find_tensor(id);
        return handle != nullptr ? &handle->dims : nullptr;
    };
    auto const observe = [&](std::vector<ExprIndex> const &indices, TensorId id) {
        auto const *dims = dims_of(id);
        if (dims == nullptr) {
            return;
        }
        for (std::size_t axis = 0; axis < indices.size() && axis < dims->size(); ++axis) {
            table.observe(indices[axis], (*dims)[axis]);
        }
    };
    for (auto const &statement : expr.statements) {
        if (statement.value == invalid_term) {
            continue;
        }
        observe(statement.target_indices, statement.target);
        ExprTerm const &term = expr.at(statement.value);
        for (std::size_t slot = 0; slot < term.operands.size() && slot < term.operand_indices.size(); ++slot) {
            observe(term.operand_indices[slot], expr.at(term.operands[slot]).tensor);
        }
    }
    for (auto const &[letter, value] : constants) {
        table.observe_constant(letter, value);
    }

    SymbolicCost total;
    for (auto const &statement : expr.statements) {
        if (statement.value == invalid_term) {
            continue;
        }
        ExprTerm const &term = expr.at(statement.value);
        if (term.kind == TermKind::Contraction && term.operand_indices.size() == 2) {
            total = add_cost(total, search::contraction_cost(search::letters_of(term.operand_indices[0]),
                                                             search::letters_of(term.operand_indices[1]),
                                                             search::letters_of(statement.target_indices), table));
            continue;
        }
        if (term.kind != TermKind::Elementwise) {
            continue;
        }
        // An elementwise term is priced too, and only here. Everywhere else in this pass the
        // question is which of two contractions is cheaper and a direct product on both sides
        // cancels; here the question is whether a direct product is worth replacing BY a
        // contraction, and a model that priced one side at nothing would answer it by
        // comparing a number against zero.
        std::set<std::string> loop = search::letters_of(statement.target_indices);
        for (auto const operand : term.operands) {
            for (auto const &index : expr.at(operand).indices) {
                loop.insert(index.letter);
            }
        }
        SymbolicCost elementwise;
        elementwise.flops    = search::poly_over(loop, table);
        elementwise.traffic  = elementwise.flops;
        elementwise.resident = elementwise.flops;
        total                = add_cost(total, elementwise);
    }
    return total;
}

} // namespace

void FactorizationPass::reset_stats() {
    RegionRewrite::reset_stats();
    _num_factorized = 0;
    _num_dissolved  = 0;
    _num_joint      = 0;
    _pending.clear();
    _pending_quadrature.clear();
    _considered.clear();
}

FactorizationRegistry &FactorizationPass::registry() const {
    return _registry != nullptr ? *_registry : global_factorization_registry();
}

std::vector<std::string> FactorizationPass::describe() const {
    if (_num_factorized == 0) {
        return {};
    }
    return {fmt::format("FactorizationPass: re-associated {} contraction(s) around a provider's factors, dissolving {} captured "
                        "intermediate(s) into the cone, {} of them decided jointly with a quadrature",
                        _num_factorized, _num_dissolved, _num_joint)};
}

bool FactorizationPass::applicable(Graph const &graph) const {
    auto &known = registry();
    return std::ranges::any_of(
        graph.tensors_map(), [&known](auto const &entry) { return !entry.second.tag.name.empty() && known.claims(entry.second.tag.name); });
}

bool FactorizationPass::run(Graph &graph) {
    _pending.clear();
    _considered.clear();
    bool modified = RegionRewrite::run(graph);

    // After the region loop, never inside it: a region is a range of positions in the node
    // vector and those positions stay live for the whole of the loop above.
    //
    // Position 0, which is correct by construction rather than by luck: the pass only accepts a
    // tagged tensor no node writes, so a fitting reading it depends on nothing this graph
    // computes and every reader of the factors comes later.
    for (auto const &pending : _pending) {
        Graph &body = graph.add_setup_at(pending.label, 0);
        pending.emit(graph, body, pending.factors);
        modified = true;
    }
    // The quadratures a joint rewrite fitted, behind the fittings for the same reason those go
    // at the front: a body reading tensors nothing here produces depends on nothing this graph
    // computes, and every reader of what it writes comes later.
    for (auto const &pending : _pending_quadrature) {
        Graph &body = graph.add_setup_at(pending.label, 0);
        pending.emit(graph, body);
        modified = true;
    }
    if (!_pending.empty() || !_pending_quadrature.empty()) {
        report(1, fmt::format("emitted {} setup bod(y/ies) holding the fittings and {} holding a quadrature", _pending.size(),
                              _pending_quadrature.size()));
    }
    _pending.clear();
    _pending_quadrature.clear();

    // A tag no contraction ever offered is a decline rather than a silence. Every other refusal
    // here reports itself because it happens with a candidate in hand; this one happens because
    // there was never a candidate, and it is the commonest thing a caller tagging an integral
    // gets: a tensor read only elementwise, or only through a dot, is not a contraction operand
    // and so nothing in the region loop ever looks at it.
    std::vector<std::string> unclaimed;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.tag.name.empty() || !registry().claims(handle.tag.name)) {
            continue;
        }
        if (std::ranges::find(_considered, id) == _considered.end()) {
            unclaimed.push_back(handle.name);
        }
    }
    std::ranges::sort(unclaimed);
    for (auto const &name : unclaimed) {
        note_skip("a tagged tensor is read by no two-operand contraction, so there is nothing to re-associate around its factors",
                  fmt::format("tensor '{}'", name));
    }
    _considered.clear();
    return modified;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity): the decomposition is one argument
// and splitting it would put the halves out of sight of each other.
bool FactorizationPass::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    bool changed = false;

    ComparisonContext ctx;
    ctx.registry = &graph.space_registry();

    // ── The joint shape, first ────────────────────────────────────────────────────────────
    //
    // A tagged tensor multiplied by a tagged denominator is the one candidate whose worth
    // cannot be decided here alone, so it is decided with the quadrature that follows it. A
    // whole rescan after each one, because the accepted rewrite dissolves a statement and
    // splices several, and every index into the list is stale afterwards. Bounded by the number
    // of such products, which is small, and terminating because an accepted rewrite leaves no
    // direct product reading the tagged tensor.
    for (bool joint = _laplace != nullptr; joint;) {
        joint = false;
        for (std::size_t position = 0; position < expr.statements.size(); ++position) {
            if (rewrite_denominator_product(graph, region, expr, position).has_value()) {
                changed = true;
                joint   = true;
                break;
            }
        }
    }

    // Index-based, because an accepted rewrite INSERTS statements and the loop has to skip
    // past what it just added rather than reconsider them.
    for (std::size_t position = 0; position < expr.statements.size(); ++position) {
        ExprStatement const statement = expr.statements[position];
        if (statement.value == invalid_term) {
            continue;
        }
        ExprTerm const term = expr.at(statement.value);
        if (term.kind != TermKind::Contraction || term.operands.size() != 2 || term.operand_indices.size() != 2) {
            continue;
        }

        // Which operand, if either, carries a tag some provider claims.
        std::size_t tagged_slot = 2;
        std::string tag;
        for (std::size_t slot = 0; slot < 2; ++slot) {
            TensorHandle const *handle = graph.find_tensor(expr.at(term.operands[slot]).tensor);
            if (handle != nullptr && !handle->tag.name.empty() && registry().claims(handle->tag.name)) {
                tagged_slot = slot;
                tag         = handle->tag.name;
                break;
            }
        }
        if (tagged_slot == 2) {
            continue;
        }
        std::size_t const other_slot   = 1 - tagged_slot;
        TensorId const    tagged_id    = expr.at(term.operands[tagged_slot]).tensor;
        std::string const tagged_name  = expr.at(term.operands[tagged_slot]).name;
        TermId const      other_leaf   = term.operands[other_slot];
        TensorId const    other_id     = expr.at(other_leaf).tensor;
        auto const        tagged_index = term.operand_indices[tagged_slot];
        auto const        other_index  = term.operand_indices[other_slot];

        if (std::ranges::find(_considered, tagged_id) == _considered.end()) {
            _considered.push_back(tagged_id);
        }

        if (written_anywhere(graph, tagged_id)) {
            note_skip("the tagged tensor is written by this graph, so its factors could go stale", fmt::format("tensor '{}'", tagged_name));
            continue;
        }
        if (!term.conjugate.empty() && term.conjugate[tagged_slot]) {
            note_skip("the tagged operand is conjugated, which a factor chain has nowhere to carry",
                      fmt::format("tensor '{}'", tagged_name));
            continue;
        }

        // Every letter this statement already uses, so a provider's new ones cannot collide.
        std::vector<std::string> used = letters_of(tagged_index);
        merge_letters(used, letters_of(other_index));
        merge_letters(used, letters_of(statement.target_indices));

        // ── The cone the tagged operand sits in ───────────────────────────────────────
        //
        // The other operand may itself be an intermediate this region formed, and the name the
        // author gave it encodes the author's bracketing rather than the problem's. Flattening
        // it in puts its own factors among the leaves the search sees, so the substituted
        // factors and the cone's factors are bracketed TOGETHER rather than either side being
        // fixed first. Where keeping the intermediate is the cheaper answer the search says so
        // on its own, since re-forming it is one of the trees over those leaves.
        std::vector<search::Factor>                        outer_pieces;
        std::vector<std::size_t>                           dissolved;        ///< Statements folded in.
        std::vector<std::array<std::vector<ExprIndex>, 3>> dissolved_shapes; ///< Their left, right and output axes.
        {
            // Reader and writer counts over the whole expression, taken here rather than once
            // per region because an accepted rewrite moves both.
            std::unordered_map<TensorId, std::size_t> readers;
            std::unordered_map<TensorId, std::size_t> writer;
            for (std::size_t other = 0; other < expr.statements.size(); ++other) {
                auto const [entry, fresh] = writer.try_emplace(expr.statements[other].target, other);
                if (!fresh) {
                    entry->second = expr.statements.size(); // more than one writer: never fold
                }
                if (expr.statements[other].value == invalid_term) {
                    continue;
                }
                for (TermId const operand : expr.at(expr.statements[other].value).operands) {
                    if (expr.at(operand).kind == TermKind::Leaf) {
                        ++readers[expr.at(operand).tensor];
                    }
                }
            }

            std::size_t                                                                                 fresh_index = 0;
            std::function<void(TermId, std::vector<ExprIndex> const &, bool, std::size_t, std::size_t)> expand =
                [&](TermId leaf_id, std::vector<ExprIndex> const &as_seen, bool conjugate, std::size_t depth, std::size_t consumer) {
                    ExprTerm const leaf      = expr.at(leaf_id);
                    auto const     keep_leaf = [&]() {
                        outer_pieces.push_back(search::Factor{.tensor = leaf.tensor, .indices = as_seen, .conjugate = conjugate});
                    };
                    // A conjugated leaf is never folded. Conjugation does distribute over a
                    // product, but carrying the flag onto every factor is a rewrite of its own
                    // and declining costs one opportunity rather than risking a wrong sign.
                    if (leaf.kind != TermKind::Leaf || depth >= 8 || conjugate) {
                        keep_leaf();
                        return;
                    }
                    if (std::ranges::find(region.internal, leaf.tensor) == region.internal.end() || readers[leaf.tensor] != 1) {
                        keep_leaf();
                        return;
                    }
                    auto const found = writer.find(leaf.tensor);
                    if (found == writer.end() || found->second >= consumer) {
                        keep_leaf();
                        return;
                    }
                    ExprStatement const definition = expr.statements[found->second];
                    if (definition.value == invalid_term || !is_zero(definition.target_prefactor) ||
                        definition.target_indices.size() != as_seen.size()) {
                        keep_leaf();
                        return;
                    }
                    ExprTerm const value = expr.at(definition.value);
                    if (value.kind != TermKind::Contraction || value.operands.size() != 2 || value.operand_indices.size() != 2 ||
                        !is_one(value.factor) || expr.at(value.operands[0]).kind != TermKind::Leaf ||
                        expr.at(value.operands[1]).kind != TermKind::Leaf) {
                        keep_leaf();
                        return;
                    }

                    // The definition's own letters, mapped onto the names the consumer uses for
                    // the same axes; everything else it mentions is summed inside it and gets a
                    // name nothing else here has.
                    std::unordered_map<std::string, ExprIndex> substitution;
                    for (std::size_t axis = 0; axis < as_seen.size(); ++axis) {
                        substitution.emplace(definition.target_indices[axis].letter, as_seen[axis]);
                    }
                    auto rename = [&](std::vector<ExprIndex> const &indices) {
                        std::vector<ExprIndex> out;
                        out.reserve(indices.size());
                        for (auto const &index : indices) {
                            auto const [entry, fresh] = substitution.try_emplace(index.letter, index);
                            if (fresh) {
                                entry->second.letter = fmt::format("~{}", fresh_index++);
                            }
                            out.push_back(entry->second);
                        }
                        return out;
                    };
                    std::vector<ExprIndex> const left  = rename(value.operand_indices[0]);
                    std::vector<ExprIndex> const right = rename(value.operand_indices[1]);

                    dissolved.push_back(found->second);
                    dissolved_shapes.push_back({left, right, as_seen});
                    expand(value.operands[0], left, !value.conjugate.empty() && value.conjugate[0], depth + 1, found->second);
                    expand(value.operands[1], right, !value.conjugate.empty() && value.conjugate[1], depth + 1, found->second);
                };
            expand(other_leaf, other_index, !term.conjugate.empty() && term.conjugate[other_slot], 0, position);

            // A folded definition's operands were read where it stood and are read where the
            // tagged contraction stands instead, so a write to one of them in between would be
            // seen by the rewrite and was not seen by the program. Falling back to the
            // unflattened operand costs the wider search rather than the rewrite.
            if (!dissolved.empty()) {
                std::size_t const earliest     = *std::ranges::min_element(dissolved);
                bool              interference = false;
                for (std::size_t between = earliest; between < position; ++between) {
                    if (std::ranges::find(dissolved, between) != dissolved.end()) {
                        continue;
                    }
                    for (auto const &piece : outer_pieces) {
                        interference = interference || expr.statements[between].target == piece.tensor;
                    }
                }
                if (interference) {
                    note_skip("a statement between an intermediate's definition and its use rewrites one of its operands, so the "
                              "cone was not flattened",
                              fmt::format("tensor '{}'", tagged_name));
                    outer_pieces.clear();
                    dissolved.clear();
                    dissolved_shapes.clear();
                    outer_pieces.push_back(search::Factor{
                        .tensor = other_id, .indices = other_index, .conjugate = !term.conjugate.empty() && term.conjugate[other_slot]});
                }
            }
        }

        struct Candidate {
            FactorizationPlan           plan;
            std::vector<RenamedFactor>  factors;
            std::vector<search::Factor> pieces; ///< The factors, then the other operand.
            search::LetterTable         table;
            search::TreePlan            tree;
            SymbolicCost                cost;
        };
        std::optional<Candidate> best;

        for (auto const &provider : registry().for_tag(tag)) {
            auto offer = provider->propose(graph, tagged_id);
            if (!offer) {
                note_skip("a provider declined", fmt::format("'{}': {}", provider->name(), offer.error()));
                continue;
            }
            FactorizationPlan plan = std::move(*offer);
            if (plan.factors.size() < 2) {
                note_skip("a provider offered fewer than two factors, which is a rename rather than a factorization",
                          fmt::format("'{}'", provider->name()));
                continue;
            }
            if (plan.factors.size() + 1 > kMaxPieces) {
                note_skip("a provider's chain has more factors than the bracketing search is allowed",
                          fmt::format("'{}': {} factor(s), cap {}", provider->name(), plan.factors.size(), kMaxPieces - 1));
                continue;
            }
            if (plan.tagged_letters.size() != tagged_index.size()) {
                note_skip("a provider's letter list does not match the tagged operand's rank", fmt::format("'{}'", provider->name()));
                continue;
            }

            // The rename. A provider letter naming one of the tagged tensor's axes becomes the
            // letter this contraction spells that axis with; anything else is a new letter and
            // gets one nothing here is using.
            std::unordered_map<std::string, std::string> rename;
            std::unordered_map<std::string, SpaceId>     new_spaces;
            for (std::size_t axis = 0; axis < plan.tagged_letters.size(); ++axis) {
                rename.emplace(plan.tagged_letters[axis], tagged_index[axis].letter);
            }
            std::vector<std::string> taken = used;
            bool                     ok    = true;
            for (auto const &factor : plan.factors) {
                if (factor.letters.size() != factor.dims.size()) {
                    note_skip("a provider's factor has a different number of letters and extents", fmt::format("'{}'", provider->name()));
                    ok = false;
                    break;
                }
                for (std::size_t axis = 0; axis < factor.letters.size(); ++axis) {
                    if (rename.contains(factor.letters[axis])) {
                        continue;
                    }
                    std::string fresh = fresh_letter(taken, factor.letters[axis]);
                    taken.push_back(fresh);
                    rename.emplace(factor.letters[axis], fresh);
                    if (axis < factor.spaces.size() && !factor.spaces[axis].empty()) {
                        if (auto const space = graph.space_registry().find(factor.spaces[axis]); space.has_value()) {
                            new_spaces.emplace(fresh, *space);
                        }
                    }
                }
            }
            if (!ok) {
                continue;
            }

            std::size_t const          count = plan.factors.size();
            std::vector<RenamedFactor> renamed(count);
            for (std::size_t which = 0; which < count; ++which) {
                for (auto const &letter : plan.factors[which].letters) {
                    std::string const &mapped = rename.at(letter);
                    SpaceId            space;
                    if (auto const found = new_spaces.find(mapped); found != new_spaces.end()) {
                        space = found->second;
                    } else {
                        for (auto const &index : tagged_index) {
                            if (index.letter == mapped) {
                                space = index.space;
                                break;
                            }
                        }
                    }
                    renamed[which].indices.push_back(ExprIndex{.letter = mapped, .space = space});
                    renamed[which].letters.push_back(mapped);
                }
            }

            // Every letter's actual extent, for the search's own ranking and for the numeric
            // veto below. Read off the operands this contraction already has, and off the plan
            // for the letters it introduces.
            //
            // An axis carrying a SYMBOLIC extent is noted too, because a capture-time number
            // for such an axis is a placeholder for whatever the next bind supplies rather
            // than a size anything runs at.
            search::LetterTable table;
            bool                any_symbolic_extent = false;
            auto const          note_extent         = [&](TensorId id, std::vector<ExprIndex> const &indices) {
                TensorHandle const *handle = graph.find_tensor(id);
                if (handle == nullptr) {
                    return;
                }
                for (std::size_t axis = 0; axis < indices.size() && axis < handle->dims.size(); ++axis) {
                    table.observe(indices[axis], handle->dims[axis]);
                    // An EMPTY dim_symbols means every axis is literal; an empty ENTRY means
                    // this one is. Anything else is a symbol or a ragged axis, both resizable.
                    if (axis < handle->dim_symbols.size() && !handle->dim_symbols[axis].empty()) {
                        any_symbolic_extent = true;
                    }
                }
            };
            note_extent(tagged_id, tagged_index);
            note_extent(other_id, other_index);
            note_extent(statement.target, statement.target_indices);
            for (auto const &piece : outer_pieces) {
                note_extent(piece.tensor, piece.indices);
            }
            for (std::size_t which = 0; which < count; ++which) {
                for (std::size_t axis = 0; axis < renamed[which].indices.size(); ++axis) {
                    table.observe(renamed[which].indices[axis], plan.factors[which].dims[axis]);
                }
            }

            // A two-factor plan has to SEPARATE the letters shared with the other operand: one
            // factor carrying all of them and one carrying none. Both or neither and there is
            // no regrouping to make, only a substitution, which is strictly more arithmetic.
            //
            // Stated for two factors and not for a chain, because it is a two-factor statement.
            // A chain has as many ways to meet the other operand as it has links, and which of
            // them pays is what the bracketing search answers; asking a chain to separate would
            // decline exactly the plans the search exists to bracket.
            if (count == 2) {
                std::vector<std::string> shared;
                for (auto const &index : tagged_index) {
                    if (contains(letters_of(other_index), index.letter)) {
                        shared.push_back(index.letter);
                    }
                }
                auto const carries_all = [&](RenamedFactor const &factor) {
                    return std::ranges::all_of(shared, [&factor](std::string const &letter) { return contains(factor.letters, letter); });
                };
                auto const carries_none = [&](RenamedFactor const &factor) {
                    return std::ranges::none_of(shared, [&factor](std::string const &letter) { return contains(factor.letters, letter); });
                };
                bool separated = false;
                if (!shared.empty()) {
                    separated =
                        (carries_all(renamed[1]) && carries_none(renamed[0])) || (carries_all(renamed[0]) && carries_none(renamed[1]));
                }
                if (!separated) {
                    note_skip("the split does not separate the letters shared with the other operand",
                              fmt::format("'{}' on '{}'", provider->name(), tagged_name));
                    continue;
                }
            }

            // The target must still be producible from the pieces. Checked rather than reasoned
            // about: the shapes that reach here are wider than the tidy ones, and a miss would
            // emit a contraction whose operands cannot make its output.
            std::vector<std::string> reachable;
            for (auto const &piece : outer_pieces) {
                merge_letters(reachable, letters_of(piece.indices));
            }
            for (auto const &factor : renamed) {
                merge_letters(reachable, factor.letters);
            }
            if (!std::ranges::all_of(letters_of(statement.target_indices),
                                     [&reachable](std::string const &letter) { return contains(reachable, letter); })) {
                note_skip("the decomposed form cannot produce the target's indices",
                          fmt::format("'{}' on '{}'", provider->name(), tagged_name));
                continue;
            }

            // The leaves the bracketing search is offered: the provider's factors, and the
            // operand the tagged tensor was being contracted with. The auxiliary letters are
            // ordinary letters to it, so a bracketing that rebuilds the tagged tensor is simply
            // the most expensive tree and the search never picks it.
            std::vector<search::Factor> pieces;
            pieces.reserve(count + outer_pieces.size());
            for (auto const &factor : renamed) {
                pieces.push_back(search::Factor{.tensor = TensorId{0}, .indices = factor.indices, .conjugate = false});
            }
            pieces.insert(pieces.end(), outer_pieces.begin(), outer_pieces.end());
            if (pieces.size() > kMaxPieces) {
                note_skip("the cone and the provider's chain together have more leaves than the bracketing search is allowed",
                          fmt::format("'{}': {} leaves, cap {}", provider->name(), pieces.size(), kMaxPieces));
                continue;
            }

            ComparisonContext tree_ctx;
            tree_ctx.registry     = ctx.registry;
            tree_ctx.bound_extent = table.lookup();

            search::TreePlan const tree = search::solve_tree(pieces, statement.target_indices, table, tree_ctx);
            if (!tree.ok) {
                note_skip("no bracketing of the substituted product could be found",
                          fmt::format("'{}' on '{}'", provider->name(), tagged_name));
                continue;
            }

            // What the region costs today: the tagged contraction, plus every statement the
            // flattening dissolved, because those disappear if this rewrite is taken and a
            // comparison that ignored them would decline a tree that pays for both.
            SymbolicCost before = search::contraction_cost(search::letters_of(tagged_index), search::letters_of(other_index),
                                                           search::letters_of(statement.target_indices), table);
            for (auto const &shape : dissolved_shapes) {
                before = add_cost(before, search::contraction_cost(search::letters_of(shape[0]), search::letters_of(shape[1]),
                                                                   search::letters_of(shape[2]), table));
            }
            SymbolicCost const after = tree.cost;

            // Cheaper SYMBOLICALLY and, where every extent is known, cheaper at those extents
            // too. The two answer different questions and the pass needs both.
            //
            // The symbolic one is about the family: two degree-three contractions beat one of
            // degree four however large the problem grows, which is the whole DF argument and
            // is what a saved graph rebound to a bigger problem will rely on.
            //
            // It is also blind to constants, and an auxiliary index larger than the product it
            // replaces is a real case rather than a pathological one. A pass that rewrote a
            // graph into something slower at the size it was captured at would be trading a
            // measurable regression for a promise, so the bound extents get a veto.
            if (compare(after, before, ctx) >= 0) {
                note_skip("the decomposed form is not symbolically cheaper",
                          fmt::format("'{}' on '{}': {} vs {}", provider->name(), tagged_name, after.flops.to_string(ctx.registry),
                                      before.flops.to_string(ctx.registry)));
                continue;
            }

            // The bound-extent veto applies only where the capture extents ARE the problem
            // size. An axis annotated symbolic is one a later bind may resize, so a number
            // read off it here describes a placeholder geometry rather than anything that
            // will run. Vetoing on such a number would let a graph captured small delete a
            // factorization the family it was annotated for needs, and delete it at the
            // capture, before the save, rather than at the bind that would have wanted it.
            // Where nothing is annotated the capture size is the contract and the veto stands.
            ExtentLookup const extent_of    = table.lookup();
            auto const         before_flops = before.flops.evaluate(extent_of);
            auto const         after_flops  = after.flops.evaluate(extent_of);
            if (any_symbolic_extent) {
                report(2, fmt::format("the extent veto abstains on '{}' ({}): a symbolic axis makes the captured size a "
                                      "placeholder, so the symbolic verdict stands alone",
                                      tagged_name, provider->name()));
            } else if (before_flops.has_value() && after_flops.has_value() && *after_flops >= *before_flops) {
                note_skip("the decomposed form is not cheaper at the extents this graph holds",
                          fmt::format("'{}' on '{}': {:g} vs {:g} flops", provider->name(), tagged_name, *after_flops, *before_flops));
                continue;
            }
            if (best.has_value() && compare(after, best->cost, ctx) >= 0) {
                continue;
            }

            best = Candidate{.plan    = std::move(plan),
                             .factors = std::move(renamed),
                             .pieces  = std::move(pieces),
                             .table   = std::move(table),
                             .tree    = tree,
                             .cost    = after};
        }

        if (!best.has_value()) {
            continue;
        }

        // The accuracy statement, before anything is rewritten. A refusal here leaves the
        // statement exactly as it was and puts the budget's own reason in the skip tally.
        ApproximationRecord record = best->plan.accuracy;
        record.pass_name           = best->plan.provider;
        if (record.setup.empty()) {
            record.setup = fmt::format("{}({})", best->plan.provider, tagged_name);
        }
        if (!approximate(graph, record)) {
            continue;
        }

        // Create the factors. Tensors only: a node added here would move the region out from
        // under the splice that is about to replace it.
        //
        // Factors under ONE name are one tensor, which is not a shortcut but the commonest
        // case there is: a metric-fitted factorization writes its tensor as B[Q,m,n] B[Q,p,q],
        // the same B twice with different letters, and a tensor-hypercontraction chain writes
        // one collocation matrix four times. Creating one per mention would fit the same thing
        // several times and store the largest tensor in the calculation several times. A
        // contraction whose operands share a tensor is already legal, so nothing downstream
        // needs to know.
        std::size_t const                             count = best->plan.factors.size();
        std::vector<TensorId>                         factor_ids(count);
        std::vector<std::pair<std::string, TensorId>> declared;
        for (std::size_t which = 0; which < count; ++which) {
            std::string const &factor_name = best->plan.factors[which].name;
            auto const found = std::ranges::find_if(declared, [&factor_name](auto const &entry) { return entry.first == factor_name; });
            if (found != declared.end()) {
                factor_ids[which] = found->second;
                continue;
            }
            factor_ids[which] = declare_scratch(graph, fmt::format("{}_{}", best->plan.provider, factor_name),
                                                best->plan.factors[which].dtype, best->plan.factors[which].dims);
            declared.emplace_back(factor_name, factor_ids[which]);
        }
        for (std::size_t which = 0; which < count; ++which) {
            best->pieces[which].tensor = factor_ids[which];
        }

        // ── Emit the tree the search chose ────────────────────────────────────────────
        EmitRequest request;
        request.graph          = &graph;
        request.expr           = &expr;
        request.pieces         = &best->pieces;
        request.tree           = &best->tree;
        request.table          = &best->table;
        request.provider       = best->plan.provider;
        request.stem           = tagged_name;
        request.dtype          = best->plan.factors[0].dtype;
        request.root_target    = statement.target;
        request.root_name      = statement.target_name;
        request.root_indices   = statement.target_indices;
        request.root_prefactor = statement.target_prefactor;
        request.root_factor    = term.factor;
        request.origin         = statement.origin;
        request.make           = [&graph](std::string const &name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
            return declare_scratch(graph, name, dtype, dims);
        };

        auto emitted = emit_tree(request);
        if (!emitted.has_value()) {
            // Nothing above touched the statement list, so a tree that will not build costs
            // this candidate and leaves the region as it was.
            note_skip("the chosen bracketing could not be emitted", fmt::format("'{}' on '{}'", best->plan.provider, tagged_name));
            continue;
        }

        expr.statements[position] = std::move(emitted->back());
        emitted->pop_back();
        expr.statements.insert(expr.statements.begin() + static_cast<std::ptrdiff_t>(position), emitted->begin(), emitted->end());
        position += emitted->size(); // past the statements just inserted, onto the one they feed

        // The definitions the flattening folded in are gone: their values now live inside the
        // tree. Erased back to front so the earlier indices stay valid, and every one of them
        // is before the insertion point, which is what keeps `position` meaningful.
        std::ranges::sort(dissolved);
        for (auto entry = dissolved.rbegin(); entry != dissolved.rend(); ++entry) {
            expr.statements.erase(expr.statements.begin() + static_cast<std::ptrdiff_t>(*entry));
            --position;
        }
        _num_dissolved += dissolved.size();

        _pending.push_back(PendingSetup{.label = record.setup, .factors = factor_ids, .emit = best->plan.emit_setup});
        ++_num_factorized;
        changed = true;
        report(2, fmt::format("factorized '{}' through {}", tagged_name, best->plan.provider));
    }

    (void)region;
    if (changed) {
        EINSUMS_LOG_INFO("FactorizationPass: re-associated {} contraction(s)", _num_factorized);
    }
    return changed;
}

/// Emit the substituted product into @p numerator and point the direct product at it.
///
/// Shared by the trial and the rewrite that follows it, because a trial priced by one code path
/// and applied by another is a measurement of the wrong program.
namespace {

bool substitute_product_operand(
    Graph const &graph, TensorExpr &expr, std::size_t position, std::size_t tagged_slot, std::string const &provider,
    std::string const &tagged_name, FactorizationPlan const &plan, std::vector<search::Factor> const &pieces, search::TreePlan const &tree,
    search::LetterTable const &table, std::vector<ExprIndex> const &tagged_index, TensorId numerator,
    std::function<TensorId(std::string const &, packed_gemm::ScalarType, std::vector<std::size_t> const &)> const &make) {
    ExprStatement const statement = expr.statements[position];

    EmitRequest request;
    request.graph          = &graph;
    request.expr           = &expr;
    request.pieces         = &pieces;
    request.tree           = &tree;
    request.table          = &table;
    request.provider       = provider;
    request.stem           = tagged_name;
    request.dtype          = plan.factors[0].dtype;
    request.root_target    = numerator;
    request.root_name      = fmt::format("{}_{}_num", provider, tagged_name);
    request.root_indices   = tagged_index;
    request.root_prefactor = PrefactorScalar{double{0}};
    request.root_factor    = PrefactorScalar{double{1}};
    request.origin         = statement.origin;
    request.make           = make;

    auto emitted = emit_tree(request);
    if (!emitted.has_value()) {
        return false;
    }

    // The direct product now reads the substituted product rather than the tagged tensor. An id
    // substitution rather than an index rewrite, because the numerator has the tagged tensor's
    // own index list by construction.
    ExprTerm updated = expr.at(statement.value);
    ExprTerm leaf;
    leaf.kind                       = TermKind::Leaf;
    leaf.tensor                     = numerator;
    leaf.name                       = request.root_name;
    leaf.indices                    = tagged_index;
    updated.operands[tagged_slot]   = expr.add(std::move(leaf));
    expr.statements[position].value = expr.add(std::move(updated));

    expr.statements.insert(expr.statements.begin() + static_cast<std::ptrdiff_t>(position), emitted->begin(), emitted->end());
    return true;
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity): the joint decision is one argument
// and splitting it would put the halves out of sight of each other.
std::optional<std::size_t> FactorizationPass::rewrite_denominator_product(Graph &graph, Region const &region, TensorExpr &expr,
                                                                          std::size_t position) {
    ExprStatement const statement = expr.statements[position];
    ExprTerm const      product   = expr.at(statement.value);
    if (product.kind != TermKind::Elementwise || product.element_kind != OpKind::DirectProduct || product.operands.size() != 2) {
        return std::nullopt;
    }

    // One operand tagged for a provider, the other a Laplace denominator. That pairing is what
    // makes this shape the one neither pass can go first on: the fit has no contraction to
    // re-associate until it substitutes, and the transform has no contraction to ride on until
    // the fit has substituted.
    std::size_t tagged_slot = 2;
    std::string tag;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        TensorHandle const *handle = graph.find_tensor(expr.at(product.operands[slot]).tensor);
        TensorHandle const *other  = graph.find_tensor(expr.at(product.operands[1 - slot]).tensor);
        if (handle != nullptr && other != nullptr && !handle->tag.name.empty() && registry().claims(handle->tag.name) &&
            other->tag.name == LaplaceTransform::tag_name()) {
            tagged_slot = slot;
            tag         = handle->tag.name;
            break;
        }
    }
    if (tagged_slot == 2) {
        return std::nullopt;
    }

    TensorId const    tagged_id   = expr.at(product.operands[tagged_slot]).tensor;
    std::string const tagged_name = expr.at(product.operands[tagged_slot]).name;
    // An elementwise term carries no per-operand index lists, which are a contraction's
    // vocabulary; a leaf carries its own. Reading them off the term instead is the mistake that
    // made this shape invisible the first time it was looked for.
    auto const tagged_index = expr.at(product.operands[tagged_slot]).indices;

    if (std::ranges::find(_considered, tagged_id) == _considered.end()) {
        _considered.push_back(tagged_id);
    }
    if (written_anywhere(graph, tagged_id)) {
        note_skip("the tagged tensor is written by this graph, so its factors could go stale", fmt::format("tensor '{}'", tagged_name));
        return std::nullopt;
    }

    ComparisonContext ctx;
    ctx.registry = &graph.space_registry();

    std::vector<std::string> used = letters_of(tagged_index);
    merge_letters(used, letters_of(statement.target_indices));

    // The invented-id space a trial names its tensors in. Far above anything a graph assigns,
    // so a trial expression can be priced without a declaration being made for a rewrite that
    // is then declined.
    TensorId trial_next = TensorId{1} << 60U;

    struct Candidate {
        FactorizationPlan           plan;
        std::vector<search::Factor> pieces;
        search::LetterTable         table;
        search::TreePlan            tree;
        SymbolicCost                after;
        std::size_t                 quadrature_points{0};
        double                      quadrature_measured{0};
        double                      quadrature_tolerance{0};
        std::string                 quadrature_label;
    };
    std::optional<Candidate> best;
    SymbolicCost const       before = expression_cost(graph, expr, {}, {});

    for (auto const &provider : registry().for_tag(tag)) {
        auto offer = provider->propose(graph, tagged_id);
        if (!offer) {
            note_skip("a provider declined", fmt::format("'{}': {}", provider->name(), offer.error()));
            continue;
        }
        FactorizationPlan plan = std::move(*offer);
        if (plan.factors.size() < 2 || plan.factors.size() > kMaxPieces) {
            note_skip("a provider's chain is not one the bracketing search is allowed to take",
                      fmt::format("'{}': {} factor(s)", provider->name(), plan.factors.size()));
            continue;
        }
        if (plan.tagged_letters.size() != tagged_index.size()) {
            note_skip("a provider's letter list does not match the tagged operand's rank", fmt::format("'{}'", provider->name()));
            continue;
        }

        std::unordered_map<std::string, std::string> rename;
        std::unordered_map<std::string, SpaceId>     new_spaces;
        for (std::size_t axis = 0; axis < plan.tagged_letters.size(); ++axis) {
            rename.emplace(plan.tagged_letters[axis], tagged_index[axis].letter);
        }
        std::vector<std::string> taken = used;
        bool                     ok    = true;
        for (auto const &factor : plan.factors) {
            if (factor.letters.size() != factor.dims.size()) {
                note_skip("a provider's factor has a different number of letters and extents", fmt::format("'{}'", provider->name()));
                ok = false;
                break;
            }
            for (std::size_t axis = 0; axis < factor.letters.size(); ++axis) {
                if (rename.contains(factor.letters[axis])) {
                    continue;
                }
                std::string fresh = fresh_letter(taken, factor.letters[axis]);
                taken.push_back(fresh);
                rename.emplace(factor.letters[axis], fresh);
                if (axis < factor.spaces.size() && !factor.spaces[axis].empty()) {
                    if (auto const space = graph.space_registry().find(factor.spaces[axis]); space.has_value()) {
                        new_spaces.emplace(fresh, *space);
                    }
                }
            }
        }
        if (!ok) {
            continue;
        }

        std::vector<search::Factor> pieces;
        search::LetterTable         table;
        bool                        any_symbolic_extent = false;
        if (TensorHandle const *handle = graph.find_tensor(tagged_id); handle != nullptr) {
            for (std::size_t axis = 0; axis < tagged_index.size() && axis < handle->dims.size(); ++axis) {
                table.observe(tagged_index[axis], handle->dims[axis]);
                if (axis < handle->dim_symbols.size() && !handle->dim_symbols[axis].empty()) {
                    any_symbolic_extent = true;
                }
            }
        }
        for (auto const &factor : plan.factors) {
            search::Factor piece;
            for (std::size_t axis = 0; axis < factor.letters.size(); ++axis) {
                std::string const &mapped = rename.at(factor.letters[axis]);
                SpaceId            space;
                if (auto const found = new_spaces.find(mapped); found != new_spaces.end()) {
                    space = found->second;
                } else {
                    for (auto const &index : tagged_index) {
                        if (index.letter == mapped) {
                            space = index.space;
                            break;
                        }
                    }
                }
                ExprIndex const index{.letter = mapped, .space = space};
                table.observe(index, factor.dims[axis]);
                piece.indices.push_back(index);
            }
            pieces.push_back(std::move(piece));
        }

        ComparisonContext tree_ctx;
        tree_ctx.registry     = ctx.registry;
        tree_ctx.bound_extent = table.lookup();

        search::TreePlan const tree = search::solve_tree(pieces, tagged_index, table, tree_ctx);
        if (!tree.ok) {
            note_skip("no bracketing of the substituted product could be found",
                      fmt::format("'{}' on '{}'", provider->name(), tagged_name));
            continue;
        }

        // ── The trial ─────────────────────────────────────────────────────────────────
        //
        // The substitution alone is strictly more arithmetic here: it rebuilds the very tensor
        // the caller already holds. What is being asked is whether the substitution plus the
        // quadrature that becomes possible because of it is worth taking, so the trial carries
        // both and nothing is declared for a trial that loses.
        TensorExpr                                             trial = expr;
        std::unordered_map<TensorId, std::vector<std::size_t>> invented;
        TensorId                                               next = trial_next;
        auto const invent = [&](std::string const & /*name*/, packed_gemm::ScalarType, std::vector<std::size_t> const &dims) {
            TensorId const id = next++;
            invented.emplace(id, dims);
            return id;
        };
        std::vector<search::Factor> trial_pieces = pieces;
        for (std::size_t which = 0; which < trial_pieces.size(); ++which) {
            trial_pieces[which].tensor = invent(plan.factors[which].name, plan.factors[which].dtype, plan.factors[which].dims);
        }
        std::vector<std::size_t> numerator_dims;
        for (auto const &index : tagged_index) {
            auto const extent = table.extent.find(index.letter);
            if (extent == table.extent.end()) {
                break;
            }
            numerator_dims.push_back(extent->second);
        }
        if (numerator_dims.size() != tagged_index.size()) {
            continue;
        }
        TensorId const trial_numerator =
            invent(fmt::format("{}_{}_num", provider->name(), tagged_name), plan.factors[0].dtype, numerator_dims);

        if (!substitute_product_operand(graph, trial, position, tagged_slot, provider->name(), tagged_name, plan, trial_pieces, tree, table,
                                        tagged_index, trial_numerator, invent)) {
            note_skip("the chosen bracketing could not be emitted", fmt::format("'{}' on '{}'", provider->name(), tagged_name));
            continue;
        }

        std::vector<TensorId> internal = region.internal;
        internal.push_back(trial_numerator);
        quadrature::RewriteOptions options;
        options.epsilon    = _laplace->epsilon();
        options.points     = _laplace->points();
        options.energy     = [this](std::string const &wanted) { return _laplace->energy(wanted); };
        options.declare    = invent;
        options.want_setup = false;
        std::vector<TensorId> claimed;

        auto const outcomes = quadrature::rewrite_denominators(graph, internal, trial, options, claimed);
        auto const applied  = std::ranges::find_if(outcomes, [](auto const &entry) { return entry.applied; });
        if (applied == outcomes.end()) {
            for (auto const &entry : outcomes) {
                if (!entry.reason.empty()) {
                    note_skip(entry.reason, entry.detail);
                }
            }
            note_skip("the substitution only pays with the quadrature that follows it, and the quadrature declined",
                      fmt::format("'{}' on '{}'", provider->name(), tagged_name));
            continue;
        }
        for (auto const &[id, dims] : applied->shapes) {
            invented.emplace(id, dims);
        }

        SymbolicCost const after = expression_cost(graph, trial, invented, applied->constant_letters);

        // The joint verdict. Both halves are asked of the PAIR, because the substitution on its
        // own rebuilds a tensor the caller already has and the transform on its own has nothing
        // to ride on: a veto taken on either alone would refuse a rewrite the other makes pay.
        if (compare(after, before, ctx) >= 0) {
            note_skip("the fit and the quadrature together are not symbolically cheaper than the region they replace",
                      fmt::format("'{}' on '{}': {} vs {}", provider->name(), tagged_name, after.flops.to_string(ctx.registry),
                                  before.flops.to_string(ctx.registry)));
            continue;
        }
        ExtentLookup const extent_of    = table.lookup();
        auto const         before_flops = before.flops.evaluate(extent_of);
        auto const         after_flops  = after.flops.evaluate(extent_of);
        if (any_symbolic_extent) {
            report(2, fmt::format("the extent veto abstains on '{}' ({}): a symbolic axis makes the captured size a placeholder",
                                  tagged_name, provider->name()));
        } else if (before_flops.has_value() && after_flops.has_value() && *after_flops >= *before_flops) {
            note_skip("the fit and the quadrature together are not cheaper at the extents this graph holds",
                      fmt::format("'{}' on '{}': {:g} vs {:g} flops", provider->name(), tagged_name, *after_flops, *before_flops));
            continue;
        }
        if (best.has_value() && compare(after, best->after, ctx) >= 0) {
            continue;
        }

        best = Candidate{.plan                 = std::move(plan),
                         .pieces               = std::move(pieces),
                         .table                = std::move(table),
                         .tree                 = tree,
                         .after                = after,
                         .quadrature_points    = applied->points,
                         .quadrature_measured  = applied->measured,
                         .quadrature_tolerance = applied->tolerance,
                         .quadrature_label     = applied->setup_label};
    }

    if (!best.has_value()) {
        return std::nullopt;
    }

    // Both accuracy statements, before anything is rewritten, and both through the one place a
    // lossy pass refuses. A budget that will not pay for the pair leaves the region exactly as
    // it was, which is why neither record is written until both are accepted.
    ApproximationRecord fit = best->plan.accuracy;
    fit.pass_name           = best->plan.provider;
    if (fit.setup.empty()) {
        fit.setup = fmt::format("{}({})", best->plan.provider, tagged_name);
    }
    ApproximationRecord quad =
        make_approximation_record(_laplace->name(), ApproximationEffect::NormRelative, best->quadrature_tolerance,
                                  best->quadrature_measured, {}, {}, best->quadrature_label, ApproximationOrigin::Measured);
    if (!approximate(graph, fit) || !approximate(graph, quad)) {
        return std::nullopt;
    }

    // ── The real thing, with the tensors declared this time ───────────────────────────
    std::size_t const                             count = best->plan.factors.size();
    std::vector<TensorId>                         factor_ids(count);
    std::vector<std::pair<std::string, TensorId>> declared;
    for (std::size_t which = 0; which < count; ++which) {
        std::string const &factor_name = best->plan.factors[which].name;
        auto const         found = std::ranges::find_if(declared, [&factor_name](auto const &entry) { return entry.first == factor_name; });
        if (found != declared.end()) {
            factor_ids[which] = found->second;
            continue;
        }
        factor_ids[which] = declare_scratch(graph, fmt::format("{}_{}", best->plan.provider, factor_name), best->plan.factors[which].dtype,
                                            best->plan.factors[which].dims);
        declared.emplace_back(factor_name, factor_ids[which]);
    }
    for (std::size_t which = 0; which < count; ++which) {
        best->pieces[which].tensor = factor_ids[which];
    }

    auto const declare = [&graph](std::string const &name, packed_gemm::ScalarType dtype, std::vector<std::size_t> const &dims) {
        return declare_scratch(graph, name, dtype, dims);
    };
    std::vector<std::size_t> numerator_dims;
    for (auto const &index : tagged_index) {
        numerator_dims.push_back(best->table.extent.at(index.letter));
    }
    TensorId const numerator =
        declare(fmt::format("{}_{}_num", best->plan.provider, tagged_name), best->plan.factors[0].dtype, numerator_dims);

    std::size_t const inserted = expr.statements.size();
    if (!substitute_product_operand(graph, expr, position, tagged_slot, best->plan.provider, tagged_name, best->plan, best->pieces,
                                    best->tree, best->table, tagged_index, numerator, declare)) {
        return std::nullopt;
    }
    std::size_t const grew = expr.statements.size() - inserted;

    std::vector<TensorId> internal = region.internal;
    internal.push_back(numerator);
    quadrature::RewriteOptions options;
    options.epsilon = _laplace->epsilon();
    options.points  = _laplace->points();
    options.energy  = [this](std::string const &wanted) { return _laplace->energy(wanted); };
    options.declare = declare;
    std::vector<TensorId> claimed;

    auto const outcomes = quadrature::rewrite_denominators(graph, internal, expr, options, claimed);
    for (auto const &outcome : outcomes) {
        if (outcome.applied && outcome.emit_setup) {
            _pending_quadrature.push_back(PendingQuadrature{.label = quad.setup, .emit = outcome.emit_setup});
        }
    }

    _pending.push_back(PendingSetup{.label = fit.setup, .factors = factor_ids, .emit = best->plan.emit_setup});
    ++_num_factorized;
    ++_num_joint;
    report(1, fmt::format("factorized '{}' through {} and decoupled its denominator in one decision: neither pays alone", tagged_name,
                          best->plan.provider));
    return grew;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
