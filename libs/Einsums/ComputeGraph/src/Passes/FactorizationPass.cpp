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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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

/// The cost of one hypothetical contraction, by the same formula @ref symbolic_cost_for uses.
///
/// Written out rather than built through an @ref EinsumDescriptor because the contraction does
/// not exist yet and constructing a descriptor for a node nobody will emit would be a second
/// way of spelling the same three polynomials.
SymbolicCost contraction_cost(std::vector<std::string> const &a, std::vector<std::string> const &b, std::vector<std::string> const &c,
                              LetterVars const &vars) {
    std::vector<std::string> all = c;
    merge_letters(all, a);
    merge_letters(all, b);

    SymbolicCost cost;
    cost.flops    = symbolic_size_for(all, vars) * 2.0;
    cost.traffic  = symbolic_size_for(c, vars) + symbolic_size_for(a, vars) + symbolic_size_for(b, vars);
    cost.resident = cost.traffic;
    return cost;
}

SymbolicCost operator+(SymbolicCost const &lhs, SymbolicCost const &rhs) {
    SymbolicCost out;
    out.flops    = lhs.flops + rhs.flops;
    out.traffic  = lhs.traffic + rhs.traffic;
    out.resident = lhs.resident + rhs.resident;
    return out;
}

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

} // namespace

void FactorizationPass::reset_stats() {
    RegionRewrite::reset_stats();
    _num_factorized = 0;
    _pending.clear();
}

FactorizationRegistry &FactorizationPass::registry() const {
    return _registry != nullptr ? *_registry : global_factorization_registry();
}

std::vector<std::string> FactorizationPass::describe() const {
    if (_num_factorized == 0) {
        return {};
    }
    return {fmt::format("FactorizationPass: re-associated {} contraction(s) around a provider's factors", _num_factorized)};
}

bool FactorizationPass::applicable(Graph const &graph) const {
    auto &known = registry();
    return std::ranges::any_of(
        graph.tensors_map(), [&known](auto const &entry) { return !entry.second.tag.name.empty() && known.claims(entry.second.tag.name); });
}

bool FactorizationPass::run(Graph &graph) {
    _pending.clear();
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
    if (!_pending.empty()) {
        report(1, fmt::format("emitted {} setup bod(y/ies) holding the fittings", _pending.size()));
    }
    _pending.clear();
    return modified;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity): the decomposition is one argument
// and splitting it would put the halves out of sight of each other.
bool FactorizationPass::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    bool changed = false;

    ComparisonContext ctx;
    ctx.registry = &graph.space_registry();

    // Index-based, because an accepted rewrite INSERTS a statement and the loop has to skip
    // past what it just added rather than reconsider it.
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
        auto const        tagged_index = term.operand_indices[tagged_slot];
        auto const        other_index  = term.operand_indices[other_slot];

        if (written_anywhere(graph, tagged_id)) {
            note_skip("the tagged tensor is written by this graph, so its factors could go stale", fmt::format("tensor '{}'", tagged_name));
            continue;
        }
        if (!term.conjugate.empty() && term.conjugate[tagged_slot]) {
            note_skip("the tagged operand is conjugated, which a factor pair has nowhere to carry",
                      fmt::format("tensor '{}'", tagged_name));
            continue;
        }

        // Every letter this statement already uses, so a provider's new one cannot collide.
        std::vector<std::string> used = letters_of(tagged_index);
        merge_letters(used, letters_of(other_index));
        merge_letters(used, letters_of(statement.target_indices));

        // Letters to their spaces, for both the rename and the cost model.
        std::vector<std::pair<std::string, SpaceId>> letter_spaces;
        auto const                                   note_space = [&letter_spaces](std::vector<ExprIndex> const &indices) {
            for (auto const &index : indices) {
                if (index.space.valid()) {
                    letter_spaces.emplace_back(index.letter, index.space);
                }
            }
        };
        note_space(tagged_index);
        note_space(other_index);
        note_space(statement.target_indices);

        struct Candidate {
            FactorizationPlan          plan;
            std::vector<RenamedFactor> factors;
            std::vector<ExprIndex>     intermediate;
            std::size_t                pair_slot{0}; ///< Which factor pairs with the other operand.
            SymbolicCost               cost;
        };
        std::optional<Candidate> best;

        for (auto const &provider : registry().for_tag(tag)) {
            auto offer = provider->propose(graph, tagged_id);
            if (!offer) {
                note_skip("a provider declined", fmt::format("'{}': {}", provider->name(), offer.error()));
                continue;
            }
            FactorizationPlan plan = std::move(*offer);
            if (plan.factors.size() != 2) {
                note_skip("a provider offered other than two factors", fmt::format("'{}'", provider->name()));
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

            LetterVars const vars{[&]() {
                auto all = letter_spaces;
                for (auto const &[letter, space] : new_spaces) {
                    all.emplace_back(letter, space);
                }
                return all;
            }()};

            // Every letter's actual extent, for the numeric veto below. Read off the operands
            // this contraction already has, and off the plan for the letters it introduces;
            // between them every letter of both forms is covered or none of it is, and a
            // partial answer disables the veto rather than guessing at it.
            std::unordered_map<std::string, double> extents;
            auto const                              note_extent = [&](TensorId id, std::vector<ExprIndex> const &indices) {
                TensorHandle const *handle = graph.find_tensor(id);
                if (handle == nullptr) {
                    return;
                }
                for (std::size_t axis = 0; axis < indices.size() && axis < handle->dims.size(); ++axis) {
                    extents.emplace(indices[axis].letter, static_cast<double>(handle->dims[axis]));
                }
            };
            note_extent(tagged_id, tagged_index);
            note_extent(expr.at(other_leaf).tensor, other_index);
            note_extent(statement.target, statement.target_indices);
            for (std::size_t which = 0; which < 2; ++which) {
                for (std::size_t axis = 0; axis < plan.factors[which].letters.size(); ++axis) {
                    auto const mapped = rename.find(plan.factors[which].letters[axis]);
                    if (mapped != rename.end()) {
                        extents.emplace(mapped->second, static_cast<double>(plan.factors[which].dims[axis]));
                    }
                }
            }
            ExtentLookup const extent_of = [&extents, &vars](SymbolicVar const &variable) -> std::optional<double> {
                for (auto const &[letter, value] : extents) {
                    if (vars.var_for(letter) == variable) {
                        return value;
                    }
                }
                return std::nullopt;
            };

            std::vector<RenamedFactor> renamed(2);
            for (std::size_t which = 0; which < 2; ++which) {
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

            // Which factor pairs with the other operand. The decomposition needs exactly one
            // factor carrying the letters the tagged tensor shares with it: that is the pair
            // whose product is the cheap intermediate. Both or neither and there is no
            // regrouping to make, only a substitution, which is strictly more arithmetic.
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
            std::size_t pair_slot = 2;
            if (!shared.empty()) {
                if (carries_all(renamed[1]) && carries_none(renamed[0])) {
                    pair_slot = 1;
                } else if (carries_all(renamed[0]) && carries_none(renamed[1])) {
                    pair_slot = 0;
                }
            }
            if (pair_slot == 2) {
                note_skip("the split does not separate the letters shared with the other operand",
                          fmt::format("'{}' on '{}'", provider->name(), tagged_name));
                continue;
            }
            std::size_t const outer_slot = 1 - pair_slot;

            // The intermediate's free indices: everything the paired factor and the other
            // operand carry that the outer factor or the target still needs. Anything else is
            // summed away inside the first contraction, which is where the saving is.
            std::vector<ExprIndex> intermediate;
            auto const             needed = [&](std::string const &letter) {
                return contains(renamed[outer_slot].letters, letter) || contains(letters_of(statement.target_indices), letter);
            };
            auto const push_needed = [&](std::vector<ExprIndex> const &source) {
                for (auto const &index : source) {
                    if (needed(index.letter) && !contains(letters_of(intermediate), index.letter)) {
                        intermediate.push_back(index);
                    }
                }
            };
            push_needed(renamed[pair_slot].indices);
            push_needed(other_index);

            // The target must still be producible from the outer factor and the intermediate.
            // Checked rather than reasoned about: the shapes that reach here are wider than the
            // tidy ones, and a miss would emit a contraction whose operands cannot make its
            // output.
            std::vector<std::string> reachable = renamed[outer_slot].letters;
            merge_letters(reachable, letters_of(intermediate));
            if (!std::ranges::all_of(letters_of(statement.target_indices),
                                     [&reachable](std::string const &letter) { return contains(reachable, letter); })) {
                note_skip("the decomposed form cannot produce the target's indices",
                          fmt::format("'{}' on '{}'", provider->name(), tagged_name));
                continue;
            }

            SymbolicCost const before =
                contraction_cost(letters_of(tagged_index), letters_of(other_index), letters_of(statement.target_indices), vars);
            SymbolicCost const after =
                contraction_cost(renamed[pair_slot].letters, letters_of(other_index), letters_of(intermediate), vars) +
                contraction_cost(renamed[outer_slot].letters, letters_of(intermediate), letters_of(statement.target_indices), vars);

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
            auto const before_flops = before.flops.evaluate(extent_of);
            auto const after_flops  = after.flops.evaluate(extent_of);
            if (before_flops.has_value() && after_flops.has_value() && *after_flops >= *before_flops) {
                note_skip("the decomposed form is not cheaper at the extents this graph holds",
                          fmt::format("'{}' on '{}': {:g} vs {:g} flops", provider->name(), tagged_name, *after_flops, *before_flops));
                continue;
            }
            if (best.has_value() && compare(after, best->cost, ctx) >= 0) {
                continue;
            }

            best = Candidate{.plan         = std::move(plan),
                             .factors      = std::move(renamed),
                             .intermediate = std::move(intermediate),
                             .pair_slot    = pair_slot,
                             .cost         = after};
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

        std::size_t const outer_slot = 1 - best->pair_slot;

        // Create the factors and the intermediate. Tensors only: a node added here would move
        // the region out from under the splice that is about to replace it.
        std::vector<TensorId> factor_ids(2);
        for (std::size_t which = 0; which < 2; ++which) {
            factor_ids[which] = declare_scratch(graph, fmt::format("{}_{}", best->plan.provider, best->plan.factors[which].name),
                                                best->plan.factors[which].dtype, best->plan.factors[which].dims);
        }

        std::vector<std::size_t> intermediate_dims;
        for (auto const &index : best->intermediate) {
            std::size_t extent = 0;
            // The extent of a letter, from whichever operand spells that axis. Read off the
            // handles rather than from the plan, because only the graph knows what the OTHER
            // operand's axes are.
            auto const from = [&](TensorId id, std::vector<ExprIndex> const &indices) {
                TensorHandle const *handle = graph.find_tensor(id);
                if (handle == nullptr) {
                    return;
                }
                for (std::size_t axis = 0; axis < indices.size() && axis < handle->dims.size(); ++axis) {
                    if (indices[axis].letter == index.letter && extent == 0) {
                        extent = handle->dims[axis];
                    }
                }
            };
            from(expr.at(other_leaf).tensor, other_index);
            for (std::size_t which = 0; which < 2; ++which) {
                for (std::size_t axis = 0; axis < best->factors[which].letters.size(); ++axis) {
                    if (best->factors[which].letters[axis] == index.letter && extent == 0) {
                        extent = best->plan.factors[which].dims[axis];
                    }
                }
            }
            intermediate_dims.push_back(extent);
        }
        TensorId const intermediate_id = declare_scratch(graph, fmt::format("{}_{}_x", best->plan.provider, tagged_name),
                                                         best->plan.factors[0].dtype, intermediate_dims);

        // Leaves for the three new tensors.
        auto make_leaf = [&expr](TensorId id, std::string name, std::vector<ExprIndex> indices) {
            ExprTerm leaf;
            leaf.kind    = TermKind::Leaf;
            leaf.tensor  = id;
            leaf.name    = std::move(name);
            leaf.indices = std::move(indices);
            return expr.add(std::move(leaf));
        };
        TermId const pair_leaf =
            make_leaf(factor_ids[best->pair_slot], best->plan.factors[best->pair_slot].name, best->factors[best->pair_slot].indices);
        TermId const outer_leaf = make_leaf(factor_ids[outer_slot], best->plan.factors[outer_slot].name, best->factors[outer_slot].indices);
        TermId const intermediate_leaf = make_leaf(intermediate_id, fmt::format("{}_x", tagged_name), best->intermediate);

        // X = paired_factor * other
        ExprTerm inner;
        inner.kind    = TermKind::Contraction;
        inner.indices = best->intermediate;
        inner.operands.assign({pair_leaf, other_leaf});
        inner.operand_indices.assign({best->factors[best->pair_slot].indices, other_index});
        inner.conjugate.assign({false, !term.conjugate.empty() && term.conjugate[other_slot]});
        inner.factor = PrefactorScalar{double{1}};

        ExprStatement inner_statement;
        inner_statement.target           = intermediate_id;
        inner_statement.target_name      = fmt::format("{}_x", tagged_name);
        inner_statement.target_indices   = best->intermediate;
        inner_statement.target_prefactor = PrefactorScalar{double{0}};
        inner_statement.value            = expr.add(std::move(inner));
        inner_statement.origin           = statement.origin;
        inner_statement.origin_kind      = OpKind::Einsum;
        inner_statement.origin_label =
            fmt::format("{}: X[{}] = {}[{}] ; {}", best->plan.provider, fmt::join(letters_of(best->intermediate), ","),
                        best->plan.factors[best->pair_slot].name, fmt::join(best->factors[best->pair_slot].letters, ","), tagged_name);

        // target = f * outer_factor * X, keeping the original destination prefactor so an
        // accumulation stays an accumulation.
        ExprTerm outer;
        outer.kind    = TermKind::Contraction;
        outer.indices = statement.target_indices;
        outer.operands.assign({outer_leaf, intermediate_leaf});
        outer.operand_indices.assign({best->factors[outer_slot].indices, best->intermediate});
        outer.conjugate.assign({false, false});
        outer.factor = term.factor;

        expr.statements[position].value        = expr.add(std::move(outer));
        expr.statements[position].origin_kind  = OpKind::Einsum;
        expr.statements[position].origin_label = fmt::format(
            "{}: {}[{}] = {}[{}] ; X", best->plan.provider, statement.target_name, fmt::join(letters_of(statement.target_indices), ","),
            best->plan.factors[outer_slot].name, fmt::join(best->factors[outer_slot].letters, ","));

        expr.statements.insert(expr.statements.begin() + static_cast<std::ptrdiff_t>(position), std::move(inner_statement));
        ++position; // past the statement just inserted, onto the one it feeds

        _pending.push_back(PendingSetup{.label = record.setup, .factors = factor_ids, .emit = best->plan.emit_setup});
        ++_num_factorized;
        changed = true;
        report(2, fmt::format("factorized '{}' through {}: one contraction became two", tagged_name, best->plan.provider));
    }

    (void)region;
    if (changed) {
        EINSUMS_LOG_INFO("FactorizationPass: re-associated {} contraction(s)", _num_factorized);
    }
    return changed;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
