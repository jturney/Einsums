//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DeltaElimination.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// The letters of an index list, without their spaces.
std::vector<std::string> letters_of(std::vector<ExprIndex> const &indices) {
    std::vector<std::string> out;
    out.reserve(indices.size());
    for (auto const &index : indices) {
        out.push_back(index.letter);
    }
    return out;
}

std::size_t count_letter(std::vector<ExprIndex> const &indices, std::string const &letter) {
    return static_cast<std::size_t>(std::ranges::count_if(indices, [&letter](ExprIndex const &i) { return i.letter == letter; }));
}

/// Whether two index lists name the same letters, in any order.
bool same_letter_set(std::vector<ExprIndex> const &lhs, std::vector<ExprIndex> const &rhs) {
    auto a = letters_of(lhs);
    auto b = letters_of(rhs);
    std::ranges::sort(a);
    std::ranges::sort(b);
    return a == b;
}

/// Whether two index lists are equal letter for letter, in order.
bool same_letters_in_order(std::vector<ExprIndex> const &lhs, std::vector<ExprIndex> const &rhs) {
    return letters_of(lhs) == letters_of(rhs);
}

/// A prefactor as the ``std::complex<double>`` a @ref PermuteDescriptor snapshot holds.
///
/// The descriptor keeps its at-capture scalars in that one type while a @ref PrefactorScalar is
/// a variant over the four, so the widening happens here rather than at each assignment. Exact
/// for every arm: a float and a double both land in a double, and the complex arms are already
/// this shape.
std::complex<double> as_complex(PrefactorScalar const &value) {
    return std::visit(
        []<typename T>(T const &scalar) -> std::complex<double> {
            if constexpr (IsComplexV<T>) {
                return {static_cast<double>(scalar.real()), static_cast<double>(scalar.imag())};
            } else {
                return {static_cast<double>(scalar), 0.0};
            }
        },
        value);
}

/// The rename a delta operand encodes: which of its two letters is contracted away and which
/// survives into the output.
struct Substitution {
    std::string link;     ///< The letter contracted away, which the other operand carries.
    std::string survivor; ///< The letter it becomes, which the output carries.
};

/// Work out the substitution, or say why there is not one.
///
/// A delta is a rename only when one of its letters is contracted against the other operand and
/// the other is free in the output. Both free is a diagonal extraction, both contracted is a
/// trace, and neither is a rename; each is real arithmetic this pass does not do.
std::optional<Substitution> substitution_for(std::vector<ExprIndex> const &delta, std::vector<ExprIndex> const &other,
                                             std::vector<ExprIndex> const &target, std::string &why) {
    if (delta.size() != 2) {
        why = "the delta operand is not rank 2";
        return std::nullopt;
    }
    std::string const &first  = delta[0].letter;
    std::string const &second = delta[1].letter;
    if (first == second) {
        why = "the delta's two letters are the same, which is a trace rather than a rename";
        return std::nullopt;
    }

    auto const classify = [&](std::string const &letter, std::string const &partner) -> std::optional<Substitution> {
        // `letter` contracted against the other operand, `partner` surviving into the output.
        if (count_letter(other, letter) != 1 || count_letter(target, letter) != 0) {
            return std::nullopt;
        }
        if (count_letter(target, partner) != 1 || count_letter(other, partner) != 0) {
            return std::nullopt;
        }
        return Substitution{.link = letter, .survivor = partner};
    };

    if (auto found = classify(first, second); found.has_value()) {
        return found;
    }
    if (auto found = classify(second, first); found.has_value()) {
        return found;
    }
    why = "the delta's letters are not one contracted and one free, so it is a diagonal or a trace";
    return std::nullopt;
}

/// One operand slot's index-space annotation, as the graph carries it NOW.
struct SlotSpace {
    SpaceId space;
    bool    inferred{false};
};

/// The space annotated on @p tensor's axis @p axis, or nothing when that axis says nothing.
///
/// Read from the HANDLE rather than from the raised @ref ExprIndex, because the raised space
/// comes from the descriptor's capture-time map, and that map holds nothing for a program
/// annotated after capture, which is every program annotated from Python. The two diagnostic
/// passes re-derive from the handles for the same reason.
std::optional<SlotSpace> annotated_space(Graph const &graph, TensorId tensor, std::size_t axis) {
    TensorHandle const *handle = graph.find_tensor(tensor);
    if (handle == nullptr || axis >= handle->spaces.size() || !handle->spaces[axis].valid()) {
        return std::nullopt;
    }
    return SlotSpace{.space = handle->spaces[axis], .inferred = handle->spaces_inferred};
}

/// A relation query that answers Unknown rather than throwing on an id this registry cannot
/// resolve, which is what a handle annotated against a different registry produces.
Tristate safe_disjoint(SpaceRegistry const &registry, SpaceId first, SpaceId second) {
    if (!first.valid() || !second.valid() || first.value() >= registry.size() || second.value() >= registry.size()) {
        return Tristate::Unknown;
    }
    return registry.is_disjoint(first, second);
}

/// The position of @p letter in @p indices, which the caller has already checked appears once.
std::size_t axis_of(std::vector<ExprIndex> const &indices, std::string const &letter) {
    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        if (indices[axis].letter == letter) {
            return axis;
        }
    }
    return indices.size();
}

/// The arena's single leaf for @p tensor, created only if this expression has none.
///
/// @ref raise_region interns one leaf per tensor, and finding that one rather than adding a second
/// is what keeps a rewrite of the leaf reaching every use of the tensor.
TermId leaf_for(TensorExpr &expr, TensorId tensor, std::string const &name) {
    for (std::size_t id = 0; id < expr.terms.size(); ++id) {
        if (expr.terms[id].kind == TermKind::Leaf && expr.terms[id].tensor == tensor) {
            return static_cast<TermId>(id);
        }
    }
    ExprTerm leaf;
    leaf.kind   = TermKind::Leaf;
    leaf.tensor = tensor;
    leaf.name   = name;
    return expr.add(std::move(leaf));
}

/// Whether any statement other than @p self reads @p tensor.
bool read_elsewhere(TensorExpr const &expr, TensorId tensor, ExprStatement const &self) {
    for (auto const &statement : expr.statements) {
        if (&statement == &self || statement.value == invalid_term) {
            continue;
        }
        for (auto const operand : expr.at(statement.value).operands) {
            auto const &leaf = expr.at(operand);
            if (leaf.kind == TermKind::Leaf && leaf.tensor == tensor) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

void DeltaElimination::reset_stats() {
    _num_eliminated  = 0;
    _num_dissolved   = 0;
    _num_zero_blocks = 0;
}

std::vector<std::string> DeltaElimination::describe() const {
    std::vector<std::string> lines;
    if (_num_eliminated != 0) {
        lines.push_back(fmt::format("DeltaElimination: removed {} contraction(s) against a Kronecker delta, dissolving {} intermediate(s)",
                                    _num_eliminated, _num_dissolved));
    }
    if (_num_zero_blocks != 0) {
        lines.push_back(
            fmt::format("DeltaElimination: reduced {} contraction(s) over disjoint spaces to their prefactor", _num_zero_blocks));
    }
    return lines;
}

bool DeltaElimination::applicable(Graph const &graph) const {
    if (std::ranges::any_of(graph.tensors_map(), [](auto const &entry) { return entry.second.tag.name == provenance_identity; })) {
        return true;
    }

    // The zero-block half needs three things at once, and a graph missing any of them cannot
    // produce a candidate. Asked here because this pass is in the default pipeline: it runs on
    // every graph anyone optimizes, and almost none of them declare a space at all.
    SpaceRegistry const &registry = graph.space_registry();
    auto const           ids      = registry.ids();
    bool                 disjoint = false;
    for (std::size_t first = 0; first < ids.size() && !disjoint; ++first) {
        for (std::size_t second = first + 1; second < ids.size(); ++second) {
            if (registry.is_disjoint(ids[first], ids[second]) == Tristate::Yes) {
                disjoint = true;
                break;
            }
        }
    }
    return disjoint && std::ranges::any_of(graph.tensors_map(), [](auto const &entry) {
               return std::ranges::any_of(entry.second.spaces, [](SpaceId id) { return id.valid(); });
           });
}

std::optional<std::string> DeltaElimination::zero_block_letter(Graph const &graph, TensorExpr const &expr, ExprTerm const &term,
                                                               ExprStatement const &statement) const {
    SpaceRegistry const &registry = graph.space_registry();

    TensorId const left  = expr.at(term.operands[0]).tensor;
    TensorId const right = expr.at(term.operands[1]).tensor;

    for (auto const &index : term.operand_indices[0]) {
        std::string const &letter = index.letter;
        if (count_letter(term.operand_indices[1], letter) == 0) {
            continue; // not shared, so nothing is being summed over two spaces
        }

        auto const on_left  = count_letter(term.operand_indices[0], letter);
        auto const on_right = count_letter(term.operand_indices[1], letter);
        auto const on_out   = count_letter(statement.target_indices, letter);

        // Both slots must be findable before anything can be said about them, and a letter that
        // survives into the output is not summed at all.
        if (on_left != 1 || on_right != 1) {
            note_skip("a shared letter repeats within an operand, so it has no single annotated slot",
                      fmt::format("target '{}'", statement.target_name));
            continue;
        }

        auto const first  = annotated_space(graph, left, axis_of(term.operand_indices[0], letter));
        auto const second = annotated_space(graph, right, axis_of(term.operand_indices[1], letter));
        if (!first.has_value() && !second.has_value()) {
            continue; // an ordinary unannotated contraction, and not a decline worth counting
        }
        if (!first.has_value() || !second.has_value()) {
            note_skip("a shared letter is annotated on one operand and not the other",
                      fmt::format("target '{}' letter '{}'", statement.target_name, letter));
            continue;
        }
        if (safe_disjoint(registry, first->space, second->space) != Tristate::Yes) {
            note_skip("nothing declared makes a shared letter's two spaces disjoint",
                      fmt::format("target '{}' letter '{}'", statement.target_name, letter));
            continue;
        }
        if (on_out != 0) {
            // A batched letter walks matrices side by side rather than summing over them, so
            // disjointness says the pairing is meaningless and NOT that the answer is zero.
            // Diagnosing that is CrossSpaceValidation's business; declining it is this pass's.
            note_skip("a letter over disjoint spaces is batched rather than summed, so the result is not zero",
                      fmt::format("target '{}' letter '{}'", statement.target_name, letter));
            continue;
        }
        if (first->inferred || second->inferred) {
            // An inference is a derivation from someone else's declaration, and this pass's
            // output is what a saved graph keeps. A rewrite is a stronger response than a
            // diagnostic, so it waits for two declarations.
            note_skip("a summed letter's disjointness rests on an inferred annotation",
                      fmt::format("target '{}' letter '{}'", statement.target_name, letter));
            continue;
        }
        return letter;
    }
    return std::nullopt;
}

bool DeltaElimination::rewrite(Graph &graph, Region const &region, TensorExpr &expr) {
    bool changed = false;

    // To a fixpoint, so a chain of deltas collapses in one visit. Each pass over the statements
    // can dissolve an intermediate that makes the NEXT one eliminable, and re-running the whole
    // pass to discover that would cost a re-raise of every region for nothing.
    bool progress = true;
    while (progress) {
        progress = false;

        for (auto &statement : expr.statements) {
            if (statement.value == invalid_term) {
                continue;
            }
            auto &term = expr.at(statement.value);
            if (term.kind != TermKind::Contraction || term.operands.size() != 2 || term.operand_indices.size() != 2) {
                continue;
            }

            // ── Provably zero ──────────────────────────────────────────────────────────
            //
            // A letter summed over two spaces that share no element has no term to sum, so the
            // contraction contributes exactly nothing and what is left is the destination's own
            // prefactor. Asked before the delta search, because a zero block subsumes whatever
            // else the statement was doing.
            if (zero_block_letter(graph, expr, term, statement).has_value()) {
                TensorId const target = statement.target;

                bool const  internal    = std::ranges::find(region.internal, target) != region.internal.end();
                bool const  overwritten = is_zero(statement.target_prefactor);
                std::size_t writers     = 0;
                for (auto const &other_statement : expr.statements) {
                    if (other_statement.target == target && other_statement.value != invalid_term) {
                        ++writers;
                    }
                }

                if (overwritten && internal && writers == 1 && !read_elsewhere(expr, target, statement)) {
                    // Zero into a buffer nothing reads and nothing outside can see. There is no
                    // value left to produce, so there is no node left to emit.
                    statement.value = invalid_term; // erased below
                    report(2, fmt::format("'{}' is a contraction over disjoint spaces into a buffer nothing reads; it is gone",
                                          statement.target_name));
                } else {
                    // Still has to be produced. `scale` by exactly zero ASSIGNS zero rather than
                    // multiplying, which is what makes this exact on a destination the program
                    // never wrote: a multiply would let an Inf already in the buffer survive.
                    ScaleDescriptor scale;
                    scale.factor = statement.target_prefactor;

                    ExprTerm replacement;
                    replacement.kind         = TermKind::Elementwise;
                    replacement.element_kind = OpKind::Scale;
                    replacement.descriptor   = OpData(std::move(scale));
                    replacement.indices      = statement.target_indices;
                    if (!overwritten) {
                        // The RMW convention: a node that READS its destination lists it among
                        // its inputs, and the schedulers rely on that to order it.
                        replacement.operands.push_back(leaf_for(expr, target, statement.target_name));
                        replacement.operand_indices.push_back(statement.target_indices);
                    }

                    statement.value            = expr.add(std::move(replacement));
                    statement.origin_kind      = OpKind::Scale;
                    statement.target_prefactor = overwritten ? PrefactorScalar{double{0}} : PrefactorScalar{double{1}};
                    statement.origin_label     = fmt::format("scale: '{}' keeps only its own prefactor", statement.target_name);
                    report(2, fmt::format("'{}' contracts over disjoint spaces; it keeps only its own prefactor", statement.target_name));
                }

                ++_num_zero_blocks;
                changed  = true;
                progress = true;
                continue;
            }

            // Which operand, if either, is a declared delta. Declared, never read from the data:
            // this pass's output is saved, and a later bind may put a different tensor behind the
            // same name.
            std::size_t delta_slot = 2;
            for (std::size_t slot = 0; slot < 2; ++slot) {
                TensorHandle const *handle = graph.find_tensor(expr.at(term.operands[slot]).tensor);
                if (handle != nullptr && handle->tag.name == provenance_identity) {
                    delta_slot = slot;
                    break;
                }
            }
            if (delta_slot == 2) {
                continue;
            }
            std::size_t const other_slot = 1 - delta_slot;

            // A conjugated delta is still a delta (its entries are real), but a conjugated OTHER
            // operand must keep its flag, and the node form this lowers to has nowhere to put
            // one. Declining is the honest answer rather than dropping a conjugation.
            if (term.conjugate.size() > other_slot && term.conjugate[other_slot]) {
                note_skip("the surviving operand is conjugated, which a permute cannot carry");
                continue;
            }

            std::string why;
            auto const  substitution =
                substitution_for(term.operand_indices[delta_slot], term.operand_indices[other_slot], statement.target_indices, why);
            if (!substitution.has_value()) {
                note_skip(why, fmt::format("target '{}'", statement.target_name));
                continue;
            }

            // The rename itself, on a copy, so a rejection below leaves the term untouched.
            std::vector<ExprIndex> renamed = term.operand_indices[other_slot];
            for (auto &index : renamed) {
                if (index.letter == substitution->link) {
                    index.letter = substitution->survivor;
                    // The surviving letter's SPACE comes from the output, not from the operand
                    // it replaced: the delta identified the two, and the output's annotation is
                    // the one a reader of the rewritten form will check against.
                    for (auto const &target_index : statement.target_indices) {
                        if (target_index.letter == substitution->survivor) {
                            index.space = target_index.space;
                            break;
                        }
                    }
                }
            }

            // After the rename the value must have exactly the output's letters, or this was not
            // a rename at all. Checked rather than reasoned about: the einsum forms that reach
            // here are wider than the tidy ones, and a mismatch here would silently produce a
            // node whose operands disagree about their extents.
            if (!same_letter_set(renamed, statement.target_indices)) {
                note_skip("the renamed operand does not carry exactly the output's letters",
                          fmt::format("target '{}'", statement.target_name));
                continue;
            }

            TermId const other_leaf = term.operands[other_slot];
            // COPIED, not referenced. `expr.add` below appends to the term arena, and a
            // reference into a vector does not survive its reallocation. The dissolve branch
            // reads this before any add and was always safe, which is how one reference came to
            // be correct on one path and dangling on the other.
            TensorId const    source_tensor = expr.at(other_leaf).tensor;
            std::string const source_name   = expr.at(other_leaf).name;

            // Can the statement disappear entirely, or must it still write its target?
            //
            // Dissolving needs three things at once: nothing outside the region can observe the
            // target, the value is a plain copy rather than a permutation or a scaling, and this
            // statement is the target's only writer here. Anything less and the target still has
            // to be produced, so a permute is emitted instead.
            bool const internal = std::ranges::find(region.internal, statement.target) != region.internal.end();
            bool const plain_copy =
                same_letters_in_order(renamed, statement.target_indices) && is_one(term.factor) && is_zero(statement.target_prefactor);
            std::size_t writers = 0;
            for (auto const &other_statement : expr.statements) {
                if (other_statement.target == statement.target && other_statement.value != invalid_term) {
                    ++writers;
                }
            }

            if (internal && plain_copy && writers == 1) {
                // Repoint every reader. `raise_region` interns ONE leaf per tensor, so the target
                // has exactly one leaf term in this arena and rewriting it reaches every use.
                for (auto &leaf : expr.terms) {
                    if (leaf.kind == TermKind::Leaf && leaf.tensor == statement.target) {
                        leaf.tensor = source_tensor;
                        leaf.name   = source_name;
                    }
                }
                statement.value = invalid_term; // erased below
                ++_num_dissolved;
                report(2, fmt::format("dissolved '{}': its readers now take '{}' directly", statement.target_name, source_name));
            } else {
                // Still has to be produced. A permute carries the reordering and both prefactors,
                // which is exactly what is left once the delta is gone.
                PermuteDescriptor permute;
                permute.a_indices = letters_of(renamed);
                permute.c_indices = letters_of(statement.target_indices);
                permute.alpha     = as_complex(term.factor);
                permute.beta      = as_complex(statement.target_prefactor);

                ExprTerm replacement;
                replacement.kind         = TermKind::Elementwise;
                replacement.element_kind = OpKind::Permute;
                replacement.descriptor   = OpData(std::move(permute));
                replacement.indices      = statement.target_indices;
                replacement.operands.push_back(other_leaf);
                // The renamed letters, so the dump shows the substitution rather than the leaf's
                // positional axis names. A before/after that rendered identically would make the
                // one diagnostic this framework offers useless for the rewrite it is diagnosing.
                replacement.operand_indices.push_back(renamed);

                statement.value        = expr.add(std::move(replacement));
                statement.origin_label = fmt::format("permute: C[{}] = A[{}]", fmt::join(letters_of(statement.target_indices), ","),
                                                     fmt::join(letters_of(renamed), ","));
                report(2, fmt::format("'{}' keeps a permute of '{}'; its target is not dissolvable", statement.target_name, source_name));
            }

            ++_num_eliminated;
            changed  = true;
            progress = true;
        }

        // Drop the statements that dissolved. Done after the sweep rather than during it, so the
        // loop above is never iterating a container it is erasing from.
        std::erase_if(expr.statements, [](ExprStatement const &statement) { return statement.value == invalid_term; });
    }

    if (changed) {
        report(1, fmt::format("removed {} delta contraction(s), dissolving {}", _num_eliminated, _num_dissolved));
        EINSUMS_LOG_INFO("DeltaElimination: removed {} delta contraction(s), dissolving {}", _num_eliminated, _num_dissolved);
    }
    return changed;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
