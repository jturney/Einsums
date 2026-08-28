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

} // namespace

void DeltaElimination::reset_stats() {
    _num_eliminated = 0;
    _num_dissolved  = 0;
}

std::vector<std::string> DeltaElimination::describe() const {
    if (_num_eliminated == 0) {
        return {};
    }
    return {fmt::format("DeltaElimination: removed {} contraction(s) against a Kronecker delta, dissolving {} intermediate(s)",
                        _num_eliminated, _num_dissolved)};
}

bool DeltaElimination::applicable(Graph const &graph) const {
    return std::ranges::any_of(graph.tensors_map(), [](auto const &entry) { return entry.second.tag.name == provenance_identity; });
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
            auto const  &source     = expr.at(other_leaf);

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
                        leaf.tensor = source.tensor;
                        leaf.name   = source.name;
                    }
                }
                statement.value = invalid_term; // erased below
                ++_num_dissolved;
                report(2, fmt::format("dissolved '{}': its readers now take '{}' directly", statement.target_name, source.name));
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
                report(2, fmt::format("'{}' keeps a permute of '{}'; its target is not dissolvable", statement.target_name, source.name));
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
