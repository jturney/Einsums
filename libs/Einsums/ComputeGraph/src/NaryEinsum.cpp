//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph/NaryEinsum.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <bit>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Passes/ContractionTreeSearch.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace search = passes::search;

namespace {

/// The most operands a plan searches over: the subset program is 3^N in their number.
constexpr std::size_t kMaxOperands = 16;

/// One step's operand as a two-operand spec spells it.
std::string spell_operand(std::vector<std::string> const &letters, bool conj) {
    std::string const joined = fmt::format("{}", fmt::join(letters, ","));
    return conj ? fmt::format("conj({})", joined) : joined;
}

} // namespace

std::vector<NaryEinsumStep> plan_nary_einsum(std::string spec, std::vector<std::vector<std::size_t>> dims,
                                             std::size_t max_intermediate_elements) {
    auto parsed = parse_nary_einsum_spec(spec);
    if (!parsed) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parsed.error().message);
    }
    std::size_t const count = parsed->operands.size();
    if (dims.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "einsum spec '{}' names {} operand(s), but {} were given", spec, count, dims.size());
    }
    if (count > kMaxOperands) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "einsum spec '{}' names {} operands; at most {} are planned", spec, count,
                                kMaxOperands);
    }

    // Every letter's extent, checked across the operands that carry it.
    search::LetterTable                          table;
    std::unordered_map<std::string, std::size_t> extent;
    std::vector<search::Factor>                  factors(count);
    for (std::size_t op = 0; op < count; ++op) {
        auto const &letters = parsed->operands[op];
        if (letters.size() != dims[op].size()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "einsum spec '{}': operand {} has {} index(es) but rank {}", spec, op + 1,
                                    letters.size(), dims[op].size());
        }
        for (std::size_t axis = 0; axis < letters.size(); ++axis) {
            auto const [it, fresh] = extent.try_emplace(letters[axis], dims[op][axis]);
            if (!fresh && it->second != dims[op][axis]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "einsum spec '{}': index '{}' has extent {} in one operand and {} in another", spec, letters[axis],
                                        it->second, dims[op][axis]);
            }
            ExprIndex const index{.letter = letters[axis], .space = {}};
            table.observe(index, dims[op][axis]);
            factors[op].indices.push_back(index);
        }
        factors[op].conjugate = parsed->conj[op];
    }
    std::vector<ExprIndex> output;
    for (auto const &letter : parsed->c_indices) {
        output.push_back(ExprIndex{.letter = letter, .space = {}});
    }

    ComparisonContext ctx;
    ctx.bound_extent      = table.lookup();
    search::TreePlan plan = search::solve_tree(factors, output, table, ctx, static_cast<double>(max_intermediate_elements));
    if (!plan.ok && max_intermediate_elements > 0) {
        plan = search::solve_tree(factors, output, table, ctx);
    }
    if (!plan.ok) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "einsum spec '{}': no contraction order was found", spec);
    }

    // The letters a set of operands keeps: those it shares with an operand outside it or the output.
    std::set<std::string> const output_letters(parsed->c_indices.begin(), parsed->c_indices.end());
    auto const                  kept_of = [&](search::Mask mask) {
        std::set<std::string> outside = output_letters;
        for (std::size_t op = 0; op < count; ++op) {
            if ((mask & (search::Mask{1} << op)) == 0) {
                outside.insert(parsed->operands[op].begin(), parsed->operands[op].end());
            }
        }
        return outside;
    };

    // Emit the tree depth first, children before parents, so every step reads results already formed.
    search::Mask const full = static_cast<search::Mask>((search::Mask{1} << count) - 1);
    struct Formed {
        std::size_t              number;
        std::vector<std::string> letters;
        bool                     conj;
    };
    std::vector<NaryEinsumStep> steps;
    auto const                  emit = [&](auto const &self, search::Mask mask) -> Formed {
        if (std::popcount(mask) == 1) {
            auto const op = static_cast<std::size_t>(std::countr_zero(mask));
            return {.number = op, .letters = parsed->operands[op], .conj = parsed->conj[op]};
        }
        search::Mask const left_mask = plan.split[mask];
        Formed const       left      = self(self, left_mask);
        Formed const       right     = self(self, mask ^ left_mask);

        // A partial product keeps its letters in the order its operands first name them; the last
        // step writes the output as the spec spells it.
        std::vector<std::string> letters;
        if (mask == full) {
            letters = parsed->c_indices;
        } else {
            std::set<std::string> const outside = kept_of(mask);
            for (auto const *list : {&left.letters, &right.letters}) {
                for (auto const &letter : *list) {
                    if (outside.contains(letter) && std::ranges::find(letters, letter) == letters.end()) {
                        letters.push_back(letter);
                    }
                }
            }
            // A partial product that sums every letter it has would be a scalar, and a two-operand
            // spec has no way to read an operand without indices. It keeps its shortest letter
            // instead, which the next contraction carries in that operand alone and so sums.
            if (letters.empty()) {
                std::string shortest;
                for (auto const *list : {&left.letters, &right.letters}) {
                    for (auto const &letter : *list) {
                        if (shortest.empty() || extent.at(letter) < extent.at(shortest)) {
                            shortest = letter;
                        }
                    }
                }
                letters.push_back(shortest);
            }
        }

        NaryEinsumStep step;
        step.left  = left.number;
        step.right = right.number;
        std::string operators;
        if (mask == full) {
            for (auto const &op : parsed->operators) {
                operators += op.render() + ' ';
            }
        }
        step.spec = fmt::format("{} <- {}{} ; {}", fmt::join(letters, ","), operators, spell_operand(left.letters, left.conj),
                                spell_operand(right.letters, right.conj));
        for (auto const &letter : letters) {
            step.dims.push_back(extent.at(letter));
        }
        steps.push_back(std::move(step));
        return {.number = count + steps.size() - 1, .letters = std::move(letters), .conj = false};
    };
    emit(emit, full);
    return steps;
}

std::size_t eager_einsum_intermediate_limit(std::size_t element_size) {
    std::size_t const budget = string_util::memory_string(config::get(option::MaxMemory));
    return budget == 0 || element_size == 0 ? 0 : budget / element_size;
}

EINSUMS_NAMESPACE_END(compute_graph)
