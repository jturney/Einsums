//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LetterBindings.hpp
 * @brief Re-derive a contraction's letter-to-space bindings from its operands' CURRENT annotations.
 *
 * Private to the pass sources. @ref einsums::compute_graph::EinsumDescriptor::letter_spaces is a
 * SNAPSHOT of what capture could see: it holds nothing for a program annotated after capture, and
 * nothing that @ref einsums::compute_graph::passes::SpacePropagation inferred afterwards. Both
 * diagnostic passes of this milestone have to see the annotations a graph carries NOW, so both
 * re-derive the bindings the same way, from one implementation.
 *
 * Unlike @ref einsums::compute_graph::detail::build_letter_spaces, which raises on a letter bound
 * to two spaces because that is a capture-time programming error, this walk RECORDS every slot
 * that binds a letter and raises nothing. Reporting the conflict is exactly what
 * @ref einsums::compute_graph::passes::CrossSpaceValidation exists to do, and a diagnostic pass
 * that threw would take the pipeline down with it.
 */

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::detail)

/// @brief One annotated index slot that binds a letter of one contraction.
struct SlotBinding {
    char const *operand{"A"};    ///< Operand the slot sits on: "A", "B" or "C".
    TensorId    tensor{0};       ///< Handle the slot sits on.
    std::string tensor_name;     ///< That handle's name, for a diagnostic.
    std::size_t slot{0};         ///< Position of the slot within the operand's index list.
    SpaceId     space;           ///< Space annotated on the slot. Always valid.
    bool        inferred{false}; ///< Whether the handle's annotation was inferred rather than declared.
};

/// @brief Every annotated slot that binds one index letter, in operand order.
struct LetterBinding {
    std::string              letter; ///< The index letter.
    std::vector<SlotBinding> slots;  ///< The slots binding it, in A, B, C order.
};

/**
 * @brief Collect the annotated slots binding each letter of one contraction.
 * @param[in] graph The graph the node belongs to, read for its tensor handles.
 * @param[in] node The contraction node. Its inputs are A and B, its single output is C.
 * @param[in] desc The node's descriptor, read for its index lists.
 * @return One entry per letter that met at least one annotated slot, sorted ascending by letter.
 *         Empty when no operand of the node carries an annotation.
 *
 * Unannotated slots and slots carrying an invalid id contribute nothing, so a partially annotated
 * operand yields a partial result rather than a wrong one.
 */
[[nodiscard]] inline std::vector<LetterBinding> letter_bindings(Graph const &graph, Node const &node, EinsumDescriptor const &desc) {
    struct Operand {
        char const                     *label;
        TensorId                        tensor;
        std::vector<std::string> const *indices;
    };

    std::array<Operand, 3> const operands{
        Operand{.label = "A", .tensor = !node.inputs.empty() ? node.inputs[0] : TensorId{0}, .indices = &desc.spec.a_indices},
        Operand{.label = "B", .tensor = node.inputs.size() > 1 ? node.inputs[1] : TensorId{0}, .indices = &desc.spec.b_indices},
        Operand{.label = "C", .tensor = !node.outputs.empty() ? node.outputs[0] : TensorId{0}, .indices = &desc.spec.c_indices},
    };

    std::vector<LetterBinding> bound;
    for (auto const &operand : operands) {
        if (operand.tensor == 0) {
            continue;
        }
        auto const *handle = graph.find_tensor(operand.tensor);
        if (handle == nullptr || handle->spaces.empty()) {
            continue;
        }
        std::size_t const slots = std::min(operand.indices->size(), handle->spaces.size());
        for (std::size_t slot = 0; slot < slots; ++slot) {
            SpaceId const id = handle->spaces[slot];
            if (!id.valid()) {
                continue; // A partially annotated tensor: this axis simply says nothing.
            }
            std::string const &letter = (*operand.indices)[slot];

            auto entry = std::ranges::find_if(bound, [&letter](LetterBinding const &candidate) { return candidate.letter == letter; });
            if (entry == bound.end()) {
                bound.push_back(LetterBinding{.letter = letter, .slots = {}});
                entry = std::prev(bound.end());
            }
            entry->slots.push_back(SlotBinding{.operand     = operand.label,
                                               .tensor      = operand.tensor,
                                               .tensor_name = handle->name,
                                               .slot        = slot,
                                               .space       = id,
                                               .inferred    = handle->spaces_inferred});
        }
    }

    std::ranges::sort(bound, [](LetterBinding const &lhs, LetterBinding const &rhs) { return lhs.letter < rhs.letter; });
    return bound;
}

/**
 * @brief Reduce collected bindings to the letter-to-space map a cost model wants.
 * @param[in] bindings The bindings, as returned by @ref letter_bindings.
 * @return One (letter, space) pair per letter whose slots AGREE, sorted ascending by letter.
 *
 * A letter whose slots disagree is dropped rather than resolved arbitrarily: the cost model then
 * treats it as an unannotated letter, which is the weaker but honest reading of a graph whose
 * annotations contradict each other. Naming the contradiction is
 * @ref einsums::compute_graph::passes::CrossSpaceValidation's job.
 */
[[nodiscard]] inline std::vector<std::pair<std::string, SpaceId>> agreed_letter_spaces(std::vector<LetterBinding> const &bindings) {
    std::vector<std::pair<std::string, SpaceId>> agreed;
    agreed.reserve(bindings.size());
    for (auto const &binding : bindings) {
        if (binding.slots.empty()) {
            continue;
        }
        SpaceId const first    = binding.slots.front().space;
        bool const    conflict = std::ranges::any_of(binding.slots, [first](SlotBinding const &slot) { return slot.space != first; });
        if (!conflict) {
            agreed.emplace_back(binding.letter, first);
        }
    }
    return agreed;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::detail)
