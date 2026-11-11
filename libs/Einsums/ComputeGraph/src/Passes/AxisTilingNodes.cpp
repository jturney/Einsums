//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// AxisTiling: which nodes the pass can slice, and the index letters it reads off them.

#include <Einsums/Config/Namespace.hpp>

#include "AxisTilingDetail.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::axis_tiling)

bool tileable_kind(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
    case OpKind::Permute:
    case OpKind::Axpby:
    case OpKind::Scale:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Dot:
        return true;
    default:
        return false;
    }
}

bool registered_sliceable(Node const &node) {
    DescriptorHooks const *hooks = descriptor_hooks(node);
    if (hooks == nullptr || !hooks->axes || !hooks->slice) {
        return false;
    }
    NodeEffects const effects = effects_of(node);
    return effects.deterministic && !effects.external_effects;
}

namespace {

/// The index letters a registered node names its operands with, the output first.
std::optional<std::vector<std::vector<std::string>>> registered_letters(Node const &node) {
    return descriptor_hooks(node)->axes(node.op_data, node);
}

} // namespace

std::optional<std::vector<SlotRef>> slots_of(Graph const &graph, Node const &node) {
    auto handle_rank = [&graph](TensorId tid) -> std::optional<std::size_t> {
        auto const *handle = graph.find_tensor(tid);
        if (handle == nullptr) {
            return std::nullopt;
        }
        return handle->rank;
    };

    std::vector<TensorId> wanted;
    switch (node.kind) {
    case OpKind::Einsum:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Dot:
        if (node.outputs.size() != 1 || node.inputs.size() < 2) {
            return std::nullopt;
        }
        wanted = {node.outputs[0], node.inputs[0], node.inputs[1]};
        break;
    case OpKind::Permute:
    case OpKind::Axpby:
        if (node.outputs.size() != 1 || node.inputs.empty()) {
            return std::nullopt;
        }
        wanted = {node.outputs[0], node.inputs[0]};
        break;
    case OpKind::Scale:
        if (node.outputs.size() != 1) {
            return std::nullopt;
        }
        wanted = {node.outputs[0]};
        break;
    default:
        if (!registered_sliceable(node) || node.outputs.size() != 1) {
            return std::nullopt;
        }
        // The destination, then the operands it reads; a destination re-listed because the node
        // reads it is the first slot already.
        wanted = {node.outputs[0]};
        for (TensorId const tid : operand_inputs(node)) {
            wanted.push_back(tid);
        }
        break;
    }

    std::vector<SlotRef> slots;
    slots.reserve(wanted.size());
    for (TensorId const tid : wanted) {
        auto const rank = handle_rank(tid);
        if (!rank.has_value()) {
            return std::nullopt;
        }
        slots.push_back(SlotRef{.tid = tid, .rank = *rank});
    }
    return slots;
}

std::optional<std::vector<std::vector<std::string>>> letters_of(Node const &node) {
    if (registered_sliceable(node)) {
        return registered_letters(node);
    }
    auto const lists = node_index_lists(node);
    if (!lists) {
        return std::nullopt;
    }
    if (lists->b == nullptr) {
        return std::vector<std::vector<std::string>>{*lists->c, *lists->a};
    }
    return std::vector<std::vector<std::string>>{*lists->c, *lists->a, *lists->b};
}

std::vector<std::string> operator_letters_of(Node const &node) {
    std::vector<std::string> out;
    if (auto const lists = node_index_lists(node); lists && lists->operators != nullptr) {
        for (auto const &op : *lists->operators) {
            auto const letters = op.letters();
            out.insert(out.end(), letters.begin(), letters.end());
        }
    }
    return out;
}

std::vector<std::string> link_letters_of(Node const &node) {
    // A registered node sums the letters an input names and its output does not.
    if (registered_sliceable(node)) {
        std::vector<std::string> summed;
        if (auto const letters = registered_letters(node); letters && !letters->empty()) {
            auto const &output = letters->front();
            for (std::size_t s = 1; s < letters->size(); ++s) {
                for (auto const &letter : (*letters)[s]) {
                    if (std::ranges::find(output, letter) == output.end() && std::ranges::find(summed, letter) == summed.end()) {
                        summed.push_back(letter);
                    }
                }
            }
        }
        return summed;
    }
    if (node.kind != OpKind::Einsum) {
        return {};
    }
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr) {
        return {};
    }
    return einsum_index_lists(*desc).link;
}

bool unify(std::vector<int> &lhs, std::vector<int> &rhs, bool &changed) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] == rhs[i]) {
            continue;
        }
        if (lhs[i] == kFree) {
            lhs[i]  = rhs[i];
            changed = true;
        } else if (rhs[i] == kFree) {
            rhs[i]  = lhs[i];
            changed = true;
        } else {
            return false;
        }
    }
    return true;
}

bool labels_are_distinct(std::vector<int> const &labels) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == kFree) {
            continue;
        }
        for (std::size_t j = i + 1; j < labels.size(); ++j) {
            if (labels[i] == labels[j]) {
                return false;
            }
        }
    }
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::axis_tiling)
