//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <complex>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

bool reads_destination(Node const &node) {
    if (DescriptorHooks const *hooks = descriptor_hooks(node); hooks != nullptr && hooks->destination) {
        return hooks->destination(node.op_data, node).reads;
    }
    switch (node.kind) {
    case OpKind::Scale:
    case OpKind::ElementTransform:
        return true;
    default:
        break;
    }
    // The LIVE prefactors, which are what the next execute() reads: a pass that folds a scale
    // into one writes it through the shared params, not the snapshot beside them.
    if (node.kind == OpKind::Axpby) {
        auto const *axpby = node.op_data.get_if<AxpbyDescriptor>();
        return axpby == nullptr || !is_zero(live_beta(*axpby));
    }
    if (auto const *e = node.op_data.get_if<EinsumDescriptor>()) {
        return !is_zero(live_c_prefactor(*e));
    }
    if (auto const *t = node.op_data.get_if<TiledEinsumDescriptor>()) {
        return t->params == nullptr || !is_zero(t->params->c_pf);
    }
    if (auto const *p = node.op_data.get_if<PermuteDescriptor>()) {
        return p->params != nullptr ? !is_zero(p->params->beta) : p->beta != 0.0;
    }
    if (auto const *p = node.op_data.get_if<TiledPermuteDescriptor>()) {
        return !is_zero(p->beta);
    }
    if (auto const *b = node.op_data.get_if<BatchedGemmDescriptor>()) {
        return b->beta != std::complex<double>{0.0, 0.0};
    }
    if (auto const *e = node.op_data.get_if<ElementwiseBinaryDescriptor>()) {
        return !is_zero(live_beta(*e));
    }
    if (auto const *t = node.op_data.get_if<TiledElementwiseDescriptor>()) {
        // A tiled scale works in place and a tiled axpy adds into Y; only the quotient has a beta.
        if (t->op != TiledElementwiseOp::Divide) {
            return true;
        }
        return t->params == nullptr || !is_zero(t->params->beta);
    }
    if (auto const *g = node.op_data.get_if<GemmDescriptor>()) {
        return !is_zero(g->beta);
    }
    return false;
}

std::optional<std::size_t> destination_operand_count(Node const &node) {
    if (node.outputs.size() != 1) {
        return std::nullopt;
    }
    if (DescriptorHooks const *hooks = descriptor_hooks(node); hooks != nullptr && hooks->destination) {
        return hooks->destination(node.op_data, node).operand_count;
    }
    switch (node.kind) {
    case OpKind::Scale:
    case OpKind::ElementTransform:
        return 0; // in place: the destination is the only tensor it lists
    case OpKind::Axpby:
        return 1; // x
    default:
        break;
    }
    if (node.op_data.holds<EinsumDescriptor>() || node.op_data.holds<TiledEinsumDescriptor>() ||
        node.op_data.holds<ElementwiseBinaryDescriptor>() || node.op_data.holds<GemmDescriptor>()) {
        return 2; // A, B
    }
    if (node.op_data.holds<PermuteDescriptor>() || node.op_data.holds<TiledPermuteDescriptor>()) {
        return 1; // A
    }
    if (auto const *t = node.op_data.get_if<TiledElementwiseDescriptor>()) {
        switch (t->op) {
        case TiledElementwiseOp::Scale:
            return 0;
        case TiledElementwiseOp::Axpy:
            return 1;
        case TiledElementwiseOp::Divide:
            return 2;
        }
    }
    return std::nullopt;
}

std::span<TensorId const> operand_inputs(Node const &node) {
    std::span<TensorId const> const all{node.inputs};
    auto const                      operands = destination_operand_count(node);
    if (!operands.has_value() || all.size() < *operands) {
        return all;
    }
    return all.first(*operands);
}

void sync_destination_input(Node &node) {
    auto const operands = destination_operand_count(node);
    if (!operands.has_value() || node.inputs.size() < *operands) {
        return;
    }
    node.inputs.resize(*operands);
    if (reads_destination(node)) {
        node.inputs.push_back(node.outputs[0]);
    }
}

std::optional<std::string> destination_rule_violation(Graph const &graph, Node const &node) {
    auto const operands = destination_operand_count(node);
    if (!operands.has_value()) {
        return std::nullopt;
    }
    bool const        reads    = reads_destination(node);
    std::size_t const expected = *operands + (reads ? 1 : 0);
    if (node.inputs.size() != expected) {
        return fmt::format("{} its destination but lists {} input(s) where its {} operand(s){} make {}", reads ? "reads" : "does not read",
                           node.inputs.size(), *operands, reads ? " and the destination" : "", expected);
    }
    if (reads && graph.buffer_of(node.inputs.back()) != graph.buffer_of(node.outputs[0])) {
        return std::string{"reads its destination but its last input is not the destination"};
    }
    return std::nullopt;
}

EINSUMS_NAMESPACE_END(compute_graph)
