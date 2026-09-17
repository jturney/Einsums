//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerLinearity.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// An overwriting permute that carries operators: `dst = alpha * P[G](src)`.
struct OperatorWrite {
    std::size_t                      index{0};
    TensorId                         source{0};
    PrefactorScalar                  alpha{double{1}};
    std::vector<PermutationOperator> operators;
};

std::optional<OperatorWrite> as_operator_write(Node const &node, std::size_t index) {
    if (node.kind != OpKind::Permute || node.inputs.size() != 1 || node.outputs.size() != 1) {
        return std::nullopt;
    }
    auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
    if (desc == nullptr || desc->operators.empty()) {
        return std::nullopt;
    }
    bool const overwrites = desc->params != nullptr ? is_zero(desc->params->beta) : desc->beta == std::complex<double>{0.0, 0.0};
    if (!overwrites) {
        return std::nullopt;
    }
    return OperatorWrite{.index     = index,
                         .source    = node.inputs[0],
                         .alpha     = desc->params != nullptr ? desc->params->alpha : PrefactorScalar{desc->alpha.real()},
                         .operators = desc->operators};
}

} // namespace

std::vector<std::string> AntisymmetrizerLinearity::explain() const {
    if (_num_merged == 0) {
        return {};
    }
    return {
        fmt::format("AntisymmetrizerLinearity: merged {} of {} antisymmetrized sum(s) into one operator", _num_merged, _num_candidates)};
}

void AntisymmetrizerLinearity::reset_stats() {
    _num_candidates = 0;
    _num_merged     = 0;
}

bool AntisymmetrizerLinearity::run(Graph &graph) {
    graph.topological_sort();

    auto const  guard          = EscapeAnalysis::over(graph);
    std::size_t original_count = graph.nodes().size();

    // Non-lifecycle writes per tensor, in program order.
    std::map<TensorId, std::vector<std::size_t>> writers;
    for (std::size_t i = 0; i < original_count; ++i) {
        if (is_lifecycle(graph.nodes()[i].kind)) {
            continue;
        }
        for (auto const out : graph.nodes()[i].outputs) {
            writers[graph.resolve_alias(out)].push_back(i);
        }
    }

    struct Site {
        TensorId        destination{0};
        std::size_t     operator_index{0}; ///< the write to rewrite
        std::size_t     axpby_index{0};    ///< the write to delete
        TensorId        own_source{0};     ///< B
        TensorId        other_source{0};   ///< A, behind the accumulated operand
        PrefactorScalar own_alpha{double{1}};
        PrefactorScalar merged_alpha{double{1}}; ///< s * c, the weight A enters with
    };
    std::vector<Site> sites;

    for (auto const &[tid, list] : writers) {
        if (list.size() != 2) {
            continue;
        }
        auto const first = as_operator_write(graph.nodes()[list[0]], list[0]);
        if (!first.has_value()) {
            continue;
        }
        Node const &second = graph.nodes()[list[1]];
        if (second.kind != OpKind::Axpby || second.inputs.size() < 2 || second.outputs.size() != 1) {
            continue;
        }
        auto const *adesc = std::get_if<AxpbyDescriptor>(&second.op_data);
        if (adesc == nullptr || !is_one(live_beta(*adesc))) {
            continue; // anything but a pure accumulate changes what the sum is
        }
        ++_num_candidates;

        TensorId const other        = graph.resolve_alias(second.inputs[0]);
        auto const    &other_writes = writers[other];
        if (other_writes.size() != 1) {
            note_skip("the accumulated operand is not settled by a single write", fmt::format("tensor #{}", tid));
            continue;
        }
        auto const behind = as_operator_write(graph.nodes()[other_writes[0]], other_writes[0]);
        if (!behind.has_value()) {
            note_skip("the accumulated operand was not written by a permutation operator", fmt::format("tensor #{}", tid));
            continue;
        }
        // The SAME operator, or the sum does not factor through one.
        if (behind->operators.size() != first->operators.size()) {
            note_skip("the two operators differ, so the sum does not factor through one", fmt::format("tensor #{}", tid));
            continue;
        }
        bool same = true;
        for (std::size_t k = 0; k < behind->operators.size() && same; ++k) {
            same = behind->operators[k].groups == first->operators[k].groups;
        }
        if (!same) {
            note_skip("the two operators differ, so the sum does not factor through one", fmt::format("tensor #{}", tid));
            continue;
        }

        auto const *dst_handle = graph.find_tensor(tid);
        if (dst_handle == nullptr || !dst_handle->is_intermediate || guard.touched_by_subtree(tid)) {
            note_skip("the destination is not a graph-owned intermediate this graph settles", fmt::format("tensor #{}", tid));
            continue;
        }

        sites.push_back(Site{.destination    = tid,
                             .operator_index = list[0],
                             .axpby_index    = list[1],
                             .own_source     = first->source,
                             .other_source   = behind->source,
                             .own_alpha      = first->alpha,
                             .merged_alpha   = PrefactorScalar{as<double>(live_alpha(*adesc)) * as<double>(behind->alpha)}});
    }

    if (sites.empty()) {
        return false;
    }

    std::vector<bool>                                      remove(original_count, false);
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;

    for (auto const &site : sites) {
        auto const *src_handle = graph.find_tensor(graph.resolve_alias(site.own_source));
        if (src_handle == nullptr) {
            continue;
        }

        auto sum = graph.create_zero_runtime_tensor_dynamic(fmt::format("_asym_sum_{}", _num_merged), src_handle->dtype, src_handle->dims);
        if (!sum) {
            note_skip("a tensor to hold the sum could not be created", fmt::format("tensor #{}", site.destination));
            continue;
        }
        TensorId const sum_id = sum.value().first;

        // sum = own_alpha * B, then sum += (s * c) * A. Both prefactors are
        // folded in here, which is what lets the operator itself run unscaled.
        std::vector<Node> group;
        group.push_back(
            graph.make_axpby_node(site.own_source, sum_id, site.own_alpha, PrefactorScalar{double{0}}, "antisym linearity: sum = a*B"));
        group.push_back(graph.make_axpby_node(site.other_source, sum_id, site.merged_alpha, PrefactorScalar{double{1}},
                                              "antisym linearity: sum += s*c*A"));

        Node &op  = graph.nodes()[site.operator_index];
        op.inputs = {sum_id};
        if (auto *pdesc = std::get_if<PermuteDescriptor>(&op.op_data)) {
            pdesc->alpha = std::complex<double>{1.0, 0.0};
            if (pdesc->params != nullptr) {
                pdesc->params->alpha = PrefactorScalar{double{1}};
            }
        }
        op.label = fmt::format("{} (linearity: over the summed source)", op.label);
        // REBUILD the executor: the closure resolved its operand when it was
        // built, so repointing inputs alone would leave the replay reading the
        // old source.
        op.execute = build_executor(OpKind::Permute, src_handle->dtype, src_handle->rank, op.op_data, graph,
                                    std::span<TensorId const>{op.inputs}, std::span<TensorId const>{op.outputs});

        remove[site.axpby_index] = true;
        inserts.emplace_back(site.operator_index, std::move(group));
        ++_num_merged;
    }

    if (inserts.empty()) {
        return false;
    }

    graph.replace_nodes(remove, std::move(inserts));
    graph.topological_sort();
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
