//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerFolding.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// What a node carrying a permutation operator offers this pass.
struct OperatorProducer {
    TensorId                         source{0}; ///< the operand the operator is applied to
    std::vector<PermutationOperator> operators; ///< the operators, as written
    std::vector<std::string>         c_indices; ///< the output index list they permute
    std::size_t                      terms{0};  ///< the expansion's term count, the factor N
    bool                             overwrites{false};
};

std::optional<OperatorProducer> read_producer(Node const &node) {
    OperatorProducer out;

    if (node.kind == OpKind::Permute) {
        auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
        if (desc == nullptr || desc->operators.empty() || node.inputs.size() != 1) {
            return std::nullopt;
        }
        out.source     = node.inputs[0];
        out.operators  = desc->operators;
        out.c_indices  = desc->c_indices;
        out.overwrites = desc->params != nullptr ? is_zero(desc->params->beta) : desc->beta == std::complex<double>{0.0, 0.0};
    } else {
        // Only the permute form is folded. An einsum's operator wraps a
        // CONTRACTION, so repointing the consumer at "the source" would mean
        // pointing it at a product that was never materialized; folding that
        // shape means rewriting the contraction itself, which is a different and
        // larger rewrite than this one.
        return std::nullopt;
    }

    out.terms = expand_permutation_operators(out.c_indices, out.operators).size();
    if (out.terms < 2) {
        return std::nullopt;
    }
    return out;
}

/// The antisymmetry the identity requires of the OTHER operand.
///
/// Built exactly as @ref AntisymmetryInference builds what it tags, so the
/// containment test below compares like with like. That is a deliberate
/// narrowness: two descriptors can state the same group through different
/// generators, and this accepts only the spelling the inference pass produces.
/// The cost is declining a fact it could have used; the alternative is closing
/// generator sets under composition to compare groups, which is a much larger
/// claim to get right for a rewrite that changes the answer when it is wrong.
std::optional<SymmetryDescriptor> required_antisymmetry(OperatorProducer const &producer) {
    SymmetryDescriptor desc;

    for (auto const &op : producer.operators) {
        std::vector<int> axes;
        for (auto const &letter : op.letters()) {
            auto const first = std::ranges::find(producer.c_indices, letter);
            if (first == producer.c_indices.end() || std::count(producer.c_indices.begin(), producer.c_indices.end(), letter) != 1) {
                return std::nullopt;
            }
            axes.push_back(static_cast<int>(first - producer.c_indices.begin()));
        }
        std::ranges::sort(axes);
        if (axes.size() < 2 || axes.back() >= kMaxSymmetryRank) {
            return std::nullopt;
        }
        for (std::size_t k = 0; k + 1 < axes.size(); ++k) {
            desc.add(SymmetryOp::swap(axes[k], axes[k + 1], -1));
        }
    }
    if (desc.empty()) {
        return std::nullopt;
    }
    return desc;
}

bool contains_all(SymmetryDescriptor const &have, SymmetryDescriptor const &need) {
    return std::ranges::all_of(need.ops, [&](SymmetryOp const &op) { return std::ranges::find(have.ops, op) != have.ops.end(); });
}

} // namespace

std::vector<std::string> AntisymmetrizerFolding::explain() const {
    if (_num_folded == 0) {
        return {};
    }
    return {fmt::format("AntisymmetrizerFolding: collapsed {} of {} antisymmetrized contraction(s) into a scalar multiple", _num_folded,
                        _num_candidates)};
}

void AntisymmetrizerFolding::reset_stats() {
    _num_candidates = 0;
    _num_folded     = 0;
}

bool AntisymmetrizerFolding::run(Graph &graph) {
    graph.topological_sort();

    auto const guard = EscapeAnalysis::over(graph);

    std::size_t const original_count = graph.nodes().size();

    // (dot position, operand slot, producer)
    struct Site {
        std::size_t      dot_index{0};
        std::size_t      slot{0};
        OperatorProducer producer;
        TensorId         result{0};
    };
    std::vector<Site> sites;

    for (std::size_t i = 0; i < original_count; ++i) {
        Node const &dot = graph.nodes()[i];
        if (dot.kind != OpKind::Dot || dot.inputs.size() != 2 || dot.outputs.size() != 1) {
            continue;
        }

        for (std::size_t slot = 0; slot < 2; ++slot) {
            TensorId const folded = graph.resolve_alias(dot.inputs[slot]);
            TensorId const other  = graph.resolve_alias(dot.inputs[1 - slot]);

            // The operand must be written exactly once in this graph and not
            // touched by a child sub-graph, or the value the identity reasons
            // about is not the value the contraction reads. That is
            // EscapeAnalysis, the same guard AntisymmetryInference tagged under.
            //
            // The first draft of this hand-rolled the scan and counted the OTHER
            // operand's own producer as interference, which is every graph of
            // this shape: the two antisymmetrized quantities are built one after
            // the other and then contracted. A second writer of an operand is
            // interference; the first one is the point.
            if (!guard.stable(folded) || !guard.stable(other)) {
                continue;
            }

            std::optional<std::size_t> producer_index;
            for (std::size_t j = 0; j < i; ++j) {
                Node const &node = graph.nodes()[j];
                if (is_lifecycle(node.kind)) {
                    continue;
                }
                if (std::ranges::any_of(node.outputs, [&](TensorId t) { return graph.resolve_alias(t) == folded; })) {
                    producer_index = j;
                }
            }
            if (!producer_index.has_value()) {
                continue;
            }

            auto producer = read_producer(graph.nodes()[*producer_index]);
            if (!producer.has_value()) {
                continue;
            }
            ++_num_candidates;

            if (!producer->overwrites) {
                note_skip("the operator node accumulates, so its output is not the operator's value", fmt::format("dot #{}", dot.id));
                continue;
            }

            auto const need = required_antisymmetry(*producer);
            if (!need.has_value()) {
                note_skip("the operator's letters do not each name one addressable axis", fmt::format("dot #{}", dot.id));
                continue;
            }

            auto const *other_handle = graph.find_tensor(other);
            if (other_handle == nullptr || other_handle->symmetry_hint == nullptr) {
                note_skip("the other operand carries no antisymmetry, so the terms do not collapse", fmt::format("dot #{}", dot.id));
                continue;
            }
            if (!contains_all(*other_handle->symmetry_hint, *need)) {
                note_skip("the other operand's antisymmetry does not cover every axis the operator permutes",
                          fmt::format("dot #{}", dot.id));
                continue;
            }

            sites.push_back(Site{.dot_index = i, .slot = slot, .producer = *producer, .result = dot.outputs[0]});
            break; // one fold per contraction
        }
    }

    if (sites.empty()) {
        return false;
    }

    // Repoint each contraction at the operator's source, then scale by the term
    // count. The scale goes in a separate node because Dot carries no prefactor,
    // and immediately after, so no reader of the result can observe the
    // unscaled value.
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;
    for (auto const &site : sites) {
        Node &dot             = graph.nodes()[site.dot_index];
        dot.inputs[site.slot] = site.producer.source;
        dot.label             = fmt::format("{} (antisymmetrizer folded, x{})", dot.label, site.producer.terms);

        // REBUILD the executor. Rewriting node.inputs alone changes what the
        // dataflow says and not what the replay does: the closure build_executor
        // returned resolved its operands when it was built, so the contraction
        // went on reading the antisymmetrized operand and the scale below then
        // multiplied an already-complete answer by N. The test caught it because
        // it compares NUMBERS against an unfolded run; a structural check would
        // have passed.
        auto const *operand_handle = graph.find_tensor(graph.resolve_alias(dot.inputs[0]));
        if (operand_handle == nullptr) {
            continue;
        }
        dot.execute = build_executor(OpKind::Dot, operand_handle->dtype, operand_handle->rank, dot.op_data, graph,
                                     std::span<TensorId const>{dot.inputs}, std::span<TensorId const>{dot.outputs});

        auto const *result_handle = graph.find_tensor(graph.resolve_alias(site.result));
        if (result_handle == nullptr) {
            continue;
        }
        // A Dot whose destination is a bare scalar registers a rank-0 tensor,
        // and build_executor refuses a Scale on one: its destination list does
        // not admit rank 0. Decline rather than throw from inside a pass; the
        // tensor-destination spelling, which is what the Python path and the
        // chemistry examples use, gives a rank-1 single-element result.
        if (result_handle->rank == 0) {
            note_skip("the contraction writes a bare scalar, which cannot carry the fold's scale node",
                      fmt::format("dot #{}", graph.nodes()[site.dot_index].id));
            continue;
        }

        ScaleDescriptor desc;
        desc.factor        = PrefactorScalar{static_cast<double>(site.producer.terms)};
        desc.params        = std::make_shared<ElementwiseParams>();
        desc.params->alpha = desc.factor;

        Node scale;
        scale.id      = graph.reserve_node_id();
        scale.kind    = OpKind::Scale;
        scale.label   = fmt::format("antisymmetrizer fold: x{}", site.producer.terms);
        scale.inputs  = {site.result};
        scale.outputs = {site.result};
        scale.op_data = OpData(std::move(desc));
        scale.execute = build_executor(OpKind::Scale, result_handle->dtype, result_handle->rank, scale.op_data, graph, {},
                                       std::span<TensorId const>{scale.outputs});

        std::vector<Node> group;
        group.push_back(std::move(scale));
        inserts.emplace_back(site.dot_index + 1, std::move(group));
        ++_num_folded;
    }

    if (inserts.empty()) {
        return false;
    }

    std::vector<bool> const remove(original_count, false);
    graph.replace_nodes(remove, std::move(inserts));
    graph.topological_sort();
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
