//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerFolding.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "AntisymmetryRules.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// What a node carrying a permutation operator offers this pass.
struct OperatorProducer {
    TensorId                         source{0};  ///< the operand the operator is applied to
    std::vector<PermutationOperator> operators;  ///< the operators, as written
    std::vector<std::string>         c_indices;  ///< the output index list they permute
    std::size_t                      terms{0};   ///< the expansion's term count, the factor N
    PrefactorScalar                  alpha{1.0}; ///< the producer's own scale, C = alpha P(A)
    bool                             overwrites{false};
};

std::optional<OperatorProducer> read_producer(Node const &node) {
    OperatorProducer out;

    if (node.kind == OpKind::Permute) {
        auto const *desc = node.op_data.get_if<PermuteDescriptor>();
        if (desc == nullptr || desc->operators.empty() || node.inputs.size() != 1) {
            return std::nullopt;
        }
        // The fold replaces the permuted tensor by its source and multiplies by the term count
        // AND by the permute's own scale, `C = alpha P(A)`. The LIVE value, since a pass that
        // folded a scale into the permute wrote it there.
        out.alpha      = desc->params != nullptr     ? desc->params->alpha
                         : desc->alpha.imag() == 0.0 ? PrefactorScalar{desc->alpha.real()}
                                                     : PrefactorScalar{desc->alpha};
        out.source     = node.inputs[0];
        out.operators  = desc->operators;
        out.c_indices  = desc->c_indices;
        out.overwrites = pure_overwrite(node);
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

/// The facts DETECTION established: a hint on a tensor nothing in the graph
/// writes and something reads.
///
/// A tensor nothing reads founds no rewrite, whatever its hint says. Requiring a
/// reader also keeps the guard off a tensor another pass orphaned: after
/// SymmetrizedAccumulation drops a transpose, its deferred destination keeps the
/// hint SymmetryPropagation derived, has no writer, and is never materialized
/// where the guard runs.
///
/// Reconstructed by walking the graph rather than threaded through from the
/// detection pass. The alternative is provenance, recording per fold which leaves
/// its premise descended from, and the conservative set is both simpler and the
/// right thing to re-check: if any detected fact stops holding, some rewrite
/// justified by it may be invalid, and this pass cannot see which.
std::vector<std::pair<TensorId, SymmetryDescriptor>> detected_leaves(EscapeAnalysis const &writers) {
    Graph const                 &graph = writers.graph();
    std::unordered_set<TensorId> read;
    for (Node const &node : graph.nodes()) {
        for (TensorId const id : node.inputs) {
            read.insert(graph.buffer_of(id));
        }
    }
    std::vector<std::pair<TensorId, SymmetryDescriptor>> leaves;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (writers.writer_count(tid) != 0 || handle.symmetry_hint == nullptr || !handle.impl_fn) {
            continue;
        }
        if (!read.contains(graph.buffer_of(tid)) && !writers.touched_by_subtree(tid)) {
            continue;
        }
        leaves.emplace_back(tid, *handle.symmetry_hint);
    }
    return leaves;
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
        if (!understands(graph, dot)) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", dot.label, describe_features(features_of(graph, dot))));
            continue;
        }

        for (std::size_t slot = 0; slot < 2; ++slot) {
            TensorId const folded = graph.buffer_of(dot.inputs[slot]);
            TensorId const other  = graph.buffer_of(dot.inputs[1 - slot]);

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
                if (std::ranges::any_of(node.outputs, [&](TensorId t) { return graph.buffer_of(t) == folded; })) {
                    producer_index = j;
                }
            }
            if (!producer_index.has_value()) {
                continue;
            }
            // The fold reads the producer's operators and source, so a producer
            // carrying a feature this pass does not know is not one it can reason about.
            if (Node const &producer_node = graph.nodes()[*producer_index]; !understands(graph, producer_node)) {
                note_skip("the node carries a feature this pass does not understand",
                          fmt::format("node '{}': {}", producer_node.label, describe_features(features_of(graph, producer_node))));
                continue;
            }

            auto producer = read_producer(graph.nodes()[*producer_index]);
            if (!producer.has_value()) {
                continue;
            }
            ++_num_candidates;

            // The dot will read the operator's SOURCE where the dot stands, not where the operator
            // read it, so nothing in between may write that source: a loop or branch cannot be
            // shown not to, and any other writer of its buffer changes what the dot would read.
            TensorId const source_buffer = graph.buffer_of(producer->source);
            bool           rewritten     = false;
            for (std::size_t j = *producer_index + 1; j < i && !rewritten; ++j) {
                Node const &between = graph.nodes()[j];
                rewritten = is_control_flow(between.kind) ||
                            (!is_lifecycle(between.kind) &&
                             std::ranges::any_of(between.outputs, [&](TensorId t) { return graph.buffer_of(t) == source_buffer; }));
            }
            if (rewritten) {
                note_skip("the operator's source is written between the operator and the contraction", fmt::format("dot #{}", dot.id));
                continue;
            }

            if (!producer->overwrites) {
                note_skip("the operator node accumulates, so its output is not the operator's value", fmt::format("dot #{}", dot.id));
                continue;
            }

            auto const need = antisymmetry::full_antisymmetry(producer->operators, producer->c_indices);
            if (!need.has_value()) {
                note_skip("the operator's letters do not each name one addressable axis", fmt::format("dot #{}", dot.id));
                continue;
            }

            auto const *other_handle = graph.find_tensor(other);
            if (other_handle == nullptr || other_handle->symmetry_hint == nullptr) {
                note_skip("the other operand carries no antisymmetry, so the terms do not collapse", fmt::format("dot #{}", dot.id));
                continue;
            }
            if (!antisymmetry::contains_all(*other_handle->symmetry_hint, *need)) {
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
        // Every decline runs BEFORE the contraction is touched: a decline after
        // the repoint would leave a Dot reading the operator's source with no
        // scale node behind it, a wrong answer rather than a missed fold.
        TensorId const operand0       = site.slot == 0 ? site.producer.source : graph.nodes()[site.dot_index].inputs[0];
        auto const    *operand_handle = graph.find_tensor(graph.buffer_of(operand0));
        if (operand_handle == nullptr) {
            continue;
        }
        auto const operand_dtype = operand_handle->dtype;
        auto const operand_rank  = operand_handle->rank;

        auto const *result_handle = graph.find_tensor(graph.buffer_of(site.result));
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
        auto const result_dtype = result_handle->dtype;
        auto const result_rank  = result_handle->rank;

        // The dot is bilinear, or sesquilinear with its first operand conjugated, and the
        // operator's signs are real: <W, alpha P V> = alpha N <W, V>, and alpha comes out
        // conjugated from the conjugated slot. A real result cannot carry an imaginary alpha,
        // which the captured producer refuses at execute; the fold keeps that refusal where it is.
        auto const           *dot_desc   = graph.nodes()[site.dot_index].op_data.get_if<DotDescriptor>();
        bool const            conjugated = dot_desc != nullptr && dot_desc->conjugated;
        PrefactorScalar const alpha =
            conjugated && site.slot == 0 ? PrefactorScalar{std::conj(as<std::complex<double>>(site.producer.alpha))} : site.producer.alpha;
        bool const complex_result =
            result_dtype == packed_gemm::ScalarType::Complex64 || result_dtype == packed_gemm::ScalarType::Complex128;
        if (!complex_result && !is_real_valued(alpha)) {
            note_skip("the operator carries a complex scale and the contraction's result is real",
                      fmt::format("dot #{}", graph.nodes()[site.dot_index].id));
            continue;
        }
        auto const            terms = static_cast<double>(site.producer.terms);
        PrefactorScalar const factor =
            complex_result ? multiply_prefactors(PrefactorScalar{terms}, alpha) : PrefactorScalar{terms * as_real<double>(alpha)};

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
        dot.execute = build_executor(OpKind::Dot, operand_dtype, operand_rank, dot.op_data, graph, std::span<TensorId const>{dot.inputs},
                                     std::span<TensorId const>{dot.outputs});

        ScaleDescriptor desc;
        desc.factor        = factor;
        desc.params        = std::make_shared<ElementwiseParams>();
        desc.params->alpha = desc.factor;

        Node scale;
        scale.id      = graph.reserve_node_id();
        scale.kind    = OpKind::Scale;
        scale.label   = fmt::format("antisymmetrizer fold: x{}", to_string(factor));
        scale.inputs  = {site.result};
        scale.outputs = {site.result};
        scale.op_data = OpData(std::move(desc));
        scale.execute =
            build_executor(OpKind::Scale, result_dtype, result_rank, scale.op_data, graph, {}, std::span<TensorId const>{scale.outputs});

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

    // THE GUARD. Everything above rests on facts read out of the tensors bound
    // when this pass ran, and rebind() repoints a graph at new ones without
    // re-running the pipeline. ContractionPlanning states the rule this runs
    // into: a measurement may found a hint, not a premise. So the premise is
    // re-checked per bound problem.
    //
    // OpKind::Setup is the mechanism, and its contract is exactly the trigger
    // needed: "A bind clears computed, and the executor then reruns". The body
    // costs one boolean per replay and a sweep per bind, which is the same cost
    // as the detection that established the facts.
    //
    // At position zero, because a guard behind the work it guards has already
    // let the wrong answer be computed. add_setup_at exists for this reason.
    if (auto leaves = detected_leaves(EscapeAnalysis::over(graph)); !leaves.empty()) {
        Graph     &body   = graph.add_setup_at("antisymmetry premise guard", 0);
        auto const anchor = graph.anchor();

        Node check;
        check.id    = body.reserve_node_id();
        check.kind  = OpKind::Custom;
        check.label = fmt::format("verify {} detected symmetry fact(s) still hold", leaves.size());
        // The body's own ids for the leaves, minted through the pointer index: the check lives in
        // the body, so an id from the parent's table would name nothing there, and the setup's
        // reads, which the parent derives from its body, would name nothing either.
        for (auto const &[tid, desc] : leaves) {
            if (auto const *handle = graph.find_tensor(graph.buffer_of(tid)); handle != nullptr) {
                check.inputs.push_back(body.find_or_register_tensor_ptr(*handle));
            }
        }
        check.execute = [anchor, leaves = std::move(leaves)]() {
            for (auto const &[tid, desc] : leaves) {
                auto const *handle = anchor->graph().find_tensor(anchor->graph().buffer_of(tid));
                if (handle == nullptr || !handle->impl_fn) {
                    continue;
                }
                bool ok = false;
                detail::dispatch_scalar_type(handle->dtype, [&]<typename T>(T /*tag*/) {
                    using Impl       = ::einsums::detail::TensorImpl<T>;
                    auto const *impl = static_cast<Impl const *>(handle->impl_fn());
                    if (impl == nullptr || impl->data() == nullptr) {
                        ok = true; // nothing bound to contradict the fact
                        return;
                    }
                    RuntimeTensorView<T> const view{*impl};
                    ok = check_symmetry(view, desc);
                });
                if (!ok) {
                    EINSUMS_THROW_EXCEPTION(
                        std::runtime_error,
                        "AntisymmetrizerFolding: tensor '{}' no longer has the symmetry this graph's optimization assumed. The "
                        "antisymmetrizer fold was justified by reading the tensors bound when the pass ran; rebinding to data "
                        "without that symmetry makes the rewrite wrong. Re-run the pass pipeline on the new binding, or bind "
                        "data carrying the same symmetry. Note that this check runs once per BIND: a caller who overwrites a "
                        "validated tensor in place should call Graph::invalidate_setup to have it run again.",
                        handle->name);
                }
            }
        };
        body.add_node(std::move(check));
    }

    graph.topological_sort();
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
