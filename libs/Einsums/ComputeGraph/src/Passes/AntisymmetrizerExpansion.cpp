//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerExpansion.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

std::vector<std::string> AntisymmetrizerExpansion::explain() const {
    if (_num_expanded == 0) {
        return {};
    }
    return {fmt::format("AntisymmetrizerExpansion: lowered {} of {} operator site(s) into {} permuted accumulation(s)", _num_expanded,
                        _num_sites, _num_terms)};
}

void AntisymmetrizerExpansion::reset_stats() {
    _num_sites    = 0;
    _num_expanded = 0;
    _num_terms    = 0;
}

bool AntisymmetrizerExpansion::run(Graph &graph) {
    // Sites are collected against the ORIGINAL numbering, which is the only one
    // a pass that has not mutated anything can speak, and which is also what
    // Graph::replace_nodes expects for both the removal mask and the insert
    // positions.
    std::size_t const        original_count = graph.nodes().size();
    std::vector<std::size_t> sites;
    for (std::size_t i = 0; i < original_count; ++i) {
        Node const &node = graph.nodes()[i];
        if (node.kind != OpKind::Einsum) {
            continue;
        }
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr) {
            continue;
        }
        // The LIVE operators, for the reason the IR writer prefers them: the
        // executor reads ``indices->spec``, and the descriptor's own copy beside
        // it is the at-capture snapshot.
        auto const &operators = desc->indices != nullptr ? desc->indices->spec.operators : desc->operators;
        if (operators.empty()) {
            continue;
        }
        // Refused where the node is built, so only a node a pass edited or a loaded one gets here:
        // the expansion's temporary takes C's type and its terms go through the same-type permute.
        if (!einsum_is_uniform(graph, node)) {
            note_skip("the operands hold different element types", fmt::format("einsum #{}", i));
            continue;
        }
        sites.push_back(i);
    }

    if (sites.empty()) {
        return false;
    }
    _num_sites += sites.size();

    auto const skip = [&](std::string_view why, std::size_t index) { note_skip(why, fmt::format("einsum #{}", index)); };

    std::vector<bool>                                      remove(original_count, false);
    std::vector<std::pair<std::size_t, std::vector<Node>>> inserts;

    for (std::size_t const index : sites) {
        {
            Node const &node = graph.nodes()[index];
            if (node.inputs.size() < 2 || node.outputs.size() != 1) {
                skip("the node does not have two inputs and one output", index);
                continue;
            }
        }
        // Copied, not referenced: create_zero_runtime_tensor_dynamic below
        // appends an Alloc node and may reallocate the node vector, which would
        // dangle anything held into it across the call.
        TensorId const a_id = graph.nodes()[index].inputs[0];
        TensorId const b_id = graph.nodes()[index].inputs[1];
        TensorId const c_id = graph.nodes()[index].outputs[0];

        auto const *c_handle = graph.find_tensor(graph.resolve_alias(c_id));
        if (c_handle == nullptr || c_handle->dims.empty() || c_handle->dtype == packed_gemm::ScalarType::Unknown) {
            skip("the destination has no usable dtype and extents for a temporary", index);
            continue;
        }

        // Read everything wanted off the descriptor BEFORE the tensor is created,
        // for the dangling reason above.
        ParsedEinsumSpec                 base;
        std::vector<PermutationOperator> operators;
        PrefactorScalar                  c_pf;
        PrefactorScalar                  ab_pf;
        bool                             conj_a = false;
        bool                             conj_b = false;
        {
            auto const *desc = graph.nodes()[index].op_data.get_if<EinsumDescriptor>();
            bool const  live = desc->indices != nullptr;
            base.c_indices   = live ? desc->indices->spec.c_indices : desc->spec.c_indices;
            base.a_indices   = live ? desc->indices->spec.a_indices : desc->spec.a_indices;
            base.b_indices   = live ? desc->indices->spec.b_indices : desc->spec.b_indices;
            operators        = live ? desc->indices->spec.operators : desc->operators;
            c_pf             = live_c_prefactor(*desc);
            ab_pf            = live_ab_prefactor(*desc);
            conj_a           = live_conj_a(*desc);
            conj_b           = live_conj_b(*desc);
        }
        base.raw = base.render();

        auto const terms = expand_permutation_operators(base.c_indices, operators);
        if (terms.size() < 2) {
            skip("the operator expands to a single term", index);
            continue;
        }

        auto scratch =
            graph.create_zero_runtime_tensor_dynamic(fmt::format("_asym_{}_{}", name(), _num_expanded), c_handle->dtype, c_handle->dims);
        if (!scratch) {
            skip("a temporary shaped like the destination could not be created", index);
            continue;
        }
        TensorId const tmp_id = scratch.value().first;

        std::vector<Node> group;
        group.reserve(terms.size() + 1);

        // The contraction, with the operator dropped: it is realized by the
        // accumulations below, and leaving it on would make the executor expand
        // it a second time.
        try {
            group.push_back(graph.make_einsum_node(a_id, b_id, tmp_id, base, PrefactorScalar{double{0}}, ab_pf, conj_a, conj_b,
                                                   fmt::format("antisym: tmp = {}", base.raw)));
        } catch (std::invalid_argument const &e) {
            skip(fmt::format("the contraction could not be rebuilt ({})", e.what()), index);
            continue;
        }

        // C's own prefactor rides on the FIRST term, so it applies once rather
        // than once per term, and the identity term is first so the term that may
        // overwrite is the one that does. ab_pf was applied by the contraction
        // above, so each term carries only its sign.
        ParsedPermuteSpec pspec;
        pspec.a_indices = base.c_indices;
        bool ok         = true;
        for (std::size_t t = 0; t < terms.size() && ok; ++t) {
            pspec.c_indices = terms[t].c_indices;
            pspec.raw       = pspec.render();
            try {
                group.push_back(graph.make_permute_node(
                    tmp_id, c_id, pspec, PrefactorScalar{terms[t].sign}, t == 0 ? c_pf : PrefactorScalar{double{1}},
                    fmt::format("antisym: C {}= {} * P({})", t == 0 ? "" : "+", terms[t].sign, fmt::join(terms[t].c_indices, ","))));
            } catch (std::invalid_argument const &e) {
                skip(fmt::format("a permuted accumulation could not be built ({})", e.what()), index);
                ok = false;
            }
        }
        if (!ok) {
            continue;
        }

        remove[index] = true;
        inserts.emplace_back(index, std::move(group));
        ++_num_expanded;
        _num_terms += terms.size();
    }

    if (inserts.empty()) {
        return false;
    }

    // The mask is sized to the ORIGINAL prefix; the Alloc nodes the scratch
    // creation appended sit past its end and are kept, which is the behaviour
    // Graph::replace_nodes documents and which every planning pass relies on.
    graph.replace_nodes(remove, std::move(inserts));
    graph.topological_sort();
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
