//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// An einsum executor holds its own copy of the node's index lists and scalars, taken from the
// descriptor by detail::einsum_params_of and detail::einsum_indices_of, and a rewriter changes an
// index list through one setter that re-derives the links. These cases pin that the copy is
// complete and that the setter keeps the derived lists in step.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

using Letters = std::vector<std::string>;

cg::EinsumDescriptor const *only_einsum(cg::Graph const &graph) {
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Einsum) {
            return node.op_data.get_if<cg::EinsumDescriptor>();
        }
    }
    return nullptr;
}

} // namespace

// The builders used to assemble the executor's state themselves and drifted: only the loader set
// the conjugation flags on the index state, and capture left its diagnostic text empty.
TEST_CASE("EinsumLiveState - a captured node's executor state is complete", "[ComputeGraph][EinsumLiveState]") {
    auto A = create_random_tensor<std::complex<double>>("A", 3, 4);
    auto B = create_random_tensor<std::complex<double>>("B", 4, 5);
    auto C = create_zero_tensor<std::complex<double>>("C", 3, 5);

    cg::Graph graph("captured");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- conj(ik) ; kj", &C, A, B);
    }

    auto const *desc = only_einsum(graph);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->site != nullptr);
    auto const params  = cg::detail::einsum_params_of(*desc);
    auto const indices = cg::detail::einsum_indices_of(*desc);
    CHECK(params.conj_a);
    CHECK(indices.spec.conj_a);
    CHECK_FALSE(indices.spec.raw.empty());
    CHECK(indices.link_indices == Letters{"k"});
}

TEST_CASE("EinsumLiveState - the executor state is taken from the descriptor", "[ComputeGraph][EinsumLiveState]") {
    auto const parsed = cg::parse_einsum_spec("i,j <- conj(i,k) ; k,j").value();
    auto const desc   = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.5}, cg::PrefactorScalar{2.0}, true, false);

    auto const params = cg::detail::einsum_params_of(desc);
    CHECK(cg::as<double>(params.c_pf) == 0.5);
    CHECK(cg::as<double>(params.ab_pf) == 2.0);
    CHECK(params.conj_a);
    auto const indices = cg::detail::einsum_indices_of(desc);
    CHECK(indices.spec.a_indices == Letters{"i", "k"});
    CHECK(indices.spec.conj_a);
    CHECK(indices.spec.raw == indices.spec.render());
}

TEST_CASE("EinsumLiveState - set_operand_indices re-derives the links", "[ComputeGraph][EinsumLiveState]") {
    auto const parsed = cg::parse_einsum_spec("ij <- ik ; kj").value();
    auto       desc   = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.0}, cg::PrefactorScalar{1.0});

    // A permutation of an operand's axes: the letter sets, and so the links, are unchanged.
    cg::set_operand_indices(desc, cg::EinsumOperand::A, {"k", "i"});
    CHECK(desc.spec.a_indices == Letters{"k", "i"});
    CHECK(cg::einsum_index_lists(desc).a == Letters{"k", "i"});
    CHECK(desc.spec.link_indices == Letters{"k"});

    // A relabeling that changes which letter is contracted re-derives them.
    cg::set_operand_indices(desc, cg::EinsumOperand::B, {"m", "j"});
    cg::set_operand_indices(desc, cg::EinsumOperand::A, {"i", "m"});
    CHECK(desc.spec.link_indices == Letters{"m"});
    CHECK(cg::detail::einsum_indices_of(desc).link_indices == Letters{"m"});
    CHECK(desc.spec.target_indices == Letters{"i", "j"});
}
