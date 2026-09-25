//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// An einsum node keeps its index lists and scalars twice: a snapshot on the descriptor, which
// analysis passes and the IR read, and live blocks the executor reads on every call. Every builder
// seeds the live blocks through one helper, and a rewriter changes an index list through one setter
// that writes both. These cases pin that the builders agree and that the setter keeps the two in step.

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

// The builders used to assemble the live blocks themselves and drifted: only the loader set the
// conjugation flags on the index block, and capture left its diagnostic text empty.
TEST_CASE("EinsumLiveState - a captured node carries complete live blocks", "[ComputeGraph][EinsumLiveState]") {
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
    REQUIRE(desc->params != nullptr);
    REQUIRE(desc->indices != nullptr);
    REQUIRE(desc->site != nullptr);
    CHECK(desc->params->conj_a);
    CHECK(desc->indices->spec.conj_a);
    CHECK_FALSE(desc->indices->spec.raw.empty());
    CHECK(desc->indices->link_indices == Letters{"k"});
}

TEST_CASE("EinsumLiveState - attach_live_state seeds every block from the snapshot", "[ComputeGraph][EinsumLiveState]") {
    auto const parsed = cg::parse_einsum_spec("i,j <- conj(i,k) ; k,j").value();
    auto       desc   = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.5}, cg::PrefactorScalar{2.0}, true, false);
    REQUIRE(desc.params == nullptr);
    cg::detail::attach_live_state(desc);

    REQUIRE(desc.params != nullptr);
    CHECK(cg::as<double>(desc.params->c_pf) == 0.5);
    CHECK(cg::as<double>(desc.params->ab_pf) == 2.0);
    CHECK(desc.params->conj_a);
    REQUIRE(desc.indices != nullptr);
    CHECK(desc.indices->spec.a_indices == Letters{"i", "k"});
    CHECK(desc.indices->spec.conj_a);
    CHECK(desc.indices->spec.raw == desc.indices->spec.render());
    CHECK(desc.site != nullptr);
}

TEST_CASE("EinsumLiveState - live_index_lists prefers the live block", "[ComputeGraph][EinsumLiveState]") {
    auto const parsed = cg::parse_einsum_spec("ij <- ik ; kj").value();
    auto       desc   = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.0}, cg::PrefactorScalar{1.0});

    // No live block: the snapshot is what runs.
    CHECK(cg::live_index_lists(desc).a == Letters{"i", "k"});
    CHECK(cg::live_index_lists(desc).link == Letters{"k"});

    cg::detail::attach_live_state(desc);
    desc.indices->spec.a_indices = {"k", "i"}; // a write the snapshot has not seen
    CHECK(cg::live_index_lists(desc).a == Letters{"k", "i"});
}

TEST_CASE("EinsumLiveState - set_operand_indices writes both copies and re-derives the links", "[ComputeGraph][EinsumLiveState]") {
    auto const parsed = cg::parse_einsum_spec("ij <- ik ; kj").value();
    auto       desc   = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.0}, cg::PrefactorScalar{1.0});
    cg::detail::attach_live_state(desc);

    // A permutation of an operand's axes: the letter sets, and so the links, are unchanged.
    cg::set_operand_indices(desc, cg::EinsumOperand::A, {"k", "i"});
    CHECK(desc.spec.a_indices == Letters{"k", "i"});
    CHECK(desc.indices->spec.a_indices == Letters{"k", "i"});
    CHECK(desc.spec.link_indices == Letters{"k"});
    CHECK(desc.indices->link_indices == Letters{"k"});
    CHECK(desc.indices->spec.raw == desc.indices->spec.render());

    // A relabeling that changes which letter is contracted re-derives them.
    cg::set_operand_indices(desc, cg::EinsumOperand::B, {"m", "j"});
    cg::set_operand_indices(desc, cg::EinsumOperand::A, {"i", "m"});
    CHECK(desc.spec.link_indices == Letters{"m"});
    CHECK(desc.indices->link_indices == Letters{"m"});
    CHECK(desc.spec.target_indices == Letters{"i", "j"});

    // Without a live block the snapshot alone is written.
    auto snapshot_only = cg::detail::build_einsum_descriptor(parsed, cg::PrefactorScalar{0.0}, cg::PrefactorScalar{1.0});
    cg::set_operand_indices(snapshot_only, cg::EinsumOperand::C, {"j", "i"});
    CHECK(snapshot_only.spec.c_indices == Letters{"j", "i"});
}
