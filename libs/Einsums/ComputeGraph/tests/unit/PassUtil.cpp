//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PassUtil.cpp
/// @brief Unit tests for the buffer I/O and interference helpers the passes share.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstddef>
#include <unordered_set>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Positions of the nodes of @p kind in @p graph, in program order.
std::vector<std::size_t> positions_of(cg::Graph const &graph, cg::OpKind kind) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].kind == kind) {
            out.push_back(i);
        }
    }
    return out;
}

} // namespace

TEST_CASE("PassUtil - conflicts is the read/write ordering rule", "[ComputeGraph][Passes][PassUtil]") {
    using cg::passes::BufferIo;
    using cg::passes::conflicts;

    BufferIo const reads_1{.reads = {1}, .writes = {}};
    BufferIo const writes_1{.reads = {}, .writes = {1}};
    BufferIo const writes_2{.reads = {}, .writes = {2}};

    CHECK_FALSE(conflicts(reads_1, reads_1)); // two reads commute
    CHECK(conflicts(writes_1, reads_1));      // RAW
    CHECK(conflicts(reads_1, writes_1));      // WAR
    CHECK(conflicts(writes_1, writes_1));     // WAW
    CHECK_FALSE(conflicts(writes_1, writes_2));
    CHECK_FALSE(conflicts(reads_1, writes_2));
}

// A Loop node's own input and output lists do not record what its body touches, so a gate that
// scanned them would let a group of nodes collapse across a loop reading the group's output.
TEST_CASE("PassUtil - buffer_io and span_interferes see what a loop body touches", "[ComputeGraph][Passes][PassUtil][ControlFlow]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto E = create_zero_tensor<double>("E", 4, 4);
    auto F = create_zero_tensor<double>("F", 4, 4);
    auto G = create_random_tensor<double>("G", 4, 4);

    bool const body_reads_group_output = GENERATE(true, false);
    CAPTURE(body_reads_group_output);

    cg::Graph graph("pass_util_loop_span");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }
    auto &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        if (body_reads_group_output) {
            cg::axpby(1.0, C, 0.0, &F); // observes the group's partial result
        } else {
            cg::axpby(1.0, G, 0.0, &F); // touches nothing the group does
        }
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &E, 1.0, A, B);
    }

    auto const einsums = positions_of(graph, cg::OpKind::Einsum);
    auto const loops   = positions_of(graph, cg::OpKind::Loop);
    REQUIRE(einsums.size() == 2);
    REQUIRE(loops.size() == 1);
    std::size_t const first = einsums.front();
    std::size_t const last  = einsums.back();
    std::size_t const loop  = loops.front();
    REQUIRE((first < loop && loop < last));

    cg::TensorId const c_id = graph.nodes()[first].outputs.front();

    cg::passes::BufferIo const loop_io = cg::passes::buffer_io(graph, graph.nodes()[loop]);
    CHECK(loop_io.reads.contains(graph.buffer_of(c_id)) == body_reads_group_output);
    CHECK_FALSE(loop_io.writes.empty()); // the body's write of F

    std::vector<bool> is_member(graph.nodes().size(), false);
    is_member[first] = true;
    is_member[last]  = true;
    std::unordered_set<cg::TensorId> writes;
    std::unordered_set<cg::TensorId> reads;
    for (std::size_t const pos : einsums) {
        writes.insert(graph.nodes()[pos].outputs.begin(), graph.nodes()[pos].outputs.end());
        reads.insert(graph.nodes()[pos].inputs.begin(), graph.nodes()[pos].inputs.end());
    }

    CHECK(cg::passes::span_interferes(graph, first, last, is_member, writes, reads, /*reject_control_flow=*/false) ==
          body_reads_group_output);
    CHECK(cg::passes::span_interferes(graph, first, last, is_member, writes, reads, /*reject_control_flow=*/true));
}
