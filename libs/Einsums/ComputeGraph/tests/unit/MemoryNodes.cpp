//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <cstddef>
#include <sstream>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("create_tensor - creates graph-owned tensor with Alloc node", "[ComputeGraph][Memory]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    cg::Graph graph("alloc_test");

    auto &C = graph.create_tensor<double, 2>("C", 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Should have 2 nodes: Alloc + Einsum
    REQUIRE(graph.num_nodes() == 2);

    // First node should be Alloc
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::Alloc);
    REQUIRE(graph.nodes()[0].label.find("alloc") != std::string::npos);

    graph.execute();

    // Verify the result
    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("free_tensor - inserts Free node", "[ComputeGraph][Memory]") {
    cg::Graph graph("free_test");

    auto &tmp = graph.create_tensor<double, 2>("tmp", 5, 5);

    // Get the tensor ID from the Alloc node
    auto *alloc_desc = std::get_if<cg::AllocDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(alloc_desc != nullptr);
    auto tmp_id = alloc_desc->tensor_id;

    // Use the tensor
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &tmp);
    }

    // Free it
    graph.free_tensor(tmp_id, "tmp", static_cast<long>(5) * 5 * sizeof(double));

    // Should have 3 nodes: Alloc, Scale, Free
    REQUIRE(graph.num_nodes() == 3);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::Alloc);
    REQUIRE(graph.nodes()[1].kind == cg::OpKind::Scale);
    REQUIRE(graph.nodes()[2].kind == cg::OpKind::Free);

    graph.execute();
}

TEST_CASE("alloc + use + free - full lifecycle", "[ComputeGraph][Memory]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto E = create_zero_tensor<double>("E", 4, 2);

    cg::Graph graph("lifecycle");

    // Allocate intermediate
    auto &T          = graph.create_tensor<double, 2>("T", 4, 5);
    auto *alloc_desc = std::get_if<cg::AllocDescriptor>(&graph.nodes()[0].op_data);
    auto  t_id       = alloc_desc->tensor_id;

    {
        cg::CaptureGuard const guard(graph);
        // T = A * B
        cg::einsum("ik;kj->ij", &T, A, B);
        // E = T * D
        cg::einsum("ik;kj->ij", &E, T, D);
    }

    // Free intermediate after use
    graph.free_tensor(t_id, "T", static_cast<long>(4) * 5 * sizeof(double));

    // Nodes: Alloc(T), Einsum(T=A*B), Einsum(E=T*D), Free(T)
    REQUIRE(graph.num_nodes() == 4);

    graph.execute();

    // Verify: E = A * B * D
    auto T_ref = create_zero_tensor<double>("Tref", 4, 5);
    auto E_ref = create_zero_tensor<double>("Eref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &T_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &E_ref, Indices{i, k}, T_ref, Indices{k, j}, D);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            REQUIRE(std::abs(E(ii, jj) - E_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("MemoryPlanning sees Alloc/Free nodes", "[ComputeGraph][Memory]") {
    cg::Graph graph("memplan_alloc");

    auto &T1    = graph.create_tensor<double, 2>("T1", 10, 10);
    auto *desc1 = std::get_if<cg::AllocDescriptor>(&graph.nodes()[0].op_data);
    auto  t1_id = desc1->tensor_id;

    {
        cg::CaptureGuard const guard(graph);
        cg::scale(1.0, &T1);
    }

    graph.free_tensor(t1_id, "T1", static_cast<long>(10) * 10 * sizeof(double));

    // Run memory planning
    auto [_m, _pass] = graph.apply<cg::passes::MemoryPlanning>();

    // Should report the tensor's memory
    REQUIRE(_pass.total_memory() > 0);

    std::ostringstream report;
    _pass.print_report(report);
    REQUIRE(report.str().find("Total tensor memory") != std::string::npos);
}

TEST_CASE("create_tensor in graph summary", "[ComputeGraph][Memory]") {
    cg::Graph graph("summary_test");

    auto &tmp = graph.create_tensor<double, 2>("my_temp", 3, 3);
    (void)tmp;

    std::ostringstream out;
    graph.print_summary(out);

    REQUIRE(out.str().find("alloc(my_temp)") != std::string::npos);
    REQUIRE(out.str().find("Alloc") != std::string::npos);
}
