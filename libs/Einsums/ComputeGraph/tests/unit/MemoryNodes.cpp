//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
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

// ── A sub-graph's operand across a Free/Materialize cycle ───────────────────
//
// Capture into a sub-graph adopts a stand-in for an operand the sub-graph does
// not own (Graph::adopt_operand): the stand-in SHARES the owner's storage block
// but holds its own TensorImpl, and that impl caches the data pointer. Free and
// Materialize relocate the block - release() frees the owned buffer outright,
// and the next replay's materialize() allocates a new one - so from the second
// replay on the stand-in's cached pointer names memory the owner no longer has.
//
// Both cases below read a PARENT-level intermediate from inside a body. A is
// changed between the two replays so a stand-in left on the old buffer is wrong
// on its VALUE, not merely on its address: reading the dead buffer returns the
// first replay's data on an allocator that has not reused the bytes yet, which
// is how this passed on macOS while faulting on Windows.

namespace {

/// ((A*B)*B)*B, the value both graphs below compute through a body.
Tensor<double, 2> chained_reference(Tensor<double, 2> const &a, Tensor<double, 2> const &b, size_t n) {
    auto r = create_zero_tensor<double>("r", n, n);
    tensor_algebra::einsum(Indices{i, j}, &r, Indices{i, k}, a, Indices{k, j}, b);
    auto s = create_zero_tensor<double>("s", n, n);
    tensor_algebra::einsum(Indices{i, j}, &s, Indices{i, k}, r, Indices{k, j}, b);
    auto o = create_zero_tensor<double>("o", n, n);
    tensor_algebra::einsum(Indices{i, j}, &o, Indices{i, k}, s, Indices{k, j}, b);
    return o;
}

} // namespace

TEST_CASE("Free + rematerialize - a setup body reads the operand's new storage", "[ComputeGraph][Memory][Setup]") {
    constexpr size_t N   = 64;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("free_remat_setup");
    auto     &R = graph.create_tensor<double, 2>("R", N, N);
    auto     &S = graph.create_tensor<double, 2>("S", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &R, A, B); // R is written at THIS level
    }
    {
        auto                  &setup = graph.add_setup("fit");
        cg::CaptureGuard const guard(setup);
        cg::einsum("ik;kj->ij", &S, R, B); // and read only inside the body
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &out, S, B); // a caller-owned result to assert on
    }

    // The floor is lowered so R, which is well under the default megabyte,
    // still gets the Materialize/Free bracket this case is about.
    auto [freed, fi] = graph.apply<cg::passes::FreeInsertion>(size_t{1024});
    REQUIRE(freed);

    auto check = [&]() {
        auto const ref = chained_reference(A, B, N);
        for (size_t ii = 0; ii < N; ii += 7) {
            for (size_t jj = 0; jj < N; jj += 5) {
                REQUIRE_THAT(out(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-10));
            }
        }
    };

    graph.execute();
    check();

    // A different A makes the replay's R a different value, so a body still
    // reading the freed buffer computes the FIRST replay's S. Without that the
    // stale read is invisible wherever the allocator has not reused the bytes.
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            A(ii, jj) += 1.0;
        }
    }
    graph.invalidate_setup();
    out.zero();
    graph.execute();
    check();
}

TEST_CASE("Free + rematerialize - a loop body reads the operand's new storage", "[ComputeGraph][Memory][Loop]") {
    constexpr size_t N   = 64;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("free_remat_loop");
    auto     &R = graph.create_tensor<double, 2>("R", N, N);
    auto     &S = graph.create_tensor<double, 2>("S", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &R, A, B);
    }
    {
        auto                  &body = graph.add_loop("once", 1, std::function<bool(size_t)>{});
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &S, R, B);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &out, S, B);
    }

    auto [freed, fi] = graph.apply<cg::passes::FreeInsertion>(size_t{1024});
    REQUIRE(freed);

    auto check = [&]() {
        auto const ref = chained_reference(A, B, N);
        for (size_t ii = 0; ii < N; ii += 7) {
            for (size_t jj = 0; jj < N; jj += 5) {
                REQUIRE_THAT(out(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-10));
            }
        }
    };

    graph.execute();
    check();

    auto decoy0 = create_zero_tensor<double>("decoy0", N, N);
    auto decoy1 = create_zero_tensor<double>("decoy1", N, N);
    auto decoy2 = create_zero_tensor<double>("decoy2", N, N);
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            A(ii, jj) += 1.0;
        }
    }
    out.zero();
    graph.execute();
    check();
}
