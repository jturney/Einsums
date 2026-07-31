//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_CSE.cpp
/// @brief Unit tests for Common Subexpression Elimination.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("CSE - empty graph", "[ComputeGraph][CSE]") {
    cg::Graph graph("cse_empty");

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
}

TEST_CASE("CSE - single node graph", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cse_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 1);
}

TEST_CASE("CSE - eliminates duplicate einsum", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("cse_test");
    // The duplicate's output must be graph-owned: user-visible outputs are a
    // contract (the user reads them directly) and are never elided.
    auto &D = graph.create_tensor<double, 2>("D", 4, 5);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    REQUIRE(graph.num_nodes() == 3); // Alloc(D) + 2 einsums

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 2); // D's producer folded away

    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("CSE - never elides a write to a user-visible tensor", "[ComputeGraph][CSE][UserVisible]") {
    // Regression for the silent-contract-break: both C and D are USER
    // tensors capturing the same computation. Folding D's producer would
    // redirect graph consumers but leave the user's D unwritten. CSE must
    // keep both producers, and both tensors must hold the result.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);

    cg::Graph graph("cse_user_visible");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 2);

    graph.execute();
    // D must actually be written (not left at its zero initialization).
    double d_norm = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - D(ii, jj)) < 1e-12);
            d_norm += std::abs(D(ii, jj));
        }
    }
    REQUIRE(d_norm > 1e-8);
}

TEST_CASE("CSE - surviving consumer of an eliminated duplicate reads the survivor", "[ComputeGraph][CSE]") {
    // Regression for the CSE soundness bug where folding a duplicate producer
    // corrupted a *downstream* consumer of that duplicate. Executor lambdas
    // resolve operands through their captured TensorSlot, not Node::inputs, so
    // CSE's TensorId metadata redirect alone is invisible at run time: a node
    // reading the eliminated duplicate's output kept reading its (now
    // never-written) buffer and silently produced zeros. The earlier tests only
    // check the survivor's value or the node count, so they missed this.
    //
    // Diamond shape: C and D are identical products (D is eliminated); out reads
    // D. After CSE, D's producer is gone and out must resolve to C's buffer via
    // Graph::redirect_slot.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 5);
    auto F   = create_random_tensor<double>("F", 5, 2);
    auto C   = create_zero_tensor<double>("C", 4, 5);
    auto out = create_zero_tensor<double>("out", 4, 2);

    cg::Graph graph("cse_surviving_consumer");
    auto     &D = graph.create_tensor<double, 2>("D", 4, 5); // graph-owned duplicate
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);   // survivor
        cg::einsum("ik;kj->ij", &D, A, B);   // duplicate (eliminated)
        cg::einsum("ik;kj->ij", &out, D, F); // consumer of the eliminated duplicate
    }

    REQUIRE(graph.num_nodes() == 4); // Alloc(D) + 3 einsums

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 3); // D's producer folded away

    graph.execute();

    // Reference: out = (A·B)·F
    auto AB = create_zero_tensor<double>("AB", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, AB, Indices{k, j}, F);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            max_abs = std::max(max_abs, std::abs(out(ii, jj)));
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
    // Guard against the failure mode being masked by an all-zero reference.
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("CSE - redirect survives a rebind of the survivor", "[ComputeGraph][CSE][Rebind]") {
    // Regression for the residual hole in the bug above: redirect_slot used to
    // be a one-time pointer copy, so rebinding the SURVIVOR after CSE left the
    // eliminated duplicate's consumers pointing at the survivor's old buffer.
    // The redirect must be durable: after rebind(C, C2) the producer writes C2
    // and out (captured against D's slot) must follow it there.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 5);
    auto F   = create_random_tensor<double>("F", 5, 2);
    auto C   = create_zero_tensor<double>("C", 4, 5);
    auto out = create_zero_tensor<double>("out", 4, 2);

    cg::Graph graph("cse_rebind_survivor");
    auto     &D = graph.create_tensor<double, 2>("D", 4, 5); // graph-owned duplicate
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);   // survivor
        cg::einsum("ik;kj->ij", &D, A, B);   // duplicate (eliminated)
        cg::einsum("ik;kj->ij", &out, D, F); // consumer of the eliminated duplicate
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 3); // Alloc(D) + 2 einsums remain

    // Rebind the survivor to a fresh buffer. C's old buffer stays zero, so a
    // stale (snapshot) redirect would make out read zeros.
    auto C2 = create_zero_tensor<double>("C2", 4, 5);
    graph.rebind(C, C2);

    graph.execute();

    auto AB = create_zero_tensor<double>("AB", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, AB, Indices{k, j}, F);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            max_abs = std::max(max_abs, std::abs(out(ii, jj)));
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
    // The producer must have written the new buffer, and the old one must
    // still be zero (proves the whole graph moved, not just the consumer).
    REQUIRE(std::abs(C2(0, 0) - AB(0, 0)) < 1e-12);
    REQUIRE(C(0, 0) == 0.0);
}

TEST_CASE("CSE - three identical einsums reduces to one", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("cse_triple");
    auto     &D = graph.create_tensor<double, 2>("D", 4, 5);
    auto     &E = graph.create_tensor<double, 2>("E", 4, 5);
    (void)D;
    (void)E;
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
        cg::einsum("ik;kj->ij", &E, A, B);
    }

    REQUIRE(graph.num_nodes() == 5); // 2 Allocs + 3 einsums

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    CHECK(graph.num_nodes() == 3); // 2 Allocs + the surviving einsum
}

TEST_CASE("CSE - does not eliminate different operations", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("cse_no_match");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 2.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("CSE - does not eliminate different inputs", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("cse_diff_inputs");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, B, A); // swapped
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("CSE - keeps a duplicate whose output a loop body reads", "[ComputeGraph][CSE][ControlFlow]") {
    // A control-flow node's Node::inputs do not list what its body reads (that
    // is what Graph::effective_io reconstructs), so the reader scan saw nobody
    // reading D and merged its producer away. Graph::redirect_slot only repoints
    // the parent's slot table, so the body kept reading D's own -- now never
    // written -- buffer and summed zeros.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 5);
    auto C   = create_zero_tensor<double>("C", 4, 5);
    auto out = create_zero_tensor<double>("out", 4, 5);

    cg::Graph graph("cse_loop_body_reader");
    auto     &D = graph.create_zero_tensor<double, 2>("D", 4, 5); // graph-owned duplicate
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B); // survivor
        cg::einsum("ik;kj->ij", &D, A, B); // duplicate
    }
    auto &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        cg::axpby(1.0, D, 0.0, &out);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);

    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    double out_norm = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(out(ii, jj) - C_ref(ii, jj)) < 1e-12);
            out_norm += std::abs(out(ii, jj));
        }
    }
    REQUIRE(out_norm > 1e-8);
}

// ═══════════════════════════════════════════════════════════════════════════
// Inside loop bodies (CSE descends the tree itself)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CSE - merges a duplicate inside a loop body", "[ComputeGraph][CSE][ControlFlow]") {
    // Iterative workloads capture EVERYTHING into a loop body, so until CSE
    // descended the tree it did nothing at all on them. The scratch is created
    // on the parent (the usual pattern) and used only by the body, which is
    // what makes it eliminable: the body's own handle for it reports
    // is_intermediate == false, so the check has to be answered at the root.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 5);
    auto out = create_zero_tensor<double>("out", 4, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("cse_in_body");
    auto     &P    = graph.create_zero_tensor<double, 2>("P", 4, 5);
    auto     &Q    = graph.create_zero_tensor<double, 2>("Q", 4, 5);
    auto     &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &P, 1.0, A, B); // survivor
        cg::einsum("ik;kj->ij", 0.0, &Q, 1.0, A, B); // duplicate
        cg::einsum("ij;ij->ij", 0.0, &out, 1.0, P, Q);
    }

    size_t const body_before = body.num_nodes();
    auto [modified, pass]    = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    CHECK(body.num_nodes() == body_before - 1);

    graph.execute();
    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            max_abs = std::max(max_abs, std::abs(out(ii, jj)));
            REQUIRE(std::abs(out(ii, jj) - C_ref(ii, jj) * C_ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("CSE - keeps a body duplicate whose output the parent reads", "[ComputeGraph][CSE][ControlFlow]") {
    // Guard F. Graph::redirect_slot repoints only the slot table of the graph
    // it is called on, so a merge inside the body cannot fix up a reader in the
    // parent: that reader would keep reading the eliminated duplicate's buffer,
    // which nothing writes any more.
    auto A    = create_random_tensor<double>("A", 4, 3);
    auto B    = create_random_tensor<double>("B", 3, 5);
    auto seen = create_zero_tensor<double>("seen", 4, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("cse_body_escapes");
    auto     &P    = graph.create_zero_tensor<double, 2>("P", 4, 5);
    auto     &Q    = graph.create_zero_tensor<double, 2>("Q", 4, 5);
    auto     &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &P, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &Q, 1.0, A, B); // would be the duplicate
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, Q, 0.0, &seen); // parent reads the body's Q
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(seen(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("CSE - body axpby copies that later diverge are not merged", "[ComputeGraph][CSE][ControlFlow]") {
    // The SCF-body case the pass used to name as its reason for never running
    // on bodies: axpby(1,H,0,F) and axpby(1,H,0,sum_HF), where F and sum_HF
    // then diverge. Guard B (single writer) is what actually rules it out -
    // both destinations are written twice - so recursing is safe here.
    auto H = create_random_tensor<double>("H", 4, 4);
    auto G = create_random_tensor<double>("G", 4, 4);

    auto F_ref = Tensor<double, 2>("Fref", 4, 4);
    auto S_ref = Tensor<double, 2>("Sref", 4, 4);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            F_ref(ii, jj) = H(ii, jj) + G(ii, jj);       // F  = H then F += G
            S_ref(ii, jj) = H(ii, jj) + 2.0 * G(ii, jj); // sum = H then sum += 2G
        }
    }

    cg::Graph graph("cse_body_diverge");
    auto     &F    = graph.create_zero_tensor<double, 2>("F", 4, 4);
    auto     &S    = graph.create_zero_tensor<double, 2>("S", 4, 4);
    auto     &body = graph.add_loop("once", 1, [](size_t iter) { return iter < 1; });
    {
        cg::CaptureGuard const guard(body);
        cg::axpby(1.0, H, 0.0, &F);
        cg::axpby(1.0, H, 0.0, &S);
        cg::axpby(1.0, G, 1.0, &F); // F and S diverge from here
        cg::axpby(2.0, G, 1.0, &S);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(F(ii, jj) - F_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(S(ii, jj) - S_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("CSE - does not merge permutes with different index orders", "[ComputeGraph][CSE][Permute]") {
    // Regression: permute_desc_equal compared only alpha and beta, so two
    // permutes of the SAME source with the same scalars but DIFFERENT index
    // orders looked like the same computation. CSE merged them and the second
    // transpose was never computed - its consumers read the first one's buffer.
    // Both outputs are 3x3x3, so the failure is silent wrong values, not a
    // shape error.
    auto A = create_random_tensor<double>("A", 3, 3, 3);

    cg::Graph graph("cse_permute_orders");
    auto     &P1 = graph.create_zero_tensor<double, 3>("P1", 3, 3, 3);
    auto     &P2 = graph.create_zero_tensor<double, 3>("P2", 3, 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("j,i,k <- i,j,k", 0.0, &P1, 1.0, A);
        cg::permute("i,k,j <- i,j,k", 0.0, &P2, 1.0, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            for (size_t kk = 0; kk < 3; kk++) {
                REQUIRE(std::abs(P1(jj, ii, kk) - A(ii, jj, kk)) < 1e-12);
                REQUIRE(std::abs(P2(ii, kk, jj) - A(ii, jj, kk)) < 1e-12);
            }
        }
    }
}

TEST_CASE("CSE - does not merge permutes differing only in the imaginary prefactor", "[ComputeGraph][CSE][Permute][Complex]") {
    // Companion to the index-order case: PermuteDescriptor recorded alpha as a
    // `double` taken from alpha.real(), so 1+3i and 1-3i both stored as 1.0 and
    // these two permutes compared equal.
    using Complex = std::complex<double>;
    auto A        = create_random_tensor<Complex>("A", 3, 3);

    cg::Graph graph("cse_permute_complex_alpha");
    auto     &P1 = graph.create_zero_tensor<Complex, 2>("P1", 3, 3);
    auto     &P2 = graph.create_zero_tensor<Complex, 2>("P2", 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", Complex{0.0, 0.0}, &P1, Complex{1.0, 3.0}, A);
        cg::permute("ji <- ij", Complex{0.0, 0.0}, &P2, Complex{1.0, -3.0}, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(P1(jj, ii) - Complex{1.0, 3.0} * A(ii, jj)) < 1e-12);
            REQUIRE(std::abs(P2(jj, ii) - Complex{1.0, -3.0} * A(ii, jj)) < 1e-12);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proportional duplicates: same computation up to one real scalar
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CSE - merges a proportional duplicate and folds the factor into its readers", "[ComputeGraph][CSE][Prefactor]") {
    auto A    = create_random_tensor<double>("A", 4, 3);
    auto B    = create_random_tensor<double>("B", 3, 5);
    auto F    = create_random_tensor<double>("F", 5, 2);
    auto out1 = create_zero_tensor<double>("out1", 4, 2);
    auto out2 = create_zero_tensor<double>("out2", 4, 2);

    cg::Graph graph("cse_proportional");
    auto     &P = graph.create_zero_tensor<double, 2>("P", 4, 5); // survivor
    auto     &Q = graph.create_zero_tensor<double, 2>("Q", 4, 5); // 0.5 * P
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &P, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &Q, 0.5, A, B);
        cg::einsum("ik;kj->ij", 0.0, &out1, 1.0, P, F);
        cg::einsum("ik;kj->ij", 0.0, &out2, 1.0, Q, F);
    }

    size_t const n_before = graph.num_nodes();
    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == n_before - 1); // Q's producer is gone

    graph.execute();

    auto AB = create_zero_tensor<double>("AB", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    auto ref = create_zero_tensor<double>("ref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &ref, Indices{i, k}, AB, Indices{k, j}, F);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            max_abs = std::max(max_abs, std::abs(ref(ii, jj)));
            REQUIRE(std::abs(out1(ii, jj) - ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(out2(ii, jj) - 0.5 * ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("CSE - declines a ratio that is not an exact power of two", "[ComputeGraph][CSE][Prefactor]") {
    // 3x is exactly representable, but folding it would make the result depend
    // on which of the two proportional nodes the pass kept.
    auto A    = create_random_tensor<double>("A", 4, 3);
    auto B    = create_random_tensor<double>("B", 3, 5);
    auto F    = create_random_tensor<double>("F", 5, 2);
    auto out2 = create_zero_tensor<double>("out2", 4, 2);

    cg::Graph graph("cse_odd_ratio");
    auto     &P = graph.create_zero_tensor<double, 2>("P", 4, 5);
    auto     &Q = graph.create_zero_tensor<double, 2>("Q", 4, 5);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &P, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &Q, 3.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &out2, 1.0, Q, F);
    }

    size_t const n_before = graph.num_nodes();
    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == n_before);
}

TEST_CASE("CSE - declines a proportional duplicate whose reader cannot take the factor", "[ComputeGraph][CSE][Prefactor]") {
    // A permute bakes its prefactor into the executor closure, so there is
    // nowhere to put the factor; the duplicate has to stay.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto QT = create_zero_tensor<double>("QT", 5, 4);

    cg::Graph graph("cse_unfoldable_reader");
    auto     &P = graph.create_zero_tensor<double, 2>("P", 4, 5);
    auto     &Q = graph.create_zero_tensor<double, 2>("Q", 4, 5);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &P, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &Q, 0.5, A, B);
        cg::permute("ji <- ij", 0.0, &QT, 1.0, Q);
    }

    size_t const n_before = graph.num_nodes();
    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == n_before);

    graph.execute();

    auto AB = create_zero_tensor<double>("AB", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(QT(jj, ii) - 0.5 * AB(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("CSE - merges proportional axpby copies", "[ComputeGraph][CSE][Prefactor][Axpby]") {
    // Axpby had no arm in the descriptor comparison at all, so even identical
    // pure-overwrite copies never merged. `Y = alpha*X` is linear in alpha, so
    // the proportional case works the same way as einsum's.
    auto X    = create_random_tensor<double>("X", 4, 5);
    auto G    = create_random_tensor<double>("G", 5, 2);
    auto out1 = create_zero_tensor<double>("out1", 4, 2);
    auto out2 = create_zero_tensor<double>("out2", 4, 2);

    cg::Graph graph("cse_axpby_proportional");
    auto     &Y1 = graph.create_zero_tensor<double, 2>("Y1", 4, 5);
    auto     &Y2 = graph.create_zero_tensor<double, 2>("Y2", 4, 5);
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(2.0, X, 0.0, &Y1);
        cg::axpby(1.0, X, 0.0, &Y2); // 0.5 * Y1
        cg::einsum("ik;kj->ij", 0.0, &out1, 1.0, Y1, G);
        cg::einsum("ik;kj->ij", 0.0, &out2, 1.0, Y2, G);
    }

    size_t const n_before = graph.num_nodes();
    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == n_before - 1);

    graph.execute();

    auto ref = create_zero_tensor<double>("ref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &ref, Indices{i, k}, X, Indices{k, j}, G);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            max_abs = std::max(max_abs, std::abs(ref(ii, jj)));
            REQUIRE(std::abs(out1(ii, jj) - 2.0 * ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(out2(ii, jj) - ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("CSE - does not merge scale with different factors", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("cse_diff_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 2);
}

TEST_CASE("CSE + DeadNodeElimination composition", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("cse_dne");
    auto     &D = graph.create_zero_tensor<double, 2>("D", 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    size_t const n_before = graph.num_nodes();
    REQUIRE(n_before >= 2);

    graph.apply<cg::passes::CSE>();
    CHECK(graph.num_nodes() < n_before);

    // DNE may or may not find further dead nodes depending on Alloc handling;
    // just verify it runs without crashing.
    auto [modified, dne] = graph.apply<cg::passes::DeadNodeElimination>();
    (void)modified;
}

TEST_CASE("CSE - deduplicates rank-3 BatchedGemm nodes (col-major)", "[ComputeGraph][CSE][HigherRank]") {
    // Two identical rank-3 strided-batched contractions → each captures as
    // OpKind::BatchedGemm. Exercises the batched_gemm_desc_equal path in
    // CSE::op_data_equal (added to handle BatchedGemm comparison).
    // Col-major default + batch-suffix pattern triggers the fast path's col_mode.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);

    cg::Graph graph("cse_rank3_col");
    auto     &D = graph.create_tensor<double, 3>("D", 3, 6, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, A, B);
    }

    size_t batched_before = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_before;
    REQUIRE(batched_before == 2);

    size_t const nodes_before = graph.num_nodes();

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    CHECK(graph.num_nodes() < nodes_before);
    size_t batched_after = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_after;
    CHECK(batched_after == 1);
}

TEST_CASE("CSE - deduplicates rank-3 BatchedGemm nodes (row-major)", "[ComputeGraph][CSE][HigherRank]") {
    // Same contraction, row-major tensors + batch-prefix pattern → triggers
    // the fast path's row_mode branch. Verifies CSE works across both layout
    // modes of the strided-batched capture.
    auto A = create_random_tensor<double>(/*row_major=*/true, "A", 4, 3, 5);
    auto B = create_random_tensor<double>(/*row_major=*/true, "B", 4, 5, 6);
    auto C = create_zero_tensor<double>(/*row_major=*/true, "C", 4, 3, 6);
    auto D = create_zero_tensor<double>(/*row_major=*/true, "D", 4, 3, 6);

    cg::Graph graph("cse_rank3_row");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bik;bkj->bij", &C, A, B);
        cg::einsum("bik;bkj->bij", &D, A, B);
    }

    // Mark D's handle intermediate so the fold is allowed (create_tensor has
    // no row-major variant; this test exercises the row_mode dedup machinery,
    // not the user-visibility guard - that has its own test above).
    for (auto &[tid, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == &D) {
            handle.is_intermediate = true;
        }
    }

    size_t batched_before = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_before;
    REQUIRE(batched_before == 2);

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    size_t batched_after = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_after;
    CHECK(batched_after == 1);
}
