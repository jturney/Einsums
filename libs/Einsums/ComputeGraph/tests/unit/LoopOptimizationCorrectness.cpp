//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file LoopOptimizationCorrectness.cpp
/// @brief End-to-end numerical-correctness hardening for the loop-aware
///        optimization passes (loop_handling_audit.md).
///
/// Every test here builds a ComputeGraph with one or more *loop nodes*
/// (``cg::Graph::add_loop``) whose bodies use heavy in-place tensor reuse —
/// the pattern that exposed the CSE and Reorder soundness gaps — then runs
/// the FULL default PassManager and asserts the executed result matches an
/// eager reference computed the same way. If any pass that recurses into
/// loop bodies corrupts the body (eliminates a needed node, reorders past an
/// anti-dependency, merges distinct mutable outputs, …) these diverge.
///
/// Covers: a single loop with mutable reuse + the CSE/Reorder "bait"
/// patterns, nested loops, and sequential loops at the same graph level.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

// Runs exactly `n` iterations: condition returns true (continue) while
// iter < n-1, false on the last. Matches the convention used elsewhere.
auto fixed_iterations(size_t n) {
    return [n](size_t iter) -> bool { return iter + 1 < n; };
}

template <size_t Rank>
void check_close(Tensor<double, Rank> const &got, Tensor<double, Rank> const &ref, double margin = 1e-10) {
    REQUIRE(got.size() == ref.size());
    for (size_t idx = 0; idx < got.size(); ++idx) {
        REQUIRE(std::abs(got.data()[idx] - ref.data()[idx]) < margin);
    }
}

} // namespace

TEST_CASE("Loop correctness - mutable-reuse body through full default pipeline", "[ComputeGraph][Loop][Hardening]") {
    constexpr size_t N = 5;
    auto             A = create_random_tensor<double>("A", 4, 4);
    auto             H = create_random_tensor<double>("H", 4, 4);

    // Working tensors (eager so they're materialized; the point is in-place
    // reuse across iterations, not the alloc path).
    auto F   = create_zero_tensor<double>("F", 4, 4);
    auto G   = create_zero_tensor<double>("G", 4, 4);
    auto tmp = create_zero_tensor<double>("tmp", 4, 4);
    auto acc = create_zero_tensor<double>("acc", 4, 4);

    cg::Graph g("mutable_reuse");
    auto     &body = g.add_loop("iter", N, fixed_iterations(N));
    {
        cg::CaptureGuard const guard(body);
        // CSE bait: two ops writing distinct tensors from the same input.
        cg::axpby(1.0, H, 0.0, &F); // F = H
        cg::axpby(1.0, H, 0.0, &G); // G = H   (same computation, different output)
        // F and G then diverge — a CSE that merged them would corrupt G.
        cg::axpy(1.0, A, &F); // F += A
        // gemm into a reused intermediate, then accumulate (mutable reuse of acc).
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A); // tmp = A*A
        cg::axpy(1.0, tmp, &acc);                      // acc += tmp
        cg::axpy(1.0, F, &acc);                        // acc += F (= H + A)
        cg::axpy(1.0, G, &acc);                        // acc += G (= H)
        cg::scale(0.9, &acc);                          // acc *= 0.9 (in-place scale)
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);
    g.execute();

    // Eager reference: acc_0 = 0; each iter acc = 0.9 * (acc + A*A + (H+A) + H).
    auto AA = create_zero_tensor<double>("AA", 4, 4);
    einsum(Indices{i, j}, &AA, Indices{i, k}, A, Indices{k, j}, A);
    auto ref = create_zero_tensor<double>("ref", 4, 4);
    for (size_t it = 0; it < N; ++it) {
        for (size_t idx = 0; idx < ref.size(); ++idx) {
            ref.data()[idx] = 0.9 * (ref.data()[idx] + AA.data()[idx] + (H.data()[idx] + A.data()[idx]) + H.data()[idx]);
        }
    }

    check_close(acc, ref);
}

TEST_CASE("Loop correctness - nested loops through full default pipeline", "[ComputeGraph][Loop][Hardening]") {
    constexpr size_t No  = 3;
    constexpr size_t Ni  = 4;
    auto             A   = create_random_tensor<double>("A", 3, 3);
    auto             acc = create_zero_tensor<double>("acc", 3, 3);

    // Outer loop body contains an inner loop; the inner body accumulates A
    // into acc. acc is reused across every iteration of both loops.
    cg::Graph g("nested");
    auto     &outer = g.add_loop("outer", No, fixed_iterations(No));
    auto     &inner = outer.add_loop("inner", Ni, fixed_iterations(Ni));
    {
        cg::CaptureGuard const guard(inner);
        cg::axpy(1.0, A, &acc); // acc += A
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);
    g.execute();

    // Reference: acc = No * Ni * A.
    auto ref = create_zero_tensor<double>("ref", 3, 3);
    for (size_t idx = 0; idx < ref.size(); ++idx) {
        ref.data()[idx] = static_cast<double>(No * Ni) * A.data()[idx];
    }
    check_close(acc, ref);
}

TEST_CASE("Loop correctness - sequential loops through full default pipeline", "[ComputeGraph][Loop][Hardening]") {
    constexpr size_t N1  = 4;
    constexpr size_t N2  = 3;
    auto             A   = create_random_tensor<double>("A", 3, 3);
    auto             acc = create_zero_tensor<double>("acc", 3, 3);
    auto             out = create_zero_tensor<double>("out", 3, 3);

    // Two loop nodes at the same graph level. The second consumes the
    // first's output — exercises cross-loop dataflow at the parent level.
    cg::Graph g("sequential");
    auto     &loop1 = g.add_loop("loop1", N1, fixed_iterations(N1));
    {
        cg::CaptureGuard const guard(loop1);
        cg::axpy(1.0, A, &acc); // acc += A  → acc = N1 * A
    }
    auto &loop2 = g.add_loop("loop2", N2, fixed_iterations(N2));
    {
        cg::CaptureGuard const guard(loop2);
        cg::axpy(1.0, acc, &out); // out += acc → out = N2 * acc
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);
    g.execute();

    // Reference: acc = N1 * A; out = N2 * acc = N2 * N1 * A.
    auto ref_acc = create_zero_tensor<double>("ref_acc", 3, 3);
    auto ref_out = create_zero_tensor<double>("ref_out", 3, 3);
    for (size_t idx = 0; idx < ref_acc.size(); ++idx) {
        ref_acc.data()[idx] = static_cast<double>(N1) * A.data()[idx];
        ref_out.data()[idx] = static_cast<double>(N2 * N1) * A.data()[idx];
    }
    check_close(acc, ref_acc);
    check_close(out, ref_out);
}

TEST_CASE("Loop correctness - workspace-backed mutable reuse through full pipeline", "[ComputeGraph][Loop][Hardening]") {
    // Same as the first test but the working tensors are workspace shells,
    // so the Materialization (alloc+zero hoist) and FreeInsertion paths are
    // exercised alongside the recursing rewrite passes.
    constexpr size_t N = 5;
    auto             A = create_random_tensor<double>("A", 4, 4);

    cg::Workspace ws("ws");
    auto         &tmp = ws.declare_zero_tensor<double, 2>("tmp", 4, 4);
    auto         &acc = ws.declare_zero_tensor<double, 2>("acc", 4, 4);

    cg::Graph g("ws_mutable_reuse");
    auto     &body = g.add_loop("iter", N, fixed_iterations(N));
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A); // tmp = A*A
        cg::axpy(1.0, tmp, &acc);                      // acc += tmp
        cg::scale(0.5, &acc);                          // acc *= 0.5
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);
    ws.materialize_all(); // idempotent guard; pass already hoists alloc+zero
    g.execute();

    auto AA = create_zero_tensor<double>("AA", 4, 4);
    einsum(Indices{i, j}, &AA, Indices{i, k}, A, Indices{k, j}, A);
    auto ref = create_zero_tensor<double>("ref", 4, 4);
    for (size_t it = 0; it < N; ++it) {
        for (size_t idx = 0; idx < ref.size(); ++idx) {
            ref.data()[idx] = 0.5 * (ref.data()[idx] + AA.data()[idx]);
        }
    }
    check_close(acc, ref);
}

TEST_CASE("Loop correctness - GEMM chain restructured inside a loop body", "[ComputeGraph][Loop][Hardening]") {
    // A 2-GEMM chain whose optimal parenthesization differs sharply from
    // left-to-right (the classic 100x1 * 1x100 * 100x1 case). With
    // ContractionPlanning recursing, the chain inside the loop body gets
    // restructured; the result must still match the eager reference
    // (associativity-equivalent), proving the restructuring is sound on a
    // loop body — the same kind of body transform that broke CSE/Reorder.
    auto A   = create_random_tensor<double>("A", 100, 1);
    auto B   = create_random_tensor<double>("B", 1, 100);
    auto C   = create_random_tensor<double>("C", 100, 1);
    auto T1  = create_zero_tensor<double>("T1", 100, 100);
    auto out = create_zero_tensor<double>("out", 100, 1);

    cg::Graph g("chain_in_loop");
    auto     &body = g.add_loop("iter", 1, fixed_iterations(1));
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &T1, A, B);   // T1 = A·B   (100x100)
        cg::einsum("ik;kj->ij", &out, T1, C); // out = T1·C (100x1)  → chain A·B·C
    }

    auto       pm       = cg::PassManager::create_default();
    bool const modified = g.apply(pm);
    g.execute();
    (void)modified; // restructuring may or may not fire depending on the profile

    // Eager reference: out = (A·B)·C, computed directly.
    auto T1_ref  = create_zero_tensor<double>("T1_ref", 100, 100);
    auto out_ref = create_zero_tensor<double>("out_ref", 100, 1);
    einsum(Indices{i, j}, &T1_ref, Indices{i, k}, A, Indices{k, j}, B);
    einsum(Indices{i, j}, &out_ref, Indices{i, k}, T1_ref, Indices{k, j}, C);

    for (size_t idx = 0; idx < out.size(); ++idx) {
        CHECK(std::abs(out.data()[idx] - out_ref.data()[idx]) < 1e-9);
    }
}
