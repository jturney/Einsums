//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/PooledTensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

void fill(RuntimeTensor<double> &t, double seed) {
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = std::sin(seed + static_cast<double>(i) * 0.125);
    }
}

} // namespace

TEST_CASE("Pooled operands keep independent alias roots", "[ComputeGraph][pool]") {
    MemoryPool pool(64 * kMiB, "graph");

    auto A = pool_empty<double>(pool, "A", {32, 24});
    auto B = pool_empty<double>(pool, "B", {24, 16});
    auto C = pool_zeros<double>(pool, "C", {32, 16});
    fill(A, 0.5);
    fill(B, 1.5);

    cg::Graph graph("pooled");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Distinct carves never overlap, so no handle may claim another as its
    // storage parent: every pooled operand stays its own hazard root, and the
    // dataflow executor sees no false dependency between them.
    size_t roots = 0;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        REQUIRE(handle.aliases == 0);
        roots++;
    }
    REQUIRE(roots >= 3);
}

TEST_CASE("Pooled operands execute like owned ones", "[ComputeGraph][pool]") {
    MemoryPool pool(64 * kMiB, "diff");

    auto Ap = pool_empty<double>(pool, "A", {32, 24});
    auto Bp = pool_empty<double>(pool, "B", {24, 16});
    auto Cp = pool_zeros<double>(pool, "C", {32, 16});
    fill(Ap, 0.5);
    fill(Bp, 1.5);

    RuntimeTensor<double> Ao("A", std::vector<size_t>{32, 24});
    RuntimeTensor<double> Bo("B", std::vector<size_t>{24, 16});
    RuntimeTensor<double> Co("C", std::vector<size_t>{32, 16});
    fill(Ao, 0.5);
    fill(Bo, 1.5);
    Co.zero();

    cg::Graph pooled("pooled");
    {
        cg::CaptureGuard guard(pooled);
        cg::einsum("ik;kj->ij", &Cp, Ap, Bp);
    }
    pooled.execute();

    cg::Graph owned("owned");
    {
        cg::CaptureGuard guard(owned);
        cg::einsum("ik;kj->ij", &Co, Ao, Bo);
    }
    owned.execute();

    REQUIRE(Cp.size() == Co.size());
    for (size_t i = 0; i < Co.size(); i++) {
        REQUIRE_THAT(Cp.data()[i], Catch::Matchers::WithinAbs(Co.data()[i], 1e-14));
    }

    // Replay is still replay: the operands did not move under the graph.
    pooled.execute();
    for (size_t i = 0; i < Co.size(); i++) {
        REQUIRE_THAT(Cp.data()[i], Catch::Matchers::WithinAbs(Co.data()[i], 1e-14));
    }
}

TEST_CASE("Pooled operands survive an optimized replay", "[ComputeGraph][pool]") {
    MemoryPool pool(64 * kMiB, "passes");

    auto A = pool_empty<double>(pool, "A", {16, 16});
    auto B = pool_empty<double>(pool, "B", {16, 16});
    auto C = pool_zeros<double>(pool, "C", {16, 16});
    fill(A, 0.25);
    fill(B, 0.75);

    cg::Graph graph("optimized");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    RuntimeTensor<double> expected("E", std::vector<size_t>{16, 16});
    expected.zero();
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < 16; k++) {
                acc += A(i, k) * B(k, j);
            }
            expected(i, j) = acc;
        }
    }

    for (size_t i = 0; i < C.size(); i++) {
        REQUIRE_THAT(C.data()[i], Catch::Matchers::WithinAbs(expected.data()[i], 1e-12));
    }

    // Nothing the pool handed out was released behind the tensors' backs.
    REQUIRE(pool.live_borrows() == 3);
}
