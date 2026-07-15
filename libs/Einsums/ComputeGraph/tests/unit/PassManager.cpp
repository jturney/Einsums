//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PassManager.cpp
/// @brief Tests for the PassManager and `PassManager::create_default()`.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("create_default - includes expected passes", "[ComputeGraph][PassManager]") {
    auto pm = cg::PassManager::create_default();

    bool has_cf = false;
    bool has_sa = false;
    bool has_gb = false;
    bool has_sp = false;
    for (auto const &pass : pm.passes()) {
        if (pass->name() == "ConstantFolding")
            has_cf = true;
        if (pass->name() == "ScaleAbsorption")
            has_sa = true;
        if (pass->name() == "GEMMBatching")
            has_gb = true;
        if (pass->name() == "SymmetryPropagation")
            has_sp = true;
    }
    CHECK(has_cf);
    CHECK(has_sa);
    CHECK(has_gb);
    CHECK(has_sp);
}

TEST_CASE("create_default - safe on empty graph", "[ComputeGraph][PassManager]") {
    cg::Graph graph("default_empty");

    auto pm = cg::PassManager::create_default();
    REQUIRE_NOTHROW(graph.apply(pm));
}

TEST_CASE("create_default - safe on simple graph", "[ComputeGraph][PassManager]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("default_simple");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("create_default - safe on Pipeline with loop", "[ComputeGraph][PassManager]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 5, 5);
    for (int it = 0; it < 5; it++) {
        tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
        linear_algebra::scale(0.9, &C_ref);
    }

    cg::Pipeline pipeline("default_loop");
    {
        auto                  &loop = pipeline.add_loop("iter", 5, [](size_t it) { return it < 4; });
        cg::CaptureGuard const guard(loop);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::scale(0.9, &C);
    }

    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    pipeline.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("create_default - safe on mixed operations", "[ComputeGraph][PassManager]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    auto D_ref = create_zero_tensor<double>("Dref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);
    tensor_algebra::permute(0.0, Indices{j, i}, &D_ref, 1.0, Indices{i, j}, C_ref);

    cg::Graph graph("default_mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
        cg::permute("ji <- ij", 0.0, &D, 1.0, C);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
            CHECK(D(ii, jj) == Catch::Approx(D_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("create_default - safe on rank-3 BatchedGemm (row-major)", "[ComputeGraph][PassManager][HigherRank]") {
    // Row-major tensors + batch-prefix pattern → row_mode fast path. The
    // full default pipeline has to stay correct on a BatchedGemm node with
    // the opposite storage order from the col-major case below.
    auto A = create_random_tensor<double>(true, "A", 4, 3, 5);
    auto B = create_random_tensor<double>(true, "B", 4, 5, 6);
    auto C = create_zero_tensor<double>(true, "C", 4, 3, 6);

    auto C_ref = create_zero_tensor<double>(true, "Cref", 4, 3, 6);
    {
        using namespace einsums::index;
        tensor_algebra::einsum(Indices{b, i, j}, &C_ref, Indices{b, i, k}, A, Indices{b, k, j}, B);
    }

    cg::Graph graph("default_rank3_row");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bik;bkj->bij", &C, A, B);
    }

    REQUIRE(graph.nodes()[0].kind == cg::OpKind::BatchedGemm);

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t bb = 0; bb < 4; bb++) {
        for (size_t ii = 0; ii < 3; ii++) {
            for (size_t jj = 0; jj < 6; jj++) {
                CHECK(C(bb, ii, jj) == Catch::Approx(C_ref(bb, ii, jj)).margin(1e-10));
            }
        }
    }
}

TEST_CASE("create_default - safe on rank-3 BatchedGemm", "[ComputeGraph][PassManager][HigherRank]") {
    // Col-major batch-suffix pattern → BatchedGemm capture. Verifies the
    // full default pipeline (including GPUPlacement, distribution passes,
    // memory planning) stays correct with a batched node.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 3, 6, 4);
    {
        using namespace einsums::index;
        tensor_algebra::einsum(Indices{i, j, b}, &C_ref, Indices{i, k, b}, A, Indices{k, j, b}, B);
    }

    cg::Graph graph("default_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
    }

    REQUIRE(graph.nodes()[0].kind == cg::OpKind::BatchedGemm);

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            for (size_t bb = 0; bb < 4; bb++) {
                CHECK(C(ii, jj, bb) == Catch::Approx(C_ref(ii, jj, bb)).margin(1e-10));
            }
        }
    }
}

namespace {
/// A deliberately unsound pass: rotates the first node to the end and vouches
/// for the order. If that node wrote a tensor a later node reads, the reader
/// now precedes its writer - the exact bug-1012 shape GEMMBatching produced
/// by appending its BatchedGemm node.
struct RotateNodesPass final : cg::OptimizerPass {
    [[nodiscard]] std::string name() const override { return "RotateNodesPass"; }
    bool                      run(cg::Graph &graph) override {
        auto &nodes = graph.nodes();
        if (nodes.size() < 2) {
            return false;
        }
        std::rotate(nodes.begin(), nodes.begin() + 1, nodes.end());
        graph.mark_sorted();
        return true;
    }
};
} // namespace

TEST_CASE("PassManager - program-order guard rejects a writer moved after its reader", "[ComputeGraph][PassManager][Validation]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    cg::Graph graph("rotate_guard");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);          // writer of C
        cg::gemm<false, false>(1.0, C, B, 0.0, &D); // reader of C
    }

    cg::PassManager pm;
    pm.add<RotateNodesPass>(); // moves the C-writer after the C-reader
    REQUIRE_THROWS_AS(graph.apply(pm), std::logic_error);
}
