//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace basics
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Workspace - construction", "[ComputeGraph][Workspace]") {
    cg::Workspace const ws("test_ws");
    CHECK(ws.name() == "test_ws");
    CHECK(ws.size() == 0);
}

TEST_CASE("Workspace - declare_tensor creates shell", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_tensor<double, 2>("A", 10, 8);

    CHECK(ws.size() == 1);
    CHECK(A.name() == "A");
    CHECK(A.dim(0) == 10);
    CHECK(A.dim(1) == 8);
    CHECK_FALSE(A.is_materialized());
    CHECK(A.Rank == 2);
}

TEST_CASE("Workspace - declare_zero_tensor sets init kind", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_zero_tensor<double, 2>("A", 5, 5);

    CHECK_FALSE(A.is_materialized());
    CHECK(ws.tensor_handles().back().init_kind == cg::InitKind::Zero);
    CHECK(ws.tensor_handles().back().alloc_state == cg::AllocState::Deferred);
}

TEST_CASE("Workspace - declare_random_tensor sets init kind", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_random_tensor<double, 2>("A", 5, 5);

    CHECK_FALSE(A.is_materialized());
    CHECK(ws.tensor_handles().back().init_kind == cg::InitKind::Random);
}

TEST_CASE("Workspace - shell tensor materializes", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_tensor<double, 2>("A", 4, 3);

    CHECK_FALSE(A.is_materialized());

    // Manually materialize (normally done by MaterializationPass)
    A.materialize();

    CHECK(A.is_materialized());
    CHECK(A.data() != nullptr);
    CHECK(A.dim(0) == 4);
    CHECK(A.dim(1) == 3);

    // Can write to it
    A(0, 0) = 42.0;
    CHECK(A(0, 0) == 42.0);
}

TEST_CASE("Workspace - materialize_fn works", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_tensor<double, 2>("A", 3, 3);

    // materialize_fn is set by declare_tensor
    auto const &handle = ws.tensor_handles().back();
    CHECK(handle.materialize_fn);

    handle.materialize_fn();

    CHECK(A.is_materialized());
    CHECK(A.data() != nullptr);
}

TEST_CASE("Workspace - multiple tensors", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("multi");

    auto &A = ws.declare_tensor<double, 2>("A", 10, 10);
    auto &B = ws.declare_zero_tensor<float, 3>("B", 5, 5, 5);
    auto &C = ws.declare_random_tensor<double, 1>("C", 100);

    CHECK(ws.size() == 3);

    // All are shell tensors
    CHECK_FALSE(A.is_materialized());
    CHECK_FALSE(B.is_materialized());
    CHECK_FALSE(C.is_materialized());

    // Handles have correct metadata
    CHECK(ws.tensor_handles()[0].name == "A");
    CHECK(ws.tensor_handles()[0].rank == 2);
    CHECK(ws.tensor_handles()[0].dims == std::vector<size_t>{10, 10});

    CHECK(ws.tensor_handles()[1].name == "B");
    CHECK(ws.tensor_handles()[1].rank == 3);
    CHECK(ws.tensor_handles()[1].init_kind == cg::InitKind::Zero);

    CHECK(ws.tensor_handles()[2].name == "C");
    CHECK(ws.tensor_handles()[2].rank == 1);
    CHECK(ws.tensor_handles()[2].init_kind == cg::InitKind::Random);
}

TEST_CASE("Workspace - tensor handle has correct alloc state", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    ws.declare_tensor<double, 2>("A", 4, 4);

    auto const &h = ws.tensor_handles().back();
    CHECK(h.alloc_state == cg::AllocState::Deferred);
    CHECK(h.tensor_ptr != nullptr); // Tensor object exists
    CHECK(h.data_ptr == nullptr);   // No data yet
}

TEST_CASE("Workspace - tensor dims preserved after materialize", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("test");

    auto &A = ws.declare_tensor<double, 4>("T", 2, 3, 4, 5);

    CHECK(A.dim(0) == 2);
    CHECK(A.dim(1) == 3);
    CHECK(A.dim(2) == 4);
    CHECK(A.dim(3) == 5);

    A.materialize();

    // Dims unchanged after materialization
    CHECK(A.dim(0) == 2);
    CHECK(A.dim(1) == 3);
    CHECK(A.dim(2) == 4);
    CHECK(A.dim(3) == 5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph::declare_tensor
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Graph declare_tensor - creates deferred intermediate", "[ComputeGraph][DeclaredTensor]") {
    cg::Graph graph("test_graph");
    auto     &T = graph.declare_tensor<double, 2>(std::string("T"), 5, 4);

    CHECK(T.dim(0) == 5);
    CHECK(T.dim(1) == 4);
    CHECK_FALSE(T.is_materialized());

    // Materialize and verify
    T.materialize();
    CHECK(T.is_materialized());
    T(0, 0) = 42.0;
    CHECK(T(0, 0) == 42.0);
}

TEST_CASE("Graph declare_zero_tensor - sets init kind", "[ComputeGraph][DeclaredTensor]") {
    cg::Graph graph("test_graph");
    auto     &T = graph.declare_zero_tensor<double, 2>(std::string("T"), 3, 3);

    CHECK_FALSE(T.is_materialized());

    // Check handle has correct init_kind
    bool found = false;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == &T) {
            CHECK(handle.init_kind == cg::InitKind::Zero);
            CHECK(handle.alloc_state == cg::AllocState::Deferred);
            CHECK_FALSE(handle.is_intermediate); // declare_tensor is user-visible
            found = true;
            break;
        }
    }
    CHECK(found);
}

TEST_CASE("Graph declare_tensor - usable in capture and execute", "[ComputeGraph][DeclaredTensor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    // Reference
    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("declared_graph");
    // Use declare_tensor for the intermediate — deferred allocation
    auto &C = graph.declare_tensor<double, 2>(std::string("C"), 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Materialize before execution
    C.materialize();
    C.zero();

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline::declare_tensor
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Pipeline declare_tensor - creates deferred tensor", "[ComputeGraph][DeclaredTensor]") {
    cg::Pipeline pipeline("test_pipeline");
    auto        &F = pipeline.declare_zero_tensor<double, 2>(std::string("F"), 6, 6);

    CHECK(F.dim(0) == 6);
    CHECK(F.dim(1) == 6);
    CHECK_FALSE(F.is_materialized());

    CHECK(pipeline.declared_handles().size() == 1);
    CHECK(pipeline.declared_handles()[0].init_kind == cg::InitKind::Zero);
}

TEST_CASE("Pipeline declare_tensor - usable in stages", "[ComputeGraph][DeclaredTensor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    // Reference
    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    cg::Pipeline pipeline("declared_pipeline");
    auto        &C = pipeline.declare_tensor<double, 2>(std::string("C"), 4, 5);

    // Stage 1: compute
    {
        auto                  &stage = pipeline.add_stage("compute");
        cg::CaptureGuard const guard(stage);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Stage 2: scale
    {
        auto                  &stage = pipeline.add_stage("scale");
        cg::CaptureGuard const guard(stage);
        cg::scale(2.0, &C);
    }

    // Materialize before execution
    C.materialize();
    C.zero();

    pipeline.execute();

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace + Pipeline integration
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Shell tensor - materialize inside a lambda", "[ComputeGraph][DeclaredTensor]") {
    // Verify materialize works when called from a lambda (like a node executor)
    using TensorType = Tensor<double, 2>;
    auto *ptr        = new TensorType(TensorType::DeferredAlloc{}, "test", 4, 5);

    CHECK_FALSE(ptr->is_materialized());
    CHECK(ptr->dim(0) == 4);
    CHECK(ptr->dim(1) == 5);

    // Simulate what the Materialize node does
    auto mat_fn = [ptr]() { ptr->materialize(); };
    mat_fn();

    CHECK(ptr->is_materialized());
    CHECK(ptr->data() != nullptr);
    (*ptr)(0, 0) = 42.0;
    CHECK((*ptr)(0, 0) == 42.0);

    delete ptr;
}

TEST_CASE("Graph declare + Materialization only + execute", "[ComputeGraph][DeclaredTensor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("deferred_mat_only");
    auto     &C = graph.declare_tensor<double, 2>(std::string("C"), 4, 5);
    CHECK_FALSE(C.is_materialized());

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    CHECK(graph.num_nodes() == 1); // Just the einsum

    // Apply ONLY the Materialization pass
    auto [modified, pass] = graph.apply<cg::passes::Materialization>();
    CHECK(modified);
    CHECK(pass.num_materialized() == 1);
    CHECK(graph.num_nodes() == 2); // Materialize + einsum

    // C should still NOT be materialized (that happens during execute)
    CHECK_FALSE(C.is_materialized());

    // Execute: Materialize runs first (position 0), then einsum
    graph.execute();

    CHECK(C.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("Graph declare + create_default + execute - full deferred path", "[ComputeGraph][DeclaredTensor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("deferred_full");
    auto     &C = graph.declare_tensor<double, 2>(std::string("C"), 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    CHECK(C.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("Pipeline declare + create_default + execute - full deferred path", "[ComputeGraph][DeclaredTensor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    cg::Pipeline pipeline("deferred_pipeline");
    auto        &C = pipeline.declare_zero_tensor<double, 2>(std::string("C"), 4, 5);

    {
        auto                  &stage = pipeline.add_stage("compute");
        cg::CaptureGuard const guard(stage);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    {
        auto                  &stage = pipeline.add_stage("scale");
        cg::CaptureGuard const guard(stage);
        cg::scale(2.0, &C);
    }

    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    pipeline.execute();

    CHECK(C.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("Workspace - materialize_all", "[ComputeGraph][Workspace]") {
    cg::Workspace ws("mat_all");

    auto &A = ws.declare_tensor<double, 2>("A", 4, 4);
    auto &B = ws.declare_zero_tensor<double, 2>("B", 3, 3);
    auto &C = ws.declare_random_tensor<double, 2>("C", 5, 5);

    CHECK_FALSE(A.is_materialized());
    CHECK_FALSE(B.is_materialized());
    CHECK_FALSE(C.is_materialized());

    ws.materialize_all();

    // All should now be materialized
    CHECK(A.is_materialized());
    CHECK(B.is_materialized());
    CHECK(C.is_materialized());

    CHECK(A.data() != nullptr);
    CHECK(B.data() != nullptr);
    CHECK(C.data() != nullptr);

    // Zero-initialized tensor should be zero
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            CHECK(B(ii, jj) == 0.0);

    // Random-initialized tensor should have non-zero data (statistically)
    bool has_nonzero = false;
    for (size_t ii = 0; ii < 5 && !has_nonzero; ii++)
        for (size_t jj = 0; jj < 5 && !has_nonzero; jj++)
            if (C(ii, jj) != 0.0)
                has_nonzero = true;
    CHECK(has_nonzero);

    // Handles should be marked Materialized
    for (auto const &h : ws.tensor_handles())
        CHECK(h.alloc_state == cg::AllocState::Materialized);

    // Calling materialize_all again should be a no-op (already materialized)
    ws.materialize_all();
    CHECK(A.is_materialized());
}

TEST_CASE("Workspace + Pipeline - materialize_all before execute", "[ComputeGraph][DeclaredTensor]") {
    auto B = create_random_tensor<double>("B", 4, 3);

    // Reference
    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, B, Indices{j, k}, B);

    cg::Workspace ws("ws");
    auto         &A = ws.declare_zero_tensor<double, 2>("A", 4, 4);

    cg::Pipeline pipeline("pipe");
    pipeline.set_workspace(ws);

    {
        auto                  &stage = pipeline.add_stage("compute");
        cg::CaptureGuard const guard(stage);
        cg::einsum("ik;jk->ij", 0.0, &A, 1.0, B, B);
    }

    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    ws.materialize_all();
    pipeline.execute();

    CHECK(A.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            CHECK(A(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("Workspace + Pipeline - tensor survives across pipelines", "[ComputeGraph][DeclaredTensor]") {
    cg::Workspace ws("calc");

    auto &shared = ws.declare_tensor<double, 2>(std::string("shared"), 3, 3);

    // Pipeline 1: write to shared
    cg::Pipeline p1("writer");
    p1.set_workspace(ws);
    {
        auto                  &stage = p1.add_stage("write");
        cg::CaptureGuard const guard(stage);
        cg::scale(1.0, &shared); // Just touch it to verify capture works
    }

    // Materialize and fill
    shared.materialize();
    shared(0, 0) = 99.0;

    p1.execute();

    // Pipeline 2: read shared (it's still alive because Workspace owns it)
    CHECK(shared(0, 0) == 99.0);

    cg::Pipeline p2("reader");
    p2.set_workspace(ws);
    // shared is still valid — workspace owns it
    CHECK(shared.is_materialized());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lambda façade: cg::make_pipeline / cg::run / Pipeline::run
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Pipeline::run() applies defaults + materializes + executes", "[ComputeGraph][Facade]") {
    // Reference computation via the classic four-step ritual.
    auto B     = create_random_tensor<double>("B_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, B, Indices{j, k}, B);

    cg::Workspace ws("ws");
    auto         &A = ws.declare_zero_tensor<double, 2>("A", 4, 4);

    cg::Pipeline pipeline("pipe");
    pipeline.set_workspace(ws);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    pipeline.add_stage("compute", [&] { cg::einsum("ik;jk->ij", 0.0, &A, 1.0, B, B); });

    pipeline.run();

    CHECK(A.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            CHECK(A(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("Pipeline::run() throws without workspace", "[ComputeGraph][Facade]") {
    cg::Pipeline pipeline("orphan");
    pipeline.add_stage("noop", [&] { /* empty body */ });
    CHECK_THROWS_AS(pipeline.run(), std::runtime_error);
}

TEST_CASE("cg::make_pipeline builds then returns by value", "[ComputeGraph][Facade]") {
    auto B     = create_random_tensor<double>("B_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, B, Indices{j, k}, B);

    cg::Workspace ws("ws");
    auto         &A = ws.declare_zero_tensor<double, 2>("A", 4, 4);

    auto pipeline = cg::make_pipeline(
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        "pipe", ws, [&](cg::Pipeline &p) { p.add_stage("compute", [&] { cg::einsum("ik;jk->ij", 0.0, &A, 1.0, B, B); }); });
    CHECK(pipeline.num_stages() == 1);
    CHECK(pipeline.stage_name(0) == "compute");

    pipeline.run();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            CHECK(A(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}

TEST_CASE("cg::run does build + run in one expression", "[ComputeGraph][Facade]") {
    auto B     = create_random_tensor<double>("B_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, B, Indices{j, k}, B);

    cg::Workspace ws("ws");
    auto         &A = ws.declare_zero_tensor<double, 2>("A", 4, 4);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::run("pipe", ws, [&](cg::Pipeline &p) { p.add_stage("compute", [&] { cg::einsum("ik;jk->ij", 0.0, &A, 1.0, B, B); }); });

    CHECK(A.is_materialized());
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            CHECK(A(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
}
