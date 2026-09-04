//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase B view_runtime tests, runtime-rank counterpart to cg::view.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("cg::view_runtime — full-axis slice produces a same-shape view", "[ComputeGraph][RuntimeTensor][View]") {
    RuntimeTensor<double> A("A", {4UL, 3UL});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(10 * i + j);

    cg::Pipeline pipe("view_rt_full");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view_runtime(A, std::vector<cg::ViewAxis>{cg::ViewAxis::full(), cg::ViewAxis::full()});
        // Initial slice metadata is set up at capture time using the parent's full shape.
        REQUIRE(slice.rank() == 2);
        REQUIRE(slice.dim(0) == 4);
        REQUIRE(slice.dim(1) == 3);
    }
    pipe.execute();
}

TEST_CASE("cg::view_runtime — constant range slice points into parent", "[ComputeGraph][RuntimeTensor][View]") {
    RuntimeTensor<double> A("A", {6UL, 3UL});
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(10 * i + j);

    cg::Pipeline               pipe("view_rt_range");
    RuntimeTensorView<double> *slice_ptr = nullptr;
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view_runtime(A, std::vector<cg::ViewAxis>{cg::ViewAxis::range(2, 5), cg::ViewAxis::full()});
        slice_ptr                    = &slice;
    }
    pipe.execute();

    // After execute, the slice executor has re-emplaced the view with the
    // resolved offset. Verify the slice covers rows 2..5 of the parent.
    REQUIRE(slice_ptr->rank() == 2);
    REQUIRE(slice_ptr->dim(0) == 3);
    REQUIRE(slice_ptr->dim(1) == 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            REQUIRE((*slice_ptr)(i, j) == Catch::Approx(static_cast<double>(10 * (i + 2) + j)));
}

TEST_CASE("cg::view_runtime — wrong axis count throws", "[ComputeGraph][RuntimeTensor][View]") {
    RuntimeTensor<double>  A("A", {3UL, 3UL});
    cg::Pipeline           pipe("view_rt_bad");
    auto                  &stage = pipe.add_stage("s");
    cg::CaptureGuard const g(stage);

    // Parent rank is 2; passing 3 axes should throw.
    REQUIRE_THROWS(cg::view_runtime(A, std::vector<cg::ViewAxis>{cg::ViewAxis::full(), cg::ViewAxis::full(), cg::ViewAxis::full()}));
}

TEST_CASE("cg::view_runtime — Drop axis reduces rank", "[ComputeGraph][RuntimeTensor][View]") {
    RuntimeTensor<double> A("A", {3UL, 3UL});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(10 * i + j);

    cg::Pipeline               pipe("view_rt_drop");
    RuntimeTensorView<double> *slice_ptr = nullptr;
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        // Drop axis 0 at index 1 (row 1), keep axis 1: result is the rank-1 row A(1, :).
        auto &slice = cg::view_runtime(A, std::vector<cg::ViewAxis>{cg::ViewAxis::drop(1), cg::ViewAxis::full()});
        slice_ptr   = &slice;
        REQUIRE(slice.rank() == 1);
        REQUIRE(slice.dim(0) == 3);
    }
    pipe.execute();

    // After execute the drop offset is resolved; the view aliases row 1 of the parent.
    REQUIRE(slice_ptr->rank() == 1);
    REQUIRE(slice_ptr->dim(0) == 3);
    for (size_t j = 0; j < 3; ++j)
        REQUIRE((*slice_ptr)(j) == Catch::Approx(10.0 + static_cast<double>(j)));
}

TEST_CASE("cg::view_runtime - views of two deferred parents keep the parents apart", "[ComputeGraph][RuntimeTensor][View][aliasing]") {
    // A deferred parent has no address, and its views must not register one. They used to carry
    // the shell's sentinel plus the slice offset, so two deferred tensors of one shape, sliced at
    // the same non-zero offset, put their views on identical byte spans and the pointer-derived
    // alias linking merged the two parents into one root. Every use of the later tensor was then
    // credited to the earlier, and a pass asking whether anything used it was told no.
    cg::Graph graph("deferred-parents");
    auto     &T  = graph.declare_zero_runtime_tensor<double>("T", {2, 2, 3, 3}, /*intermediate=*/true);
    auto     &r2 = graph.declare_zero_runtime_tensor<double>("r2", {2, 2, 3, 3}, /*intermediate=*/true);

    RuntimeTensor<double> src("src", {3UL, 3UL});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            src(i, j) = static_cast<double>(10 * i + j);
    {
        cg::CaptureGuard const guard(graph);
        auto const             slice =
            std::vector<cg::ViewAxis>{cg::ViewAxis::drop(1), cg::ViewAxis::drop(1), cg::ViewAxis::full(), cg::ViewAxis::full()};
        auto &Tv = cg::view_runtime(T, slice);
        auto &rv = cg::view_runtime(r2, slice);
        cg::axpby(1.0, src, 0.0, &Tv);
        cg::axpby(2.0, src, 0.0, &rv);
    }
    graph.link_alias_storage();

    cg::TensorId const t_id = graph.find_tensor_id_by_ptr(&T);
    cg::TensorId const r_id = graph.find_tensor_id_by_ptr(&r2);
    REQUIRE(t_id != 0);
    REQUIRE(r_id != 0);
    CHECK(graph.resolve_alias(t_id) == t_id);
    CHECK(graph.resolve_alias(r_id) == r_id);
    size_t views = 0;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (!handle.name.starts_with("view_rt")) {
            continue;
        }
        views++;
        CHECK(handle.data_ptr == nullptr);
        CHECK((graph.resolve_alias(id) == t_id || graph.resolve_alias(id) == r_id));
    }
    CHECK(views == 2);

    // Both parents are used, through their views, so both get storage and the values land apart.
    cg::passes::Materialization mat;
    REQUIRE(mat.run(graph));
    CHECK(mat.num_materialized() == 2);
    CHECK(mat.num_unused() == 0);
    graph.execute();
    CHECK(T(1, 1, 2, 1) == Catch::Approx(21.0));
    CHECK(r2(1, 1, 2, 1) == Catch::Approx(42.0));
    CHECK(T(0, 0, 2, 1) == Catch::Approx(0.0));
}
