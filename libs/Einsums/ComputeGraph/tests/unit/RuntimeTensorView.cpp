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
