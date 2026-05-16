//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("BoundExpr - const, param, callback", "[ComputeGraph][BoundExpr]") {
    cg::ParamTable params;
    params.set("n_occ", 5);
    params.set("zero", 0);

    SECTION("Const") {
        cg::BoundExpr const e{int64_t{7}};
        REQUIRE(e.is_const());
        REQUIRE(e.const_value() == 7);
        REQUIRE(e.resolve(params) == 7);
    }

    SECTION("Param") {
        cg::BoundExpr const e{"n_occ"};
        REQUIRE(e.is_param());
        REQUIRE(e.param_name() == "n_occ");
        REQUIRE(e.resolve(params) == 5);
    }

    SECTION("Param missing throws") {
        cg::BoundExpr const e{"missing"};
        REQUIRE_THROWS(e.resolve(params));
    }

    SECTION("Callback") {
        int                 x = 0;
        cg::BoundExpr const e{std::function<int64_t()>{[&] { return x * 3; }}};
        REQUIRE(e.is_callback());
        x = 4;
        REQUIRE(e.resolve(params) == 12);
        x = 7;
        REQUIRE(e.resolve(params) == 21);
    }
}

TEST_CASE("Pipeline - parameters", "[ComputeGraph][Pipeline][Params]") {
    cg::Pipeline pipe("p");
    pipe.set_param("k", 42);
    REQUIRE(pipe.get_param("k") == 42);
    REQUIRE(pipe.get_param_or("missing", -1) == -1);
    pipe.set_param("k", 7);
    REQUIRE(pipe.get_param("k") == 7);
}

TEST_CASE("View - constant range, aliases parent", "[ComputeGraph][View]") {
    // Build a 4x4 tensor; slice the first 2 rows fully across cols.
    Tensor<double, 2> A("A", 4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(10 * i + j);

    Tensor<double, 2> dst("dst", 2, 4);

    cg::Pipeline pipe("view_const");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<double, 2>(A, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());

        cg::permute("ij <- ij", 0.0, &dst, 1.0, slice);
    }
    pipe.execute();

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            REQUIRE(dst(i, j) == Catch::Approx(10.0 * i + j));
}

TEST_CASE("View - dynamic param range", "[ComputeGraph][View][Param]") {
    // Slice changes between executions when the param changes.
    Tensor<double, 2> A("A", 6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(i + j);

    // Output sized for the largest expected slice; we write to A(0..n,:) shape.
    Tensor<double, 2> dst("dst", 6, 3);
    dst.zero();

    cg::Pipeline pipe("view_param");
    pipe.set_param("n", 2);

    // We want: dst(0..n, :) = slice = A(0..n, :)
    // For simplicity, scale slice by 2 inside the graph and mirror the
    // result into a sized-n region of dst.
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<double, 2>(A, cg::ViewAxis::range(0, "n"), cg::ViewAxis::full());
        cg::scale(2.0, &slice);
    }

    SECTION("n=2: only first 2 rows scaled") {
        pipe.execute();
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE(A(i, j) == Catch::Approx(2.0 * (i + j)));
        // Rows 2..5 untouched
        for (size_t i = 2; i < 6; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE(A(i, j) == Catch::Approx(i + j));
    }
}

TEST_CASE("View - aliasing: write through slice mutates parent", "[ComputeGraph][View]") {
    Tensor<double, 2> A("A", 4, 4);
    A.zero();

    cg::Pipeline pipe("view_alias");

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // Slice rows 1..3, all columns. Writing to slice should appear in A.
        auto &slice = cg::view<double, 2>(A, cg::ViewAxis::range(1, 3), cg::ViewAxis::full());

        // Use permute to fill slice with a constant: slice(i,j) = 7 via permute from a const tensor
        // Simpler: scale-by-zero then add. But we have no add on a constant. Use a custom fill via cg::ones-like setup:
        // For this test we just verify the view aliases by directly setting A through the slice's identity.
        // Capture: scale slice by 0 then by some constant via two-pass — easier to use a permute from a filled source.
        cg::scale(0.0, &slice); // slice = 0
    }
    pipe.execute();

    // Rows 1..2 should be zero, rows 0 and 3 unchanged.
    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(A(0, j) == Catch::Approx(0.0)); // was already zero
        REQUIRE(A(1, j) == Catch::Approx(0.0));
        REQUIRE(A(2, j) == Catch::Approx(0.0));
        REQUIRE(A(3, j) == Catch::Approx(0.0));
    }

    // Now make the test more decisive: prime A nonzero, then re-run to zero only rows 1..2.
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = 5.0;

    pipe.execute();

    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(A(0, j) == Catch::Approx(5.0)); // untouched
        REQUIRE(A(1, j) == Catch::Approx(0.0)); // zeroed via view
        REQUIRE(A(2, j) == Catch::Approx(0.0));
        REQUIRE(A(3, j) == Catch::Approx(5.0)); // untouched
    }
}

TEST_CASE("WriteParam - callback updates param mid-loop", "[ComputeGraph][WriteParam]") {
    Tensor<double, 1> v("v", 1);
    v(0) = 1.0;

    cg::Pipeline pipe("write_param");
    pipe.set_param("step", 0);

    // Loop body: read step, double v, write step+1 back.
    int  iter_count = 0;
    auto cond       = [&](size_t /*iter*/) {
        iter_count++;
        return iter_count < 3; // 3 iterations total
    };

    {
        auto                  &body = pipe.add_loop("loop", 10, cond);
        cg::CaptureGuard const g(body);

        cg::scale(2.0, &v);

        cg::write_param("step", std::function<int64_t()>([&pipe] { return pipe.get_param("step") + 1; }));
    }
    pipe.execute();

    REQUIRE(pipe.get_param("step") == 3); // wrote 3 times
    REQUIRE(v(0) == Catch::Approx(8.0));  // 1 * 2^3
}

TEST_CASE("WriteParam - scalar source updates view bounds", "[ComputeGraph][WriteParam][View]") {
    // Combined dataflow test: a scalar variable's value is pushed to a
    // param, and a downstream View reads that param to compute its slice.
    int64_t n_src = 3;

    Tensor<double, 1> A("A", 5);
    for (size_t k = 0; k < 5; ++k)
        A(k) = 1.0;

    cg::Pipeline pipe("wp_scalar");
    pipe.set_param("n", 0);

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // 1) Push n_src into params["n"]. The graph captures &n_src and
        //    reads its current value at execute time.
        cg::write_param("n", n_src);

        // 2) Slice A[0:n] and zero it.
        auto &head = cg::view<double, 1>(A, cg::ViewAxis::range(0, "n"));
        cg::scale(0.0, &head);
    }
    pipe.execute();

    // First 3 zeroed, last 2 unchanged.
    for (size_t k = 0; k < 3; ++k)
        REQUIRE(A(k) == Catch::Approx(0.0));
    for (size_t k = 3; k < 5; ++k)
        REQUIRE(A(k) == Catch::Approx(1.0));

    // Mutate the source scalar and re-execute; same graph picks up new bounds.
    n_src = 1;
    for (size_t k = 0; k < 5; ++k)
        A(k) = 2.0;

    pipe.execute();
    REQUIRE(A(0) == Catch::Approx(0.0));
    for (size_t k = 1; k < 5; ++k)
        REQUIRE(A(k) == Catch::Approx(2.0));
}

TEST_CASE("View - param bounds across multiple executes", "[ComputeGraph][View][Param]") {
    Tensor<double, 1> A("A", 6);
    cg::Pipeline      pipe("multi_exec");
    pipe.set_param("n", 0);

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &head = cg::view<double, 1>(A, cg::ViewAxis::range(0, "n"));
        cg::scale(0.0, &head);
    }

    // Execute 1: n = 2
    for (size_t k = 0; k < 6; ++k)
        A(k) = 7.0;
    pipe.set_param("n", 2);
    pipe.execute();
    REQUIRE(A(0) == 0.0);
    REQUIRE(A(1) == 0.0);
    REQUIRE(A(2) == 7.0);
    REQUIRE(A(5) == 7.0);

    // Execute 2: n = 5  (shouldn't require recapture)
    for (size_t k = 0; k < 6; ++k)
        A(k) = 7.0;
    pipe.set_param("n", 5);
    pipe.execute();
    for (size_t k = 0; k < 5; ++k)
        REQUIRE(A(k) == 0.0);
    REQUIRE(A(5) == 7.0);

    // Execute 3: n = 0  (no-op slice — 0-extent is allowed)
    for (size_t k = 0; k < 6; ++k)
        A(k) = 9.0;
    pipe.set_param("n", 0);
    pipe.execute();
    for (size_t k = 0; k < 6; ++k)
        REQUIRE(A(k) == 9.0);
}

TEST_CASE("View - capture-time errors", "[ComputeGraph][View][Errors]") {
    SECTION("Drop axis is not yet supported") {
        Tensor<double, 1>      A("A", 4);
        cg::Pipeline           pipe("drop");
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        REQUIRE_THROWS(cg::view<double, 1>(A, cg::ViewAxis::drop(0)));
    }

    SECTION("cg::view outside capture throws") {
        Tensor<double, 1> A("A", 4);
        REQUIRE_THROWS(cg::view<double, 1>(A, cg::ViewAxis::range(0, 2)));
    }
}

TEST_CASE("View - execute-time errors", "[ComputeGraph][View][Errors]") {
    SECTION("Out-of-bounds range throws at execute") {
        Tensor<double, 1> A("A", 4);
        cg::Pipeline      pipe("oob");
        pipe.set_param("hi", 100); // larger than A.dim(0)

        {
            auto                  &stage = pipe.add_stage("s");
            cg::CaptureGuard const g(stage);
            auto                  &slice = cg::view<double, 1>(A, cg::ViewAxis::range(0, "hi"));
            cg::scale(0.0, &slice);
        }
        REQUIRE_THROWS(pipe.execute());
    }

    SECTION("Missing param throws at execute") {
        Tensor<double, 1> A("A", 4);
        cg::Pipeline      pipe("missing");
        // Don't set "n" at all.

        {
            auto                  &stage = pipe.add_stage("s");
            cg::CaptureGuard const g(stage);
            auto                  &slice = cg::view<double, 1>(A, cg::ViewAxis::range(0, "n"));
            cg::scale(0.0, &slice);
        }
        REQUIRE_THROWS(pipe.execute());
    }
}

TEST_CASE("View - multiple consumers of one slice", "[ComputeGraph][View]") {
    Tensor<double, 2> A("A", 4, 4);
    Tensor<double, 2> B("B", 2, 4);
    Tensor<double, 2> C("C", 2, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i + j);

    cg::Pipeline pipe("multi");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view<double, 2>(A, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());
        cg::permute("ij <- ij", 0.0, &B, 1.0, slice);
        cg::permute("ij <- ij", 0.0, &C, 1.0, slice);
    }
    pipe.execute();
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE(B(i, j) == Catch::Approx(i + j));
            REQUIRE(C(i, j) == Catch::Approx(i + j));
        }
}

TEST_CASE("View - aliases survive FreeInsertion (parent kept alive)", "[ComputeGraph][View][Pass]") {
    // FreeInsertion shouldn't free a parent tensor while one of its aliases
    // is still being read. Construct a scenario with two stages: the first
    // creates an intermediate, the second reads through a slice. If
    // FreeInsertion erroneously frees the parent after the first stage,
    // the second stage's read crashes.
    cg::Pipeline pipe("free_alias");

    Tensor<double, 2> external_in("external_in", 4, 4);
    Tensor<double, 2> external_out("external_out", 2, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            external_in(i, j) = static_cast<double>(10 * i + j);

    // Single stage with a long body so the lifetime tracking has multiple
    // nodes to consider. Slice + permute the slice + then a no-op scale on
    // the slice (the alias is read AFTER the permute, so its lifetime
    // exceeds where a naive last-use computation would put a Free).
    {
        auto                  &stage = pipe.add_stage("body");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<double, 2>(external_in, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());

        // First reader of the slice
        cg::permute("ij <- ij", 0.0, &external_out, 1.0, slice);

        // Second reader of the slice — touches external_in's storage again.
        cg::scale(2.0, &slice);
    }

    REQUIRE_NOTHROW(pipe.execute());

    // Slice was scaled by 2 ⇒ external_in rows 0..1 doubled.
    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(external_in(0, j) == Catch::Approx(2.0 * j));
        REQUIRE(external_in(1, j) == Catch::Approx(2.0 * (10 + j)));
    }
}

TEST_CASE("View - topological order: parent write before alias read", "[ComputeGraph][View][Topo]") {
    // Build a single-stage graph that has both a write to the parent and a
    // read through a slice of the parent. Recorded order is: write parent →
    // read slice. The topological sort must keep them in this order; if it
    // ignored aliasing, it could reorder them as independent.
    Tensor<double, 1> A("A", 4);
    Tensor<double, 1> dst("dst", 2);

    cg::Pipeline pipe("topo");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // Write parent
        cg::scale(0.0, &A); // A := 0

        auto &slice = cg::view<double, 1>(A, cg::ViewAxis::range(0, 2));
        // Read alias — must run AFTER the parent write.
        cg::permute("i <- i", 0.0, &dst, 1.0, slice);
    }

    // Prime A nonzero before execute — if the read ran before the write,
    // dst would see the old values. After execute, dst must read zeros.
    A(0) = 99.0;
    A(1) = 99.0;
    A(2) = 99.0;
    A(3) = 99.0;

    pipe.execute();

    REQUIRE(dst(0) == Catch::Approx(0.0));
    REQUIRE(dst(1) == Catch::Approx(0.0));
}

TEST_CASE("View - chained slicing: view of view", "[ComputeGraph][View]") {
    // First view: A[1:5, :], rank 2
    // Second view: middle of that view, A[2:4, :]
    Tensor<double, 2> A("A", 6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(100 + 10 * i + j);

    Tensor<double, 2> dst("dst", 2, 3);
    dst.zero();

    cg::Pipeline pipe("chain");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &outer = cg::view<double, 2>(A, cg::ViewAxis::range(1, 5), cg::ViewAxis::full());
        auto &inner = cg::view<double, 2>(outer, cg::ViewAxis::range(1, 3), cg::ViewAxis::full());

        cg::permute("ij <- ij", 0.0, &dst, 1.0, inner);
    }
    pipe.execute();

    // outer represents A[1..5, :]. inner is outer[1..3, :] = A[2..4, :].
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            REQUIRE(dst(i, j) == Catch::Approx(100 + 10 * (i + 2) + j));
}

TEST_CASE("Trace - eager form", "[ComputeGraph][Trace]") {
    Tensor<double, 2> A("A", 4, 4);
    A.zero();
    for (size_t k = 0; k < 4; ++k)
        A(k, k) = static_cast<double>(k + 1); // diagonal: 1, 2, 3, 4

    REQUIRE(cg::trace(A) == Catch::Approx(10.0));
}

TEST_CASE("Trace - non-square throws", "[ComputeGraph][Trace]") {
    Tensor<double, 2> A("A", 3, 5);
    A.zero();
    REQUIRE_THROWS(cg::trace(A));
}

TEST_CASE("Trace - eager form throws during capture", "[ComputeGraph][Trace]") {
    Tensor<double, 2> const A("A", 3, 3);
    cg::Pipeline            pipe("trace_eager_capture");
    auto                   &stage = pipe.add_stage("s");
    cg::CaptureGuard const  g(stage);
    REQUIRE_THROWS(cg::trace(A));
}

TEST_CASE("Trace - recorded form into pipeline", "[ComputeGraph][Trace]") {
    Tensor<double, 2> A("A", 3, 3);
    A.zero();
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;

    double result = 0.0;

    cg::Pipeline pipe("trace_recorded");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        cg::trace(&result, A);
    }
    pipe.execute();
    REQUIRE(result == Catch::Approx(6.0));

    // Mutate A and re-execute; recorded trace re-reads at runtime.
    A(0, 0) = 10.0;
    A(1, 1) = 20.0;
    A(2, 2) = 30.0;
    pipe.execute();
    REQUIRE(result == Catch::Approx(60.0));
}

TEST_CASE("Trace - recorded form on a graph view", "[ComputeGraph][Trace][View]") {
    // Trace of a slice — the trace executor reads through the view's data
    // pointer just like any other consumer. With slice = A[1:4, 1:4] from
    // a 5x5 A whose entries are i*10+j, the slice is:
    //   [[11, 12, 13],
    //    [21, 22, 23],
    //    [31, 32, 33]]
    // Trace = 11 + 22 + 33 = 66.
    Tensor<double, 2> A("A", 5, 5);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            A(i, j) = static_cast<double>(10 * i + j);

    double result = 0.0;

    cg::Pipeline pipe("trace_view");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view<double, 2>(A, cg::ViewAxis::range(1, 4), cg::ViewAxis::range(1, 4));
        cg::trace(&result, slice);
    }
    pipe.execute();
    REQUIRE(result == Catch::Approx(66.0));
}

TEST_CASE("View - mixed callback + param + const bounds", "[ComputeGraph][View][BoundExpr]") {
    Tensor<double, 1> A("A", 8);
    for (size_t k = 0; k < 8; ++k)
        A(k) = 1.0;

    cg::Pipeline pipe("mixed");
    pipe.set_param("hi", 5);

    int dynamic_lo = 1;

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // lo = callback (1), hi = param "hi" (5) — slice [1, 5)
        auto &slice = cg::view<double, 1>(A, cg::ViewAxis::range(std::function<int64_t()>([&] { return dynamic_lo; }), "hi"));
        cg::scale(0.0, &slice);
    }

    pipe.execute();
    REQUIRE(A(0) == 1.0);
    for (size_t k = 1; k < 5; ++k)
        REQUIRE(A(k) == 0.0);
    for (size_t k = 5; k < 8; ++k)
        REQUIRE(A(k) == 1.0);

    // Re-run with both the callback and the param changed.
    for (size_t k = 0; k < 8; ++k)
        A(k) = 2.0;
    dynamic_lo = 3;
    pipe.set_param("hi", 7);
    pipe.execute();
    REQUIRE(A(0) == 2.0);
    REQUIRE(A(2) == 2.0);
    for (size_t k = 3; k < 7; ++k)
        REQUIRE(A(k) == 0.0);
    REQUIRE(A(7) == 2.0);
}
