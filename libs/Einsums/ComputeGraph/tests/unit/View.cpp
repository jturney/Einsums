//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <array>
#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// A value of element type T carrying @p x, with an imaginary part for a complex T so a copy that
// drops it, or a view that lands a real part's width off, cannot pass. Both parts of every value
// used here are small multiples of one half, which every type stores exactly, so the moves and
// doublings these tests make are compared exactly.
template <typename T>
T val(double x) {
    return testing::prefactor<T>(x, -0.5 * x);
}

// Within a few ulps of the element type, relative to the expected value with a floor of one.
template <typename T>
auto near(T want) {
    return CheckWithinRel(want, 1000.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon());
}

} // namespace

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

TEMPLATE_LIST_TEST_CASE("View - constant range, aliases parent", "[ComputeGraph][View]", testing::AllScalarTypes) {
    using T = TestType;
    // Build a 4x4 tensor; slice the first 2 rows fully across cols.
    Tensor<T, 2> A("A", 4, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = val<T>(10 * i + j);

    Tensor<T, 2> dst("dst", 2, 4);

    cg::Pipeline pipe("view_const");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<T, 2>(A, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());

        cg::permute("ij <- ij", 0.0, &dst, 1.0, slice);
    }
    pipe.execute();

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            REQUIRE(dst(i, j) == val<T>(10.0 * i + j));
}

TEMPLATE_LIST_TEST_CASE("View - dynamic param range", "[ComputeGraph][View][Param]", testing::AllScalarTypes) {
    using T = TestType;
    // Slice changes between executions when the param changes.
    Tensor<T, 2> A("A", 6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = val<T>(i + j);

    // Output sized for the largest expected slice; we write to A(0..n,:) shape.
    Tensor<T, 2> dst("dst", 6, 3);
    dst.zero();

    cg::Pipeline pipe("view_param");
    pipe.set_param("n", 2);

    // We want: dst(0..n, :) = slice = A(0..n, :)
    // For simplicity, scale slice by 2 inside the graph and mirror the
    // result into a sized-n region of dst.
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<T, 2>(A, cg::ViewAxis::range(0, "n"), cg::ViewAxis::full());
        cg::scale(2.0, &slice);
    }

    SECTION("n=2: only first 2 rows scaled") {
        pipe.execute();
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE(A(i, j) == val<T>(2.0 * (i + j)));
        // Rows 2..5 untouched
        for (size_t i = 2; i < 6; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE(A(i, j) == val<T>(i + j));
    }
}

TEMPLATE_LIST_TEST_CASE("View - aliasing: write through slice mutates parent", "[ComputeGraph][View]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 2> A("A", 4, 4);
    A.zero();

    cg::Pipeline pipe("view_alias");

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // Slice rows 1..3, all columns. Writing to slice should appear in A.
        auto &slice = cg::view<T, 2>(A, cg::ViewAxis::range(1, 3), cg::ViewAxis::full());

        // Use permute to fill slice with a constant: slice(i,j) = 7 via permute from a const tensor
        // Simpler: scale-by-zero then add. But we have no add on a constant. Use a custom fill via cg::ones-like setup:
        // For this test we just verify the view aliases by directly setting A through the slice's identity.
        // Capture: scale slice by 0 then by some constant via two-pass, easier to use a permute from a filled source.
        cg::scale(0.0, &slice); // slice = 0
    }
    pipe.execute();

    // Rows 1..2 should be zero, rows 0 and 3 unchanged.
    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(A(0, j) == val<T>(0.0)); // was already zero
        REQUIRE(A(1, j) == val<T>(0.0));
        REQUIRE(A(2, j) == val<T>(0.0));
        REQUIRE(A(3, j) == val<T>(0.0));
    }

    // Now make the test more decisive: prime A nonzero, then re-run to zero only rows 1..2.
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = val<T>(5.0);

    pipe.execute();

    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(A(0, j) == val<T>(5.0)); // untouched
        REQUIRE(A(1, j) == val<T>(0.0)); // zeroed via view
        REQUIRE(A(2, j) == val<T>(0.0));
        REQUIRE(A(3, j) == val<T>(5.0)); // untouched
    }
}

TEMPLATE_LIST_TEST_CASE("View - a view made outside capture still orders against its parent", "[ComputeGraph][View][aliasing]",
                        testing::AllScalarTypes) {
    using T = TestType;
    // cg::view() sets TensorHandle::aliases itself, but a view sliced OUTSIDE a
    // capture never goes through it: it reaches the graph as an ordinary
    // operand on first use. Registered with aliases == 0 it looked unrelated to
    // its parent, so the scheduler was free to order a read of the parent
    // before the writes through the views, silently returning a stale result.
    // Graph::link_alias_storage now recovers the relationship from the storage.
    //
    // The pattern is not exotic: a captured view must outlive the graph, which
    // pushes callers to build their views up front.
    constexpr size_t N = 6;
    Tensor<T, 2>     A("A", N, N);
    Tensor<T, 1>     ones("ones", N);
    T                out{0};
    A.zero();
    ones.set_all(T{1});

    // Sliced eagerly, before any capture exists.
    std::vector<TensorView<T, 1>> rows;
    rows.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        rows.push_back(A(static_cast<int>(i), All));
    }

    cg::Graph graph("view_outside_capture");
    {
        cg::CaptureGuard const capture(graph);
        for (size_t i = 0; i < N; ++i) {
            cg::scale(0.0, &rows[i]); // write every row through a view
            cg::axpy(static_cast<T>(static_cast<RemoveComplexT<T>>(i + 1)), ones, &rows[i]);
        }
        cg::dot(&out, A, A); // read the parent: must come after all of them
    }
    graph.execute();

    T expected{0};
    for (size_t i = 0; i < N; ++i) {
        expected += static_cast<RemoveComplexT<T>>(N * (i + 1) * (i + 1));
    }
    REQUIRE_THAT(out, near(expected));

    // Disjoint slices must still be free to run concurrently: recovering the
    // parent link must not collapse every view into a whole-tensor access.
    auto const &deps = graph.dependencies();
    REQUIRE(deps.levels.size() > 1);
    REQUIRE(deps.levels.front().size() > 1);
}

TEMPLATE_LIST_TEST_CASE("WriteParam - callback updates param mid-loop", "[ComputeGraph][WriteParam]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 1> v("v", 1);
    v(0) = val<T>(1.0);

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
    REQUIRE(v(0) == val<T>(8.0));         // 1 * 2^3
}

TEMPLATE_LIST_TEST_CASE("WriteParam - scalar source updates view bounds", "[ComputeGraph][WriteParam][View]", testing::AllScalarTypes) {
    using T = TestType;
    // Combined dataflow test: a scalar variable's value is pushed to a
    // param, and a downstream View reads that param to compute its slice.
    int64_t n_src = 3;

    Tensor<T, 1> A("A", 5);
    for (size_t k = 0; k < 5; ++k)
        A(k) = val<T>(1.0);

    cg::Pipeline pipe("wp_scalar");
    pipe.set_param("n", 0);

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // 1) Push n_src into params["n"]. The graph captures &n_src and
        //    reads its current value at execute time.
        cg::write_param("n", n_src);

        // 2) Slice A[0:n] and zero it.
        auto &head = cg::view<T, 1>(A, cg::ViewAxis::range(0, "n"));
        cg::scale(0.0, &head);
    }
    pipe.execute();

    // First 3 zeroed, last 2 unchanged.
    for (size_t k = 0; k < 3; ++k)
        REQUIRE(A(k) == val<T>(0.0));
    for (size_t k = 3; k < 5; ++k)
        REQUIRE(A(k) == val<T>(1.0));

    // Mutate the source scalar and re-execute; same graph picks up new bounds.
    n_src = 1;
    for (size_t k = 0; k < 5; ++k)
        A(k) = val<T>(2.0);

    pipe.execute();
    REQUIRE(A(0) == val<T>(0.0));
    for (size_t k = 1; k < 5; ++k)
        REQUIRE(A(k) == val<T>(2.0));
}

TEMPLATE_LIST_TEST_CASE("View - param bounds across multiple executes", "[ComputeGraph][View][Param]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 1> A("A", 6);
    cg::Pipeline pipe("multi_exec");
    pipe.set_param("n", 0);

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &head = cg::view<T, 1>(A, cg::ViewAxis::range(0, "n"));
        cg::scale(0.0, &head);
    }

    // Execute 1: n = 2
    for (size_t k = 0; k < 6; ++k)
        A(k) = val<T>(7.0);
    pipe.set_param("n", 2);
    pipe.execute();
    REQUIRE(A(0) == val<T>(0.0));
    REQUIRE(A(1) == val<T>(0.0));
    REQUIRE(A(2) == val<T>(7.0));
    REQUIRE(A(5) == val<T>(7.0));

    // Execute 2: n = 5  (shouldn't require recapture)
    for (size_t k = 0; k < 6; ++k)
        A(k) = val<T>(7.0);
    pipe.set_param("n", 5);
    pipe.execute();
    for (size_t k = 0; k < 5; ++k)
        REQUIRE(A(k) == val<T>(0.0));
    REQUIRE(A(5) == val<T>(7.0));

    // Execute 3: n = 0  (no-op slice, 0-extent is allowed)
    for (size_t k = 0; k < 6; ++k)
        A(k) = val<T>(9.0);
    pipe.set_param("n", 0);
    pipe.execute();
    for (size_t k = 0; k < 6; ++k)
        REQUIRE(A(k) == val<T>(9.0));
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

TEMPLATE_LIST_TEST_CASE("View - multiple consumers of one slice", "[ComputeGraph][View]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 2> A("A", 4, 4);
    Tensor<T, 2> B("B", 2, 4);
    Tensor<T, 2> C("C", 2, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = val<T>(i + j);

    cg::Pipeline pipe("multi");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view<T, 2>(A, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());
        cg::permute("ij <- ij", 0.0, &B, 1.0, slice);
        cg::permute("ij <- ij", 0.0, &C, 1.0, slice);
    }
    pipe.execute();
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            REQUIRE(B(i, j) == val<T>(i + j));
            REQUIRE(C(i, j) == val<T>(i + j));
        }
}

TEMPLATE_LIST_TEST_CASE("View - aliases survive FreeInsertion (parent kept alive)", "[ComputeGraph][View][Pass]", testing::AllScalarTypes) {
    using T = TestType;
    // FreeInsertion shouldn't free a parent tensor while one of its aliases
    // is still being read. Construct a scenario with two stages: the first
    // creates an intermediate, the second reads through a slice. If
    // FreeInsertion erroneously frees the parent after the first stage,
    // the second stage's read crashes.
    cg::Pipeline pipe("free_alias");

    Tensor<T, 2> external_in("external_in", 4, 4);
    Tensor<T, 2> external_out("external_out", 2, 4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            external_in(i, j) = val<T>(10 * i + j);

    // Single stage with a long body so the lifetime tracking has multiple
    // nodes to consider. Slice + permute the slice + then a no-op scale on
    // the slice (the alias is read AFTER the permute, so its lifetime
    // exceeds where a naive last-use computation would put a Free).
    {
        auto                  &stage = pipe.add_stage("body");
        cg::CaptureGuard const g(stage);

        auto &slice = cg::view<T, 2>(external_in, cg::ViewAxis::range(0, 2), cg::ViewAxis::full());

        // First reader of the slice
        cg::permute("ij <- ij", 0.0, &external_out, 1.0, slice);

        // Second reader of the slice, touches external_in's storage again.
        cg::scale(2.0, &slice);
    }

    REQUIRE_NOTHROW(pipe.execute());

    // Slice was scaled by 2 ⇒ external_in rows 0..1 doubled.
    for (size_t j = 0; j < 4; ++j) {
        REQUIRE(external_in(0, j) == val<T>(2.0 * j));
        REQUIRE(external_in(1, j) == val<T>(2.0 * (10 + j)));
    }
}

TEMPLATE_LIST_TEST_CASE("View - topological order: parent write before alias read", "[ComputeGraph][View][Topo]", testing::AllScalarTypes) {
    using T = TestType;
    // Build a single-stage graph that has both a write to the parent and a
    // read through a slice of the parent. Recorded order is: write parent →
    // read slice. The topological sort must keep them in this order; if it
    // ignored aliasing, it could reorder them as independent.
    Tensor<T, 1> A("A", 4);
    Tensor<T, 1> dst("dst", 2);

    cg::Pipeline pipe("topo");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // Write parent
        cg::scale(0.0, &A); // A := 0

        auto &slice = cg::view<T, 1>(A, cg::ViewAxis::range(0, 2));
        // Read alias, must run AFTER the parent write.
        cg::permute("i <- i", 0.0, &dst, 1.0, slice);
    }

    // Prime A nonzero before execute, if the read ran before the write,
    // dst would see the old values. After execute, dst must read zeros.
    A(0) = val<T>(99.0);
    A(1) = val<T>(99.0);
    A(2) = val<T>(99.0);
    A(3) = val<T>(99.0);

    pipe.execute();

    REQUIRE(dst(0) == val<T>(0.0));
    REQUIRE(dst(1) == val<T>(0.0));
}

TEMPLATE_LIST_TEST_CASE("View - chained slicing: view of view", "[ComputeGraph][View]", testing::AllScalarTypes) {
    using T = TestType;
    // First view: A[1:5, :], rank 2
    // Second view: middle of that view, A[2:4, :]
    Tensor<T, 2> A("A", 6, 3);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = val<T>(100 + 10 * i + j);

    Tensor<T, 2> dst("dst", 2, 3);
    dst.zero();

    cg::Pipeline pipe("chain");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &outer = cg::view<T, 2>(A, cg::ViewAxis::range(1, 5), cg::ViewAxis::full());
        auto &inner = cg::view<T, 2>(outer, cg::ViewAxis::range(1, 3), cg::ViewAxis::full());

        cg::permute("ij <- ij", 0.0, &dst, 1.0, inner);
    }
    pipe.execute();

    // outer represents A[1..5, :]. inner is outer[1..3, :] = A[2..4, :].
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            REQUIRE(dst(i, j) == val<T>(100 + 10 * (i + 2) + j));
}

TEMPLATE_LIST_TEST_CASE("View - typed permute_view transposes, aliases parent", "[ComputeGraph][View][Permute]", testing::AllScalarTypes) {
    using T = TestType;
    // A is 3x4; the transpose view is 4x3 with At(i,j) == A(j,i).
    Tensor<T, 2> A("A", 3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            A(i, j) = val<T>(10 * i + j);

    Tensor<T, 2> dst("dst", 4, 3);
    dst.zero();

    cg::Pipeline pipe("permute_view");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &At = cg::permute_view(A, std::array<size_t, 2>{1, 0}); // A^T (4x3)
        cg::permute("ij <- ij", 0.0, &dst, 1.0, At);                 // dst = At
    }
    pipe.execute();

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j)
            REQUIRE(dst(i, j) == val<T>(10.0 * j + i)); // At(i,j) = A(j,i)
}

TEMPLATE_LIST_TEST_CASE("View - permute_view rank-3 axis permutation", "[ComputeGraph][View][Permute]", testing::AllScalarTypes) {
    using T = TestType;
    // R is 2x3x4; permute axes to (4,2,3) via perm {2,0,1}: V(i,j,k) = R(j,k,i).
    Tensor<T, 3> R("R", 2, 3, 4);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                R(i, j, k) = val<T>(100 * i + 10 * j + k);

    Tensor<T, 3> dst("dst", 4, 2, 3);
    dst.zero();

    cg::Pipeline pipe("permute_view3");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        auto &V = cg::permute_view(R, std::array<size_t, 3>{2, 0, 1});
        cg::permute("ijk <- ijk", 0.0, &dst, 1.0, V);
    }
    pipe.execute();

    for (size_t i = 0; i < 4; ++i)         // result axis 0 <- R axis 2 (k)
        for (size_t j = 0; j < 2; ++j)     // result axis 1 <- R axis 0 (i)
            for (size_t k = 0; k < 3; ++k) // result axis 2 <- R axis 1 (j)
                REQUIRE(dst(i, j, k) == val<T>(100.0 * j + 10.0 * k + i));
}

TEST_CASE("View - permute_view rejects a non-bijection", "[ComputeGraph][View][Permute][Errors]") {
    Tensor<double, 2>      A("A", 3, 3);
    cg::Pipeline           pipe("permute_view_bad");
    auto                  &stage = pipe.add_stage("s");
    cg::CaptureGuard const g(stage);
    // {0, 0} is not a permutation of [0, 2).
    REQUIRE_THROWS(cg::permute_view(A, std::array<size_t, 2>{0, 0}));
}

TEMPLATE_LIST_TEST_CASE("Trace - eager form", "[ComputeGraph][Trace]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 2> A("A", 4, 4);
    A.zero();
    for (size_t k = 0; k < 4; ++k)
        A(k, k) = val<T>(k + 1); // diagonal: 1, 2, 3, 4

    REQUIRE(cg::trace(A) == val<T>(10.0));
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

TEMPLATE_LIST_TEST_CASE("Trace - recorded form into pipeline", "[ComputeGraph][Trace]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 2> A("A", 3, 3);
    A.zero();
    A(0, 0) = val<T>(1.0);
    A(1, 1) = val<T>(2.0);
    A(2, 2) = val<T>(3.0);

    T result{0};

    cg::Pipeline pipe("trace_recorded");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        cg::trace(&result, A);
    }
    pipe.execute();
    REQUIRE(result == val<T>(6.0));

    // Mutate A and re-execute; recorded trace re-reads at runtime.
    A(0, 0) = val<T>(10.0);
    A(1, 1) = val<T>(20.0);
    A(2, 2) = val<T>(30.0);
    pipe.execute();
    REQUIRE(result == val<T>(60.0));
}

TEMPLATE_LIST_TEST_CASE("Trace - recorded form on a graph view", "[ComputeGraph][Trace][View]", testing::AllScalarTypes) {
    using T = TestType;
    // Trace of a slice, the trace executor reads through the view's data
    // pointer just like any other consumer. With slice = A[1:4, 1:4] from
    // a 5x5 A whose entries are i*10+j, the slice is:
    //   [[11, 12, 13],
    //    [21, 22, 23],
    //    [31, 32, 33]]
    // Trace = 11 + 22 + 33 = 66.
    Tensor<T, 2> A("A", 5, 5);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            A(i, j) = val<T>(10 * i + j);

    T result{0};

    cg::Pipeline pipe("trace_view");
    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);
        auto                  &slice = cg::view<T, 2>(A, cg::ViewAxis::range(1, 4), cg::ViewAxis::range(1, 4));
        cg::trace(&result, slice);
    }
    pipe.execute();
    REQUIRE(result == val<T>(66.0));
}

TEMPLATE_LIST_TEST_CASE("View - mixed callback + param + const bounds", "[ComputeGraph][View][BoundExpr]", testing::AllScalarTypes) {
    using T = TestType;
    Tensor<T, 1> A("A", 8);
    for (size_t k = 0; k < 8; ++k)
        A(k) = val<T>(1.0);

    cg::Pipeline pipe("mixed");
    pipe.set_param("hi", 5);

    int dynamic_lo = 1;

    {
        auto                  &stage = pipe.add_stage("s");
        cg::CaptureGuard const g(stage);

        // lo = callback (1), hi = param "hi" (5), slice [1, 5)
        auto &slice = cg::view<T, 1>(A, cg::ViewAxis::range(std::function<int64_t()>([&] { return dynamic_lo; }), "hi"));
        cg::scale(0.0, &slice);
    }

    pipe.execute();
    REQUIRE(A(0) == val<T>(1.0));
    for (size_t k = 1; k < 5; ++k)
        REQUIRE(A(k) == val<T>(0.0));
    for (size_t k = 5; k < 8; ++k)
        REQUIRE(A(k) == val<T>(1.0));

    // Re-run with both the callback and the param changed.
    for (size_t k = 0; k < 8; ++k)
        A(k) = val<T>(2.0);
    dynamic_lo = 3;
    pipe.set_param("hi", 7);
    pipe.execute();
    REQUIRE(A(0) == val<T>(2.0));
    REQUIRE(A(2) == val<T>(2.0));
    for (size_t k = 3; k < 7; ++k)
        REQUIRE(A(k) == val<T>(0.0));
    REQUIRE(A(7) == val<T>(2.0));
}

TEMPLATE_LIST_TEST_CASE("batched_gemm - one node for many independent GEMMs", "[ComputeGraph][BatchedGemm]", testing::AllScalarTypes) {
    using T = TestType;
    // The GEMMBatching pass could already produce this node; the point of the
    // op is reaching it without emitting one node per GEMM first, since capture
    // costs tens of microseconds a node and a big batch pays that N times.
    constexpr size_t          N = 12, m = 5, k = 4, n = 3;
    std::vector<Tensor<T, 2>> A, B, C, want;
    for (size_t i = 0; i < N; ++i) {
        A.emplace_back("a", m, k);
        B.emplace_back("b", k, n);
        C.emplace_back("c", m, n);
        want.emplace_back("w", m, n);
        for (size_t p = 0; p < m; ++p)
            for (size_t q = 0; q < k; ++q)
                A[i](p, q) = static_cast<T>(p + 2 * q + i);
        for (size_t p = 0; p < k; ++p)
            for (size_t q = 0; q < n; ++q)
                B[i](p, q) = static_cast<T>(3 * p - q + i);
        C[i].set_all(val<T>(1.0));
        want[i].set_all(val<T>(1.0));
        // reference: want = 2*A*B + 0.5*want
        for (size_t p = 0; p < m; ++p)
            for (size_t q = 0; q < n; ++q) {
                T acc{0};
                for (size_t r = 0; r < k; ++r)
                    acc += A[i](p, r) * B[i](r, q);
                want[i](p, q) = T{2} * acc + T{0.5} * want[i](p, q);
            }
    }

    std::vector<Tensor<T, 2> const *> a_ptr, b_ptr;
    std::vector<Tensor<T, 2> *>       c_ptr;
    for (size_t i = 0; i < N; ++i) {
        a_ptr.push_back(&A[i]);
        b_ptr.push_back(&B[i]);
        c_ptr.push_back(&C[i]);
    }

    SECTION("captured: exactly one node") {
        cg::Graph graph("batched");
        {
            cg::CaptureGuard const capture(graph);
            cg::batched_gemm(2.0, a_ptr, b_ptr, 0.5, c_ptr);
        }
        REQUIRE(graph.num_nodes() == 1);
        graph.execute();
        for (size_t i = 0; i < N; ++i)
            for (size_t p = 0; p < m; ++p)
                for (size_t q = 0; q < n; ++q)
                    REQUIRE_THAT(C[i](p, q), near(want[i](p, q)));
    }

    SECTION("eager: same result outside capture") {
        cg::batched_gemm(2.0, a_ptr, b_ptr, 0.5, c_ptr);
        for (size_t i = 0; i < N; ++i)
            for (size_t p = 0; p < m; ++p)
                for (size_t q = 0; q < n; ++q)
                    REQUIRE_THAT(C[i](p, q), near(want[i](p, q)));
    }

    SECTION("a member that cannot share the batch's scalars is rejected") {
        Tensor<T, 2> odd("odd", m + 1, n);
        auto         bad = c_ptr;
        bad.back()       = &odd;
        REQUIRE_THROWS_AS(cg::batched_gemm(1.0, a_ptr, b_ptr, 0.0, bad), std::invalid_argument);
        REQUIRE_THROWS_AS(cg::batched_gemm(1.0, decltype(a_ptr){}, decltype(b_ptr){}, 0.0, decltype(c_ptr){}), std::invalid_argument);
    }
}
