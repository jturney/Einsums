//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("Rebind - basic tensor rebind", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    // Capture graph with A1
    cg::Graph graph("rebind_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    // Execute with A1
    graph.execute();

    auto C_ref1 = create_zero_tensor<double>("Cr1", 4, 5);
    reference_einsum("ij <- ik ; kj", &C_ref1, A1, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref1(ii, jj)) < 1e-12);
        }
    }

    // Rebind A1 → A2 (one line!)
    graph.rebind(A1, A2);

    // Execute again, should now use A2
    C.zero();
    graph.execute();

    auto C_ref2 = create_zero_tensor<double>("Cr2", 4, 5);
    reference_einsum("ij <- ik ; kj", &C_ref2, A2, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref2(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - dimension mismatch throws", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 5, 3); // Different first dim!
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_mismatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    REQUIRE_THROWS_AS(graph.rebind(A1, A2), std::invalid_argument);
}

TEST_CASE("Rebind - scale operation", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 3, 3);
    auto A2 = create_random_tensor<double>("A2", 3, 3);

    cg::Graph graph("rebind_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A1);
    }

    graph.execute();

    auto A2_copy = Tensor<double, 2>(A2);
    graph.rebind(A1, A2);
    graph.execute();

    // A2 should now be doubled
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A2(ii, jj) - 2.0 * A2_copy(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - string einsum", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_string");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A1, B);
    }

    graph.execute();

    graph.rebind(A1, A2);
    C.zero();
    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cr", 4, 5);
    reference_einsum("ij <- ik ; kj", &C_ref, A2, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("update_prefactors - correct after CSE removes an earlier einsum", "[ComputeGraph][Rebind][CSE]") {
    // Regression: update_prefactors used to locate the EinsumParams by
    // counting einsum nodes before the target in _nodes and indexing
    // _params_store by that ordinal. Any pass that removes or reorders
    // einsum nodes (CSE here) desynced the ordinal, so the update landed on
    // a dead einsum's params and the target silently kept its old scalars.
    // The params handle now lives on the EinsumDescriptor itself.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 4);
    auto C   = create_zero_tensor<double>("C", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("update_pf_after_cse");
    auto     &D = graph.create_tensor<double, 2>("D", 4, 4); // graph-owned duplicate
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);   // survivor
        cg::einsum("ik;kj->ij", &D, A, B);   // duplicate, removed by CSE
        cg::einsum("ik;kj->ij", &out, C, D); // target of update_prefactors
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 3); // Alloc(D) + 2 einsums

    // The target einsum is now the last node; scale its contribution by 2.
    cg::NodeId const out_id = graph.nodes()[graph.num_nodes() - 1].id;
    graph.update_prefactors(out_id, 0.0, 2.0);

    graph.execute();

    auto AB = create_zero_tensor<double>("AB", 4, 4);
    reference_einsum("ij <- ik ; kj", &AB, A, B);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 4);
    reference_einsum("ij <- ik ; kj", 0.0, &out_ref, 2.0, AB, AB);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            max_abs = std::max(max_abs, std::abs(out(ii, jj)));
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("update_prefactors - finds its einsum after Materialization inserts nodes", "[ComputeGraph][Rebind]") {
    // Defends node identity across pass-inserted nodes. Materialize, Free and Initialize nodes
    // used to keep the default id 0, which is also the first captured node's id, so a lookup by
    // the einsum's id found the Materialize placed in front of it and threw "not an einsum".
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 4);

    cg::Graph graph("update_pf_after_materialize");
    auto     &C = graph.declare_tensor<double, 2>(std::string("C"), 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }
    cg::NodeId const einsum_id = graph.nodes()[0].id;

    auto [modified, pass] = graph.apply<cg::passes::Materialization>();
    REQUIRE(modified);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::Materialize);

    graph.update_prefactors(einsum_id, 0.0, 2.0);
    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    reference_einsum("ij <- ik ; kj", 0.0, &C_ref, 2.0, A, B);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("update_prefactors - changes computation", "[ComputeGraph][Rebind]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);

    cg::Graph graph("update_pf");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Get the einsum node's ID
    cg::NodeId const einsum_id = graph.nodes()[0].id;

    // Execute with c_pf=0, ab_pf=1
    auto C_save = Tensor<double, 2>(C);
    graph.execute();
    auto C_v1 = Tensor<double, 2>(C);

    // Update prefactors to c_pf=1, ab_pf=2
    graph.update_prefactors(einsum_id, 1.0, 2.0);

    // Execute again, should compute C = 1*C + 2*A*B
    graph.execute();

    // Verify: C_new = C_v1 + 2*A*B
    auto C_ref = Tensor<double, 2>(C_v1);
    reference_einsum("ij <- ik ; kj", 1.0, &C_ref, 2.0, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - the pointer index follows the rebound tensor", "[ComputeGraph][Rebind]") {
    // Every by-address lookup on a Graph goes through the pointer index, and a
    // rebind moves what a slot names. The index used to be left behind: it went
    // on naming the tensor the graph was captured over and answered "not
    // registered" for the one it had just been rebound to, so a caller who
    // rebound and then asked the graph about the new tensor was told the graph
    // had never seen it.
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_ptr_index");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    cg::TensorId const a_id = graph.find_tensor_id_by_ptr(&A1);
    REQUIRE(a_id != 0);

    graph.rebind(A1, A2);

    // The index answers about the storage the slot names NOW,
    CHECK(graph.find_tensor_id_by_ptr(&A2) == a_id);
    // and stops claiming the address it was moved off is registered.
    CHECK(graph.find_tensor_id_by_ptr(&A1) == cg::TensorId{0});
    // The handle and the index agree, which is the invariant every by-address
    // lookup relies on.
    auto const *handle = graph.find_tensor_by_ptr(&A2);
    REQUIRE(handle != nullptr);
    CHECK(handle->id == a_id);
    CHECK(handle->tensor_ptr == static_cast<void *>(&A2));
}

TEST_CASE("Rebind - a rebound tensor answers the liveness-checked lookup", "[ComputeGraph][Rebind]") {
    // The by-object metadata readers pair the address with the caller's liveness
    // token, so that a tensor allocated on top of a dead one does not inherit its
    // id. A rebind moved the address and left the token, which made the pair name
    // a tensor that never existed: the new object with the previous object's
    // lifetime. The graph then disowned the very tensor it had just been rebound
    // to, and reading an annotation off it threw "is not registered".
    RuntimeTensor<double> A1("A1", std::vector<size_t>{4, 3});
    RuntimeTensor<double> A2("A2", std::vector<size_t>{4, 3});
    RuntimeTensor<double> B("B", std::vector<size_t>{3, 5});
    RuntimeTensor<double> C("C", std::vector<size_t>{4, 5});
    A1.zero();
    A2.zero();
    B.zero();
    C.zero();

    cg::Graph graph("rebind_live_lookup");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    cg::TensorId const a_id = graph.live_tensor_id_by_ptr(&A1, A1.liveness_token());
    REQUIRE(a_id != 0);

    graph.rebind(A1, A2);

    CHECK(graph.live_tensor_id_by_ptr(&A2, A2.liveness_token()) == a_id);
    CHECK(graph.live_tensor_id_by_ptr(&A1, A1.liveness_token()) == cg::TensorId{0});
}

namespace {

RuntimeTensor<double> filled(char const *name, std::size_t rows, std::size_t cols, double base) {
    RuntimeTensor<double> out(name, std::vector<size_t>{rows, cols});
    for (std::size_t n = 0; n < out.size(); ++n) {
        out.data()[n] = base + 0.25 * static_cast<double>(n % 7);
    }
    return out;
}

void check_against(RuntimeTensor<double> const &got, RuntimeTensor<double> const &a, RuntimeTensor<double> const &b) {
    RuntimeTensor<double> want("want", std::vector<size_t>{got.dim(0), got.dim(1)});
    want.zero();
    reference_einsum("ik;kj->ij", &want, a, b);
    for (std::size_t n = 0; n < want.size(); ++n) {
        INFO("element " << n);
        CHECK_THAT(got.data()[n], Catch::Matchers::WithinAbs(want.data()[n], 1e-12));
    }
}

} // namespace

TEST_CASE("Rebind - a tensor allocated over a dropped operand's address leaves the operand alone", "[ComputeGraph][Rebind]") {
    // A graph keeps an operand's storage alive after the caller drops its tensor, and the
    // caller's address is free from then on. A loop body captured the dropped operand; the
    // caller then captured a new tensor that landed at the same address, and rebound it. Handles
    // were identified by the caller's address, so the rebind matched the body's handle for the
    // dropped operand as well, and the body computed with the rebound tensor.
    std::optional<RuntimeTensor<double>> a{filled("A", 4, 3, 1.0)};
    auto const                           kept = *a; // the dropped operand's values, for the check
    auto const                           b    = filled("B", 3, 5, -0.5);
    RuntimeTensor<double>                e("E", std::vector<size_t>{4, 5});
    RuntimeTensor<double>                f("F", std::vector<size_t>{4, 5});
    e.zero();
    f.zero();

    cg::Graph graph("dropped operand");
    {
        auto                  &body = graph.add_loop("once", 1, [](size_t) { return false; });
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &e, *a, b);
    }

    // The same address, a different tensor.
    a.reset();
    a.emplace(filled("A2", 4, 3, 7.0));
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &f, *a, b);
    }
    auto const a3 = filled("A3", 4, 3, -2.0);
    graph.rebind(*a, const_cast<RuntimeTensor<double> &>(a3));

    graph.execute();
    check_against(e, kept, b);
    check_against(f, a3, b);
}

namespace {

/// A declared zero tensor `C` that an accumulating einsum writes, optimized so the graph holds
/// its Materialize and Initialize nodes, and caller storage holding 7.0 to bind it to.
struct DeclaredAccumulator {
    Tensor<double, 2>  A = create_random_tensor<double>("A", 3, 4);
    Tensor<double, 2>  B = create_random_tensor<double>("B", 4, 5);
    cg::Graph          graph{"declared_accumulator"};
    Tensor<double, 2> &C;
    Tensor<double, 2>  user{"user", 3, 5};

    explicit DeclaredAccumulator(bool optimized = true) : C(graph.declare_zero_tensor<double, 2>(std::string("C"), 3, 5)) {
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
        }
        if (optimized) {
            optimize();
        }
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                user(i, j) = 7.0;
            }
        }
    }

    void optimize() {
        auto pm = cg::PassManager::create_default();
        graph.apply(pm);
    }

    /// A rebind redefines C: the bound tensor's contents are its value, so the accumulation
    /// starts from 7, and the graph's own shell, which nothing reads any more, is never given
    /// storage.
    void check_redefined() {
        auto expected = create_zero_tensor<double>("expected", 3, 5);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                expected(i, j) = 7.0;
            }
        }
        reference_einsum("ik;kj->ij", 1.0, &expected, 1.0, A, B);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                CHECK_THAT(user(i, j), Catch::Matchers::WithinAbs(expected(i, j), 1e-12));
            }
        }
        CHECK_FALSE(C.is_materialized());
    }
};

} // namespace

// A rebind after optimizing, so the Materialize and Initialize nodes were built over the graph's
// shell for C. They used to keep acting on it: every replay allocated and zeroed a tensor nothing
// read, and the value in `user` was right only because they missed it. The lifecycle executors now
// look the tensor up when they run and do nothing for one a rebind gave to the caller's storage.
TEST_CASE("Rebind - a declared tensor rebound to caller storage leaves its lifecycle", "[ComputeGraph][Rebind]") {
    DeclaredAccumulator fixture;
    fixture.graph.rebind(fixture.C, fixture.user);
    fixture.graph.execute();
    fixture.check_redefined();
}

// The same, reached through bind by manifest name, which repoints the slot through the same rebind.
TEST_CASE("Rebind - a declared tensor bound by name leaves its lifecycle", "[ComputeGraph][Rebind]") {
    DeclaredAccumulator fixture;
    fixture.graph.bind("C", fixture.user);
    fixture.graph.execute();
    fixture.check_redefined();
}

// The order Graph::bind's documentation recommends: bind, then optimize. The passes then see a
// tensor the caller supplied, so they must give it no Materialize and no Initialize; an Initialize
// built over the handle at that point would zero the caller's contents on every replay.
TEST_CASE("Rebind - a declared tensor bound before optimizing keeps the caller's contents", "[ComputeGraph][Rebind]") {
    SECTION("rebind") {
        DeclaredAccumulator fixture(/*optimized=*/false);
        fixture.graph.rebind(fixture.C, fixture.user);
        fixture.optimize();
        fixture.graph.execute();
        fixture.check_redefined();
    }
    SECTION("bind by name") {
        DeclaredAccumulator fixture(/*optimized=*/false);
        fixture.graph.bind("C", fixture.user);
        fixture.optimize();
        fixture.graph.execute();
        fixture.check_redefined();
    }
}

// Binding a slot to the tensor it already reads redefines nothing: C keeps its lifecycle, so the
// graph still allocates and zeroes it, and the accumulation starts from zero. Marking it as the
// caller's would leave the graph's own tensor without storage.
TEST_CASE("Rebind - a declared tensor rebound to itself keeps its lifecycle", "[ComputeGraph][Rebind]") {
    DeclaredAccumulator fixture;
    fixture.graph.rebind(fixture.C, fixture.C);
    fixture.graph.execute();

    REQUIRE(fixture.C.is_materialized());
    auto expected = create_zero_tensor<double>("expected", 3, 5);
    reference_einsum("ik;kj->ij", 0.0, &expected, 1.0, fixture.A, fixture.B);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            CHECK_THAT(fixture.C(i, j), Catch::Matchers::WithinAbs(expected(i, j), 1e-12));
        }
    }
}

// A scratch the graph frees after its last use, rebound to caller storage once the Free exists.
// The Free looks its tensor up when it runs, which after the rebind is the caller's, and releasing
// that would hand the caller back a tensor with no storage. It does nothing instead.
TEST_CASE("Rebind - a Free built before a rebind leaves the caller's storage alone", "[ComputeGraph][Rebind]") {
    constexpr size_t n = 8;
    auto             A = create_random_tensor<double>("A", n, n);
    auto             B = create_random_tensor<double>("B", n, n);
    auto             C = create_zero_tensor<double>("C", n, n);

    cg::Graph          graph("freed_scratch");
    Tensor<double, 2> *scratch = nullptr;
    {
        cg::CaptureGuard const guard(graph);
        auto                  &tmp = graph.create_zero_tensor<double, 2>("tmp", n, n);
        scratch                    = &tmp;
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, B);
    }
    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(std::size_t{0});
    graph.apply(pm);
    REQUIRE(std::ranges::any_of(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Free; }));

    Tensor<double, 2> user("user", n, n);
    graph.rebind(*scratch, user);
    graph.execute();

    REQUIRE(user.is_materialized());
    Tensor<double, 2> AB("AB", n, n);
    Tensor<double, 2> ABB("ABB", n, n);
    reference_einsum("ik;kj->ij", 0.0, &AB, 1.0, A, B);
    reference_einsum("ik;kj->ij", 0.0, &ABB, 1.0, AB, B);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            CHECK_THAT(user(i, j), Catch::Matchers::WithinAbs(AB(i, j), 1e-12));
            CHECK_THAT(C(i, j), Catch::Matchers::WithinAbs(ABB(i, j), 1e-12));
        }
    }
}
