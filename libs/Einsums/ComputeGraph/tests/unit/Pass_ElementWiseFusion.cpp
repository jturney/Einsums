//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ElementWiseFusion.cpp
/// @brief Unit tests for the ElementWiseFusion optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

using einsums::testing::reference_einsum;

namespace {

// A few ulps of the element type: every check here composes at most three
// order-one scalars or sums four order-one products.
template <typename T>
constexpr double tol() {
    return 1000.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
}

template <typename T>
T pf(double re, double im) {
    return testing::prefactor<T>(re, im);
}

// Element by element over two tensors of one shape, however they are stored.
template <typename GotType, typename WantType>
void require_close(GotType const &got, WantType const &want) {
    using T = typename GotType::ValueType;
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(std::abs(got.data()[i] - want.data()[i]) <= tol<T>() * (1.0 + std::abs(want.data()[i])));
    }
}

} // namespace

TEST_CASE("ElementWiseFusion - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("ewf_empty");

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    CHECK_FALSE(modified);
    CHECK(ewf.num_fused() == 0);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - single node", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 3);

    cg::Graph graph("ewf_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(pf<T>(2.0, 0.5), &A);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    CHECK_FALSE(modified);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - fuses consecutive scales", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    // Complex types scale by complex factors, so the fused factor must be their
    // complex product, not the product of the real parts.
    using T        = TestType;
    auto    A      = create_random_tensor<T>("A", 3, 3);
    T const first  = pf<T>(2.0, 0.5);
    T const second = pf<T>(3.0, -1.5);
    auto    A_ref  = Tensor<T, 2>(A);
    for (size_t i = 0; i < A_ref.size(); i++) {
        A_ref.data()[i] = second * (first * A_ref.data()[i]);
    }

    cg::Graph graph("ewf_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(first, &A);
        cg::scale(second, &A);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(ewf.num_fused() == 1);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(A, A_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - three consecutive scales fuse to one", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    using T       = TestType;
    auto    A     = create_random_tensor<T>("A", 3, 3);
    T const s1    = pf<T>(2.0, 0.5);
    T const s2    = pf<T>(3.0, -1.5);
    T const s3    = pf<T>(4.0, 0.1);
    auto    A_ref = Tensor<T, 2>(A);
    for (size_t i = 0; i < A_ref.size(); i++) {
        A_ref.data()[i] = s3 * (s2 * (s1 * A_ref.data()[i]));
    }

    cg::Graph graph("ewf_triple");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(s1, &A);
        cg::scale(s2, &A);
        cg::scale(s3, &A);
    }

    REQUIRE(graph.num_nodes() == 3);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();

    CHECK(modified);
    CHECK(ewf.num_fused() == 2);
    CHECK(graph.num_nodes() == 1);

    graph.execute();
    require_close(A, A_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - no fusion for different tensors", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 3);
    auto B  = create_random_tensor<T>("B", 3, 3);

    cg::Graph graph("ewf_no_fuse");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(pf<T>(2.0, 0.5), &A);
        cg::scale(pf<T>(3.0, -1.5), &B);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE_FALSE(modified);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - scale separated by einsum does not fuse", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 3, 3);
    auto B  = create_random_tensor<T>("B", 3, 3);

    cg::Graph graph("ewf_barrier");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(pf<T>(2.0, 0.5), &A);
        cg::einsum("ik;kj->ij", T{0}, &A, T{1}, A, B); // NOLINT
        cg::scale(pf<T>(3.0, -1.5), &A);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();

    CHECK_FALSE(modified);
    CHECK(ewf.num_fused() == 0);
    CHECK(graph.num_nodes() == 3);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - fuses consecutive rank-3 scales", "[ComputeGraph][Passes][HigherRank]",
                        testing::AllScalarTypes) {
    using T        = TestType;
    auto    A      = create_random_tensor<T>("A", 4, 3, 5);
    T const first  = pf<T>(2.0, 0.5);
    T const second = pf<T>(3.0, -1.5);
    auto    A_ref  = Tensor<T, 3>(A);
    for (size_t i = 0; i < A_ref.size(); i++) {
        A_ref.data()[i] = second * (first * A_ref.data()[i]);
    }

    cg::Graph graph("ewf_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(first, &A);
        cg::scale(second, &A);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(ewf.num_fused() == 1);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(A, A_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - fused output feeding a downstream consumer stays correct", "[ComputeGraph][Passes]",
                        testing::AllScalarTypes) {
    // Consumer-bearing topology (node-position hazard): the fused node's POSITION
    // matters because a later gemm reads the scaled tensor. The fused
    // replacement must stay ahead of that reader.
    using T        = TestType;
    auto    A      = create_random_tensor<T>("A", 4, 4);
    auto    C      = create_random_tensor<T>("C", 4, 4);
    auto    D      = create_zero_tensor<T>("D", 4, 4);
    T const first  = pf<T>(2.0, 0.5);
    T const second = pf<T>(3.0, -1.5);

    auto C_ref = Tensor<T, 2>(C);
    for (size_t i = 0; i < C_ref.size(); i++) {
        C_ref.data()[i] = second * (first * C_ref.data()[i]);
    }

    cg::Graph graph("ewf_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(first, &C);
        cg::scale(second, &C);
        cg::gemm<false, false>(T{1}, C, A, T{0}, &D); // consumer of the fused scales
    }
    REQUIRE(graph.num_nodes() == 3);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 2);

    graph.execute();

    auto D_ref = create_zero_tensor<T>("D_ref", 4, 4);
    reference_einsum("ij <- ik ; kj", &D_ref, C_ref, A);
    require_close(C, C_ref);
    require_close(D, D_ref);
}

// ═══════════════════════════════════════════════════════════════════════════════
// axpby chains
// ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - fuses consecutive axpby on the same pair", "[ComputeGraph][Passes][Axpby]",
                        testing::AllScalarTypes) {
    // Y = a1*X + b1*Y then Y = a2*X + b2*Y is Y = (a2 + b2*a1)*X + (b2*b1)*Y.
    // axpby reads its scalars from live shared params, so the fused node is
    // ONE sweep over Y with new scalars, not two executors called in turn.
    // Complex scalars make the composition's cross terms visible.
    using T    = TestType;
    auto    X  = create_random_tensor<T>("X", 4, 5);
    auto    Y  = create_random_tensor<T>("Y", 4, 5);
    T const a1 = pf<T>(2.0, 0.5), b1 = pf<T>(3.0, -0.25);
    T const a2 = pf<T>(5.0, -1.0), b2 = pf<T>(7.0, 0.75);

    auto Y_ref = Tensor<T, 2>("Yref", 4, 5);
    for (size_t i = 0; i < Y_ref.size(); i++) {
        T const after_first = a1 * X.data()[i] + b1 * Y.data()[i];
        Y_ref.data()[i]     = a2 * X.data()[i] + b2 * after_first;
    }

    cg::Graph graph("ewf_axpby");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(a1, X, b1, &Y);
        cg::axpby(a2, X, b2, &Y);
    }
    REQUIRE(graph.num_nodes() == 2);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    CHECK(ewf.num_fused() == 1);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(Y, Y_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - three consecutive axpby fuse to one", "[ComputeGraph][Passes][Axpby]",
                        testing::AllScalarTypes) {
    using T    = TestType;
    auto    X  = create_random_tensor<T>("X", 4, 5);
    auto    Y  = create_random_tensor<T>("Y", 4, 5);
    T const a1 = pf<T>(2.0, 0.5), b1 = pf<T>(3.0, -0.25);
    T const a2 = pf<T>(5.0, -1.0), b2 = pf<T>(7.0, 0.75);
    T const a3 = pf<T>(0.5, 0.125), b3 = pf<T>(0.25, -0.5);

    auto Y_ref = Tensor<T, 2>("Yref", 4, 5);
    for (size_t i = 0; i < Y_ref.size(); i++) {
        T acc           = Y.data()[i];
        acc             = a1 * X.data()[i] + b1 * acc;
        acc             = a2 * X.data()[i] + b2 * acc;
        acc             = a3 * X.data()[i] + b3 * acc;
        Y_ref.data()[i] = acc;
    }

    cg::Graph graph("ewf_axpby3");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(a1, X, b1, &Y);
        cg::axpby(a2, X, b2, &Y);
        cg::axpby(a3, X, b3, &Y);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    CHECK(ewf.num_fused() == 2);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(Y, Y_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - axpby on a different source does not fuse", "[ComputeGraph][Passes][Axpby]",
                        testing::AllScalarTypes) {
    // Composing needs the SAME X: Y = a1*X1 + b1*Y then Y = a2*X2 + b2*Y is a
    // three-operand update, which no single axpby expresses.
    using T    = TestType;
    auto    X1 = create_random_tensor<T>("X1", 4, 5);
    auto    X2 = create_random_tensor<T>("X2", 4, 5);
    auto    Y  = create_random_tensor<T>("Y", 4, 5);
    T const a1 = pf<T>(2.0, 0.5), b1 = pf<T>(3.0, -0.25);
    T const a2 = pf<T>(5.0, -1.0), b2 = pf<T>(7.0, 0.75);

    auto Y_ref = Tensor<T, 2>("Yref", 4, 5);
    for (size_t i = 0; i < Y_ref.size(); i++) {
        T const after_first = a1 * X1.data()[i] + b1 * Y.data()[i];
        Y_ref.data()[i]     = a2 * X2.data()[i] + b2 * after_first;
    }

    cg::Graph graph("ewf_axpby_diff_src");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(a1, X1, b1, &Y);
        cg::axpby(a2, X2, b2, &Y);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 2);

    graph.execute();
    require_close(Y, Y_ref);
}

TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - a fused axpby that drops Y stops listing it as an input", "[ComputeGraph][Passes][Axpby]",
                        testing::AllScalarTypes) {
    // b2*b1 == 0 means the composed op overwrites Y, so Y must leave the input
    // list - liveness passes read that to decide what is still needed.
    using T    = TestType;
    auto    X  = create_random_tensor<T>("X", 4, 5);
    auto    Y  = create_random_tensor<T>("Y", 4, 5);
    T const a1 = pf<T>(2.0, 0.5), b1 = pf<T>(3.0, -0.25);
    T const a2 = pf<T>(5.0, -1.0);

    // beta 0 on the second: the earlier value is discarded, so Y = a2*X.
    auto Y_ref = Tensor<T, 2>("Yref", 4, 5);
    for (size_t i = 0; i < Y_ref.size(); i++) {
        Y_ref.data()[i] = a2 * X.data()[i];
    }

    cg::Graph graph("ewf_axpby_overwrite");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(a1, X, b1, &Y);
        cg::axpby(a2, X, T{0}, &Y); // beta 0: the earlier value is discarded
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 1);

    auto const &fused = graph.nodes()[0];
    CHECK(fused.inputs.size() == 1); // Y no longer read
    auto const *desc = fused.op_data.get_if<cg::AxpbyDescriptor>();
    REQUIRE(desc != nullptr);
    CHECK(einsums::compute_graph::is_zero(desc->beta));

    graph.execute();
    require_close(Y, Y_ref);
}

// Two `Y += a*X` accumulations compose into one sweep. Both were invisible to
// this pass while cg::axpy recorded its own opaque node kind, so a chain of
// the most natural spelling of an accumulation fused into nothing.
TEMPLATE_LIST_TEST_CASE("ElementWiseFusion - fuses a pair of axpy accumulations", "[ComputeGraph][Passes]", testing::AllScalarTypes) {
    using T        = TestType;
    auto    X      = create_random_tensor<T>("X", 6, 4);
    auto    Y      = create_random_tensor<T>("Y", 6, 4);
    T const first  = pf<T>(2.0, 0.5);
    T const second = pf<T>(3.0, -1.5);

    auto Y_ref = Tensor<T, 2>(Y);
    for (size_t i = 0; i < Y_ref.size(); i++) {
        Y_ref.data()[i] += first * X.data()[i];
        Y_ref.data()[i] += second * X.data()[i];
    }

    cg::Graph graph("fuse_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(first, X, &Y);
        cg::axpy(second, X, &Y);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::ElementWiseFusion>();

    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(Y, Y_ref);
}
