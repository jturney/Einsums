//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// AxpbyDescriptor: axpby (Y = alpha*X + beta*Y) now carries a descriptor so
// passes can read its scalars, with a shared-params block the executor reads on
// every call (single source of truth - a mutation through the descriptor takes
// effect on replay, avoiding the snapshot/executor desync).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <limits>
#include <variant>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// The single Axpby node's descriptor (non-const, so tests can mutate params).
cg::AxpbyDescriptor *find_axpby_desc(cg::Graph &graph) {
    for (auto &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Axpby) {
            if (auto *d = node.op_data.get_if<cg::AxpbyDescriptor>()) {
                return d;
            }
        }
    }
    return nullptr;
}

size_t axpby_input_count(cg::Graph &graph) {
    for (auto &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Axpby) {
            return node.inputs.size();
        }
    }
    return 0;
}

/// Within 100 epsilon of @p want, relative to it and absolute near zero, for any element type.
template <typename T>
bool near(T got, T want) {
    return std::abs(got - want) <= 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon() * (1.0 + std::abs(want));
}

} // namespace

TEST_CASE("AxpbyDescriptor - snapshot exposes alpha/beta", "[ComputeGraph][Axpby][descriptor]") {
    auto X = create_random_tensor<double>("X", 4);
    auto Y = create_zero_tensor<double>("Y", 4);

    cg::Graph graph("axpby-desc");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(2.0, X, 3.0, &Y);
    }

    auto *d = find_axpby_desc(graph);
    REQUIRE(d != nullptr);
    CHECK(cg::as<double>(d->alpha) == 2.0);
    CHECK(cg::as<double>(d->beta) == 3.0);
    REQUIRE(d->params != nullptr);
    CHECK(cg::as<double>(d->params->alpha) == 2.0);
    CHECK(cg::as<double>(d->params->beta) == 3.0);
}

TEMPLATE_LIST_TEST_CASE("AxpbyDescriptor - executor reads live params (single source of truth)", "[ComputeGraph][Axpby][descriptor]",
                        testing::AllScalarTypes) {
    using T       = TestType;
    T const alpha = testing::prefactor<T>(2.0, 0.5);
    T const beta  = testing::prefactor<T>(3.0, -1.0);
    auto    X     = create_random_tensor<T>("X", 4);
    auto    Y     = create_random_tensor<T>("Y", 4);

    std::vector<T> x0(4);
    std::vector<T> y0(4);
    for (size_t i = 0; i < 4; ++i) {
        x0[i] = X(i);
        y0[i] = Y(i);
    }

    cg::Graph graph("axpby-live");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(alpha, X, beta, &Y); // Y = alpha*X + beta*Y
    }

    graph.execute();
    for (size_t i = 0; i < 4; ++i) {
        CHECK(near<T>(Y(i), alpha * x0[i] + beta * y0[i]));
    }

    // Fold beta -> 0 through the shared params. If the executor still used the
    // baked value it would keep accumulating (Y += beta*Y); reading live params it
    // overwrites: Y = alpha*X.
    auto *d = find_axpby_desc(graph);
    REQUIRE(d != nullptr);
    d->params->beta = cg::PrefactorScalar{0.0};

    graph.execute();
    for (size_t i = 0; i < 4; ++i) {
        CHECK(near<T>(Y(i), alpha * x0[i]));
    }
    // The at-capture snapshot is unchanged by the params mutation.
    CHECK(cg::as<T>(d->beta) == beta);
}

TEST_CASE("AxpbyDescriptor - inputs encode accumulate vs overwrite", "[ComputeGraph][Axpby][descriptor]") {
    auto X = create_random_tensor<double>("X", 4);
    auto Y = create_zero_tensor<double>("Y", 4);

    cg::Graph accumulate("axpby-acc");
    {
        cg::CaptureGuard const guard(accumulate);
        cg::axpby(1.0, X, 1.0, &Y); // beta != 0: reads its destination
    }
    CHECK(axpby_input_count(accumulate) == 2);

    cg::Graph overwrite("axpby-ovr");
    {
        cg::CaptureGuard const guard(overwrite);
        cg::axpby(1.0, X, 0.0, &Y); // beta == 0: pure overwrite
    }
    CHECK(axpby_input_count(overwrite) == 1);
}
