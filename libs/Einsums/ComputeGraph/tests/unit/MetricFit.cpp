//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/MetricFitFactorization.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <memory>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

std::size_t const nbf  = 5;
std::size_t const naux = 7;

/// A symmetric positive-definite metric, built as R R^T + a shift so nothing is singular.
Tensor<double, 2> make_metric() {
    auto seed = create_random_tensor<double>("seed", naux, naux);
    auto out  = create_zero_tensor<double>("J", naux, naux);
    for (std::size_t p = 0; p < naux; ++p) {
        for (std::size_t q = 0; q < naux; ++q) {
            double sum = 0.0;
            for (std::size_t r = 0; r < naux; ++r) {
                sum += seed(p, r) * seed(q, r);
            }
            out(p, q) = sum + (p == q ? 2.0 : 0.0);
        }
    }
    return out;
}

/// The tensor a metric fit is EXACT for: T[m,n,p,q] = sum_PQ R[P,m,n] Jinv[P,Q] R[Q,p,q].
///
/// Built by inverting the metric directly rather than through the square root, so the
/// reference owes nothing to the code path under test: if the provider's J^{-1/2} were wrong,
/// B B would not reproduce this.
Tensor<double, 4> exact_fit(Tensor<double, 3> const &R, Tensor<double, 2> const &J) {
    auto inverse = Tensor<double, 2>("Jinv", naux, naux);
    inverse      = J;
    linear_algebra::invert(&inverse);

    auto out = create_zero_tensor<double>("T", nbf, nbf, nbf, nbf);
    for (std::size_t m = 0; m < nbf; ++m) {
        for (std::size_t n = 0; n < nbf; ++n) {
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    double sum = 0.0;
                    for (std::size_t a = 0; a < naux; ++a) {
                        for (std::size_t b = 0; b < naux; ++b) {
                            sum += R(a, m, n) * inverse(a, b) * R(b, p, q);
                        }
                    }
                    out(m, n, p, q) = sum;
                }
            }
        }
    }
    return out;
}

} // namespace

TEST_CASE("MetricFit - an exactly fitted tensor is replaced and the contraction reproduces it",
          "[ComputeGraph][Factorization][MetricFit]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("D", nbf, nbf);

    // The reference: the four-index contraction, done directly.
    auto      C = create_zero_tensor<double>("C", nbf, nbf);
    cg::Graph reference("reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    reference.execute();
    auto const expected = C;

    auto      Cf = create_zero_tensor<double>("C", nbf, nbf);
    cg::Graph graph("fitted");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &Cf, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    auto provider = std::make_shared<cg::MetricFitFactorization>("eri", R, J);

    cg::FactorizationRegistry registry;
    registry.add(provider);

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);

    // T is exactly the metric-fitted product, so the measured error is rounding and nothing
    // else. This is the assertion that the fitting is the RIGHT arithmetic rather than merely
    // arithmetic of the right shape: a wrong inverse square root would show up here as a
    // number of order one.
    REQUIRE(provider->measured_error() < 1e-10);

    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();

    for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
            INFO("i=" << i << " j=" << j);
            REQUIRE(std::abs(Cf(i, j) - expected(i, j)) < 1e-9);
        }
    }

    // And the graph carries what it now computes, with the measured bound rather than a
    // number anyone asserted.
    REQUIRE(graph.approximations().size() == 1);
    REQUIRE(graph.approximations()[0].pass_name == "MetricFit");
    REQUIRE(graph.approximations()[0].bound == Catch::Approx(provider->measured_error()));
}

TEST_CASE("MetricFit - one tensor is fitted, not two", "[ComputeGraph][Factorization][MetricFit]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("D", nbf, nbf);
    auto C = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("shared_factor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    // The provider offers two factors under one name, which is one buffer. Two would fit the
    // same thing twice and store it twice, and at DF's real sizes B is the largest tensor in
    // the calculation.
    std::size_t fitted_tensors = 0;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name.find("MetricFit_B") != std::string::npos) {
            ++fitted_tensors;
        }
    }
    REQUIRE(fitted_tensors == 1);
}

TEST_CASE("MetricFit - a declared bound is recorded instead of a measured one", "[ComputeGraph][Factorization][MetricFit]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("D", nbf, nbf);
    auto C = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("declared");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    auto                      provider = std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-5);
    cg::FactorizationRegistry registry;
    registry.add(provider);
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    REQUIRE(provider->measured_error() < 0.0); // nothing was measured
    REQUIRE(graph.approximations()[0].bound == Catch::Approx(1e-5));
}

TEST_CASE("MetricFit - a tensor whose shape the fit cannot produce is declined", "[ComputeGraph][Factorization][MetricFit]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = create_random_tensor<double>("T", nbf, nbf, nbf, nbf + 1);
    auto D = create_random_tensor<double>("D", nbf, nbf + 1);
    auto C = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("wrong_shape");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());
}

TEST_CASE("MetricFit - a fitted graph cannot be saved yet, and the refusal names the eigendecomposition",
          "[ComputeGraph][Factorization][MetricFit][SaveLoad]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("D", nbf, nbf);
    auto C = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("saveable");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, T, D);
    }
    graph.annotate_tag(T, cg::ProvenanceTag{.name = "eri"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    // The gap, asserted rather than left to be discovered. The rewritten BODY is perfectly
    // saveable; the fitting is not, because a symmetric inverse square root needs an
    // eigendecomposition and Syev is not in the reconstructible set. So the one thing a
    // factorization exists to make reusable across problems is the one thing that cannot yet
    // be written to a file.
    //
    // When Syev joins the set this test goes red, which is the reminder to replace it with a
    // round-trip.
    auto const text = cg::save_graph_string(graph);
    REQUIRE_FALSE(text.has_value());
    REQUIRE_THAT(text.error().message, Catch::Matchers::ContainsSubstring("Syev"));
    REQUIRE_THAT(text.error().message, Catch::Matchers::ContainsSubstring("setup("));
}
