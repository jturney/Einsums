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

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-12));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);

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
    REQUIRE(graph.approximations()[0].bound == Catch::Approx(1e-12));
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
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-12));
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

TEST_CASE("MetricFit - a well-conditioned metric drops nothing, and a singular one says how much",
          "[ComputeGraph][Factorization][MetricFit]") {
    auto R = create_random_tensor<double>("R", naux, nbf, nbf);
    auto J = make_metric();
    auto T = exact_fit(R, J);
    auto D = create_random_tensor<double>("D", nbf, nbf);
    auto C = create_zero_tensor<double>("C", nbf, nbf);

    auto const build = [&](Tensor<double, 2> &metric, Tensor<double, 4> &tagged, Tensor<double, 2> &out) {
        auto graph = std::make_unique<cg::Graph>("conditioning");
        {
            cg::CaptureGuard const guard(*graph);
            cg::einsum("m,n,p,q ; p,q -> m,n", &out, tagged, D);
        }
        graph->annotate_tag(tagged, cg::ProvenanceTag{.name = "eri"});
        return graph;
    };

    std::string const key = cg::MetricFitFactorization::dropped_param_name("MetricFit", "T");

    SECTION("a metric with no null directions keeps every one") {
        auto                      graph = build(J, T, C);
        cg::FactorizationRegistry registry;
        registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-12));
        cg::passes::FactorizationPass factorization(registry);
        cg::PassManager               pm;
        pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
        REQUIRE(graph->apply(pm));

        auto defaults = cg::PassManager::create_default();
        graph->apply(defaults);
        graph->execute();

        REQUIRE(graph->params_ptr()->get(key) == 0);
    }

    SECTION("a metric with a null direction reports it, on every refit") {
        // A metric with one unambiguously NEGATIVE eigenvalue, spelled as a diagonal so there
        // is nothing to argue about: the guarded inverse square root has to throw that
        // direction away, and the fitted space is then smaller than the auxiliary set the
        // caller supplied. That is the case an asserted bound says nothing about, and nothing
        // else would tell them.
        //
        // Negative rather than merely tiny, deliberately, because the guard tests `x > 0` and
        // a near-singular metric whose smallest eigenvalue lands at +1e-17 slips through it
        // and produces an enormous 1/sqrt instead. Real orthogonalizations drop below a
        // THRESHOLD for exactly that reason; this one does not have one yet, and a test that
        // pretended otherwise would be testing a guard this code does not have.
        auto singular = create_zero_tensor<double>("J", naux, naux);
        for (std::size_t q = 0; q < naux; ++q) {
            singular(q, q) = 2.0;
        }
        singular(0, 0) = -1.0;
        auto tagged    = create_zero_tensor<double>("T", nbf, nbf, nbf, nbf);
        auto out       = create_zero_tensor<double>("C", nbf, nbf);

        auto                      graph = build(singular, tagged, out);
        cg::FactorizationRegistry registry;
        registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, singular, 1e-3));
        cg::passes::FactorizationPass factorization(registry);
        cg::PassManager               pm;
        pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
        REQUIRE(graph->apply(pm));

        auto defaults = cg::PassManager::create_default();
        graph->apply(defaults);
        graph->execute();

        REQUIRE(graph->params_ptr()->get(key) >= 1);
    }
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
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-12));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());
}

TEST_CASE("MetricFit - a fitted graph saves, loads, and refits for the problem it is bound to",
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
    registry.add(std::make_shared<cg::MetricFitFactorization>("eri", R, J, 1e-5));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    // Saved BEFORE any resource pass, which is the documented flow rather than a convenience:
    // a Materialize node carries an allocating closure and allocation is re-derived on load.
    // Nothing in the fitting blocks it now, the eigendecomposition included.
    REQUIRE(graph.serializability_report().empty());
    auto const text = cg::save_graph_string(graph);
    if (!text.has_value()) {
        UNSCOPED_INFO(text.error().message);
    }
    REQUIRE(text.has_value());

    // The four-index tensor is gone from the interface: nothing reads it any more, so a
    // caller of the loaded graph is not asked for the very thing the factorization exists to
    // avoid needing.
    auto loaded = cg::load_graph_string(*text);
    if (!loaded.has_value()) {
        UNSCOPED_INFO(loaded.error().message);
    }
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->manifest().find("T") == nullptr);

    // A fresh set of storage for the same problem. R and J are bindable because the fitting
    // reads them, which is what makes a refit per bound problem possible at all.
    auto R2 = create_zero_tensor<double>("R", naux, nbf, nbf);
    auto J2 = create_zero_tensor<double>("J", naux, naux);
    auto D2 = create_zero_tensor<double>("D", nbf, nbf);
    auto C2 = create_zero_tensor<double>("C", nbf, nbf);
    R2      = R;
    J2      = J;
    D2      = D;

    loaded->bind("R", R2, "J", J2, "D", D2, "C", C2);

    auto defaults = cg::PassManager::create_default();
    loaded->apply(defaults);
    loaded->execute();

    // The reference: the four-index contraction done directly, which the loaded graph never
    // forms and still reproduces.
    auto      expected = create_zero_tensor<double>("expected", nbf, nbf);
    cg::Graph reference("reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &expected, T, D);
    }
    reference.execute();

    for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
            INFO("i=" << i << " j=" << j);
            REQUIRE(std::abs(C2(i, j) - expected(i, j)) < 1e-9);
        }
    }

    // The accuracy statement travelled with the structure, because it says what the graph
    // computes and no machine could recover it from the node list.
    REQUIRE(loaded->approximations().size() == 1);
    REQUIRE(loaded->approximations()[0].pass_name == "MetricFit");
    REQUIRE(loaded->approximations()[0].bound == Catch::Approx(1e-5));
}
