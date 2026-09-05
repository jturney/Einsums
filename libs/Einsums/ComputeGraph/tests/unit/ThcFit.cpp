//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/ComputeGraph/ThcFactorization.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

std::size_t const nbf  = 4;
std::size_t const ngrd = 6; ///< At most nbf*(nbf+1)/2, so the fit's normal equations are solvable.
std::size_t const naux = 3;

/// One problem a grid fit is EXACT for, and the four-index tensor it produces.
///
/// The three-index tensor is BUILT from the grid rather than fitted to it: with
/// @c B[A,m,n] @c = @c sum_P @c C[A,P] @c X[m,P] @c X[n,P] the least-squares solution is @c C
/// itself, so @c Z comes out as @c C^T @c C and the chain reproduces the four-index tensor to
/// rounding. That is what lets this test compare against a dense contraction rather than
/// against a tolerance nobody derived.
struct Problem {
    Tensor<double, 2> collocation;
    Tensor<double, 2> weights;
    Tensor<double, 3> three_index;
    Tensor<double, 4> dense;
};

Problem make_problem() {
    Problem problem{.collocation = create_random_tensor<double>("X", nbf, ngrd),
                    .weights     = create_random_tensor<double>("C", naux, ngrd),
                    .three_index = create_zero_tensor<double>("B", naux, nbf, nbf),
                    .dense       = create_zero_tensor<double>("M", nbf, nbf, nbf, nbf)};

    for (std::size_t a = 0; a < naux; ++a) {
        for (std::size_t m = 0; m < nbf; ++m) {
            for (std::size_t n = 0; n < nbf; ++n) {
                double sum = 0.0;
                for (std::size_t p = 0; p < ngrd; ++p) {
                    sum += problem.weights(a, p) * problem.collocation(m, p) * problem.collocation(n, p);
                }
                problem.three_index(a, m, n) = sum;
            }
        }
    }
    for (std::size_t m = 0; m < nbf; ++m) {
        for (std::size_t n = 0; n < nbf; ++n) {
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    double sum = 0.0;
                    for (std::size_t a = 0; a < naux; ++a) {
                        sum += problem.three_index(a, m, n) * problem.three_index(a, p, q);
                    }
                    problem.dense(m, n, p, q) = sum;
                }
            }
        }
    }
    return problem;
}

} // namespace

TEST_CASE("ThcFit - a five-factor grid chain replaces the tensor and the contraction reproduces it", "[ComputeGraph][Factorization][Thc]") {
    Problem problem = make_problem();
    auto    T       = create_random_tensor<double>("T", nbf, nbf);
    auto    C       = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph reference("thc_reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, problem.dense, T);
    }
    reference.execute();
    auto const expected = C;

    auto      Cf = create_zero_tensor<double>("C", nbf, nbf);
    cg::Graph graph("thc_factorized");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &Cf, problem.dense, T);
    }
    graph.annotate_tag(problem.dense, cg::ProvenanceTag{.name = "eri"});
    // Annotated for its FAMILY, which is what a caller does when the size in hand is not the
    // size that matters. It is also what gives the collocation matrix's basis axis a symbol,
    // since a factor is annotated only when every one of its axes resolves.
    graph.annotate_dims(problem.dense, {"nbf", "nbf", "nbf", "nbf"});
    // The grid is a space of the GRAPH, which is why a caller registers it: propose() is handed
    // a const graph precisely so a provider cannot make one.
    REQUIRE(cg::ThcFactorization::register_grid_space(graph).valid());
    // And the basis is a space too. Annotated here rather than left anonymous because the
    // pass's self-check compares its reported cost against the flops of the nodes it emitted,
    // and a letter that resolves to a space in one derivation and anonymously in the other is
    // two names for one extent: the check would fire on a rewrite that is perfectly correct.
    cg::SpaceId const ao = graph.space_registry().register_space(cg::make_index_space("ao", "b", 0.0, cg::GrowthClass::linear(), "nbf"));
    graph.annotate_spaces(problem.dense, {ao, ao, ao, ao});
    graph.annotate_spaces(T, {ao, ao});
    graph.annotate_spaces(Cf, {ao, ao});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::ThcFactorization>("eri", problem.three_index, problem.collocation, 1e-8));

    cg::passes::FactorizationPass factorization(registry);
    // Deliberately WITHOUT set_verify_costs, and the reason is worth stating rather than
    // leaving as an omission. That check compares the cost the pass computes from the algebra
    // against the flops of the nodes the lowering emitted, and the two read their letters'
    // spaces from different snapshots: the raised algebra takes them from the capture-time
    // letter map, which holds nothing for a program annotated afterwards, while the emitted
    // nodes are built after the annotation and see it. A letter that resolves to a space in
    // one derivation and anonymously in the other is two names for one extent, so the check
    // fires on a rewrite that is perfectly correct. The chain and cone cases in
    // Factorization.cpp keep it on, over programs nothing annotates, which is where it means
    // something.
    cg::PassManager factorization_manager;
    factorization_manager.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(factorization_manager));
    REQUIRE(factorization.num_factorized() == 1);

    // Six leaves, five binary contractions, and none of them a rank-four tensor: rebuilding the
    // integral is the most expensive tree over those leaves and the search never picks it.
    std::size_t contractions = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
    }
    REQUIRE(contractions == 5);

    std::size_t collocations = 0;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == "Thc_X") {
            ++collocations;
            // The grid axis carries the space and its symbol, which is what makes a saved graph
            // rebindable at a grid of another size.
            REQUIRE(handle.dim_symbols.size() == 2);
            REQUIRE(handle.dim_symbols[1] == cg::ThcFactorization::grid_dim_symbol());
        }
        if (handle.name.rfind("Thc_M_x", 0) == 0) {
            REQUIRE(handle.rank < 4);
        }
    }
    // ONE collocation matrix for four mentions of it. Creating one per mention would fit and
    // store the same thing four times.
    REQUIRE(collocations == 1);

    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();

    for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
            INFO("i=" << i << " j=" << j);
            REQUIRE(std::abs(Cf(i, j) - expected(i, j)) < 1e-9);
        }
    }

    REQUIRE(graph.approximations().size() == 1);
    REQUIRE(graph.approximations()[0].pass_name == "Thc");

    // What the fit is worth, measured rather than asserted, on this bind. The problem was built
    // so the fit is exact, so the residual is rounding.
    double const residual       = graph.params_ptr()->get(cg::ThcFactorization::residual_param_name("Thc", "M"));
    double const reference_norm = graph.params_ptr()->get(cg::ThcFactorization::reference_param_name("Thc", "M"));
    REQUIRE(reference_norm > 0.0);
    REQUIRE(std::sqrt(residual / reference_norm) < 1e-10);
}

TEST_CASE("ThcFit - a tensor whose basis the grid does not span is declined", "[ComputeGraph][Factorization][Thc]") {
    Problem problem = make_problem();
    // A rank-4 tensor over a DIFFERENT number of basis functions. The provider cannot produce
    // it and says so rather than offering a chain whose extents do not fit.
    auto other = create_random_tensor<double>("M2", nbf + 1, nbf + 1, nbf + 1, nbf + 1);
    auto T     = create_random_tensor<double>("T", nbf + 1, nbf + 1);
    auto C     = create_zero_tensor<double>("C", nbf + 1, nbf + 1);

    cg::Graph graph("thc_mismatched");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, other, T);
    }
    graph.annotate_tag(other, cg::ProvenanceTag{.name = "eri"});
    cg::ThcFactorization::register_grid_space(graph);

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::ThcFactorization>("eri", problem.three_index, problem.collocation, 1e-8));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());
    bool named_the_basis = false;
    for (auto const &[reason, count] : factorization.skip_reasons()) {
        if (reason.find("a provider declined") != std::string::npos) {
            named_the_basis = true;
        }
    }
    REQUIRE(named_the_basis);
}

TEST_CASE("ThcFit - a threshold that could not mean anything is refused at construction", "[ComputeGraph][Factorization][Thc]") {
    Problem problem = make_problem();
    // Zero is spellable, because it is exactly the bare positivity guard and a caller
    // reproducing an earlier result has to be able to ask for it. A negative one is not.
    REQUIRE_NOTHROW(cg::ThcFactorization("eri", problem.three_index, problem.collocation, 1e-8, 0.0));
    REQUIRE_THROWS_AS(cg::ThcFactorization("eri", problem.three_index, problem.collocation, 1e-8, -1e-12), std::invalid_argument);
    REQUIRE_THROWS_AS(cg::ThcFactorization("eri", problem.three_index, problem.collocation, -1.0), std::invalid_argument);
}

TEST_CASE("ThcFit - a fitted graph saves, loads, rebinds and refits", "[ComputeGraph][Factorization][Thc][SaveLoad]") {
    Problem problem = make_problem();
    auto    T       = create_random_tensor<double>("T", nbf, nbf);
    auto    C       = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("thc_saveable");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, problem.dense, T);
    }
    graph.annotate_tag(problem.dense, cg::ProvenanceTag{.name = "eri"});
    graph.annotate_dims(problem.dense, {"nbf", "nbf", "nbf", "nbf"});
    cg::ThcFactorization::register_grid_space(graph);

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<cg::ThcFactorization>("eri", problem.three_index, problem.collocation, 1e-8));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    // Saved BEFORE any resource pass, which is the documented flow: a Materialize node carries
    // an allocating closure and allocation is re-derived on load.
    REQUIRE(graph.serializability_report().empty());
    auto const text = cg::save_graph_string(graph);
    if (!text.has_value()) {
        UNSCOPED_INFO(text.error().message);
    }
    REQUIRE(text.has_value());

    auto loaded = cg::load_graph_string(*text);
    if (!loaded.has_value()) {
        UNSCOPED_INFO(loaded.error().message);
    }
    REQUIRE(loaded.has_value());
    // The four-index tensor is gone from the interface, and the collocation matrix is in it:
    // a caller of the loaded graph is asked for the grid rather than for the very tensor the
    // factorization exists to avoid needing.
    REQUIRE(loaded->manifest().find("M") == nullptr);
    REQUIRE(loaded->manifest().find("X") != nullptr);

    auto X2 = create_zero_tensor<double>("X", nbf, ngrd);
    auto B2 = create_zero_tensor<double>("B", naux, nbf, nbf);
    auto T2 = create_zero_tensor<double>("T", nbf, nbf);
    auto C2 = create_zero_tensor<double>("C", nbf, nbf);
    X2      = problem.collocation;
    B2      = problem.three_index;
    T2      = T;

    loaded->bind("X", X2, "B", B2, "T", T2, "C", C2);
    auto defaults = cg::PassManager::create_default();
    loaded->apply(defaults);
    loaded->execute();

    auto      expected = create_zero_tensor<double>("expected", nbf, nbf);
    cg::Graph reference("thc_reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &expected, problem.dense, T);
    }
    reference.execute();

    for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
            INFO("i=" << i << " j=" << j);
            REQUIRE(std::abs(C2(i, j) - expected(i, j)) < 1e-9);
        }
    }

    REQUIRE(loaded->approximations().size() == 1);
    REQUIRE(loaded->approximations()[0].pass_name == "Thc");
    REQUIRE(loaded->approximations()[0].bound == Catch::Approx(1e-8));

    // Refitted on the bind rather than replayed: the residual is a property of what is bound
    // now, and a loaded graph that had remembered one would report the capture's.
    double const residual       = loaded->params_ptr()->get(cg::ThcFactorization::residual_param_name("Thc", "M"));
    double const reference_norm = loaded->params_ptr()->get(cg::ThcFactorization::reference_param_name("Thc", "M"));
    REQUIRE(reference_norm > 0.0);
    REQUIRE(std::sqrt(residual / reference_norm) < 1e-10);
}

// ── The amplitude mode: a fit OF the tagged tensor ───────────────────────────

TEST_CASE("ThcFit - an amplitude is fitted from itself and refitted at every update", "[ComputeGraph][Factorization][Thc][Amplitude]") {
    // The integral fit reads a three-index tensor the caller hands over and CLAIMS it
    // approximates the tagged one; its factors do not move when the tagged tensor does. An
    // amplitude fit projects the tagged tensor onto the grid basis, so its factors are a function
    // of that tensor and have to be redone wherever it is written. That is what lets a tensor a
    // solver rewrites every iteration be factorized at all.
    std::size_t const iterations = 3;

    // A grid the amplitude lies exactly in the span of: T[m,n,p,q] = sum_PQ X[m,P] X[n,P] Z[P,Q]
    // X[p,Q] X[q,Q] by construction, so the substitution is exact and the case compares against
    // an un-rewritten replay rather than against a tolerance nobody derived.
    auto X              = create_random_tensor<double>("X", nbf, ngrd);
    auto W              = create_random_tensor<double>("W", ngrd, ngrd);
    auto seed_amplitude = [&](Tensor<double, 4> &into) {
        for (std::size_t m = 0; m < nbf; ++m) {
            for (std::size_t n = 0; n < nbf; ++n) {
                for (std::size_t p = 0; p < nbf; ++p) {
                    for (std::size_t q = 0; q < nbf; ++q) {
                        double sum = 0.0;
                        for (std::size_t a = 0; a < ngrd; ++a) {
                            for (std::size_t b = 0; b < ngrd; ++b) {
                                sum += X(m, a) * X(n, a) * W(a, b) * X(p, b) * X(q, b);
                            }
                        }
                        into(m, n, p, q) = sum;
                    }
                }
            }
        }
    };

    auto ones = create_zero_tensor<double>("ones", nbf, nbf, nbf, nbf);
    for (std::size_t m = 0; m < nbf; ++m) {
        for (std::size_t n = 0; n < nbf; ++n) {
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    ones(m, n, p, q) = 1.0;
                }
            }
        }
    }
    auto operand = create_random_tensor<double>("T", nbf, nbf);

    // One iteration: a contraction over the amplitude, and an update that divides a residual by
    // a denominator into it. The residual is a scaling of the amplitude, so it stays in the
    // grid's span and the fit stays exact; what is under test is the refit, not a solver.
    auto capture = [&](cg::Graph &graph, Tensor<double, 4> &amplitude, Tensor<double, 2> &result,
                       Tensor<double, 4> &residual) -> cg::Graph & {
        cg::Graph &body = graph.add_loop("amplitude_iteration", iterations, [iterations](std::size_t it) { return it + 1 < iterations; });
        cg::CaptureGuard const guard(body);
        cg::einsum("m,n,p,q ; p,q -> m,n", &result, amplitude, operand);
        cg::direct_product(0.5, amplitude, ones, 0.0, &residual);
        cg::direct_division(1.0, residual, ones, 0.0, &amplitude);
        return body;
    };

    auto reference_amplitude = create_zero_tensor<double>("Tamp", nbf, nbf, nbf, nbf);
    auto reference_result    = create_zero_tensor<double>("C", nbf, nbf);
    auto reference_residual  = create_zero_tensor<double>("R", nbf, nbf, nbf, nbf);
    seed_amplitude(reference_amplitude);
    cg::Graph reference("amplitude_reference");
    capture(reference, reference_amplitude, reference_result, reference_residual);
    auto reference_passes = cg::PassManager::create_default();
    reference.apply(reference_passes);
    reference.execute();

    auto amplitude = create_zero_tensor<double>("Tamp", nbf, nbf, nbf, nbf);
    auto result    = create_zero_tensor<double>("C", nbf, nbf);
    auto residual  = create_zero_tensor<double>("R", nbf, nbf, nbf, nbf);
    seed_amplitude(amplitude);
    cg::Graph  graph("amplitude_factorized");
    cg::Graph &body = capture(graph, amplitude, result, residual);
    graph.annotate_tag(amplitude, cg::ProvenanceTag{.name = "amplitude"});
    cg::ThcFactorization::register_grid_space(graph);

    auto fit_residual  = create_zero_tensor<double>("thc_residual", 1);
    auto fit_reference = create_zero_tensor<double>("thc_reference", 1);

    auto provider = cg::ThcFactorization::for_amplitude("amplitude", RuntimeTensorView<double>{amplitude},
                                                        std::vector<RuntimeTensorView<double>>{RuntimeTensorView<double>{X}}, 1e-6, 1e-12);
    provider->report_residual_into(fit_residual, fit_reference);

    cg::FactorizationRegistry registry;
    registry.add(provider);
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::make_shared<cg::passes::ProvenancePropagation>());
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    pm.set_verbosity(3);
    bool const fired = graph.apply(pm);
    INFO(pm.explain());
    REQUIRE(fired);
    CHECK(factorization.num_factorized() == 1);

    // One record, MEASURED, naming the caller's own tensor.
    REQUIRE(graph.approximations().size() == 1);
    CHECK(graph.approximations().front().origin == cg::ApproximationOrigin::Measured);
    CHECK(graph.approximations().front().measurement == fit_residual.name());

    // The fitting is in the parent's setup, which is what the first iteration reads, and again in
    // the body, which is what every iteration after it reads.
    std::size_t setups = 0;
    for (auto const &node : graph.nodes()) {
        if (std::get_if<cg::SetupDescriptor>(&node.op_data) != nullptr) {
            ++setups;
        }
    }
    CHECK(setups == 1);
    CHECK(body.num_nodes() > 3); // the refit is in there beside the three captured statements

    auto graph_passes = cg::PassManager::create_default();
    graph.apply(graph_passes);
    graph.execute();

    for (std::size_t m = 0; m < nbf; ++m) {
        for (std::size_t n = 0; n < nbf; ++n) {
            CHECK_THAT(result(m, n), Catch::Matchers::WithinRel(reference_result(m, n), 1.0e-8));
        }
    }

    // And the measurement is a real one: the fit is exact here, so the residual is rounding
    // against a reference norm that is not.
    CHECK(fit_reference(0) > 0.0);
    CHECK(std::sqrt(fit_residual(0) / fit_reference(0)) < 1e-8);
}

TEST_CASE("ThcFit - an amplitude provider handed a different tensor declines", "[ComputeGraph][Factorization][Thc][Amplitude]") {
    // The caller's claim that the tensor they tagged is the tensor they handed over is CHECKED,
    // because a fit OF a tensor is only sound if it reads the tensor the pass substitutes away.
    auto X     = create_random_tensor<double>("X", nbf, ngrd);
    auto mine  = create_random_tensor<double>("mine", nbf, nbf, nbf, nbf);
    auto other = create_random_tensor<double>("other", nbf, nbf, nbf, nbf);
    auto T     = create_random_tensor<double>("T", nbf, nbf);
    auto C     = create_zero_tensor<double>("C", nbf, nbf);

    cg::Graph graph("amplitude_mismatched");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, mine, T);
    }
    graph.annotate_tag(mine, cg::ProvenanceTag{.name = "amplitude"});
    cg::ThcFactorization::register_grid_space(graph);

    auto provider = cg::ThcFactorization::for_amplitude("amplitude", RuntimeTensorView<double>{other},
                                                        std::vector<RuntimeTensorView<double>>{RuntimeTensorView<double>{X}});
    cg::FactorizationRegistry registry;
    registry.add(provider);
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    pm.set_verbosity(3);
    CHECK_FALSE(graph.apply(pm));
    CHECK(factorization.num_factorized() == 0);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("provider declined") != std::string::npos);
}
