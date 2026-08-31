//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A provider that offers nothing, which is all these cases need: the registry's job is to
/// hold providers and hand back the ones claiming a tag, and it does that without ever
/// calling propose.
class NamedProvider : public cg::FactorizationProvider {
  public:
    NamedProvider(std::string name, std::string tag) : _name(std::move(name)), _tag(std::move(tag)) {}

    [[nodiscard]] std::string name() const override { return _name; }
    [[nodiscard]] std::string tag() const override { return _tag; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const & /*graph*/,
                                                                       cg::TensorId /*tensor*/) const override {
        // QUALIFIED, and it has to be. This file says `using namespace einsums;` at global
        // scope, so an unqualified `unexpected` here is a name two using-directives can both
        // supply: ours and the standard library's. libstdc++ and libc++ resolve it in our
        // favour and the MSVC STL reports it as ambiguous, which makes the unqualified
        // spelling a portability trap rather than a style choice. The library's own sources
        // are safe unqualified because they sit INSIDE einsums, where the enclosing namespace
        // hides anything a using-directive brings in.
        return einsums::unexpected(std::string{"this provider offers nothing"});
    }

  private:
    std::string _name;
    std::string _tag;
};

std::shared_ptr<cg::FactorizationProvider> provider(std::string name, std::string tag) {
    return std::make_shared<NamedProvider>(std::move(name), std::move(tag));
}

} // namespace

TEST_CASE("Factorization - a registry hands back every provider claiming a tag, in registration order", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.claims("eri"));

    registry.add(provider("AutoDF", "eri"));
    registry.add(provider("AutoCholesky", "eri"));
    registry.add(provider("AutoTHC", "eri_df_b"));

    REQUIRE(registry.size() == 3);
    REQUIRE(registry.claims("eri"));
    REQUIRE(registry.claims("eri_df_b"));
    REQUIRE_FALSE(registry.claims("fock"));

    auto const eri = registry.for_tag("eri");
    REQUIRE(eri.size() == 2);
    // Registration order, not some container's iteration order: a pass that costs several
    // offers and finds two of them equal breaks the tie somehow, and a tie broken by
    // iteration order makes the optimizer's output depend on nothing anyone can see.
    REQUIRE(eri[0]->name() == "AutoDF");
    REQUIRE(eri[1]->name() == "AutoCholesky");

    REQUIRE(registry.for_tag("fock").empty());
}

TEST_CASE("Factorization - two providers cannot share a name", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    registry.add(provider("AutoDF", "eri"));

    // Not a tidiness rule. The name is what the approximation record carries and what the
    // report names, and two providers answering to one of them makes which ran a function of
    // registration order.
    REQUIRE_THROWS_WITH(registry.add(provider("AutoDF", "coulomb_metric")),
                        Catch::Matchers::ContainsSubstring("already registered") && Catch::Matchers::ContainsSubstring("'eri'"));
    REQUIRE(registry.size() == 1);
}

TEST_CASE("Factorization - a provider can be removed and the registry emptied", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    registry.add(provider("AutoDF", "eri"));
    registry.add(provider("AutoTHC", "eri"));

    REQUIRE(registry.remove("AutoDF"));
    REQUIRE_FALSE(registry.remove("AutoDF"));
    REQUIRE(registry.size() == 1);
    REQUIRE(registry.for_tag("eri").front()->name() == "AutoTHC");

    registry.clear();
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.claims("eri"));
}

TEST_CASE("Factorization - a null provider is refused rather than stored", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    REQUIRE_THROWS_WITH(registry.add(nullptr), Catch::Matchers::ContainsSubstring("null"));
    REQUIRE(registry.size() == 0);
}

TEST_CASE("Factorization - the process registry is one object", "[ComputeGraph][Factorization]") {
    auto             &registry = cg::global_factorization_registry();
    std::size_t const before   = registry.size();

    registry.add(provider("TestOnlyProvider", "test_only_tag"));
    REQUIRE(cg::global_factorization_registry().size() == before + 1);
    REQUIRE(cg::global_factorization_registry().claims("test_only_tag"));

    // Put it back. A process-global registry a test adds to and leaves is a test that changes
    // what every later one sees.
    REQUIRE(registry.remove("TestOnlyProvider"));
    REQUIRE(cg::global_factorization_registry().size() == before);
}

// ── The pass ───────────────────────────────────────────────────────────────

namespace {

/// A low-rank split that is EXACT by construction: the test builds the tagged tensor as the
/// very product this provider claims it is, so the only difference between the two forms is
/// the order the sum is taken in.
///
/// That is deliberately the re-associating tier rather than the bitwise one. It is what a real
/// factorization is - DF's saving comes from contracting the factors in the other order, and
/// re-association is exactly what changes a floating-point result - so a provider test that
/// demanded bit equality would be testing something no provider can deliver.
class ExactLowRank : public cg::FactorizationProvider {
  public:
    ExactLowRank(Tensor<double, 3> &left, Tensor<double, 3> &right) : _left(&left), _right(&right) {}

    [[nodiscard]] std::string name() const override { return "ExactLowRank"; }
    [[nodiscard]] std::string tag() const override { return "test_lowrank"; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
        cg::TensorHandle const *handle = graph.find_tensor(tensor);
        if (handle == nullptr || handle->rank != 4) {
            return einsums::unexpected(std::string{"this provider only factorizes a rank-4 tensor"});
        }
        std::size_t const rank = _left->dim(0);

        cg::FactorizationPlan plan;
        plan.provider       = name();
        plan.tagged_letters = {"m", "n", "p", "q"};
        plan.factors.push_back(cg::FactorTensor{.name    = "left",
                                                .letters = {"Q", "m", "n"},
                                                .dims    = {rank, handle->dims[0], handle->dims[1]},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.factors.push_back(cg::FactorTensor{.name    = "right",
                                                .letters = {"Q", "p", "q"},
                                                .dims    = {rank, handle->dims[2], handle->dims[3]},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        // Exact, and recorded anyway: "this result is an exact low-rank form of the integrals"
        // is a statement about what the graph computes and belongs in the file with everything
        // else that is.
        plan.accuracy = cg::make_approximation_record(name(), cg::ApproximationEffect::NormRelative, 0.0, 0.0);

        auto *left      = _left;
        auto *right     = _right;
        plan.emit_setup = [left, right](cg::Graph &parent, cg::Graph &body, std::vector<cg::TensorId> const &factors) {
            auto                  *l = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
            auto                  *r = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
            cg::CaptureGuard const guard(body);
            cg::permute("Q,m,n <- Q,m,n", 0.0, l, 1.0, *left);
            cg::permute("Q,p,q <- Q,p,q", 0.0, r, 1.0, *right);
        };
        return plan;
    }

  private:
    Tensor<double, 3> *_left;
    Tensor<double, 3> *_right;
};

} // namespace

TEST_CASE("Factorization - a tagged operand is replaced by its factors and the contraction re-associated",
          "[ComputeGraph][Factorization]") {
    std::size_t const n    = 4;
    std::size_t const rank = 3;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    // M is EXACTLY the product the provider will claim it is. Written out rather than
    // contracted, so the reference this test compares against owes nothing to the machinery
    // under test.
    for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t nn = 0; nn < n; ++nn) {
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = 0; q < n; ++q) {
                    double sum = 0.0;
                    for (std::size_t r = 0; r < rank; ++r) {
                        sum += left(r, m, nn) * right(r, p, q);
                    }
                    M(m, nn, p, q) = sum;
                }
            }
        }
    }

    cg::Graph reference("reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    reference.execute();
    auto const expected = C;

    auto      Cf = create_zero_tensor<double>("C", n, n);
    cg::Graph graph("factorized");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &Cf, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactLowRank>(left, right));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);

    // One contraction became two, plus a setup node holding the fitting.
    REQUIRE(graph.nodes().front().kind == cg::OpKind::Setup);

    // The factors have to be materialized, which is the ordinary lifecycle pass doing its job
    // on tensors this pass created.
    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);

    graph.execute();

    // Not bitwise: the rewrite sums over the auxiliary index in a different order, which is
    // the re-associating tier and is what makes the factorization worth doing.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(std::abs(Cf(i, j) - expected(i, j)) < 1e-10);
        }
    }

    // And the graph says what it now computes.
    REQUIRE(graph.approximations().size() == 1);
    REQUIRE(graph.approximations()[0].pass_name == "ExactLowRank");
}

TEST_CASE("Factorization - a split that is not cheaper is declined", "[ComputeGraph][Factorization]") {
    std::size_t const n = 4;
    // A rank ABOVE n*n, so the two contractions together cost more than the one they replace.
    // Substituting an approximation and coming out slower is the worst of both, and the cost
    // comparison is the only thing standing between the pass and doing it.
    std::size_t const rank = 32;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("too_expensive");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactLowRank>(left, right));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());
    REQUIRE(std::ranges::none_of(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Setup; }));

    // Declined by the BOUND-EXTENT veto rather than by the symbolic comparison, and the
    // distinction is the point. Symbolically two degree-three contractions beat one of degree
    // four whatever the extents are, which is the DF argument and is right for a family; it is
    // also blind to constants, and an auxiliary index larger than the product it replaces is
    // an ordinary case rather than a pathological one. The numeric check is what keeps the
    // pass from trading a measurable regression for that promise.
    bool named_the_cost = false;
    for (auto const &[reason, count] : factorization.skip_reasons()) {
        if (reason.find("not cheaper at the extents") != std::string::npos) {
            named_the_cost = true;
        }
    }
    REQUIRE(named_the_cost);
}

TEST_CASE("Factorization - a symbolic axis makes the bound-extent veto abstain", "[ComputeGraph][Factorization]") {
    std::size_t const n = 4;
    // The SAME rank and the same numbers as the decline case above: at these extents the two
    // contractions cost more than the one they replace. The only difference is that this graph
    // says its extents are a FAMILY rather than a geometry, and that has to be enough to change
    // the answer. The captured 4 is then a placeholder for whatever a later bind supplies, and
    // a veto read off it would delete the factorization the family was annotated for, at the
    // capture and before the save, rather than at the bind that would have wanted it.
    std::size_t const rank = 32;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    // M is exactly the product the provider claims it is, as in the success case above, so the
    // rewrite this abstention allows can be checked and not merely counted.
    for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t nn = 0; nn < n; ++nn) {
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = 0; q < n; ++q) {
                    double sum = 0.0;
                    for (std::size_t r = 0; r < rank; ++r) {
                        sum += left(r, m, nn) * right(r, p, q);
                    }
                    M(m, nn, p, q) = sum;
                }
            }
        }
    }

    cg::Graph reference("reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    reference.execute();
    auto const expected = C;

    auto      Cf = create_zero_tensor<double>("C", n, n);
    cg::Graph graph("symbolic_family");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &Cf, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    // Every axis of the contraction carries one symbol, so there is no literal extent here for
    // the veto to hold the pass to.
    graph.annotate_dims(M, {"n", "n", "n", "n"});
    graph.annotate_dims(T, {"n", "n"});
    graph.annotate_dims(Cf, {"n", "n"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactLowRank>(left, right));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);

    // Specifically NOT declined for its extents, which is the whole difference from the case
    // above and the reason this test spends the same numbers on the opposite outcome.
    for (auto const &[reason, count] : factorization.skip_reasons()) {
        REQUIRE(reason.find("not cheaper at the extents") == std::string::npos);
    }

    // Abstaining is not rewriting blind. The symbolic comparison still ran and still had to
    // accept, and what it accepted still has to compute the right thing.
    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(std::abs(Cf(i, j) - expected(i, j)) < 1e-10);
        }
    }
}

TEST_CASE("Factorization - a tagged tensor the graph writes is declined", "[ComputeGraph][Factorization]") {
    std::size_t const n    = 4;
    std::size_t const rank = 3;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto S     = create_random_tensor<double>("S", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("written");
    {
        cg::CaptureGuard const guard(graph);
        // M is PRODUCED here, so a fitting of it computed once per bound problem would be a
        // fitting of whatever it held before this ran.
        cg::permute("m,n,p,q <- m,n,p,q", 0.0, &M, 1.0, S);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactLowRank>(left, right));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);

    bool named_the_writer = false;
    for (auto const &[reason, count] : factorization.skip_reasons()) {
        if (reason.find("written by this graph") != std::string::npos) {
            named_the_writer = true;
        }
    }
    REQUIRE(named_the_writer);
}

TEST_CASE("Factorization - an accuracy budget the split cannot fit refuses it", "[ComputeGraph][Factorization]") {
    std::size_t const n    = 4;
    std::size_t const rank = 3;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("over_budget");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    // A provider that costs more than the caller will pay. The pass asks BEFORE it rewrites
    // anything, so the refusal leaves the graph exactly as it was rather than half-rewritten
    // with an unrecorded approximation in it.
    class Coarse : public ExactLowRank {
      public:
        using ExactLowRank::ExactLowRank;
        [[nodiscard]] std::string                                  name() const override { return "Coarse"; }
        [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
            auto plan = ExactLowRank::propose(graph, tensor);
            if (plan) {
                plan->provider       = name();
                plan->accuracy.bound = 1e-2;
            }
            return plan;
        }
    };

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<Coarse>(left, right));
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-6);

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());
    REQUIRE(std::ranges::none_of(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Setup; }));

    // Under a budget it fits inside, the same pass applies.
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-1);
    cg::passes::FactorizationPass affordable(registry);
    cg::PassManager               pm2;
    pm2.add(std::shared_ptr<cg::OptimizerPass>(&affordable, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm2));
    REQUIRE(affordable.num_factorized() == 1);
}

TEST_CASE("Factorization - the fitting runs once and the replays skip it", "[ComputeGraph][Factorization]") {
    std::size_t const n    = 4;
    std::size_t const rank = 3;

    auto left  = create_random_tensor<double>("left", rank, n, n);
    auto right = create_random_tensor<double>("right", rank, n, n);
    auto M     = create_zero_tensor<double>("M", n, n, n, n);
    auto T     = create_random_tensor<double>("T", n, n);
    auto C     = create_zero_tensor<double>("C", n, n);

    for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t nn = 0; nn < n; ++nn) {
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = 0; q < n; ++q) {
                    double sum = 0.0;
                    for (std::size_t r = 0; r < rank; ++r) {
                        sum += left(r, m, nn) * right(r, p, q);
                    }
                    M(m, nn, p, q) = sum;
                }
            }
        }
    }

    cg::Graph graph("replayed");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_lowrank"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactLowRank>(left, right));
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));

    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);

    graph.execute();
    auto const first = C;

    // The whole reason a factorization is affordable: the fitting is paid once and every
    // replay after it reads the factors that are already there.
    C.zero();
    graph.execute();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(C(i, j) == first(i, j));
        }
    }
}
