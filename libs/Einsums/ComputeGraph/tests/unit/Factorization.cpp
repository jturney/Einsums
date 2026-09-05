//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

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

// ── The chain plan ─────────────────────────────────────────────────────────

namespace {

/// A three-factor plan, exact by construction, whose auxiliary letters are TWO rather than one.
///
/// @c M[m,n,p,q] @c = @c sum_QR @c A[Q,m,n] @c D[Q,R] @c B[R,p,q]. There is no way to write that
/// as a split of two factors, which is exactly what a plan naming a factor list buys: the pass
/// binarizes the chain, so a provider states the algebra and never a bracketing.
///
/// Deliberately synthetic. The chain a real method produces is the tensor hypercontraction one,
/// and a test of the PASS should not need a fitting to run.
class ExactChain : public cg::FactorizationProvider {
  public:
    ExactChain(Tensor<double, 3> &a, Tensor<double, 2> &d, Tensor<double, 3> &b) : _a(&a), _d(&d), _b(&b) {}

    [[nodiscard]] std::string name() const override { return "ExactChain"; }
    [[nodiscard]] std::string tag() const override { return "test_chain"; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
        cg::TensorHandle const *handle = graph.find_tensor(tensor);
        if (handle == nullptr || handle->rank != 4) {
            return einsums::unexpected(std::string{"this provider only factorizes a rank-4 tensor"});
        }
        std::size_t const left_aux  = _a->dim(0);
        std::size_t const right_aux = _b->dim(0);

        cg::FactorizationPlan plan;
        plan.provider       = name();
        plan.tagged_letters = {"m", "n", "p", "q"};
        plan.factors.push_back(cg::FactorTensor{.name    = "A",
                                                .letters = {"Q", "m", "n"},
                                                .dims    = {left_aux, handle->dims[0], handle->dims[1]},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.factors.push_back(cg::FactorTensor{.name    = "D",
                                                .letters = {"Q", "R"},
                                                .dims    = {left_aux, right_aux},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.factors.push_back(cg::FactorTensor{.name    = "B",
                                                .letters = {"R", "p", "q"},
                                                .dims    = {right_aux, handle->dims[2], handle->dims[3]},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.accuracy = cg::make_approximation_record(name(), cg::ApproximationEffect::NormRelative, 0.0, 0.0);

        auto *a         = _a;
        auto *d         = _d;
        auto *b         = _b;
        plan.emit_setup = [a, d, b](cg::Graph &parent, cg::Graph &body, std::vector<cg::TensorId> const &factors) {
            auto                  *fa = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
            auto                  *fd = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
            auto                  *fb = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[2]).tensor_ptr);
            cg::CaptureGuard const guard(body);
            cg::permute("Q,m,n <- Q,m,n", 0.0, fa, 1.0, *a);
            cg::permute("Q,R <- Q,R", 0.0, fd, 1.0, *d);
            cg::permute("R,p,q <- R,p,q", 0.0, fb, 1.0, *b);
        };
        return plan;
    }

  private:
    Tensor<double, 3> *_a;
    Tensor<double, 2> *_d;
    Tensor<double, 3> *_b;
};

/// A provider whose fit READS the tagged tensor, which is what an amplitude fit is.
///
/// Every other provider in this file, and both shipped ones, fit a quantity the caller handed
/// over and merely claim it approximates the tagged tensor. That is a fixed factorization: the
/// factors are the same however often the tagged tensor moves, so a tensor a loop body rewrites
/// every iteration cannot be factorized that way at all. This one projects the tagged matrix
/// onto a fixed orthonormal basis, ``A = M U`` with ``B = U^T``, so the substitution is exact
/// for any ``M`` whose rows lie in that basis and the fit has to be redone whenever ``M`` does.
class ProjectedAmplitude : public cg::FactorizationProvider {
  public:
    ProjectedAmplitude(Tensor<double, 2> &basis, Tensor<double, 2> &transposed, double *report = nullptr)
        : _basis(&basis), _transposed(&transposed), _report(report) {}

    [[nodiscard]] std::string name() const override { return "ProjectedAmplitude"; }
    [[nodiscard]] std::string tag() const override { return "test_amplitude"; }

    /// The tensor the squared residual of the projection is reported in.
    static std::string residual_tensor() { return "ProjectedAmplitude.residual_squared"; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
        cg::TensorHandle const *handle = graph.find_tensor(tensor);
        if (handle == nullptr || handle->rank != 2) {
            return einsums::unexpected(std::string{"this provider only factorizes a rank-2 tensor"});
        }
        std::size_t const rank = _basis->dim(1);

        cg::FactorizationPlan plan;
        plan.provider         = name();
        plan.tagged_letters   = {"m", "n"};
        plan.fits_from_tagged = true;
        plan.factors.push_back(cg::FactorTensor{.name    = "projected",
                                                .letters = {"m", "Q"},
                                                .dims    = {handle->dims[0], rank},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.factors.push_back(cg::FactorTensor{.name    = "basisT",
                                                .letters = {"Q", "n"},
                                                .dims    = {rank, handle->dims[1]},
                                                .spaces  = {},
                                                .dtype   = einsums::packed_gemm::ScalarType::Float64});
        plan.accuracy = cg::make_approximation_record(name(), cg::ApproximationEffect::NormRelative, 0.0, 0.0);

        // The record's BOUND is written once; the record names the TENSOR carrying what a bind
        // found, which is the rule for a rewrite inside a loop read back: the fitting rewrites
        // that value at every update, so what stands in it when the solver stops is the fit's
        // residual at the last iteration.
        if (_report != nullptr) {
            plan.accuracy.measurement = residual_tensor();
        }

        auto *basis      = _basis;
        auto *transposed = _transposed;
        auto *report     = _report;
        auto  rows       = handle->dims[0];
        auto  cols       = handle->dims[1];
        plan.emit_setup  = [basis, transposed, report, tensor, rows, cols](cg::Graph &parent, cg::Graph &body,
                                                                           std::vector<cg::TensorId> const &factors) {
            auto *a = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[0]).tensor_ptr);
            auto *b = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(factors[1]).tensor_ptr);
            auto *m = static_cast<einsums::RuntimeTensor<double> *>(parent.tensor(tensor).tensor_ptr);

            cg::CaptureGuard const guard(body);
            // The projection reads the tagged tensor, which is what makes this a fit OF it.
            cg::einsum("m,n ; n,Q -> m,Q", a, *m, *basis);
            cg::permute("Q,n <- Q,n", 0.0, b, 1.0, *transposed);
            if (report == nullptr) {
                return;
            }
            // What the fit is worth on THIS bind, measured rather than asserted, and rewritten at
            // every refit. Named after the graph it is declared in, because this fitting is
            // emitted twice for a re-fitted amplitude and the storage auditor keys its duplicate
            // check on the NAME rather than on the buffer.
            auto &residual = body.declare_runtime_tensor<double>(fmt::format("{}_projection_residual", body.name()), {rows, cols},
                                                                 /*intermediate=*/true);
            cg::CaptureContext::current().get_or_register_scalar(report, residual_tensor());
            cg::permute("m,n <- m,n", 0.0, &residual, 1.0, *m);
            cg::einsum("m,Q ; Q,n -> m,n", 1.0, &residual, -1.0, *a, *b);
            cg::dot(report, residual, residual);
        };
        return plan;
    }

  private:
    Tensor<double, 2> *_basis;
    Tensor<double, 2> *_transposed;
    double            *_report;
};

/// The same plan, declaring nothing about what it reads. Used to pin the decline: a provider
/// that says nothing gets the fixed-quantity answer, which is the one that cannot be wrong.
class SilentAmplitude : public ProjectedAmplitude {
  public:
    using ProjectedAmplitude::ProjectedAmplitude;

    [[nodiscard]] std::string name() const override { return "SilentAmplitude"; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
        auto plan = ProjectedAmplitude::propose(graph, tensor);
        if (plan) {
            plan->provider         = name();
            plan->fits_from_tagged = false;
        }
        return plan;
    }
};

/// An orthonormal basis of @p rank columns in @p n dimensions, and its transpose.
std::pair<Tensor<double, 2>, Tensor<double, 2>> orthonormal_basis(std::size_t n, std::size_t rank, int seed) {
    auto raw = create_random_tensor<double>("raw", n, rank);
    (void)seed;
    // Modified Gram-Schmidt, in double, over a handful of columns.
    for (std::size_t col = 0; col < rank; ++col) {
        for (std::size_t prev = 0; prev < col; ++prev) {
            double overlap = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                overlap += raw(row, prev) * raw(row, col);
            }
            for (std::size_t row = 0; row < n; ++row) {
                raw(row, col) -= overlap * raw(row, prev);
            }
        }
        double norm = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            norm += raw(row, col) * raw(row, col);
        }
        norm = std::sqrt(norm);
        for (std::size_t row = 0; row < n; ++row) {
            raw(row, col) /= norm;
        }
    }
    auto transposed = create_zero_tensor<double>("basisT", rank, n);
    for (std::size_t col = 0; col < rank; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            transposed(col, row) = raw(row, col);
        }
    }
    return {raw, transposed};
}

} // namespace

namespace {

/// One CCSD-shaped iteration: a contraction over the amplitude, a residual built from it, and an
/// update that divides the residual by a denominator into the amplitude.
///
/// The denominator is all ones and the residual is a left multiplication, so the amplitude stays
/// in the row space of the fixed basis at every iteration and the projected factorization stays
/// EXACT. What is under test is where the fitting goes and how often it runs, not the accuracy
/// of an approximation.
struct AmplitudeLoop {
    RuntimeTensor<double> *amplitude;
    RuntimeTensor<double> *result;
    cg::Graph             *body;
};

AmplitudeLoop capture_amplitude_iteration(cg::Graph &graph, RuntimeTensor<double> &M, RuntimeTensor<double> &T, RuntimeTensor<double> &C,
                                          RuntimeTensor<double> &S, RuntimeTensor<double> &R, RuntimeTensor<double> &copy,
                                          RuntimeTensor<double> &ones, std::size_t iterations) {
    cg::Graph &body = graph.add_loop("amplitude_iteration", iterations, [iterations](std::size_t it) { return it + 1 < iterations; });
    {
        cg::CaptureGuard const guard(body);
        // The residual reads a COPY of the amplitude rather than the amplitude, so exactly one
        // two-operand contraction offers the tagged tensor as an operand. A tagged tensor two
        // contractions read is factorized once per reader today, and that is a separate question
        // from where the fitting goes.
        cg::axpby(1.0, M, 0.0, &copy);
        cg::einsum("m,n ; n,j -> m,j", &C, M, T);
        cg::einsum("m,k ; k,n -> m,n", &R, S, copy);
        cg::direct_division(1.0, R, ones, 0.0, &M);
    }
    return AmplitudeLoop{.amplitude = &M, .result = &C, .body = &body};
}

} // namespace

TEST_CASE("Factorization - an amplitude the loop body updates is refitted in the body", "[ComputeGraph][Factorization][Amplitude]") {
    std::size_t const n = 8, rank = 2, iterations = 3;

    auto [basis, transposed] = orthonormal_basis(n, rank, 7);
    auto weights             = create_random_tensor<double>("weights", n, rank);

    // M = W U^T, so M's rows lie in the basis and M U U^T is M exactly. A left multiplication
    // keeps them there, which is what the residual below is.
    auto seed_amplitude = [&](RuntimeTensor<double> &into) {
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                double sum = 0.0;
                for (std::size_t q = 0; q < rank; ++q) {
                    sum += weights(row, q) * transposed(q, col);
                }
                into(row, col) = sum;
            }
        }
    };

    auto T    = RuntimeTensor<double>(create_random_tensor<double>("T", n, n));
    auto S    = RuntimeTensor<double>(create_random_tensor<double>("S", n, n));
    auto ones = RuntimeTensor<double>(create_zero_tensor<double>("ones", n, n));
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            ones(row, col) = 1.0;
        }
    }

    // The reference arm: the same iteration with no factorization at all.
    auto M_ref    = RuntimeTensor<double>(create_zero_tensor<double>("M", n, n));
    auto C_ref    = RuntimeTensor<double>(create_zero_tensor<double>("C", n, n));
    auto R_ref    = RuntimeTensor<double>(create_zero_tensor<double>("R", n, n));
    auto copy_ref = RuntimeTensor<double>(create_zero_tensor<double>("Mcopy", n, n));
    seed_amplitude(M_ref);
    cg::Graph reference("amplitude_reference");
    capture_amplitude_iteration(reference, M_ref, T, C_ref, S, R_ref, copy_ref, ones, iterations);
    auto reference_default = cg::PassManager::create_default();
    reference.apply(reference_default);
    reference.execute();

    auto M    = RuntimeTensor<double>(create_zero_tensor<double>("M", n, n));
    auto C    = RuntimeTensor<double>(create_zero_tensor<double>("C", n, n));
    auto R    = RuntimeTensor<double>(create_zero_tensor<double>("R", n, n));
    auto copy = RuntimeTensor<double>(create_zero_tensor<double>("Mcopy", n, n));
    seed_amplitude(M);
    cg::Graph  graph("amplitude_factorized");
    auto const loop = capture_amplitude_iteration(graph, M, T, C, S, R, copy, ones, iterations);
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_amplitude"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ProjectedAmplitude>(basis, transposed));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    // The tag is declared on the graph and read inside the body, which are two handles for one
    // buffer; the analysis phase is what carries it across.
    pm.add(std::make_shared<cg::passes::ProvenancePropagation>());
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    pm.set_verbosity(3);
    REQUIRE(graph.apply(pm));
    CHECK(factorization.num_factorized() == 1);

    // The fitting is in BOTH places, and each is there for one of the two things a re-fit has to
    // do: the parent's setup fits what is bound before the loop runs, and the body's copy re-fits
    // after each update.
    std::size_t setups_in_parent = 0;
    for (auto const &node : graph.nodes()) {
        if (std::get_if<cg::SetupDescriptor>(&node.op_data) != nullptr) {
            ++setups_in_parent;
        }
    }
    CHECK(setups_in_parent == 1);
    std::size_t setups_in_body = 0;
    for (auto const &node : loop.body->nodes()) {
        if (std::get_if<cg::SetupDescriptor>(&node.op_data) != nullptr) {
            ++setups_in_body;
        }
    }
    CHECK(setups_in_body == 0);

    auto graph_default = cg::PassManager::create_default();
    graph.apply(graph_default);
    INFO(fmt::format("{}", fmt::join(cg::passes::duplicate_materializations(graph), ", ")));
    CHECK(cg::passes::duplicate_materializations(graph).empty());
    CHECK(cg::passes::stranded_materializations(graph).empty());
    graph.execute();

    // Exact, because the projection is exact for an amplitude in this basis, and the point of the
    // case is that it stays exact across iterations: a fit made once before the loop would be the
    // first iteration's and would be visibly wrong by the third.
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(C_ref(row, col), 1.0e-10));
        }
    }
}

TEST_CASE("Factorization - the record on a converged output is the last iteration's fit", "[ComputeGraph][Factorization][Amplitude]") {
    // The rule for an approximation record inside a loop, asserted rather than argued.
    //
    // A record is written once, at optimize time, so its BOUND is what the structure claims and
    // cannot be what a particular bind found. What a bind leaves behind is a parameter, and the
    // record names which one. A fit re-fitted at every amplitude update rewrites that parameter
    // once per iteration, so what stands in it when the solver stops is the fit's residual at the
    // LAST iteration, which is the iteration the answer came out of. There is no per-iteration
    // history and there should not be: the intermediate fits are error in a quantity nobody kept.
    std::size_t const n = 8, rank = 3;

    auto [basis, transposed] = orthonormal_basis(n, rank, 23);
    auto T                   = RuntimeTensor<double>(create_random_tensor<double>("T", n, n));
    auto S                   = RuntimeTensor<double>(create_random_tensor<double>("S", n, n));
    auto ones                = RuntimeTensor<double>(create_zero_tensor<double>("ones", n, n));
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            ones(row, col) = 1.0;
        }
    }

    // The residual of the projection for a given amplitude, written out: what the graph's own
    // parameter has to agree with.
    auto expected_residual = [&](RuntimeTensor<double> const &M) {
        double total = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                double fitted = 0.0;
                for (std::size_t q = 0; q < rank; ++q) {
                    double projected = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        projected += M(row, k) * basis(k, q);
                    }
                    fitted += projected * transposed(q, col);
                }
                double const diff = M(row, col) - fitted;
                total += diff * diff;
            }
        }
        return total;
    };

    auto run = [&](std::size_t iterations) {
        // A RANDOM amplitude, not one in the basis: the fit is inexact on purpose, because a
        // residual of zero at every iteration would agree with any rule at all.
        auto M    = RuntimeTensor<double>(create_random_tensor<double>("M", n, n));
        auto C    = RuntimeTensor<double>(create_zero_tensor<double>("C", n, n));
        auto R    = RuntimeTensor<double>(create_zero_tensor<double>("R", n, n));
        auto copy = RuntimeTensor<double>(create_zero_tensor<double>("Mcopy", n, n));
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                M(row, col) = 0.25 * static_cast<double>(row + 1) - 0.1 * static_cast<double>(col);
            }
        }

        auto       graph = std::make_unique<cg::Graph>("amplitude_record");
        auto const loop  = capture_amplitude_iteration(*graph, M, T, C, S, R, copy, ones, iterations);
        graph->annotate_tag(M, cg::ProvenanceTag{.name = "test_amplitude"});

        auto                      measured_residual = std::make_unique<double>(0.0);
        cg::FactorizationRegistry registry;
        registry.add(std::make_shared<ProjectedAmplitude>(basis, transposed, measured_residual.get()));
        cg::passes::FactorizationPass factorization(registry);
        cg::PassManager               pm;
        pm.add(std::make_shared<cg::passes::ProvenancePropagation>());
        pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
        REQUIRE(graph->apply(pm));

        auto defaults = cg::PassManager::create_default();
        graph->apply(defaults);
        graph->execute();

        (void)loop;
        return std::tuple{std::move(graph), expected_residual(M), *measured_residual};
    };

    auto const [one_graph, one_expected, one_measured]       = run(1);
    auto const [three_graph, three_expected, three_measured] = run(3);

    // The record says where to read the number, which is what makes it readable at all without
    // knowing how this provider spells its diagnostics.
    REQUIRE(one_graph->approximations().size() == 1);
    CHECK(one_graph->approximations().front().measurement == ProjectedAmplitude::residual_tensor());
    CHECK(one_graph->approximations().front().origin == cg::ApproximationOrigin::Asserted);

    // And what stands in it is the LAST iteration's fit, not the first one's.
    CHECK_THAT(one_measured, Catch::Matchers::WithinRel(one_expected, 1.0e-10));
    CHECK_THAT(three_measured, Catch::Matchers::WithinRel(three_expected, 1.0e-10));
    CHECK(three_measured != one_measured);
    CHECK(three_expected > 0.0);
}

TEST_CASE("Factorization - a fit that does not read the updated tensor is declined", "[ComputeGraph][Factorization][Amplitude]") {
    std::size_t const n = 8, rank = 2, iterations = 2;

    auto [basis, transposed] = orthonormal_basis(n, rank, 11);
    auto M                   = RuntimeTensor<double>(create_random_tensor<double>("M", n, n));
    auto T                   = RuntimeTensor<double>(create_random_tensor<double>("T", n, n));
    auto S                   = RuntimeTensor<double>(create_random_tensor<double>("S", n, n));
    auto C                   = RuntimeTensor<double>(create_zero_tensor<double>("C", n, n));
    auto R                   = RuntimeTensor<double>(create_zero_tensor<double>("R", n, n));
    auto copy                = RuntimeTensor<double>(create_zero_tensor<double>("Mcopy", n, n));
    auto ones                = RuntimeTensor<double>(create_zero_tensor<double>("ones", n, n));

    cg::Graph graph("amplitude_silent");
    capture_amplitude_iteration(graph, M, T, C, S, R, copy, ones, iterations);
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_amplitude"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<SilentAmplitude>(basis, transposed));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::make_shared<cg::passes::ProvenancePropagation>());
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    pm.set_verbosity(3);
    CHECK_FALSE(graph.apply(pm));
    CHECK(factorization.num_factorized() == 0);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("does not read the tagged tensor") != std::string::npos);
}

TEST_CASE("Factorization - a tagged tensor written by anything but an update still declines", "[ComputeGraph][Factorization][Amplitude]") {
    std::size_t const n = 8, rank = 2, iterations = 2;

    auto [basis, transposed] = orthonormal_basis(n, rank, 13);
    auto M                   = RuntimeTensor<double>(create_random_tensor<double>("M", n, n));
    auto T                   = RuntimeTensor<double>(create_random_tensor<double>("T", n, n));
    auto S                   = RuntimeTensor<double>(create_random_tensor<double>("S", n, n));
    auto C                   = RuntimeTensor<double>(create_zero_tensor<double>("C", n, n));

    cg::Graph  graph("amplitude_wrong_writer");
    cg::Graph &body = graph.add_loop("iteration", iterations, [iterations](std::size_t it) { return it + 1 < iterations; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("m,n ; n,j -> m,j", &C, M, T);
        // A contraction, not a division of a residual by a denominator. The tensor still moves
        // every iteration and there is still nothing a fit made once could describe, but the
        // statement is not one this analysis will call an update.
        cg::einsum("m,k ; k,n -> m,n", &M, S, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_amplitude"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ProjectedAmplitude>(basis, transposed));

    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::make_shared<cg::passes::ProvenancePropagation>());
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    pm.set_verbosity(3);
    CHECK_FALSE(graph.apply(pm));
    CHECK(factorization.num_factorized() == 0);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("could go stale") != std::string::npos);
}

TEST_CASE("Factorization - what counts as an amplitude update, and what does not", "[ComputeGraph][Factorization][Amplitude]") {
    // The three answers the analysis gives, asked of it directly. What a pass does with them is
    // the business of the cases above; what they ARE is decided here, because the whole of the
    // third mode of the stale-factor refusal rests on this one recognition.
    std::size_t const n    = 4;
    auto              M    = RuntimeTensor<double>(create_random_tensor<double>("M", n, n));
    auto              R    = RuntimeTensor<double>(create_random_tensor<double>("R", n, n));
    auto              D    = RuntimeTensor<double>(create_random_tensor<double>("D", n, n));
    auto              S    = RuntimeTensor<double>(create_random_tensor<double>("S", n, n));
    auto              step = RuntimeTensor<double>(create_zero_tensor<double>("step", n, n));

    SECTION("a residual divided by a denominator into the amplitude") {
        cg::Graph body("update_division");
        {
            cg::CaptureGuard const guard(body);
            cg::direct_division(1.0, R, D, 0.0, &M);
        }
        auto const found = cg::amplitude_update_writer(body, body.live_tensor_id_by_ptr(&M, {}));
        CHECK(found.has_value());
    }

    SECTION("the same update written as a step a host's DIIS can read") {
        cg::Graph body("update_step");
        {
            cg::CaptureGuard const guard(body);
            cg::direct_division(1.0, R, D, 0.0, &step);
            cg::axpby(1.0, step, 1.0, &M);
        }
        auto const found = cg::amplitude_update_writer(body, body.live_tensor_id_by_ptr(&M, {}));
        CHECK(found.has_value());
    }

    SECTION("a contraction is not an update") {
        cg::Graph body("update_contraction");
        {
            cg::CaptureGuard const guard(body);
            cg::einsum("m,k ; k,n -> m,n", &M, S, R);
        }
        auto const found = cg::amplitude_update_writer(body, body.live_tensor_id_by_ptr(&M, {}));
        REQUIRE_FALSE(found.has_value());
        CHECK(found.error().find("not an amplitude update") != std::string::npos);
    }

    SECTION("an accumulation from something no division produced is not an update") {
        cg::Graph body("update_bare_axpby");
        {
            cg::CaptureGuard const guard(body);
            cg::axpby(1.0, R, 1.0, &M);
        }
        auto const found = cg::amplitude_update_writer(body, body.live_tensor_id_by_ptr(&M, {}));
        REQUIRE_FALSE(found.has_value());
        CHECK(found.error().find("no division in this body produced") != std::string::npos);
    }

    SECTION("two writers are not an update, whatever they are") {
        cg::Graph body("update_two_writers");
        {
            cg::CaptureGuard const guard(body);
            cg::direct_division(1.0, R, D, 0.0, &M);
            cg::axpby(1.0, S, 1.0, &M);
        }
        auto const found = cg::amplitude_update_writer(body, body.live_tensor_id_by_ptr(&M, {}));
        REQUIRE_FALSE(found.has_value());
        CHECK(found.error().find("value-writers") != std::string::npos);
    }
}

TEST_CASE("Factorization - a three-factor chain is substituted and binarized by the pass", "[ComputeGraph][Factorization][Chain]") {
    std::size_t const n    = 6;
    std::size_t const left = 2;
    std::size_t const rght = 3;

    auto A = create_random_tensor<double>("A", left, n, n);
    auto D = create_random_tensor<double>("D", left, rght);
    auto B = create_random_tensor<double>("B", rght, n, n);
    auto M = create_zero_tensor<double>("M", n, n, n, n);
    auto T = create_random_tensor<double>("T", n, n);
    auto C = create_zero_tensor<double>("C", n, n);

    // M is EXACTLY the chain the provider claims it is, written out rather than contracted so
    // the reference owes nothing to the machinery under test.
    for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t nn = 0; nn < n; ++nn) {
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = 0; q < n; ++q) {
                    double sum = 0.0;
                    for (std::size_t Q = 0; Q < left; ++Q) {
                        for (std::size_t R = 0; R < rght; ++R) {
                            sum += A(Q, m, nn) * D(Q, R) * B(R, p, q);
                        }
                    }
                    M(m, nn, p, q) = sum;
                }
            }
        }
    }

    cg::Graph reference("chain_reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    reference.execute();
    auto const expected = C;

    auto      Cf = create_zero_tensor<double>("C", n, n);
    cg::Graph graph("chain_factorized");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &Cf, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_chain"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactChain>(A, D, B));

    cg::passes::FactorizationPass factorization(registry);
    // The cost line is what the report offers as evidence the rewrite paid, and it is derived
    // from the algebra alone. Checking it against the nodes the lowering emitted is what makes
    // either derivation evidence.
    factorization.set_verify_costs(true);
    cg::PassManager pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);
    REQUIRE(factorization.cost_mismatches().empty());

    // One contraction became three: four leaves make three interior contractions in any binary
    // tree, and the pass emitted the tree rather than a three-operand node nothing can lower.
    std::size_t contractions = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
    }
    REQUIRE(contractions == 3);
    REQUIRE(graph.nodes().front().kind == cg::OpKind::Setup);

    // And no intermediate the rewrite introduced is rank four, which is the tagged tensor the
    // substitution exists to avoid forming. It is not a rule anywhere in the pass: rebuilding
    // it is simply the most expensive tree and the search never picks it.
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name.rfind("ExactChain_M_x", 0) == 0) {
            REQUIRE(handle.rank < 4);
        }
    }

    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(std::abs(Cf(i, j) - expected(i, j)) < 1e-10);
        }
    }

    REQUIRE(graph.approximations().size() == 1);
    REQUIRE(graph.approximations()[0].pass_name == "ExactChain");
}

TEST_CASE("Factorization - one factor is a rename rather than a factorization, and is declined", "[ComputeGraph][Factorization][Chain]") {
    // A plan naming a single factor says the tagged tensor IS that factor under other letters.
    // There is nothing to re-associate around, and accepting it would substitute a tensor for
    // itself and record an approximation for having done so.
    class OneFactor : public cg::FactorizationProvider {
      public:
        [[nodiscard]] std::string                                  name() const override { return "OneFactor"; }
        [[nodiscard]] std::string                                  tag() const override { return "test_one"; }
        [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const &graph, cg::TensorId tensor) const override {
            cg::TensorHandle const *handle = graph.find_tensor(tensor);
            cg::FactorizationPlan   plan;
            plan.provider       = name();
            plan.tagged_letters = {"m", "n", "p", "q"};
            plan.factors.push_back(cg::FactorTensor{.name    = "same",
                                                    .letters = {"m", "n", "p", "q"},
                                                    .dims    = {handle->dims[0], handle->dims[1], handle->dims[2], handle->dims[3]},
                                                    .spaces  = {},
                                                    .dtype   = einsums::packed_gemm::ScalarType::Float64});
            plan.accuracy   = cg::make_approximation_record(name(), cg::ApproximationEffect::NormRelative, 0.0, 0.0);
            plan.emit_setup = [](cg::Graph &, cg::Graph &, std::vector<cg::TensorId> const &) {};
            return plan;
        }
    };

    std::size_t const n = 4;
    auto              M = create_random_tensor<double>("M", n, n, n, n);
    auto              T = create_random_tensor<double>("T", n, n);
    auto              C = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("one_factor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("m,n,p,q ; p,q -> m,n", &C, M, T);
    }
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_one"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<OneFactor>());
    cg::passes::FactorizationPass factorization(registry);
    cg::PassManager               pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));

    REQUIRE_FALSE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 0);
    REQUIRE(graph.approximations().empty());

    bool named_the_count = false;
    for (auto const &[reason, count] : factorization.skip_reasons()) {
        if (reason.find("fewer than two factors") != std::string::npos) {
            named_the_count = true;
        }
    }
    REQUIRE(named_the_count);
}

TEST_CASE("Factorization - the cone is re-associated, not just the tagged contraction", "[ComputeGraph][Factorization][Chain]") {
    // The author writes an intermediate and the tagged contraction reads it. The name is the
    // author's bracketing rather than the problem's, so the pass flattens it in and brackets
    // the whole cone at once: here the outer product is worth never forming, which is a
    // decision that cannot be reached one contraction at a time.
    std::size_t const n    = 6;
    std::size_t const left = 2;
    std::size_t const rght = 2;

    auto A  = create_random_tensor<double>("A", left, n, n);
    auto D  = create_random_tensor<double>("D", left, rght);
    auto B  = create_random_tensor<double>("B", rght, n, n);
    auto t1 = create_random_tensor<double>("t1", n);
    auto t2 = create_random_tensor<double>("t2", n);
    auto M  = create_zero_tensor<double>("M", n, n, n, n);

    for (std::size_t m = 0; m < n; ++m) {
        for (std::size_t nn = 0; nn < n; ++nn) {
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = 0; q < n; ++q) {
                    double sum = 0.0;
                    for (std::size_t Q = 0; Q < left; ++Q) {
                        for (std::size_t R = 0; R < rght; ++R) {
                            sum += A(Q, m, nn) * D(Q, R) * B(R, p, q);
                        }
                    }
                    M(m, nn, p, q) = sum;
                }
            }
        }
    }

    auto const build = [&](cg::Graph &graph, Tensor<double, 2> &out) {
        auto                  &U = graph.scratch<double, 2>("U", n, n);
        cg::CaptureGuard const guard(graph);
        cg::einsum("p ; q -> p,q", &U, t1, t2);
        cg::einsum("m,n,p,q ; p,q -> m,n", &out, M, U);
    };

    auto      expected = create_zero_tensor<double>("C", n, n);
    cg::Graph reference("cone_reference");
    build(reference, expected);
    auto defaults_for_reference = cg::PassManager::create_default();
    reference.apply(defaults_for_reference);
    reference.execute();

    auto      C = create_zero_tensor<double>("C", n, n);
    cg::Graph graph("cone_factorized");
    build(graph, C);
    graph.annotate_tag(M, cg::ProvenanceTag{.name = "test_chain"});

    cg::FactorizationRegistry registry;
    registry.add(std::make_shared<ExactChain>(A, D, B));

    cg::passes::FactorizationPass factorization(registry);
    factorization.set_verify_costs(true);
    cg::PassManager pm;
    pm.add(std::shared_ptr<cg::OptimizerPass>(&factorization, [](cg::OptimizerPass *) {}));
    REQUIRE(graph.apply(pm));
    REQUIRE(factorization.num_factorized() == 1);
    REQUIRE(factorization.cost_mismatches().empty());

    // The author's intermediate is gone: its two factors joined the leaves and the search put
    // them somewhere the outer product never has to be formed.
    REQUIRE(factorization.num_dissolved() == 1);

    // Five leaves, four contractions, and none of them writes a rank-four tensor: the tagged
    // tensor is the most expensive thing the search could rebuild and it never does.
    std::size_t contractions = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
    }
    REQUIRE(contractions == 4);
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name.rfind("ExactChain_M_x", 0) == 0) {
            REQUIRE(handle.rank < 4);
        }
    }

    // The cost line has to say the rewrite paid, and it is checked against the nodes rather
    // than believed: a report that offers a rewrite to nothing as evidence went unread once.
    bool reported = false;
    for (auto const &dump : factorization.last_dumps()) {
        if (dump.changed) {
            REQUIRE_FALSE(dump.cost_after.empty());
            REQUIRE(dump.cost_after != dump.cost_before);
            reported = true;
        }
    }
    REQUIRE(reported);

    auto defaults = cg::PassManager::create_default();
    graph.apply(defaults);
    graph.execute();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(std::abs(C(i, j) - expected(i, j)) < 1e-10);
        }
    }
}
