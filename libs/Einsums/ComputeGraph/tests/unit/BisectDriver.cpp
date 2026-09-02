//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// BisectDriver: which pass moved the answer.
//
// The load-bearing cases here are the ones with a DELIBERATELY WRONG pass in the list. A driver
// exercised only on pipelines where nothing is wrong proves that it can say "clean", which is the
// answer it gives when it is broken as well as when it is working. So the wrong pass is injected,
// and the assertion is that the driver names it, and names it first.
//
// The two perturbing passes below differ in exactly one thing: the tier they declare. The same
// arithmetic change is a divergence for one and within bounds for the other, which is what makes
// this a test of the BOUND rather than of the comparison.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/BisectDriver.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// The shape LayoutAssignment responds to: a rank-3 graph-owned intermediate whose captured axis
// order costs a copy at each end. Chosen so that at least one REAL structural pass fires, because
// a driver reporting "clean" over a pipeline where nothing ran has not been tested at all.
constexpr size_t kI = 6, kK = 3, kX = 7, kJ = 5, kY = 4;
constexpr size_t kN = 6;

/// Keeps every trial's tensors alive without letting two trials share one.
///
/// This is the discipline the driver's own documentation asks a builder for, written out: the
/// builder mints fresh buffers per call and hands ownership to a pool that outlives the run, so no
/// trial can start from another trial's answer.
struct Pool {
    std::vector<std::shared_ptr<RuntimeTensor<double>>> kept;

    RuntimeTensor<double> &add(RuntimeTensor<double> tensor) {
        kept.push_back(std::make_shared<RuntimeTensor<double>>(std::move(tensor)));
        return *kept.back();
    }
};

/// A chain of two contractions into one interface output.
cg::BisectDriver::Builder chain_builder(Pool &pool) {
    einsums::seed_random(31);
    auto const a = create_random_tensor<double>("a", kI, kK);
    auto const b = create_random_tensor<double>("b", kK, kX, kJ);
    auto const d = create_random_tensor<double>("d", kJ, kY);

    return [&pool, a, b, d](cg::Graph &graph) {
        auto &A = pool.add(RuntimeTensor<double>(a));
        auto &B = pool.add(RuntimeTensor<double>(b));
        auto &D = pool.add(RuntimeTensor<double>(d));
        auto &R = pool.add(RuntimeTensor<double>(create_zero_tensor<double>("R", kI, kX, kY)));
        auto &W = graph.declare_runtime_tensor<double>("W", {kI, kJ, kX}, /*intermediate=*/true);

        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,x <- i,k ; k,x,j", 0.0, &W, 1.0, A, B);
        cg::einsum("i,x,y <- i,j,x ; j,y", 0.0, &R, 1.0, W, D);
    };
}

/// Scale the first contraction's product prefactor, and lie about the tier while doing it.
///
/// The shape a real bug takes: a pass that changes the arithmetic and reports itself as belonging
/// to a tier that promises it did not.
class PerturbingPass : public cg::OptimizerPass {
  public:
    PerturbingPass(std::string name, double factor, cg::PassTier tier) : _name(std::move(name)), _factor(factor), _tier(tier) {}

    [[nodiscard]] std::string   name() const override { return _name; }
    [[nodiscard]] cg::PassPhase phase() const override { return cg::PassPhase::StructuralAlgebraic; }
    [[nodiscard]] cg::PassTier  tier() const override { return _tier; }

    bool run(cg::Graph &graph) override {
        for (auto &node : graph.nodes()) {
            auto *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
            if (desc == nullptr || !desc->params) {
                continue;
            }
            // Both spellings, as any honest rewriter must: the live value is what the executor
            // reads and the snapshot is what analysis reads.
            double const scaled = cg::as<double>(desc->params->ab_pf) * _factor;
            desc->params->ab_pf = scaled;
            desc->ab_prefactor  = scaled;
            return true;
        }
        return false;
    }

  private:
    std::string  _name;
    double       _factor{1.0};
    cg::PassTier _tier{cg::PassTier::BitwiseExact};
};

/// Declares the algebraic phase and changes nothing, so a trial over it must read "did not fire".
class InertPass : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string   name() const override { return "InertPass"; }
    [[nodiscard]] cg::PassPhase phase() const override { return cg::PassPhase::StructuralAlgebraic; }
    [[nodiscard]] cg::PassTier  tier() const override { return cg::PassTier::BitwiseExact; }

    bool run(cg::Graph &) override { return false; }
};

std::shared_ptr<cg::OptimizerPass> perturbing(std::string name, double factor, cg::PassTier tier) {
    return std::make_shared<PerturbingPass>(std::move(name), factor, tier);
}

} // namespace

TEST_CASE("BisectDriver - the real pipeline comes back clean", "[ComputeGraph][BisectDriver]") {
    Pool             pool;
    cg::BisectDriver driver(chain_builder(pool));

    auto const report = driver.run();

    INFO(report.to_string());
    CHECK(report.baseline_ok);
    CHECK(report.clean());
    CHECK_FALSE(report.trials.empty());
    // Something has to have run, or "clean" means "nothing was tried".
    CHECK(std::ranges::any_of(report.trials, [](cg::BisectTrial const &t) { return t.modified; }));
}

TEST_CASE("BisectDriver - it names the wrong pass, and names it first", "[ComputeGraph][BisectDriver]") {
    Pool             pool;
    cg::BisectDriver driver(chain_builder(pool));
    driver.set_passes(
        {std::make_shared<InertPass>(), perturbing("BrokenPass", 1.5, cg::PassTier::BitwiseExact), std::make_shared<InertPass>()});

    auto const report = driver.run();

    INFO(report.to_string());
    REQUIRE(report.baseline_ok);
    CHECK_FALSE(report.clean());
    REQUIRE(report.first_divergence() < report.trials.size());
    CHECK(report.trials[report.first_divergence()].name == "BrokenPass");
    CHECK(report.trials[report.first_divergence()].norm_relative > 0.1);
    CHECK(report.to_string().find("BrokenPass") != std::string::npos);
}

TEST_CASE("BisectDriver - the same change is a divergence or not, by tier", "[ComputeGraph][BisectDriver]") {
    // One part in 1e15 is far above bit equality and far below a thousand epsilon. So the verdict
    // is decided entirely by what the pass CLAIMS, which is the property under test.
    constexpr double kTiny = 1.0 + 1.0e-15;

    {
        Pool             pool;
        cg::BisectDriver driver(chain_builder(pool));
        driver.set_passes({perturbing("ClaimsBitwise", kTiny, cg::PassTier::BitwiseExact)});
        auto const report = driver.run();
        INFO(report.to_string());
        CHECK_FALSE(report.clean());
        CHECK(report.trials.at(0).diverged);
    }
    {
        Pool             pool;
        cg::BisectDriver driver(chain_builder(pool));
        driver.set_passes({perturbing("ClaimsReAssociating", kTiny, cg::PassTier::ReAssociating)});
        auto const report = driver.run();
        INFO(report.to_string());
        CHECK(report.clean());
        CHECK_FALSE(report.trials.at(0).diverged);
        CHECK(report.trials.at(0).modified);
        CHECK(report.trials.at(0).norm_relative > 0.0); // it really did move something
    }
}

TEST_CASE("BisectDriver - a pass that did not fire is not blamed", "[ComputeGraph][BisectDriver]") {
    Pool             pool;
    cg::BisectDriver driver(chain_builder(pool));
    driver.set_passes({std::make_shared<InertPass>()});

    auto const report = driver.run();

    INFO(report.to_string());
    CHECK(report.clean());
    CHECK_FALSE(report.trials.at(0).modified);
    CHECK_FALSE(report.trials.at(0).diverged);
    CHECK(report.to_string().find("did not fire") != std::string::npos);
}

TEST_CASE("BisectDriver - cumulative mode blames the pass that was added", "[ComputeGraph][BisectDriver]") {
    Pool             pool;
    cg::BisectDriver driver(chain_builder(pool));
    driver.set_passes(
        {std::make_shared<InertPass>(), std::make_shared<InertPass>(), perturbing("BrokenPass", 1.5, cg::PassTier::BitwiseExact)});
    driver.set_mode(cg::BisectMode::Cumulative);

    auto const report = driver.run();

    INFO(report.to_string());
    REQUIRE(report.first_divergence() < report.trials.size());
    // Every trial after this one inherits the damage, which is exactly why the report names the
    // FIRST one past its bound rather than the worst.
    CHECK(report.first_divergence() == 2);
    CHECK(report.trials[2].name == "BrokenPass");
}

TEST_CASE("BisectDriver - widening the bound changes the verdict", "[ComputeGraph][BisectDriver]") {
    Pool             pool;
    cg::BisectDriver driver(chain_builder(pool));
    driver.set_passes({perturbing("ClaimsBitwise", 1.0 + 1.0e-15, cg::PassTier::BitwiseExact)});

    // A bitwise-exact bound is zero, and zero times anything is still zero: widening cannot
    // excuse a tier that promised bit equality, which is the point of scaling the TIER's bound
    // rather than adding a floor to it.
    driver.set_bound_scale(1.0e6);
    CHECK(driver.bound_scale() == 1.0e6);
    CHECK_FALSE(driver.run().clean());

    CHECK_THROWS_AS(driver.set_bound_scale(0.0), std::invalid_argument);
    CHECK_THROWS_AS(driver.set_bound_scale(-1.0), std::invalid_argument);
}

TEST_CASE("BisectDriver - a program with nothing to read says so", "[ComputeGraph][BisectDriver]") {
    // No interface output means nothing to compare, and reporting that as "clean" would be a
    // driver that passes every pipeline it cannot see.
    cg::BisectDriver driver([](cg::Graph &graph) {
        auto                  &T = graph.declare_runtime_tensor<double>("T", {kN, kN}, /*intermediate=*/true);
        cg::CaptureGuard const capture(graph);
        cg::scale(2.0, &T);
    });

    auto const report = driver.run();

    CHECK_FALSE(report.baseline_ok);
    CHECK_FALSE(report.clean());
    INFO(report.to_string());
    CHECK(report.to_string().find("nothing to compare") != std::string::npos);
}

TEST_CASE("BisectDriver - an empty builder is refused", "[ComputeGraph][BisectDriver]") {
    CHECK_THROWS_AS(cg::BisectDriver(cg::BisectDriver::Builder{}), std::invalid_argument);
}

TEST_CASE("tier_bound - each tier's promise as a number", "[ComputeGraph][BisectDriver]") {
    constexpr double eps = std::numeric_limits<double>::epsilon();

    CHECK(cg::tier_bound(cg::PassTier::BitwiseExact, eps) == 0.0);
    CHECK(cg::tier_bound(cg::PassTier::Tuning, eps) == 0.0);

    // Above every measured member of the tier (the worst is 6.5e-16) and far below what a wrong
    // contraction produces, which is the separation the constant has to sit inside.
    double const re_associating = cg::tier_bound(cg::PassTier::ReAssociating, eps);
    CHECK(re_associating > 1.0e-14);
    CHECK(re_associating < 1.0e-9);

    // A lossy pass declares its own tolerance; a constant here would be a lie.
    CHECK(std::isinf(cg::tier_bound(cg::PassTier::Lossy, eps)));

    // The bound follows the element type, so a float program is not held to a double's epsilon.
    CHECK(cg::tier_bound(cg::PassTier::ReAssociating, std::numeric_limits<float>::epsilon()) > re_associating);
}
