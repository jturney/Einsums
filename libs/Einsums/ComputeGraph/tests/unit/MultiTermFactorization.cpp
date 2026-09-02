//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// MultiTermFactorization: contraction orders and shared intermediates, chosen together.
//
// The shape underneath every case below, and why it is the one:
//
//   T1[i,l] = A[i,k] B[k,l]      R1[i,j] = T1[i,l] C[l,j]     // bracketed left to right
//   T2[k,j] = B[k,l] D[l,j]      R2[i,j] = A[i,k] T2[k,j]     // bracketed right to left
//
// Two products of three tensors, written with opposite bracketing, sharing A and B. Nothing here
// is a duplicate node, so CSE sees nothing; nothing here is a sum into one output, so
// DistributiveFactoring sees nothing; and each chain on its own is already what
// ContractionPlanning would leave it as. The shared (A B) only exists once both terms are looked
// at together, which is the whole claim of this pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/MultiTermFactorization.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// Chosen so that (A B) C is the cheaper bracketing for BOTH terms, which is what makes the shared
// (A B) worth building once: i*l*(k+j) < k*j*(l+i). With these it is 3*3*24 against 144*6.
constexpr size_t kI = 3, kK = 12, kL = 3, kJ = 12;

struct Chain {
    RuntimeTensor<double> A, B, C, D, R1, R2;
};

Chain make_chain(unsigned seed) {
    einsums::seed_random(seed);
    return Chain{.A  = create_random_tensor<double>("A", kI, kK),
                 .B  = create_random_tensor<double>("B", kK, kL),
                 .C  = create_random_tensor<double>("C", kL, kJ),
                 .D  = create_random_tensor<double>("D", kL, kJ),
                 .R1 = create_zero_tensor<double>("R1", kI, kJ),
                 .R2 = create_zero_tensor<double>("R2", kI, kJ)};
}

/// Capture the two chains of the file note into @p graph.
void capture_chains(cg::Graph &graph, Chain &t) {
    auto &T1 = graph.declare_runtime_tensor<double>("T1", {kI, kL}, /*intermediate=*/true);
    auto &T2 = graph.declare_runtime_tensor<double>("T2", {kK, kJ}, /*intermediate=*/true);

    cg::CaptureGuard const guard(graph);
    cg::einsum("i,l <- i,k ; k,l", 0.0, &T1, 1.0, t.A, t.B);
    cg::einsum("i,j <- i,l ; l,j", 0.0, &t.R1, 1.0, T1, t.C);
    cg::einsum("k,j <- k,l ; l,j", 0.0, &T2, 1.0, t.B, t.D);
    cg::einsum("i,j <- i,k ; k,j", 0.0, &t.R2, 1.0, t.A, T2);
}

std::shared_ptr<cg::passes::MultiTermFactorization> searching_pass() {
    auto pass = std::make_shared<cg::passes::MultiTermFactorization>();
    pass->set_search_enabled(true);
    return pass;
}

/// Run the chains, optionally through the search, and hand back both outputs.
std::pair<RuntimeTensor<double>, RuntimeTensor<double>> run_chains(unsigned seed, bool with_search) {
    Chain     t = make_chain(seed);
    cg::Graph graph("chains");
    capture_chains(graph, t);

    cg::PassManager pm;
    if (with_search) {
        pm.add(searching_pass());
    }
    pm.add<cg::passes::Materialization>();
    graph.apply(pm);
    graph.execute();
    return {std::move(t.R1), std::move(t.R2)};
}

/// A pass that does nothing but report the allowance it was handed.
class BudgetProbe : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "BudgetProbe"; }

    bool run(cg::Graph &) override {
        _unlimited = budget().unlimited();
        _remaining = budget().remaining();
        _seen      = true;
        return false;
    }

    [[nodiscard]] bool                      seen() const { return _seen; }
    [[nodiscard]] bool                      unlimited() const { return _unlimited; }
    [[nodiscard]] std::chrono::milliseconds remaining() const { return _remaining; }

  private:
    bool                      _seen{false};
    bool                      _unlimited{true};
    std::chrono::milliseconds _remaining{0};
};

} // namespace

TEST_CASE("MultiTermFactorization - the shared product neither term asked for", "[ComputeGraph][MultiTermFactorization]") {
    Chain     t = make_chain(3);
    cg::Graph graph("share");
    capture_chains(graph, t);

    auto            pass = searching_pass();
    cg::PassManager pm;
    pm.add(pass);
    REQUIRE(pm.run(graph));

    // Both author-named intermediates dissolve, because the bracketing they encode is an artifact
    // of how the equations were written down rather than of what has to be computed.
    CHECK(pass->num_inlined() == 2);
    CHECK(pass->num_shared() == 1);
    CHECK(pass->num_rebracketed() == 2);
    CHECK_FALSE(pass->was_cut_off());

    // Three contractions where there were four: one shared product plus one per term.
    size_t contractions = 0;
    for (auto const &node : graph.nodes()) {
        contractions += node.kind == cg::OpKind::Einsum ? 1 : 0;
    }
    CHECK(contractions == 3);
}

TEST_CASE("MultiTermFactorization - the answer does not change", "[ComputeGraph][MultiTermFactorization]") {
    auto const [plain1, plain2]   = run_chains(11, /*with_search=*/false);
    auto const [search1, search2] = run_chains(11, /*with_search=*/true);

    // Re-associating, so a norm-relative bound rather than bit equality: the shared form sums the
    // same products in a different order and that is the tier's whole definition.
    for (size_t i = 0; i < kI; i++) {
        for (size_t j = 0; j < kJ; j++) {
            INFO("R[" << i << "," << j << "]");
            CHECK_THAT(search1(i, j), Catch::Matchers::WithinRel(plain1(i, j), 1.0e-12));
            CHECK_THAT(search2(i, j), Catch::Matchers::WithinRel(plain2(i, j), 1.0e-12));
        }
    }
}

TEST_CASE("MultiTermFactorization - the search is off unless it is asked for", "[ComputeGraph][MultiTermFactorization]") {
    Chain     t = make_chain(5);
    cg::Graph graph("off");
    capture_chains(graph, t);

    // No set_search_enabled, and the option defaults to false. A pass whose runtime is a function
    // of how many candidates a graph offers does not get to run because it happened to be linked.
    auto            pass = std::make_shared<cg::passes::MultiTermFactorization>();
    cg::PassManager pm;
    pm.add(pass);
    // The skip tally is what a decline is read through, and `explain` grows that section from
    // verbosity 2 up. A report that said only "no optimizations applied" cannot tell "already
    // optimal" from "switched off", which is the one distinction this case is about.
    pm.set_verbosity(2);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_shared() == 0);
    CHECK_FALSE(pass->search_enabled());

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("structural search is switched off") != std::string::npos);
}

TEST_CASE("MultiTermFactorization - the same graph gets the same plan", "[ComputeGraph][MultiTermFactorization]") {
    // A search whose answer varies between runs makes every measurement against it noise, which is
    // why the subset program walks masks in integer order and takes only strict improvements.
    std::vector<std::string> first;
    for (int trial = 0; trial < 3; trial++) {
        Chain     t = make_chain(7);
        cg::Graph graph("determinism");
        capture_chains(graph, t);

        auto            pass = searching_pass();
        cg::PassManager pm;
        pm.add(pass);
        pass->set_dump(true);
        REQUIRE(pm.run(graph));

        std::vector<std::string> labels;
        for (auto const &node : graph.nodes()) {
            labels.push_back(node.label);
        }
        if (trial == 0) {
            first = labels;
        }
        CHECK(labels == first);
    }
}

TEST_CASE("MultiTermFactorization - it reports what it did", "[ComputeGraph][MultiTermFactorization]") {
    Chain     t = make_chain(13);
    cg::Graph graph("report");
    capture_chains(graph, t);

    cg::PassManager pm;
    pm.add(searching_pass());
    REQUIRE(pm.run(graph));

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("MultiTermFactorization") != std::string::npos);
    CHECK(report.find("shared intermediate") != std::string::npos);
    CHECK(report.find("structural-algebraic") != std::string::npos);
}

TEST_CASE("MultiTermFactorization - a factor cap is a decline, not an approximation", "[ComputeGraph][MultiTermFactorization]") {
    Chain     t = make_chain(17);
    cg::Graph graph("cap");
    capture_chains(graph, t);

    auto pass = searching_pass();
    pass->set_max_factors(2); // both terms flatten to three factors
    cg::PassManager pm;
    pm.add(pass);
    pm.set_verbosity(2); // see the note in the switched-off case
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_shared() == 0);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("not a product this pass can model") != std::string::npos);
}

TEST_CASE("MultiTermFactorization - a graph with nothing to share is left alone", "[ComputeGraph][MultiTermFactorization]") {
    auto A = RuntimeTensor<double>(create_random_tensor<double>("A", kI, kK));
    auto B = RuntimeTensor<double>(create_random_tensor<double>("B", kK, kL));
    auto R = RuntimeTensor<double>(create_zero_tensor<double>("R", kI, kL));

    cg::Graph graph("nothing");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,l <- i,k ; k,l", 0.0, &R, 1.0, A, B);
        cg::einsum("i,l <- i,k ; k,l", 1.0, &R, 2.0, A, B);
    }

    // Two two-factor terms: nothing to re-bracket and no pair that is not the whole term.
    auto            pass = searching_pass();
    cg::PassManager pm;
    pm.add(pass);
    CHECK_FALSE(pm.run(graph));
    CHECK(pass->num_shared() == 0);
    CHECK(pass->num_rebracketed() == 0);
}

// ══════════════════════════════════════════════════════════════════════════════
// The budget, which this is the first pass to need
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SearchBudget - a default budget bounds nothing", "[ComputeGraph][MultiTermFactorization]") {
    cg::SearchBudget const unlimited;
    CHECK(unlimited.unlimited());
    CHECK_FALSE(unlimited.expired());

    // Zero is how a caller REQUESTS unlimited, so it must not read as an allowance already spent.
    cg::SearchBudget const zero{std::chrono::milliseconds{0}};
    CHECK(zero.unlimited());
    CHECK_FALSE(zero.expired());
}

TEST_CASE("SearchBudget - an allowance in the past is spent", "[ComputeGraph][MultiTermFactorization]") {
    cg::SearchBudget const spent{std::chrono::milliseconds{-1}};
    CHECK(spent.unlimited()); // negative is not a deadline, it is a caller asking for no bound

    cg::SearchBudget const generous{std::chrono::milliseconds{60000}};
    CHECK_FALSE(generous.unlimited());
    CHECK_FALSE(generous.expired());
    CHECK(generous.remaining() > std::chrono::milliseconds{0});
}

TEST_CASE("PassManager - the budget it hands out is the one it was set", "[ComputeGraph][MultiTermFactorization]") {
    cg::Graph graph("budget");
    {
        auto                  &S = graph.declare_runtime_tensor<double>("S", {2, 2}, true);
        cg::CaptureGuard const guard(graph);
        cg::scale(1.0, &S);
    }

    auto probe = std::make_shared<BudgetProbe>();
    {
        cg::PassManager pm;
        pm.add(probe);
        pm.set_optimizer_budget(5000);
        pm.run(graph);
        REQUIRE(probe->seen());
        CHECK_FALSE(probe->unlimited());
        CHECK(probe->remaining() > std::chrono::milliseconds{0});
    }
    {
        // An explicit zero is a request for no bound, and it must win over the option the same way
        // an explicit `enable` wins over `einsums:pass:disable`.
        auto            second = std::make_shared<BudgetProbe>();
        cg::PassManager pm;
        pm.add(second);
        pm.set_optimizer_budget(0);
        pm.run(graph);
        REQUIRE(second->seen());
        CHECK(second->unlimited());
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Shapes the three-factor case does not reach
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("MultiTermFactorization - a four-factor term emits its whole tree", "[ComputeGraph][MultiTermFactorization]") {
    // Four factors means the subset program actually has interior nodes to choose, and the
    // emission has to declare an intermediate for each one rather than only for the shared pair.
    constexpr size_t kM = 4;
    einsums::seed_random(23);
    RuntimeTensor<double> A = create_random_tensor<double>("A", kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kK, kL);
    RuntimeTensor<double> C = create_random_tensor<double>("C", kL, kM);
    RuntimeTensor<double> E = create_random_tensor<double>("E", kM, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kL, kJ);

    auto run = [&](bool with_search) {
        RuntimeTensor<double> R1 = create_zero_tensor<double>("R1", kI, kJ);
        RuntimeTensor<double> R2 = create_zero_tensor<double>("R2", kI, kJ);

        cg::Graph graph("four");
        {
            auto &T1 = graph.declare_runtime_tensor<double>("T1", {kI, kL}, true);
            auto &T2 = graph.declare_runtime_tensor<double>("T2", {kI, kM}, true);
            auto &T3 = graph.declare_runtime_tensor<double>("T3", {kK, kJ}, true);

            cg::CaptureGuard const guard(graph);
            cg::einsum("i,l <- i,k ; k,l", 0.0, &T1, 1.0, A, B);
            cg::einsum("i,m <- i,l ; l,m", 0.0, &T2, 1.0, T1, C);
            cg::einsum("i,j <- i,m ; m,j", 0.0, &R1, 1.0, T2, E);
            cg::einsum("k,j <- k,l ; l,j", 0.0, &T3, 1.0, B, D);
            cg::einsum("i,j <- i,k ; k,j", 0.0, &R2, 1.0, A, T3);
        }

        cg::PassManager pm;
        if (with_search) {
            pm.add(searching_pass());
        }
        pm.add<cg::passes::Materialization>();
        graph.apply(pm);
        graph.execute();
        return std::pair{std::move(R1), std::move(R2)};
    };

    auto const [plain1, plain2]   = run(false);
    auto const [search1, search2] = run(true);

    for (size_t i = 0; i < kI; i++) {
        for (size_t j = 0; j < kJ; j++) {
            INFO("[" << i << "," << j << "]");
            CHECK_THAT(search1(i, j), Catch::Matchers::WithinRel(plain1(i, j), 1.0e-12));
            CHECK_THAT(search2(i, j), Catch::Matchers::WithinRel(plain2(i, j), 1.0e-12));
        }
    }
}

TEST_CASE("MultiTermFactorization - a batched letter stays outermost", "[ComputeGraph][MultiTermFactorization]") {
    // A letter carried by every operand is a batch dimension: it is neither contracted nor free to
    // one side, so it must stay in every intermediate the tree builds. Getting that wrong is not a
    // slower plan, it is a wrong shape, which is why it has a case of its own.
    constexpr size_t kB = 3;
    einsums::seed_random(29);
    RuntimeTensor<double> A = create_random_tensor<double>("A", kB, kI, kK);
    RuntimeTensor<double> B = create_random_tensor<double>("B", kB, kK, kL);
    RuntimeTensor<double> C = create_random_tensor<double>("C", kB, kL, kJ);
    RuntimeTensor<double> D = create_random_tensor<double>("D", kB, kL, kJ);

    auto run = [&](bool with_search) {
        RuntimeTensor<double> R1 = create_zero_tensor<double>("R1", kB, kI, kJ);
        RuntimeTensor<double> R2 = create_zero_tensor<double>("R2", kB, kI, kJ);

        cg::Graph graph("batched");
        {
            auto &T1 = graph.declare_runtime_tensor<double>("T1", {kB, kI, kL}, true);
            auto &T2 = graph.declare_runtime_tensor<double>("T2", {kB, kK, kJ}, true);

            cg::CaptureGuard const guard(graph);
            cg::einsum("b,i,l <- b,i,k ; b,k,l", 0.0, &T1, 1.0, A, B);
            cg::einsum("b,i,j <- b,i,l ; b,l,j", 0.0, &R1, 1.0, T1, C);
            cg::einsum("b,k,j <- b,k,l ; b,l,j", 0.0, &T2, 1.0, B, D);
            cg::einsum("b,i,j <- b,i,k ; b,k,j", 0.0, &R2, 1.0, A, T2);
        }

        cg::PassManager pm;
        if (with_search) {
            pm.add(searching_pass());
        }
        pm.add<cg::passes::Materialization>();
        graph.apply(pm);
        graph.execute();
        return std::pair{std::move(R1), std::move(R2)};
    };

    auto const [plain1, plain2]   = run(false);
    auto const [search1, search2] = run(true);

    for (size_t b = 0; b < kB; b++) {
        for (size_t i = 0; i < kI; i++) {
            for (size_t j = 0; j < kJ; j++) {
                INFO("[" << b << "," << i << "," << j << "]");
                CHECK_THAT(search1(b, i, j), Catch::Matchers::WithinRel(plain1(b, i, j), 1.0e-12));
                CHECK_THAT(search2(b, i, j), Catch::Matchers::WithinRel(plain2(b, i, j), 1.0e-12));
            }
        }
    }
}
