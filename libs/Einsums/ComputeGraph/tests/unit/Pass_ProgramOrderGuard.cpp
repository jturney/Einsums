//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ProgramOrderGuard.cpp
/// @brief What the program-order guard covers, and what it deliberately does not.
///
/// Position IS program order in this IR, so a pass that moves a writer past a
/// surviving reader silently changes which value that reader observes.
/// ``PassManager::run`` guards against it by comparing an observed-writes map
/// taken before and after each pass.
///
/// A guard nobody has watched fire is not a guard, which is what the
/// deliberately-broken pass below is for: it performs the exact rewrite the
/// guard exists to reject.
///
/// The third case pins a KNOWN GAP rather than a behaviour anyone wants. The
/// guard walks the top-level graph only, while fourteen passes opt into
/// ``recurse_into_subgraphs()`` and rewrite loop bodies. Extending it over the
/// sub-graph tree was tried on 2026-08-08 and withdrawn: its "was there an
/// earlier writer of this buffer in scan order" test is a proxy for "this read
/// has a producer", and Reorder trips the proxy without changing semantics -
/// sixteen control-flow fuzz programs flagged, all 798 numerically identical
/// with the check disabled. When the guard learns to reason from real
/// dependence edges instead of scan position, the third case flips from
/// REQUIRE_NOTHROW to REQUIRE_THROWS_WITH("iter/body") and this comment goes.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Moves the FIRST node to the back, which is the rewrite the guard rejects
/// whenever a later node reads what that node wrote.
class WriterToBack : public cg::OptimizerPass {
  public:
    explicit WriterToBack(bool recurse) : _recurse(recurse) {}

    [[nodiscard]] std::string name() const override { return "WriterToBack"; }
    [[nodiscard]] bool        recurse_into_subgraphs() const override { return _recurse; }

    bool run(cg::Graph &graph) override {
        auto &nodes = graph.nodes();
        if (nodes.size() < 2 || _done) {
            return false;
        }
        auto moved = std::move(nodes.front());
        nodes.erase(nodes.begin());
        nodes.push_back(std::move(moved));
        _done = true;
        return true;
    }

  private:
    bool _recurse;
    bool _done{false};
};

/// Touches nothing, so the guard must stay quiet: the known-true case, without
/// which "throws" would be indistinguishable from "throws at everything".
class NoOpButReports : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "NoOpButReports"; }
    [[nodiscard]] bool        recurse_into_subgraphs() const override { return true; }
    bool                      run(cg::Graph                      &/*graph*/) override { return false; }
};

/// C = A * B, then D = C * B. The second node READS what the first wrote, so
/// moving the first node to the back inverts a genuine producer/consumer pair.
void capture_producer_consumer(cg::Graph &into, RuntimeTensor<double> &A, RuntimeTensor<double> &B, RuntimeTensor<double> &C,
                               RuntimeTensor<double> &D) {
    cg::CaptureGuard const guard(into);
    cg::einsum("i,j <- i,k ; k,j", 0.0, &C, 1.0, A, B);
    cg::einsum("i,j <- i,k ; k,j", 0.0, &D, 1.0, C, B);
}

struct Fixture {
    Tensor<double, 2>     A{create_random_tensor<double>("A", 4, 4)};
    Tensor<double, 2>     B{create_random_tensor<double>("B", 4, 4)};
    RuntimeTensor<double> A_rt{A};
    RuntimeTensor<double> B_rt{B};
    RuntimeTensor<double> C_rt{"C", std::vector<size_t>{4, 4}};
    RuntimeTensor<double> D_rt{"D", std::vector<size_t>{4, 4}};
};

} // namespace

TEST_CASE("Program-order guard rejects a writer moved behind its reader", "[ComputeGraph][Passes][ProgramOrder]") {
    Fixture   f;
    cg::Graph graph("flat");
    capture_producer_consumer(graph, f.A_rt, f.B_rt, f.C_rt, f.D_rt);

    cg::PassManager pm;
    pm.add(std::make_shared<WriterToBack>(/*recurse=*/false));
    REQUIRE_THROWS_WITH(graph.apply(pm), Catch::Matchers::ContainsSubstring("broke program order"));
}

TEST_CASE("Program-order guard stays quiet on a pass that changes nothing", "[ComputeGraph][Passes][ProgramOrder]") {
    Fixture   f;
    cg::Graph graph("outer");
    auto     &body = graph.add_loop("iter", 2, [](size_t it) { return it + 1 < 2; });
    capture_producer_consumer(body, f.A_rt, f.B_rt, f.C_rt, f.D_rt);

    cg::PassManager pm;
    pm.add(std::make_shared<NoOpButReports>());
    REQUIRE_NOTHROW(graph.apply(pm));
}

TEST_CASE("KNOWN GAP: the guard does not see inside a loop body", "[ComputeGraph][Passes][ProgramOrder]") {
    Fixture   f;
    cg::Graph graph("outer");
    auto     &body = graph.add_loop("iter", 3, [](size_t it) { return it + 1 < 3; });
    capture_producer_consumer(body, f.A_rt, f.B_rt, f.C_rt, f.D_rt);

    // The parent holds one Loop node and no reader/writer pair at all, so the
    // broken rewrite lands entirely in the body.
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(body.num_nodes() == 2);

    cg::PassManager pm;
    pm.add(std::make_shared<WriterToBack>(/*recurse=*/true));

    // Asserting the CURRENT behaviour, which is a gap and not a guarantee: the
    // same rewrite that throws at the top level is accepted here. See the file
    // header for why the sub-graph sweep was withdrawn and what has to change
    // before this becomes REQUIRE_THROWS_WITH("iter/body").
    REQUIRE_NOTHROW(graph.apply(pm));
}
