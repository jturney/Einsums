//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// How many times a setup body has run, read off the tensor it accumulates into.
///
/// The bodies below all add ONE to every element of their output, so the value in a slot is
/// the run count and nothing has to be counted outside the graph. A lambda counter would
/// work for an eagerly built node and not for one a save and a load rebuilt, and half these
/// cases are about exactly that.
double run_count(Tensor<double, 2> const &t) {
    return t(0, 0);
}

} // namespace

TEST_CASE("Setup - the body runs on the first replay and no replay after it", "[ComputeGraph][Setup]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_once");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }

    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
    REQUIRE(out(0, 0) == Catch::Approx(1.0));

    graph.execute();
    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
    REQUIRE(out(0, 0) == Catch::Approx(1.0));
}

TEST_CASE("Setup - the parent node reports the body's reads and writes as its own", "[ComputeGraph][Setup]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;

    cg::Graph graph("setup_effective_io");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }

    // The Setup node lists nothing of its own: its body was captured after the node existed,
    // exactly as a loop body is. What orders it against the rest of the graph is the subtree
    // expansion, and this is the assertion that the expansion reaches through a setup body -
    // without it the consumer of a fitted tensor has no edge to its producer and the sort is
    // free to run them in either order.
    cg::Node const *setup_node = nullptr;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Setup) {
            setup_node = &node;
        }
    }
    REQUIRE(setup_node != nullptr);
    REQUIRE(setup_node->inputs.empty());
    REQUIRE(setup_node->outputs.empty());

    auto const [eff_in, eff_out] = graph.effective_io(*setup_node);
    cg::TensorId const ones_id   = graph.live_tensor_id_by_ptr(&ones, {});
    cg::TensorId const fit_id    = graph.live_tensor_id_by_ptr(&fit, {});
    REQUIRE(std::find(eff_in.begin(), eff_in.end(), ones_id) != eff_in.end());
    REQUIRE(std::find(eff_out.begin(), eff_out.end(), fit_id) != eff_out.end());
}

TEST_CASE("Setup - a body placed before its consumers runs before them", "[ComputeGraph][Setup]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_placed");
    // The CONSUMER is captured first, which is the situation a pass that introduces a fitting
    // is always in: the nodes that will read the factors already exist. Appending the setup
    // behind them puts a writer behind its readers, and the dependency sort reads that as an
    // anti-dependency and orders it exactly the wrong way round.
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }
    {
        auto                  &body = graph.add_setup_at("fit", 0);
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }

    REQUIRE(graph.nodes().front().kind == cg::OpKind::Setup);

    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
    REQUIRE(out(0, 0) == Catch::Approx(1.0));
}

TEST_CASE("Setup - a bind puts the body back to work", "[ComputeGraph][Setup][Bind]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_rebind");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }

    graph.execute();
    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));

    // Fresh storage for the same interface: a different problem as far as anything here can
    // tell, so the factors on hand are about the previous one.
    auto ones2  = create_zero_tensor<double>("ones", 2, 2);
    auto fit2   = create_zero_tensor<double>("fit", 2, 2);
    auto out2   = create_zero_tensor<double>("out", 2, 2);
    ones2(0, 0) = 1.0;
    ones2(0, 1) = 1.0;
    ones2(1, 0) = 1.0;
    ones2(1, 1) = 1.0;

    graph.bind("ones", ones2, "fit", fit2, "out", out2);
    graph.execute();
    REQUIRE(run_count(fit2) == Catch::Approx(1.0));
    REQUIRE(out2(0, 0) == Catch::Approx(1.0));
    // The first problem's slot was not touched again.
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
}

TEST_CASE("Setup - a matching key lets a re-bind skip the refit, and a different one does not", "[ComputeGraph][Setup][Bind]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_key");
    graph.set_setup_key("problem-a");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }

    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
    REQUIRE(graph.setup_key() == "problem-a");

    // Re-bind under the SAME key. The caller is saying this is the same problem, so the
    // factors already in `fit2` stand and the body is not run into them.
    auto ones2  = create_zero_tensor<double>("ones", 2, 2);
    auto fit2   = create_zero_tensor<double>("fit", 2, 2);
    auto out2   = create_zero_tensor<double>("out", 2, 2);
    ones2(0, 0) = 1.0;
    ones2(0, 1) = 1.0;
    ones2(1, 0) = 1.0;
    ones2(1, 1) = 1.0;
    fit2(0, 0)  = 1.0; // the caller's own copy of the previous fit
    fit2(0, 1)  = 1.0;
    fit2(1, 0)  = 1.0;
    fit2(1, 1)  = 1.0;

    graph.bind("ones", ones2, "fit", fit2, "out", out2);
    graph.execute();
    REQUIRE(run_count(fit2) == Catch::Approx(1.0)); // not refitted to 2
    REQUIRE(out2(0, 0) == Catch::Approx(1.0));

    // A different key is a different problem and the body runs again.
    graph.set_setup_key("problem-b");
    graph.bind("ones", ones2, "fit", fit2, "out", out2);
    graph.execute();
    REQUIRE(run_count(fit2) == Catch::Approx(2.0));
}

TEST_CASE("Setup - run_setup pulls the body forward, and force ignores both guards", "[ComputeGraph][Setup]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_run");
    REQUIRE_FALSE(graph.has_setup());
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }
    REQUIRE(graph.has_setup());

    graph.run_setup();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));

    // Idempotent: the same guards apply here as in a replay.
    graph.run_setup();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));

    // And the replay that follows does not repeat the work either.
    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));

    // The escape hatch for a caller who changed a bound tensor's CONTENTS, which nothing
    // can observe.
    graph.run_setup(/*force=*/true);
    REQUIRE(run_count(fit) == Catch::Approx(2.0));
}

TEST_CASE("Setup - a key alone does not skip a body that has never run", "[ComputeGraph][Setup]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_first_run");
    graph.set_setup_key("problem-a");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }

    // The computed key starts empty, and the guard compares against it rather than merely
    // checking that a key was set: a graph that has fitted nothing must fit.
    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));
}

TEST_CASE("Setup - the default pass manager does not free what a setup body produced", "[ComputeGraph][Setup][Passes]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_free");
    // A GRAPH-OWNED deferred intermediate, which is what a factorization pass will create
    // for its factors and exactly the shape FreeInsertion reclaims: written once, read once,
    // never touched again within one replay. Reclaiming it is correct for an ordinary
    // intermediate and wrong for this one, because the second replay does not rewrite it.
    auto &fit = graph.scratch_zero<double, 2>("fit", 2, 2);
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    graph.execute();
    REQUIRE(out(0, 0) == Catch::Approx(1.0));

    // The replay that would read released storage if the fit had been freed.
    out.zero();
    graph.execute();
    REQUIRE(out(0, 0) == Catch::Approx(1.0));
}

TEST_CASE("Setup - a saved graph reloads having fitted nothing, and fits once", "[ComputeGraph][Setup][SaveLoad]") {
    auto ones  = create_zero_tensor<double>("ones", 2, 2);
    auto fit   = create_zero_tensor<double>("fit", 2, 2);
    auto out   = create_zero_tensor<double>("out", 2, 2);
    ones(0, 0) = 1.0;
    ones(0, 1) = 1.0;
    ones(1, 0) = 1.0;
    ones(1, 1) = 1.0;

    cg::Graph graph("setup_saved");
    {
        auto                  &body = graph.add_setup("fit");
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fit);
    }
    graph.execute();
    REQUIRE(run_count(fit) == Catch::Approx(1.0));

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());

    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());

    auto ones2  = create_zero_tensor<double>("ones", 2, 2);
    auto fit2   = create_zero_tensor<double>("fit", 2, 2);
    auto out2   = create_zero_tensor<double>("out", 2, 2);
    ones2(0, 0) = 1.0;
    ones2(0, 1) = 1.0;
    ones2(1, 0) = 1.0;
    ones2(1, 1) = 1.0;

    loaded->bind("ones", ones2, "fit", fit2, "out", out2);
    loaded->execute();
    // The loaded graph had not fitted anything, whatever the saving process had done.
    REQUIRE(run_count(fit2) == Catch::Approx(1.0));
    REQUIRE(out2(0, 0) == Catch::Approx(1.0));

    loaded->execute();
    REQUIRE(run_count(fit2) == Catch::Approx(1.0));
}

TEST_CASE("Setup - a body carrying an unwritable node is refused by the serializability report", "[ComputeGraph][Setup][SaveLoad]") {
    auto ones = create_zero_tensor<double>("ones", 2, 2);
    auto fit  = create_zero_tensor<double>("fit", 2, 2);

    cg::Graph graph("setup_blocker");
    auto     &body = graph.add_setup("fit");
    {
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, ones, &fit);
    }
    // A closure-conditioned loop inside the setup body: the blocker walk has to reach
    // through a Setup body exactly as it reaches through a Loop body.
    {
        auto                  &inner = body.add_loop("inner", 2, [](std::size_t it) { return it < 1; });
        cg::CaptureGuard const inner_guard(inner);
        cg::scale(1.0, &fit);
    }

    auto const blockers = graph.serializability_report();
    REQUIRE_FALSE(blockers.empty());
    bool named_the_setup_path = false;
    for (auto const &blocker : blockers) {
        if (blocker.subgraph_path.find("setup(") != std::string::npos) {
            named_the_setup_path = true;
        }
    }
    REQUIRE(named_the_setup_path);
}
