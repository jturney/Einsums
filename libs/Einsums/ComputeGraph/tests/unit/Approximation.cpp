//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A graph with one contraction, so there is a manifest to name outputs in.
cg::Graph make_graph(std::string const &name, Tensor<double, 2> &A, Tensor<double, 2> &B, Tensor<double, 2> &C) {
    cg::Graph graph(name);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    return graph;
}

cg::ApproximationRecord relative(std::string pass, double bound, std::vector<std::string> outputs = {}) {
    return cg::ApproximationRecord{.pass_name = std::move(pass),
                                   .tolerance = bound,
                                   .effect    = cg::ApproximationEffect::NormRelative,
                                   .bound     = bound,
                                   .outputs   = std::move(outputs),
                                   .spaces    = {},
                                   .setup     = {}};
}

cg::ApproximationRecord absolute(std::string pass, double bound, std::vector<std::string> outputs = {}) {
    return cg::ApproximationRecord{.pass_name = std::move(pass),
                                   .tolerance = bound,
                                   .effect    = cg::ApproximationEffect::ElementWise,
                                   .bound     = bound,
                                   .outputs   = std::move(outputs),
                                   .spaces    = {},
                                   .setup     = {}};
}

/// A pass that does nothing but try to spend accuracy, so the decline path has a caller.
class SpendingPass : public cg::OptimizerPass {
  public:
    explicit SpendingPass(cg::ApproximationRecord record) : _record(std::move(record)) {}

    [[nodiscard]] std::string name() const override { return "SpendingPass"; }

    bool run(cg::Graph &graph) override {
        applied = approximate(graph, _record);
        return applied;
    }

    bool applied{false};

  private:
    cg::ApproximationRecord _record;
};

} // namespace

// ── Composition ────────────────────────────────────────────────────────────

TEST_CASE("Approximation - an absolute effect composes by the triangle inequality", "[ComputeGraph][Approximation]") {
    REQUIRE(cg::compose_approximation(cg::ApproximationEffect::ElementWise, 1e-3, 2e-3) == Catch::Approx(3e-3));
    REQUIRE(cg::compose_approximation(cg::ApproximationEffect::EnergyLike, 1e-3, 2e-3) == Catch::Approx(3e-3));
}

TEST_CASE("Approximation - a relative effect composes with the product term, not without it", "[ComputeGraph][Approximation]") {
    // The whole point of the rule. The second pass's error is relative to the ALREADY
    // perturbed result, so a plain sum is an under-estimate, and an accuracy contract that
    // under-estimates is one that reports a graph as inside a budget it is outside of.
    double const composed = cg::compose_approximation(cg::ApproximationEffect::NormRelative, 0.1, 0.2);
    REQUIRE(composed == Catch::Approx(0.32));
    REQUIRE(composed > 0.1 + 0.2);
}

// ── Recording and reading back ─────────────────────────────────────────────

TEST_CASE("Approximation - a graph with no records is in exact mode", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("exact", A, B, C);

    REQUIRE(graph.approximations().empty());
    auto const tolerance = graph.approximation_tolerance();
    REQUIRE(tolerance.relative == 0.0);
    REQUIRE(tolerance.absolute == 0.0);
}

TEST_CASE("Approximation - a record widens the side its effect is stated in", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("widen", A, B, C);

    graph.note_approximation(relative("DF", 1e-5));
    REQUIRE(graph.approximation_tolerance().relative == Catch::Approx(1e-5));
    REQUIRE(graph.approximation_tolerance().absolute == 0.0);

    // Absolute and relative bounds do not convert, so a second record of the other kind
    // widens the OTHER side rather than being folded into the first. This is the honest
    // representation of a result that is off by so much in norm and so much per element,
    // and it is why the two are allowed to coexist on one output.
    graph.note_approximation(absolute("Rounding", 1e-8));
    REQUIRE(graph.approximation_tolerance("C").relative == Catch::Approx(1e-5));
    REQUIRE(graph.approximation_tolerance("C").absolute == Catch::Approx(1e-8));
}

TEST_CASE("Approximation - records on disjoint outputs do not compose with each other", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("disjoint", A, B, C);

    graph.note_approximation(relative("First", 1e-4, {"A"}));
    graph.note_approximation(relative("Second", 2e-4, {"B"}));

    REQUIRE(graph.accuracy_spent(cg::ApproximationEffect::NormRelative, "A") == Catch::Approx(1e-4));
    REQUIRE(graph.accuracy_spent(cg::ApproximationEffect::NormRelative, "B") == Catch::Approx(2e-4));
    // The graph-wide question is the worst case, which counts both.
    REQUIRE(graph.accuracy_spent(cg::ApproximationEffect::NormRelative) ==
            Catch::Approx(cg::compose_approximation(cg::ApproximationEffect::NormRelative, 1e-4, 2e-4)));
}

TEST_CASE("Approximation - a record naming no outputs applies to all of them", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("everywhere", A, B, C);

    graph.note_approximation(relative("Everywhere", 1e-4));
    REQUIRE(graph.accuracy_spent(cg::ApproximationEffect::NormRelative, "C") == Catch::Approx(1e-4));
    REQUIRE(graph.accuracy_spent(cg::ApproximationEffect::NormRelative, "anything_at_all") == Catch::Approx(1e-4));
}

// ── Refusals ───────────────────────────────────────────────────────────────

TEST_CASE("Approximation - a bound that is not a number is refused", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("bad_bound", A, B, C);

    REQUIRE_FALSE(graph.can_approximate(relative("Negative", -1.0)).empty());
    REQUIRE_FALSE(graph.can_approximate(relative("Infinite", std::numeric_limits<double>::infinity())).empty());
    REQUIRE_THROWS_WITH(graph.note_approximation(relative("Negative", -1.0)),
                        Catch::Matchers::ContainsSubstring("how large its effect is"));
}

TEST_CASE("Approximation - two effects coexist until a budget claims to cap them", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("mixed", A, B, C);

    graph.note_approximation(relative("DF", 1e-5, {"C"}));

    // No budget: nothing needs a single composed number, composition is per effect, and the
    // tolerance carries the two sides separately. Refusing here would make an ordinary
    // pipeline (a factorization then a precision change) unexpressible for no gain.
    REQUIRE(graph.can_approximate(absolute("THC", 1e-6, {"C"})).empty());
    graph.note_approximation(absolute("THC", 1e-6, {"C"}));
    REQUIRE(graph.approximation_tolerance("C").relative == Catch::Approx(1e-5));
    REQUIRE(graph.approximation_tolerance("C").absolute == Catch::Approx(1e-6));

    // A budget is a claim to be a cap, and it can only cap its own units. A third record
    // over the same output is refused, naming the record the budget does not cover, rather
    // than admitted under a cap that governs half the error.
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-2);
    std::string const reason = graph.can_approximate(relative("More", 1e-6, {"C"}));
    REQUIRE_FALSE(reason.empty());
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("does not cap"));
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("THC"));

    // Over a DIFFERENT output there is nothing uncapped to trip over.
    REQUIRE(graph.can_approximate(relative("More", 1e-6, {"A"})).empty());
}

TEST_CASE("Approximation - a budget refuses the record that would exceed it", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("budget", A, B, C);

    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-4);
    REQUIRE(graph.accuracy_budget_value() == Catch::Approx(1e-4));

    REQUIRE(graph.can_approximate(relative("Small", 6e-5, {"C"})).empty());
    graph.note_approximation(relative("Small", 6e-5, {"C"}));

    // 6e-5 composed with 6e-5 is over 1e-4, so the second one does not fit even though it
    // would have fitted on its own.
    std::string const reason = graph.can_approximate(relative("Second", 6e-5, {"C"}));
    REQUIRE_FALSE(reason.empty());
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("budget"));
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("Second"));

    // On a different output the budget is untouched.
    REQUIRE(graph.can_approximate(relative("Elsewhere", 6e-5, {"A"})).empty());
}

TEST_CASE("Approximation - a budget in other units is not one this pass can spend against", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("units", A, B, C);

    graph.set_accuracy_budget(cg::ApproximationEffect::ElementWise, 1e-6);
    std::string const reason = graph.can_approximate(relative("DF", 1e-9));
    REQUIRE_FALSE(reason.empty());
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("norm-relative"));
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("element-wise"));

    graph.clear_accuracy_budget();
    REQUIRE(graph.can_approximate(relative("DF", 1e-9)).empty());
}

TEST_CASE("Approximation - a budget that is not a bound is refused when it is set", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("bad_budget", A, B, C);

    REQUIRE_THROWS_WITH(graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, -1.0),
                        Catch::Matchers::ContainsSubstring("not a bound"));
}

// ── The pass-facing spelling ───────────────────────────────────────────────

TEST_CASE("Approximation - a pass declines through the skip tally rather than by throwing", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("declines", A, B, C);

    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-9);

    SpendingPass pass(relative("SpendingPass", 1e-3));
    REQUIRE_FALSE(pass.run(graph));
    REQUIRE_FALSE(pass.applied);
    REQUIRE(graph.approximations().empty());

    auto const skips = pass.skip_reasons();
    REQUIRE_FALSE(skips.empty());
    REQUIRE_THAT(skips.front().first, Catch::Matchers::ContainsSubstring("lossy"));

    // The same pass under a budget it fits inside applies and records.
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-2);
    SpendingPass affordable(relative("SpendingPass", 1e-3));
    REQUIRE(affordable.run(graph));
    REQUIRE(graph.approximations().size() == 1);
}

// ── Saving ─────────────────────────────────────────────────────────────────

TEST_CASE("Approximation - records survive a save and a load, in order", "[ComputeGraph][Approximation][SaveLoad]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("saved", A, B, C);

    graph.note_approximation(cg::ApproximationRecord{.pass_name = "AutoDF",
                                                     .tolerance = 1e-5,
                                                     .effect    = cg::ApproximationEffect::NormRelative,
                                                     .bound     = 4e-6,
                                                     .outputs   = {"C"},
                                                     .spaces    = {},
                                                     .setup     = "df_fit"});
    graph.note_approximation(relative("AutoTHC", 2e-6, {"C"}));

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());

    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->approximations().size() == 2);

    // Order is content, not presentation: a relative composition is not commutative, so a
    // reader re-deriving the composed bound has to see them in the order they were applied.
    REQUIRE(loaded->approximations()[0].pass_name == "AutoDF");
    REQUIRE(loaded->approximations()[0].tolerance == Catch::Approx(1e-5));
    REQUIRE(loaded->approximations()[0].bound == Catch::Approx(4e-6));
    REQUIRE(loaded->approximations()[0].effect == cg::ApproximationEffect::NormRelative);
    REQUIRE(loaded->approximations()[0].outputs == std::vector<std::string>{"C"});
    REQUIRE(loaded->approximations()[0].setup == "df_fit");
    REQUIRE(loaded->approximations()[1].pass_name == "AutoTHC");

    REQUIRE(loaded->approximation_tolerance("C").relative == Catch::Approx(graph.approximation_tolerance("C").relative));
}

TEST_CASE("Approximation - a record enters the content hash and a budget does not", "[ComputeGraph][Approximation][SaveLoad]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("hashed", A, B, C);

    std::uint64_t const bare = graph.content_hash();

    // A budget is what a caller was willing to spend, which is a property of the run rather
    // than of the graph. It is not saved, so it cannot move the hash.
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-3);
    REQUIRE(graph.content_hash() == bare);

    // A record is a statement about what the graph now computes, so it does.
    graph.note_approximation(relative("DF", 1e-5));
    REQUIRE(graph.content_hash() != bare);
}

TEST_CASE("Approximation - a load installs the file's history whatever this process budgets", "[ComputeGraph][Approximation][SaveLoad]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("history", A, B, C);

    graph.note_approximation(relative("Generous", 1e-2));
    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());

    // A file says what WAS applied. Re-checking it against this process's budget would
    // refuse a graph that was perfectly legal where it was built, which is a loader acting
    // on provenance rather than reading it.
    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());
    loaded->set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-9);
    REQUIRE(loaded->approximations().size() == 1);
    REQUIRE(loaded->approximations()[0].bound == Catch::Approx(1e-2));

    // The budget still governs anything applied FROM HERE.
    REQUIRE_FALSE(loaded->can_approximate(relative("Another", 1e-6)).empty());
}

TEST_CASE("Approximation - a move carries the records and the budget", "[ComputeGraph][Approximation]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("moved", A, B, C);

    graph.note_approximation(relative("DF", 1e-5));
    graph.set_accuracy_budget(cg::ApproximationEffect::NormRelative, 1e-3);

    // `load_graph` returns a Graph BY VALUE, so every load goes through this path; a member
    // missing from move_members_from is silently dropped by all of them.
    cg::Graph const moved = std::move(graph);
    REQUIRE(moved.approximations().size() == 1);
    REQUIRE(moved.accuracy_budget_value() == Catch::Approx(1e-3));
}

// ── Names ──────────────────────────────────────────────────────────────────

TEST_CASE("Approximation - every effect round-trips through its name", "[ComputeGraph][Approximation]") {
    for (auto const effect :
         {cg::ApproximationEffect::ElementWise, cg::ApproximationEffect::NormRelative, cg::ApproximationEffect::EnergyLike}) {
        auto const name = cg::approximation_effect_name(effect);
        INFO("effect name: " << name);
        REQUIRE(cg::approximation_effect_from_name(name).has_value());
        REQUIRE(*cg::approximation_effect_from_name(name) == effect);
    }
    REQUIRE_FALSE(cg::approximation_effect_from_name("not-an-effect").has_value());
}

// ── Where a number came from ───────────────────────────────────────────────

TEST_CASE("Approximation - a bound is asserted unless a pass says it measured one", "[ComputeGraph][Approximation]") {
    // The default is the safe one. A provider that says nothing about provenance is claiming
    // nothing, and a claim is what most of them have: measuring an approximation's error means
    // computing the exact answer it exists to avoid computing.
    auto const claimed = cg::make_approximation_record("Claimed", cg::ApproximationEffect::NormRelative, 1e-5, 1e-5);
    REQUIRE(claimed.origin == cg::ApproximationOrigin::Asserted);

    auto const known = cg::make_approximation_record("Known", cg::ApproximationEffect::NormRelative, 1e-5, 4e-6, {}, {}, "",
                                                     cg::ApproximationOrigin::Measured);
    REQUIRE(known.origin == cg::ApproximationOrigin::Measured);
}

TEST_CASE("Approximation - the origin survives a save, and an older file's number is not promoted",
          "[ComputeGraph][Approximation][SaveLoad]") {
    auto A     = create_random_tensor<double>("A", 3, 4);
    auto B     = create_random_tensor<double>("B", 4, 5);
    auto C     = create_zero_tensor<double>("C", 3, 5);
    auto graph = make_graph("origin", A, B, C);

    graph.note_approximation(cg::make_approximation_record("Truncation", cg::ApproximationEffect::NormRelative, 1e-5, 4e-6, {"C"}, {}, "",
                                                           cg::ApproximationOrigin::Measured));

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());

    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->approximations()[0].origin == cg::ApproximationOrigin::Measured);

    // The same file with the key taken out, which is what every file written before the key
    // existed looks like. It reads back as ASSERTED: a number whose provenance nobody recorded
    // does not become evidence because a newer build opened it.
    std::string older = *text;
    auto const  key   = older.find(R"("origin":"measured",)");
    REQUIRE(key != std::string::npos);
    older.erase(key, std::string_view{R"("origin":"measured",)"}.size());

    auto older_loaded = cg::load_graph_string(older);
    if (!older_loaded.has_value()) {
        UNSCOPED_INFO(older_loaded.error().message);
    }
    REQUIRE(older_loaded.has_value());
    REQUIRE(older_loaded->approximations()[0].origin == cg::ApproximationOrigin::Asserted);
    REQUIRE(older_loaded->approximations()[0].bound == Catch::Approx(4e-6));
}

TEST_CASE("Approximation - every origin round-trips through its name", "[ComputeGraph][Approximation]") {
    for (auto const origin : {cg::ApproximationOrigin::Measured, cg::ApproximationOrigin::Asserted}) {
        auto const name = cg::approximation_origin_name(origin);
        INFO("origin name: " << name);
        REQUIRE(cg::approximation_origin_from_name(name).has_value());
        REQUIRE(*cg::approximation_origin_from_name(name) == origin);
    }
    REQUIRE_FALSE(cg::approximation_origin_from_name("guessed").has_value());
}
