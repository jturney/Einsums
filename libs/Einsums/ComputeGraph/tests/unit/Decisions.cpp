//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// What a pass did, declined and said, as Decision records: every reporting call makes one, the
/// report text is rendered from them, and each is stamped with the graph it was made in.

#include <Einsums/ComputeGraph.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

namespace cg = einsums::compute_graph;

namespace {

/// A pass defined outside the library that makes one record of every kind on each graph it runs on.
class Recording : public cg::OptimizerPass {
  public:
    static constexpr cg::Reason Done{"recording.done", "did the thing"};
    static constexpr cg::Reason Deep{"recording.deep", "did a detail"};
    static constexpr cg::Reason Coded{"recording.coded", "a coded reason"};
    static constexpr cg::Reason Noted{"recording.noted", "worth saying"};

    [[nodiscard]] std::string name() const override { return "Recording"; }
    [[nodiscard]] bool        recurse_into_subgraphs() const override { return true; }

    bool transform(cg::Graph & /*graph*/) override {
        report(2, "an uncoded note");
        note_skip("an uncoded reason", "detail one");
        note_skip(Coded, "detail two", cg::Decision{.tensors = {"T"}});
        note_skip(Coded, "detail three");
        applied(Done, "did the thing to T", cg::Decision{.tensors = {"T"}, .facts = {cg::Fact{.name = "bytes", .value = 64, .unit = "B"}}});
        applied(Deep, "a level-two detail", {}, 2);
        note(Noted, "something worth saying");
        return false;
    }
};

std::vector<cg::Decision> of_code(std::vector<cg::Decision> const &decisions, std::string_view code) {
    std::vector<cg::Decision> out;
    std::ranges::copy_if(decisions, std::back_inserter(out), [&](cg::Decision const &d) { return d.code == code; });
    return out;
}

} // namespace

TEST_CASE("Every reporting call of a pass makes a decision", "[ComputeGraph][Decisions]") {
    cg::Graph       graph("recorded");
    auto            pass = std::make_shared<Recording>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    auto const decisions = pass->decisions();
    REQUIRE(decisions.size() == 7);
    CHECK(std::ranges::all_of(decisions, [](cg::Decision const &d) { return d.pass_name == "Recording" && d.graph == "recorded"; }));

    // A call that gives only words is a record without a code.
    CHECK(decisions.at(0).kind == cg::DecisionKind::Note);
    CHECK(decisions.at(0).code.empty());
    CHECK(decisions.at(0).text == "an uncoded note");
    CHECK(decisions.at(0).level == 2);
    CHECK(decisions.at(1).kind == cg::DecisionKind::Declined);
    CHECK(decisions.at(1).code.empty());
    CHECK(decisions.at(1).reason == "an uncoded reason");
    CHECK(decisions.at(1).detail == "detail one");

    auto const coded = of_code(decisions, Recording::Coded.code);
    REQUIRE(coded.size() == 2);
    CHECK(coded.at(0).kind == cg::DecisionKind::Declined);
    CHECK(coded.at(0).reason == Recording::Coded.text);
    CHECK(coded.at(0).tensors == std::vector<std::string>{"T"});

    auto const done = of_code(decisions, Recording::Done.code);
    REQUIRE(done.size() == 1);
    CHECK(done.at(0).kind == cg::DecisionKind::Applied);
    CHECK(done.at(0).fact("bytes") == 64);
    CHECK(done.at(0).facts.at(0).unit == "B");
    CHECK(done.at(0).fact("absent") == 0);
    CHECK(of_code(decisions, Recording::Noted.code).at(0).kind == cg::DecisionKind::Note);

    // The tally counts declines by their words, coded or not, as it always has.
    auto const skips = pass->skip_reasons();
    REQUIRE(skips.size() == 2);
    CHECK(skips.at(0) == std::pair<std::string, std::size_t>{std::string{Recording::Coded.text}, 2});
    CHECK(skips.at(1) == std::pair<std::string, std::size_t>{"an uncoded reason", 1});

    // explain() renders the level-one applied records, and nothing else.
    CHECK(pass->explain() == std::vector<std::string>{"did the thing to T"});
}

TEST_CASE("Decisions name the graph or body they were made in, and clear per apply", "[ComputeGraph][Decisions]") {
    cg::Graph graph("parent");
    graph.add_loop("again", 1, [](std::size_t) { return false; });
    auto            pass = std::make_shared<Recording>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    auto const done = of_code(pass->decisions(), Recording::Done.code);
    REQUIRE(done.size() == 2);
    CHECK(done.at(0).graph == "parent");
    CHECK(done.at(1).graph == "again/body");

    graph.apply(manager);
    CHECK(of_code(pass->decisions(), Recording::Done.code).size() == 2);
}

TEST_CASE("The manager and the graph hand back every pass's decisions", "[ComputeGraph][Decisions]") {
    cg::Graph       graph("collected");
    auto            first  = std::make_shared<Recording>();
    auto            second = std::make_shared<Recording>();
    cg::PassManager manager;
    manager.add(first);
    manager.add(second);
    graph.apply(manager);
    CHECK(manager.decisions().size() == first->decisions().size() + second->decisions().size());

    CHECK(graph.decisions().empty());
    graph.optimize();
    CHECK_FALSE(graph.decisions().empty());
    CHECK(std::ranges::all_of(graph.decisions(), [](cg::Decision const &d) { return !d.pass_name.empty() && d.graph == "collected"; }));
}
