//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PassSwitches.cpp
/// @brief Switching individual passes off by name, and saying so.
///
/// An optimizer that rewrites mathematics will eventually produce a wrong
/// number, and the bug history in this module says so with evidence: the
/// baked-lambda redirect, the full-cover alias miss, the Kahn FIFO hoist, the
/// pass-stats recursion reset. Each produced a plausible graph and a wrong
/// result, and each was found by bisecting against eager. Bisection needs one
/// thing from the pass manager, which is the ability to run the same pipeline
/// with one pass switched off and no rebuild.
///
/// Two sources say a pass is off, and the cases below pin how they compose:
///
///  * ``einsums:pass:disable``, the user's blunt instrument from a command line.
///  * @ref cg::PassManager::disable / @ref cg::PassManager::enable, which is what
///    a driver uses when it has to run N pipelines with a different pass off in
///    each; mutating process-global configuration N times would be the
///    alternative, and a driver that did that could not be run concurrently.
///
/// The more specific statement wins, so an explicit ``enable`` overrides the
/// option. And a switch name matching no pass is REPORTED rather than ignored,
/// which is the case at the end of this file: a mistyped pass name that quietly
/// does nothing recreates exactly the silent-typo failure mode that string-keyed
/// configuration was removed from this library to avoid.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Set ``einsums:pass:disable`` for one case and put it back afterwards. The
/// option is process-global, so a case that left it set would silently change
/// every case Catch2 happens to run after it, and the randomized ordering makes
/// that a flake rather than a reproducible failure.
class ScopedPassDisable {
  public:
    explicit ScopedPassDisable(std::string const &value) : _saved(config::get(option::PassDisable)) {
        config::set(option::PassDisable, value);
    }
    ~ScopedPassDisable() { config::set(option::PassDisable, _saved); }

    ScopedPassDisable(ScopedPassDisable const &)                = delete;
    ScopedPassDisable(ScopedPassDisable &&)                     = delete;
    ScopedPassDisable &operator=(ScopedPassDisable const &)     = delete;
    ScopedPassDisable &operator=(ScopedPassDisable &&) noexcept = delete;

  private:
    std::string _saved;
};

/// A graph with two scales of one tensor, which ElementWiseFusion merges and
/// nothing else in the default cleanup set touches. Small enough that "did the
/// pass run" is answerable from the node count alone.
cg::Graph two_scales(Tensor<double, 2> &tensor) {
    cg::Graph graph("two scales");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &tensor);
        cg::scale(3.0, &tensor);
    }
    return graph;
}

bool contains(std::vector<std::string> const &haystack, std::string const &needle) {
    return std::ranges::find(haystack, needle) != haystack.end();
}

} // namespace

TEST_CASE("a pass switched off by name does not run", "[ComputeGraph][PassManager][Switches]") {
    auto tensor = create_random_tensor<double>("A", 4, 4);

    // The baseline: the pass is in the pipeline and it fires.
    {
        auto graph = two_scales(tensor);
        auto pm    = cg::PassManager();
        pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
        CHECK(pm.run(graph));
        CHECK(pm.disabled_passes().empty());
    }

    // Switched off, the same pipeline leaves the graph alone.
    {
        auto graph  = two_scales(tensor);
        auto before = graph.num_nodes();
        auto pm     = cg::PassManager();
        pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
        pm.disable("ElementWiseFusion");
        CHECK_FALSE(pm.run(graph));
        CHECK(graph.num_nodes() == before);
        CHECK(pm.disabled_passes() == std::vector<std::string>{"ElementWiseFusion"});
        CHECK(pm.unmatched_switches().empty());
    }
}

TEST_CASE("the option switches a pass off with no code change", "[ComputeGraph][PassManager][Switches]") {
    auto                    tensor = create_random_tensor<double>("A", 4, 4);
    ScopedPassDisable const disabled("ElementWiseFusion");

    auto graph  = two_scales(tensor);
    auto before = graph.num_nodes();
    auto pm     = cg::PassManager();
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    CHECK_FALSE(pm.run(graph));
    CHECK(graph.num_nodes() == before);
    CHECK(pm.disabled_passes() == std::vector<std::string>{"ElementWiseFusion"});
}

TEST_CASE("the option's list is comma separated and trimmed", "[ComputeGraph][PassManager][Switches]") {
    auto                    tensor = create_random_tensor<double>("A", 4, 4);
    ScopedPassDisable const disabled(" ElementWiseFusion , CSE ");

    auto graph = two_scales(tensor);
    auto pm    = cg::PassManager();
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    pm.add(std::make_shared<cg::passes::CSE>());
    CHECK_FALSE(pm.run(graph));
    CHECK(pm.disabled_passes() == std::vector<std::string>{"ElementWiseFusion", "CSE"});
}

TEST_CASE("an explicit enable beats the option", "[ComputeGraph][PassManager][Switches]") {
    // The more specific statement about this pipeline wins. A program that named
    // a pass and silently did not get it, because a shell variable set for an
    // unrelated bisect run said otherwise, is the surprise this rule removes.
    auto                    tensor = create_random_tensor<double>("A", 4, 4);
    ScopedPassDisable const disabled("ElementWiseFusion");

    auto graph = two_scales(tensor);
    auto pm    = cg::PassManager();
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    pm.enable("ElementWiseFusion");
    CHECK(pm.run(graph));
    CHECK(pm.disabled_passes().empty());
}

TEST_CASE("a switch may be set before the pass it names is added", "[ComputeGraph][PassManager][Switches]") {
    // Switches are consulted at run(), not at the call, so `disable` then
    // `populate_default` works. A driver that had to order the two would be
    // relying on something nothing states.
    auto tensor = create_random_tensor<double>("A", 4, 4);
    auto graph  = two_scales(tensor);

    auto pm = cg::PassManager();
    pm.disable("ElementWiseFusion");
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    CHECK_FALSE(pm.run(graph));
    CHECK(pm.disabled_passes() == std::vector<std::string>{"ElementWiseFusion"});
}

TEST_CASE("a switch name matching no pass is reported, not ignored", "[ComputeGraph][PassManager][Switches]") {
    auto tensor = create_random_tensor<double>("A", 4, 4);
    auto graph  = two_scales(tensor);

    auto pm = cg::PassManager();
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    pm.disable("ElementWiseFusionn"); // the typo a bisect script makes at 2am
    CHECK(pm.run(graph));             // the real pass still ran

    CHECK(pm.disabled_passes().empty());
    CHECK(pm.unmatched_switches() == std::vector<std::string>{"ElementWiseFusionn"});
    CHECK(pm.explain().find("ElementWiseFusionn") != std::string::npos);
    CHECK(pm.explain().find("matching no pass") != std::string::npos);
}

TEST_CASE("explain names the passes it switched off", "[ComputeGraph][PassManager][Switches]") {
    // A report that omits this cannot distinguish "nothing to optimize" from
    // "the pass that would have optimized it was switched off", and the second
    // is usually a leftover environment variable.
    auto tensor = create_random_tensor<double>("A", 4, 4);
    auto graph  = two_scales(tensor);

    auto pm = cg::PassManager();
    pm.add(std::make_shared<cg::passes::ElementWiseFusion>());
    pm.disable("ElementWiseFusion");
    CHECK_FALSE(pm.run(graph));

    auto const report = pm.explain();
    INFO("report:\n" << report);
    CHECK(report.find("switched off for this run") != std::string::npos);
    CHECK(report.find("ElementWiseFusion") != std::string::npos);
    // And it does NOT claim the pipeline had nothing to do, which is the wrong
    // conclusion to hand someone bisecting.
    CHECK(report.find("no optimizations applied") == std::string::npos);
}

TEST_CASE("a switched-off analysis pass stays off through the re-analysis sweep", "[ComputeGraph][PassManager][Switches][Analysis]") {
    // PassManager::run re-runs the analysis passes once when a structural pass
    // moved the node set, so annotations describe the graph that will execute.
    // That second loop has its own disable check, and a pass switched off in the
    // first loop but not the second would run exactly once - the worst of both,
    // and invisible without this case.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("analysis sweep");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager();
    pm.populate_default();
    pm.disable("SpacePropagation");
    graph.apply(pm);

    CHECK(contains(pm.disabled_passes(), "SpacePropagation"));
    // Exactly once in the report, however many loops consulted the switch.
    auto const  report = pm.explain();
    std::size_t count = 0, at = 0;
    while ((at = report.find("SpacePropagation", at)) != std::string::npos) {
        ++count;
        at += 1;
    }
    INFO("report:\n" << report);
    CHECK(count == 1);
}

TEST_CASE("the phase managers partition the default pipeline", "[ComputeGraph][PassManager][Phases]") {
    // The in-place populate_* methods exist for the Python binding, which cannot
    // receive a PassManager by value. They have to agree with the static
    // factories they mirror, or a Python caller running the documented load path
    // (resource then tuning) would be running a different set of passes than a
    // C++ caller doing the same thing.
    auto const names = [](cg::PassManager const &pm) {
        std::vector<std::string> out;
        for (auto const &pass : pm.passes()) {
            out.push_back(pass->name());
        }
        return out;
    };

    auto in_place = [](void (cg::PassManager::*populate)()) {
        cg::PassManager pm;
        (pm.*populate)();
        return pm;
    };

    auto analysis   = in_place(&cg::PassManager::populate_analysis);
    auto structural = in_place(&cg::PassManager::populate_structural);
    auto resource   = in_place(&cg::PassManager::populate_resource);
    auto tuning     = in_place(&cg::PassManager::populate_tuning);

    CHECK(names(analysis) == names(cg::PassManager::analysis_pass_manager()));
    CHECK(names(structural) == names(cg::PassManager::structural_pass_manager()));
    CHECK(names(resource) == names(cg::PassManager::resource_pass_manager()));
    CHECK(names(tuning) == names(cg::PassManager::tuning_pass_manager()));

    // Together they are the default pipeline exactly: no pass in two phases, and
    // none left out. A pass missing from every phase view would be a pass a
    // loaded graph never runs, which is how an output silently stops being
    // re-derived.
    std::vector<std::string> union_of;
    for (auto const *pm : {&analysis, &structural, &resource, &tuning}) {
        for (auto const &name : names(*pm)) {
            union_of.push_back(name);
        }
    }
    auto expected = names(cg::PassManager::create_default());
    std::ranges::sort(union_of);
    std::ranges::sort(expected);
    CHECK(union_of == expected);
}
