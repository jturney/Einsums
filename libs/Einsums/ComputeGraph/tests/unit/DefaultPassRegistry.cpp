//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DefaultPassRegistry.cpp
/// @brief Passes from outside the library run in the default pipelines beside their anchors.
///
/// A registration lasts for the process, so these cases live in a binary of their own: every
/// default pipeline built here carries the passes registered below, which would make another
/// file's assertions about the default pipeline depend on which tests ran first.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// How many times each test pass has run, by name.
std::map<std::string, int> &runs() {
    static std::map<std::string, int> counts;
    return counts;
}

/// A pass that changes nothing and counts its runs.
class CountingPass final : public cg::OptimizerPass {
  public:
    explicit CountingPass(std::string name) : _name(std::move(name)) {}
    [[nodiscard]] std::string name() const override { return _name; }
    bool                      run(cg::Graph                      &/*graph*/) override {
        ++runs()[_name];
        return false;
    }

  private:
    std::string _name;
};

std::function<std::shared_ptr<cg::OptimizerPass>()> counting(std::string name) {
    return [name = std::move(name)] { return std::make_shared<CountingPass>(name); };
}

/// The registrations every case here sees, made once whatever order the cases run in.
void register_test_passes() {
    static bool const registered = [] {
        cg::register_default_pass("DeadNodeElimination", cg::PassAnchor::After, counting("TestAfterDne1"), "test");
        cg::register_default_pass("DeadNodeElimination", cg::PassAnchor::After, counting("TestAfterDne2"), "test");
        cg::register_default_pass("DeadNodeElimination", cg::PassAnchor::Before, counting("TestBeforeDne"), "test");
        cg::register_default_pass("TestAfterDne1", cg::PassAnchor::After, counting("TestAfterOutside"), "test");
        cg::register_default_pass("GPUPlacement", cg::PassAnchor::After, counting("TestAfterGpuPlacement"), "test");
        return true;
    }();
    (void)registered;
}

std::vector<std::string> names_of(cg::PassManager const &pm) {
    std::vector<std::string> names;
    for (auto const &pass : pm.passes()) {
        names.push_back(pass->name());
    }
    return names;
}

/// The @p count names that follow @p anchor in @p names, or fewer at the end of the list.
std::vector<std::string> after(std::vector<std::string> const &names, std::string const &anchor, std::size_t count) {
    auto const it = std::ranges::find(names, anchor);
    REQUIRE(it != names.end());
    auto const end = std::min(names.end(), it + 1 + static_cast<std::ptrdiff_t>(count));
    return {it + 1, end};
}

} // namespace

TEST_CASE("Default pass registry - registered passes stand beside their anchors in registration order", "[ComputeGraph][Passes]") {
    register_test_passes();
    auto const names = names_of(cg::PassManager::create_default());

    // Before the anchor, and after it in the order registered, with the pass anchored on a
    // registered pass right behind that pass.
    auto const dne = std::ranges::find(names, std::string{"DeadNodeElimination"});
    REQUIRE(dne != names.begin());
    CHECK(*(dne - 1) == "TestBeforeDne");
    CHECK(after(names, "DeadNodeElimination", 3) == std::vector<std::string>{"TestAfterDne1", "TestAfterOutside", "TestAfterDne2"});

    // An anchor gated by a backend stands only where the backend is built.
    bool const has_gpu_placement = std::ranges::find(names, std::string{"GPUPlacement"}) != names.end();
    CHECK((std::ranges::find(names, std::string{"TestAfterGpuPlacement"}) != names.end()) == has_gpu_placement);
    if (has_gpu_placement) {
        CHECK(after(names, "GPUPlacement", 1) == std::vector<std::string>{"TestAfterGpuPlacement"});
    }
}

TEST_CASE("Default pass registry - every list built from the default pipeline carries them", "[ComputeGraph][Passes]") {
    register_test_passes();

    // O1 holds DeadNodeElimination, so it carries the passes anchored there.
    auto const o1 = names_of(cg::PassManager::create_for(cg::OptLevel::O1));
    CHECK(after(o1, "DeadNodeElimination", 3) == std::vector<std::string>{"TestAfterDne1", "TestAfterOutside", "TestAfterDne2"});
    CHECK(std::ranges::find(o1, std::string{"TestAfterGpuPlacement"}) == o1.end());

    // A registered pass declares no phase, so it is tuning, and goes where its phase puts it.
    auto const tuning = names_of(cg::PassManager::tuning_pass_manager());
    CHECK(std::ranges::find(tuning, std::string{"TestAfterDne1"}) != tuning.end());
    auto const structural = names_of(cg::PassManager::structural_pass_manager());
    CHECK(std::ranges::find(structural, std::string{"TestAfterDne1"}) == structural.end());

    // Every manager gets a fresh pass of its own.
    auto const first  = cg::PassManager::create_default();
    auto const second = cg::PassManager::create_default();
    auto const find   = [](cg::PassManager const &pm) {
        return std::ranges::find_if(pm.passes(), [](auto const &pass) { return pass->name() == "TestAfterDne1"; })->get();
    };
    CHECK(find(first) != find(second));
}

TEST_CASE("Default pass registry - a registered pass runs when the graph is optimized", "[ComputeGraph][Passes]") {
    register_test_passes();
    RuntimeTensor<double> A{"A", {2UL, 2UL}};
    RuntimeTensor<double> C{"C", {2UL, 2UL}};
    A.zero();
    C.zero();
    cg::Graph graph("default_pass_registry");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, A);
    }

    int const before = runs()["TestAfterDne1"];
    graph.optimize();
    CHECK(runs()["TestAfterDne1"] == before + 1);
}

TEST_CASE("Default pass registry - registrations that could never run, or collide, are refused", "[ComputeGraph][Passes]") {
    register_test_passes();
    CHECK_THROWS_AS(cg::register_default_pass("NoSuchPass", cg::PassAnchor::After, counting("TestNowhere"), "test"), std::invalid_argument);
    CHECK_THROWS_AS(cg::register_default_pass("CSE", cg::PassAnchor::After, counting("TestAfterDne1"), "test"), std::invalid_argument);
    CHECK_THROWS_AS(cg::register_default_pass("CSE", cg::PassAnchor::After, counting("DeadNodeElimination"), "test"),
                    std::invalid_argument);
    CHECK_THROWS_AS(cg::register_default_pass("CSE", cg::PassAnchor::After, {}, "test"), std::invalid_argument);
    CHECK_THROWS_AS(cg::register_default_pass(
                        "CSE", cg::PassAnchor::After, [] { return std::shared_ptr<cg::OptimizerPass>{}; }, "test"),
                    std::invalid_argument);
    CHECK_THROWS_AS(cg::register_default_pass("CSE", cg::PassAnchor::After, counting("TestNoOwner"), ""), std::invalid_argument);

    auto const registered = cg::registered_default_passes();
    REQUIRE(registered.size() == 5);
    CHECK(registered[0].name == "TestAfterDne1");
    CHECK(registered[2].where == cg::PassAnchor::Before);
    CHECK(registered[3].anchor == "TestAfterDne1");
    CHECK(registered[4].owner == "test");
}
