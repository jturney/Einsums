//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Capture and run one trivial graph under the given name, then let it die.
void run_and_destroy(std::string const &name) {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph(name);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();
}

/// RAII: set the profiler-save config key for one test, restore on exit.
struct ProfilerSaveKey {
    explicit ProfilerSaveKey(std::string const &value) { GlobalConfigMap::get_singleton().set_string("profiler-save", value); }
    ~ProfilerSaveKey() { GlobalConfigMap::get_singleton().set_string("profiler-save", ""); }
};

size_t count_occurrences(std::string const &haystack, std::string const &needle) {
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

} // namespace

TEST_CASE("Graph registry - live graphs are exported", "[ComputeGraph][registry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("registry_live_graph");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.execute();

    REQUIRE(cg::registered_graphs_json().find("registry_live_graph") != std::string::npos);
}

TEST_CASE("Graph registry - dead graphs are not serialized when nobody collects", "[ComputeGraph][registry]") {
    // The default state of a test process: profiling enabled but no
    // profiler-save configured and no viewer attached. A dying graph must NOT
    // pay to_json() then - that serialization is O(graph) and runs inside
    // whatever phase drops the graph; it measured 41 ms of a 345 ms DLPNO
    // transform phase before it was gated.
    run_and_destroy("registry_uncollected_graph");
    REQUIRE(cg::registered_graphs_json().find("registry_uncollected_graph") == std::string::npos);
}

TEST_CASE("Graph registry - dead graphs survive when profiler-save is configured", "[ComputeGraph][registry]") {
    ProfilerSaveKey const key("registry-test-session.json");

    run_and_destroy("registry_saved_graph");
    REQUIRE(cg::registered_graphs_json().find("registry_saved_graph") != std::string::npos);
}

TEST_CASE("Graph registry - dead-graph cache replaces by name instead of growing", "[ComputeGraph][registry]") {
    ProfilerSaveKey const key("registry-test-session.json");

    // A loop that builds and destroys a same-named graph - the shape of any
    // phase-per-iteration workload - must re-cache one entry, not append one
    // per death; the cache used to grow without bound.
    for (int rep = 0; rep < 3; ++rep) {
        run_and_destroy("registry_replaced_graph");
    }
    auto const json = cg::registered_graphs_json();
    REQUIRE(count_occurrences(json, "\"registry_replaced_graph\"") == 1);
}
