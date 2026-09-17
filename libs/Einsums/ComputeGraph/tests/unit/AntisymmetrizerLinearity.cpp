//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerLinearity.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// `V = a*P[spec](B)` then `V += s*W`, with `W = c*P[spec](A)`: the shape a
/// residual writes when it antisymmetrizes two quantities and adds them.
void capture_sum(cg::Graph &graph, std::string const &w_spec, std::string const &v_spec, size_t n, RuntimeTensor<double> const &A,
                 RuntimeTensor<double> const &B, RuntimeTensor<double> &out) {
    auto                  &W = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
    auto                  &V = graph.create_zero_runtime_tensor<double>("V", {n, n, n}, true);
    cg::CaptureGuard const capture(graph);
    cg::permute(cg::PermuteFormatString(w_spec), 0.0, &W, 2.0, A); // c = 2
    cg::permute(cg::PermuteFormatString(v_spec), 0.0, &V, 3.0, B); // a = 3
    cg::axpby(0.5, W, 1.0, &V);                                    // s = 1/2
    cg::axpby(1.0, V, 0.0, &out);
}

std::shared_ptr<cg::passes::AntisymmetrizerLinearity> merge(cg::Graph &graph) {
    auto            pass = std::make_shared<cg::passes::AntisymmetrizerLinearity>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    return pass;
}

} // namespace

// The merge rewrites which additions happen and in what order, so the only claim
// worth testing is that the VALUE survives. Prefactors on both operators and on
// the accumulation, so a fold that dropped one would show.
TEST_CASE("AntisymmetrizerLinearity - the merged form computes the same values", "[ComputeGraph][AntisymmetrizerLinearity]") {
    size_t const          n       = 4;
    auto                  a_typed = create_random_tensor<double>("a", n, n, n);
    auto                  b_typed = create_random_tensor<double>("b", n, n, n);
    RuntimeTensor<double> A(a_typed);
    RuntimeTensor<double> B(b_typed);

    std::string const spec = "i,j,k <- P(i/j/k) i,j,k";

    RuntimeTensor<double> plain("plain", {n, n, n});
    cg::Graph             unmerged("unmerged");
    capture_sum(unmerged, spec, spec, n, A, B, plain);
    unmerged.execute();

    RuntimeTensor<double> merged_out("merged", {n, n, n});
    cg::Graph             merged("merged");
    capture_sum(merged, spec, spec, n, A, B, merged_out);
    auto const pass = merge(merged);
    REQUIRE(pass->num_merged() == 1);
    merged.execute();

    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                std::vector<size_t> const at{i, j, k};
                worst = std::max(worst, std::abs(plain(at) - merged_out(at)));
            }
        }
    }
    INFO("worst element-wise difference " << worst);
    CHECK(worst < 1e-12);

    // Not two zero tensors: the comparison has to have something to compare.
    double magnitude = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                magnitude = std::max(magnitude, std::abs(plain(std::vector<size_t>{i, j, k})));
            }
        }
    }
    REQUIRE(magnitude > 1e-6);
}

TEST_CASE("AntisymmetrizerLinearity - two different operators do not factor", "[ComputeGraph][AntisymmetrizerLinearity]") {
    // P(i/j/k)(A) + P(i/jk)(B) is not P(anything)(A+B). Merging them would be
    // arithmetic the identity does not license.
    size_t const          n       = 4;
    auto                  a_typed = create_random_tensor<double>("a", n, n, n);
    auto                  b_typed = create_random_tensor<double>("b", n, n, n);
    RuntimeTensor<double> A(a_typed);
    RuntimeTensor<double> B(b_typed);

    RuntimeTensor<double> out("out", {n, n, n});
    cg::Graph             graph("mismatched");
    capture_sum(graph, "i,j,k <- P(i/j/k) i,j,k", "i,j,k <- P(i/jk) i,j,k", n, A, B, out);

    auto const pass = merge(graph);
    CHECK(pass->num_candidates() == 1);
    CHECK(pass->num_merged() == 0);
    CHECK(pass->explain().empty());
}

TEST_CASE("AntisymmetrizerLinearity - a graph with no operator sum is untouched", "[ComputeGraph][AntisymmetrizerLinearity]") {
    size_t const          n       = 4;
    auto                  a_typed = create_random_tensor<double>("a", n, n, n);
    RuntimeTensor<double> A(a_typed);
    RuntimeTensor<double> out("out", {n, n, n});

    cg::Graph graph("plain");
    {
        auto                  &W = graph.create_zero_runtime_tensor<double>("W", {n, n, n}, true);
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k <- P(i/j/k) i,j,k", 0.0, &W, 1.0, A);
        cg::axpby(1.0, W, 0.0, &out);
    }
    std::size_t const before = graph.num_nodes();

    auto const pass = merge(graph);
    CHECK(pass->num_merged() == 0);
    CHECK(graph.num_nodes() == before);
}
