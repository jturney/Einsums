//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerFolding.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryInference.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Capture `result = dot( P[spec](wsrc), P[spec](vsrc) )`, which is the shape
/// the toy's naive (T) energy has: an antisymmetrized quantity contracted
/// against another antisymmetrized quantity.
void capture_energy(cg::Graph &graph, std::string const &spec, std::vector<size_t> const &dims, RuntimeTensor<double> const &wsrc,
                    RuntimeTensor<double> const &vsrc, RuntimeTensor<double> &result) {
    auto                  &W = graph.create_zero_runtime_tensor<double>("W", dims, true);
    auto                  &V = graph.create_zero_runtime_tensor<double>("V", dims, true);
    cg::CaptureGuard const capture(graph);
    cg::permute(cg::PermuteFormatString(spec), 0.0, &W, 1.0, wsrc);
    cg::permute(cg::PermuteFormatString(spec), 0.0, &V, 1.0, vsrc);
    cg::dot_python(&result, W, V);
}

double run(cg::Graph &graph, RuntimeTensor<double> &result) {
    graph.execute();
    return result(std::vector<size_t>{0});
}

} // namespace

// THE test. The fold discards N-1 of the operator's terms and multiplies by N,
// which is only the same number if the identity holds. Comparing a folded run
// against an unfolded one on the same data is the whole claim.
TEST_CASE("AntisymmetrizerFolding - the folded contraction gives the same value", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 4;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    std::string const spec = "i,j,k <- P(i/j/k) i,j,k";

    RuntimeTensor<double> plain_result("plain", {1});
    cg::Graph             plain("unfolded");
    capture_energy(plain, spec, dims, wsrc, vsrc, plain_result);
    double const unfolded = run(plain, plain_result);

    RuntimeTensor<double> folded_result("folded", {1});
    cg::Graph             folded("folded");
    capture_energy(folded, spec, dims, wsrc, vsrc, folded_result);

    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(inference);
    manager.add(fold);
    folded.apply(manager);

    REQUIRE(inference->num_tagged() == 2); // both antisymmetrizer outputs
    REQUIRE(fold->num_folded() == 1);

    double const value = run(folded, folded_result);
    INFO("unfolded=" << unfolded << " folded=" << value);
    // ReAssociating, not bitwise: the discarded terms were equal in exact
    // arithmetic and merely close in floating point.
    REQUIRE_THAT(value, Catch::Matchers::WithinRel(unfolded, 1e-12));
    REQUIRE(std::abs(unfolded) > 1e-6); // the comparison is not two zeros
}

// The coset operator the triples correction uses. Its output is antisymmetric
// under nothing, so AntisymmetryInference tags neither operand and the fold has
// no premise. A pass that fired here would be wrong, not merely optimistic.
TEST_CASE("AntisymmetrizerFolding - a coset operator is not folded", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 3;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    RuntimeTensor<double> result("r", {1});
    cg::Graph             graph("coset");
    capture_energy(graph, "i,j,k <- P(i/jk) i,j,k", dims, wsrc, vsrc, result);

    auto            inference = std::make_shared<cg::passes::AntisymmetryInference>();
    auto            fold      = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(inference);
    manager.add(fold);
    graph.apply(manager);

    CHECK(inference->num_tagged() == 0);
    CHECK(fold->num_folded() == 0);
}

// Without the inference pass there is no hint, so the fold must decline even
// though the operator is one it could otherwise collapse. The premise has to be
// established, not assumed from the operator's presence.
TEST_CASE("AntisymmetrizerFolding - declines without an established premise", "[ComputeGraph][AntisymmetrizerFolding]") {
    size_t const              n = 3;
    std::vector<size_t> const dims{n, n, n};
    auto                      w_typed = create_random_tensor<double>("wsrc", n, n, n);
    auto                      v_typed = create_random_tensor<double>("vsrc", n, n, n);
    RuntimeTensor<double>     wsrc(w_typed);
    RuntimeTensor<double>     vsrc(v_typed);

    RuntimeTensor<double> result("r", {1});
    cg::Graph             graph("no_inference");
    capture_energy(graph, "i,j,k <- P(i/j/k) i,j,k", dims, wsrc, vsrc, result);

    auto            fold = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(fold);
    graph.apply(manager);

    CHECK(fold->num_candidates() >= 1);
    CHECK(fold->num_folded() == 0);
    CHECK(fold->explain().empty());
}

TEST_CASE("AntisymmetrizerFolding - a graph with no operator is untouched", "[ComputeGraph][AntisymmetrizerFolding]") {
    auto                  a_typed = create_random_tensor<double>("a", 6);
    auto                  b_typed = create_random_tensor<double>("b", 6);
    RuntimeTensor<double> A(a_typed);
    RuntimeTensor<double> B(b_typed);
    RuntimeTensor<double> result("r", {1});

    cg::Graph graph("plain");
    {
        cg::CaptureGuard const capture(graph);
        cg::dot_python(&result, A, B);
    }
    std::size_t const before = graph.num_nodes();

    auto            fold = std::make_shared<cg::passes::AntisymmetrizerFolding>();
    cg::PassManager manager;
    manager.add(fold);
    graph.apply(manager);

    CHECK(fold->num_candidates() == 0);
    CHECK(fold->num_folded() == 0);
    CHECK(graph.num_nodes() == before);
}
