//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file NodeFeatures.cpp
/// @brief What a node carries, and the check that a pass leaves alone what it does not understand.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Sets einsums:pass:verify for one test and restores it.
struct VerifyPasses {
    bool const previous = config::get(option::PassVerify);
    VerifyPasses() { config::set(option::PassVerify, true); }
    ~VerifyPasses() { config::set(option::PassVerify, previous); }
};

cg::NodeFeatures only_node(cg::Graph const &graph) {
    REQUIRE(graph.nodes().size() == 1);
    return cg::features_of(graph, graph.nodes()[0]);
}

/// Removes every einsum, the crudest possible rewrite, asking @ref understands first or not.
class EinsumRemover : public cg::OptimizerPass {
  public:
    EinsumRemover(std::optional<cg::NodeFeatures> understood, bool asks) : _understood(understood), _asks(asks) {}

    [[nodiscard]] std::string                     name() const override { return "EinsumRemover"; }
    [[nodiscard]] std::optional<cg::NodeFeatures> understood_features() const override { return _understood; }

    bool run(cg::Graph &graph) override {
        std::vector<bool> remove(graph.nodes().size(), false);
        bool              any = false;
        for (std::size_t i = 0; i < graph.nodes().size(); i++) {
            auto const &node = graph.nodes()[i];
            if (node.kind == cg::OpKind::Einsum && (!_asks || understands(graph, node))) {
                remove[i] = true;
                any       = true;
            }
        }
        if (any) {
            graph.erase_nodes(remove);
        }
        return any;
    }

  private:
    std::optional<cg::NodeFeatures> _understood;
    bool                            _asks;
};

/// One plain contraction and one wrapped in P(i/j), over caller tensors.
void capture_plain_and_antisymmetrized(cg::Graph &graph, Tensor<double, 2> &C, Tensor<double, 2> &X, Tensor<double, 2> const &A) {
    cg::CaptureGuard const guard(graph);
    cg::einsum("ik;kj->ij", &C, A, A);
    cg::einsum("i,j <- P(i/j) i,k ; k,j", 0.0, &X, 1.0, A, A);
}

} // namespace

// Defends: the classifier every declaration is checked against. A feature it failed to see would
// let a pass rewrite a node it was never taught, which is the whole failure this exists to stop.
TEST_CASE("features_of names what a node carries", "[ComputeGraph][NodeFeatures]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    SECTION("a plain contraction carries nothing") {
        cg::Graph graph("plain");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &C, A, B);
        }
        CHECK(only_node(graph).empty());
    }
    SECTION("permutation operators") {
        cg::Graph graph("operators");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- P(i/j) i,k ; k,j", 0.0, &C, 1.0, A, B);
        }
        CHECK(only_node(graph) == cg::NodeFeature::PermutationOperators);
    }
    SECTION("conjugation and a complex prefactor") {
        using cd     = std::complex<double>;
        auto      Z  = create_random_tensor<cd>("Z", 3, 3);
        auto      ZC = create_zero_tensor<cd>("ZC", 3, 3);
        cg::Graph graph("complex");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", cd{0}, &ZC, cd{0.5, 0.5}, Z, Z, /*conj_a=*/true, /*conj_b=*/false);
        }
        CHECK(only_node(graph) == (cg::NodeFeature::Conjugation | cg::NodeFeature::ComplexPrefactor));
    }
    SECTION("mixed precision") {
        auto      F = create_random_tensor<float>("F", 3, 3);
        cg::Graph graph("mixed");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, F, A);
        }
        CHECK(only_node(graph) == cg::NodeFeature::MixedPrecision);
    }
    SECTION("a bare scalar") {
        double    result = 0.0;
        cg::Graph graph("scalar");
        {
            cg::CaptureGuard const guard(graph);
            cg::dot(&result, A, B);
        }
        CHECK(only_node(graph) == cg::NodeFeature::RawScalar);
    }
    SECTION("a view, and the node that makes it") {
        auto      D = create_random_tensor<double>("D", 3, 4);
        auto      E = create_zero_tensor<double>("E", 3, 3);
        cg::Graph graph("view");
        {
            cg::CaptureGuard const guard(graph);
            auto                  &slice = cg::view(D, cg::ViewAxis::full(), cg::ViewAxis::range(0, 3));
            cg::einsum("ik;kj->ij", &E, slice, A);
        }
        auto const &contraction = graph.nodes().back();
        REQUIRE(contraction.kind == cg::OpKind::Einsum);
        CHECK(cg::features_of(graph, contraction).covers(cg::NodeFeature::Views));
    }
    SECTION("control flow") {
        cg::Graph graph("loop");
        auto     &body = graph.add_loop("twice", 2, [](std::size_t) { return false; });
        {
            cg::CaptureGuard const guard(body);
            cg::einsum("ik;kj->ij", &C, A, B);
        }
        CHECK(only_node(graph).covers(cg::NodeFeature::ControlFlow));
    }
    SECTION("a redirected slot") {
        auto      S1 = create_zero_tensor<double>("S1", 3, 3);
        auto      S2 = create_zero_tensor<double>("S2", 3, 3);
        cg::Graph graph("redirect");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &S1, A, A);
            cg::einsum("ik;kj->ij", &S2, A, A);
            cg::einsum("ik;kj->ij", &C, S2, B);
        }
        graph.erase_nodes({false, true, false});
        graph.redirect_slot(graph.nodes()[1].inputs[0], graph.nodes()[0].outputs[0]);
        CHECK(cg::features_of(graph, graph.nodes()[1]) == cg::NodeFeature::RedirectedSlot);
    }
}

// Defends: the promise a declaration makes is checked, not trusted. A pass that removes or rewrites
// a node carrying a feature it does not understand is named, with the node and the feature, the
// moment it does so, rather than surfacing later as a wrong number.
TEST_CASE("a pass that touches a node it does not understand is caught", "[ComputeGraph][NodeFeatures][Verify]") {
    auto               A = create_random_tensor<double>("A", 3, 3);
    auto               C = create_zero_tensor<double>("C", 3, 3);
    auto               X = create_zero_tensor<double>("X", 3, 3);
    VerifyPasses const verifying;

    SECTION("removing it is reported") {
        cg::Graph graph("careless");
        capture_plain_and_antisymmetrized(graph, C, X, A);
        cg::PassManager manager;
        manager.add(std::make_shared<EinsumRemover>(cg::NodeFeatures{}, /*asks=*/false));
        try {
            (void)manager.run(graph);
            FAIL("a pass that removed a node it does not understand was not reported");
        } catch (std::logic_error const &error) {
            std::string const message = error.what();
            CHECK_THAT(message, Catch::Matchers::ContainsSubstring("pass 'EinsumRemover' changed nodes carrying features"));
            CHECK_THAT(message, Catch::Matchers::ContainsSubstring("removed node"));
            CHECK_THAT(message, Catch::Matchers::ContainsSubstring("permutation operators"));
        }
    }

    SECTION("a pass that asks leaves it alone") {
        cg::Graph graph("careful");
        capture_plain_and_antisymmetrized(graph, C, X, A);
        cg::PassManager manager;
        manager.add(std::make_shared<EinsumRemover>(cg::NodeFeatures{}, /*asks=*/true));
        CHECK(manager.run(graph));
        REQUIRE(graph.nodes().size() == 1);
        CHECK(cg::features_of(graph, graph.nodes()[0]) == cg::NodeFeature::PermutationOperators);
    }

    SECTION("a pass that declares the feature may touch it") {
        cg::Graph graph("taught");
        capture_plain_and_antisymmetrized(graph, C, X, A);
        cg::PassManager manager;
        manager.add(std::make_shared<EinsumRemover>(cg::NodeFeatures{cg::NodeFeature::PermutationOperators}, /*asks=*/false));
        CHECK(manager.run(graph));
        CHECK(graph.nodes().empty());
    }

    SECTION("a pass that declares nothing is not checked") {
        cg::Graph graph("outside");
        capture_plain_and_antisymmetrized(graph, C, X, A);
        cg::PassManager manager;
        manager.add(std::make_shared<EinsumRemover>(std::nullopt, /*asks=*/false));
        CHECK(manager.run(graph));
        CHECK(graph.nodes().empty());
    }
}
