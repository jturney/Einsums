//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_DeviceEligibility.cpp
/// @brief Tests for Node::eligible, compute_device_eligibility and the DeviceEligibility pass: the
///        "can this node run on a device" half of what GPUPlacement used to decide in one piece.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The pass records nothing without a usable device, so the tests that expect one skip there.
#define DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU()                                                                                              \
    do {                                                                                                                                   \
        if (!::einsums::gpu::gpu_available()) {                                                                                            \
            SKIP("no usable GPU device (gpu::gpu_available() is false)");                                                                  \
        }                                                                                                                                  \
    } while (0)

/// Whether the compiled backend runs a plain real double-precision kernel. MPS takes float32 only;
/// the same constant the pass keys on.
constexpr bool backend_takes_float64 = !gpu::has_mps;

RuntimeTensor<float> matrix(char const *name, std::size_t rows, std::size_t cols) {
    RuntimeTensor<float> t(name, std::vector<std::size_t>{rows, cols});
    t.zero();
    return t;
}

RuntimeTensor<double> matrix64(char const *name, std::size_t rows, std::size_t cols) {
    RuntimeTensor<double> t(name, std::vector<std::size_t>{rows, cols});
    t.zero();
    return t;
}

cg::Node const *find_kind(cg::Graph const &graph, cg::OpKind kind) {
    for (auto const &node : graph.nodes()) {
        if (node.kind == kind) {
            return &node;
        }
    }
    return nullptr;
}

/// Every node of @p graph and of the graphs beneath it.
void for_all_nodes(cg::Graph &graph, std::function<void(cg::Node const &)> const &visit) {
    for (auto const &node : graph.nodes()) {
        visit(node);
    }
    graph.for_each_subgraph([&](cg::Graph &sub) {
        for (auto const &node : sub.nodes()) {
            visit(node);
        }
    });
}

} // namespace

TEST_CASE("EligiblePlaces - the host is always eligible and a device is by ordinal", "[ComputeGraph][DeviceEligibility]") {
    cg::EligiblePlaces places;
    CHECK(places.host());
    CHECK_FALSE(places.any_device());
    CHECK_FALSE(places.device(0));

    places.add_device(0);
    CHECK(places.any_device());
    CHECK(places.device(0));
    CHECK_FALSE(places.device(1));
    CHECK(places.host());

    places.add_device(3);
    CHECK(places.device(3));
    CHECK_FALSE(places.device(2));

    // An ordinal outside the range names nothing and changes nothing.
    cg::EligiblePlaces const before = places;
    places.add_device(-1);
    places.add_device(cg::EligiblePlaces::max_devices);
    CHECK(places == before);
    CHECK_FALSE(places.device(-1));
    CHECK_FALSE(places.device(cg::EligiblePlaces::max_devices));

    places.clear_devices();
    CHECK(places == cg::EligiblePlaces{});
    CHECK(places.host());
}

TEST_CASE("DeviceEligibility - a node starts host only", "[ComputeGraph][DeviceEligibility]") {
    cg::Node const node;
    CHECK(node.eligible.host());
    CHECK_FALSE(node.eligible.any_device());
}

TEST_CASE("DeviceEligibility - nodes the dispatcher cannot take have no device", "[ComputeGraph][DeviceEligibility]") {
    auto A = matrix("A", 64, 64);
    auto B = matrix("B", 64, 64);
    auto C = matrix("C", 64, 64);

    RuntimeTensor<float> R("R", std::vector<std::size_t>{4, 4, 4, 4});
    RuntimeTensor<float> S("S", std::vector<std::size_t>{4, 4, 4, 4});
    RuntimeTensor<float> T("T", std::vector<std::size_t>{4, 4, 4, 4});
    R.zero();
    S.zero();
    T.zero();

    cg::Graph graph("device-eligibility-ineligible");
    {
        cg::CaptureGuard const guard(graph);
        // OpKind::Gemm carries a GemmDescriptor, which no device dispatcher looks at.
        cg::gemm(1.0F, A, B, 0.0F, &C);
        // A rank-4 contraction matches neither the GEMM nor the GEMV shape, and no strided batch.
        cg::einsum("ijkl <- ijmn ; mnkl", 0.0F, &T, 1.0F, R, S);
    }

    auto const *gemm = find_kind(graph, cg::OpKind::Gemm);
    REQUIRE(gemm != nullptr);
    auto const *tensor_contraction = find_kind(graph, cg::OpKind::Einsum);
    REQUIRE(tensor_contraction != nullptr);

    // The shared function asks nothing of the machine, so this holds on every build.
    cg::passes::compute_device_eligibility(graph);
    CHECK(gemm->eligible.host());
    CHECK_FALSE(gemm->eligible.any_device());
    CHECK(tensor_contraction->eligible.host());
    CHECK_FALSE(tensor_contraction->eligible.any_device());
}

TEST_CASE("DeviceEligibility - a dtype the backend rejects has no device", "[ComputeGraph][DeviceEligibility]") {
    auto A = matrix64("A", 64, 64);
    auto B = matrix64("B", 64, 64);
    auto C = matrix64("C", 64, 64);

    cg::Graph graph("device-eligibility-float64");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }

    cg::passes::compute_device_eligibility(graph);
    CHECK(graph.nodes()[0].eligible.any_device() == backend_takes_float64);
    if constexpr (gpu::has_mps) {
        CHECK_FALSE(graph.nodes()[0].eligible.any_device());
    }
}

TEST_CASE("DeviceEligibility - a GEMM, a GEMV, a scale and an axpby have the device", "[ComputeGraph][DeviceEligibility]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();
    auto A = matrix("A", 64, 64);
    auto B = matrix("B", 64, 64);
    auto C = matrix("C", 64, 64);
    auto D = matrix("D", 64, 64);

    RuntimeTensor<float> x("x", std::vector<std::size_t>{64});
    RuntimeTensor<float> y("y", std::vector<std::size_t>{64});
    x.zero();
    y.zero();

    cg::Graph graph("device-eligibility-eligible");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
        cg::einsum("i <- ij ; j", 0.0F, &y, 1.0F, A, x);
        cg::scale(2.0F, &D);
        cg::axpby(1.0F, A, 1.0F, &D);
    }
    REQUIRE(graph.num_nodes() == 4);

    cg::passes::DeviceEligibility pass;
    CHECK(pass.run(graph));
    CHECK(pass.num_eligible() == 4);
    for (auto const &node : graph.nodes()) {
        INFO("node " << node.id << " " << cg::op_kind_name(node.kind));
        CHECK(node.eligible.host());
        CHECK(node.eligible.device(0));
        CHECK_FALSE(node.eligible.device(1));
    }
    CHECK_FALSE(pass.explain().empty());

    // Idempotent: a second run finds every set as it left it.
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_eligible() == 4);
}

TEST_CASE("DeviceEligibility - float64 is rejected under MPS and accepted elsewhere", "[ComputeGraph][DeviceEligibility]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();
    auto A = matrix64("A", 64, 64);
    auto B = matrix64("B", 64, 64);
    auto C = matrix64("C", 64, 64);

    cg::Graph graph("device-eligibility-float64-pass");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
        cg::scale(2.0, &C);
    }

    cg::passes::DeviceEligibility pass;
    pass.run(graph);
    for (auto const &node : graph.nodes()) {
        CHECK(node.eligible.any_device() == backend_takes_float64);
    }
}

TEST_CASE("DeviceEligibility - a graph is recomputed when it changes under the set", "[ComputeGraph][DeviceEligibility]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();
    auto A = matrix("A", 64, 64);
    auto B = matrix("B", 64, 64);
    auto C = matrix("C", 64, 64);

    cg::Graph graph("device-eligibility-stale");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
    }
    cg::EligiblePlaces wrong;
    wrong.add_device(5);
    graph.set_eligible(0, wrong);

    // A stale device is dropped, not merged.
    cg::passes::DeviceEligibility pass;
    CHECK(pass.run(graph));
    CHECK(graph.nodes()[0].eligible.device(0));
    CHECK_FALSE(graph.nodes()[0].eligible.device(5));
}

TEST_CASE("DeviceEligibility - nodes inside a loop body get their set", "[ComputeGraph][DeviceEligibility][Loop]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();
    auto A = matrix("A", 64, 64);
    auto B = matrix("B", 64, 64);
    auto C = matrix("C", 64, 64);

    cg::Graph graph("device-eligibility-loop");
    auto     &body = graph.add_loop("iter", 1, [](std::size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
    }

    cg::passes::DeviceEligibility pass;
    CHECK(pass.run(graph));
    REQUIRE(body.num_nodes() == 1);
    CHECK(body.nodes()[0].eligible.device(0));
    // The loop node itself is control flow and has no kernel.
    CHECK_FALSE(graph.nodes()[0].eligible.any_device());
}

TEST_CASE("DeviceEligibility - every node GPUPlacement places is eligible", "[ComputeGraph][DeviceEligibility][GPU]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();

    // Large enough for the byte threshold to place a float32 GEMM and a scale; one rank-4
    // contraction no device can run; one GEMM too small to be worth it but eligible all the same.
    auto build = [](cg::Graph &graph, RuntimeTensor<float> &A, RuntimeTensor<float> &B, RuntimeTensor<float> &C,
                    RuntimeTensor<float> &small_a, RuntimeTensor<float> &small_b, RuntimeTensor<float> &small_c, RuntimeTensor<float> &R,
                    RuntimeTensor<float> &S, RuntimeTensor<float> &T) {
        auto &body = graph.add_loop("iter", 1, [](std::size_t) { return false; });
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
            cg::scale(2.0F, &C);
            cg::einsum("ij <- ik ; kj", 0.0F, &small_c, 1.0F, small_a, small_b);
            cg::einsum("ijkl <- ijmn ; mnkl", 0.0F, &T, 1.0F, R, S);
        }
        {
            cg::CaptureGuard const guard(body);
            cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
        }
    };

    auto A = matrix("A", 128, 128);
    auto B = matrix("B", 128, 128);
    auto C = matrix("C", 128, 128);
    auto a = matrix("a", 2, 2);
    auto b = matrix("b", 2, 2);
    auto c = matrix("c", 2, 2);

    RuntimeTensor<float> R("R", std::vector<std::size_t>{16, 16, 16, 16});
    RuntimeTensor<float> S("S", std::vector<std::size_t>{16, 16, 16, 16});
    RuntimeTensor<float> T("T", std::vector<std::size_t>{16, 16, 16, 16});
    R.zero();
    S.zero();
    T.zero();

    // A fixed throughput model rather than detect_default(), which measures the host.
    auto fixed = [] {
        cg::passes::GPUPlacement placement;
        placement.cpu_throughput_gflops  = 50.0;
        placement.gpu_throughput_gflops  = 5000.0;
        placement.pcie_bandwidth_gbs     = 12.0;
        placement.gpu_launch_overhead_us = 10.0;
        return placement;
    };

    // Placement on its own, as a test of the pass constructed alone.
    cg::Graph alone("placement-alone");
    build(alone, A, B, C, a, b, c, R, S, T);
    auto placement_alone = fixed();
    placement_alone.run(alone);

    // Eligibility first, then placement, as the default pipeline runs them.
    cg::Graph piped("placement-after-eligibility");
    build(piped, A, B, C, a, b, c, R, S, T);
    cg::passes::DeviceEligibility eligibility;
    eligibility.run(piped);
    auto placement_piped = fixed();
    placement_piped.run(piped);

    CHECK(placement_alone.num_placed() > 0);
    CHECK(placement_piped.num_placed() == placement_alone.num_placed());

    // The same decisions either way, node for node.
    std::vector<cg::Target> targets_alone;
    std::vector<cg::Target> targets_piped;
    for_all_nodes(alone, [&](cg::Node const &n) { targets_alone.push_back(n.target); });
    for_all_nodes(piped, [&](cg::Node const &n) { targets_piped.push_back(n.target); });
    CHECK(targets_alone == targets_piped);

    // Every node placed on the device can run there; and eligibility is wider than placement.
    std::size_t placed   = 0;
    std::size_t eligible = 0;
    for (auto *graph : {&alone, &piped}) {
        for_all_nodes(*graph, [&](cg::Node const &node) {
            if (node.target == cg::Target::GPU) {
                ++placed;
                CHECK(node.eligible.device(0));
            }
            if (node.eligible.any_device()) {
                ++eligible;
            }
            if (node.kind == cg::OpKind::Einsum && node.op_data.holds<cg::EinsumDescriptor>() &&
                node.op_data.get_if<cg::EinsumDescriptor>()->spec.target_indices.size() == 4) {
                CHECK_FALSE(node.eligible.any_device());
            }
        });
    }
    CHECK(placed > 0);
    CHECK(eligible > placed);
}

TEST_CASE("DeviceEligibility - runs immediately before GPUPlacement in the default pipeline", "[ComputeGraph][DeviceEligibility]") {
    auto const               pm = cg::PassManager::create_default();
    std::vector<std::string> names;
    for (auto const &pass : pm.passes()) {
        names.push_back(pass->name());
    }
    auto const eligibility = std::ranges::find(names, std::string{"DeviceEligibility"});
    auto const placement   = std::ranges::find(names, std::string{"GPUPlacement"});
    REQUIRE(eligibility != names.end());
    REQUIRE(placement != names.end());
    CHECK(eligibility + 1 == placement);
}

TEST_CASE("DeviceEligibility - the set is not saved and does not stop a save", "[ComputeGraph][DeviceEligibility][SaveLoad]") {
    DEVICE_ELIGIBILITY_SKIP_WITHOUT_GPU();
    auto A = matrix("A", 64, 64);
    auto B = matrix("B", 64, 64);
    auto C = matrix("C", 64, 64);

    cg::Graph graph("device-eligibility-save");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
    }
    auto const before = cg::save_graph_string(graph);
    REQUIRE(before.has_value());

    cg::passes::DeviceEligibility pass;
    REQUIRE(pass.run(graph));
    REQUIRE(graph.nodes()[0].eligible.device(0));

    // It writes a field a save leaves out, so the graph is not machine-dependent for it.
    CHECK(graph.machine_dependent_passes().empty());
    auto const after = cg::save_graph_string(graph);
    REQUIRE(after.has_value());
    CHECK(*after == *before);
    CHECK(after->find("eligible") == std::string::npos);

    // A loaded graph starts host only.
    auto loaded = cg::load_graph_string(*after);
    REQUIRE(loaded.has_value());
    for (auto const &node : loaded->nodes()) {
        CHECK_FALSE(node.eligible.any_device());
    }
}
