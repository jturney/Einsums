//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ===========================================================================
// Helper: count transfer nodes by kind and optionally by tensor id
// ===========================================================================
namespace {

struct TransferCounts {
    size_t                                   h2d_total{0};
    size_t                                   d2h_total{0};
    std::unordered_map<cg::TensorId, size_t> h2d_per_tensor;
    std::unordered_map<cg::TensorId, size_t> d2h_per_tensor;
};

TransferCounts count_transfers(std::vector<cg::Node> const &nodes) {
    TransferCounts c;
    for (auto const &n : nodes) {
        if (n.kind == cg::OpKind::HostToDevice) {
            c.h2d_total++;
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            if (desc)
                c.h2d_per_tensor[desc->tensor_id]++;
        } else if (n.kind == cg::OpKind::DeviceToHost) {
            c.d2h_total++;
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            if (desc)
                c.d2h_per_tensor[desc->tensor_id]++;
        }
    }
    return c;
}

size_t count_gpu_nodes(std::vector<cg::Node> const &nodes) {
    size_t count = 0;
    for (auto const &n : nodes) {
        if (n.target == cg::Target::GPU && n.kind != cg::OpKind::HostToDevice && n.kind != cg::OpKind::DeviceToHost)
            count++;
    }
    return count;
}

} // namespace

// ===========================================================================
// Infrastructure tests
// ===========================================================================

TEST_CASE("Residency enum on TensorHandle defaults to Host", "[ComputeGraph][GPU]") {
    cg::TensorHandle const h;
    CHECK(h.residency == cg::Residency::Host);
}

TEST_CASE("Node Target defaults to CPU", "[ComputeGraph][GPU]") {
    cg::Node const n;
    CHECK(n.target == cg::Target::CPU);
}

TEST_CASE("op_kind_name for transfer ops", "[ComputeGraph][GPU]") {
    CHECK(std::string(cg::op_kind_name(cg::OpKind::HostToDevice)) == "HostToDevice");
    CHECK(std::string(cg::op_kind_name(cg::OpKind::DeviceToHost)) == "DeviceToHost");
}

TEST_CASE("TransferDescriptor in OpData variant", "[ComputeGraph][GPU]") {
    cg::OpData data = cg::TransferDescriptor{.tensor_id = 42, .size_bytes = 1024};
    auto      *desc = std::get_if<cg::TransferDescriptor>(&data);
    REQUIRE(desc != nullptr);
    CHECK(desc->tensor_id == 42);
    CHECK(desc->size_bytes == 1024);
}

// ===========================================================================
// GPUPlacement tests
// ===========================================================================

TEST_CASE("GPUPlacement - large Einsum placed on GPU", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("gpu-placement");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 1);
    CHECK(graph.nodes()[0].kind == cg::OpKind::Einsum);
    CHECK(graph.nodes()[0].target == cg::Target::CPU);

    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    CHECK(modified);
    CHECK(pass.num_placed() == 1);
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
}

TEST_CASE("GPUPlacement - double precision rejected on MPS", "[ComputeGraph][GPU]") {
    // MPS only supports float32 GEMM. Double-precision operations should stay on CPU.
    auto A = create_random_tensor<double>("A", 64, 64);
    auto B = create_random_tensor<double>("B", 64, 64);
    auto C = create_zero_tensor<double>("C", 64, 64);

    cg::Graph graph("double-rejected");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    if constexpr (einsums::gpu::has_mps) {
        CHECK_FALSE(modified);
        CHECK(pass.num_placed() == 0);
        CHECK(graph.nodes()[0].target == cg::Target::CPU);
    }
    // On CUDA/HIP/mock, double is supported and would be placed.
}

TEST_CASE("GPUPlacement - small operation stays on CPU", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 2, 2);
    auto B = create_random_tensor<float>("B", 2, 2);
    auto C = create_zero_tensor<float>("C", 2, 2);

    cg::Graph graph("gpu-placement-small");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // High thresholds, 2x2 won't qualify.
    cg::passes::GPUPlacement pass(1000000, 1000000);
    bool const               modified = pass.run(graph);

    CHECK_FALSE(modified);
    CHECK(pass.num_placed() == 0);
    CHECK(graph.nodes()[0].target == cg::Target::CPU);
}

TEST_CASE("GPUPlacement - mix of large and small nodes places selectively", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto x = create_random_tensor<float>("x", 2, 2);
    auto y = create_random_tensor<float>("y", 2, 2);
    auto z = create_zero_tensor<float>("z", 2, 2);

    cg::Graph graph("mixed-placement");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &z, 1.0, x, y);
    }

    REQUIRE(graph.num_nodes() == 2);

    // Use thresholds that the 64x64 GEMM passes but the 2x2 does not.
    // 64x64 tensors: 3 * 64*64*8 = 98304 bytes > 65536 default threshold.
    // 2x2 tensors:   3 * 2*2*8   = 96 bytes    < 65536.
    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    CHECK(modified);

    // Large einsum on GPU, small stays on CPU.
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
    CHECK(graph.nodes()[1].target == cg::Target::CPU);
}

TEST_CASE("GPUPlacement - places a GEMM inside a loop body", "[ComputeGraph][GPU][Loop]") {
    // The hot GEMM lives entirely inside an SCF-style loop body. The
    // loop-aware placement walks the tree and must place it on GPU, a
    // flat-graph-only pass would leave it on CPU.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("gpu-placement-loop");
    auto     &body = graph.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    cg::passes::GPUPlacement pass;
    bool const               modified = pass.run(graph);

    if constexpr (einsums::gpu::has_gpu || einsums::gpu::is_mock) {
        CHECK(modified);
        CHECK(pass.num_placed() == 1);
        // The body's einsum is the placed node, not the parent Loop node.
        CHECK(body.nodes()[0].target == cg::Target::GPU);
        // The parent only holds the Loop node, still CPU/control-flow.
        CHECK(graph.nodes()[0].target == cg::Target::CPU);
    }
}

TEST_CASE("GPUPlacement - shared budget across loop boundary", "[ComputeGraph][GPU][Loop]") {
    // Two large GEMMs compete for device memory: one at the parent level,
    // one inside the loop body. With a budget that fits only one, the
    // shared-budget placement must place exactly one (the larger), not
    // both: proving parent and body draw from the same budget.
    auto Abig = create_random_tensor<float>("Abig", 256, 256);
    auto Bbig = create_random_tensor<float>("Bbig", 256, 256);
    auto Cbig = create_zero_tensor<float>("Cbig", 256, 256);
    auto Asm  = create_random_tensor<float>("Asm", 128, 128);
    auto Bsm  = create_random_tensor<float>("Bsm", 128, 128);
    auto Csm  = create_zero_tensor<float>("Csm", 128, 128);

    cg::Graph graph("gpu-shared-budget");
    {
        cg::CaptureGuard const guard(graph); // parent-level large GEMM
        cg::einsum("ik;kj->ij", 0.0, &Cbig, 1.0, Abig, Bbig);
    }
    auto &body = graph.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body); // body-level smaller GEMM
        cg::einsum("ik;kj->ij", 0.0, &Csm, 1.0, Asm, Bsm);
    }

    if constexpr (einsums::gpu::has_gpu || einsums::gpu::is_mock) {
        // Budget fits the 256^3 GEMM (3 * 256*256*4 = 786432 bytes) but not
        // both it and the 128 GEMM (196608 bytes).
        size_t const saved = einsums::gpu::available_device_memory();
        einsums::gpu::set_mock_device_memory_limit(800000);

        cg::passes::GPUPlacement pass;
        pass.run(graph);
        CHECK(pass.num_placed() == 1);
        // The larger (parent) GEMM wins the budget; the body GEMM stays CPU.
        CHECK(graph.nodes()[0].target == cg::Target::GPU);
        CHECK(body.nodes()[0].target == cg::Target::CPU);

        einsums::gpu::set_mock_device_memory_limit(saved);
    }
}

TEST_CASE("TransferInsertion - inserts transfers inside a loop body", "[ComputeGraph][GPU][Loop]") {
    // After GPUPlacement marks a body GEMM as GPU, TransferInsertion must
    // recurse and insert H2D before it (and D2H after) so the body is
    // self-contained, otherwise the GPU op reads host-only memory.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("xfer-loop");
    auto     &body = graph.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    if constexpr (einsums::gpu::has_gpu || einsums::gpu::is_mock) {
        cg::PassManager pm;
        pm.add<cg::passes::GPUPlacement>();
        pm.add<cg::passes::TransferInsertion>();
        graph.apply(pm);

        // The body should now contain H2D transfer node(s).
        size_t h2d = 0;
        for (auto const &n : body.nodes()) {
            if (n.kind == cg::OpKind::HostToDevice) {
                h2d++;
            }
        }
        CHECK(h2d >= 1);
    }
}

TEST_CASE("GPUPlacement - idempotent (running twice has no effect)", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("idempotent");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    auto [modified1, pass1] = graph.apply<cg::passes::GPUPlacement>();
    CHECK(modified1);
    CHECK(pass1.num_placed() == 1);

    // Run again, already placed, should be no-op.
    auto [modified2, pass2] = graph.apply<cg::passes::GPUPlacement>();
    CHECK_FALSE(modified2);
    CHECK(pass2.num_placed() == 0);
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
}

TEST_CASE("GPUPlacement - empty graph is a no-op", "[ComputeGraph][GPU]") {
    cg::Graph graph("empty");

    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    CHECK_FALSE(modified);
    CHECK(pass.num_placed() == 0);
}

TEST_CASE("GPUPlacement - Scale op is GPU-capable", "[ComputeGraph][GPU]") {
    auto C = create_random_tensor<float>("C", 128, 128);

    cg::Graph graph("scale-placement");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
    }

    REQUIRE(graph.num_nodes() == 1);
    CHECK(graph.nodes()[0].kind == cg::OpKind::Scale);

    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    CHECK(modified);
    CHECK(pass.num_placed() == 1);
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
}

TEST_CASE("GPUPlacement - cost model rejects when transfer overhead dominates", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("cost-model");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Manually set estimated_flops to a small value where transfer overhead dominates.
    graph.nodes()[0].estimated_flops = 1000;  // tiny: 1000 FLOPs
    graph.nodes()[0].estimated_bytes = 98304; // but large transfer

    cg::passes::GPUPlacement pass;
    // With default cost model: cpu_time = 1000 / 50e9 = 0.02 us
    // gpu_time = 1000 / 5000e9 + 98304 / 12e9 + 10e-6 ≈ 0 + 8.2us + 10us = 18.2us
    // gpu_time >> cpu_time → rejected
    bool const modified = pass.run(graph);

    CHECK_FALSE(modified);
    CHECK(pass.num_placed() == 0);
    CHECK(graph.nodes()[0].target == cg::Target::CPU);
}

TEST_CASE("GPUPlacement - cost model accepts when compute dominates", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("cost-model-accept");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Large flops, reasonable transfer.
    graph.nodes()[0].estimated_flops = 500000000; // 500M FLOPs
    graph.nodes()[0].estimated_bytes = 98304;

    cg::passes::GPUPlacement pass;
    // cpu_time = 500e6 / 50e9 = 10ms
    // gpu_time = 500e6 / 5000e9 + 98304 / 12e9 + 10us ≈ 0.1ms + 0.008ms + 0.01ms = 0.118ms
    // gpu_time << cpu_time → accepted
    bool const modified = pass.run(graph);

    CHECK(modified);
    CHECK(pass.num_placed() == 1);
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
}

TEST_CASE("GPUPlacement - budget limits placement", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);
    auto E = create_zero_tensor<float>("E", 128, 128);

    cg::Graph graph("budget-limited");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &E, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 3);

    // Each 128x128 float einsum has A(64K) + B(64K) + output(64K) = ~192K bytes per node.
    // Set budget to fit 1 node but not all 3.
    einsums::gpu::set_mock_device_memory_limit(200000); // ~200 KB

    cg::passes::GPUPlacement pass;
    bool const               modified = pass.run(graph);

    // Should place some but not all nodes.
    CHECK(modified);
    CHECK(pass.num_placed() >= 1);
    CHECK(pass.num_placed() < 3);

    // Count how many are on GPU.
    size_t gpu_count = 0;
    for (auto const &n : graph.nodes()) {
        if (n.target == cg::Target::GPU)
            gpu_count++;
    }
    CHECK(gpu_count == pass.num_placed());

    // Restore default limit so other tests aren't affected.
    einsums::gpu::set_mock_device_memory_limit(0);
}

TEST_CASE("GPUPlacement - unlimited budget places all candidates", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("unlimited-budget");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    // Default mock limit is huge (system RAM / 2), so both should fit.
    einsums::gpu::set_mock_device_memory_limit(0); // reset to default
    auto [modified, pass] = graph.apply<cg::passes::GPUPlacement>();

    CHECK(modified);
    CHECK(pass.num_placed() == 2);
}

// ===========================================================================
// TransferInsertion tests
// ===========================================================================

TEST_CASE("TransferInsertion - precise H2D/D2H counts for single GPU node", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("transfer-insertion-precise");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    graph.apply<cg::passes::GPUPlacement>();
    REQUIRE(graph.nodes()[0].target == cg::Target::GPU);

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK(modified);

    auto tc = count_transfers(graph.nodes());

    // The einsum reads A, B, C and writes C.
    // All three inputs need H2D (A, B, C all start on Host).
    // C is written by GPU → residency becomes Device.
    // No later CPU node reads C → no D2H needed... unless there's an implicit
    // "the user will read C after execute()" assumption. Our pass only inserts
    // D2H when a later CPU NODE reads it. Since there's no later node, no D2H.
    CHECK(tc.h2d_total >= 2); // A and B definitely. C depends on whether it's in inputs.
    CHECK(pass.num_transfers() == tc.h2d_total + tc.d2h_total);

    // Verify transfer nodes have valid TransferDescriptors with non-zero size_bytes.
    for (auto const &n : graph.nodes()) {
        if (n.kind == cg::OpKind::HostToDevice || n.kind == cg::OpKind::DeviceToHost) {
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            REQUIRE(desc != nullptr);
            CHECK(desc->size_bytes > 0);
            CHECK_FALSE(n.label.empty());
        }
    }
}

TEST_CASE("TransferInsertion - skips H2D for dead input (c_prefactor=0)", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("dead-input");
    {
        cg::CaptureGuard const guard(graph);
        // c_prefactor=0.0 means C's initial value is not read.
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    graph.apply<cg::passes::GPUPlacement>();
    REQUIRE(graph.nodes()[0].target == cg::Target::GPU);

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK(modified);

    auto tc = count_transfers(graph.nodes());

    // C should NOT get an H2D because its input value is dead (c_pf=0).
    // Only A and B need H2D.
    CHECK(tc.h2d_total == 2); // A and B, not C

    // The graph should still execute correctly.
    auto C_ref = Tensor<float, 2>("C_ref", 128, 128);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    graph.apply<cg::passes::TransferElimination>();
    graph.execute();

    for (size_t ii = 0; ii < 128; ii++) {
        for (size_t jj = 0; jj < 128; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-4f));
        }
    }
}

TEST_CASE("TransferInsertion - GPU→GPU chain: no D2H between consecutive GPU nodes", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("gpu-chain");
    {
        cg::CaptureGuard const guard(graph);
        // Node 1: C = A * B  (GPU)
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        // Node 2: D = C * B  (GPU), C flows directly GPU→GPU
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    graph.apply<cg::passes::GPUPlacement>();

    // Both should be on GPU.
    CHECK(graph.nodes()[0].target == cg::Target::GPU);
    CHECK(graph.nodes()[1].target == cg::Target::GPU);

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK(modified);

    auto tc = count_transfers(graph.nodes());

    // Find C's tensor ID by looking at node 1's outputs.
    // Between the two GPU nodes, C should NOT get a D2H+H2D round-trip.
    // After node 1, C is on Device. Node 2 reads C and is also GPU,
    // so C's residency (Device or Both) satisfies node 2's input requirement.

    // No D2H between the two GPU nodes (C flows GPU→GPU without round-trip).
    // But final D2H nodes are inserted for user-visible tensors left on device
    // (C and D are both non-intermediate and Device-resident after the GPU nodes).
    // Check that there are NO D2H between the two GPU nodes, only at the end.

    // Find the GPU nodes in the graph and verify no D2H appears between them.
    bool found_first_gpu = false;
    bool d2h_between_gpu = false;
    for (auto const &n : graph.nodes()) {
        if (n.target == cg::Target::GPU && n.kind == cg::OpKind::Einsum) {
            if (found_first_gpu) {
                break; // Second GPU node, stop checking.
            }
            found_first_gpu = true;
            continue;
        }
        if (found_first_gpu && n.kind == cg::OpKind::DeviceToHost) {
            d2h_between_gpu = true; // D2H found between the two GPU nodes, bad!
        }
    }
    CHECK_FALSE(d2h_between_gpu); // No D2H round-trip between consecutive GPU nodes.
    CHECK(tc.d2h_total > 0);      // But there should be final D2H for user-visible results.
}

TEST_CASE("TransferInsertion - D2H inserted when CPU node reads GPU output", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("d2h-needed");
    {
        cg::CaptureGuard const guard(graph);
        // Node 1: C = A * B  (will be GPU)
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        // Node 2: D = 2.0 * C (will be CPU due to small thresholds? No, Scale is GPU-capable)
        // We need a node that stays on CPU. Use a Permute instead.
        cg::permute("ij <- ij", 0.0, &D, 1.0, C);
    }

    REQUIRE(graph.num_nodes() == 2);

    // Place only large-enough nodes on GPU. Permute of 64x64 will also be placed.
    // To force the second node to CPU, use custom thresholds.
    // Actually, let's just manually check what happens and verify D2H is inserted
    // when a CPU node needs GPU output.
    // Strategy: place node 1 on GPU, leave node 2 on CPU.
    graph.nodes()[0].target = cg::Target::GPU;
    // node 1 is GPU, node 2 is CPU and reads C.

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK(modified);

    auto tc = count_transfers(graph.nodes());

    // Node 1 (GPU) writes C → C becomes Device-resident.
    // Node 2 (CPU) reads C → needs D2H for C.
    CHECK(tc.d2h_total >= 1);
}

TEST_CASE("TransferInsertion - no transfers for CPU-only graph", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 4, 4);
    auto B = create_random_tensor<float>("B", 4, 4);
    auto C = create_zero_tensor<float>("C", 4, 4);

    cg::Graph graph("no-transfers");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK_FALSE(modified);
    CHECK(pass.num_transfers() == 0);
}

TEST_CASE("TransferInsertion - empty graph is a no-op", "[ComputeGraph][GPU]") {
    cg::Graph graph("empty");

    auto [modified, pass] = graph.apply<cg::passes::TransferInsertion>();
    CHECK_FALSE(modified);
    CHECK(pass.num_transfers() == 0);
}

TEST_CASE("TransferInsertion - residency updated on TensorHandle", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("residency-update");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // All tensors start as Host.
    for (auto const &[tid, handle] : graph.tensors_map()) {
        CHECK(handle.residency == cg::Residency::Host);
    }

    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();

    // After insertion, input tensors that got H2D should be Both.
    // Output tensor written by GPU should be Device.
    bool found_non_host = false;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.residency != cg::Residency::Host)
            found_non_host = true;
    }
    CHECK(found_non_host);
}

// ===========================================================================
// TransferElimination tests
// ===========================================================================

TEST_CASE("TransferElimination - no redundancy when insertion is optimal", "[ComputeGraph][GPU]") {
    // TransferInsertion already tracks residency, so consecutive GPU nodes
    // sharing an input (A, B) do NOT get duplicate H2D nodes.
    // Elimination should be a no-op in this case.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("elim-noop");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);
    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();

    auto before = count_transfers(graph.nodes());

    // Shared tensors A, B should already have exactly 1 H2D each.
    for (auto const &[tid, count] : before.h2d_per_tensor) {
        CHECK(count == 1);
    }

    // Elimination should find nothing redundant.
    auto [elim_modified, elim_pass] = graph.apply<cg::passes::TransferElimination>();
    CHECK_FALSE(elim_modified);
    CHECK(elim_pass.num_eliminated() == 0);
}

TEST_CASE("TransferElimination - removes manually injected redundant H2D", "[ComputeGraph][GPU]") {
    // Manually construct a graph with a redundant H2D to test that
    // elimination actually removes it when redundancy exists.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("elim-manual");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();

    // Manually duplicate an H2D node to create redundancy.
    auto &nodes = graph.nodes();
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        if (nodes[idx].kind == cg::OpKind::HostToDevice) {
            // Insert a duplicate right after.
            cg::Node dup = nodes[idx]; // copy
            nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(idx) + 1, std::move(dup));
            break;
        }
    }
    graph.mark_sorted();

    size_t const nodes_before = graph.num_nodes();

    auto [elim_modified, elim_pass] = graph.apply<cg::passes::TransferElimination>();

    CHECK(elim_modified);
    CHECK(elim_pass.num_eliminated() >= 1);
    CHECK(graph.num_nodes() < nodes_before);
}

TEST_CASE("TransferElimination - removes manually injected redundant D2H", "[ComputeGraph][GPU]") {
    // Manually inject a duplicate D2H node and verify elimination removes it.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("elim-d2h");
    {
        cg::CaptureGuard const guard(graph);
        // C = A * B (GPU)
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        // D = C (CPU reads C, so D2H for C is needed)
        cg::permute("ij <- ij", 0.0, &D, 1.0, C);
    }

    // Force einsum to GPU, permute stays CPU.
    graph.nodes()[0].target = cg::Target::GPU;

    auto [ins_modified, ins_pass] = graph.apply<cg::passes::TransferInsertion>();
    REQUIRE(ins_modified);

    // Manually duplicate a D2H node to create redundancy.
    auto &nodes = graph.nodes();
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        if (nodes[idx].kind == cg::OpKind::DeviceToHost) {
            cg::Node dup = nodes[idx]; // copy
            nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(idx) + 1, std::move(dup));
            break;
        }
    }
    graph.mark_sorted();

    size_t const nodes_before = graph.num_nodes();

    auto [elim_modified, elim_pass] = graph.apply<cg::passes::TransferElimination>();

    CHECK(elim_modified);
    CHECK(elim_pass.num_eliminated() >= 1);
    CHECK(graph.num_nodes() < nodes_before);

    // After elimination, at most one D2H per tensor.
    auto tc = count_transfers(graph.nodes());
    for (auto const &[tid, count] : tc.d2h_per_tensor) {
        CHECK(count <= 1);
    }
}

TEST_CASE("TransferElimination - Belady eviction under memory pressure", "[ComputeGraph][GPU]") {
    // Three large GEMMs, each reading different tensors but sharing B.
    // With tight budget, some tensors must be evicted between operations.
    auto A1 = create_random_tensor<float>("A1", 128, 128);
    auto A2 = create_random_tensor<float>("A2", 128, 128);
    auto A3 = create_random_tensor<float>("A3", 128, 128);
    auto B  = create_random_tensor<float>("B", 128, 128);
    auto C1 = create_zero_tensor<float>("C1", 128, 128);
    auto C2 = create_zero_tensor<float>("C2", 128, 128);
    auto C3 = create_zero_tensor<float>("C3", 128, 128);

    cg::Graph graph("belady-eviction");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C1, 1.0, A1, B);
        cg::einsum("ik;kj->ij", 0.0, &C2, 1.0, A2, B);
        cg::einsum("ik;kj->ij", 0.0, &C3, 1.0, A3, B);
    }

    REQUIRE(graph.num_nodes() == 3);

    // Each 64x64 double tensor = 32,768 bytes.
    // After einsum 1: H2D(A1), H2D(B), H2D(C1) → 98,304 bytes on device.
    // Then GPU einsum 1 runs, C1 output tracked → still 98,304 (C1 was already counted).
    // Einsum 2 needs H2D(A2) and H2D(C2). B is already on device.
    // A2 = +32,768 → 131,072. C2 = +32,768 → 163,840.
    // Set budget to 100,000 bytes so even the first H2D batch barely fits,
    // and the second einsum's inputs definitely need eviction.
    einsums::gpu::set_mock_device_memory_limit(100000);

    // Place all on GPU manually (bypass GPUPlacement budget check).
    for (auto &n : graph.nodes())
        n.target = cg::Target::GPU;

    graph.apply<cg::passes::TransferInsertion>();

    auto         tc_before    = count_transfers(graph.nodes());
    size_t const nodes_before = graph.num_nodes();

    auto [modified, pass] = graph.apply<cg::passes::TransferElimination>();

    auto tc_after = count_transfers(graph.nodes());

    // Eviction should have inserted at least one D2H_evict node.
    // The total D2H count after should be >= the D2H count from before (which is 0,
    // since all nodes are GPU and there are no CPU consumers).
    CHECK(tc_after.d2h_total > tc_before.d2h_total);

    // Verify that eviction nodes have "D2H_evict" labels.
    size_t evict_count = 0;
    for (auto const &n : graph.nodes()) {
        if (n.kind == cg::OpKind::DeviceToHost && n.label.find("evict") != std::string::npos) {
            evict_count++;
            // Each eviction node should have a valid TransferDescriptor.
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            REQUIRE(desc != nullptr);
            CHECK(desc->size_bytes > 0);
        }
    }
    CHECK(evict_count > 0);

    // The graph should still be valid and executable.
    graph.execute();

    // Restore default limit.
    einsums::gpu::set_mock_device_memory_limit(0);
}

TEST_CASE("GPUDiagnostics - reports correct counts after pipeline", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("diagnostics-test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();
    pm.add<cg::passes::GPUDiagnostics>();
    graph.apply(pm);

    // Get the diagnostics pass from the PassManager.
    auto const *diag = dynamic_cast<cg::passes::GPUDiagnostics const *>(pm.passes().back().get());
    REQUIRE(diag != nullptr);

    CHECK(diag->gpu_nodes() == 2);
    CHECK(diag->h2d_transfers() >= 2); // At least A and B need H2D
    CHECK(diag->total_transfer_bytes() > 0);
    CHECK(diag->peak_device_bytes() > 0);

    // print_report shouldn't crash.
    std::ostringstream oss;
    diag->print_report(oss);
    CHECK_FALSE(oss.str().empty());
    CHECK(oss.str().find("GPU nodes") != std::string::npos);
}

TEST_CASE("GPUDiagnostics - aggregates node counts inside a loop body", "[ComputeGraph][GPU][Loop]") {
    // Two CPU einsums inside a loop body. A flat-graph-only pass would
    // count zero nodes (the top-level graph holds just the Loop node); the
    // aggregating pass must count the body's nodes.
    auto A = create_random_tensor<float>("A", 16, 16);
    auto B = create_random_tensor<float>("B", 16, 16);
    auto C = create_zero_tensor<float>("C", 16, 16);
    auto D = create_zero_tensor<float>("D", 16, 16);

    cg::Graph graph("diag-loop");
    auto     &body = graph.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    cg::passes::GPUDiagnostics diag;
    diag.run(graph);

    // 3 CPU nodes: the parent-level Loop node (counted as non-GPU) plus
    // the two body einsums. The key point is that the body's nodes are
    // now included in the count, a flat-graph-only pass would report 1.
    CHECK(diag.cpu_nodes() == 3);
    CHECK(diag.gpu_nodes() == 0);
}

TEST_CASE("MemoryPlanning - device memory tracking", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("mem-planning-device");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    // Apply GPU passes first.
    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();
    graph.apply<cg::passes::TransferElimination>();

    auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();

    CHECK_FALSE(modified); // analysis only
    CHECK(mem.total_memory() > 0);
    CHECK(mem.peak_memory() > 0);

    // With GPU nodes, device memory should be tracked.
    CHECK(mem.device_total_memory() > 0);
    CHECK(mem.device_peak_memory() > 0);
    CHECK(mem.device_peak_memory() <= mem.device_total_memory());

    // If tensors have non-overlapping lifetimes, there should be reuse savings.
    // With chained GEMMs, A is only used in node 1 and could be freed before D is created.
    // device_reuse_savings() = device_total - device_peak, may be > 0.

    std::ostringstream oss;
    mem.print_report(oss);
    CHECK(oss.str().find("Device") != std::string::npos);
}

TEST_CASE("MemoryPlanning - CPU-only graph has zero device memory", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 4, 4);
    auto B = create_random_tensor<float>("B", 4, 4);
    auto C = create_zero_tensor<float>("C", 4, 4);

    cg::Graph graph("mem-cpu-only");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // No GPU passes applied.
    auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();

    CHECK(mem.device_total_memory() == 0);
    CHECK(mem.device_peak_memory() == 0);
    CHECK(mem.device_reuse_savings() == 0);
}

TEST_CASE("Tensor set_data swap mechanism works", "[ComputeGraph][GPU]") {
    Tensor<float, 2> A("A", 4, 4);
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
            A(i, j) = static_cast<float>(i * 4 + j + 1);

    // Allocate shadow
    auto shadow_result = einsums::gpu::device_malloc(16 * sizeof(float));
    REQUIRE(shadow_result.has_value());
    void *shadow = shadow_result.value();

    // Copy host → shadow
    einsums::gpu::memcpy_host_to_device(shadow, A.data(), 16 * sizeof(float));

    // Swap tensor to point at shadow
    float *original = A.data();
    A.set_data(static_cast<float *>(shadow));
    CHECK(A.data() == static_cast<float *>(shadow));
    CHECK(A(0, 0) == Catch::Approx(1.0f));

    // Write through tensor (should write to shadow)
    A(0, 0) = 99.0f;
    CHECK(static_cast<float *>(shadow)[0] == Catch::Approx(99.0f));

    // Restore and flush
    A.set_data(original);
    CHECK(A(0, 0) == Catch::Approx(1.0f)); // Original untouched

    einsums::gpu::memcpy_device_to_host(A.data(), shadow, 16 * sizeof(float));
    CHECK(A(0, 0) == Catch::Approx(99.0f)); // Flushed from shadow

    einsums::gpu::device_free(shadow);
}

TEST_CASE("available_device_memory and set_mock_device_memory_limit", "[ComputeGraph][GPU]") {
    // Default: returns a large value (system RAM / 2 or 4GB fallback).
    einsums::gpu::set_mock_device_memory_limit(0);
    size_t const default_mem = einsums::gpu::available_device_memory();
    CHECK(default_mem > 0);
    CHECK(default_mem >= static_cast<long>(1024 * 1024)); // At least 1 MB.

    // Set a custom limit.
    einsums::gpu::set_mock_device_memory_limit(42000);
    CHECK(einsums::gpu::available_device_memory() == 42000);

    // Reset.
    einsums::gpu::set_mock_device_memory_limit(0);
    CHECK(einsums::gpu::available_device_memory() == default_mem);
}

TEST_CASE("TransferElimination - nothing to eliminate in CPU-only graph", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("no-elimination");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::TransferElimination>();
    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("TransferElimination - empty graph is a no-op", "[ComputeGraph][GPU]") {
    cg::Graph graph("empty");

    auto [modified, pass] = graph.apply<cg::passes::TransferElimination>();
    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("TransferElimination - residency updated on TensorHandle", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("elim-residency");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
    }

    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();
    graph.apply<cg::passes::TransferElimination>();

    // After full pipeline, check that residency on handles is coherent
    // (not all stuck at Host).
    bool found_device = false;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.residency == cg::Residency::Device || handle.residency == cg::Residency::Both) {
            found_device = true;
        }
    }
    CHECK(found_device);
}

// ===========================================================================
// Full pipeline: placement → insertion → elimination → execute
// ===========================================================================

TEST_CASE("GPU pass pipeline end-to-end: chained GEMMs", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    // Reference: compute on CPU.
    auto C_ref = Tensor<float, 2>(C);
    auto D_ref = Tensor<float, 2>(D);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D_ref, 1.0, Indices{i, k}, C_ref, Indices{k, j}, B);

    cg::Graph graph("gpu-pipeline");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    cg::PassManager pm;
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();

    bool const modified = graph.apply(pm);
    CHECK(modified);

    // Verify structure.
    CHECK(count_gpu_nodes(graph.nodes()) == 2);

    // Execute: mock backend runs CPU lambdas, transfer nodes are no-ops.
    graph.execute();

    // Verify BOTH intermediates and final output.
    for (size_t ii = 0; ii < 128; ii++) {
        for (size_t jj = 0; jj < 128; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-4f));
            CHECK(D(ii, jj) == Catch::Approx(D_ref(ii, jj)).margin(1e-4f));
        }
    }
}

TEST_CASE("GPU pass pipeline: mixed GPU/CPU graph produces correct results", "[ComputeGraph][GPU]") {
    // A * B → C (GPU, large), then scale C by 2.0 (CPU, forced small threshold).
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    // Reference.
    auto C_ref = Tensor<float, 2>(C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    cg::Graph graph("mixed-gpu-cpu");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::scale(2.0, &C);
    }

    REQUIRE(graph.num_nodes() == 2);

    // Force einsum to GPU, scale to CPU to test the data-flow boundary.
    graph.nodes()[0].target = cg::Target::GPU;
    // node[1] stays CPU (Scale).

    graph.apply<cg::passes::TransferInsertion>();
    graph.apply<cg::passes::TransferElimination>();

    // There must be a D2H for C between GPU einsum and CPU scale.
    auto tc = count_transfers(graph.nodes());
    CHECK(tc.d2h_total >= 1);

    // Execute and verify correctness.
    graph.execute();

    for (size_t ii = 0; ii < 128; ii++) {
        for (size_t jj = 0; jj < 128; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-4f));
        }
    }
}

TEST_CASE("GPU runtime fallback when dispatch is not applicable", "[ComputeGraph][GPU]") {
    // Test CPU fallback for a GPU-placed node that isn't GEMM-dispatchable.
    // Scale is GPU-capable for placement but has no gpu::blas dispatch,
    // so the executor falls back to the CPU lambda.
    auto C = create_random_tensor<float>("C", 128, 128);

    // Reference.
    auto C_ref = Tensor<float, 2>(C);
    linear_algebra::scale(2.0f, &C_ref);

    cg::Graph graph("fallback-scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
    }

    // Apply GPU pipeline.
    cg::PassManager pm;
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();
    graph.apply(pm);

    // Scale should be placed on GPU (it's GPU-capable and large enough).
    bool found_gpu_scale = false;
    for (auto const &n : graph.nodes()) {
        if (n.target == cg::Target::GPU && n.kind == cg::OpKind::Scale)
            found_gpu_scale = true;
    }
    CHECK(found_gpu_scale);

    // Execute: GPU GEMM dispatch won't apply (it's a Scale, not Einsum),
    // so the CPU lambda runs as fallback with data swapped to shadows.
    REQUIRE_NOTHROW(graph.execute());

    // Result should be correct (computed by CPU fallback on shadow data).
    for (size_t ii = 0; ii < 128; ii++) {
        for (size_t jj = 0; jj < 128; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-4f));
        }
    }
}

TEST_CASE("StreamAssignment - transfers get stream 1, compute gets stream 0", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    cg::Graph graph("stream-assignment");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    graph.apply<cg::passes::GPUPlacement>();
    graph.apply<cg::passes::TransferInsertion>();

    auto [modified, pass] = graph.apply<cg::passes::StreamAssignment>();

    CHECK(modified);
    CHECK(pass.num_assigned() > 0);

    for (auto const &n : graph.nodes()) {
        if (n.kind == cg::OpKind::HostToDevice || n.kind == cg::OpKind::DeviceToHost) {
            CHECK(n.stream_id == 1); // transfer stream
        } else {
            CHECK(n.stream_id == 0); // compute stream
        }
    }
}

TEST_CASE("StreamAssignment - no transfers means no assignments", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 4, 4);
    auto B = create_random_tensor<float>("B", 4, 4);
    auto C = create_zero_tensor<float>("C", 4, 4);

    cg::Graph graph("no-streams");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // No GPU passes, all nodes stay CPU with default stream_id=0.
    auto [modified, pass] = graph.apply<cg::passes::StreamAssignment>();

    CHECK_FALSE(modified);
    CHECK(pass.num_assigned() == 0);
}

TEST_CASE("GPU pipeline: GEMM then Scale stays on GPU", "[ComputeGraph][GPU]") {
    // Scale should run on GPU via gpu::blas::scal when the tensor is already on device.
    // No D2H round-trip between GEMM and Scale.
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);

    // Reference.
    auto C_ref = Tensor<float, 2>(C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0f, &C_ref);

    cg::Graph graph("gemm-then-scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::scale(2.0, &C);
    }

    REQUIRE(graph.num_nodes() == 2);

    // Apply GPU passes.
    cg::PassManager pm;
    pm.add<cg::passes::ScaleAbsorption>();
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();
    graph.apply(pm);

    // Both einsum and scale should be on GPU.
    size_t gpu_count = 0;
    for (auto const &n : graph.nodes()) {
        if (n.target == cg::Target::GPU)
            gpu_count++;
    }
    CHECK(gpu_count >= 2);

    // There should be no D2H between GEMM and Scale (C stays on device).
    bool d2h_between = false;
    bool found_gemm  = false;
    for (auto const &n : graph.nodes()) {
        if (n.kind == cg::OpKind::Einsum && n.target == cg::Target::GPU) {
            found_gemm = true;
            continue;
        }
        if (found_gemm && n.kind == cg::OpKind::DeviceToHost) {
            // If there's a D2H before Scale, it's a round-trip
            d2h_between = true;
        }
        if (found_gemm && n.kind == cg::OpKind::Scale) {
            break; // Found Scale after GEMM, check is done
        }
    }
    CHECK_FALSE(d2h_between);

    // Execute and verify correctness.
    graph.execute();

    for (size_t ii = 0; ii < 128; ii++) {
        for (size_t jj = 0; jj < 128; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-4f));
        }
    }
}

TEST_CASE("GPU pass pipeline: node ordering is valid after all passes", "[ComputeGraph][GPU]") {
    auto A = create_random_tensor<float>("A", 128, 128);
    auto B = create_random_tensor<float>("B", 128, 128);
    auto C = create_zero_tensor<float>("C", 128, 128);
    auto D = create_zero_tensor<float>("D", 128, 128);

    cg::Graph graph("ordering");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::GPUPlacement>();
    pm.add<cg::passes::TransferInsertion>();
    pm.add<cg::passes::TransferElimination>();
    graph.apply(pm);

    // Verify: H2D nodes come before the GPU node that needs them.
    // D2H nodes come after the GPU node that produces the data.
    auto const                      &nodes = graph.nodes();
    std::unordered_set<cg::TensorId> available_on_device;
    std::unordered_set<cg::TensorId> available_on_host;

    // All tensors start on host.
    for (auto const &[tid, handle] : graph.tensors_map()) {
        available_on_host.insert(tid);
    }

    for (auto const &n : nodes) {
        if (n.kind == cg::OpKind::HostToDevice) {
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            REQUIRE(desc != nullptr);
            // Must be on host to transfer.
            CHECK(available_on_host.count(desc->tensor_id) > 0);
            available_on_device.insert(desc->tensor_id);
        } else if (n.kind == cg::OpKind::DeviceToHost) {
            auto const *desc = std::get_if<cg::TransferDescriptor>(&n.op_data);
            REQUIRE(desc != nullptr);
            // Must be on device to transfer back.
            CHECK(available_on_device.count(desc->tensor_id) > 0);
            available_on_host.insert(desc->tensor_id);
        } else if (n.target == cg::Target::GPU) {
            // All inputs must be on device.
            for (auto tid : n.inputs) {
                CHECK(available_on_device.count(tid) > 0);
            }
            // Outputs go to device.
            for (auto tid : n.outputs) {
                available_on_device.insert(tid);
                available_on_host.erase(tid); // invalidate host copy
            }
        } else {
            // CPU node: all inputs must be on host.
            for (auto tid : n.inputs) {
                CHECK(available_on_host.count(tid) > 0);
            }
            // Outputs go to host.
            for (auto tid : n.outputs) {
                available_on_host.insert(tid);
                available_on_device.erase(tid); // invalidate device copy
            }
        }
    }
}
