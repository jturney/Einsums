//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PlacementCheapest.cpp
/// @brief Tests for the cost-based placement policy: the rule alone on hand-built queries, then the
///        estimates the node runner computes, then whole graphs priced by a fixed cost model.
///
/// Every model here is built by hand, so no decision depends on the machine running the test.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/PlacementPolicy.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

#define CHEAPEST_SKIP_WITHOUT_GPU()                                                                                                        \
    do {                                                                                                                                   \
        if (!::einsums::gpu::gpu_available()) {                                                                                            \
            SKIP("no usable GPU device (gpu::gpu_available() is false)");                                                                  \
        }                                                                                                                                  \
    } while (0)

/// A model with round numbers: the host does @p cpu_gflops, the device 1000 GFLOP/s, and a byte
/// crosses between them at @p pcie_gbps. Launch latency is @p launch_us.
cg::CostModel make_model(double cpu_gflops, double cpu_mem_gbps, double pcie_gbps, double launch_us) {
    cg::CostModel model;
    model.source = "test";

    model.cpu.name                      = "test cpu";
    model.cpu.device_type               = cg::DeviceType::CPU;
    model.cpu.mem_bandwidth_gbps        = cpu_mem_gbps;
    model.cpu.kernel_launch_overhead_us = 0.0;
    model.cpu.alloc_overhead_us         = 0.0;
    model.cpu.gemm_efficiency           = {{.M = 16, .N = 16, .K = 16, .gflops = cpu_gflops}};

    model.gpu.name                      = "test gpu";
    model.gpu.device_type               = cg::DeviceType::GPU;
    model.gpu.mem_bandwidth_gbps        = 1000.0;
    model.gpu.device_bandwidth_gbps     = 1000.0;
    model.gpu.pcie_bandwidth_gbps       = pcie_gbps;
    model.gpu.gpu_launch_latency_us     = launch_us;
    model.gpu.kernel_launch_overhead_us = 0.0;
    model.gpu.alloc_overhead_us         = 0.0;
    model.gpu.gemm_efficiency           = {{.M = 16, .N = 16, .K = 16, .gflops = 1000.0}};
    return model;
}

/// Ten bytes cross in one microsecond: 4096 bytes take 10 us at 0.4096 GB/s.
constexpr double kSlowLink = 0.4096;

cg::EligiblePlaces with_device() {
    cg::EligiblePlaces places;
    places.add_device(0);
    return places;
}

/// What the policy answers for a node with the given estimates and inputs.
struct Ask {
    cg::Place                         planned{cg::Place::host()};
    cg::EligiblePlaces                eligible{with_device()};
    std::vector<cg::PlacementOperand> inputs;
    std::optional<double>             host_us;
    std::optional<double>             device_us;
    std::optional<double>             run_gain_us;
    std::optional<std::size_t>        run_boundary_bytes;

    [[nodiscard]] cg::Place of(cg::PlacementPolicy const &policy) const {
        cg::Node const           node;
        cg::PlacementQuery const query{.node               = node,
                                       .planned            = planned,
                                       .eligible           = eligible,
                                       .inputs             = inputs,
                                       .outputs            = {},
                                       .host_us            = host_us,
                                       .device_us          = device_us,
                                       .run_gain_us        = run_gain_us,
                                       .run_boundary_bytes = run_boundary_bytes};
        return policy.choose(query);
    }
};

cg::PlacementOperand operand(std::size_t bytes, cg::Validity valid) {
    return {.id = cg::TensorId{1}, .bytes = bytes, .valid = valid};
}

float pattern(std::size_t k, int seed) {
    return static_cast<float>((k * 7 + static_cast<std::size_t>(seed) * 13) % 17) / 8.0F - 1.0F;
}

void fill(RuntimeTensor<float> &t, int seed) {
    for (std::size_t k = 0; k < t.size(); ++k) {
        t.data()[k] = pattern(k, seed);
    }
}

void poison(RuntimeTensor<float> &t) {
    for (std::size_t k = 0; k < t.size(); ++k) {
        t.data()[k] = 1e30F;
    }
}

RuntimeTensor<float> mat(char const *name, std::size_t extent) {
    return RuntimeTensor<float>(name, std::vector<std::size_t>{extent, extent});
}

void check_close(RuntimeTensor<float> const &got, RuntimeTensor<float> const &want, char const *what) {
    REQUIRE(got.size() == want.size());
    for (std::size_t k = 0; k < want.size(); ++k) {
        INFO(what << " element " << k);
        REQUIRE(std::abs(got.data()[k] - want.data()[k]) <= 1e-3F * (1.0F + std::abs(want.data()[k])));
    }
}

/// Asks the policy it wraps, with its model, and remembers what each query carried.
class Recording final : public cg::PlacementPolicy {
  public:
    struct Seen {
        cg::Node const            *node;
        cg::Place                  chosen;
        std::optional<double>      host_us;
        std::optional<double>      device_us;
        std::optional<double>      run_gain_us;
        std::optional<std::size_t> run_boundary_bytes;
    };

    explicit Recording(cg::PlacementPolicy const &inner) : _inner(inner) {}

    [[nodiscard]] cg::Place choose(cg::PlacementQuery const &query) const override {
        cg::Place const        chosen = _inner.choose(query);
        std::scoped_lock const lock(_mutex);
        _seen.push_back({.node               = &query.node,
                         .chosen             = chosen,
                         .host_us            = query.host_us,
                         .device_us          = query.device_us,
                         .run_gain_us        = query.run_gain_us,
                         .run_boundary_bytes = query.run_boundary_bytes});
        return chosen;
    }
    [[nodiscard]] std::string          name() const override { return "recording"; }
    [[nodiscard]] cg::CostModel const *cost_model() const noexcept override { return _inner.cost_model(); }

    [[nodiscard]] Seen const *find(cg::Node const *node) const {
        std::scoped_lock const lock(_mutex);
        for (auto const &s : _seen) {
            if (s.node == node) {
                return &s;
            }
        }
        return nullptr;
    }
    void forget() {
        std::scoped_lock const lock(_mutex);
        _seen.clear();
    }

  private:
    cg::PlacementPolicy const &_inner;
    mutable std::mutex         _mutex;
    mutable std::vector<Seen>  _seen;
};

/// Reads the graph's copy counts as a node ends.
class CopiesAtEnd final : public cg::ExecutionObserver {
  public:
    CopiesAtEnd(cg::Graph const &graph, cg::NodeId node) : _graph(graph), _node(node) {}
    void on_node_end(cg::ExecutionEvent const &event) override {
        if (event.node == _node) {
            _downloads = _graph.device_transfers().downloads;
        }
    }
    [[nodiscard]] std::size_t downloads() const { return _downloads; }

  private:
    cg::Graph const         &_graph;
    cg::NodeId               _node;
    std::atomic<std::size_t> _downloads{static_cast<std::size_t>(-1)};
};

std::shared_ptr<cg::Executor> make_executor(int which) {
    switch (which) {
    case 1:
        return std::make_shared<cg::SequentialExecutor>();
    case 2:
        return std::make_shared<cg::OpenMPExecutor>();
    case 3:
        return std::make_shared<cg::DataflowExecutor>();
    default:
        return nullptr;
    }
}

char const *executor_name(int which) {
    constexpr char const *names[] = {"built-in replay", "SequentialExecutor", "OpenMPExecutor", "DataflowExecutor"};
    return names[which];
}

void eligibility(cg::Graph &graph) {
    cg::passes::DeviceEligibility pass;
    pass.run(graph);
}

/// The positions of the nodes of @p kind, in order; graph-owned scratch adds an allocation node
/// that is no part of what these tests place.
std::vector<std::size_t> nodes_of(cg::Graph const &graph, cg::OpKind kind) {
    std::vector<std::size_t> found;
    for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].kind == kind) {
            found.push_back(i);
        }
    }
    return found;
}

std::shared_ptr<cg::PlacementPolicy const> borrowed(cg::PlacementPolicy const &policy) {
    return std::shared_ptr<cg::PlacementPolicy const>(&policy, [](auto *) {});
}

constexpr char const *kGemm = "ij <- ik ; kj";

} // namespace

TEST_CASE("CheapestPlacement - a node with no device place or a missing estimate runs where it is planned",
          "[ComputeGraph][PlacementPolicy][Cheapest]") {
    cg::CheapestPlacement const policy(make_model(1.0, 10.0, 10.0, 5.0));
    CHECK(policy.name() == "cheapest");
    CHECK_FALSE(policy.follows_plan());
    REQUIRE(policy.cost_model() != nullptr);
    CHECK(policy.cost_model()->has_gpu());

    for (auto const planned : {cg::Place::host(), cg::Place::device(0)}) {
        Ask ask{.planned = planned, .inputs = {}, .host_us = 1000.0, .device_us = 1.0};
        ask.eligible = cg::EligiblePlaces{};
        CHECK(ask.of(policy) == planned); // priced to win on the device, but the device is not a place

        ask.eligible = with_device();
        ask.host_us.reset();
        CHECK(ask.of(policy) == planned);
        ask.host_us   = 1000.0;
        ask.device_us = std::nullopt;
        CHECK(ask.of(policy) == planned);
    }
}

TEST_CASE("CheapestPlacement - the rule", "[ComputeGraph][PlacementPolicy][Cheapest]") {
    // 1e4 bytes cross per microsecond, and a launch costs 5.
    cg::CheapestPlacement const policy(make_model(1.0, 10.0, 10.0, 5.0));

    SECTION("a node that pays for itself goes to the device") {
        Ask ask{.inputs = {operand(10000, cg::Validity::Host)}, .host_us = 1000.0, .device_us = 10.0};
        CHECK(ask.of(policy) == cg::Place::device(0)); // 1000 against 10 + 5 + 1
    }

    SECTION("inputs only the device holds keep a host-planned node there") {
        Ask ask{.inputs = {operand(1000000, cg::Validity::Device)}, .host_us = 10.0, .device_us = 10.0};
        CHECK(ask.of(policy) == cg::Place::device(0)); // 10 + 100 to fetch against 10 + 5
    }

    SECTION("inputs only the host holds, and a small gain, keep a node on the host") {
        Ask ask{.planned = cg::Place::device(0), .inputs = {operand(100000, cg::Validity::Host)}, .host_us = 10.0, .device_us = 2.0};
        CHECK(ask.of(policy) == cg::Place::host()); // 10 against 2 + 5 + 10, and not the plan
    }

    SECTION("inputs current on both sides cost nothing to cross") {
        Ask ask{.inputs = {operand(100000000, cg::Validity::Both)}, .host_us = 10.0, .device_us = 2.0};
        CHECK(ask.of(policy) == cg::Place::device(0)); // 10 against 2 + 5
    }

    SECTION("a run whose gain exceeds its boundary traffic and one launch enters") {
        Ask ask{.inputs             = {operand(100000, cg::Validity::Host)},
                .host_us            = 10.0,
                .device_us          = 2.0,
                .run_gain_us        = 100.0,
                .run_boundary_bytes = 10000}; // 1 + 5 against a gain of 100
        CHECK(ask.of(policy) == cg::Place::device(0));
    }

    SECTION("a run whose gain falls short of its boundary traffic does not") {
        Ask ask{.inputs             = {operand(100000, cg::Validity::Host)},
                .host_us            = 10.0,
                .device_us          = 2.0,
                .run_gain_us        = 100.0,
                .run_boundary_bytes = 1000000}; // 100 + 5 against a gain of 100
        CHECK(ask.of(policy) == cg::Place::host());
    }

    SECTION("a run with no boundary figure is not consulted") {
        Ask ask{.inputs = {operand(100000, cg::Validity::Host)}, .host_us = 10.0, .device_us = 2.0, .run_gain_us = 100.0};
        CHECK(ask.of(policy) == cg::Place::host());
    }

    SECTION("an exact tie keeps the planned place") {
        for (auto const planned : {cg::Place::host(), cg::Place::device(0)}) {
            Ask ask{.planned = planned, .inputs = {}, .host_us = 10.0, .device_us = 5.0}; // 10 against 5 + 5
            CHECK(ask.of(policy) == planned);
        }
    }

    SECTION("outputs are never charged") {
        cg::Node const             node;
        cg::PlacementOperand const out = operand(1000000000, cg::Validity::Host);
        cg::PlacementQuery const   query{.node      = node,
                                         .planned   = cg::Place::host(),
                                         .eligible  = with_device(),
                                         .inputs    = {},
                                         .outputs   = std::span<cg::PlacementOperand const>(&out, 1),
                                         .host_us   = 1000.0,
                                         .device_us = 10.0};
        CHECK(policy.choose(query) == cg::Place::device(0));
    }
}

TEST_CASE("CheapestPlacement - the built-in value", "[ComputeGraph][PlacementPolicy][Cheapest]") {
    auto const &names = cg::builtin_placement_policy_names();
    CHECK(std::ranges::find(names, std::string{"cheapest"}) != names.end());

    cg::PlacementPolicy const &policy = cg::builtin_placement_policy("cheapest");
    CHECK(policy.name() == "cheapest");
    CHECK(policy.cost_model() != nullptr);
    CHECK(&cg::builtin_placement_policy("cheapest") == &policy);

    // The policies that price nothing leave the runner nothing to price.
    for (char const *other : {"planned", "host", "device"}) {
        CHECK(cg::builtin_placement_policy(other).cost_model() == nullptr);
    }
}

TEST_CASE("CheapestPlacement - a small lone node stays on the host", "[ComputeGraph][PlacementPolicy][Cheapest]") {
    CHEAPEST_SKIP_WITHOUT_GPU();
    constexpr std::size_t E = 4;

    cg::CheapestPlacement const model(make_model(100.0, 40.0, 10.0, 5.0));
    Recording                   recording(model);
    for (int which = 0; which < 4; ++which) {
        DYNAMIC_SECTION(executor_name(which)) {
            auto      A = mat("A", E), B = mat("B", E), C = mat("C", E), want = mat("want", E);
            cg::Graph graph{"lone"};
            {
                cg::CaptureGuard const guard(graph);
                cg::einsum("ij <- ik ; kj", 0.0F, &C, 1.0F, A, B);
            }
            eligibility(graph);
            REQUIRE(graph.nodes()[0].eligible.any_device());
            graph.set_placement_policy(borrowed(recording));
            graph.set_executor(make_executor(which));

            for (int replay = 0; replay < 3; ++replay) {
                INFO("replay " << replay);
                fill(A, 1);
                fill(B, 2);
                poison(C);
                graph.execute();

                want.zero();
                testing::reference_einsum(kGemm, 0.0F, &want.impl(), 1.0F, A.impl(), B.impl());
                check_close(C, want, "C");
                CHECK_FALSE(graph.ran_on_device(0));
                auto const copies = graph.device_transfers();
                CHECK(copies.uploads == 0);
                CHECK(copies.downloads == 0);
            }
            auto const *seen = recording.find(&graph.nodes()[0]);
            REQUIRE(seen != nullptr);
            REQUIRE(seen->host_us.has_value());
            REQUIRE(seen->device_us.has_value());
            CHECK(*seen->host_us < *seen->device_us + 5.0);
            recording.forget();
        }
    }
}

TEST_CASE("CheapestPlacement - a run pays for the entry its members cannot pay alone", "[ComputeGraph][PlacementPolicy][Cheapest]") {
    CHEAPEST_SKIP_WITHOUT_GPU();
    constexpr std::size_t E = 32; // 4096 bytes a matrix, 10 us across the link

    // Each GEMM takes about 35 us on the host and nothing on the device. Alone, one is not worth
    // the launch (20 us) and its two inputs (20 us); the pair is, with the four matrices that
    // cross (40 us) and one launch.
    cg::CheapestPlacement const model(make_model(1.867, 2.048, kSlowLink, 20.0));
    Recording                   recording(model);

    for (int which = 0; which < 4; ++which) {
        DYNAMIC_SECTION(executor_name(which)) {
            auto      A = mat("A", E), B = mat("B", E), D = mat("D", E), C2 = mat("C2", E);
            auto      want1 = mat("want1", E), want2 = mat("want2", E);
            cg::Graph graph{"run"};
            {
                cg::CaptureGuard const guard(graph);
                auto                  &C1 = graph.create_runtime_tensor<float>("C1", {E, E});
                cg::einsum("ij <- ik ; kj", 0.0F, &C1, 1.0F, A, B);
                cg::einsum("ij <- ik ; kj", 0.0F, &C2, 1.0F, C1, D);
            }
            eligibility(graph);
            auto const gemms = nodes_of(graph, cg::OpKind::Einsum);
            REQUIRE(gemms.size() == 2);
            graph.set_placement_policy(borrowed(recording));
            graph.set_executor(make_executor(which));

            for (int replay = 0; replay < 3; ++replay) {
                INFO("replay " << replay);
                fill(A, 1);
                fill(B, 2);
                fill(D, 3);
                poison(C2);
                graph.execute();

                want1.zero();
                want2.zero();
                testing::reference_einsum(kGemm, 0.0F, &want1.impl(), 1.0F, A.impl(), B.impl());
                testing::reference_einsum(kGemm, 0.0F, &want2.impl(), 1.0F, want1.impl(), D.impl());
                check_close(C2, want2, "C2");
                CHECK(graph.ran_on_device(gemms[0]));
                CHECK(graph.ran_on_device(gemms[1]));
            }

            // The run is both nodes; A, B and D come in and only C2 goes out, C1 being scratch the
            // run itself reads.
            for (std::size_t const i : gemms) {
                auto const *seen = recording.find(&graph.nodes()[i]);
                REQUIRE(seen != nullptr);
                REQUIRE(seen->run_boundary_bytes.has_value());
                CHECK(*seen->run_boundary_bytes == 4 * E * E * sizeof(float));
                REQUIRE(seen->run_gain_us.has_value());
                REQUIRE(seen->host_us.has_value());
                REQUIRE(seen->device_us.has_value());
                CHECK(*seen->run_gain_us == Catch::Approx(2.0 * (*seen->host_us - *seen->device_us)));
                CHECK(*seen->host_us - *seen->device_us > 30.0);
                CHECK(*seen->host_us - *seen->device_us < 40.0);
            }
            recording.forget();
        }
    }
}

TEST_CASE("CheapestPlacement - a result read outside the run is boundary traffic", "[ComputeGraph][PlacementPolicy][Cheapest]") {
    CHEAPEST_SKIP_WITHOUT_GPU();
    constexpr std::size_t E = 32;

    cg::CheapestPlacement const model(make_model(1.867, 2.048, kSlowLink, 20.0));
    Recording                   recording(model);

    auto      A = mat("A", E), B = mat("B", E), C = mat("C", E), want1 = mat("want1", E), want2 = mat("want2", E);
    cg::Graph graph{"outside"};
    {
        cg::CaptureGuard const guard(graph);
        auto                  &S = graph.create_runtime_tensor<float>("S", {E, E});
        cg::einsum("ij <- ik ; kj", 0.0F, &S, 1.0F, A, B);
        cg::gemm(1.0F, S, B, 0.0F, &C); // OpKind::Gemm has no device dispatcher, so this node is not a member
    }
    eligibility(graph);
    auto const producer = nodes_of(graph, cg::OpKind::Einsum);
    auto const reader   = nodes_of(graph, cg::OpKind::Gemm);
    REQUIRE(producer.size() == 1);
    REQUIRE(reader.size() == 1);
    REQUIRE(graph.nodes()[producer[0]].eligible.any_device());
    REQUIRE_FALSE(graph.nodes()[reader[0]].eligible.any_device());
    graph.set_placement_policy(borrowed(recording));

    fill(A, 1);
    fill(B, 2);
    poison(C);
    graph.execute();

    want1.zero();
    want2.zero();
    testing::reference_einsum(kGemm, 0.0F, &want1.impl(), 1.0F, A.impl(), B.impl());
    testing::reference_einsum(kGemm, 0.0F, &want2.impl(), 1.0F, want1.impl(), B.impl());
    check_close(C, want2, "C");

    // A and B come in, and S goes out because the host-only GEMM reads it.
    auto const *seen = recording.find(&graph.nodes()[producer[0]]);
    REQUIRE(seen != nullptr);
    REQUIRE(seen->run_boundary_bytes.has_value());
    CHECK(*seen->run_boundary_bytes == 3 * E * E * sizeof(float));
}

TEST_CASE("CheapestPlacement - inputs only the device holds keep a host-planned node on the device",
          "[ComputeGraph][PlacementPolicy][Cheapest]") {
    CHEAPEST_SKIP_WITHOUT_GPU();
    constexpr std::size_t E = 32; // 4096 bytes a matrix, 10 us across the link

    // X = A B is worth its entry (29 us gained, against a launch of 5 and 20 to send A and B), and
    // scaling X on the host takes 2 us. With X only on the device the scale costs 12 there to
    // bring X down, against 5 on the device: it stays. The run's gain does not cover its boundary,
    // so nothing but the location of X moves the scale.
    cg::CheapestPlacement const model(make_model(2.26, 2.048, kSlowLink, 5.0));
    Recording                   recording(model);

    for (int which = 0; which < 4; ++which) {
        DYNAMIC_SECTION(executor_name(which)) {
            auto      A = mat("A", E), B = mat("B", E), X = mat("X", E), want = mat("want", E);
            cg::Graph graph{"resident"};
            {
                cg::CaptureGuard const guard(graph);
                cg::einsum("ij <- ik ; kj", 0.0F, &X, 1.0F, A, B);
                cg::scale(2.0F, &X);
            }
            eligibility(graph);
            REQUIRE(graph.nodes().size() == 2);
            REQUIRE(graph.nodes()[1].target == cg::Target::CPU);
            graph.set_placement_policy(borrowed(recording));
            graph.set_executor(make_executor(which));
            auto observer = std::make_shared<CopiesAtEnd>(graph, graph.nodes()[1].id);
            graph.set_observer(observer);

            for (int replay = 0; replay < 3; ++replay) {
                INFO("replay " << replay);
                fill(A, 1);
                fill(B, 2);
                poison(X);
                graph.execute();

                want.zero();
                testing::reference_einsum(kGemm, 0.0F, &want.impl(), 1.0F, A.impl(), B.impl());
                for (std::size_t k = 0; k < want.size(); ++k) {
                    want.data()[k] *= 2.0F;
                }
                check_close(X, want, "X");

                CHECK(graph.ran_on_device(0));
                auto const copies = graph.device_transfers();
                if constexpr (gpu::has_unified_memory) {
                    // One memory, so X is never "only on the device" and the scale is not worth a launch.
                    CHECK_FALSE(graph.ran_on_device(1));
                    CHECK(copies.uploads == 0);
                    CHECK(copies.downloads == 0);
                } else {
                    CHECK(graph.ran_on_device(1));
                    // Nothing came down before the scale finished: A, B and X went up, and X comes
                    // down once, when the replay ends.
                    CHECK(observer->downloads() == 0);
                    CHECK(copies.uploads == 3);
                    CHECK(copies.upload_bytes == 3 * E * E * sizeof(float));
                    CHECK(copies.downloads == 1);
                    CHECK(copies.download_bytes == E * E * sizeof(float));
                }
            }
            recording.forget();
        }
    }
}
