//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PlacementPolicy.cpp
/// @brief Tests for the placement policy a replay asks as each node is dispatched: the built-in
///        policies, the option and the installed object, the nodes a policy is and is not asked
///        about, and that every result is the same wherever the nodes ran.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/PlacementPolicy.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <algorithm>
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

#define PLACEMENT_SKIP_WITHOUT_GPU()                                                                                                       \
    do {                                                                                                                                   \
        if (!::einsums::gpu::gpu_available()) {                                                                                            \
            SKIP("no usable GPU device (gpu::gpu_available() is false)");                                                                  \
        }                                                                                                                                  \
    } while (0)

constexpr std::size_t N = 16; // matrix extent
constexpr std::size_t Q = 3;  // rank-4 extent

float pattern(std::size_t k, int seed) {
    return static_cast<float>((k * 7 + static_cast<std::size_t>(seed) * 13) % 17) / 8.0F - 1.0F;
}

template <typename T>
void fill(T &t, int seed) {
    for (std::size_t k = 0; k < t.size(); ++k) {
        t.data()[k] = pattern(k, seed);
    }
}

/// A destination the graph overwrites: poisoned, so a missed write is deterministic.
template <typename T>
void poison(T &t) {
    for (std::size_t k = 0; k < t.size(); ++k) {
        t.data()[k] = 1e30F;
    }
}

RuntimeTensor<float> mat(char const *name, std::size_t rows, std::size_t cols) {
    return RuntimeTensor<float>(name, std::vector<std::size_t>{rows, cols});
}

/// C = A B over square column-major matrices of extent N, from the values now held.
std::vector<float> product(float const *a, float const *b) {
    std::vector<float> c(N * N, 0.0F);
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < N; ++k) {
            for (std::size_t i = 0; i < N; ++i) {
                c[i + j * N] += a[i + k * N] * b[k + j * N];
            }
        }
    }
    return c;
}

void check_close(float const *got, std::vector<float> const &want, char const *what) {
    for (std::size_t k = 0; k < want.size(); ++k) {
        INFO(what << " element " << k);
        REQUIRE(std::abs(got[k] - want[k]) <= 1e-4F * (1.0F + std::abs(want[k])));
    }
}

/// One graph exercising every kind of node a placement policy meets, with the values it must
/// produce. Built in place and not moved: the graph holds the tensors by address.
struct Workload {
    RuntimeTensor<float> A        = mat("A", N, N);
    RuntimeTensor<float> B        = mat("B", N, N);
    RuntimeTensor<float> C_einsum = mat("C_einsum", N, N); // GEMM einsum
    RuntimeTensor<float> C_gemm   = mat("C_gemm", N, N);   // OpKind::Gemm, which no device dispatcher takes
    RuntimeTensor<float> C_acc    = mat("C_acc", N, N);    // accumulating einsum: nonzero C prefactor
    RuntimeTensor<float> C_second = mat("C_second", N, N); // reads C_einsum, so it follows the first GEMM
    RuntimeTensor<float> C_body   = mat("C_body", N, N);   // written inside the loop body
    RuntimeTensor<float> x{"x", std::vector<std::size_t>{N}};
    RuntimeTensor<float> y{"y", std::vector<std::size_t>{N}};
    RuntimeTensor<float> S = mat("S", N, N); // scaled in place
    RuntimeTensor<float> X = mat("X", N, N);
    RuntimeTensor<float> Y = mat("Y", N, N); // axpby destination
    RuntimeTensor<float> R{"R", std::vector<std::size_t>{Q, Q, Q, Q}};
    RuntimeTensor<float> R2{"R2", std::vector<std::size_t>{Q, Q, Q, Q}};
    RuntimeTensor<float> T{"T", std::vector<std::size_t>{Q, Q, Q, Q}};
    Tensor<float, 2>     V{"V", N, N};     // the parent of a view
    Tensor<float, 2>     W{"W", N / 2, N}; // what the view's node writes

    // Node positions, in the order captured.
    std::size_t gemm_einsum{}, gemv{}, scale{}, axpby{}, rank4{}, gemm_kind{}, accumulate{}, second{}, view_node{};
    cg::Graph  *body{nullptr};

    cg::Graph graph{"placement-workload"};

    Workload() {
        auto &loop_body = graph.add_loop("iter", 2, [](std::size_t) { return false; });
        body            = &loop_body;
        {
            cg::CaptureGuard const guard(loop_body);
            cg::einsum("ij <- ik ; kj", 0.0F, &C_body, 1.0F, A, B);
        }
        auto const next = [&] { return graph.num_nodes(); };

        auto &slice = [&]() -> auto & {
            cg::CaptureGuard const guard(graph);
            return cg::view<float, 2>(V, cg::ViewAxis::range(0, N / 2), cg::ViewAxis::full());
        }();
        {
            cg::CaptureGuard const guard(graph);
            gemm_einsum = next();
            cg::einsum("ij <- ik ; kj", 0.0F, &C_einsum, 1.0F, A, B);
            gemv = next();
            cg::einsum("i <- ij ; j", 0.0F, &y, 1.0F, A, x);
            scale = next();
            cg::scale(2.0F, &S);
            axpby = next();
            cg::axpby(0.5F, X, 2.0F, &Y);
            rank4 = next();
            cg::einsum("ijkl <- ijmn ; mnkl", 0.0F, &T, 1.0F, R, R2);
            gemm_kind = next();
            cg::gemm(1.0F, A, B, 0.0F, &C_gemm);
            accumulate = next();
            cg::einsum("ij <- ik ; kj", 1.0F, &C_acc, 1.0F, A, B);
            second = next();
            cg::einsum("ij <- ik ; kj", 0.0F, &C_second, 1.0F, C_einsum, B);
            view_node = next();
            cg::axpby(1.0F, slice, 0.0F, &W);
        }
    }

    /// Put every operand back to its starting values; the graph's nodes are not all idempotent.
    void reset() {
        fill(A, 1);
        fill(B, 2);
        fill(x, 3);
        fill(S, 4);
        fill(X, 5);
        fill(Y, 6);
        fill(C_acc, 7);
        fill(R, 8);
        fill(R2, 9);
        fill(V, 10);
        for (auto *dest : {&C_einsum, &C_gemm, &C_second, &C_body, &y, &T}) {
            poison(*dest);
        }
        poison(W);
    }

    /// Check every result against loops written here, which share no code with the engine.
    void verify() const {
        auto const ab = product(A.data(), B.data());
        check_close(C_einsum.data(), ab, "GEMM einsum");
        check_close(C_gemm.data(), ab, "OpKind::Gemm");
        check_close(C_body.data(), ab, "loop body");
        check_close(C_second.data(), product(C_einsum.data(), B.data()), "chained GEMM");

        std::vector<float> acc(N * N);
        {
            RuntimeTensor<float> c0 = mat("c0", N, N);
            fill(c0, 7);
            for (std::size_t k = 0; k < acc.size(); ++k) {
                acc[k] = c0.data()[k] + ab[k];
            }
        }
        check_close(C_acc.data(), acc, "accumulating einsum");

        std::vector<float> want_y(N, 0.0F);
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t i = 0; i < N; ++i) {
                want_y[i] += A.data()[i + j * N] * x.data()[j];
            }
        }
        check_close(y.data(), want_y, "GEMV einsum");

        std::vector<float> want_s(N * N), want_y2(N * N);
        for (std::size_t k = 0; k < N * N; ++k) {
            want_s[k]  = 2.0F * pattern(k, 4);
            want_y2[k] = 0.5F * X.data()[k] + 2.0F * pattern(k, 6);
        }
        check_close(S.data(), want_s, "scale");
        check_close(Y.data(), want_y2, "axpby");

        std::vector<float> want_t(Q * Q * Q * Q, 0.0F);
        for (std::size_t l = 0; l < Q; ++l) {
            for (std::size_t k = 0; k < Q; ++k) {
                for (std::size_t j = 0; j < Q; ++j) {
                    for (std::size_t i = 0; i < Q; ++i) {
                        for (std::size_t n = 0; n < Q; ++n) {
                            for (std::size_t m = 0; m < Q; ++m) {
                                want_t[i + Q * (j + Q * (k + Q * l))] +=
                                    R.data()[i + Q * (j + Q * (m + Q * n))] * R2.data()[m + Q * (n + Q * (k + Q * l))];
                            }
                        }
                    }
                }
            }
        }
        check_close(T.data(), want_t, "rank-4 contraction");

        std::vector<float> want_w(N / 2 * N);
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t i = 0; i < N / 2; ++i) {
                want_w[i + j * (N / 2)] = V.data()[i + j * N];
            }
        }
        check_close(W.data(), want_w, "node reading a view");
    }

    /// Nodes of the parent graph the device runs when it is offered everything it can.
    [[nodiscard]] std::vector<std::size_t> device_nodes() const { return {gemm_einsum, gemv, scale, axpby, accumulate, second}; }

    [[nodiscard]] std::vector<std::size_t> host_nodes() const { return {rank4, gemm_kind, view_node}; }
};

void eligibility(cg::Graph &graph) {
    cg::passes::DeviceEligibility pass;
    pass.run(graph);
}

/// Place by a fixed throughput model rather than detect_default(), with a byte threshold that
/// sends the GEMMs and leaves the smaller nodes where they are.
void place(cg::Graph &graph) {
    cg::passes::GPUPlacement placement(2048);
    placement.cpu_throughput_gflops  = 50.0;
    placement.gpu_throughput_gflops  = 5000.0;
    placement.pcie_bandwidth_gbs     = 12.0;
    placement.gpu_launch_overhead_us = 10.0;
    placement.run(graph);
}

/// Asks the policy it wraps and remembers every query.
class Recording final : public cg::PlacementPolicy {
  public:
    struct Seen {
        cg::Node const                   *node;
        cg::Place                         planned;
        cg::Place                         chosen;
        cg::EligiblePlaces                eligible;
        std::vector<cg::PlacementOperand> inputs;
        std::vector<cg::PlacementOperand> outputs;
    };

    explicit Recording(cg::PlacementPolicy const &inner) : _inner(inner) {}

    [[nodiscard]] cg::Place choose(cg::PlacementQuery const &query) const override {
        cg::Place const        chosen = _inner.choose(query);
        std::scoped_lock const lock(_mutex);
        _seen.push_back({.node     = &query.node,
                         .planned  = query.planned,
                         .chosen   = chosen,
                         .eligible = query.eligible,
                         .inputs   = std::vector<cg::PlacementOperand>(query.inputs.begin(), query.inputs.end()),
                         .outputs  = std::vector<cg::PlacementOperand>(query.outputs.begin(), query.outputs.end())});
        return chosen;
    }
    [[nodiscard]] std::string name() const override { return "recording(" + _inner.name() + ")"; }

    [[nodiscard]] std::vector<Seen> seen() const {
        std::scoped_lock const lock(_mutex);
        return _seen;
    }

    [[nodiscard]] Seen const *find(std::vector<Seen> const &all, cg::Node const *node) const {
        for (auto const &s : all) {
            if (s.node == node) {
                return &s;
            }
        }
        return nullptr;
    }

  private:
    cg::PlacementPolicy const &_inner;
    mutable std::mutex         _mutex;
    mutable std::vector<Seen>  _seen;
};

/// Always answers with one fixed place.
class Fixed final : public cg::PlacementPolicy {
  public:
    explicit Fixed(cg::Place place) : _place(place) {}
    [[nodiscard]] cg::Place   choose(cg::PlacementQuery const &) const override { return _place; }
    [[nodiscard]] std::string name() const override { return "fixed"; }

  private:
    cg::Place _place;
};

/// Sets the placement option and puts back what it found.
class PlacementOption {
  public:
    explicit PlacementOption(std::string value) : _previous(config::get(option::GraphPlacement)) {
        config::set(option::GraphPlacement, std::move(value));
    }
    ~PlacementOption() { config::set(option::GraphPlacement, _previous); }
    PlacementOption(PlacementOption const &)            = delete;
    PlacementOption &operator=(PlacementOption const &) = delete;

  private:
    std::string _previous;
};

cg::PlannedPlacement const        planned_policy;
cg::HostPlacement const           host_policy;
cg::EligibleDevicePlacement const device_policy;

enum class Mode { Unplaced, Planned, Host, Device };

char const *mode_name(Mode mode) {
    switch (mode) {
    case Mode::Unplaced:
        return "no placement";
    case Mode::Planned:
        return "planned";
    case Mode::Host:
        return "host";
    case Mode::Device:
        return "device";
    }
    return "?";
}

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

} // namespace

TEST_CASE("PlacementPolicy - the built-in policies answer as documented", "[ComputeGraph][PlacementPolicy]") {
    CHECK(planned_policy.follows_plan());
    CHECK_FALSE(host_policy.follows_plan());
    CHECK_FALSE(device_policy.follows_plan());
    CHECK(planned_policy.name() == "planned");
    CHECK(host_policy.name() == "host");
    CHECK(device_policy.name() == "device");

    cg::Node const     node;
    cg::EligiblePlaces with_device;
    with_device.add_device(0);
    for (auto const eligible : {cg::EligiblePlaces{}, with_device}) {
        for (auto const planned : {cg::Place::host(), cg::Place::device(0)}) {
            cg::PlacementQuery const query{.node = node, .planned = planned, .eligible = eligible, .inputs = {}, .outputs = {}};
            CHECK(planned_policy.choose(query) == planned);
            CHECK(host_policy.choose(query) == cg::Place::host());
            CHECK(device_policy.choose(query) == (eligible.any_device() ? cg::Place::device(0) : cg::Place::host()));
        }
    }

    // Target maps onto a place: CPU is the host, GPU is device 0.
    CHECK(cg::Place::of(cg::Target::CPU) == cg::Place::host());
    CHECK(cg::Place::of(cg::Target::GPU) == cg::Place::device(0));
    CHECK(with_device.contains(cg::Place::device(0)));
    CHECK_FALSE(with_device.contains(cg::Place::device(1)));
    CHECK(cg::EligiblePlaces{}.contains(cg::Place::host()));
}

TEST_CASE("PlacementPolicy - results match the unplaced run under every policy and executor", "[ComputeGraph][PlacementPolicy][Executor]") {
    PLACEMENT_SKIP_WITHOUT_GPU();

    for (Mode const mode : {Mode::Unplaced, Mode::Planned, Mode::Host, Mode::Device}) {
        for (int which = 0; which < 4; ++which) {
            DYNAMIC_SECTION(mode_name(mode) << " under " << executor_name(which)) {
                Workload w;
                switch (mode) {
                case Mode::Unplaced:
                    break;
                case Mode::Planned:
                    place(w.graph);
                    break;
                case Mode::Host:
                    place(w.graph);
                    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&host_policy, [](auto *) {}));
                    break;
                case Mode::Device:
                    eligibility(w.graph);
                    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&device_policy, [](auto *) {}));
                    break;
                }
                w.graph.set_executor(make_executor(which));

                for (int replay = 0; replay < 3; ++replay) {
                    INFO("replay " << replay);
                    w.reset();
                    w.graph.execute();
                    w.verify();

                    // Where each node ran, read from the dispatch record and not from timing.
                    auto const &nodes = w.graph.nodes();
                    for (std::size_t i = 0; i < nodes.size(); ++i) {
                        INFO("node " << i);
                        switch (mode) {
                        case Mode::Unplaced:
                        case Mode::Host:
                            CHECK_FALSE(w.graph.ran_on_device(i));
                            break;
                        case Mode::Planned:
                            CHECK(w.graph.ran_on_device(i) == (nodes[i].target == cg::Target::GPU));
                            break;
                        case Mode::Device:
                            break;
                        }
                    }
                    if (mode == Mode::Device) {
                        for (std::size_t i : w.device_nodes()) {
                            INFO("device node " << i);
                            CHECK(w.graph.ran_on_device(i));
                        }
                        for (std::size_t i : w.host_nodes()) {
                            INFO("host node " << i);
                            CHECK_FALSE(w.graph.ran_on_device(i));
                        }
                        CHECK(w.body->ran_on_device(0));
                    } else {
                        if (mode != Mode::Planned) {
                            CHECK_FALSE(w.body->ran_on_device(0));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("PlacementPolicy - a planned graph run with the plan placed something", "[ComputeGraph][PlacementPolicy]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    place(w.graph);
    std::size_t placed = 0;
    for (auto const &node : w.graph.nodes()) {
        placed += node.target == cg::Target::GPU ? 1 : 0;
    }
    // The comparisons above mean nothing if the plan put nothing on the device.
    CHECK(placed > 0);
    CHECK(w.body->nodes()[0].target == cg::Target::GPU);
}

TEST_CASE("PlacementPolicy - a policy that returns the plan changes nothing", "[ComputeGraph][PlacementPolicy]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    place(w.graph);
    Recording recording(planned_policy);
    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&recording, [](auto *) {}));

    w.reset();
    w.graph.execute();
    w.verify();

    auto const &nodes = w.graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        INFO("node " << i);
        CHECK(w.graph.ran_on_device(i) == (nodes[i].target == cg::Target::GPU));
    }
}

TEST_CASE("PlacementPolicy - a node reading a view is not offered, and every other node is", "[ComputeGraph][PlacementPolicy]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    eligibility(w.graph);
    // The view's node is eligible, so a policy that asked about it would send it to the device.
    REQUIRE(w.graph.nodes()[w.view_node].eligible.any_device());

    Recording recording(device_policy);
    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&recording, [](auto *) {}));
    w.reset();
    w.graph.execute();
    w.verify();

    auto const seen = recording.seen();
    CHECK(recording.find(seen, &w.graph.nodes()[w.view_node]) == nullptr);
    CHECK_FALSE(w.graph.ran_on_device(w.view_node));
    for (std::size_t const i : w.device_nodes()) {
        INFO("node " << i);
        auto const *entry = recording.find(seen, &w.graph.nodes()[i]);
        REQUIRE(entry != nullptr);
        CHECK(entry->chosen == cg::Place::device(0));
    }
    for (std::size_t const i : {w.rank4, w.gemm_kind}) {
        INFO("node " << i);
        auto const *entry = recording.find(seen, &w.graph.nodes()[i]);
        REQUIRE(entry != nullptr);
        CHECK(entry->chosen == cg::Place::host());
        CHECK_FALSE(entry->eligible.any_device());
    }
}

TEST_CASE("PlacementPolicy - a place outside the eligible set is refused", "[ComputeGraph][PlacementPolicy]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    eligibility(w.graph);
    REQUIRE_FALSE(w.graph.nodes()[w.rank4].eligible.any_device());

    // The policy sends everything to the device, including the nodes no device kernel exists for.
    w.graph.set_placement_policy(std::make_shared<Fixed>(cg::Place::device(0)));
    w.reset();
    try {
        w.graph.execute();
        FAIL("a policy that chose a place the node cannot run should have thrown");
    } catch (std::logic_error const &error) {
        std::string const what = error.what();
        CHECK(what.find("fixed") != std::string::npos);
        CHECK(what.find("device 0") != std::string::npos);
    }

    // A device that does not exist is outside every set.
    w.graph.set_placement_policy(std::make_shared<Fixed>(cg::Place::device(1)));
    CHECK_THROWS_AS(w.graph.execute(), std::logic_error);
}

TEST_CASE("PlacementPolicy - a loop body is replayed under its parent's policy", "[ComputeGraph][PlacementPolicy][Loop]") {
    PLACEMENT_SKIP_WITHOUT_GPU();

    SECTION("inherited") {
        Workload w;
        eligibility(w.graph);
        eligibility(*w.body);
        Recording recording(device_policy);
        w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&recording, [](auto *) {}));
        w.reset();
        w.graph.execute();
        w.verify();

        auto const  seen  = recording.seen();
        auto const *entry = recording.find(seen, &w.body->nodes()[0]);
        REQUIRE(entry != nullptr);
        CHECK(entry->chosen == cg::Place::device(0));
        CHECK(w.body->ran_on_device(0));
    }

    SECTION("a body keeps a policy of its own") {
        Workload w;
        eligibility(w.graph);
        w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&device_policy, [](auto *) {}));
        w.body->set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&host_policy, [](auto *) {}));
        w.reset();
        w.graph.execute();
        w.verify();
        CHECK_FALSE(w.body->ran_on_device(0));
        CHECK(w.graph.ran_on_device(w.gemm_einsum));
    }

    SECTION("inherited under a concurrent executor") {
        Workload w;
        eligibility(w.graph);
        Recording recording(device_policy);
        w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&recording, [](auto *) {}));
        w.graph.set_executor(std::make_shared<cg::DataflowExecutor>());
        w.reset();
        w.graph.execute();
        w.verify();
        CHECK(recording.find(recording.seen(), &w.body->nodes()[0]) != nullptr);
    }
}

TEST_CASE("PlacementPolicy - a query says where each operand is current", "[ComputeGraph][PlacementPolicy]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    eligibility(w.graph);
    Recording recording(device_policy);
    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&recording, [](auto *) {}));
    w.reset();
    w.graph.execute();
    w.verify();

    auto const  seen  = recording.seen();
    auto const *first = recording.find(seen, &w.graph.nodes()[w.gemm_einsum]);
    REQUIRE(first != nullptr);
    REQUIRE(first->inputs.size() == 2);
    REQUIRE(first->outputs.size() == 1);
    for (auto const &operand : first->inputs) {
        CHECK(operand.bytes == N * N * sizeof(float));
        CHECK(operand.valid == cg::Validity::Host);
    }
    CHECK(first->outputs[0].bytes == N * N * sizeof(float));
    CHECK(first->planned == cg::Place::host());

    // The second GEMM reads the first one's result. It was written on the device, so on a build
    // whose device has its own memory that copy alone is current; where the memory is shared there
    // is no second copy.
    auto const *second = recording.find(seen, &w.graph.nodes()[w.second]);
    REQUIRE(second != nullptr);
    REQUIRE(second->inputs.size() == 2);
    cg::Validity const expected = gpu::has_unified_memory ? cg::Validity::Host : cg::Validity::Device;
    CHECK(second->inputs[0].valid == expected);
}

TEST_CASE("PlacementPolicy - the option selects the built-in policy", "[ComputeGraph][PlacementPolicy][Options]") {
    PLACEMENT_SKIP_WITHOUT_GPU();

    SECTION("the default is planned") {
        CHECK(config::get(option::GraphPlacement) == "planned");
        Workload w;
        place(w.graph);
        w.reset();
        w.graph.execute();
        w.verify();
        for (std::size_t i = 0; i < w.graph.nodes().size(); ++i) {
            CHECK(w.graph.ran_on_device(i) == (w.graph.nodes()[i].target == cg::Target::GPU));
        }
    }

    SECTION("host runs everything on the host, though the plan sent nodes to the device") {
        PlacementOption const option("host");
        Workload              w;
        place(w.graph);
        REQUIRE(w.graph.nodes()[w.gemm_einsum].target == cg::Target::GPU);
        w.reset();
        w.graph.execute();
        w.verify();
        for (std::size_t i = 0; i < w.graph.nodes().size(); ++i) {
            CHECK_FALSE(w.graph.ran_on_device(i));
        }
    }

    SECTION("device runs every eligible node on the device, though the plan left it") {
        PlacementOption const option("device");
        Workload              w;
        eligibility(w.graph);
        REQUIRE(w.graph.nodes()[w.gemm_einsum].target == cg::Target::CPU);
        w.reset();
        w.graph.execute();
        w.verify();
        for (std::size_t const i : w.device_nodes()) {
            CHECK(w.graph.ran_on_device(i));
        }
        CHECK(w.body->ran_on_device(0));
    }

    SECTION("a policy installed on the graph wins") {
        PlacementOption const option("device");
        Workload              w;
        eligibility(w.graph);
        w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&host_policy, [](auto *) {}));
        w.reset();
        w.graph.execute();
        w.verify();
        for (std::size_t i = 0; i < w.graph.nodes().size(); ++i) {
            CHECK_FALSE(w.graph.ran_on_device(i));
        }
        CHECK_FALSE(w.body->ran_on_device(0));
    }

    SECTION("an unknown value is an error naming the valid ones") {
        PlacementOption const option("fastest");
        Workload              w;
        w.reset();
        try {
            w.graph.execute();
            FAIL("an unknown placement should have thrown");
        } catch (std::invalid_argument const &error) {
            std::string const what = error.what();
            CHECK(what.find("fastest") != std::string::npos);
            for (char const *valid : {"planned", "host", "device"}) {
                CHECK(what.find(valid) != std::string::npos);
            }
        }
    }

    // Restored by each section's guard.
    CHECK(config::get(option::GraphPlacement) == "planned");
}

TEST_CASE("PlacementPolicy - nothing about a policy reaches a saved graph", "[ComputeGraph][PlacementPolicy][SaveLoad]") {
    PLACEMENT_SKIP_WITHOUT_GPU();
    Workload w;
    eligibility(w.graph);
    auto const before = cg::save_graph_string(w.graph);

    w.graph.set_placement_policy(std::shared_ptr<cg::PlacementPolicy const>(&device_policy, [](auto *) {}));
    w.reset();
    w.graph.execute();

    CHECK(w.graph.machine_dependent_passes().empty());
    auto const after = cg::save_graph_string(w.graph);
    CHECK(before.has_value() == after.has_value());
    if (before.has_value() && after.has_value()) {
        CHECK(*before == *after);
        CHECK(after->find("placement") == std::string::npos);
    }
}
