//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The width contract: a node planned at width w runs with exactly w threads
// available to its kernel, and the thread that ran it is left exactly as it was
// found. Everything here observes the OpenMP thread ceiling from inside a
// node's kernel, which is the value every threaded kernel in the tree forks
// from.

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The thread ceiling this build can actually move. A build without OpenMP has
/// no ICV to set, so `omp_set_num_threads` is a no-op there and every width
/// assertion below would be testing the stub rather than the contract.
bool openmp_ceiling_is_settable() {
    int const before = hardware::get_max_threads();
    hardware::set_num_threads(before + 1);
    bool const moved = hardware::get_max_threads() == before + 1;
    hardware::set_num_threads(before);
    return moved;
}

/// What one node saw while it ran.
struct Observation {
    std::atomic<int> max_threads{-1}; ///< OpenMP ceiling inside the kernel
    std::atomic<int> worker_id{-2};   ///< pool worker index, or -1 on the calling thread
};

/// A node whose entire effect is to record the thread state its kernel sees.
cg::Node observing_node(std::string label, std::uint16_t width, Observation *out) {
    cg::Node node;
    node.kind         = cg::OpKind::Custom;
    node.label        = std::move(label);
    node.thread_width = width;
    node.execute      = [out]() {
        out->max_threads.store(hardware::get_max_threads(), std::memory_order_relaxed);
        out->worker_id.store(task_pool::TaskPool::current_worker_id(), std::memory_order_relaxed);
    };
    return node;
}

/// A width no thread in the process already has, so an observation of it can
/// only have come from the executor's wrap. On a four-core machine the caller's
/// own ceiling is 4, which would make a width of 4 unfalsifiable.
std::uint16_t distinct_width(int baseline) {
    return baseline == 4 ? std::uint16_t{3} : std::uint16_t{4};
}

/// What an UNPLANNED node is entitled to see: the pin a pool worker sets at
/// startup, or the caller's own ceiling when `help_until` runs the node on the
/// thread that called execute().
bool is_unplanned_ceiling(int observed, int baseline) {
    return observed == 1 || observed == baseline;
}

} // namespace

TEST_CASE("ThreadWidth - a planned width reaches the kernel", "[ComputeGraph][Executor][ThreadWidth]") {
    if (!openmp_ceiling_is_settable()) {
        SKIP("build has no OpenMP runtime, so there is no thread ceiling to plan");
    }

    int const           baseline = hardware::get_max_threads();
    std::uint16_t const wide     = distinct_width(baseline);

    Observation observed;
    cg::Graph   graph("width_reaches_kernel");
    graph.add_node(observing_node("wide", wide, &observed));

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(observed.max_threads.load() == static_cast<int>(wide));
}

TEST_CASE("ThreadWidth - an unplanned node pays nothing", "[ComputeGraph][Executor][ThreadWidth]") {
    int const baseline = hardware::get_max_threads();

    Observation observed;
    cg::Graph   graph("unplanned");
    graph.add_node(observing_node("plain", 0, &observed));

    cg::DataflowExecutor df;
    graph.execute(df);

    int const seen = observed.max_threads.load();
    if (observed.worker_id.load() >= 0) {
        // On a pool worker the answer is the startup pin, exactly as before
        // widths existed.
        REQUIRE(seen == 1);
    } else {
        // help_until ran it on the calling thread, which no one has narrowed.
        REQUIRE(seen == baseline);
    }
    REQUIRE(is_unplanned_ceiling(seen, baseline));
}

TEST_CASE("ThreadWidth - a wide node leaves its thread as it found it", "[ComputeGraph][Executor][ThreadWidth]") {
    if (!openmp_ceiling_is_settable()) {
        SKIP("build has no OpenMP runtime, so there is no thread ceiling to plan");
    }

    int const           baseline = hardware::get_max_threads();
    std::uint16_t const wide     = distinct_width(baseline);

    // Enough of each to land on every worker a wide node could have widened.
    size_t const             wide_count   = task_pool::TaskPool::get_singleton().num_workers() + 1;
    size_t const             narrow_count = 4 * wide_count;
    std::vector<Observation> wide_seen(wide_count);
    std::vector<Observation> narrow_seen(narrow_count);

    cg::Graph graph("restores_after_wide");
    for (size_t i = 0; i < wide_count; i++) {
        graph.add_node(observing_node("wide_" + std::to_string(i), wide, &wide_seen[i]));
    }
    for (size_t i = 0; i < narrow_count; i++) {
        graph.add_node(observing_node("narrow_" + std::to_string(i), 0, &narrow_seen[i]));
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    for (auto const &obs : wide_seen) {
        REQUIRE(obs.max_threads.load() == static_cast<int>(wide));
    }
    for (auto const &obs : narrow_seen) {
        // The nodes are all independent, so a narrow one may well run BEFORE a
        // wide one; what must never happen is a narrow node inheriting a width
        // it was not planned with.
        REQUIRE(obs.max_threads.load() != static_cast<int>(wide));
        REQUIRE(is_unplanned_ceiling(obs.max_threads.load(), baseline));
    }
}

TEST_CASE("ThreadWidth - the calling thread survives a wide replay", "[ComputeGraph][Executor][ThreadWidth]") {
    if (!openmp_ceiling_is_settable()) {
        SKIP("build has no OpenMP runtime, so there is no thread ceiling to plan");
    }

    int const           baseline      = hardware::get_max_threads();
    int const           blas_baseline = blas::get_num_threads_this_thread();
    std::uint16_t const wide          = distinct_width(baseline);

    // More wide nodes than workers, so help_until is all but certain to run one
    // of them on this thread - the case that makes restoring the SAVED value
    // rather than the worker pin necessary.
    size_t const             count = 4 * (task_pool::TaskPool::get_singleton().num_workers() + 1);
    std::vector<Observation> observed(count);

    cg::Graph graph("caller_survives");
    for (size_t i = 0; i < count; i++) {
        graph.add_node(observing_node("wide_" + std::to_string(i), wide, &observed[i]));
    }

    // Replayed rather than run once, and the replays are counted: whether the
    // caller ends up running a node is the scheduler's business, and this case
    // asserts nothing at all about the caller on a replay where the workers
    // drained the queue before it could help. One replay that lands a node here
    // is enough to exercise the restore; without one the case is vacuous and
    // says so rather than passing.
    cg::DataflowExecutor df;
    bool                 ran_on_caller = false;
    for (int attempt = 0; attempt < 20; attempt++) {
        graph.execute(df);

        for (auto const &obs : observed) {
            REQUIRE(obs.max_threads.load() == static_cast<int>(wide));
            ran_on_caller = ran_on_caller || obs.worker_id.load() < 0;
        }
        REQUIRE(hardware::get_max_threads() == baseline);
        if (blas_baseline > 0) {
            // The vendor count is the half a raised OpenMP ICV hides: a vendor
            // that reports the calling thread's ICV when the thread set no
            // count of its own answers with the width while the guard is up, so
            // a guard that read its baseline through that ICV would restore the
            // width here and pin the caller to it for good.
            REQUIRE(blas::get_num_threads_this_thread() == blas_baseline);
        }
    }
    if (!ran_on_caller) {
        SKIP("the workers drained every replay, so the calling thread never ran a wide node");
    }
}

TEST_CASE("ThreadWidth - a throwing wide kernel still restores", "[ComputeGraph][Executor][ThreadWidth]") {
    if (!openmp_ceiling_is_settable()) {
        SKIP("build has no OpenMP runtime, so there is no thread ceiling to plan");
    }

    int const           baseline = hardware::get_max_threads();
    std::uint16_t const wide     = distinct_width(baseline);

    size_t const count = 4 * (task_pool::TaskPool::get_singleton().num_workers() + 1);

    cg::Graph failing("wide_throws");
    for (size_t i = 0; i < count; i++) {
        cg::Node node;
        node.kind         = cg::OpKind::Custom;
        node.label        = "thrower_" + std::to_string(i);
        node.thread_width = wide;
        node.execute      = []() { throw std::runtime_error("kernel failed while wide"); };
        failing.add_node(std::move(node));
    }

    cg::DataflowExecutor df;
    REQUIRE_THROWS(failing.execute(df));

    // The calling thread first, then every worker: a guard that unwound without
    // restoring would leave a widened thread behind, and the next unplanned
    // node to land on it would see the width.
    REQUIRE(hardware::get_max_threads() == baseline);

    std::vector<Observation> observed(count);
    cg::Graph                after("after_throw");
    for (size_t i = 0; i < count; i++) {
        after.add_node(observing_node("narrow_" + std::to_string(i), 0, &observed[i]));
    }
    after.execute(df);

    for (auto const &obs : observed) {
        REQUIRE(is_unplanned_ceiling(obs.max_threads.load(), baseline));
    }
}

TEST_CASE("ThreadWidth - the other executors ignore widths", "[ComputeGraph][Executor][ThreadWidth]") {
    int const           baseline = hardware::get_max_threads();
    std::uint16_t const wide     = distinct_width(baseline);

    Observation seq_seen;
    cg::Graph   seq_graph("sequential_widths");
    seq_graph.add_node(observing_node("wide", wide, &seq_seen));

    cg::SequentialExecutor seq;
    seq_graph.execute(seq);
    REQUIRE(seq_seen.max_threads.load() == baseline);
    REQUIRE(hardware::get_max_threads() == baseline);

    Observation omp_seen;
    cg::Graph   omp_graph("openmp_widths");
    omp_graph.add_node(observing_node("wide", wide, &omp_seen));

    cg::OpenMPExecutor omp;
    omp_graph.execute(omp);
    REQUIRE(omp_seen.max_threads.load() == baseline);
    REQUIRE(hardware::get_max_threads() == baseline);
}

TEST_CASE("ThreadWidth - widths do not change results", "[ComputeGraph][Executor][ThreadWidth]") {
    auto A = create_random_tensor<double>("A", 12, 9);
    auto B = create_random_tensor<double>("B", 9, 7);
    auto C = create_zero_tensor<double>("C", 12, 7);
    auto D = create_zero_tensor<double>("D", 12, 7);

    cg::Graph reference("width_reference");
    {
        cg::CaptureGuard const guard(reference);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(3.0, &C);
    }
    cg::SequentialExecutor seq;
    reference.execute(seq);

    cg::Graph planned("width_planned");
    {
        cg::CaptureGuard const guard(planned);
        cg::einsum("ik;kj->ij", &D, A, B);
        cg::scale(3.0, &D);
    }
    for (auto &node : planned.nodes()) {
        node.thread_width = distinct_width(hardware::get_max_threads());
    }

    cg::DataflowExecutor df;
    planned.execute(df);

    for (size_t r = 0; r < 12; r++) {
        for (size_t c = 0; c < 7; c++) {
            REQUIRE(std::abs(C(r, c) - D(r, c)) < 1e-12);
        }
    }
}

TEST_CASE("ThreadWidth - moldability follows the kernel and the vendor", "[ComputeGraph][Executor][ThreadWidth]") {
    auto kind_node = [](cg::OpKind kind) {
        cg::Node node;
        node.kind = kind;
        return node;
    };

    // Kernels einsums threads itself fork from the ICV the executor sets, so
    // they are governable whatever BLAS was linked.
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::Permute)));
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::HPTTPermute)));
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::Custom)));

    // A vendor call is governable only where the vendor listens.
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::Gemm)) == cg::blas_route_is_moldable());
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::Gemv)) == cg::blas_route_is_moldable());

    // A contraction is not a vendor call: PackedGemm's engine keeps the work in
    // its own tiled loops rather than deferring to a GEMM the wrappers' fence
    // would clamp, so the width reaches a kernel einsums threads. The two
    // batched-GEMM kinds are einsums' own loop over the batch and always were.
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::Einsum)));
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::BatchedGemm)));
    REQUIRE(cg::kernel_moldability(kind_node(cg::OpKind::GroupedBatchedGemm)));

    // Containers hold no kernel; their bodies carry the widths.
    REQUIRE_FALSE(cg::kernel_moldability(kind_node(cg::OpKind::Loop)));
    REQUIRE_FALSE(cg::kernel_moldability(kind_node(cg::OpKind::Conditional)));

    // The two ways a vendor can be told apart are independent facts, and at
    // most one of them holds: MKL keeps its own runtime, OpenBLAS reads ours.
    REQUIRE_FALSE((blas::has_per_thread_control() && blas::threads_with_openmp()));
}

TEST_CASE("ThreadWidth - a node width sends a contraction to the packed loops", "[ComputeGraph][Executor][ThreadWidth]") {
    // The mechanism the (T)-shaped contractions depend on, checked one level
    // below the executor. A vendor GEMM issued under a node width is clamped to
    // one thread by the BLAS wrappers' fence, so PackedGemm's engine declines
    // its own deferring fast paths there and packs the contraction instead.
    // Outside such a scope nothing changes, which is the half worth pinning:
    // eager callers and width-1 nodes must keep the route they always had.
    if (!blas::threads_with_openmp()) {
        SKIP("the fence only applies to a vendor that reads our OpenMP ICV");
    }
    if (hardware::get_max_threads() < 2) {
        SKIP("a fenced width has to be a width above one");
    }

    // "abc <- abd ; cd": two M indices, one N, one link. The DLPNO triples
    // shape, and after dim coalescing a single-M/N/K contraction - which is
    // exactly what the single-GEMM deferral picks up.
    constexpr size_t n     = 48;
    auto             K     = create_random_tensor<double>("K_xvvv", n, n, n);
    auto             t     = create_random_tensor<double>("t_kj", n, n);
    auto             eager = create_zero_tensor<double>("eager", n, n, n);
    auto             wide  = create_zero_tensor<double>("wide", n, n, n);

    cg::einsum("abd;cd->abc", &eager, K, t);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "packed_gemm");
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");

    {
        // What WidthGuard does around a wide node, minus the executor. The
        // ceiling this thread already carries is above one, so it is the width
        // being held; the flag is the half the guard adds.
        blas::set_moldable_width_scope(true);
        cg::einsum("abd;cd->abc", &wide, K, t);
        blas::set_moldable_width_scope(false);
    }
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "packed_gemm");
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");

    // Two kernels, one answer. The sums are reassociated, nothing else.
    for (size_t e = 0; e < eager.size(); e++) {
        REQUIRE(std::abs(eager.data()[e] - wide.data()[e]) < 1e-10 * std::max(1.0, std::abs(eager.data()[e])));
    }

    // Leaving the scope restores the route, so nothing about the eager path is
    // conditional on a wide call having happened earlier.
    cg::einsum("abd;cd->abc", &eager, K, t);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");
}

TEST_CASE("ThreadWidth - a rank-2 GEMM under a width leaves the vendor route", "[ComputeGraph][Executor][ThreadWidth]") {
    // The other half of the same seam: a matrix times a matrix is taken by the
    // string dispatch's own gemm_direct route before PackedGemm ever sees it,
    // so the width has to be honored there instead.
    if (!blas::threads_with_openmp()) {
        SKIP("the fence only applies to a vendor that reads our OpenMP ICV");
    }
    if (hardware::get_max_threads() < 2) {
        SKIP("a fenced width has to be a width above one");
    }

    constexpr size_t n     = 96;
    auto             A     = create_random_tensor<double>("A", n, n);
    auto             B     = create_random_tensor<double>("B", n, n);
    auto             eager = create_zero_tensor<double>("C_eager", n, n);
    auto             wide  = create_zero_tensor<double>("C_wide", n, n);

    cg::einsum("ik;kj->ij", &eager, A, B);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "gemm_direct");

    {
        blas::set_moldable_width_scope(true);
        cg::einsum("ik;kj->ij", &wide, A, B);
        blas::set_moldable_width_scope(false);
    }
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "packed_gemm");
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");

    for (size_t e = 0; e < eager.size(); e++) {
        REQUIRE(std::abs(eager.data()[e] - wide.data()[e]) < 1e-10 * std::max(1.0, std::abs(eager.data()[e])));
    }

    cg::einsum("ik;kj->ij", &eager, A, B);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "gemm_direct");
}

// ── Kernel route pins ───────────────────────────────────────────────────────
// A width decides how fast a contraction runs. Left to itself it also decides
// WHICH kernel runs it, and the vendor GEMM and the packed loops disagree in
// the last bit, so a plan that re-picks widths from wall-clock measurements
// re-picks the last digit of the answer with them. A pin settles the route on
// the node, deterministically, before any width is chosen.

namespace {

/// The site the node's dispatch reads, or null for a node that has none.
packed_gemm::ContractionSite *site_of(cg::Node &node) {
    auto *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    return desc != nullptr ? desc->site.get() : nullptr;
}

/// Pin every contraction in @p graph, as a plan-time chooser would.
size_t pin_every_contraction(cg::Graph &graph, packed_gemm::KernelRoute route) {
    size_t pinned = 0;
    for (auto &node : graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            site->route = route;
            pinned++;
        }
    }
    return pinned;
}

} // namespace

TEST_CASE("ThreadWidth - a pinned route decides the kernel, not the width", "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    // Run on the calling thread, through the sequential executor, so the
    // dispatch's thread-local route names are the ones this thread reads back.
    // That executor also ignores widths, so nothing here is fenced: whatever
    // route fires, the pin is the only thing that could have chosen it.
    constexpr size_t n = 48;
    auto             K = create_random_tensor<double>("K_xvvv", n, n, n);
    auto             t = create_random_tensor<double>("t_kj", n, n);
    auto             C = create_zero_tensor<double>("C", n, n, n);

    cg::Graph graph("route_pin_single_k");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("abd;cd->abc", &C, K, t);
    }
    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Packed) == 1);

    cg::SequentialExecutor seq;
    graph.execute(seq);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "packed_gemm");
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    REQUIRE(packed_gemm::last_route_pin() == packed_gemm::KernelRoute::Packed);

    // Unpinned, the same node at the same width takes the deferring fast path
    // again - so the pin is what moved it, and removing the pin restores
    // exactly what the node did before pins existed.
    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Adaptive) == 1);
    graph.execute(seq);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");
    REQUIRE(packed_gemm::last_route_pin() == packed_gemm::KernelRoute::Adaptive);
}

TEST_CASE("ThreadWidth - a pinned rank-2 GEMM leaves the string dispatch's vendor route",
          "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    // The gate one level up: a matrix times a matrix never reaches PackedGemm,
    // because the string dispatch's own vendor-GEMM route takes it first. That
    // gate has to read the same pin, or the shape would take one route here and
    // the other inside the engine.
    //
    // The route reads "gemm_direct_runtime" rather than "gemm_direct" because a
    // REPLAYED node reaches the dispatch through rank-erased operands: since
    // build_executor took over the einsum lowering, a graph node runs the same
    // way whether its operands were declared as Tensor<T,2> or as runtime-rank
    // ones. The two spellings name the same vendor GEMM - the runtime ladder
    // upcasts to TensorView<T,2> and calls the same helper - and the EAGER form
    // of this call, a few tests up, still reports "gemm_direct".
    constexpr size_t n = 96;
    auto             A = create_random_tensor<double>("A", n, n);
    auto             B = create_random_tensor<double>("B", n, n);
    auto             C = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("route_pin_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    cg::SequentialExecutor seq;

    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Packed) == 1);
    graph.execute(seq);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "packed_gemm");
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");

    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Vendor) == 1);
    graph.execute(seq);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "gemm_direct_runtime");
    REQUIRE(packed_gemm::last_route_pin() == packed_gemm::KernelRoute::Vendor);

    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Adaptive) == 1);
    graph.execute(seq);
    REQUIRE(std::string(cg::dispatch::last_dispatch_route()) == "gemm_direct_runtime");
}

TEST_CASE("ThreadWidth - a pin survives the decline memo it contradicts", "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    // A plain single-M/N/K GEMM is the one shape the engine deliberately turns
    // away, so the vendor can have it - and it MEMOIZES the refusal on the site.
    // A memo that ignored the route would answer "declined" to a pinned-packed
    // call and hand the shape straight back to the vendor, silently undoing the
    // pin from the second replay onward.
    constexpr size_t n = 64;
    auto             A = create_random_tensor<double>("A", n, n, n);
    auto             B = create_random_tensor<double>("B", n, n);
    auto             C = create_zero_tensor<double>("C", n, n, n);

    cg::Graph graph("route_pin_memo");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("abd;cd->abc", &C, A, B);
    }
    cg::SequentialExecutor seq;

    // Two unpinned replays: the first records the decline, the second reads it.
    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Adaptive) == 1);
    graph.execute(seq);
    graph.execute(seq);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");

    // Now pin, with that decline sitting on the site.
    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Packed) == 1);
    graph.execute(seq);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    graph.execute(seq);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");

    // And back: the memo written under the pin must not answer for the
    // unpinned regime either.
    REQUIRE(pin_every_contraction(graph, packed_gemm::KernelRoute::Adaptive) == 1);
    graph.execute(seq);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");
}

TEST_CASE("ThreadWidth - pinned routes make widths bit-identical", "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    if (!openmp_ceiling_is_settable()) {
        SKIP("build has no OpenMP runtime, so there is no thread ceiling to plan");
    }

    // Two link indices, so the engine's multi-K flatten path is in play: that
    // is the shape where the vendor GEMM and the packed loops were measured to
    // disagree in the last bit. Compared with ==, not a tolerance - a tolerance
    // is what this whole campaign is trying to stop accepting.
    constexpr size_t n      = 40;
    auto             A      = create_random_tensor<double>("A", n, n, n);
    auto             B      = create_random_tensor<double>("B", n, n, n);
    auto             narrow = create_zero_tensor<double>("narrow", n, n);
    auto             wide   = create_zero_tensor<double>("wide", n, n);

    auto build = [&](cg::Graph &graph, Tensor<double, 2> *out) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("acd;bcd->ab", out, A, B);
    };

    cg::Graph narrow_graph("route_pin_narrow");
    build(narrow_graph, &narrow);
    cg::Graph wide_graph("route_pin_wide");
    build(wide_graph, &wide);

    REQUIRE(pin_every_contraction(narrow_graph, packed_gemm::KernelRoute::Packed) == 1);
    REQUIRE(pin_every_contraction(wide_graph, packed_gemm::KernelRoute::Packed) == 1);

    // The two width plans the trial could land on for the same node.
    for (auto &node : narrow_graph.nodes()) {
        node.thread_width = 1;
    }
    std::uint16_t const w = distinct_width(hardware::get_max_threads());
    for (auto &node : wide_graph.nodes()) {
        node.thread_width = w;
    }

    cg::DataflowExecutor df;
    narrow_graph.execute(df);
    wide_graph.execute(df);

    for (size_t e = 0; e < narrow.size(); e++) {
        REQUIRE(narrow.data()[e] == wide.data()[e]);
    }

    // The pins are exactly where they were put: replaying does not touch them.
    for (auto &node : narrow_graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            REQUIRE(site->route == packed_gemm::KernelRoute::Packed);
        }
    }
    for (auto &node : wide_graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            REQUIRE(site->route == packed_gemm::KernelRoute::Packed);
        }
    }
}

TEST_CASE("ThreadWidth - the planner pins every contraction it may widen", "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    if (hardware::get_max_threads() < 2) {
        SKIP("a plan that cannot widen anything pins nothing, by design");
    }

    constexpr size_t n = 40;
    auto             A = create_random_tensor<double>("A", n, n, n);
    auto             B = create_random_tensor<double>("B", n, n, n);
    auto             C = create_zero_tensor<double>("C", n, n);

    cg::Graph graph("route_pin_planner");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("acd;bcd->ab", &C, A, B);
    }

    // Nothing is pinned until something plans.
    for (auto &node : graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            REQUIRE(site->route == packed_gemm::KernelRoute::Adaptive);
        }
    }

    graph.plan_threads();
    size_t pinned = 0;
    for (auto &node : graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            REQUIRE(site->route == packed_gemm::KernelRoute::Packed);
            pinned++;
        }
    }
    REQUIRE(pinned == 1);

    // Re-planning is what the timed width trial does between replays. It may
    // move widths; it may not move a route.
    graph.plan_threads();
    for (auto &node : graph.nodes()) {
        if (auto *site = site_of(node); site != nullptr) {
            REQUIRE(site->route == packed_gemm::KernelRoute::Packed);
        }
    }
}

TEST_CASE("ThreadWidth - an eager contraction is never pinned", "[ComputeGraph][Executor][ThreadWidth][RoutePin]") {
    // Eager callers pass no site, so there is nowhere for a pin to live and the
    // route comes from the thread regime exactly as it always did.
    constexpr size_t n = 48;
    auto             K = create_random_tensor<double>("K_xvvv", n, n, n);
    auto             t = create_random_tensor<double>("t_kj", n, n);
    auto             C = create_zero_tensor<double>("C", n, n, n);

    cg::einsum("abd;cd->abc", &C, K, t);
    REQUIRE(packed_gemm::last_route_pin() == packed_gemm::KernelRoute::Adaptive);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "single_k_gemm");
}
