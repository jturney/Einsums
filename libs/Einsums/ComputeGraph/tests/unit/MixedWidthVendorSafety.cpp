//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One constraint, pinned: a node-scoped thread width must never reach the
// vendor BLAS.
//
// An OpenMP-built OpenBLAS obeys each caller's `omp_get_max_threads()` but does
// not support concurrent callers whose values differ. Its server syncs a
// process-global thread count to whatever the current caller's ICV says, frees
// the shared pack buffers when that global shrinks - under whatever GEMM is
// still in flight - and sizes its work queue from the global while forking the
// team from the caller. Two threads calling at different widths therefore
// corrupt each other's results or deadlock outright, which is a property of the
// library and not something einsums can arrange around.
//
// Two things keep that from happening. `blas_route_is_moldable` reports false
// unless the vendor scopes its thread count per caller, so the planner never
// writes a width onto a node whose kernel reaches such a vendor. And the
// `VendorWidthFence` in the BLAS level-2/3 wrappers clamps any vendor call made
// under `blas::moldable_width_scope` back to one thread, which holds at run
// time no matter where the width on the node came from.
//
// The shape below is the one that broke: three independent chains, each
// alternating a contraction with an accumulating permute that reads what the
// contraction wrote, replayed by DataflowExecutor with widths of 4, 4 and 2
// imposed across the chains. That puts wide and narrow callers in flight at the
// same moment, which nothing else in the suite does.
//
// What the mixed-width rounds are compared against is worth being exact about.
// A width is not allowed to change an ANSWER, but it does change the ROUTE: a
// contraction issued under a width the fence would clamp stays in PackedGemm's
// own tiled loops instead of deferring to a vendor GEMM, and reassociated sums
// move last bits. So the rounds are checked two ways - against the sequential
// reference at a tolerance far below the order-1 stale blocks this case exists
// for, and against each other bit for bit, which is the check that actually
// catches an intermittent defect in state the vendor shares between callers.
// The width-1 and OpenMPExecutor replays run the reference's own routes and are
// still compared exactly.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Extent of every dimension. Big enough that the contraction reaches the
/// packed-GEMM engine rather than a generic loop and that the permutes build
/// multi-threaded HPTT plans, which is the whole point: a smaller number tests
/// the same code path with none of the threading that breaks it.
constexpr size_t kExtent = 48;

/// Six index orders, three chains: enough concurrency for the executor to have
/// wide and narrow nodes running together.
constexpr size_t kPermutations = 6;
constexpr size_t kChains       = 3;

/// Replays of the mixed-width configuration. The corruption was intermittent -
/// a few rounds in every hundred - so one round proves nothing.
constexpr int kRounds = 200;

/// The machine the width budget rations, which is the thread count a plan has
/// to claim to be for or the executor discards it as stale.
unsigned machine_threads() {
    auto &budget = task_pool::WidthBudget::get_singleton();
    budget.sync_machine_width();
    return std::max(1U, budget.total());
}

/// One chain's buffers. `scratch` is overwritten once per permutation and read
/// once per permutation, so the chain is strictly serial within itself and
/// every chain runs concurrently with the others.
struct Chain {
    Tensor<double, 3>              scratch;
    Tensor<double, 3>              w;
    std::vector<Tensor<double, 3>> k;
    std::vector<Tensor<double, 2>> t;

    Chain(std::string const &tag, size_t n, size_t permutations)
        : scratch(create_zero_tensor<double>(tag + "_scratch", n, n, n)), w(create_zero_tensor<double>(tag + "_w", n, n, n)) {
        for (size_t p = 0; p < permutations; p++) {
            k.push_back(create_random_tensor<double>(tag + "_k" + std::to_string(p), n, n, n));
            t.push_back(create_random_tensor<double>(tag + "_t" + std::to_string(p), n, n));
        }
    }
};

/// Contract into scratch, then read scratch into the accumulator under one of
/// the six index orders. The spec strings are checked at compile time, so the
/// orders have to be literals rather than a table walked at run time.
void accumulate(size_t idx, Tensor<double, 3> *w, Tensor<double, 3> const &scratch) {
    double const c_prefactor = idx == 0 ? 0.0 : 1.0;
    switch (idx % 6) {
    case 0:
        cg::permute("abc <- abc", c_prefactor, w, 1.0, scratch);
        break;
    case 1:
        cg::permute("abc <- acb", c_prefactor, w, 1.0, scratch);
        break;
    case 2:
        cg::permute("abc <- bac", c_prefactor, w, 1.0, scratch);
        break;
    case 3:
        cg::permute("abc <- bca", c_prefactor, w, 1.0, scratch);
        break;
    case 4:
        cg::permute("abc <- cab", c_prefactor, w, 1.0, scratch);
        break;
    default:
        cg::permute("abc <- cba", c_prefactor, w, 1.0, scratch);
        break;
    }
}

void capture(cg::Graph &graph, std::vector<Chain> &chains) {
    cg::CaptureGuard const guard(graph);
    for (auto &chain : chains) {
        for (size_t idx = 0; idx < chain.k.size(); idx++) {
            // The packed-GEMM route: a vendor GEMM with no width in scope, the
            // engine's own tiled loops with one. Overwrites scratch, so it must
            // wait for the previous permutation's read of it.
            cg::einsum("abd;cd->abc", 0.0, &chain.scratch, 1.0, chain.k[idx], chain.t[idx]);
            accumulate(idx, &chain.w, chain.scratch);
        }
    }
}

/// Give each chain its own width. Nodes are captured chain by chain and every
/// chain is a strict serial run, so program order is already a topological
/// order and the sort leaves the blocks where they are; even if it did not, any
/// assignment that leaves different widths in flight together is the
/// configuration under test.
void impose_widths(cg::Graph &graph, std::vector<std::uint16_t> const &widths) {
    auto        &nodes           = graph.nodes();
    size_t const nodes_per_chain = std::max<size_t>(1, nodes.size() / kChains);
    for (size_t n = 0; n < nodes.size(); n++) {
        nodes[n].thread_width = widths[(n / nodes_per_chain) % widths.size()];
    }
}

/// What a replay differed from the reference by, if anything.
struct Mismatch {
    bool   any{false};
    size_t chain{0};
    size_t wrong{0};
    size_t total{0};
    double worst{0.0};
};

Mismatch compare(std::vector<Chain> const &reference, std::vector<Chain> const &trial, double tol) {
    Mismatch result;
    for (size_t c = 0; c < reference.size(); c++) {
        size_t const n     = reference[c].w.size();
        size_t       wrong = 0;
        double       worst = 0.0;
        for (size_t e = 0; e < n; e++) {
            double const a = reference[c].w.data()[e];
            double const b = trial[c].w.data()[e];
            // Negated form so a NaN anywhere registers as a mismatch.
            if (!(std::abs(a - b) <= tol * std::max(1.0, std::abs(a)))) {
                wrong++;
                worst = std::max(worst, std::abs(a - b));
            }
        }
        if (wrong > 0 && !result.any) {
            result = Mismatch{true, c, wrong, n, worst};
        }
    }
    return result;
}

void reset(std::vector<Chain> &chains) {
    for (auto &chain : chains) {
        std::memset(chain.w.data(), 0, chain.w.size() * sizeof(double));
        std::memset(chain.scratch.data(), 0, chain.scratch.size() * sizeof(double));
    }
}

/// A tolerance is owed once the width run is not running the same kernel as the
/// reference, which is every configuration where the width actually buys
/// something: PackedGemm keeps a contraction in its own tiled loops rather than
/// deferring it to a GEMM the fence would clamp, and on a vendor with per-caller
/// control (MKL) the fence is inert and the vendor GEMM legitimately runs wide.
/// Either way the sums are reassociated, which moves last bits and nothing more.
/// The order-1 stale-block corruption this case exists for is many orders above
/// this, and the round-to-round comparison below is bit-exact regardless, which
/// is the stronger check for an intermittent defect anyway.
constexpr double kWideTolerance = 1e-12;

/// Every element of every accumulator, in one flat vector - the bit-for-bit
/// snapshot the rounds are compared against.
std::vector<double> snapshot(std::vector<Chain> const &chains) {
    std::vector<double> flat;
    for (auto const &chain : chains) {
        flat.insert(flat.end(), chain.w.data(), chain.w.data() + chain.w.size());
    }
    return flat;
}

/// Copy the reference's inputs so both sides contract identical numbers.
void mirror_inputs(std::vector<Chain> const &from, std::vector<Chain> &to) {
    for (size_t c = 0; c < from.size(); c++) {
        for (size_t p = 0; p < from[c].k.size(); p++) {
            std::memcpy(to[c].k[p].data(), from[c].k[p].data(), from[c].k[p].size() * sizeof(double));
            std::memcpy(to[c].t[p].data(), from[c].t[p].data(), from[c].t[p].size() * sizeof(double));
        }
    }
}

} // namespace

TEST_CASE("MixedWidthVendorSafety - mixed widths never corrupt a vendor contraction", "[ComputeGraph][Executor][ThreadWidth]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("mixed widths need at least four threads to be mixed");
    }

    std::vector<Chain> reference;
    std::vector<Chain> trial;
    for (size_t c = 0; c < kChains; c++) {
        reference.emplace_back("ref" + std::to_string(c), kExtent, kPermutations);
        trial.emplace_back("try" + std::to_string(c), kExtent, kPermutations);
    }
    mirror_inputs(reference, trial);

    // The oracle: one node at a time, no widths anywhere.
    cg::Graph reference_graph("mixed_width_reference");
    capture(reference_graph, reference);
    cg::SequentialExecutor sequential;
    reference_graph.execute(sequential);

    cg::Graph trial_graph("mixed_width_trial");
    capture(trial_graph, trial);

    cg::DataflowExecutor dataflow;

    // Width 1 everywhere first: dependency-driven overlap with nothing for the
    // fence to do, which separates a width bug from an ordering bug.
    for (auto &node : trial_graph.nodes()) {
        node.thread_width = 1;
    }
    reset(trial);
    trial_graph.execute(dataflow);
    {
        // Exact: no WidthGuard anywhere, so every kernel is the one the
        // reference ran, on the same route.
        auto const bad = compare(reference, trial, 0.0);
        INFO(
            fmt::format("width 1: chain {} differs in {} of {} elements, worst |diff| {:.3e}", bad.chain, bad.wrong, bad.total, bad.worst));
        REQUIRE_FALSE(bad.any);
    }

    // The configuration that used to corrupt. Widths are clamped to the machine
    // because a width above it is a share of threads that do not exist, and the
    // graph is told which machine the plan is for or the executor discards it as
    // stale and runs everything at width 1.
    auto const                       wide = static_cast<std::uint16_t>(std::min<unsigned>(4, p));
    auto const                       thin = static_cast<std::uint16_t>(std::min<unsigned>(2, p));
    std::vector<std::uint16_t> const widths{wide, wide, thin};
    impose_widths(trial_graph, widths);
    trial_graph.set_planned_thread_count(p);
    cg::reset_stale_thread_plan_fallbacks();

    std::vector<double> first_round;
    for (int round = 0; round < kRounds; round++) {
        reset(trial);
        trial_graph.execute(dataflow);
        auto const bad = compare(reference, trial, kWideTolerance);
        if (bad.any) {
            FAIL(fmt::format("round {} at widths {}/{}/{}: chain {} differs from the sequential reference in {} of {} elements, "
                             "worst |diff| {:.3e}",
                             round, wide, wide, thin, bad.chain, bad.wrong, bad.total, bad.worst));
        }
        // The corruption was intermittent and lived in state the vendor shared
        // between concurrent callers, so the sharpest instrument is the mixed
        // width configuration against ITSELF: the routes are fixed, no kernel in
        // this graph reassociates with the thread count, and a single differing
        // bit between two rounds is a caller reading what another one wrote.
        auto const now = snapshot(trial);
        if (round == 0) {
            first_round = now;
        } else if (now != first_round) {
            size_t at = 0;
            while (at < now.size() && now[at] == first_round[at]) {
                at++;
            }
            FAIL(fmt::format("round {} at widths {}/{}/{} differs from round 0 at flat element {} ({} against {}); the same graph at "
                             "the same widths is not replaying deterministically",
                             round, wide, wide, thin, at, now[at], first_round[at]));
        }
    }

    // The widths that ran are the mixed ones this case imposed. Without both
    // checks the case is green for the wrong reason: a rejected plan or a
    // re-plan back to width 1 would have run the whole loop in the one
    // configuration that was never in question.
    REQUIRE(cg::stale_thread_plan_fallbacks() == 0);
    REQUIRE(wide != thin);
    size_t wide_nodes = 0;
    size_t thin_nodes = 0;
    for (auto const &node : trial_graph.nodes()) {
        wide_nodes += node.thread_width == wide ? 1 : 0;
        thin_nodes += node.thread_width == thin ? 1 : 0;
    }
    REQUIRE(wide_nodes > 0);
    REQUIRE(thin_nodes > 0);

    // The executor that ignores widths entirely, for cross-executor parity: the
    // same graph, still carrying the widths, must land on the same bits.
    cg::OpenMPExecutor openmp;
    reset(trial);
    trial_graph.execute(openmp);
    {
        auto const bad = compare(reference, trial, 0.0);
        INFO(fmt::format("openmp executor: chain {} differs in {} of {} elements, worst |diff| {:.3e}", bad.chain, bad.wrong, bad.total,
                         bad.worst));
        REQUIRE_FALSE(bad.any);
    }
}
