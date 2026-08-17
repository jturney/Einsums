//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One constraint, over a random corpus: a thread width changes how much of the
// machine a node gets and nothing else. The bits a graph produces must not
// depend on the widths its nodes carry.
//
// The suite's differential fuzzers all compare a captured graph against an
// eager or numpy oracle, and none of them can put a width on a node - the
// python surface exposes plan_threads() but neither Node::thread_width nor the
// node list, so the moldable replay path has no fuzz coverage from that side.
// This case supplies it in C++: random small graphs built from the kernel
// families that carry widths (vendor GEMM through the packed route, HPTT
// permutes, elementwise scale/axpby, rank-3 contractions), each replayed three
// ways and compared bit for bit.
//
// The three replays separate the two ways a width could show up in an answer:
//
//   * SequentialExecutor, no widths anywhere. The oracle. Its kernels run on
//     the calling thread at the machine's own thread ceiling.
//   * DataflowExecutor with every width 1. Same dependency-driven overlap as
//     the width run, but no WidthGuard anywhere. A difference here is a
//     thread-count or ordering effect that has nothing to do with widths, and
//     the failure says so.
//   * DataflowExecutor with drawn widths, mixed across the graph. A difference
//     here that the width-1 replay did not show is a width-dependent result,
//     which is the bug this case exists to find.
//
// The width-1 replay is compared to the reference exactly: it runs the
// reference's own kernels on the reference's own routes, so a single differing
// bit is a finding. The wide replay is not owed that, because a width changes
// which kernel runs - PackedGemm keeps a contraction in its own tiled loops
// rather than deferring to a vendor GEMM the fence would clamp, and on MKL the
// vendor GEMM itself goes wide - and reassociated sums move last bits. It is
// compared to the reference at a few hundred epsilon, which is orders of
// magnitude below the whole wrong blocks the vendor defect produced, and to its
// own earlier rounds bit for bit, which is what an intermittent defect in state
// shared between callers actually shows up in.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Graphs drawn. Each is a fresh shape, program, dtype and width assignment.
constexpr int kGraphs = 120;

/// Replays of each graph's width assignment. The vendor defect that motivated
/// this case showed up in a few rounds per hundred, so a single replay of a
/// single assignment proves very little; breadth across assignments is the
/// budget better spent here, with MixedWidthVendorSafety carrying the depth on
/// the one shape known to break.
constexpr int kRounds = 4;

/// Seeds are `kBaseSeed + graph index`, so a failure names a seed that
/// regenerates exactly that graph on its own.
constexpr std::uint32_t kBaseSeed = 0x51D4E110U;

/// Pool sizes. Large enough that a step can always find an output distinct
/// from its inputs, small enough that a program of a dozen steps reuses
/// buffers heavily - shared read operands and write-after-read chains are what
/// give the executor something to overlap.
constexpr size_t kMats  = 5;
constexpr size_t kCubes = 4;
constexpr size_t kFacs  = 3;

/// The machine the width budget rations. A plan has to claim this number or
/// the executor discards it as stale and runs everything at width 1.
unsigned machine_threads() {
    auto &budget = task_pool::WidthBudget::get_singleton();
    budget.sync_machine_width();
    return std::max(1U, budget.total());
}

/// Every buffer a program can touch. Matrices and cubes carry the rank-2 and
/// rank-3 work; `facs` are the rank-2 factors of a rank-3 contraction and the
/// targets of a rank-3 reduction, so they are sized to the cube extent rather
/// than the matrix one.
template <typename T>
struct World {
    std::vector<Tensor<T, 2>> mats;
    std::vector<Tensor<T, 3>> cubes;
    std::vector<Tensor<T, 2>> facs;

    World(std::string const &tag, size_t extent2, size_t extent3) {
        for (size_t i = 0; i < kMats; i++) {
            mats.push_back(create_random_tensor<T>(tag + "_m" + std::to_string(i), extent2, extent2));
        }
        for (size_t i = 0; i < kCubes; i++) {
            cubes.push_back(create_random_tensor<T>(tag + "_c" + std::to_string(i), extent3, extent3, extent3));
        }
        for (size_t i = 0; i < kFacs; i++) {
            facs.push_back(create_random_tensor<T>(tag + "_f" + std::to_string(i), extent3, extent3));
        }
    }
};

/// Flat view of a world's storage, in a fixed order, for snapshot, restore and
/// comparison. Two worlds built with the same extents produce the same layout.
template <typename T>
std::vector<std::pair<T *, size_t>> buffers(World<T> &world) {
    std::vector<std::pair<T *, size_t>> out;
    for (auto &t : world.mats) {
        out.emplace_back(t.data(), t.size());
    }
    for (auto &t : world.cubes) {
        out.emplace_back(t.data(), t.size());
    }
    for (auto &t : world.facs) {
        out.emplace_back(t.data(), t.size());
    }
    return out;
}

template <typename T>
std::vector<std::vector<T>> snapshot(World<T> &world) {
    std::vector<std::vector<T>> saved;
    for (auto const &[ptr, count] : buffers(world)) {
        saved.emplace_back(ptr, ptr + count);
    }
    return saved;
}

/// Puts a world back to the drawn inputs. Every replay starts from these, not
/// from whatever the last one left behind: a step may accumulate into a buffer
/// it did not write first, so the starting contents are part of the program.
template <typename T>
void restore(World<T> &world, std::vector<std::vector<T>> const &saved) {
    auto const slots = buffers(world);
    for (size_t b = 0; b < slots.size(); b++) {
        std::copy(saved[b].begin(), saved[b].end(), slots[b].first);
    }
}

/// Where two worlds first differ, if they do.
struct Mismatch {
    bool   any{false};
    size_t buffer{0};
    size_t element{0};
    size_t wrong{0};
    double reference{0.0};
    double trial{0.0};
};

/// The tolerance a replay that ran a node wide is compared at. A width does not
/// change what a graph computes, but it does change which kernel computes it:
/// PackedGemm keeps a contraction in its own tiled loops rather than deferring
/// to a vendor GEMM the fence would clamp, and on MKL the fence is inert and the
/// vendor GEMM itself runs wide. Both reassociate sums. A few hundred epsilon in
/// the dtype's own precision covers that and stays orders of magnitude below any
/// real corruption, which arrives as whole wrong blocks.
template <typename T>
constexpr double wide_tolerance() {
    return 500.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// What separates two replays that ran the same kernels on different threads.
///
/// The sequential reference runs on the calling thread and the dataflow trial
/// runs on pool workers. A threaded vendor BLAS is free to answer those two
/// with different internal thread counts, and a dot product summed in two
/// orders lands a unit or so apart in the last place. That is not the failure
/// this case looks for - an ordering bug moves an element, not its last bit -
/// so the bound is a handful of epsilon rather than zero.
template <typename T>
constexpr double ordering_tolerance() {
    return 8.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <typename T>
Mismatch compare(World<T> &reference, World<T> &trial, double tol) {
    // The negated form makes a NaN register as bad.
    Mismatch   result;
    auto const lhs = buffers(reference);
    auto const rhs = buffers(trial);
    for (size_t b = 0; b < lhs.size(); b++) {
        for (size_t e = 0; e < lhs[b].second; e++) {
            T const a = lhs[b].first[e];
            T const c = rhs[b].first[e];
            if (!(std::abs(static_cast<double>(a) - static_cast<double>(c)) <= tol * std::max(1.0, std::abs(static_cast<double>(a))))) {
                if (!result.any) {
                    result = Mismatch{true, b, e, 0, static_cast<double>(a), static_cast<double>(c)};
                }
                result.wrong++;
            }
        }
    }
    return result;
}

/// Whether a replay wrote anything at all. Two worlds that were never touched
/// compare equal, so a replay that silently executed nothing would pass every
/// check below without testing a thing.
template <typename T>
bool differs_from(World<T> &world, std::vector<std::vector<T>> const &saved) {
    auto const slots = buffers(world);
    for (size_t b = 0; b < slots.size(); b++) {
        if (!std::equal(saved[b].begin(), saved[b].end(), slots[b].first)) {
            return true;
        }
    }
    return false;
}

/// A reference value that is not finite means the generator overflowed rather
/// than the executor misbehaved, and an exact comparison of NaN against NaN
/// would then fail for a reason that has nothing to do with widths.
template <typename T>
bool all_finite(World<T> &world) {
    for (auto const &[ptr, count] : buffers(world)) {
        for (size_t e = 0; e < count; e++) {
            if (!std::isfinite(static_cast<double>(ptr[e]))) {
                return false;
            }
        }
    }
    return true;
}

/// The kernel families a width can reach. Each maps to one capture call; the
/// pools are chosen so that no step can produce an output that overlaps one of
/// its inputs, which capture rejects outright.
enum class Kind {
    Gemm,       ///< vendor GEMM through the packed route
    GemmT,      ///< same, with A transposed
    Hadamard,   ///< elementwise contraction, generic route
    Permute2,   ///< HPTT, rank 2
    Scale2,     ///< elementwise, in place
    Axpby2,     ///< elementwise, two operands
    Contract3,  ///< rank-3 by rank-2, contracting the last index
    Contract3B, ///< rank-3 by rank-2, contracting the middle index
    Reduce3,    ///< rank-3 by rank-3 down to rank 2
    Permute3,   ///< HPTT, rank 3, six index orders
    Scale3,
    Axpby3,
};

constexpr Kind kKinds[] = {Kind::Gemm,      Kind::GemmT,      Kind::Hadamard, Kind::Permute2, Kind::Scale2, Kind::Axpby2,
                           Kind::Contract3, Kind::Contract3B, Kind::Reduce3,  Kind::Permute3, Kind::Scale3, Kind::Axpby3};

/// Specs are walked by the generator, so they are built from `string_view` at
/// run time rather than from literals. Dispatch does not depend on which
/// constructor produced them: the consteval form only pre-computes the operand
/// rank check, which the runtime form performs against the parsed spec instead.
constexpr char const *kPermute3Specs[] = {"abc <- abc", "abc <- acb", "abc <- bac", "abc <- bca", "abc <- cab", "abc <- cba"};
constexpr char const *kPermute2Specs[] = {"ij <- ij", "ji <- ij"};

/// One captured operation. Indices address the pool that the kind implies.
struct Step {
    Kind   kind{Kind::Gemm};
    size_t out{0};
    size_t lhs{0};
    size_t rhs{0};
    size_t spec{0};
    double alpha{1.0};
    double beta{0.0};
};

/// Prefactors drawn from exactly representable values, including the 0 that
/// makes an accumulation an overwrite and the 1 that makes it a pure add.
constexpr double kPrefactors[] = {0.0, 0.25, 0.5, 1.0, -1.0, 2.0};

size_t draw_index(std::mt19937 &rng, size_t count) {
    return std::uniform_int_distribution<size_t>(0, count - 1)(rng);
}

/// An index into a pool that is not `avoid`, so a step never writes a buffer
/// it also reads through a different index list.
size_t draw_index_other(std::mt19937 &rng, size_t count, size_t avoid) {
    size_t const drawn = std::uniform_int_distribution<size_t>(0, count - 2)(rng);
    return drawn >= avoid ? drawn + 1 : drawn;
}

double draw_prefactor(std::mt19937 &rng) {
    return kPrefactors[draw_index(rng, std::size(kPrefactors))];
}

/// A random straight-line program. Nothing constrains the shape beyond operand
/// validity: whatever dependencies fall out of the pool reuse are what the
/// executor gets to overlap, and a program that happens to serialize is as
/// valid a case as one that fans out.
std::vector<Step> draw_program(std::mt19937 &rng, size_t steps, size_t extent2, size_t extent3) {
    std::vector<Step> program;
    program.reserve(steps);
    for (size_t s = 0; s < steps; s++) {
        Step step;
        step.kind  = kKinds[draw_index(rng, std::size(kKinds))];
        step.beta  = draw_prefactor(rng);
        step.alpha = draw_prefactor(rng);

        switch (step.kind) {
        case Kind::Gemm:
        case Kind::GemmT:
            // A contraction multiplies magnitudes by its summed extent, and a
            // dozen of them chained would overflow a float. Damping the
            // prefactor by the extent keeps every value order 1 without
            // constraining anything the test is about.
            step.alpha /= static_cast<double>(extent2);
            [[fallthrough]];
        case Kind::Hadamard:
        case Kind::Axpby2:
            step.out = draw_index(rng, kMats);
            step.lhs = draw_index_other(rng, kMats, step.out);
            step.rhs = draw_index_other(rng, kMats, step.out);
            break;
        case Kind::Permute2:
            step.out  = draw_index(rng, kMats);
            step.lhs  = draw_index_other(rng, kMats, step.out);
            step.spec = draw_index(rng, std::size(kPermute2Specs));
            break;
        case Kind::Scale2:
            step.out = draw_index(rng, kMats);
            break;
        case Kind::Contract3:
        case Kind::Contract3B:
            step.alpha /= static_cast<double>(extent3);
            step.out = draw_index(rng, kCubes);
            step.lhs = draw_index_other(rng, kCubes, step.out);
            step.rhs = draw_index(rng, kFacs);
            break;
        case Kind::Reduce3:
            step.alpha /= static_cast<double>(extent3 * extent3);
            step.out = draw_index(rng, kFacs);
            step.lhs = draw_index(rng, kCubes);
            step.rhs = draw_index(rng, kCubes);
            break;
        case Kind::Permute3:
            step.out  = draw_index(rng, kCubes);
            step.lhs  = draw_index_other(rng, kCubes, step.out);
            step.spec = draw_index(rng, std::size(kPermute3Specs));
            break;
        case Kind::Scale3:
            step.out = draw_index(rng, kCubes);
            break;
        case Kind::Axpby3:
            step.out = draw_index(rng, kCubes);
            step.lhs = draw_index_other(rng, kCubes, step.out);
            break;
        }
        program.push_back(step);
    }

    // A terminating overwrite, so every program provably changes its world and
    // the comparison against the reference is never vacuous. The draws above
    // could in principle produce a program whose every step is a scale by 1.
    Step terminal;
    terminal.kind  = Kind::Gemm;
    terminal.out   = draw_index(rng, kMats);
    terminal.lhs   = draw_index_other(rng, kMats, terminal.out);
    terminal.rhs   = draw_index_other(rng, kMats, terminal.out);
    terminal.alpha = 1.0 / static_cast<double>(extent2);
    terminal.beta  = 0.0;
    program.push_back(terminal);

    return program;
}

template <typename T>
void apply(Step const &step, World<T> &world) {
    auto const alpha = static_cast<T>(step.alpha);
    auto const beta  = static_cast<T>(step.beta);

    switch (step.kind) {
    case Kind::Gemm:
        cg::einsum(cg::EinsumFormatString(std::string_view{"ik;kj->ij"}), beta, &world.mats[step.out], alpha, world.mats[step.lhs],
                   world.mats[step.rhs]);
        break;
    case Kind::GemmT:
        cg::einsum(cg::EinsumFormatString(std::string_view{"ki;kj->ij"}), beta, &world.mats[step.out], alpha, world.mats[step.lhs],
                   world.mats[step.rhs]);
        break;
    case Kind::Hadamard:
        cg::einsum(cg::EinsumFormatString(std::string_view{"ij;ij->ij"}), beta, &world.mats[step.out], alpha, world.mats[step.lhs],
                   world.mats[step.rhs]);
        break;
    case Kind::Permute2:
        cg::permute(cg::PermuteFormatString(std::string_view{kPermute2Specs[step.spec]}), beta, &world.mats[step.out], alpha,
                    world.mats[step.lhs]);
        break;
    case Kind::Scale2:
        cg::scale(alpha, &world.mats[step.out]);
        break;
    case Kind::Axpby2:
        cg::axpby(alpha, world.mats[step.lhs], beta, &world.mats[step.out]);
        break;
    case Kind::Contract3:
        cg::einsum(cg::EinsumFormatString(std::string_view{"abd;cd->abc"}), beta, &world.cubes[step.out], alpha, world.cubes[step.lhs],
                   world.facs[step.rhs]);
        break;
    case Kind::Contract3B:
        cg::einsum(cg::EinsumFormatString(std::string_view{"abc;cd->abd"}), beta, &world.cubes[step.out], alpha, world.cubes[step.lhs],
                   world.facs[step.rhs]);
        break;
    case Kind::Reduce3:
        cg::einsum(cg::EinsumFormatString(std::string_view{"abc;abd->cd"}), beta, &world.facs[step.out], alpha, world.cubes[step.lhs],
                   world.cubes[step.rhs]);
        break;
    case Kind::Permute3:
        cg::permute(cg::PermuteFormatString(std::string_view{kPermute3Specs[step.spec]}), beta, &world.cubes[step.out], alpha,
                    world.cubes[step.lhs]);
        break;
    case Kind::Scale3:
        cg::scale(alpha, &world.cubes[step.out]);
        break;
    case Kind::Axpby3:
        cg::axpby(alpha, world.cubes[step.lhs], beta, &world.cubes[step.out]);
        break;
    }
}

template <typename T>
void capture(cg::Graph &graph, std::vector<Step> const &program, World<T> &world) {
    cg::CaptureGuard const guard(graph);
    for (auto const &step : program) {
        apply(step, world);
    }
}

/// How a graph's widths are drawn. Uniform is the general case; the other
/// three aim at boundaries the uniform draw reaches only rarely - every node
/// at one extreme or the other, every node claiming the whole machine at once
/// so the budget has to park most of them, and the rung set a real plan uses.
enum class WidthMode { Uniform, Extremes, AllWide, Rungs };

constexpr WidthMode kWidthModes[] = {WidthMode::Uniform, WidthMode::Extremes, WidthMode::AllWide, WidthMode::Rungs};

std::uint16_t draw_width(std::mt19937 &rng, WidthMode mode, unsigned machine) {
    switch (mode) {
    case WidthMode::Uniform:
        return static_cast<std::uint16_t>(std::uniform_int_distribution<unsigned>(1, machine)(rng));
    case WidthMode::Extremes:
        return static_cast<std::uint16_t>(std::uniform_int_distribution<unsigned>(0, 1)(rng) == 0 ? 1 : machine);
    case WidthMode::AllWide:
        return static_cast<std::uint16_t>(machine);
    case WidthMode::Rungs:
    default: {
        std::vector<unsigned> rungs;
        for (unsigned w = 1; w <= machine; w *= 2) {
            rungs.push_back(w);
        }
        if (rungs.back() != machine) {
            rungs.push_back(machine);
        }
        return static_cast<std::uint16_t>(rungs[draw_index(rng, rungs.size())]);
    }
    }
}

/// What a failure needs to name the case beyond its seed. Widths are omitted
/// before they are drawn rather than printed as an empty list.
std::string describe(std::vector<Step> const &program, std::vector<std::uint16_t> const &widths) {
    if (widths.empty()) {
        return fmt::format("{} steps", program.size());
    }
    return fmt::format("{} steps, widths [{}]", program.size(), fmt::join(widths, ","));
}

/// Draws one graph, replays it three ways and returns how many nodes carried a
/// width above 1, so the caller can prove the corpus tested what it claims to.
template <typename T>
size_t run_one(std::mt19937 &rng, std::uint32_t seed, unsigned machine) {
    // Rank 2 is cheap enough to run wide; rank 3 costs the cube of its extent
    // in storage and the fourth power in work, so it gets its own smaller
    // range to keep the whole corpus inside a few seconds.
    size_t const extent2 = std::uniform_int_distribution<size_t>(8, 48)(rng);
    size_t const extent3 = std::uniform_int_distribution<size_t>(8, 24)(rng);
    size_t const steps   = std::uniform_int_distribution<size_t>(4, 12)(rng);

    auto const program = draw_program(rng, steps, extent2, extent3);

    World<T>   reference(fmt::format("mwd_ref_{}", seed), extent2, extent3);
    World<T>   trial(fmt::format("mwd_try_{}", seed), extent2, extent3);
    auto const initial = snapshot(reference);
    restore(trial, initial);

    cg::Graph reference_graph(fmt::format("mixed_width_diff_ref_{}", seed));
    capture(reference_graph, program, reference);
    cg::SequentialExecutor sequential;
    reference_graph.execute(sequential);

    if (!differs_from(reference, initial)) {
        FAIL(fmt::format("seed {}: the sequential replay wrote nothing, so nothing below compares anything", seed));
    }
    if (!all_finite(reference)) {
        FAIL(fmt::format("seed {}: the drawn program overflowed, so the reference is not comparable ({})", seed, describe(program, {})));
    }

    cg::Graph trial_graph(fmt::format("mixed_width_diff_try_{}", seed));
    capture(trial_graph, program, trial);
    cg::DataflowExecutor dataflow;

    // Width 1 everywhere: the same concurrency with no WidthGuard anywhere, so
    // a difference here is not about widths at all.
    for (auto &node : trial_graph.nodes()) {
        node.thread_width = 1;
    }
    restore(trial, initial);
    trial_graph.execute(dataflow);
    {
        auto const bad = compare(reference, trial, ordering_tolerance<T>());
        if (bad.any) {
            FAIL(fmt::format("seed {}: dataflow at width 1 differs from the sequential reference in {} elements, first at buffer {} "
                             "element {} ({} against {}); this is an ordering or thread-count difference, not a width one. {}",
                             seed, bad.wrong, bad.buffer, bad.element, bad.reference, bad.trial, describe(program, {})));
        }
    }

    WidthMode const            mode = kWidthModes[draw_index(rng, std::size(kWidthModes))];
    std::vector<std::uint16_t> widths;
    size_t                     wide = 0;
    for (auto &node : trial_graph.nodes()) {
        std::uint16_t const w = draw_width(rng, mode, machine);
        node.thread_width     = w;
        widths.push_back(w);
        wide += w > 1 ? 1 : 0;
    }
    trial_graph.set_planned_thread_count(machine);

    std::vector<std::vector<T>> first_round;
    for (int round = 0; round < kRounds; round++) {
        restore(trial, initial);
        trial_graph.execute(dataflow);
        auto const bad = compare(reference, trial, wide == 0 ? ordering_tolerance<T>() : wide_tolerance<T>());
        if (bad.any) {
            FAIL(fmt::format("seed {}, round {}: dataflow at drawn widths differs from the sequential reference in {} elements, first "
                             "at buffer {} element {} ({} against {}). A width changed a result. {}",
                             seed, round, bad.wrong, bad.buffer, bad.element, bad.reference, bad.trial, describe(program, widths)));
        }
        // The one thing a width may not move at all is the answer from one
        // replay of ITSELF to the next. The routes are fixed by the width
        // assignment and none of these kernels reassociates with the thread
        // count, so any difference between rounds is nondeterminism - which is
        // the shape the vendor defect this case guards took.
        auto const now = snapshot(trial);
        if (round == 0) {
            first_round = now;
        } else if (now != first_round) {
            FAIL(fmt::format("seed {}, round {}: dataflow at drawn widths differs from its own round 0. The same graph at the same "
                             "widths is not replaying deterministically. {}",
                             seed, round, describe(program, widths)));
        }
    }

    return wide;
}

} // namespace

TEST_CASE("MixedWidthDifferential - drawn graphs replay identically at any width", "[ComputeGraph][Executor][ThreadWidth]") {
    unsigned const machine = machine_threads();
    if (machine < 2) {
        SKIP("a width can only differ from 1 on a machine with more than one thread");
    }

    cg::reset_stale_thread_plan_fallbacks();

    size_t wide_nodes = 0;
    for (int g = 0; g < kGraphs; g++) {
        auto const   seed = static_cast<std::uint32_t>(kBaseSeed + g);
        std::mt19937 rng(seed);
        // Every third graph in single precision. The vendor route, the HPTT
        // plans and the packing all divide their work by element size, so a
        // float64-only corpus would leave half of each untested.
        if (g % 3 == 1) {
            wide_nodes += run_one<float>(rng, seed, machine);
        } else {
            wide_nodes += run_one<double>(rng, seed, machine);
        }
    }

    // Without these the case is green for the wrong reason: a corpus that drew
    // width 1 everywhere, or a plan the executor threw away as stale, would
    // have run every replay in the one configuration that was never in doubt.
    REQUIRE(wide_nodes > 0);
    REQUIRE(cg::stale_thread_plan_fallbacks() == 0);
}
