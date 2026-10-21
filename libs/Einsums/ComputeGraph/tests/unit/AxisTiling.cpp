//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AxisTiling.cpp
/// @brief What AxisTiling decides, on the shape it was designed against.
///
/// The program below is the density-fitted MP2 correlation energy written over all four of its
/// orbital indices, at extents small enough to run in a unit test. It is the one shape where
/// every part of the decision has a knowable right answer: the occupied axes are the ones to
/// slice, the virtual ones are not, and the reason the virtual ones are not is structural
/// rather than a matter of their size.
///
/// Each case asserts the MECHANISM. "The pass fired" would pass on a decision that sliced the
/// wrong axes at the wrong depth, so what is asserted is which axes, at what extents, into how
/// many slices, in chunks of what, and what the largest intermediate measures on both sides of
/// the rewrite.

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph.hpp>
// Not reached through the umbrella header, which does not carry this pass.
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t nocc = 4;
constexpr std::size_t nvir = 6;
constexpr std::size_t naux = 7;

/// Bytes one rank-four tensor over the four orbital indices costs.
constexpr std::size_t whole_bytes = nocc * nvir * nocc * nvir * sizeof(double);
/// Bytes one occupied PAIR of that tensor costs.
constexpr std::size_t pair_bytes = nvir * nvir * sizeof(double);
/// Bytes one occupied ROW of it costs, which is what slicing a single axis leaves.
constexpr std::size_t row_bytes = nocc * nvir * nvir * sizeof(double);

/// A reproducible filling, so two arms of one comparison see identical inputs.
RuntimeTensor<double> filled(std::string name, std::vector<std::size_t> dims, unsigned seed) {
    RuntimeTensor<double> out(std::move(name), std::move(dims));
    for (std::size_t i = 0; i < out.size(); ++i) {
        seed          = (seed * 1103515245U) + 12345U;
        out.data()[i] = (static_cast<double>((seed >> 8U) % 2048U) / 1024.0) - 1.0;
    }
    return out;
}

/// The buffers the caller owns, kept alive for as long as the graph that reads them.
///
/// Rank-erased, because the emitted body names its operands' static types and a rank-erased
/// operand is the one it can name. A statically ranked capture is declined with that reason.
struct Problem {
    RuntimeTensor<double> fitted{filled("B", {naux, nocc, nvir}, 7U)};
    RuntimeTensor<double> denominator{filled("D", {nocc, nvir, nocc, nvir}, 19U)};
    RuntimeTensor<double> energy{"E", std::vector<std::size_t>{1}};
};

/// E = sum_iajb (2 K[i,a,j,b] - K[i,b,j,a]) K[i,a,j,b] D[i,a,j,b], with K = sum_Q B B.
///
/// The integral is formed twice on purpose, and here the duplicate is load-bearing rather than
/// inherited: two contractions are what give the emitted chunk two grouped families, and the
/// exchange permutation reads the copy rather than the tensor the amplitude reads, which is what
/// the occupied-exchange case below is about. Everything here is a graph-owned intermediate over
/// all four indices, which is what makes the program the one that needs tiling.
void capture(cg::Graph &graph, Problem &problem) {
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &again       = graph.scratch_runtime<double>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<double>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, problem.fitted, problem.fitted);
    cg::permute("i,a,j,b <- i,b,j,a", &exchange, again);
    cg::axpby(2.0, again, 0.0, &combination);
    cg::axpby(-1.0, exchange, 1.0, &combination);
    // The rank-one destination is the graph-native scalar handle; a bare double would register
    // a rank-zero slot the accumulation has no axpby to write through.
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// The same energy with the exchange term dropped, so nothing permutes the virtual axes.
///
/// Without that permutation BOTH pairs of axes carry a slice labelling, and which one the pass
/// takes is decided by traffic alone. That is the half of the decision the four-index program
/// above cannot exercise, because there the virtual pair is rejected before any size is looked
/// at.
void capture_without_exchange(cg::Graph &graph, Problem &problem) {
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::axpby(2.0, integral, 0.0, &combination);
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// The four-index energy whose exchange term permutes the OCCUPIED pair rather than the virtual
/// one.
///
/// Not a physical program: it is the shape the symmetry-partner sentence of the design is about.
/// At a fixed pair the body holds the ``(i,j)`` slab, and this exchange wants the ``(j,i)`` one,
/// which the body is not at. So the occupied pair stops being a candidate and the pass says why.
void capture_with_occupied_exchange(cg::Graph &graph, Problem &problem) {
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &again       = graph.scratch_runtime<double>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<double>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, problem.fitted, problem.fitted);
    cg::permute("i,a,j,b <- j,a,i,b", &exchange, again);
    cg::axpby(2.0, again, 0.0, &combination);
    cg::axpby(-1.0, exchange, 1.0, &combination);
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// The buffers of a program whose contraction is batched over the occupied pair: every operand
/// carries both occupied indices, so no operand is shared between pairs and a block of pairs
/// reads nothing less often than a grouped chunk does.
struct BatchedProblem {
    RuntimeTensor<double> left{filled("X", {nvir, nvir, nocc, nocc}, 3U)};
    RuntimeTensor<double> right{filled("Y", {nvir, nvir, nocc, nocc}, 5U)};
    RuntimeTensor<double> denominator{filled("D", {nvir, nvir, nocc, nocc}, 19U)};
    RuntimeTensor<double> energy{"E", std::vector<std::size_t>{1}};
};

/// E = sum (2 K - K^T) K D over every pair, with K[a,b,i,j] = sum_c X[a,c,i,j] Y[c,b,i,j].
///
/// The occupied indices are the LAST two, so a pair of every operand is a column-major matrix
/// a vendor GEMM can read and the grouped chunk takes the batched-GEMM family. The exchange
/// permutes a with b, which keeps the virtual axes out of the decision.
void capture_pair_batched(cg::Graph &graph, BatchedProblem &problem) {
    std::vector<std::size_t> const shape{nvir, nvir, nocc, nocc};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &again       = graph.scratch_runtime<double>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<double>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("a,b,i,j <- a,c,i,j ; c,b,i,j", &integral, problem.left, problem.right);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::einsum("a,b,i,j <- a,c,i,j ; c,b,i,j", &again, problem.left, problem.right);
    cg::permute("a,b,i,j <- b,a,i,j", &exchange, again);
    cg::axpby(2.0, again, 0.0, &combination);
    cg::axpby(-1.0, exchange, 1.0, &combination);
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// An energy computed by plain loops, with the sum of its terms' magnitudes, which bounds what a
/// reordering of the sum can move it by.
template <typename T>
struct Expected {
    T      value{0};
    double magnitude{0.0};
};

/// The batched program's energy by plain loops over the reference contraction.
Expected<double> batched_reference(BatchedProblem &problem) {
    RuntimeTensor<double> integral("Kref", std::vector<std::size_t>{nvir, nvir, nocc, nocc});
    integral.zero();
    testing::reference_einsum("a,b,i,j <- a,c,i,j ; c,b,i,j", &integral, problem.left, problem.right);
    Expected<double> out;
    for (std::size_t a = 0; a < nvir; ++a) {
        for (std::size_t b = 0; b < nvir; ++b) {
            for (std::size_t i = 0; i < nocc; ++i) {
                for (std::size_t j = 0; j < nocc; ++j) {
                    double const k    = integral(a, b, i, j);
                    double const term = ((2.0 * k) - integral(b, a, i, j)) * k * problem.denominator(a, b, i, j);
                    out.value += term;
                    out.magnitude += std::abs(term);
                }
            }
        }
    }
    return out;
}

/// Run the pass alone at @p cap bytes and hand back what it decided.
std::shared_ptr<cg::passes::AxisTiling> decide(cg::Graph &graph, std::int64_t cap) {
    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(cap);
    cg::apply_single_pass(*tiling, graph);
    return tiling;
}

/// The reasons the pass reported, as a set of prefixes a case can look for.
bool declined_because(cg::passes::AxisTiling const &tiling, std::string_view fragment) {
    for (auto const &[reason, count] : tiling.skip_reasons()) {
        if (reason.find(fragment) != std::string::npos) {
            return count > 0;
        }
    }
    return false;
}

} // namespace

TEST_CASE("AxisTiling picks the occupied axes of the four-index energy", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // A cap between one occupied pair and one occupied row: the pair fits, the row does not, so
    // the only feasible depth is one and the only feasible set is both occupied axes.
    REQUIRE(pair_bytes < 400);
    REQUIRE(row_bytes > 400);
    auto const tiling = decide(graph, 400);

    // The axes, by the letter the contraction that writes the seed names them with. i and j are
    // the occupied ones; a and b are rejected because the exchange permutation exchanges them,
    // so a slice of the permuted tensor would need a slice of its source at a different pair.
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->axis_names() == std::vector<std::string>{"K[0]", "K[2]"});
    CHECK(tiling->axis_extents() == std::vector<std::int64_t>{nocc, nocc});

    CHECK(tiling->slice_count() == nocc * nocc);
    CHECK(tiling->depth() == 1);
    CHECK(tiling->iterations() == nocc * nocc);

    // The water number in miniature: o^2 v^2 becomes v^2.
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(tiling->largest_after() == pair_bytes);

    // Every four-index intermediate streams, and the three-index integral streams too, because
    // one of its axes is an occupied one. Nothing is left whole but the energy.
    std::vector<std::string> const streamed = tiling->streamed();
    for (auto const &name : {"K", "T", "K_again", "K_exchange", "Kbar", "B", "D"}) {
        INFO("expected '" << name << "' among the streamed tensors");
        CHECK(std::ranges::find(streamed, std::string{name}) != streamed.end());
    }
    CHECK(tiling->whole() == std::vector<std::string>{"E"});
    CHECK(tiling->accumulator() == "E");
}

TEST_CASE("AxisTiling declines a program whose largest intermediate already fits", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    auto const tiling = decide(graph, static_cast<std::int64_t>(whole_bytes));
    CHECK(tiling->slice_count() == 0);
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(declined_because(*tiling, "already fits the cap"));
}

TEST_CASE("AxisTiling holds a declared result whole and says how to let it stream", "[ComputeGraph][Pass][AxisTiling]") {
    // A declared tensor is a result unless it is declared an intermediate: the caller may read it
    // after execute(), so it has to exist whole. The pass used to leave such a tensor whole without
    // a word, and a program over its memory budget gave no sign of why nothing was streamed.
    auto const a = filled("A", {6, 5}, 3U);
    auto const b = filled("B", {7, 5}, 5U);
    auto const d = filled("D", {7, 5, 3}, 7U);
    for (bool const intermediate : {false, true}) {
        CAPTURE(intermediate);
        RuntimeTensor<double> e("E", std::vector<std::size_t>{6, 3});
        cg::Graph             graph("declared result");
        auto                 &product = graph.declare_runtime_tensor<double>("T", std::vector<std::size_t>{6, 7, 5}, intermediate);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j,k <- i,k ; j,k", &product, a, b);
            cg::einsum("i,m <- i,j,k ; j,k,m", &e, product, d);
        }
        auto const tiling = decide(graph, 400);
        CHECK(tiling->num_tiled() == (intermediate ? 1 : 0));
        CHECK(declined_because(*tiling, "declared as a result") == !intermediate);
    }
}

TEST_CASE("AxisTiling slices an axis a contraction sums, when its destination collects the sum", "[ComputeGraph][Pass][AxisTiling]") {
    //   H[Q,i]  = sum_m A[Q,m] C[m,i]
    //   out[k]  = 0.5 out[k] + 2 sum_Qi H[Q,i] G[Q,i,k]
    // The second contraction sums both of H's axes, so every candidate slices a summed letter, and
    // the pass used to decline them all. out keeps none of them and nothing in the region reads it,
    // so it can collect the sum: scaled by 0.5 once before the loop, added to at every iteration.
    // out starts nonzero, so a prefactor applied per block, or not at all, shows in the result.
    constexpr std::size_t naux = 7;
    constexpr std::size_t m    = 4;
    constexpr std::size_t ni   = 6;
    constexpr std::size_t nk   = 3;
    auto const            a    = filled("A", {naux, m}, 3U);
    auto const            c    = filled("C", {m, ni}, 5U);
    auto const            g    = filled("G", {naux, ni, nk}, 7U);
    auto                  out  = filled("out", {nk}, 11U);
    auto const            out0 = out;

    cg::Graph graph("summed axis");
    {
        auto                  &half = graph.declare_runtime_tensor<double>("H", std::vector<std::size_t>{naux, ni}, /*intermediate=*/true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,i <- Q,m ; m,i", &half, a, c);
        cg::einsum("k <- Q,i ; Q,i,k", 0.5, &out, 2.0, half, g);
    }
    auto const tiling = decide(graph, 100);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->summed_into() == std::vector<std::string>{"out"});

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();

    RuntimeTensor<double> half("Href", std::vector<std::size_t>{naux, ni});
    half.zero();
    testing::reference_einsum("Q,i <- Q,m ; m,i", &half, a, c);
    auto expected = out0;
    testing::reference_einsum("k <- Q,i ; Q,i,k", 0.5, &expected, 2.0, half, g);
    for (std::size_t n = 0; n < nk; ++n) {
        INFO("element " << n);
        CHECK_THAT(out.data()[n], Catch::Matchers::WithinAbs(expected.data()[n], 1e-12));
    }
}

TEST_CASE("AxisTiling does not report a backed tensor a loop body writes as a held result", "[ComputeGraph][Pass][AxisTiling]") {
    // A backed tensor is its storage's to hold, not memory's, so it is never the declared result
    // the report asks the caller to redeclare. A body's handle for a tensor its parent declared
    // carries no backing of its own, and the check read the body's handle, so inside a loop it
    // reported the tensor anyway.
    std::size_t const v = 5;
    auto const        a = filled("A", {v, 4}, 3U);
    auto const        g = filled("G", {4, v, 6}, 5U);

    cg::TensorBacking backing;
    backing.kind     = "test.memory";
    backing.location = {{"name", "W"}};
    backing.write    = true;
    backing.prepare  = [](std::vector<std::size_t> const &dims, packed_gemm::ScalarType dtype) {
        cg::TensorBacking ready;
        ready.kind       = "test.memory";
        ready.location   = {{"name", "W"}};
        ready.write      = true;
        ready.dims       = dims;
        ready.dtype      = dtype;
        ready.read_slab  = [](void *, std::vector<std::pair<std::size_t, std::size_t>> const &) {};
        ready.write_slab = [](void const *, std::vector<std::pair<std::size_t, std::size_t>> const &) {};
        return ready;
    };

    cg::Graph graph("backed in a loop");
    auto     &w = graph.declare_runtime_tensor<double>("W", std::vector<std::size_t>{v, v, 6});
    graph.annotate_backing(w, backing);
    {
        auto                  &body = graph.add_loop("iterate", 2, [](std::size_t it) { return it + 1 < 2; });
        cg::CaptureGuard const guard(body);
        cg::einsum("i,j,k <- i,m ; m,j,k", &w, a, g);
    }
    REQUIRE(v * v * 6 * sizeof(double) > 100);
    auto const tiling = decide(graph, 100);
    CHECK_FALSE(declined_because(*tiling, "declared as a result"));
    auto const lines = tiling->explain();
    CHECK(std::ranges::none_of(lines, [](std::string const &line) { return line.find("held 'W'") != std::string::npos; }));
}

TEST_CASE("AxisTiling slices a backed operand where its blocks are the fewest runs of its storage", "[ComputeGraph][Pass][AxisTiling]") {
    //   H[i,j]  = sum_k Bk[i,j,k] v[k]          Bk backed, stored as declared
    //   out[m]  = sum_ij H[i,j] W[i,j,m]
    // Slicing i or j moves the same bytes, since the two have one extent. A block of j is one run of
    // Bk's storage per k; a block of i, the first stored axis, is one per (j, k). With a run costed
    // the pass takes j; with runs free it has no reason to prefer either and keeps the first, i.
    std::size_t const n  = 6;
    std::size_t const nk = 4;
    auto const        v  = filled("v", {nk}, 3U);
    auto const        w  = filled("W", {n, n, 2}, 5U);

    cg::TensorBacking backing;
    backing.kind      = "test.memory";
    backing.location  = {{"name", "Bk"}};
    backing.dims      = {n, n, nk};
    backing.dtype     = packed_gemm::ScalarType::Float64;
    backing.read_slab = [](void *, std::vector<std::pair<std::size_t, std::size_t>> const &) {};

    auto const chosen = [&](std::int64_t run_bytes) {
        std::int64_t const previous = config::get(option::GraphStorageRunBytes);
        config::set(option::GraphStorageRunBytes, run_bytes);
        RuntimeTensor<double> out("out", std::vector<std::size_t>{2});
        cg::Graph             graph("backed operand");
        auto                 &bk = graph.declare_runtime_tensor<double>("Bk", std::vector<std::size_t>{n, n, nk});
        graph.annotate_backing(bk, backing);
        auto &half = graph.declare_runtime_tensor<double>("H", std::vector<std::size_t>{n, n}, /*intermediate=*/true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,j,k ; k", &half, bk, v);
            cg::einsum("m <- i,j ; i,j,m", &out, half, w);
        }
        auto const tiling = decide(graph, static_cast<std::int64_t>(n * sizeof(double)));
        config::set(option::GraphStorageRunBytes, previous);
        REQUIRE(tiling->num_tiled() == 1);
        return tiling->axis_letters();
    };
    CHECK(chosen(0) == std::vector<std::string>{"i"});
    CHECK(chosen(65536) == std::vector<std::string>{"j"});
}

TEST_CASE("AxisTiling is off when the cap is zero", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    auto const tiling = decide(graph, 0);
    CHECK(tiling->slice_count() == 0);
    CHECK(declined_because(*tiling, "memory cap is zero"));
}

TEST_CASE("AxisTiling takes the block the cap allows", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // Three pairs would fit. Each factor of the integral lacks the other occupied index, so the
    // pairs are cut into blocks rather than grouped. A block of three on four occupied indices
    // takes as many blocks as a block of two, so it is two; and one ranged axis reads fewer bytes
    // than a block over both that the cap would allow (none would: two by two is four pairs).
    auto const tiling = decide(graph, 3 * static_cast<std::int64_t>(pair_bytes));
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->emission() == "range");
    CHECK(tiling->block_extents() == std::vector<std::int64_t>{1, 2});
    CHECK(tiling->depth() == 2);
    CHECK(tiling->iterations() == (nocc * nocc) / 2);
    CHECK(tiling->loops() == 1);
    CHECK(tiling->largest_after() == 2 * pair_bytes);
}

TEST_CASE("AxisTiling prefers the axes that stream the fewest bytes", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 no exchange");
    capture_without_exchange(graph, problem);

    // Without the exchange every pair of axes is a candidate. At this cap an occupied pair
    // fits (v^2 doubles) but two do not, so the occupied pair can only be taken a point at a
    // time, and each factor of the integral is read once per value of the other occupied index:
    // 2 o^2 Q v elements. Slicing i and b instead leaves a slice of o v that fits twice, so b can
    // be cut in blocks of two, and the factor carrying b is re-read once per i while the one
    // carrying i is re-read once per BLOCK of b: o^2 Q v + o (v / 2) Q v, which is less.
    REQUIRE(2 * pair_bytes > 400);
    REQUIRE(2 * nocc * nvir * sizeof(double) <= 400);
    auto const tiling = decide(graph, 400);

    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "b"});
    CHECK(tiling->emission() == "range");
    CHECK(tiling->block_extents() == std::vector<std::int64_t>{1, 2});
    CHECK(tiling->largest_after() == 2 * nocc * nvir * sizeof(double));
}

TEST_CASE("AxisTiling reads the cap the caller states over the option", "[ComputeGraph][Pass][AxisTiling]") {
    cg::passes::AxisTiling tiling;
    CHECK(tiling.memory_cap() == config::get(option::GraphTilingMemoryCap));
    tiling.set_memory_cap(4096);
    CHECK(tiling.memory_cap() == 4096);
}

// ── The rewrite ─────────────────────────────────────────────────────────────

namespace {

/// The energy the program computes with no tiling on it, which every tiled arm is held to.
double untiled_energy(Problem &problem) {
    cg::Graph graph("mp2 untiled");
    capture(graph, problem);
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();
    return problem.energy.data()[0];
}

/// The loop node the rewrite emitted, or nothing when it emitted none.
cg::Graph const *loop_body(cg::Graph const &graph) {
    for (auto const &node : graph.nodes()) {
        if (auto const *loop = node.op_data.get_if<cg::LoopDescriptor>(); loop != nullptr && loop->body) {
            return loop->body.get();
        }
    }
    return nullptr;
}

/// A body-declared tensor's dims, by name.
std::vector<std::size_t> body_dims(cg::Graph const &body, std::string_view name) {
    for (auto const &[id, handle] : body.tensors_map()) {
        if (handle.name == name) {
            return handle.dims;
        }
    }
    return {};
}

} // namespace

TEST_CASE("AxisTiling replays the same energy from the loop it emits", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const reference = untiled_energy(problem);

    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();

    // Re-associating: the reduction is summed pair by pair rather than over the whole
    // four-index tensor, so the two agree to the tier's bound and not to the bit.
    double const tiled = problem.energy.data()[0];
    INFO("untiled " << reference << " against tiled " << tiled);
    CHECK(std::abs(tiled - reference) <= cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14);
    CHECK(tiled != 0.0);

    // Replays restart the sweep rather than continuing it.
    problem.energy.data()[0] = 0.0;
    graph.execute();
    CHECK(problem.energy.data()[0] == Catch::Approx(tiled));
}

TEST_CASE("AxisTiling keeps a contraction's permutation operator", "[ComputeGraph][Pass][AxisTiling]") {
    // Defends the member contractions the pass emits. It re-rendered each one's spec from the
    // index lists and conjugation flags alone, so an operator the contraction carried was not in
    // the text it captured, and a tiled P(ab) contraction summed one term instead of two.
    //
    // Returns the energy and, when tiled, the pass that decided.
    auto const run = [](std::string const &spec, bool tiled) {
        Problem                                 problem;
        std::vector<std::size_t> const          shape{nocc, nvir, nocc, nvir};
        cg::Graph                               graph(tiled ? "operator tiled" : "operator untiled");
        std::shared_ptr<cg::passes::AxisTiling> tiling;
        {
            auto                  &integral    = graph.scratch_runtime<double>("K", shape);
            auto                  &amplitude   = graph.scratch_runtime<double>("T", shape);
            auto                  &combination = graph.scratch_runtime<double>("Kbar", shape);
            cg::CaptureGuard const guard(graph);
            cg::einsum(cg::EinsumFormatString(spec), &integral, problem.fitted, problem.fitted);
            cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
            cg::axpby(2.0, integral, 0.0, &combination);
            cg::dot_python(&problem.energy, combination, amplitude);
        }
        if (tiled) {
            tiling = decide(graph, 400);
        }
        auto manager = cg::PassManager::create_default();
        graph.apply(manager);
        problem.energy.data()[0] = 0.0;
        graph.execute();
        return std::pair{problem.energy.data()[0], tiling};
    };
    auto const agrees = [](double tiled, double reference) {
        INFO("untiled " << reference << " against tiled " << tiled);
        REQUIRE(reference != 0.0);
        CHECK(std::abs(tiled - reference) <= cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14);
    };

    SECTION("an operator on axes the pass does not slice travels with each member") {
        std::string const spec         = "i,a,j,b <- P(ab) Q,i,a ; Q,j,b";
        auto const [reference, unused] = run(spec, false);
        auto const [tiled, tiling]     = run(spec, true);
        REQUIRE(tiling->num_tiled() == 1);
        agrees(tiled, reference);
    }
    SECTION("an operator on the axes it would slice keeps them whole, with the reason") {
        std::string const spec         = "i,a,j,b <- P(ij) Q,i,a ; Q,j,b";
        auto const [reference, unused] = run(spec, false);
        auto const [tiled, tiling]     = run(spec, true);
        CHECK(declined_because(*tiling, "a permutation operator exchanges one of the candidate's sliced axes"));
        agrees(tiled, reference);
    }
}

TEST_CASE("AxisTiling emits a loop whose body declares the slice, not the slab", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);

    // The parent holds the zeroing of the accumulation and the loop, and nothing of the seven
    // nodes the region was.
    REQUIRE(graph.num_nodes() == 2);
    CHECK(graph.nodes()[0].kind == cg::OpKind::Scale);
    CHECK(graph.nodes()[1].kind == cg::OpKind::Loop);

    auto const *body = loop_body(graph);
    REQUIRE(body != nullptr);

    // Every four-index intermediate is re-declared at ONE pair. The sliced axes are gone rather
    // than one element wide: a loop variable is not an axis, and dropping them is what makes the
    // body's contraction the ordinary matrix product the hand-written pair loop performs.
    std::vector<std::size_t> const pair{nvir, nvir};
    for (auto const &name : {"K#0", "T#0", "K_again#0", "K_exchange#0", "Kbar#0"}) {
        INFO("body intermediate '" << name << "'");
        CHECK(body_dims(*body, name) == pair);
    }
    // The accumulation's partial is one element, and the caller's energy is not re-declared.
    CHECK(body_dims(*body, "axtile_partial#0") == std::vector<std::size_t>{1});

    // A slice of a caller's tensor is a VIEW of the caller's buffer, never a copy: every View
    // node in the body names a parent, and the integral's views name the caller's integral.
    std::size_t integral_views = 0;
    for (auto const &node : body->nodes()) {
        if (node.kind != cg::OpKind::View) {
            continue;
        }
        auto const *desc = node.op_data.get_if<cg::ViewDescriptor>();
        REQUIRE(desc != nullptr);
        auto const *parent = body->find_tensor(desc->parent_id);
        REQUIRE(parent != nullptr);
        if (parent->name == "B") {
            ++integral_views;
            CHECK(parent->data_ptr == problem.fitted.data());
        }
    }
    // Two: one per operand of the contraction, at two different occupied indices.
    CHECK(integral_views == 2);
}

TEST_CASE("AxisTiling leaves the storage invariants intact", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    REQUIRE(decide(graph, 400)->num_tiled() == 1);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);

    CHECK(cg::passes::duplicate_materializations(graph).empty());
    CHECK(cg::passes::stranded_materializations(graph).empty());

    // A body-declared intermediate's lifecycle is HOISTED to the parent, ahead of the loop,
    // which is where the placement rules put a loop body's workspace: a buffer the body reuses
    // on every iteration is allocated once. A setup body's workspace is the case that stays
    // inside, and this is not one.
    auto const *body = loop_body(graph);
    REQUIRE(body != nullptr);
    for (auto const &node : body->nodes()) {
        INFO("body node '" << node.label << "'");
        CHECK(node.kind != cg::OpKind::Materialize);
    }

    std::map<std::string, std::size_t> allocated;
    std::size_t                        loop_position = graph.num_nodes();
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        auto const &node = graph.nodes()[i];
        if (node.kind == cg::OpKind::Loop) {
            loop_position = i;
        }
        if (node.kind != cg::OpKind::Materialize) {
            continue;
        }
        for (auto const tid : node.outputs) {
            auto const *handle = graph.find_tensor(tid);
            if (handle != nullptr && handle->name.find('#') != std::string::npos) {
                INFO("'" << handle->name << "' is materialized behind the loop it feeds");
                CHECK(i < loop_position);
                ++allocated[handle->name];
            }
        }
    }
    // One lifecycle each, and the buffer that is allocated is the PAIR rather than the slab.
    for (auto const &name : {"K#0", "T#0", "K_again#0", "K_exchange#0", "Kbar#0", "axtile_partial#0"}) {
        INFO("body intermediate '" << name << "'");
        CHECK(allocated[name] == 1);
    }
    CHECK(body_dims(*body, "K#0") == std::vector<std::size_t>{nvir, nvir});
}

TEST_CASE("AxisTiling leaves a pair a permutation exchanges whole, with the reason", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 occupied exchange");
    capture_with_occupied_exchange(graph, problem);

    auto const tiling = decide(graph, 400);

    // The occupied pair is gone from the decision, and the tally says what took it out. The
    // pass then takes the virtual pair, which this program's permutation does leave in place:
    // a partner contraction is not emitted and the candidate is simply not offered.
    CHECK(declined_because(*tiling, "a permutation exchanges two of the candidate's sliced axes"));
    CHECK(tiling->axis_letters() == std::vector<std::string>{"a", "b"});
    CHECK(tiling->largest_after() == tiling->depth() * nocc * nocc * sizeof(double));
}

// ── Chunks ──────────────────────────────────────────────────────────────────

namespace {

/// How many nodes of each kind the loop body holds.
std::map<cg::OpKind, std::size_t> body_kinds(cg::Graph const &graph) {
    std::map<cg::OpKind, std::size_t> out;
    auto const                       *body = loop_body(graph);
    if (body == nullptr) {
        return out;
    }
    for (auto const &node : body->nodes()) {
        ++out[node.kind];
    }
    return out;
}

/// Capture, tile at @p cap, optimize, replay, and hand back the energy.
double tiled_energy(Problem &problem, std::int64_t cap, std::shared_ptr<cg::passes::AxisTiling> *out = nullptr) {
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, cap);
    if (out != nullptr) {
        *out = tiling;
    }
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();
    return problem.energy.data()[0];
}

} // namespace

TEST_CASE("AxisTiling emits a chunk as one grouped node per family", "[ComputeGraph][Pass][AxisTiling]") {
    BatchedProblem problem;
    cg::Graph      graph("pair batched chunked");
    capture_pair_batched(graph, problem);

    // Every operand of the contraction carries both occupied indices, so no block would read any
    // of them less often, and the pairs are grouped. Three pairs would fit; the chunk has to divide
    // the sixteen slices, so the depth is two.
    auto const tiling = decide(graph, 3 * static_cast<std::int64_t>(pair_bytes));
    REQUIRE(tiling->depth() == 2);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "grouped");
    CHECK(tiling->block_extents().empty());
    CHECK(tiling->loops() == 1);

    auto const kinds = body_kinds(graph);
    // Two contractions, one permutation, two accumulations, one reduction and the accumulation
    // into the energy: seven captured nodes, and at a chunk of two each becomes ONE grouped node
    // rather than two ungrouped ones. Nothing of the ungrouped families survives.
    CHECK(kinds.at(cg::OpKind::GroupedBatchedGemm) == 2);
    CHECK(kinds.at(cg::OpKind::GroupedPermute) == 1);
    CHECK(kinds.at(cg::OpKind::GroupedDirectProduct) == 1);
    CHECK(kinds.at(cg::OpKind::GroupedDot) == 1);
    // Two combinations plus the accumulation into the energy.
    CHECK(kinds.at(cg::OpKind::GroupedAxpby) == 3);
    CHECK(kinds.count(cg::OpKind::Einsum) == 0);
    CHECK(kinds.count(cg::OpKind::Dot) == 0);

    // The chunk's members each hold their own pair, so the store scales with the chunk.
    CHECK(tiling->largest_after() == 2 * pair_bytes);
    CHECK(body_dims(*loop_body(graph), "K#1") == std::vector<std::size_t>{nvir, nvir});
}

TEST_CASE("AxisTiling gets the same energy from a grouped chunk as from untiled", "[ComputeGraph][Pass][AxisTiling]") {
    BatchedProblem problem;
    auto const     reference = batched_reference(problem);

    cg::Graph graph("pair batched chunked");
    capture_pair_batched(graph, problem);
    auto const tiling = decide(graph, 3 * static_cast<std::int64_t>(pair_bytes));
    REQUIRE(tiling->emission() == "grouped");
    REQUIRE(tiling->depth() == 2);
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();

    double const tiled = problem.energy.data()[0];
    INFO("grouped " << tiled << ", reference " << reference.value);
    CHECK(std::abs(tiled - reference.value) <= 1e-12 * reference.magnitude);
}

TEST_CASE("AxisTiling gets the same energy from a block of pairs as from one", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const reference = untiled_energy(problem);

    std::shared_ptr<cg::passes::AxisTiling> per_pair;
    std::shared_ptr<cg::passes::AxisTiling> per_chunk;
    double const                            one  = tiled_energy(problem, 400, &per_pair);
    double const                            many = tiled_energy(problem, 3 * static_cast<std::int64_t>(pair_bytes), &per_chunk);

    REQUIRE(per_pair->depth() == 1);
    REQUIRE(per_chunk->depth() == 2);
    REQUIRE(per_chunk->emission() == "range");
    CHECK(per_chunk->largest_after() == 2 * per_pair->largest_after());

    double const bound = cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14;
    INFO("per pair " << one << ", per chunk " << many << ", untiled " << reference);
    CHECK(std::abs(many - one) <= bound);
    CHECK(std::abs(many - reference) <= bound);
}

// ── Range blocks ────────────────────────────────────────────────────────────

namespace {

/// A reproducible filling of any element type; a complex element gets an imaginary part too.
template <typename T>
RuntimeTensor<T> filled_as(std::string name, std::vector<std::size_t> dims, unsigned seed) {
    RuntimeTensor<T> out(std::move(name), std::move(dims));
    auto             next = [&seed] {
        seed = (seed * 1103515245U) + 12345U;
        return (static_cast<double>((seed >> 8U) % 2048U) / 1024.0) - 1.0;
    };
    for (std::size_t i = 0; i < out.size(); ++i) {
        if constexpr (IsComplexV<T>) {
            auto const re = static_cast<RemoveComplexT<T>>(next());
            out.data()[i] = T{re, static_cast<RemoveComplexT<T>>(next())};
        } else {
            out.data()[i] = static_cast<T>(next());
        }
    }
    return out;
}

/// How far a re-associated sum may move from the reference, relative to the sum of its terms'
/// magnitudes.
template <typename T>
double sum_tolerance() {
    return 256.0 * static_cast<double>(std::numeric_limits<RemoveComplexT<T>>::epsilon());
}

/// The density-fitted MP2 program of the cases above, at any element type and extents.
template <typename T>
struct Mp2 {
    std::size_t      o;
    std::size_t      v;
    std::size_t      naux;
    RuntimeTensor<T> fitted;
    RuntimeTensor<T> denominator;
    RuntimeTensor<T> energy{"E", std::vector<std::size_t>{1}};

    Mp2(std::size_t occupied, std::size_t virtuals, std::size_t auxiliary)
        : o(occupied), v(virtuals), naux(auxiliary), fitted(filled_as<T>("B", {auxiliary, occupied, virtuals}, 7U)),
          denominator(filled_as<T>("D", {occupied, virtuals, occupied, virtuals}, 19U)) {}

    /// Bytes @p n occupied pairs of a four-index intermediate hold.
    [[nodiscard]] std::int64_t pairs(std::size_t n) const { return static_cast<std::int64_t>(n * v * v * sizeof(T)); }
};

template <typename T>
void capture_mp2(cg::Graph &graph, Mp2<T> &p) {
    std::vector<std::size_t> const shape{p.o, p.v, p.o, p.v};
    auto                          &integral    = graph.scratch_runtime<T>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<T>("T", shape);
    auto                          &again       = graph.scratch_runtime<T>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<T>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<T>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, p.fitted, p.fitted);
    cg::direct_product(T{1}, integral, p.denominator, T{0}, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, p.fitted, p.fitted);
    cg::permute("i,a,j,b <- i,b,j,a", &exchange, again);
    cg::axpby(T{2}, again, T{0}, &combination);
    cg::axpby(T{-1}, exchange, T{1}, &combination);
    cg::dot_python(&p.energy, combination, amplitude);
}

/// The MP2 energy by plain loops over the reference contraction.
template <typename T>
Expected<T> mp2_reference(Mp2<T> &p) {
    RuntimeTensor<T> integral("Kref", std::vector<std::size_t>{p.o, p.v, p.o, p.v});
    integral.zero();
    testing::reference_einsum("i,a,j,b <- Q,i,a ; Q,j,b", &integral, p.fitted, p.fitted);
    Expected<T> out;
    for (std::size_t i = 0; i < p.o; ++i) {
        for (std::size_t a = 0; a < p.v; ++a) {
            for (std::size_t j = 0; j < p.o; ++j) {
                for (std::size_t b = 0; b < p.v; ++b) {
                    T const k    = integral(i, a, j, b);
                    T const term = ((T{2} * k) - integral(i, b, j, a)) * k * p.denominator(i, a, j, b);
                    out.value += term;
                    out.magnitude += static_cast<double>(std::abs(term));
                }
            }
        }
    }
    return out;
}

/// Optimize a tiled graph as a caller would, replay it, and hand back the energy.
template <typename T>
T replay_mp2(cg::Graph &graph, Mp2<T> &p) {
    p.energy.data()[0] = T{0};
    graph.execute();
    return p.energy.data()[0];
}

/// Every loop body the rewrite emitted, in the parent's order.
std::vector<cg::Graph const *> loop_bodies(cg::Graph const &graph) {
    std::vector<cg::Graph const *> out;
    for (auto const &node : graph.nodes()) {
        if (auto const *loop = node.op_data.get_if<cg::LoopDescriptor>(); loop != nullptr && loop->body) {
            out.push_back(loop->body.get());
        }
    }
    return out;
}

/// How many nodes of each kind one body holds.
std::map<cg::OpKind, std::size_t> kinds_of(cg::Graph const &body) {
    std::map<cg::OpKind, std::size_t> out;
    for (auto const &node : body.nodes()) {
        ++out[node.kind];
    }
    return out;
}

/// Whether a body holds any grouped node.
bool holds_grouped(cg::Graph const &body) {
    auto const kinds = kinds_of(body);
    for (auto const kind : {cg::OpKind::GroupedBatchedGemm, cg::OpKind::GroupedPermute, cg::OpKind::GroupedAxpby, cg::OpKind::GroupedDot,
                            cg::OpKind::GroupedDirectProduct}) {
        if (kinds.contains(kind)) {
            return true;
        }
    }
    return false;
}

} // namespace

TEMPLATE_TEST_CASE("AxisTiling cuts both occupied axes of DF-MP2 into square blocks", "[ComputeGraph][Pass][AxisTiling]", float, double,
                   std::complex<double>) {
    using T = TestType;
    Mp2<T>    p(4, 6, 7);
    cg::Graph graph("mp2 blocks");
    capture_mp2(graph, p);

    // Four pairs fit. Each factor of the integral carries one occupied index and is re-read once
    // per block of the other, so two by two re-reads each twice, where one by four re-reads one of
    // them four times: the square streams fewer bytes.
    auto const tiling = decide(graph, p.pairs(4));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->emission() == "range");
    CHECK(tiling->block_extents() == std::vector<std::int64_t>{2, 2});
    CHECK(tiling->depth() == 4);
    CHECK(tiling->loops() == 1);
    CHECK(tiling->iterations() == 4);
    CHECK(tiling->largest_after() == static_cast<std::size_t>(p.pairs(4)));

    // One node per captured statement, none grouped, and the block keeps its axes.
    auto const bodies = loop_bodies(graph);
    REQUIRE(bodies.size() == 1);
    auto const kinds = kinds_of(*bodies[0]);
    CHECK(kinds.at(cg::OpKind::Einsum) == 2);
    CHECK(kinds.at(cg::OpKind::Permute) == 1);
    CHECK(kinds.at(cg::OpKind::DirectProduct) == 1);
    CHECK(kinds.at(cg::OpKind::Dot) == 1);
    // Two combinations plus the accumulation into the energy.
    CHECK(kinds.at(cg::OpKind::Axpby) == 3);
    CHECK_FALSE(holds_grouped(*bodies[0]));
    CHECK(body_dims(*bodies[0], "K#0") == std::vector<std::size_t>{2, p.v, 2, p.v});

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const expected = mp2_reference(p);
    T const    energy   = replay_mp2(graph, p);
    INFO("tiled " << energy << ", reference " << expected.value);
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<T>() * expected.magnitude);

    // Replays restart the sweep of block starts and ends rather than continuing it.
    CHECK(replay_mp2(graph, p) == energy);
}

TEMPLATE_TEST_CASE("AxisTiling runs each remainder block through a loop of its own", "[ComputeGraph][Pass][AxisTiling]", float, double,
                   std::complex<double>) {
    using T = TestType;
    Mp2<T>    p(5, 6, 7);
    cg::Graph graph("mp2 ragged blocks");
    capture_mp2(graph, p);

    // Five occupied indices in blocks of two leave a remainder of one on each axis, so the
    // iteration space is four loops: two by two, two by one, one by two and one by one.
    auto const tiling = decide(graph, p.pairs(4));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "range");
    CHECK(tiling->block_extents() == std::vector<std::int64_t>{2, 2});
    CHECK(tiling->loops() == 4);
    CHECK(tiling->iterations() == 4 + 2 + 2 + 1);

    auto const bodies = loop_bodies(graph);
    REQUIRE(bodies.size() == 4);
    std::vector<std::vector<std::size_t>> const shapes{{2, p.v, 2, p.v}, {2, p.v, 1, p.v}, {1, p.v, 2, p.v}, {1, p.v, 1, p.v}};
    for (std::size_t l = 0; l < bodies.size(); ++l) {
        INFO("loop " << l);
        CHECK(body_dims(*bodies[l], "K#0") == shapes[l]);
        CHECK_FALSE(holds_grouped(*bodies[l]));
    }

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const expected = mp2_reference(p);
    T const    energy   = replay_mp2(graph, p);
    INFO("tiled " << energy << ", reference " << expected.value);
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<T>() * expected.magnitude);
    CHECK(replay_mp2(graph, p) == energy);
}

TEMPLATE_TEST_CASE("AxisTiling blocks the particle ladder so the shared integrals are read once per block",
                   "[ComputeGraph][Pass][AxisTiling]", float, double, std::complex<double>) {
    // The density-fitted particle ladder of a CCSD residual, written over every index:
    //   H[Q,a,f,i,j]  = sum_e Bvv[Q,a,e] Tau[i,j,e,f]
    //   r2[i,j,a,b]  += sum_Qf H[Q,a,f,i,j] Bvv[Q,b,f]
    // The integrals carry no occupied index, so any blocking of the pairs reads them once per
    // block, and the five-index intermediate is what the cap forbids holding whole.
    using T                    = TestType;
    constexpr std::size_t o    = 4;
    constexpr std::size_t v    = 5;
    constexpr std::size_t naux = 6;
    auto                  bvv  = filled_as<T>("Bvv", {naux, v, v}, 11U);
    auto                  tau  = filled_as<T>("Tau", {o, o, v, v}, 13U);
    auto                  r2   = filled_as<T>("r2", {o, o, v, v}, 17U);

    RuntimeTensor<T> expected = r2;
    {
        RuntimeTensor<T> half("Href", std::vector<std::size_t>{naux, v, v, o, o});
        half.zero();
        testing::reference_einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        testing::reference_einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", T{1}, &expected, T{1}, half, bvv);
    }

    cg::Graph graph("ladder");
    {
        auto                  &half = graph.scratch_runtime<T>("H", std::vector<std::size_t>{naux, v, v, o, o});
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", T{1}, &r2, T{1}, half, bvv);
    }
    // Two pairs of the intermediate fit and one occupied row of it (four pairs) does not, so no
    // candidate can keep an occupied axis whole and every one of them is cut into blocks.
    auto const tiling = decide(graph, static_cast<std::int64_t>(2 * naux * v * v * sizeof(T)));
    INFO("sliced " << fmt::format("{}", fmt::join(tiling->axis_letters(), ",")) << " in blocks of "
                   << fmt::format("{}", fmt::join(tiling->block_extents(), ",")));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "range");

    // Each loop holds the two contractions once, over the whole block.
    for (auto const *body : loop_bodies(graph)) {
        auto const kinds = kinds_of(*body);
        CHECK(kinds.at(cg::OpKind::Einsum) == 2);
        CHECK_FALSE(holds_grouped(*body));
    }

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();

    // The block's contractions take a matrix-product route, not the generic loop.
    std::string const route{cg::dispatch::last_dispatch_route()};
    INFO("last route " << route);
    CHECK(route.rfind("generic", 0) != 0);

    double scale = 0.0;
    for (std::size_t n = 0; n < expected.size(); ++n) {
        scale = std::max(scale, static_cast<double>(std::abs(expected.data()[n])));
    }
    for (std::size_t n = 0; n < expected.size(); ++n) {
        INFO("element " << n);
        CHECK(std::abs(r2.data()[n] - expected.data()[n]) <= sum_tolerance<T>() * static_cast<double>(naux * v) * scale);
    }
}

TEST_CASE("AxisTiling keeps an operand it slices on every axis by cutting one of them into blocks", "[ComputeGraph][Pass][AxisTiling]") {
    // K[i,a] = X[i] w[a], E = sum K D. Every candidate slices every axis of some operand: i
    // takes all of X, a all of w, and both all of K. A point leaves such an operand nothing, so
    // the pass used to decline this program outright; a block keeps the axis.
    constexpr std::size_t ni = 8;
    constexpr std::size_t na = 4;
    auto                  x  = filled_as<double>("X", {ni}, 3U);
    auto                  w  = filled_as<double>("w", {na}, 5U);
    auto                  d  = filled_as<double>("D", {ni, na}, 7U);
    RuntimeTensor<double> energy("E", std::vector<std::size_t>{1});

    double want = 0.0;
    for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t a = 0; a < na; ++a) {
            want += x(i) * w(a) * d(i, a);
        }
    }

    auto const build = [&](cg::Graph &graph) {
        auto                  &outer = graph.scratch_runtime<double>("K", std::vector<std::size_t>{ni, na});
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,a <- i ; a", &outer, x, w);
        cg::dot_python(&energy, outer, d);
    };

    SECTION("a block fits, so the program is tiled") {
        cg::Graph graph("outer product");
        build(graph);
        auto const tiling = decide(graph, static_cast<std::int64_t>(2 * na * sizeof(double)));
        REQUIRE(tiling->num_tiled() == 1);
        CHECK(tiling->emission() == "range");
        auto manager = cg::PassManager::create_default();
        graph.apply(manager);
        energy.data()[0] = 0.0;
        graph.execute();
        CHECK_THAT(energy.data()[0], Catch::Matchers::WithinRel(want, 1e-12));
    }
    SECTION("only points fit, so it is declined with the reason") {
        // One element: a point of both axes fits and a block of either does not.
        cg::Graph graph("outer product");
        build(graph);
        auto const tiling = decide(graph, static_cast<std::int64_t>(sizeof(double)));
        CHECK(tiling->num_tiled() == 0);
        CHECK(declined_because(*tiling, "every axis of one operand is sliced"));
    }
}

// ── Regions ─────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("AxisTiling tiles the ladder chain of a residual whose other terms sum over its occupied indices",
                   "[ComputeGraph][Pass][AxisTiling]", float, double, std::complex<double>) {
    // A coupled-cluster residual in miniature, in the order its terms are written:
    //   Tau[i,j,a,b]  = t1[i,a] t1[j,b] + t2[i,j,a,b]
    //   F[a,e]        = sum_mnf Tau[m,n,a,f] g[m,n,e,f]            sums Tau's occupied indices
    //   r2[i,j,a,b]   = sum_e t2[i,j,a,e] F[b,e]
    //   H[Q,a,f,i,j]  = sum_e Bvv[Q,a,e] Tau[i,j,e,f]              the ladder, over the cap
    //   r2[i,j,a,b]  += sum_Qf H[Q,a,f,i,j] Bvv[Q,b,f]
    //   X[i,j,a,b]    = sum_e Tau[i,j,a,e] F[e,b];  r2 += X + X[j,i,b,a]   exchanges i with j
    // Taken as one run, any slice of the occupied pair reaches F's contraction, which sums over it,
    // and the symmetrizing permutation, which exchanges it, so no occupied slicing survives. The
    // ladder's own statements need neither: the region is H's writer and reader, and everything
    // they share with the other terms fits the cap and stays whole.
    using T                    = TestType;
    constexpr std::size_t o    = 4;
    constexpr std::size_t v    = 5;
    constexpr std::size_t naux = 6;
    auto                  t1   = filled_as<T>("t1", {o, v}, 3U);
    auto                  t2   = filled_as<T>("t2", {o, o, v, v}, 5U);
    auto                  g    = filled_as<T>("g", {o, o, v, v}, 7U);
    auto                  bvv  = filled_as<T>("Bvv", {naux, v, v}, 11U);
    RuntimeTensor<T>      r2("r2", std::vector<std::size_t>{o, o, v, v});
    r2.zero();

    // The reference, statement by statement.
    RuntimeTensor<T> expected("r2ref", std::vector<std::size_t>{o, o, v, v});
    expected.zero();
    {
        RuntimeTensor<T> tau("Tauref", std::vector<std::size_t>{o, o, v, v});
        RuntimeTensor<T> fae("Fref", std::vector<std::size_t>{v, v});
        RuntimeTensor<T> half("Href", std::vector<std::size_t>{naux, v, v, o, o});
        RuntimeTensor<T> x("Xref", std::vector<std::size_t>{o, o, v, v});
        tau.zero();
        fae.zero();
        half.zero();
        x.zero();
        testing::reference_einsum("i,j,a,b <- i,a ; j,b", &tau, t1, t1);
        for (std::size_t n = 0; n < tau.size(); ++n) {
            tau.data()[n] += t2.data()[n];
        }
        testing::reference_einsum("a,e <- m,n,a,f ; m,n,e,f", &fae, tau, g);
        testing::reference_einsum("i,j,a,b <- i,j,a,e ; b,e", &expected, t2, fae);
        testing::reference_einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        testing::reference_einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", T{1}, &expected, T{1}, half, bvv);
        testing::reference_einsum("i,j,a,b <- i,j,a,e ; e,b", &x, tau, fae);
        for (std::size_t i = 0; i < o; ++i) {
            for (std::size_t j = 0; j < o; ++j) {
                for (std::size_t a = 0; a < v; ++a) {
                    for (std::size_t b = 0; b < v; ++b) {
                        expected(i, j, a, b) += x(i, j, a, b) + x(j, i, b, a);
                    }
                }
            }
        }
    }

    cg::Graph graph("residual");
    {
        std::vector<std::size_t> const doubles{o, o, v, v};
        auto                          &tau  = graph.scratch_runtime<T>("Tau", doubles);
        auto                          &fae  = graph.scratch_runtime<T>("F", std::vector<std::size_t>{v, v});
        auto                          &half = graph.scratch_runtime<T>("H", std::vector<std::size_t>{naux, v, v, o, o});
        auto                          &x    = graph.scratch_runtime<T>("X", doubles);
        auto                          &xp   = graph.scratch_runtime<T>("Xp", doubles);
        cg::CaptureGuard const         guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", &tau, t1, t1);
        cg::axpby(T{1}, t2, T{1}, &tau);
        cg::einsum("a,e <- m,n,a,f ; m,n,e,f", &fae, tau, g);
        cg::einsum("i,j,a,b <- i,j,a,e ; b,e", &r2, t2, fae);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", T{1}, &r2, T{1}, half, bvv);
        cg::einsum("i,j,a,b <- i,j,a,e ; e,b", &x, tau, fae);
        cg::permute("j,i,b,a <- i,j,a,b", &xp, x);
        cg::axpby(T{1}, x, T{1}, &r2);
        cg::axpby(T{1}, xp, T{1}, &r2);
    }
    std::size_t const captured = graph.num_nodes();

    // Four pairs of H fit; Tau, F and X fit whole, so the region does not follow them.
    auto const tiling = decide(graph, static_cast<std::int64_t>(4 * naux * v * v * sizeof(T)));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->axis_names().front().starts_with("H["));
    INFO("sliced " << fmt::format("{}", fmt::join(tiling->axis_letters(), ",")) << " blocks "
                   << fmt::format("{}", fmt::join(tiling->block_extents(), ",")));
    // An occupied index: one i of H is exactly the four pairs the cap allows. Slicing it labels
    // Tau's occupied axes, which F's contraction sums over, so this is the decision a region that
    // followed Tau into F would have had to give up.
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i"});

    // The loop holds the ladder's two contractions and nothing else; the other eight statements
    // run as captured, around it.
    auto const bodies = loop_bodies(graph);
    REQUIRE(!bodies.empty());
    for (auto const *body : bodies) {
        CHECK(kinds_of(*body).at(cg::OpKind::Einsum) == 2);
        CHECK(kinds_of(*body).count(cg::OpKind::Permute) == 0);
        CHECK(kinds_of(*body).count(cg::OpKind::Axpby) == 0);
    }
    CHECK(graph.num_nodes() == captured - 2 + bodies.size());

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();

    double scale = 0.0;
    for (std::size_t n = 0; n < expected.size(); ++n) {
        scale = std::max(scale, static_cast<double>(std::abs(expected.data()[n])));
    }
    for (std::size_t n = 0; n < expected.size(); ++n) {
        INFO("element " << n);
        CHECK(std::abs(r2.data()[n] - expected.data()[n]) <= sum_tolerance<T>() * static_cast<double>(naux * v * v) * scale);
    }
}

TEST_CASE("AxisTiling gives every intermediate over the cap a region of its own", "[ComputeGraph][Pass][AxisTiling]") {
    // Two independent energies in one graph, each with its own four-index intermediates.
    Mp2<double> first(4, 6, 7);
    Mp2<double> second(5, 6, 7);
    cg::Graph   graph("two energies");
    capture_mp2(graph, first);
    capture_mp2(graph, second);

    auto const tiling = decide(graph, first.pairs(1));
    CHECK(tiling->num_tiled() == 2);
    CHECK(tiling->explain().size() >= 2);
    CHECK(loop_bodies(graph).size() == 2);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    first.energy.data()[0]  = 0.0;
    second.energy.data()[0] = 0.0;
    graph.execute();
    for (auto *p : {&first, &second}) {
        auto const expected = mp2_reference(*p);
        INFO("o = " << p->o);
        CHECK(std::abs(p->energy.data()[0] - expected.value) <= sum_tolerance<double>() * expected.magnitude);
    }
}

TEST_CASE("AxisTiling declines a region it cannot gather around a statement between its own", "[ComputeGraph][Pass][AxisTiling]") {
    // K = B B, v[i,j] = K M, u = v v as a matrix product, T = K u, E = sum T D. The product of v
    // with itself is no statement this pass slices, and it sits between two that it must: it
    // reads what the region writes before it and writes what the region reads after it, so the
    // region can neither move down past it nor up past it.
    Problem               problem;
    auto                  weights = filled("M", {nvir, nvir}, 23U);
    RuntimeTensor<double> u_out("u", std::vector<std::size_t>{nocc, nocc});
    cg::Graph             graph("ungatherable");
    {
        std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
        auto                          &integral = graph.scratch_runtime<double>("K", shape);
        auto                          &pairs    = graph.scratch_runtime<double>("v", std::vector<std::size_t>{nocc, nocc});
        auto                          &scaled   = graph.scratch_runtime<double>("T", shape);
        cg::CaptureGuard const         guard(graph);
        cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
        cg::einsum("i,j <- i,a,j,b ; a,b", &pairs, integral, weights);
        cg::gemm(1.0, pairs, pairs, 0.0, &u_out);
        cg::einsum("i,a,j,b <- i,a,j,b ; i,j", &scaled, integral, u_out);
        cg::dot_python(&problem.energy, scaled, problem.denominator);
    }
    // Two pairs of K fit, so the occupied pair can be cut into blocks and v, which carries
    // nothing but the pair, keeps an axis; that is the candidate that reaches the gathering.
    REQUIRE(2 * pair_bytes <= 600);
    auto const  tiling = decide(graph, 600);
    std::string said   = fmt::format("largest before {}; ", tiling->largest_before());
    for (auto const &[reason, count] : tiling->skip_reasons()) {
        said += fmt::format("{} x {}; ", count, reason);
    }
    INFO(said);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "cannot be gathered into one loop"));
}

TEST_CASE("AxisTiling reads an input whole when its producer cannot join the loop", "[ComputeGraph][Pass][AxisTiling]") {
    // The integral's factor is squared in place by an element transform, a kind no loop this pass
    // emits carries. The region around the four-index integral reads that factor, and the factor is
    // over the cap, but a tensor the region only READS is an input: it is read through slices of
    // its whole buffer rather than pulling its producer into the loop and giving up.
    Problem   problem;
    double    reference = 0.0;
    cg::Graph graph("squared factor");
    {
        auto                  &factor = graph.scratch_runtime<double>("Bsq", std::vector<std::size_t>{naux, nocc, nvir});
        auto                  &K      = graph.scratch_runtime<double>("K", std::vector<std::size_t>{nocc, nvir, nocc, nvir});
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, problem.fitted, 0.0, &factor);
        cg::element_transform(&factor, "square");
        cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &K, factor, factor);
        cg::dot_python(&problem.energy, K, problem.denominator);
    }
    {
        RuntimeTensor<double> squared("Bsq_ref", std::vector<std::size_t>{naux, nocc, nvir});
        for (std::size_t n = 0; n < squared.size(); ++n) {
            squared.data()[n] = problem.fitted.data()[n] * problem.fitted.data()[n];
        }
        RuntimeTensor<double> integral("Kref", std::vector<std::size_t>{nocc, nvir, nocc, nvir});
        integral.zero();
        testing::reference_einsum("i,a,j,b <- Q,i,a ; Q,j,b", &integral, squared, squared);
        for (std::size_t n = 0; n < integral.size(); ++n) {
            reference += integral.data()[n] * problem.denominator.data()[n];
        }
    }

    REQUIRE(naux * nocc * nvir * sizeof(double) > 400);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);
    auto const whole    = tiling->whole();
    auto const streamed = tiling->streamed();
    CHECK(std::ranges::find(streamed, std::string{"Bsq"}) != streamed.end());
    CHECK(std::ranges::find(whole, std::string{"Bsq"}) == whole.end());
    for (auto const *body : loop_bodies(graph)) {
        CHECK(kinds_of(*body).count(cg::OpKind::ElementTransform) == 0);
        CHECK(kinds_of(*body).count(cg::OpKind::Axpby) <= 1);
    }

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();
    CHECK_THAT(problem.energy.data()[0], Catch::Matchers::WithinRel(reference, 1e-12));
}

TEST_CASE("AxisTiling builds its loop over the graph's operands, not the caller's", "[ComputeGraph][Pass][AxisTiling]") {
    // Capture adopts each operand's storage so the caller may drop its own tensor before the graph
    // runs. The pass used to build its loop body over the caller's tensor instead, and a program
    // that captured in a function and returned only the graph read freed memory in apply(). Here
    // each operand is replaced, at the same address, by a tensor of another shape, which a body
    // built over the caller's tensor would read, and the result is still the captured one.
    constexpr std::size_t                ni = 6;
    constexpr std::size_t                nj = 7;
    constexpr std::size_t                nk = 5;
    constexpr std::size_t                nm = 3;
    std::optional<RuntimeTensor<double>> a{filled("A", {ni, nk}, 3U)};
    std::optional<RuntimeTensor<double>> b{filled("B", {nj, nk}, 5U)};
    std::optional<RuntimeTensor<double>> d{filled("D", {nj, nk, nm}, 7U)};
    RuntimeTensor<double>                e("E", std::vector<std::size_t>{ni, nm});
    e.zero();

    RuntimeTensor<double> expected("E_ref", std::vector<std::size_t>{ni, nm});
    expected.zero();
    {
        RuntimeTensor<double> product("T_ref", std::vector<std::size_t>{ni, nj, nk});
        product.zero();
        testing::reference_einsum("i,j,k <- i,k ; j,k", &product, *a, *b);
        testing::reference_einsum("i,m <- i,j,k ; j,k,m", &expected, product, *d);
    }

    cg::Graph graph("dropped operands");
    {
        auto                  &product = graph.scratch_runtime<double>("T", std::vector<std::size_t>{ni, nj, nk});
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,k <- i,k ; j,k", &product, *a, *b);
        cg::einsum("i,m <- i,j,k ; j,k,m", &e, product, *d);
    }
    for (auto *operand : {&a, &b, &d}) {
        operand->reset();
        operand->emplace(filled("other", {2, 2}, 13U));
    }

    REQUIRE(ni * nj * nk * sizeof(double) > 400);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(graph.verify().empty());

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();
    for (std::size_t n = 0; n < expected.size(); ++n) {
        INFO("element " << n);
        CHECK_THAT(e.data()[n], Catch::Matchers::WithinAbs(expected.data()[n], 1e-12));
    }
}

// ── Loop bodies ─────────────────────────────────────────────────────────────

namespace {

/// The loop body the capture made, and the integral the parent declared for it.
template <typename T>
struct InLoop {
    cg::Graph        &body;
    RuntimeTensor<T> &integral;
};

/// The MP2 program captured into the body of a loop of @p iterations, its intermediates declared
/// on the PARENT, which is how an iterative method declares its scratch: the body names each of
/// them through a handle of its own.
template <typename T>
InLoop<T> capture_mp2_in_loop(cg::Graph &graph, Mp2<T> &p, std::size_t iterations) {
    std::vector<std::size_t> const shape{p.o, p.v, p.o, p.v};
    auto                          &integral    = graph.scratch_runtime<T>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<T>("T", shape);
    auto                          &again       = graph.scratch_runtime<T>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<T>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<T>("Kbar", shape);
    auto                  &body = graph.add_loop("iterate", iterations, [iterations](std::size_t it) { return it + 1 < iterations; });
    cg::CaptureGuard const guard(body);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, p.fitted, p.fitted);
    cg::direct_product(T{1}, integral, p.denominator, T{0}, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, p.fitted, p.fitted);
    cg::permute("i,a,j,b <- i,b,j,a", &exchange, again);
    cg::axpby(T{2}, again, T{0}, &combination);
    cg::axpby(T{-1}, exchange, T{1}, &combination);
    cg::dot_python(&p.energy, combination, amplitude);
    return {.body = body, .integral = integral};
}

} // namespace

TEMPLATE_TEST_CASE("AxisTiling tiles a region inside a loop body, over intermediates the parent declared",
                   "[ComputeGraph][Pass][AxisTiling]", float, double, std::complex<double>) {
    // The body names the parent's intermediates through handles of its own, registered with no
    // ownership claim, so read off the body alone they are not intermediates at all. The pass
    // reads intermediacy off the declaration, so the body is tiled as the top level would be,
    // and the loop it emits nests inside the body and runs once per outer iteration.
    using T = TestType;
    Mp2<T>      p(5, 6, 7);
    cg::Graph   graph("mp2 in a loop");
    auto const &body = capture_mp2_in_loop(graph, p, 3).body;

    auto const tiling = decide(graph, p.pairs(4));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "range");
    CHECK(tiling->block_extents() == std::vector<std::int64_t>{2, 2});

    // The parent still holds one loop, and that loop's body now holds the four loops the blocks
    // and their remainders need, each declaring its block rather than the whole integral.
    REQUIRE(graph.num_nodes() == 1);
    auto const inner = loop_bodies(body);
    REQUIRE(inner.size() == 4);
    CHECK(body_dims(*inner[0], "K#0") == std::vector<std::size_t>{2, p.v, 2, p.v});

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const expected = mp2_reference(p);
    T const    energy   = replay_mp2(graph, p);
    INFO("tiled " << energy << ", reference " << expected.value);
    // Every outer iteration recomputes the same energy, so the last one is checked; a nested
    // sweep that continued from where the previous iteration left it would not reproduce it.
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<T>() * expected.magnitude);
    CHECK(replay_mp2(graph, p) == energy);
}

TEST_CASE("AxisTiling keeps whole a body's intermediate that the parent reads after the loop", "[ComputeGraph][Pass][AxisTiling]") {
    // The integral is formed in the body and read again by the parent once the loop is done, so
    // the region inside cannot re-declare it a block at a time: the loop writes blocks of the
    // parent's buffer instead, and the parent's reduction reads every element.
    Mp2<double>           p(4, 6, 7);
    RuntimeTensor<double> after("E_after", std::vector<std::size_t>{1});
    cg::Graph             graph("mp2 read after the loop");
    auto const            captured = capture_mp2_in_loop(graph, p, 2);
    auto const           &body     = captured.body;
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&after, captured.integral, p.denominator);
    }

    auto const tiling = decide(graph, p.pairs(4));
    REQUIRE(tiling->num_tiled() == 1);
    auto const inner = loop_bodies(body);
    REQUIRE(!inner.empty());
    for (auto const *nested : inner) {
        CHECK(body_dims(*nested, "K#0").empty());
    }

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    p.energy.data()[0] = 0.0;
    graph.execute();

    RuntimeTensor<double> integral("Kref", std::vector<std::size_t>{p.o, p.v, p.o, p.v});
    integral.zero();
    testing::reference_einsum("i,a,j,b <- Q,i,a ; Q,j,b", &integral, p.fitted, p.fitted);
    double want = 0.0;
    for (std::size_t n = 0; n < integral.size(); ++n) {
        want += integral.data()[n] * p.denominator.data()[n];
    }
    CHECK_THAT(after.data()[0], Catch::Matchers::WithinRel(want, 1e-12));
    auto const expected = mp2_reference(p);
    CHECK(std::abs(p.energy.data()[0] - expected.value) <= sum_tolerance<double>() * expected.magnitude);
}

// ── The cap derived from the memory budget ──────────────────────────────────

namespace {

/// Every byte the four-index program names: the factor, the denominator, the energy and the five
/// four-index intermediates.
constexpr std::size_t program_bytes = (naux * nocc * nvir * sizeof(double)) + whole_bytes + sizeof(double) + (5 * whole_bytes);

/// --einsums:graph:tiling-budget-share at @p fraction for one case, restored on the way out.
struct Share {
    double previous{config::get(option::GraphTilingBudgetShare)};

    explicit Share(double fraction) { config::set(option::GraphTilingBudgetShare, fraction); }
    Share(Share const &)            = delete;
    Share &operator=(Share const &) = delete;
    ~Share() { config::set(option::GraphTilingBudgetShare, previous); }
};

/// --einsums:max-memory at @p bytes for one case, restored on the way out. The share is off, so
/// the budget is the only bound; the cases for the share set it themselves.
struct Budget {
    std::string previous{config::get(option::MaxMemory)};
    Share       share{1.0};

    explicit Budget(std::size_t bytes) { config::set(option::MaxMemory, std::to_string(bytes) + "b"); }
    explicit Budget(std::string const &spec) { config::set(option::MaxMemory, spec); }
    Budget(Budget const &)            = delete;
    Budget &operator=(Budget const &) = delete;
    ~Budget() { config::set(option::MaxMemory, previous); }
};

/// Run the pass alone under the cap the option gives, which by default is derived.
std::shared_ptr<cg::passes::AxisTiling> decide_by_default(cg::Graph &graph) {
    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    cg::apply_single_pass(*tiling, graph);
    return tiling;
}

std::vector<std::string> labels_of(cg::Graph const &graph) {
    std::vector<std::string> out;
    for (auto const &node : graph.nodes()) {
        out.push_back(node.label);
    }
    return out;
}

} // namespace

TEST_CASE("AxisTiling derives its cap from the memory budget by default", "[ComputeGraph][Pass][AxisTiling]") {
    CHECK(config::get(option::GraphTilingMemoryCap) < 0);
    CHECK(cg::passes::AxisTiling{}.memory_cap() < 0);
    CHECK(config::get(option::GraphTilingBudgetShare) > 0.0);
    CHECK(config::get(option::GraphTilingBudgetShare) < 1.0);

    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // The budget falls half an intermediate short of the program, so the integral may hold what
    // the budget leaves after everything else: half of itself.
    Budget const budget(program_bytes - (whole_bytes / 2));
    auto const   tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() >= 1);
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(tiling->largest_after() <= whole_bytes / 2);
    CHECK(loop_body(graph) != nullptr);
}

TEST_CASE("AxisTiling leaves a program that fits the memory budget as captured", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);
    auto const before = labels_of(graph);

    Budget const budget(program_bytes + 4096);
    auto const   tiling = decide_by_default(graph);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "fit the memory budget together"));
    // Declined before anything was read or reordered, which is what the default pipeline relies
    // on to carry the pass at no cost to every program that fits.
    CHECK(labels_of(graph) == before);
}

TEST_CASE("AxisTiling says when the tensors besides an intermediate already exceed the budget", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // The factor and the denominator alone are over this, so no slice of any intermediate fits
    // beside them, and tiling the small ones would shrink nothing that matters.
    Budget const budget(std::size_t{2000});
    auto const   tiling = decide_by_default(graph);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "other tensors alone exceed the memory budget"));
}

TEST_CASE("AxisTiling cuts a seed that outweighs the rest into its smallest slices when nothing fits the budget",
          "[ComputeGraph][Pass][AxisTiling]") {
    // H(Q,a,b) = A(Q,a) B(Q,b), then r(a,b) = sum_Q H(Q,a,b) w(Q): the intermediate is 1920 bytes and
    // everything outside the loop about a kilobyte. A budget below the inputs admits no slice, and
    // the pass used to leave H whole, which on the water ladder held a 79 MB intermediate against a
    // 2 MB budget. Cutting the seed that outweighs the rest still brings the peak down to a slice.
    constexpr std::size_t Q  = 8;
    constexpr std::size_t A_ = 6;
    constexpr std::size_t B_ = 5;
    auto const            a  = filled_as<double>("A", {Q, A_}, 3U);
    auto const            b  = filled_as<double>("B", {Q, B_}, 5U);
    auto const            w  = filled_as<double>("w", {Q}, 11U);
    RuntimeTensor<double> r("r", std::vector<std::size_t>{A_, B_});

    cg::Graph graph("best effort");
    auto     &h = graph.scratch_runtime<double>("H", std::vector<std::size_t>{Q, A_, B_});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,b <- Q,a ; Q,b", 0.0, &h, 1.0, a, b);
        cg::einsum("a,b <- Q,a,b ; Q", 0.0, &r, 1.0, h, w);
    }

    Budget const budget(std::size_t{512});
    auto const   tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(declined_because(*tiling, "no slicing fits the memory budget"));
    CHECK(tiling->largest_before() == Q * A_ * B_ * sizeof(double));
    CHECK(tiling->largest_after() < tiling->largest_before());

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();

    RuntimeTensor<double> whole("H_ref", std::vector<std::size_t>{Q, A_, B_});
    RuntimeTensor<double> want("r_ref", std::vector<std::size_t>{A_, B_});
    testing::reference_einsum("Q,a,b <- Q,a ; Q,b", &whole, a, b);
    testing::reference_einsum("a,b <- Q,a,b ; Q", &want, whole, w);
    for (std::size_t i = 0; i < A_; ++i) {
        for (std::size_t j = 0; j < B_; ++j) {
            CHECK_THAT(r(i, j), Catch::Matchers::WithinRel(want(i, j), 1e-12));
        }
    }
}

TEST_CASE("AxisTiling has no budget to derive a cap from when the memory ceiling is off", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    Budget const budget(std::string{"0"});
    auto const   tiling = decide_by_default(graph);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "--einsums:max-memory, which is zero"));
}

TEST_CASE("The default pipeline tiles a program the memory budget cannot hold", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const untiled = untiled_energy(problem);

    // The pipeline's CSE merges the duplicated integral before the pass runs, which is the point
    // of running it after the algebraic rewrites: it budgets the program that will execute. So the
    // budget falls short of the program less that copy.
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);
    Budget const budget(program_bytes - whole_bytes - (whole_bytes / 2));
    auto         manager = cg::PassManager::create_default();
    graph.apply(manager);
    for (auto const &pass : manager.passes()) {
        if (auto const *tiling = dynamic_cast<cg::passes::AxisTiling const *>(pass.get()); tiling != nullptr) {
            for (auto const &[reason, count] : tiling->skip_reasons()) {
                UNSCOPED_INFO("AxisTiling declined: " << reason);
            }
        }
    }
    for (auto const &node : graph.nodes()) {
        UNSCOPED_INFO("node: " << node.label);
    }
    REQUIRE(loop_body(graph) != nullptr);

    problem.energy.data()[0] = 0.0;
    graph.execute();
    CHECK_THAT(problem.energy.data()[0], Catch::Matchers::WithinRel(untiled, 1e-12));
}

TEST_CASE("AxisTiling counts a loop body's budget against every tensor of the tree", "[ComputeGraph][Pass][AxisTiling]") {
    // The body names only the program's own tensors, which fit the budget. The parent also holds
    // a tensor the body never names, and with it the program does not fit: a body that counted
    // only what it names would decline.
    Mp2<double>           p(nocc, nvir, naux);
    RuntimeTensor<double> other = filled("X", {8192}, 3U);
    RuntimeTensor<double> norm("N", std::vector<std::size_t>{1});
    cg::Graph             graph("mp2 in a loop beside a large tensor");
    auto const           &body = capture_mp2_in_loop(graph, p, 2).body;
    {
        cg::CaptureGuard const guard(graph);
        cg::dot_python(&norm, other, other);
    }
    std::size_t const other_bytes = other.size() * sizeof(double) + sizeof(double);

    Budget const budget(program_bytes + other_bytes - (whole_bytes / 2));
    auto const   tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK_FALSE(loop_bodies(body).empty());
    CHECK(tiling->largest_after() <= whole_bytes / 2);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    p.energy.data()[0] = 0.0;
    graph.execute();
    auto const expected = mp2_reference(p);
    CHECK(std::abs(p.energy.data()[0] - expected.value) <= sum_tolerance<double>() * expected.magnitude);
}

namespace {

/// Every byte the tree around @p graph allocates once optimized: each tensor object once, views
/// not at all, block buffers of every emitted loop included.
std::size_t tree_bytes(cg::Graph const &graph) {
    std::set<void const *> objects;
    std::size_t            total = 0;
    auto const             count = [&](cg::Graph const &g) {
        std::set<cg::TensorId> named;
        for (auto const &node : g.nodes()) {
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (cg::TensorId const tid : *ids) {
                    named.insert(g.buffer_of(tid));
                }
            }
        }
        for (auto const &[tid, handle] : g.tensors_map()) {
            if (handle.is_tensor_view || g.buffer_of(tid) != tid || handle.tensor_ptr == nullptr || !named.contains(tid)) {
                continue;
            }
            if (objects.insert(handle.tensor_ptr).second) {
                total += handle.total_bytes();
            }
        }
    };
    count(graph);
    graph.for_each_descendant([&](cg::Graph const &g) { count(g); });
    return total;
}

} // namespace

TEST_CASE("AxisTiling holds every loop's buffers to the memory budget, remainders included", "[ComputeGraph][Pass][AxisTiling]") {
    // Five occupied orbitals cut into blocks leave a remainder, and the remainder loop declares
    // buffers of its own that live beside the full block's for the whole replay. Held to the
    // budget one block at a time, a block that fit alone overran it once its remainder was added.
    Mp2<double> p(5, nvir, naux);
    cg::Graph   graph("mp2 with remainders");
    capture_mp2(graph, p);
    std::size_t const before = tree_bytes(graph);
    std::size_t const four   = p.o * p.v * p.o * p.v * sizeof(double);

    std::size_t const limit = before - (3 * four);
    Budget const      budget(limit);
    auto const        tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() >= 1);
    INFO("sliced " << fmt::format("{}", fmt::join(tiling->axis_letters(), ",")) << " in blocks of "
                   << fmt::format("{}", fmt::join(tiling->block_extents(), ",")) << ", " << tiling->loops() << " loops");
    CHECK(tree_bytes(graph) <= limit);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const   expected = mp2_reference(p);
    double const energy   = replay_mp2(graph, p);
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<double>() * expected.magnitude);
}

TEST_CASE("AxisTiling streams an intermediate over its share of a budget the program fits", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const untiled = untiled_energy(problem);
    cg::Graph    graph("mp2 full axis");
    capture(graph, problem);

    // The program fits four times over, but each intermediate is twice its share.
    Budget const budget(4 * program_bytes);
    Share const  share(static_cast<double>(whole_bytes / 2) / static_cast<double>(4 * program_bytes));
    auto const   tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() >= 1);
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(tiling->largest_after() <= whole_bytes / 2);
    REQUIRE(loop_body(graph) != nullptr);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();
    CHECK_THAT(problem.energy.data()[0], Catch::Matchers::WithinRel(untiled, 1e-12));
}

TEST_CASE("AxisTiling leaves a program alone when it fits and the share is off", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);
    auto const before = labels_of(graph);

    // The same budget and an intermediate as large, with the rule turned off at either end.
    Budget const budget(4 * program_bytes);
    double const fraction = GENERATE(0.0, 1.0);
    Share const  share(fraction);
    auto const   tiling = decide_by_default(graph);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "no intermediate holds more than its share"));
    CHECK(labels_of(graph) == before);
}

TEST_CASE("AxisTiling holds a loop's buffers to their share together, remainders included", "[ComputeGraph][Pass][AxisTiling]") {
    // Five occupied orbitals leave a remainder loop whose buffers live beside the full block's.
    // The share bounds them together, not the largest block alone.
    Mp2<double> p(5, nvir, naux);
    cg::Graph   graph("mp2 with remainders");
    capture_mp2(graph, p);
    std::size_t const before = tree_bytes(graph);
    std::size_t const four   = p.o * p.v * p.o * p.v * sizeof(double);

    // A budget the program fits many times over, whose share is less than one four-index tensor.
    std::size_t const budget_bytes = 16 * before;
    std::size_t const share_bytes  = (3 * four) / 5;
    Budget const      budget(budget_bytes);
    Share const       share(static_cast<double>(share_bytes) / static_cast<double>(budget_bytes));
    auto const        tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() >= 1);
    INFO("sliced " << fmt::format("{}", fmt::join(tiling->axis_letters(), ",")) << " in blocks of "
                   << fmt::format("{}", fmt::join(tiling->block_extents(), ",")) << ", " << tiling->loops() << " loops");
    CHECK(tiling->largest_after() <= share_bytes);
    // The four-index seed left the tree and every loop's buffers came in, together within the
    // share.
    CHECK(tree_bytes(graph) <= before - four + share_bytes);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const   expected = mp2_reference(p);
    double const energy   = replay_mp2(graph, p);
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<double>() * expected.magnitude);
}

TEST_CASE("AxisTiling gives a loop the budget leaves when the program is over its budget", "[ComputeGraph][Pass][AxisTiling]") {
    // A program over its budget streams because it must, so its loop may take what the budget
    // leaves. Held to the share as well, the CCSD tutorial's file-backed stage cut its ladder into
    // blocks of 2 x 4 under a 16 MB budget while the whole replay held 11.6 MB; taking the room,
    // its blocks are 7 x 9 and an iteration takes 53 ms instead of 73. A program that fits keeps
    // the share's cap (the case above).
    Mp2<double> p(5, nvir, naux);
    cg::Graph   graph("mp2 over budget");
    capture_mp2(graph, p);
    std::size_t const before = tree_bytes(graph);
    std::size_t const four   = p.o * p.v * p.o * p.v * sizeof(double);

    // Half a four-index tensor short of the program, and a share a fifth of that.
    std::size_t const budget_bytes = before - (four / 2);
    std::size_t const share_bytes  = four / 10;
    Budget const      budget(budget_bytes);
    Share const       share(static_cast<double>(share_bytes) / static_cast<double>(budget_bytes));
    auto const        tiling = decide_by_default(graph);
    REQUIRE(tiling->num_tiled() >= 1);
    INFO("sliced " << fmt::format("{}", fmt::join(tiling->axis_letters(), ",")) << " in blocks of "
                   << fmt::format("{}", fmt::join(tiling->block_extents(), ",")) << ", largest " << tiling->largest_after());
    CHECK(tiling->largest_after() > share_bytes);
    CHECK(tree_bytes(graph) <= budget_bytes);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    auto const   expected = mp2_reference(p);
    double const energy   = replay_mp2(graph, p);
    CHECK(std::abs(energy - expected.value) <= sum_tolerance<double>() * expected.magnitude);
}

TEST_CASE("AxisTiling does not count a body's tensors against a sibling that runs before it", "[ComputeGraph][Pass][AxisTiling]") {
    // A setup body builds H1 and contracts it into R; the loop after it reads R and builds X, which
    // only the loop names. The setup has finished before the loop starts, so X is not live beside
    // H1. Counted as if it were, X left the setup no room under the budget and H1 was held whole.
    auto const a  = filled("A", {8, 3}, 3U);
    auto const b  = filled("B", {3, 8}, 5U);
    auto const c  = filled("C2", {8, 4}, 7U);
    auto const u  = filled("u", {64}, 11U);
    auto const u2 = filled("u2", {64}, 13U);
    auto const z  = filled("z", {64}, 17U);
    auto const wm = filled("w", {4, 64}, 19U);

    cg::Graph graph("setup then loop");
    auto     &r  = graph.declare_runtime_tensor<double>("R", std::vector<std::size_t>{8, 4});
    auto     &h1 = graph.declare_runtime_tensor<double>("H1", std::vector<std::size_t>{8, 8}, /*intermediate=*/true);
    auto     &x  = graph.declare_runtime_tensor<double>("X", std::vector<std::size_t>{64, 64}, /*intermediate=*/true);
    auto     &y  = graph.declare_runtime_tensor<double>("y", std::vector<std::size_t>{64});
    auto     &q  = graph.declare_runtime_tensor<double>("q", std::vector<std::size_t>{8, 64});
    {
        auto                  &setup = graph.add_setup("build R");
        cg::CaptureGuard const guard(setup);
        cg::einsum("i,j <- i,k ; k,j", &h1, a, b);
        cg::einsum("i,m <- i,j ; j,m", &r, h1, c);
    }
    {
        auto                  &body = graph.add_loop("iterate", 2, [](std::size_t it) { return it + 1 < 2; });
        cg::CaptureGuard const guard(body);
        cg::einsum("a,b <- a ; b", &x, u, u2);
        cg::einsum("a <- a,b ; b", &y, x, z);
        cg::einsum("i,a <- i,m ; m,a", &q, r, wm);
    }

    // The setup alone, H1 held at half: room enough only once H1 is cut, and none at all with the
    // loop's tensors counted beside it.
    std::size_t const              element = sizeof(double);
    std::size_t const              setup   = element * (a.size() + b.size() + c.size() + r.size());
    Budget const                   budget(setup + (h1.size() * element / 2));
    auto const                     tiling   = decide_by_default(graph);
    std::vector<std::string> const streamed = tiling->streamed();
    CHECK(std::ranges::find(streamed, std::string{"H1"}) != streamed.end());
}

// ── The triangle: two exchangeable axes run only the tiles on and above their diagonal ─────────

namespace {

/// The density-fitted particle ladder over an amplitude with a declared symmetry, at extents a
/// test can afford. The amplitude is made to HAVE the symmetry it declares.
template <typename T>
struct LadderProblem {
    std::size_t      o;
    std::size_t      v    = 4;
    std::size_t      naux = 5;
    RuntimeTensor<T> bvv;
    RuntimeTensor<T> tau;
    RuntimeTensor<T> r2;

    /// @p sign +1 builds tau(i,j,e,f) = tau(j,i,f,e); -1 builds tau antisymmetric in (i,j) and in
    /// (e,f) separately, as a spin-orbital amplitude is. Zero leaves it general and undeclared.
    LadderProblem(std::size_t occupied, int sign, std::size_t virt = 4, std::size_t aux = 5)
        : o{occupied}, v{virt}, naux{aux}, bvv{filled_as<T>("Bvv", {naux, v, v}, 11U)}, tau{"Tau", std::vector<std::size_t>{o, o, v, v}},
          r2{filled_as<T>("r2", {o, o, v, v}, 17U)} {
        auto const raw = filled_as<T>("raw", {o, o, v, v}, 13U);
        for (std::size_t i = 0; i < o; ++i)
            for (std::size_t j = 0; j < o; ++j)
                for (std::size_t e = 0; e < v; ++e)
                    for (std::size_t f = 0; f < v; ++f) {
                        if (sign > 0) {
                            tau(i, j, e, f) = raw(i, j, e, f) + raw(j, i, f, e);
                        } else if (sign < 0) {
                            tau(i, j, e, f) = raw(i, j, e, f) - raw(j, i, e, f) - raw(i, j, f, e) + raw(j, i, f, e);
                        } else {
                            tau(i, j, e, f) = raw(i, j, e, f);
                        }
                    }
        if (sign > 0) {
            tau.set_symmetry(SymmetryDescriptor{{SymmetryOp::group_swap({0, 2}, {1, 3})}});
        } else if (sign < 0) {
            tau.set_symmetry(SymmetryDescriptor{{SymmetryOp::swap(0, 1, -1), SymmetryOp::swap(2, 3, -1)}});
        }
    }

    /// r2 = c_pf r2 + the ladder, by plain loops.
    RuntimeTensor<T> expected(T c_pf) const {
        RuntimeTensor<T> out = r2;
        RuntimeTensor<T> half("Href", std::vector<std::size_t>{naux, v, v, o, o});
        half.zero();
        testing::reference_einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        testing::reference_einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", c_pf, &out, T{1}, half, bvv);
        return out;
    }

    void capture(cg::Graph &graph, T c_pf) {
        auto                  &half = graph.scratch_runtime<T>("H", std::vector<std::size_t>{naux, v, v, o, o});
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", &half, bvv, tau);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", c_pf, &r2, T{1}, half, bvv);
    }

    /// The cap that holds a tile of the intermediate and not the whole of it.
    [[nodiscard]] std::int64_t cap() const { return static_cast<std::int64_t>(naux * v * v * o * o * sizeof(T) / 2); }
};

/// Run the default pipeline over a graph AxisTiling has already rewritten, which materializes the
/// buffers the emitted bodies declared, then replay it.
void materialize_and_execute(cg::Graph &graph) {
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();
}

/// AxisTiling at @p cap with its triangle block pinned to @p block.
std::shared_ptr<cg::passes::AxisTiling> decide_pinned(cg::Graph &graph, std::int64_t cap, std::int64_t block) {
    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(cap);
    tiling->set_block_extent(block);
    cg::apply_single_pass(*tiling, graph);
    return tiling;
}

template <typename T>
void require_close(RuntimeTensor<T> const &got, RuntimeTensor<T> const &want, double terms) {
    double scale = 0.0;
    for (std::size_t n = 0; n < want.size(); ++n) {
        scale = std::max(scale, static_cast<double>(std::abs(want.data()[n])));
    }
    for (std::size_t n = 0; n < want.size(); ++n) {
        INFO("element " << n);
        REQUIRE(std::abs(got.data()[n] - want.data()[n]) <= sum_tolerance<T>() * terms * scale);
    }
}

} // namespace

TEMPLATE_TEST_CASE("AxisTiling runs the ladder over a triangle of tiles when tau is declared symmetric",
                   "[ComputeGraph][Pass][AxisTiling][Triangle]", float, double, std::complex<double>) {
    using T = TestType;
    // o = 6 in blocks of 2 divides (three diagonal tiles, three above); o = 7 in blocks of 3 leaves
    // a remainder of one, which brings in the last-column and remainder-diagonal loops.
    auto const [o, block, iterations, loops] =
        GENERATE(table<std::size_t, std::int64_t, std::size_t, std::size_t>({{6, 2, 6, 2}, {7, 3, 6, 4}}));
    // Accumulating onto r2, and overwriting an r2 full of NaN, which every element must lose.
    T const c_pf = GENERATE(T{1}, T{0});

    LadderProblem<T> problem(o, +1);
    if (c_pf == T{0}) {
        for (std::size_t n = 0; n < problem.r2.size(); ++n) {
            problem.r2.data()[n] = T{std::numeric_limits<RemoveComplexT<T>>::quiet_NaN()};
        }
    }
    auto const want = problem.expected(c_pf);

    cg::Graph graph("triangle_ladder");
    problem.capture(graph, c_pf);
    auto const tiling = decide_pinned(graph, problem.cap(), block);
    INFO("o " << o << ", block " << block << ", c_pf " << c_pf);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "triangle");
    CHECK(tiling->triangle_sign() == +1);
    CHECK(tiling->iterations() == iterations);
    CHECK(tiling->loops() == loops);

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling mirrors an antisymmetric ladder with the opposite sign", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // Spin-orbital tau, antisymmetric in (i,j): exchanging i and j alone negates the ladder.
    LadderProblem<double> problem(6, -1);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("antisymmetric_ladder");
    problem.capture(graph, 1.0);
    auto const tiling = decide_pinned(graph, problem.cap(), 2);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "triangle");
    CHECK(tiling->triangle_sign() == -1);

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling runs every tile when nothing declares tau's symmetry", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    LadderProblem<double> problem(6, 0);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("undeclared_ladder");
    problem.capture(graph, 1.0);
    auto const tiling = decide_pinned(graph, problem.cap(), 2);
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "range");
    CHECK(tiling->triangle_sign() == 0);
    CHECK(declined_because(*tiling, "not provably unchanged"));

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling's triangle refuses to run once tau no longer has its declared symmetry",
          "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // The declaration is a promise the triangle computes half its tiles from, so the loop checks it
    // on every replay and a broken one stops the run rather than giving a wrong residual.
    LadderProblem<double> problem(6, +1);

    cg::Graph graph("broken_promise");
    problem.capture(graph, 1.0);
    auto const tiling = decide_pinned(graph, problem.cap(), 2);
    REQUIRE(tiling->emission() == "triangle");

    materialize_and_execute(graph); // Holds.
    problem.tau(0, 1, 2, 3) += 1.0;
    REQUIRE_THROWS_AS(graph.execute(), std::runtime_error);
}

TEST_CASE("AxisTiling runs every tile when no triangle block fits beside the mirror's scratch",
          "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // A cap that holds one occupied pair of the intermediate and no two-by-two tile of it: the
    // exchange is provable, but the smallest triangle is over the cap, so the pass takes points
    // and says that it was the fit, not the proof, that decided.
    LadderProblem<double> problem(6, +1);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("triangle_too_big");
    problem.capture(graph, 1.0);
    auto const tiling = decide(graph, static_cast<std::int64_t>(problem.naux * problem.v * problem.v * sizeof(double)));
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() != "triangle");
    CHECK(tiling->triangle_sign() == 0);
    CHECK(declined_because(*tiling, "no block of two or more fits"));

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling runs a ladder that fits as a triangle when the triangle pays", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // Nothing here is over any cap, so nothing is streamed; the exchange alone makes the region a
    // seed. At these extents each kernel does enough arithmetic that halving the tiles saves more
    // than the extra kernels and the mirror cost, whatever block the cost model picks.
    LadderProblem<double> problem(8, +1, 16, 32);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("fitting_ladder");
    problem.capture(graph, 1.0);
    std::shared_ptr<cg::passes::AxisTiling> tiling;
    SECTION("under a cap the caller set") {
        tiling = decide(graph, std::int64_t{1} << 40);
        CHECK(declined_because(*tiling, "already fits the cap"));
    }
    SECTION("under the memory budget") {
        tiling = std::make_shared<cg::passes::AxisTiling>();
        cg::apply_single_pass(*tiling, graph);
        CHECK(declined_because(*tiling, "fit the memory budget"));
    }
    REQUIRE(tiling->num_tiled() == 1);
    CHECK(tiling->emission() == "triangle");
    CHECK(tiling->triangle_sign() == +1);

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling leaves a ladder that fits alone when its triangle does not pay", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // Three occupied orbitals take blocks of two: a full diagonal tile, a mirrored edge and a
    // remainder diagonal, three tiles of near-empty kernels against one. The launch overhead the
    // cost model charges per kernel decides, so the region runs as captured.
    LadderProblem<double> problem(3, +1);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("cheap_ladder");
    problem.capture(graph, 1.0);
    auto const tiling = decide(graph, std::int64_t{1} << 40);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "no cheaper than running it whole"));

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling picks its triangle block from what the tile's layout costs", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    // The CCSD particle ladder over pairs of virtuals: W(a,b,e,f) = B(Q,a,e) B(Q,b,f), then
    // r2(i,j,a,b) += tau(i,j,e,f) W(a,b,e,f). W is symmetric under (a,e) <-> (b,f) and tau is
    // declared so, so the pair (a,b) runs as a triangle. Each tile's W interleaves B's letters, so
    // the engine scatters it in runs of block x block elements.
    //
    // The pass priced tiles by their arithmetic alone, at a rate that was flat across every block
    // it could pick, and the only difference left was the triangle's redundant diagonal: the
    // smallest block won however slowly its short runs wrote. Priced against two hand-built
    // profiles, the choice must follow the scatter's cost: the smallest block when it is flat,
    // a larger one when short runs are dear.
    constexpr std::size_t o = 2, v = 12, naux = 6;
    auto const            bvv = filled_as<double>("Bvv", {naux, v, v}, 11U);
    auto const            raw = filled_as<double>("raw", {o, o, v, v}, 13U);
    RuntimeTensor<double> tau("Tau", std::vector<std::size_t>{o, o, v, v});
    for (std::size_t i = 0; i < o; ++i)
        for (std::size_t j = 0; j < o; ++j)
            for (std::size_t e = 0; e < v; ++e)
                for (std::size_t f = 0; f < v; ++f)
                    tau(i, j, e, f) = raw(i, j, e, f) + raw(j, i, f, e);
    tau.set_symmetry(SymmetryDescriptor{{SymmetryOp::group_swap({0, 2}, {1, 3})}});
    RuntimeTensor<double> const r2_start = filled_as<double>("r2", {o, o, v, v}, 17U);

    RuntimeTensor<double> want = r2_start;
    {
        RuntimeTensor<double> w_ref("Wref", std::vector<std::size_t>{v, v, v, v});
        testing::reference_einsum("a,b,e,f <- Q,a,e ; Q,b,f", &w_ref, bvv, bvv);
        testing::reference_einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", 1.0, &want, 1.0, tau, w_ref);
    }

    // Arithmetic at one rate whatever the shape, nothing for a kernel call or a packed panel, and
    // memory too fast to matter: only the scatter separates the blocks.
    auto const profile = [](std::vector<cg::RunCostPoint> scatter) {
        cg::CostModel model;
        model.cpu.gemm_efficiency           = {{.M = 64, .N = 64, .K = 64, .gflops = 50.0}};
        model.cpu.kernel_launch_overhead_us = 0.0;
        model.cpu.mem_bandwidth_gbps        = 1e6;
        model.cpu.scatter_cost              = std::move(scatter);
        return model;
    };
    auto const tile = [&](cg::CostModel model) {
        RuntimeTensor<double> r2 = r2_start;
        cg::Graph             graph("pair_ladder");
        {
            auto                  &w = graph.scratch_runtime<double>("W", std::vector<std::size_t>{v, v, v, v});
            cg::CaptureGuard const guard(graph);
            cg::einsum("a,b,e,f <- Q,a,e ; Q,b,f", &w, bvv, bvv);
            cg::einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", 1.0, &r2, 1.0, tau, w);
        }
        auto tiling = std::make_shared<cg::passes::AxisTiling>(std::move(model));
        tiling->set_memory_cap(std::int64_t{1} << 40);
        cg::apply_single_pass(*tiling, graph);
        REQUIRE(tiling->num_tiled() == 1);
        REQUIRE(tiling->emission() == "triangle");
        materialize_and_execute(graph);
        require_close(r2, want, static_cast<double>(naux * v));
        return tiling->block_extents().front();
    };

    std::int64_t const flat = tile(profile({{.run = 1, .ns_per_element = 1.0}, {.run = 1024, .ns_per_element = 1.0}}));
    // A tile of block b writes W in runs of b x b and r2 in runs of o x o x b: free from 24 up,
    // so blocks of 6 write both for nothing while blocks of 2 pay 20 ns an element.
    std::int64_t const steep = tile(profile({{.run = 4, .ns_per_element = 20.0}, {.run = 24, .ns_per_element = 0.0}}));
    CHECK(flat == 2);
    CHECK(steep > flat);
}

TEST_CASE("AxisTiling streams the seed alone when taking in what its labels reach leaves nothing to slice",
          "[ComputeGraph][Pass][AxisTiling]") {
    // The CCSD update cycle in miniature: W(a,e,b,f) = B(Q,a,e) B(Q,b,f), tau = t,
    // r(i,j,a,b) = tau(i,j,e,f) W(a,e,b,f), and t += r with each electron's indices exchanged.
    // Sliced on W's e and f, the labels run tau -> t -> r and back into W's a and b, where they
    // meet the e and f they started from. The region takes in every intermediate over the cap
    // that the labels reach, so with a cap below tau's size it took in the whole cycle and no
    // labelling held: on the CCSD tutorial a budget the other tensors nearly filled left the seed
    // a 13 KB cap, and the 63 MB ladder intermediate stayed whole, four times over the budget.
    // Taking a tensor in is a choice, since one left out stays whole and is written through
    // slices; the region falls back to the seed and its own readers and writers.
    constexpr std::size_t o = 2, v = 6, naux = 3;
    auto const            bvv = filled_as<double>("Bvv", {naux, v, v}, 21U);
    auto const            t0  = filled_as<double>("t0", {o, o, v, v}, 23U);

    // Reference, by plain loops.
    RuntimeTensor<double> want = t0;
    {
        RuntimeTensor<double> w_ref("Wref", std::vector<std::size_t>{v, v, v, v});
        RuntimeTensor<double> r_ref("Rref", std::vector<std::size_t>{o, o, v, v});
        w_ref.zero();
        r_ref.zero();
        testing::reference_einsum("a,e,b,f <- Q,a,e ; Q,b,f", &w_ref, bvv, bvv);
        testing::reference_einsum("i,j,a,b <- i,j,e,f ; a,e,b,f", &r_ref, t0, w_ref);
        testing::reference_permute("j,i,b,a <- i,j,a,b", 1.0, &want, 1.0, r_ref);
    }

    RuntimeTensor<double> out("out", std::vector<std::size_t>{o, o, v, v});
    cg::Graph             graph("update_cycle");
    {
        auto                  &t   = graph.scratch_runtime<double>("t", std::vector<std::size_t>{o, o, v, v});
        auto                  &tau = graph.scratch_runtime<double>("tau", std::vector<std::size_t>{o, o, v, v});
        auto                  &w   = graph.scratch_runtime<double>("W", std::vector<std::size_t>{v, v, v, v});
        auto                  &r   = graph.scratch_runtime<double>("r", std::vector<std::size_t>{o, o, v, v});
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, t0, 0.0, &t);
        cg::axpby(1.0, t, 0.0, &tau);
        cg::einsum("a,e,b,f <- Q,a,e ; Q,b,f", &w, bvv, bvv);
        cg::einsum("i,j,a,b <- i,j,e,f ; a,e,b,f", &r, tau, w);
        cg::permute("j,i,b,a <- i,j,a,b", 1.0, &t, 1.0, r);
        cg::axpby(1.0, t, 0.0, &out);
    }

    // Below tau, t and r (1152 bytes each) and well above a slice of W.
    auto const  tiling = decide(graph, 1000);
    std::string reasons;
    for (auto const &[reason, count] : tiling->skip_reasons()) {
        reasons += reason + "; ";
    }
    CAPTURE(reasons);
    REQUIRE(tiling->num_tiled() >= 1);
    CHECK(declined_because(*tiling, "holds the others whole"));

    materialize_and_execute(graph);
    require_close(out, want, static_cast<double>(naux * v * v));
}

TEST_CASE("AxisTiling leaves a ladder that fits alone when nothing proves the exchange", "[ComputeGraph][Pass][AxisTiling][Triangle]") {
    LadderProblem<double> problem(8, 0, 16, 32);
    auto const            want = problem.expected(1.0);

    cg::Graph graph("undeclared_fitting_ladder");
    problem.capture(graph, 1.0);
    auto const tiling = decide(graph, std::int64_t{1} << 40);
    CHECK(tiling->num_tiled() == 0);
    CHECK(declined_because(*tiling, "already fits the cap"));

    materialize_and_execute(graph);
    require_close(problem.r2, want, static_cast<double>(problem.naux * problem.v));
}

TEST_CASE("AxisTiling builds a body from letters no spec may hold", "[ComputeGraph][Pass][AxisTiling]") {
    // A pass that lowers its algebra builds nodes straight from the algebra's letters, which name
    // an elementwise op's axes by position ('#0') and a renamed summed letter '~0'. The body this
    // pass emitted spelled each contraction as text for the capture to parse back, and the parser
    // refused those letters: "output index '#0' has a non-letter character". The body now hands the
    // capture the index lists themselves, so the letters never meet the parser.
    std::size_t const     n = 6;
    std::size_t const     k = 5;
    auto const            a = filled_as<double>("A", {n, k}, 3U);
    auto const            b = filled_as<double>("B", {k, n}, 5U);
    auto const            w = filled_as<double>("w", {n}, 7U);
    RuntimeTensor<double> e("e", std::vector<std::size_t>{n});
    e.zero();

    cg::Graph graph("pass_built_letters");
    auto     &half = graph.scratch_runtime<double>("H", std::vector<std::size_t>{n, n});
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &half, a, b);
    }
    auto const           h_id = graph.find_tensor_by_ptr(&half)->id;
    auto const           w_id = graph.register_tensor(cg::make_handle(w, 0));
    auto const           e_id = graph.register_tensor(cg::make_handle(e, 0));
    cg::ParsedEinsumSpec spec;
    spec.c_indices = {"#0"};
    spec.a_indices = {"#0", "~0"};
    spec.b_indices = {"~0"};
    spec.raw       = spec.render();
    graph.add_node(graph.make_einsum_node(h_id, w_id, e_id, spec, 0.0, 1.0));

    auto const tiling = decide(graph, static_cast<std::int64_t>(n * n * sizeof(double) / 2));
    REQUIRE(tiling->num_tiled() == 1);
    materialize_and_execute(graph);

    RuntimeTensor<double> h_want("Href", std::vector<std::size_t>{n, n});
    h_want.zero();
    testing::reference_einsum("i,j <- i,k ; k,j", &h_want, a, b);
    RuntimeTensor<double> want("want", std::vector<std::size_t>{n});
    want.zero();
    testing::reference_einsum("i <- i,j ; j", &want, h_want, w);
    require_close(e, want, static_cast<double>(n * k));
}
