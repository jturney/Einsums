//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DirectApplyRecursion.cpp
/// @brief Every pass that opts into sub-graph recursion, applied directly with
///        ``Graph::apply<P>()`` to a graph whose work lives in a loop body and a conditional.
///
/// ``Graph::apply<P>()`` reaches loop bodies and branches through the same driver as
/// ``PassManager`` (``run_pass_tree``). RecursionPlumbing.cpp pins that the driver visits the
/// sub-graphs and which passes opt in; nothing there checks that a pass is CORRECT on what it
/// finds inside one. Here the top level holds only a Loop node and a Conditional node, so any
/// rewrite a pass makes happens inside a sub-graph, and the executed numbers are checked
/// against brute-force references (``reference_einsum`` / ``reference_permute``) applied once
/// per iteration.
///
/// The same program is also captured flat, at the top level of a second graph. A pass that
/// rewrites the flat program must rewrite each sub-graph copy the same way: that is what
/// shows the pass reached the body rather than silently stopping at the loop header.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerExpansion.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerFolding.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerLinearity.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryDetection.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetryInference.hpp>
#include <Einsums/ComputeGraph/Passes/FactorizationPass.hpp>
#include <Einsums/ComputeGraph/Passes/LaplaceTransform.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;
using einsums::testing::reference_permute;

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t kI          = 4;
constexpr std::size_t kK          = 3;
constexpr std::size_t kJ          = 5;
constexpr std::size_t kIterations = 3;

using RT = RuntimeTensor<double>;

/// Every tensor the program touches. Inputs and outputs alike are compared after
/// execution, so a pass that writes into a caller's input is caught as well.
struct Operands {
    RT A{create_random_tensor<double>("A", kI, kK)};
    RT B1{create_random_tensor<double>("B1", kK, kJ)};
    RT B2{create_random_tensor<double>("B2", kK, kJ)};
    RT S{create_random_tensor<double>("S", kK, kI)};
    RT M{create_random_tensor<double>("M", kK, kK)};
    RT X{create_random_tensor<double>("X", kI, kJ)};
    RT Y{create_random_tensor<double>("Y", kI, kJ)};
    RT R{create_random_tensor<double>("R", kI, kJ)};
    RT G1{create_random_tensor<double>("G1", kI, kJ)};
    RT G2{create_random_tensor<double>("G2", kI, kJ)};
    RT Z{create_random_tensor<double>("Z", kI, kJ)};
    RT v{create_random_tensor<double>("v", kK)};
    RT W3{create_random_tensor<double>("W3", kK, kI, kI)};
    RT O{create_random_tensor<double>("O", kI, kI)};
    RT TEI{create_random_tensor<double>("TEI", kK, kK, kK, kK)};
    RT Dm{create_random_tensor<double>("Dm", kK, kK)};
    RT J{create_random_tensor<double>("J", kK, kK)};
    RT K{create_random_tensor<double>("K", kK, kK)};

    void                                                           capture(cg::Graph &graph);
    void                                                           reference();
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const;
};

constexpr std::array kMembers = {&Operands::A,  &Operands::B1, &Operands::B2,  &Operands::S,  &Operands::M, &Operands::X,
                                 &Operands::Y,  &Operands::R,  &Operands::G1,  &Operands::G2, &Operands::Z, &Operands::v,
                                 &Operands::W3, &Operands::O,  &Operands::TEI, &Operands::Dm, &Operands::J, &Operands::K};
constexpr std::array kNames   = {"A", "B1", "B2", "S", "M", "X", "Y", "R", "G1", "G2", "Z", "v", "W3", "O", "TEI", "Dm", "J", "K"};

/// Capture the mixed program into @p graph. One statement per pattern a pass exists to rewrite:
///   - a scale followed by an einsum accumulating into the same tensor (ScaleAbsorption),
///   - two einsums sharing operand A and accumulating into R (DistributiveFactoring),
///   - a permute whose only reader is an einsum (PermuteFusion),
///   - two independent GEMM-shaped einsums of one shape (GEMMBatching),
///   - two axpby updates of one pair (ElementWiseFusion),
///   - a two-step chain through a graph-owned intermediate (ContractionPlanning, InplaceOptimization),
///   - the 2J - K transpose pair over one operand (LinearCombinationContractionFolding),
///   - J and K contracted from one four-index stream (StreamContractionFusion).
void capture_program(cg::Graph &graph, Operands &o) {
    auto &P = graph.create_zero_runtime_tensor<double>("P", std::vector<std::size_t>{kI, kK});
    auto &T = graph.create_zero_runtime_tensor<double>("T", std::vector<std::size_t>{kI, kK});

    cg::CaptureGuard const guard(graph);
    cg::scale(3.0, &o.R);
    cg::einsum("ij <- ik ; kj", 1.0, &o.R, 1.0, o.A, o.B1);
    cg::einsum("ij <- ik ; kj", 1.0, &o.R, 1.0, o.A, o.B2);

    cg::permute("ik <- ki", 0.0, &P, 1.0, o.S);
    cg::einsum("ij <- ik ; kj", 0.0, &o.G1, 1.0, P, o.B1);
    cg::einsum("ij <- ik ; kj", 0.0, &o.G2, 1.0, o.A, o.B2);

    cg::axpby(0.5, o.X, 2.0, &o.Y);
    cg::axpby(1.0, o.X, 0.5, &o.Y);

    cg::einsum("ik <- il ; lk", 0.0, &T, 1.0, o.A, o.M);
    cg::einsum("ij <- ik ; kj", 0.0, &o.Z, 1.0, T, o.B1);

    cg::einsum("i,j <- k ; k,i,j", 0.0, &o.O, 2.0, o.v, o.W3);
    cg::einsum("i,j <- k ; k,j,i", 1.0, &o.O, -1.0, o.v, o.W3);

    cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &o.J, 2.0, o.TEI, o.Dm);
    cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &o.K, -1.0, o.TEI, o.Dm);
}

/// One execution of the program, by brute force and by hand.
void reference_once(Operands &o) {
    for (std::size_t n = 0; n < o.R.size(); ++n) {
        o.R.data()[n] *= 3.0;
    }
    reference_einsum("ij <- ik ; kj", 1.0, &o.R, 1.0, o.A, o.B1);
    reference_einsum("ij <- ik ; kj", 1.0, &o.R, 1.0, o.A, o.B2);

    RT P("P", std::vector<std::size_t>{kI, kK});
    reference_permute("ik <- ki", 0.0, &P, 1.0, o.S);
    reference_einsum("ij <- ik ; kj", 0.0, &o.G1, 1.0, P, o.B1);
    reference_einsum("ij <- ik ; kj", 0.0, &o.G2, 1.0, o.A, o.B2);

    for (std::size_t n = 0; n < o.Y.size(); ++n) {
        double const x    = o.X.data()[n];
        double const once = 0.5 * x + 2.0 * o.Y.data()[n];
        o.Y.data()[n]     = 1.0 * x + 0.5 * once;
    }

    RT T("T", std::vector<std::size_t>{kI, kK});
    reference_einsum("ik <- il ; lk", 0.0, &T, 1.0, o.A, o.M);
    reference_einsum("ij <- ik ; kj", 0.0, &o.Z, 1.0, T, o.B1);

    reference_einsum("ij <- k ; kij", 0.0, &o.O, 2.0, o.v, o.W3);
    reference_einsum("ij <- k ; kji", 1.0, &o.O, -1.0, o.v, o.W3);

    reference_einsum("ij <- ijkl ; kl", 0.0, &o.J, 2.0, o.TEI, o.Dm);
    reference_einsum("ij <- ikjl ; kl", 0.0, &o.K, -1.0, o.TEI, o.Dm);
}

void Operands::capture(cg::Graph &graph) {
    capture_program(graph, *this);
}

void Operands::reference() {
    reference_once(*this);
}

std::vector<std::pair<char const *, RT const *>> Operands::tensors() const {
    std::vector<std::pair<char const *, RT const *>> out;
    for (std::size_t m = 0; m < kMembers.size(); ++m) {
        out.emplace_back(kNames[m], &(this->*kMembers[m]));
    }
    return out;
}

/// ``T1 = A B`` then ``out = T1 C`` with a 100x1 * 1x100 * 100x1 shape, whose cheap
/// parenthesization is the other one, through a graph-owned interior.
struct ChainProgram {
    RT A{create_random_tensor<double>("A", 100, 1)};
    RT B{create_random_tensor<double>("B", 1, 100)};
    RT C{create_random_tensor<double>("C", 100, 1)};
    RT out{create_random_tensor<double>("out", 100, 1)};

    void capture(cg::Graph &graph) {
        auto                  &T1 = graph.create_zero_runtime_tensor<double>("T1", std::vector<std::size_t>{100, 100});
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, T1, C);
    }
    void reference() {
        RT T1("T1", std::vector<std::size_t>{100, 100});
        reference_einsum("ij <- ik ; kj", 0.0, &T1, 1.0, A, B);
        reference_einsum("ij <- ik ; kj", 0.0, &out, 1.0, T1, C);
    }
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const {
        return {{"A", &A}, {"B", &B}, {"C", &C}, {"out", &out}};
    }
};

/// ``R += A B1`` and ``R += A B2``, the two-term sum over a shared operand. Accumulating,
/// so every iteration of the loop shows in the answer.
struct FactorProgram {
    RT A{create_random_tensor<double>("A", kI, kK)};
    RT B1{create_random_tensor<double>("B1", kK, kJ)};
    RT B2{create_random_tensor<double>("B2", kK, kJ)};
    RT R{create_random_tensor<double>("R", kI, kJ)};

    void capture(cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
    }
    void reference() {
        reference_einsum("ij <- ik ; kj", 1.0, &R, 1.0, A, B1);
        reference_einsum("ij <- ik ; kj", 1.0, &R, 1.0, A, B2);
    }
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const {
        return {{"A", &A}, {"B1", &B1}, {"B2", &B2}, {"R", &R}};
    }
};

/// J and K from one four-index operand large enough to clear the stream threshold.
struct StreamProgram {
    static constexpr std::size_t kN = 40;

    RT TEI{create_random_tensor<double>("TEI", kN, kN, kN, kN)};
    RT D{create_random_tensor<double>("D", kN, kN)};
    RT J{create_random_tensor<double>("J", kN, kN)};
    RT K{create_random_tensor<double>("K", kN, kN)};

    void capture(cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J, 2.0, TEI, D);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K, -1.0, TEI, D);
    }
    void reference() {
        reference_einsum("ij <- ijkl ; kl", 0.0, &J, 2.0, TEI, D);
        reference_einsum("ij <- ikjl ; kl", 0.0, &K, -1.0, TEI, D);
    }
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const {
        return {{"TEI", &TEI}, {"D", &D}, {"J", &J}, {"K", &K}};
    }
};

/// Two independent GEMMs of one shape and one set of prefactors.
struct BatchProgram {
    RT A1{create_random_tensor<double>("A1", 8, 6)};
    RT B1{create_random_tensor<double>("B1", 6, 7)};
    RT A2{create_random_tensor<double>("A2", 8, 6)};
    RT B2{create_random_tensor<double>("B2", 6, 7)};
    RT C1{create_random_tensor<double>("C1", 8, 7)};
    RT C2{create_random_tensor<double>("C2", 8, 7)};

    void capture(cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C1, 1.0, A1, B1);
        cg::einsum("ik;kj->ij", 0.0, &C2, 1.0, A2, B2);
    }
    void reference() {
        reference_einsum("ij <- ik ; kj", 0.0, &C1, 1.0, A1, B1);
        reference_einsum("ij <- ik ; kj", 0.0, &C2, 1.0, A2, B2);
    }
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const {
        return {{"A1", &A1}, {"B1", &B1}, {"A2", &A2}, {"B2", &B2}, {"C1", &C1}, {"C2", &C2}};
    }
};

/// ``X = A B``, ``Y = 2 (X * B)`` elementwise, ``out = Y A``: X dies at the elementwise
/// product whose output is aligned with it, so Y can take X's storage.
struct InplaceProgram {
    RT A{create_random_tensor<double>("A", 6, 6)};
    RT B{create_random_tensor<double>("B", 6, 6)};
    RT out{create_random_tensor<double>("out", 6, 6)};

    void capture(cg::Graph &graph) {
        auto                  &X = graph.create_zero_runtime_tensor<double>("X", std::vector<std::size_t>{6, 6});
        auto                  &Y = graph.create_zero_runtime_tensor<double>("Y", std::vector<std::size_t>{6, 6});
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &X, 1.0, A, B);
        cg::direct_product(2.0, X, B, 0.0, &Y);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, Y, A);
    }
    void reference() {
        RT X("X", std::vector<std::size_t>{6, 6});
        RT Y("Y", std::vector<std::size_t>{6, 6});
        reference_einsum("ij <- ik ; kj", 0.0, &X, 1.0, A, B);
        reference_einsum("ij <- ij ; ij", 0.0, &Y, 2.0, X, B);
        reference_einsum("ij <- ik ; kj", 0.0, &out, 1.0, Y, A);
    }
    [[nodiscard]] std::vector<std::pair<char const *, RT const *>> tensors() const { return {{"A", &A}, {"B", &B}, {"out", &out}}; }
};

/// A fast CPU whose GEMM time is dominated by FLOP count, so ContractionPlanning prefers the
/// cheap parenthesization on any machine.
cg::CostModel skewed_model() {
    cg::CostModel model;
    model.cpu.peak_gflops_fp64          = 100.0;
    model.cpu.mem_bandwidth_gbps        = 40.0;
    model.cpu.kernel_launch_overhead_us = 0.1;
    model.cpu.name                      = "TestCPU";
    return model;
}

/// Contractions ruinously slow and memory traffic free, so DistributiveFactoring factors.
cg::CostModel favors_factoring() {
    cg::CostModel model;
    model.cpu.peak_gflops_fp64          = 1e-3;
    model.cpu.mem_bandwidth_gbps        = 1e6;
    model.cpu.kernel_launch_overhead_us = 0.0;
    model.cpu.alloc_overhead_us         = 0.0;
    model.cpu.gemm_efficiency.clear();
    return model;
}

/// Every tensor of @p got within a relative tolerance of the same tensor of @p want.
template <typename Program>
void require_close(Program const &got, Program const &want, std::string_view where) {
    auto const got_tensors  = got.tensors();
    auto const want_tensors = want.tensors();
    REQUIRE(got_tensors.size() == want_tensors.size());
    for (std::size_t m = 0; m < got_tensors.size(); ++m) {
        RT const &g = *got_tensors[m].second;
        RT const &w = *want_tensors[m].second;
        REQUIRE(g.size() == w.size());
        double worst = 0.0;
        for (std::size_t n = 0; n < g.size(); ++n) {
            double const scale = std::max(1.0, std::abs(w.data()[n]));
            worst              = std::max(worst, std::abs(g.data()[n] - w.data()[n]) / scale);
        }
        INFO(where << ": operand " << got_tensors[m].first);
        REQUIRE(worst < 1.0e-11);
    }
}

void require_well_formed(cg::Graph const &graph) {
    auto const problems = graph.verify();
    INFO(fmt::format("{}", fmt::join(problems, "\n")));
    REQUIRE(problems.empty());
}

/// The statistic a pass keeps for the rewrites it made, when it keeps one that counts per
/// site. A running total over the whole sub-graph tree within one apply.
template <typename P>
std::optional<std::size_t> rewrite_count(P const &pass) {
    namespace ps = cg::passes;
    if constexpr (std::is_same_v<P, ps::ScaleAbsorption>) {
        return pass.num_absorbed();
    } else if constexpr (std::is_same_v<P, ps::ElementWiseFusion>) {
        return pass.num_fused();
    } else if constexpr (std::is_same_v<P, ps::PermuteFusion>) {
        return pass.num_rewrites();
    } else if constexpr (std::is_same_v<P, ps::GEMMBatching>) {
        return pass.num_batches();
    } else if constexpr (std::is_same_v<P, ps::InplaceOptimization>) {
        return pass.num_merged();
    } else if constexpr (std::is_same_v<P, ps::DistributiveFactoring>) {
        return pass.num_groups();
    } else if constexpr (std::is_same_v<P, ps::LinearCombinationContractionFolding>) {
        return pass.num_groups();
    } else if constexpr (std::is_same_v<P, ps::ContractionPlanning>) {
        return pass.chains_restructured();
    } else if constexpr (std::is_same_v<P, ps::StreamContractionFusion>) {
        return pass.num_groups();
    } else if constexpr (std::is_same_v<P, ps::ConstantFolding>) {
        return pass.num_folded();
    } else if constexpr (std::is_same_v<P, ps::LayoutAssignment>) {
        return pass.num_relaid_out();
    } else {
        return std::nullopt;
    }
}

/// What the pass did to the program captured flat.
struct FlatOutcome {
    bool                       modified{false};
    std::optional<std::size_t> count;
};

/// Apply @p Pass (constructed from @p args) to @p Program captured flat and to the same
/// program captured in a loop body and both branches of a conditional, then execute both and
/// compare against the brute-force reference. Returns what the pass did to the flat capture,
/// so a caller can require that the comparison was not vacuous.
template <typename Program, typename Pass, typename... Args>
FlatOutcome check_direct_apply(Args const &...args) {
    // The flat program: what this pass does when nothing hides the program from it.
    Program   flat_ops;
    Program   flat_ref = flat_ops;
    cg::Graph flat("direct_apply_flat");
    flat_ops.capture(flat);
    auto [flat_modified, flat_pass] = flat.template apply<Pass>(args...);
    require_well_formed(flat);
    std::size_t const flat_nodes = flat.num_nodes();
    flat.execute();
    flat_ref.reference();
    require_close(flat_ops, flat_ref, "flat");

    // The nested program: the same statements, once in a loop body and once in each branch.
    Program loop_ops, then_ops, else_ops;
    Program loop_ref = loop_ops, then_ref = then_ops, else_ref = else_ops;

    cg::Graph graph("direct_apply_nested");
    auto     &body = graph.add_loop("iters", kIterations + 2, [](std::size_t iter) { return iter + 1 < kIterations; });
    loop_ops.capture(body);
    auto [then_g, else_g] = graph.add_conditional("branch", [] { return true; });
    then_ops.capture(then_g);
    else_ops.capture(else_g);

    auto [modified, pass] = graph.template apply<Pass>(args...);
    require_well_formed(graph);

    // The pass reached every copy, and did to each what it did to the flat program.
    std::vector<cg::Graph const *> subs;
    std::as_const(graph).for_each_subgraph([&subs](cg::Graph const &sub) { subs.push_back(&sub); });
    REQUIRE(subs.size() == 3);
    for (cg::Graph const *sub : subs) {
        INFO("sub-graph " << sub->name());
        CHECK(sub->num_nodes() == flat_nodes);
    }
    CHECK(modified == flat_modified);
    auto const flat_count = rewrite_count(flat_pass);
    if (flat_count.has_value()) {
        // Three copies and nothing at the top level, so a counter kept across the whole tree
        // reads three times the flat count. A driver that stopped at the loop header reads zero,
        // and a counter reset per sub-graph reads one copy's worth.
        INFO("flat count " << *flat_count);
        CHECK(*rewrite_count(pass) == 3 * *flat_count);
    }

    graph.execute();
    for (std::size_t iter = 0; iter < kIterations; ++iter) {
        loop_ref.reference();
    }
    then_ref.reference();
    require_close(loop_ops, loop_ref, "loop body");
    require_close(then_ops, then_ref, "taken branch");
    require_close(else_ops, else_ref, "untaken branch");

    // A replay runs the rewritten bodies again from the state the first run left.
    graph.execute();
    for (std::size_t iter = 0; iter < kIterations; ++iter) {
        loop_ref.reference();
    }
    then_ref.reference();
    require_close(loop_ops, loop_ref, "loop body, replay");
    require_close(then_ops, then_ref, "taken branch, replay");
    require_close(else_ops, else_ref, "untaken branch, replay");

    return {.modified = flat_modified, .count = flat_count};
}

/// The passes whose default instance rewrites the mixed program, so the sweep is not vacuous
/// for them. The others either need a tuned instance (the targeted cases below) or have
/// nothing in a dense double program to act on, where the sweep still pins that they leave a
/// body correct.
template <typename P>
constexpr bool kRewritesMixedProgram =
    std::is_same_v<P, cg::passes::ScaleAbsorption> || std::is_same_v<P, cg::passes::ElementWiseFusion> ||
    std::is_same_v<P, cg::passes::PermuteFusion> || std::is_same_v<P, cg::passes::LinearCombinationContractionFolding> ||
    std::is_same_v<P, cg::passes::Reorder> || std::is_same_v<P, cg::passes::RegionIdentity>;

} // namespace

// Defends: Graph::apply<P>() driving an opt-in pass into a loop body and both branches of a
// conditional, the path a caller takes when applying one pass by hand. Each pass must leave
// every sub-graph well formed, compute the same numbers as the program run by brute force
// (three loop iterations, the taken branch once, the untaken branch not at all), and treat
// each sub-graph copy of the program the way it treats the program captured flat. The list
// is every pass whose recurse_into_subgraphs() is true; the first REQUIRE fails if one opts out.
TEMPLATE_TEST_CASE("Direct apply - an opt-in pass is correct inside a loop body and a conditional",
                   "[ComputeGraph][Recursion][DirectApply]", cg::passes::AntisymmetrizerExpansion, cg::passes::AntisymmetrizerFolding,
                   cg::passes::AntisymmetrizerLinearity, cg::passes::AntisymmetryDetection, cg::passes::AntisymmetryInference,
                   cg::passes::ConstantFolding, cg::passes::ContractionPlanning, cg::passes::CrossSpaceValidation,
                   cg::passes::DeltaElimination, cg::passes::DistributiveFactoring, cg::passes::ElementWiseFusion,
                   cg::passes::FactorizationPass, cg::passes::GEMMBatching, cg::passes::InplaceOptimization, cg::passes::LaplaceTransform,
                   cg::passes::LayoutAssignment, cg::passes::LinearCombinationContractionFolding, cg::passes::MultiTermFactorization,
                   cg::passes::PermuteFusion, cg::passes::ProvenancePropagation, cg::passes::RegionIdentity, cg::passes::Reorder,
                   cg::passes::ScaleAbsorption, cg::passes::ScalingAnalysis, cg::passes::SpacePropagation, cg::passes::StreamAssignment,
                   cg::passes::StreamContractionFusion, cg::passes::SymmetrizedAccumulation, cg::passes::SymmetryPropagation,
                   cg::passes::TiledExpansion, cg::passes::TransferElimination, cg::passes::TransferInsertion) {
    using Pass = TestType;
    REQUIRE(Pass{}.recurse_into_subgraphs());

    FlatOutcome const outcome = check_direct_apply<Operands, Pass>();
    if constexpr (kRewritesMixedProgram<Pass>) {
        CHECK(outcome.modified);
        CHECK(outcome.count.value_or(1) > 0);
    }
}

// Defends: ContractionPlanning restructuring a chain inside a loop body when applied directly.
// The mixed program's chain is too small for the default cost model to reorder, so a tuned
// model is passed through apply's forwarded constructor arguments.
TEST_CASE("Direct apply - ContractionPlanning restructures a chain inside a loop body", "[ComputeGraph][Recursion][DirectApply]") {
    FlatOutcome const outcome = check_direct_apply<ChainProgram, cg::passes::ContractionPlanning>(skewed_model());
    CHECK(outcome.count.value_or(0) == 1);
}

// Defends: DistributiveFactoring rewriting a shared-operand sum inside a loop body when
// applied directly. The sum accumulates, so a rewrite that ran the body the wrong number of
// times, or dropped a term, shows in the answer.
TEST_CASE("Direct apply - DistributiveFactoring factors a sum inside a loop body", "[ComputeGraph][Recursion][DirectApply]") {
    FlatOutcome const outcome = check_direct_apply<FactorProgram, cg::passes::DistributiveFactoring>(favors_factoring());
    CHECK(outcome.count.value_or(0) == 1);
}

// Defends: StreamContractionFusion fusing two contractions over one stream inside a loop
// body. Its fused node is an opaque executor, so the numbers are the only thing that shows it
// read the right operands.
TEST_CASE("Direct apply - StreamContractionFusion fuses a stream inside a loop body", "[ComputeGraph][Recursion][DirectApply]") {
    FlatOutcome const outcome = check_direct_apply<StreamProgram, cg::passes::StreamContractionFusion>();
    CHECK(outcome.count.value_or(0) == 1);
}

// Defends: GEMMBatching forming a batch inside a loop body when applied directly.
TEST_CASE("Direct apply - GEMMBatching batches GEMMs inside a loop body", "[ComputeGraph][Recursion][DirectApply]") {
    FlatOutcome const outcome = check_direct_apply<BatchProgram, cg::passes::GEMMBatching>();
    CHECK(outcome.count.value_or(0) == 1);
}

// Defends: InplaceOptimization merging a body intermediate onto a dying one when applied
// directly. A merge that aliased the wrong buffer, or one that outlived the iteration, would
// corrupt the next iteration's product.
TEST_CASE("Direct apply - InplaceOptimization merges storage inside a loop body", "[ComputeGraph][Recursion][DirectApply]") {
    FlatOutcome const outcome = check_direct_apply<InplaceProgram, cg::passes::InplaceOptimization>();
    CHECK(outcome.count.value_or(0) == 1);
}
