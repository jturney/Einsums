//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkOuterProduct.cpp
/// @brief Where the packed engine starts beating the generic loop on a
///        contraction with NO link index.
///
/// An outer product has nothing to contract, so no GEMM, GEMV or GER path in
/// StringDispatch's ladder can express it once the output is rank-3 or higher -
/// those helpers are all rank-2-or-less. PackedGemm can (it synthesizes K=1),
/// but it declines below a size threshold and lets the generic nested loop keep
/// the shape. This benchmark is what that threshold should be read from.
///
/// The two specs measured are the CCSD `t1*t1` terms, and they differ in a way
/// that matters: in `ijab <- ib ; ja` the second operand's letters are adjacent
/// in C, while in `ijab <- ia ; jb` the two operands' letters alternate. Same
/// flops, same output, different memory walk - so the crossover is measured for
/// both rather than assumed to be one number.
///
/// Run just this: ctest -R BenchmarkOuterProduct
/// The threshold under test lives in EinsumPackedGemm.hpp
/// (`defer_small_outer_to_generic`); a "declined" line below means the size is
/// under it and the generic loop ran.

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <fmt/format.h>

#include <array>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;
namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

namespace {

/// One `ijab <- <a_idx> ; <b_idx>` case: the operands, the output, and the
/// parsed spec both routes are handed.
struct OuterCase {
    RuntimeTensor<double>    A, B, C;
    cg::ParsedEinsumSpec     parsed;
    std::vector<std::string> links; // empty by construction; that IS the case
};

/// @param a_idx  A's two letters, drawn from {i,j,a,b}.
/// @param b_idx  B's two letters: the other two.
OuterCase make_case(std::vector<std::string> const &a_idx, std::vector<std::string> const &b_idx, size_t n_occ, size_t n_vir) {
    auto extent = [&](std::string const &s) { return (s == "i" || s == "j") ? n_occ : n_vir; };

    OuterCase c{.A      = RuntimeTensor<double>("A", std::vector<size_t>{extent(a_idx[0]), extent(a_idx[1])}),
                .B      = RuntimeTensor<double>("B", std::vector<size_t>{extent(b_idx[0]), extent(b_idx[1])}),
                .C      = RuntimeTensor<double>("C", std::vector<size_t>{n_occ, n_occ, n_vir, n_vir}),
                .parsed = {},
                .links  = {}};
    c.parsed.c_indices = {"i", "j", "a", "b"};
    c.parsed.a_indices = a_idx;
    c.parsed.b_indices = b_idx;

    // Deterministic, non-trivial values: a zero operand would let a lazy path
    // skip work the real one does.
    for (size_t k = 0; k < c.A.size(); k++) {
        c.A.data()[k] = 1.0 + 0.5 * static_cast<double>(k % 13);
    }
    for (size_t k = 0; k < c.B.size(); k++) {
        c.B.data()[k] = 2.0 - 0.25 * static_cast<double>(k % 7);
    }
    c.C.zero();
    return c;
}

/// The packed engine, called exactly as StringDispatch calls it.
bool run_packed(OuterCase &c) {
    packed_gemm::ContractionSpec spec;
    spec.c_indices      = c.parsed.c_indices;
    spec.a_indices      = c.parsed.a_indices;
    spec.b_indices      = c.parsed.b_indices;
    spec.link_indices   = c.links;
    spec.target_indices = c.parsed.c_indices;
    spec.all_indices    = c.parsed.c_indices;
    return packed_gemm::try_packed_gemm<RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(spec, 1.0, &c.C, 1.0, c.A,
                                                                                                             c.B);
}

void bench_one(std::vector<std::string> const &a_idx, std::vector<std::string> const &b_idx, size_t n_occ, size_t n_vir) {
    auto         c        = make_case(a_idx, b_idx, n_occ, n_vir);
    size_t const elements = c.C.size();
    auto const   name     = fmt::format("ijab <- {} ; {}", fmt::join(a_idx, ""), fmt::join(b_idx, ""));

    // Reps scaled so a small case is not measured through timer noise and a
    // large one does not dominate the suite.
    int const reps = elements > (1u << 22) ? 3 : (elements > (1u << 18) ? 10 : (elements > (1u << 12) ? 50 : 2000));

    auto t_generic = time_us(
        "generic", [&]() { cgd::generic_string_einsum(c.parsed, c.links, 1.0, &c.C, 1.0, c.A, c.B); }, reps);

    bool const packed_ran = run_packed(c);
    auto       t_packed   = time_us(
        "packed", [&]() { (void)run_packed(c); }, reps);

    double const per_element_generic = t_generic.avg * 1e3 / static_cast<double>(elements);
    if (packed_ran) {
        fmt::println("[OuterProduct {:>18}] {:>10} elems  generic {:9.2f} us ({:.2f} ns/elem)  packed {:9.2f} us  -> {:.2f}x", name,
                     elements, t_generic.avg, per_element_generic, t_packed.avg, t_generic.avg / t_packed.avg);
        publish_benchmark_result(name.c_str(), "t_packed", static_cast<int>(elements), t_packed);
    } else {
        fmt::println("[OuterProduct {:>18}] {:>10} elems  generic {:9.2f} us ({:.2f} ns/elem)  packed DECLINED (under threshold)", name,
                     elements, t_generic.avg, per_element_generic);
    }
    publish_benchmark_result(name.c_str(), "t_generic", static_cast<int>(elements), t_generic);
}

// (occupied, virtual) pairs spanning the CCSD toy model up to a production-sized
// one: 16 output elements through 14.7M, spanning the crossover in both directions.
constexpr std::array<std::pair<size_t, size_t>, 9> kSizes{
    {{2, 2}, {4, 4}, {6, 8}, {8, 12}, {10, 16}, {20, 24}, {30, 32}, {40, 48}, {60, 64}}};

} // namespace

EINSUMS_TEST_CASE("Bench OuterProduct: adjacent operand letters (ijab <- ib ; ja)", "[ComputeGraph][OuterProduct][benchmark]") {
    LabeledSection0();
    for (auto const &[n_occ, n_vir] : kSizes) {
        bench_one({"i", "b"}, {"j", "a"}, n_occ, n_vir);
    }
}

EINSUMS_TEST_CASE("Bench OuterProduct: alternating operand letters (ijab <- ia ; jb)", "[ComputeGraph][OuterProduct][benchmark]") {
    LabeledSection0();
    for (auto const &[n_occ, n_vir] : kSizes) {
        bench_one({"i", "a"}, {"j", "b"}, n_occ, n_vir);
    }
}
