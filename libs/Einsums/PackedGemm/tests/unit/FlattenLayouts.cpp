//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Layout independence of the multi-K flatten path.
//
// The flatten route copies A and B into flat M*K / K*N buffers, either with
// HPTT or with a scalar gather. HPTT takes no stride vector - its sizes[0] is
// the fastest-varying axis - so the choice used to be made by a hand-rolled
// column-major contiguity test, which sent every row-major operand to the
// gather even though the same bytes in ascending-stride order describe the
// tensor perfectly well. These tests pin the route by layout: the same
// contraction, over byte-identical operands, expressed column-major and
// row-major, must take HPTT both ways and agree; a genuinely non-dense view
// must still fall to the gather.
//
// The shape is the benchmark's "ab <- acd ; dbc": M = a, N = b, and two link
// indices that are adjacent in A but split in B, so they cannot coalesce into
// a single K and the contraction stays on the multi-K flatten path.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {
constexpr size_t NA = 24, NB = 20, NC = 12, ND = 16;

/// The operands in the benchmark's column-major order. Every tensor below is
/// memcpy'd from these, which is what makes the native and the reversed
/// row-major runs the same bytes in the same order.
struct Operands {
    std::vector<double> a = std::vector<double>(NA * NC * ND);
    std::vector<double> b = std::vector<double>(ND * NB * NC);

    Operands() {
        std::mt19937                           gen(20260913);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto &v : a) {
            v = dist(gen);
        }
        for (auto &v : b) {
            v = dist(gen);
        }
    }

    double a_at(size_t ia, size_t ic, size_t id) const { return a[ia + ic * NA + id * NA * NC]; }
    double b_at(size_t id, size_t ib, size_t ic) const { return b[id + ib * ND + ic * ND * NB]; }
};

/// C[a,b] = sum_{c,d} A[a,c,d] B[d,b,c], plus sum|terms| per element.
///
/// The packed path accumulates the same NC*ND products in a different order,
/// so the comparison has to allow the reassociation bound, which is set by the
/// magnitude of the TERMS rather than of the result (see the same reasoning in
/// BatchedMultiK.cpp).
struct Reference {
    std::vector<double> value = std::vector<double>(NA * NB);
    std::vector<double> mag   = std::vector<double>(NA * NB);

    explicit Reference(Operands const &ops) {
        for (size_t ia = 0; ia < NA; ia++) {
            for (size_t ib = 0; ib < NB; ib++) {
                double sum = 0.0, m = 0.0;
                for (size_t ic = 0; ic < NC; ic++) {
                    for (size_t id = 0; id < ND; id++) {
                        double const term = ops.a_at(ia, ic, id) * ops.b_at(id, ib, ic);
                        sum += term;
                        m += std::abs(term);
                    }
                }
                value[ia + ib * NA] = sum;
                mag[ia + ib * NA]   = m;
            }
        }
    }
};

double reassociation_tol(double term_magnitude) {
    // 32x headroom over the strict n*eps*scale bound (n = NC*ND = 192).
    return 32.0 * static_cast<double>(NC * ND) * std::numeric_limits<double>::epsilon() * term_magnitude;
}

/// Compare a result laid out column-major over (a, b) against the reference.
void check_against(Reference const &ref, double const *c) {
    for (size_t e = 0; e < NA * NB; e++) {
        REQUIRE_THAT(c[e], Catch::Matchers::WithinRel(ref.value[e], 1e-12) ||
                               Catch::Matchers::WithinAbs(ref.value[e], reassociation_tol(ref.mag[e])));
    }
}

RuntimeTensor<double> filled(std::vector<size_t> const &dims, bool row_major, std::vector<double> const &src) {
    RuntimeTensor<double> t(std::string("op"), dims, row_major);
    std::memcpy(t.data(), src.data(), src.size() * sizeof(double));
    return t;
}
} // namespace

TEST_CASE("Flatten path: column-major operands take HPTT", "[PackedGemm][FlattenLayouts]") {
    Operands const  ops;
    Reference const ref(ops);

    auto A = filled({NA, NC, ND}, /*row_major=*/false, ops.a);
    auto B = filled({ND, NB, NC}, /*row_major=*/false, ops.b);
    auto C = RuntimeTensor<double>(std::string("C"), std::vector<size_t>{NA, NB}, /*row_major=*/false);

    cg::einsum("acd;dbc->ab", &C, A, B);

    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "flatten_gemm_hptt");
    check_against(ref, C.data());
}

TEST_CASE("Flatten path: row-major operands take HPTT too", "[PackedGemm][FlattenLayouts]") {
    // The same bytes, described the way the benchmark harness describes them:
    // a row-major tensor over reversed dims, with the labels reversed to
    // match. This is the case the old column-major contiguity test rejected.
    Operands const  ops;
    Reference const ref(ops);

    auto A = filled({ND, NC, NA}, /*row_major=*/true, ops.a);
    auto B = filled({NC, NB, ND}, /*row_major=*/true, ops.b);
    auto C = RuntimeTensor<double>(std::string("C"), std::vector<size_t>{NB, NA}, /*row_major=*/true);

    cg::einsum("dca;cbd->ba", &C, A, B);

    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "flatten_gemm_hptt");
    // C is row-major over (b, a), which is column-major over (a, b): the same
    // bytes the reference is written in.
    check_against(ref, C.data());
}

TEST_CASE("Flatten path: a non-dense view still gathers", "[PackedGemm][FlattenLayouts]") {
    // A view that drops the tail of a middle axis is dense in NO axis order,
    // so HPTT cannot describe it without an outer-size argument the wrapper
    // does not plumb through. The gather has to pick it up, and be right.
    Operands const  ops;
    Reference const ref(ops);

    auto A_big = RuntimeTensor<double>(std::string("A_big"), std::vector<size_t>{NA, NC + 3, ND}, /*row_major=*/false);
    for (size_t id = 0; id < ND; id++) {
        for (size_t ic = 0; ic < NC; ic++) {
            std::memcpy(A_big.data() + ic * NA + id * NA * (NC + 3), ops.a.data() + ic * NA + id * NA * NC, NA * sizeof(double));
        }
    }

    auto B = filled({ND, NB, NC}, /*row_major=*/false, ops.b);
    auto C = RuntimeTensor<double>(std::string("C"), std::vector<size_t>{NA, NB}, /*row_major=*/false);

    RuntimeTensorView<double> A = A_big(AllT{}, Range{0, NC}, AllT{});
    cg::einsum("acd;dbc->ab", &C, A, B);

    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "flatten_gemm_gather");
    check_against(ref, C.data());
}
