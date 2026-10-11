//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// coalesce_plan merges the M and N groups one tensor at a time.
//
// The shape is a block of the DF-CCSD particle ladder: C(Q,a,f,j) <- A(Q,a,e) ; B(j,e,f), where B is
// a slice of a rank-4 amplitude tensor at fixed i over a range of j. f and j are adjacent in C but
// have e between them in B. Merged jointly they stay two N dims, which sends C through a scratch
// block and a scatter; merged per tensor C sees one N dim and the tile kernel writes it directly,
// while pack_B still gathers B's two.

#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {

namespace pg = einsums::packed_gemm;

template <typename T, size_t Rank>
void check_close(Tensor<T, Rank> const &C, Tensor<T, Rank> const &C_ref, int64_t k) {
    double const tol = 64.0 * static_cast<double>(k) * std::numeric_limits<RemoveComplexT<T>>::epsilon();
    for (size_t idx = 0; idx < C.size(); ++idx) {
        T const got  = C.data()[idx];
        T const want = C_ref.data()[idx];
        REQUIRE(std::isfinite(std::abs(got)));
        REQUIRE(std::abs(got - want) <= tol * (1.0 + std::abs(want)));
    }
}

} // namespace

TEST_CASE("coalesce: N dims adjacent in C but split in B merge in C only", "[PackedGemm][Coalesce]") {
    // Column-major. C(Q,a,f,j) and A(Q,a,e) are dense; B is the (j, e, f) slice at one i of a
    // (ni, no, ne, nf) amplitude tensor, so j strides ni and e sits between j and f.
    struct Extents {
        int64_t nQ, na, ne, nf, nj, ni, no;
    };
    // The benzene extents are the ones that matter: there the pack_B saving of B's own order is
    // tiny next to C, and pricing the N group at its own C stride used to pick B's order and scatter
    // C. The small extents check the same merge where no ordering question is close.
    auto const x = GENERATE(Extents{5, 4, 3, 6, 2, 3, 4}, Extents{420, 93, 93, 93, 3, 21, 21});
    INFO("nQ " << x.nQ << " nj " << x.nj);

    int64_t const   qa = x.nQ * x.na;
    int64_t const   bf = x.ni * x.no * x.ne; // B's f stride
    pg::PackingPlan plan;
    plan.valid       = true;
    plan.m_dims      = {{.tensor_pos = 0, .size = x.nQ, .tensor_stride = 1}, {.tensor_pos = 1, .size = x.na, .tensor_stride = x.nQ}};
    plan.c_m_dims    = plan.m_dims;
    plan.n_dims      = {{.tensor_pos = 0, .size = x.nj, .tensor_stride = x.ni}, {.tensor_pos = 2, .size = x.nf, .tensor_stride = bf}};
    plan.c_n_dims    = {{.tensor_pos = 3, .size = x.nj, .tensor_stride = qa * x.nf}, {.tensor_pos = 2, .size = x.nf, .tensor_stride = qa}};
    plan.k_dims_in_a = {{.tensor_pos = 2, .size = x.ne, .tensor_stride = qa}};
    plan.k_dims_in_b = {{.tensor_pos = 1, .size = x.ne, .tensor_stride = x.ni * x.no}};
    plan.M_total     = qa;
    plan.N_total     = x.nf * x.nj;
    plan.K_total     = x.ne;

    pg::coalesce_plan(plan);

    REQUIRE(plan.coalesced);
    REQUIRE(plan.c_m_dims.size() == 1);
    REQUIRE(plan.m_dims.size() == 1);
    REQUIRE(plan.c_n_dims.size() == 1);
    REQUIRE(plan.c_n_dims[0].size == plan.N_total);
    REQUIRE(plan.c_n_dims[0].tensor_stride == qa);
    REQUIRE(plan.n_dims.size() == 2);

    // Both vectors still enumerate one flat N coordinate in the same order: every flat index names
    // the same (f, j) in C as in B.
    for (int64_t n = 0; n < plan.N_total; ++n) {
        int64_t const c_off = pg::flat_to_offset(n, plan.c_n_dims);
        int64_t const b_off = pg::flat_to_offset(n, plan.n_dims);
        INFO("flat n = " << n);
        REQUIRE((c_off / qa) % x.nf == b_off / bf);          // f
        REQUIRE(c_off / (qa * x.nf) == (b_off % bf) / x.ni); // j
    }
}

TEST_CASE("coalesce: a C order kept for the direct write is not reordered for the operand", "[PackedGemm][Coalesce]") {
    // C(Q,x,y,z) <- A(Q,e) ; B(x,e,y,z), column-major, extents Q 64, x 2, y 5, z 7, e 3. C's N
    // group (x, y, z) tiles end to end in C, and the direct write wants it whole. B's own unit
    // stride is x, which the gather would rather have second-fastest; moving it there splits C's
    // group into three and sends C back through the scatter.
    pg::PackingPlan plan;
    plan.valid       = true;
    plan.m_dims      = {{.tensor_pos = 0, .size = 64, .tensor_stride = 1}};
    plan.c_m_dims    = plan.m_dims;
    plan.n_dims      = {{.tensor_pos = 0, .size = 2, .tensor_stride = 1},
                        {.tensor_pos = 2, .size = 5, .tensor_stride = 6},
                        {.tensor_pos = 3, .size = 7, .tensor_stride = 30}};
    plan.c_n_dims    = {{.tensor_pos = 1, .size = 2, .tensor_stride = 64 * 5 * 7},
                        {.tensor_pos = 2, .size = 5, .tensor_stride = 64 * 7},
                        {.tensor_pos = 3, .size = 7, .tensor_stride = 64}};
    plan.k_dims_in_a = {{.tensor_pos = 1, .size = 3, .tensor_stride = 64}};
    plan.k_dims_in_b = {{.tensor_pos = 1, .size = 3, .tensor_stride = 2}};
    plan.M_total     = 64;
    plan.N_total     = 70;
    plan.K_total     = 3;

    pg::coalesce_plan(plan);

    REQUIRE(plan.c_n_dims.size() == 1);
    REQUIRE(plan.c_n_dims[0].size == 70);
    REQUIRE(plan.c_n_dims[0].tensor_stride == 64);
}

namespace {

constexpr int64_t nQ = 24;
constexpr int64_t na = 10;
constexpr int64_t ne = 9;
constexpr int64_t nf = 13;

/// The (j, e, f) block of a rank-4 amplitude tensor, j over a range of three: with j strided (the
/// ladder's i-major slice) or with j the slice's unit-stride axis. The second matters to the vendor
/// GEMM paths: C's group is single, the operand's is not, and the operand's first dim has unit
/// stride, so a path that read only that dim would hand BLAS a valid-looking but wrong layout.
template <typename T>
TensorView<T, 3> jef_block(Tensor<T, 4> &Tau, bool j_unit) {
    return j_unit ? Tau(Range{1, 4}, 1, All, All) : Tau(1, Range{1, 4}, All, All);
}

template <typename T>
Tensor<T, 4> amplitudes(bool j_unit) {
    return j_unit ? create_random_tensor<T>("Tau", 5, 3, ne, nf) : create_random_tensor<T>("Tau", 3, 5, ne, nf);
}

} // namespace

TEMPLATE_TEST_CASE("coalesce: a split N group in B still writes C directly", "[PackedGemm][Coalesce]", float, double) {
    using T           = TestType;
    bool const j_unit = GENERATE(false, true);
    INFO("j unit stride in B: " << j_unit);
    auto const   A    = create_random_tensor<T>("A", nQ, na, ne);
    auto         Tau  = amplitudes<T>(j_unit);
    auto const   B    = jef_block(Tau, j_unit);
    auto         C    = create_random_tensor<T>("C", nQ, na, nf, 3);
    Tensor<T, 4> Cref = C;

    testing::reference_einsum("Qafj <- Qae ; jef", T{0}, &Cref, T{1}, A, B);

    pg::last_contraction_route() = "";
    pg::last_packed_blocking()   = {.c_scattered = true};
    bool const handled           = tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0}, Indices{Q, a, f, j}, &C, T{1},
                                                                                                 Indices{Q, a, e}, A, Indices{j, e, f}, B);
    REQUIRE(handled);
    REQUIRE(std::string(pg::last_contraction_route()) == "packed");
    REQUIRE_FALSE(pg::last_packed_blocking().c_scattered);
    check_close(C, Cref, ne);
}

TEMPLATE_TEST_CASE("coalesce: a split M group in A still writes C directly", "[PackedGemm][Coalesce]", float, double) {
    using T           = TestType;
    bool const j_unit = GENERATE(false, true);
    INFO("j unit stride in A: " << j_unit);
    auto         Tau  = amplitudes<T>(j_unit);
    auto const   A    = jef_block(Tau, j_unit);
    auto const   B    = create_random_tensor<T>("B", nQ, na, ne);
    auto         C    = create_random_tensor<T>("C", nf, 3, nQ, na);
    Tensor<T, 4> Cref = C;

    testing::reference_einsum("fjQa <- jef ; Qae", T{0}, &Cref, T{1}, A, B);

    pg::last_contraction_route() = "";
    pg::last_packed_blocking()   = {.c_scattered = true};
    bool const handled           = tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0}, Indices{f, j, Q, a}, &C, T{1},
                                                                                                 Indices{j, e, f}, A, Indices{Q, a, e}, B);
    REQUIRE(handled);
    REQUIRE(std::string(pg::last_contraction_route()) == "packed");
    REQUIRE_FALSE(pg::last_packed_blocking().c_scattered);
    check_close(C, Cref, ne);
}
