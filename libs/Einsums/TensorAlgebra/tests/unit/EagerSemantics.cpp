//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The typed-Indices einsum held to semantics the string engine already had: lone summed indices
// and the output-aliasing policy. These cases lived beside the graph parity tests; they test this
// engine alone, so they live with it, and go when it does.

#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <stdexcept>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace ta = einsums::tensor_algebra;

namespace {

template <typename T, size_t Rank>
void require_close(Tensor<T, Rank> const &got, Tensor<T, Rank> const &want, double tol = 1e-10) {
    auto const n = got.size();
    for (size_t flat = 0; flat < n; ++flat) {
        REQUIRE(std::abs(got.data()[flat] - want.data()[flat]) <= tol * (1.0 + std::abs(want.data()[flat])));
    }
}

// C(i,j) = (sum_k S(i,j,k)) * W(i,j)  -- empty link, lone k summed in A (P1).
Tensor<double, 2> lone_empty_link_reference(Tensor<double, 3> const &S, Tensor<double, 2> const &W) {
    Tensor<double, 2> ref{"ref", S.dim(0), S.dim(1)};
    for (size_t a = 0; a < S.dim(0); ++a)
        for (size_t b = 0; b < S.dim(1); ++b) {
            double s = 0.0;
            for (size_t c = 0; c < S.dim(2); ++c)
                s += S(a, b, c);
            ref(a, b) = s * W(a, b);
        }
    return ref;
}

// C(j,k) = sum_l sum_p A(j,l) * B(p,l,k)  -- shared link l, lone p summed in B.
Tensor<double, 2> link_plus_lone_reference(Tensor<double, 2> const &A, Tensor<double, 3> const &B) {
    Tensor<double, 2> ref{"ref", A.dim(0), B.dim(2)};
    for (size_t jj = 0; jj < A.dim(0); ++jj)
        for (size_t kk = 0; kk < B.dim(2); ++kk) {
            double s = 0.0;
            for (size_t ll = 0; ll < A.dim(1); ++ll)
                for (size_t pp = 0; pp < B.dim(0); ++pp)
                    s += A(jj, ll) * B(pp, ll, kk);
            ref(jj, kk) = s;
        }
    return ref;
}

} // namespace

TEST_CASE("eager parity - lone summed index empty link ij<-ijk;ij", "[TensorAlgebra][EagerSemantics][lone-summed]") {
    // The eager path used to leave the lone k unsummed and read only k = 0.
    auto S   = create_random_tensor<double>("S", 3, 4, 5);
    auto W   = create_random_tensor<double>("W", 3, 4);
    auto ref = lone_empty_link_reference(S, W);

    auto C_eager = create_zero_tensor<double>("Ce", 3, 4);
    ta::einsum(Indices{i, j}, &C_eager, Indices{i, j, k}, S, Indices{i, j}, W);

    require_close(C_eager, ref);
}

TEST_CASE("eager parity - lone summed index with shared link jk<-jl;plk", "[TensorAlgebra][EagerSemantics][lone-summed]") {
    // The eager path used to throw std::out_of_range on the lone p here (a stride
    // access past the operand rank); p lives only in B.
    auto A   = create_random_tensor<double>("A", 3, 4);
    auto B   = create_random_tensor<double>("B", 2, 4, 5);
    auto ref = link_plus_lone_reference(A, B);

    auto C_eager = create_zero_tensor<double>("Ce", 3, 5);
    ta::einsum(Indices{j, k}, &C_eager, Indices{j, l}, A, Indices{p, l, k}, B);

    require_close(C_eager, ref);
}

TEST_CASE("eager aliasing - in-place update survives the generic algorithm", "[TensorAlgebra][EagerSemantics][aliasing]") {
    // Same defect, same fix, in the eager generic algorithm. Eager is the
    // oracle the differential tests compare against, so a silent zero here
    // would agree with a silent zero in the graph path and neither would fail.
    auto const A0 = create_random_tensor<double>("A0", 3, 4);
    auto const v  = create_random_tensor<double>("v", 4);

    auto expected = create_zero_tensor<double>("E", 3, 4);
    for (size_t a = 0; a < 3; ++a)
        for (size_t b = 0; b < 4; ++b)
            expected(a, b) = A0(a, b) * v(b);

    auto C = A0;
    ta::einsum(Indices{i, j}, &C, Indices{i, j}, C, Indices{j}, v);
    require_close(C, expected);
}

TEST_CASE("eager aliasing - typed-Indices dispatcher matches the policy", "[TensorAlgebra][EagerSemantics][aliasing]") {
    auto C = create_random_tensor<double>("C", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    // Contraction with C as an input: rejected.
    REQUIRE_THROWS_AS(ta::einsum(Indices{i, j}, &C, Indices{i, k}, C, Indices{k, j}, B), std::invalid_argument);

    // Elementwise in-place (identical index lists): allowed and correct.
    auto D        = create_random_tensor<double>("D", 4, 4);
    auto expected = create_zero_tensor<double>("E", 4, 4);
    for (size_t a = 0; a < 4; ++a)
        for (size_t b = 0; b < 4; ++b)
            expected(a, b) = D(a, b) * B(a, b);
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &D, Indices{i, j}, D, Indices{i, j}, B));
    require_close(D, expected);

    // A and B sharing a tensor: always allowed.
    auto A  = create_random_tensor<double>("A", 4, 4);
    auto C2 = create_zero_tensor<double>("C2", 4, 4);
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &C2, Indices{i, k}, A, Indices{k, j}, A));
}
