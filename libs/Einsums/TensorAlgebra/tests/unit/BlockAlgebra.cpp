//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

// A full contraction of block tensors into a scalar scales the scalar by its prefactor once and adds
// every block's contribution once. The reduction over blocks started from the already scaled scalar and
// then added its total back onto the scalar, so the scaled value counted twice.
TEMPLATE_TEST_CASE("block tensors contracted to a scalar apply the scalar's prefactor once", "[tensor-algebra][block-tensor]", float,
                   double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    using T = TestType;

    // Two 3x3 blocks on the diagonal of a 6x6.
    BlockTensor<T, 2> A("A", 3, 3), B("B", 3, 3);
    A.zero();
    B.zero();
    T expected_sum{0};
    for (int d = 0; d < 6; d++) {
        A(d, d) = T(d + 1);
        B(d, d) = T(2);
        expected_sum += T(d + 1) * T(2);
    }

    tensor_algebra::detail::AlgorithmChoice choice{};
    // ij,ij is a dot and takes the dot route; ij,ji reaches the block backend's reduction.
    SECTION("ij,ij") {
        Tensor<T, 0> C("C");
        C = T(10);
        einsum(T(0.5), Indices{}, &C, T(1), Indices{i, j}, A, Indices{i, j}, B, &choice);
        T const got = C;
        INFO("got " << got << ", route " << static_cast<int>(choice));
        CHECK(std::abs(got - (T(5) + expected_sum)) < 1e-4);
    }
    SECTION("ij,ji") {
        Tensor<T, 0> C("C");
        C = T(10);
        einsum(T(0.5), Indices{}, &C, T(1), Indices{i, j}, A, Indices{j, i}, B, &choice);
        T const got = C;
        INFO("got " << got << ", route " << static_cast<int>(choice));
        CHECK(std::abs(got - (T(5) + expected_sum)) < 1e-4);
    }
}
