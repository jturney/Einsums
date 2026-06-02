//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/Diagonal.hpp>

#include <numbers>
#include <string>

namespace einsums {

/**
 * Create a random positive or negative definite matrix.
 * A positive definite matrix is a symmetric matrix whose eigenvalues are all positive.
 * Similarly for negative definite matrices.
 *
 * This function first generates a set of random eigenvectors, making sure they are non-singular.
 * Then, it uses these to form an orthonormal eigenbasis for the new matrix. Then, it generates
 * the eigenvalues. The eigenvalues are distributed using a Maxwell-Boltzmann distribution
 * with the given mean, defaulting to 1. Then, the returned matrix is formed
 * by computing @f$P^TDP@f$. If the mean is negative, then the result will be a negative
 * definite matrix.
 *
 * @param[in] name The name for the matrix.
 * @param[in] rows The number of rows.
 * @param[in] cols The number of columns. Should equal the number of rows.
 * @param[in] mean The mean for the eigenvalues. Defaults to 1.
 * @return A new positive definite or negative definite matrix.
 *
 * @versionadded{1.0.0}
 */
template <typename T = double>
auto create_random_definite(std::string const &name, int rows, int cols, RemoveComplexT<T> mean = RemoveComplexT<T>{1.0}) -> Tensor<T, 2> {
    using Real = RemoveComplexT<T>;

    if (rows != cols) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only make square positive definite matrices.");
    }
    Tensor<T, 2> Evecs("name", rows, cols);

    Tensor<T, 2>              Temp = Evecs;
    BufferVector<blas::int_t> pivs;

    // Make sure the eigenvectors are non-singular.
    do {
        Evecs = create_random_tensor<T>("name", rows, cols);
        Temp  = Evecs;
    } while (linear_algebra::getrf(&Temp, &pivs) > 0);

    // QR decompose Evecs to get a random matrix of orthonormal eigenvectors.
    auto pair = linear_algebra::qr(Evecs);

    Evecs = std::get<0>(pair);

    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::default_random_engine engine;

    // Create random eigenvalues. Need to calculate the standard deviation from the mean.
    auto normal =
        std::normal_distribution<Real>(0, std::abs(mean) / Real{2.0} / std::numbers::sqrt2_v<Real> / std::numbers::inv_sqrtpi_v<Real>);

    Tensor<T, 1> Evals("name2", rows);

    for (int i = 0; i < rows; i++) {
        // Maxwell-Boltzmann distribute the eigenvalues. Make sure they are positive.
        do {
            Real val1 = normal(engine), val2 = normal(engine), val3 = normal(engine);

            Evals(i) = std::sqrt(val1 * val1 + val2 * val2 + val3 * val3);
            if (mean < Real{0.0}) {
                Evals(i) = -Evals(i);
            }
        } while (Evals(i) == Real{0.0}); // Make sure they are non-zero.
    }

    // Create the tensor.
    Tensor<T, 2> ret = diagonal(Evals);

    linear_algebra::gemm<false, false>(1.0, ret, Evecs, 0.0, &Temp);
    linear_algebra::gemm<true, false>(1.0, Evecs, Temp, 0.0, &ret);

    ret.set_name(name);

    return ret;
}

/**
 * @brief Python-bindable runtime-rank create_random_definite.
 *
 * Builds a square SPD (or SND if mean < 0) matrix and returns it as a
 * RuntimeTensor. Adapter over the typed Tensor<T, 2> entry point.
 */
template <typename T = double>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("create_random_definite", double) APIARY_INSTANTIATE_AS("create_random_definite", float)
    APIARY_INSTANTIATE_AS("create_random_definite", std::complex<double>)
        APIARY_INSTANTIATE_AS("create_random_definite", std::complex<float>) auto create_random_definite(std::string const &name, int n)
            -> RuntimeTensor<T> {
    return RuntimeTensor<T>(create_random_definite<T>(name, n, n));
}

} // namespace einsums