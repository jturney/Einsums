//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/concepts/complex.hpp>
#include <einsums/tensor/tensor_fwd.hpp>
#include <einsums/tensor_base/common.hpp>

#include <range/v3/view/cartesian_product.hpp>

#include <complex>
#include <omp.h>
#include <random>
#include <string>

namespace einsums {

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param name The name of the new tensor.
 * @param index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 */
template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_tensor(const std::string &name, MultiIndex... index) -> tensor<T, sizeof...(MultiIndex)> {
    tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);

    double const lower_bound = -1.0;
    double const upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine             re;

    if constexpr (std::is_same_v<T, std::complex<float>>) {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(),
                          [&]() { return T{static_cast<float>(unif(re)), static_cast<float>(unif(re))}; });
        }
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(),
                          [&]() { return T{static_cast<double>(unif(re)), static_cast<double>(unif(re))}; });
        }
    } else {
#pragma omp parallel default(none) shared(A, re, unif)
        {
            auto tid       = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin     = A.vector_data().begin() + chunksize * tid;
            auto end       = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() { return static_cast<T>(unif(re)); });
        }
    }

    if constexpr (Normalize && sizeof...(MultiIndex) == 2) {
        for (int col = 0; col < A.dim(-1); col++) {
            remove_complex_t<T> scale{1};
            remove_complex_t<T> sumsq{0};

            auto column = A(all, col);
            // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
            // auto column = collapsed(All, col);
            sum_square(column, &scale, &sumsq);
            T value = scale * sqrt(sumsq);
            column /= value;
        }
    }

    return A;
}

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_gpu_tensor(const std::string &name, MultiIndex... index) -> DeviceTensor<T, sizeof...(MultiIndex)> {
    DeviceTensor<T, sizeof...(MultiIndex)> out{name, einsums::detail::DEV_ONLY, index...};
    out = create_random_tensor<T, Normalize, MultiIndex...>(name, index...);
    return out;
}
#endif

} // namespace einsums
