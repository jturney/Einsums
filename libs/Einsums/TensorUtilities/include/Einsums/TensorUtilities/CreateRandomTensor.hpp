//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <concepts>
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
 * @param[in] row_major Whether the tensor should be row-major or column-major.
 * @param[in] name The name of the new tensor.
 * @param[in] distribution The random distribution to use for generating the random numbers.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionaddeddesc{2.0.0}
 *      Added the row_major parameter.
 * @endversion
 */
template <typename T = double, bool Normalize = false, typename Distribution, std::integral... MultiIndex>
    requires requires(Distribution dist) {
        { dist(einsums::random_engine) } -> std::same_as<T>;
    }
auto create_random_tensor(bool row_major, std::string const &name, Distribution &&distribution, MultiIndex... index)
    -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating random tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(row_major, name, std::forward<MultiIndex>(index)...);
    // Serial fill: einsums::random_engine is a shared global LCG and
    // std::uniform_real_distribution carries mutable internal state. Wrapping
    // this loop in #pragma omp parallel for races on both. TSan reported
    // 1647 hits at this line on arm64 (1305 on amd64), top of the report
    // list and the source of every downstream SIMD-load race when the
    // generated buffer is read by consumers. The parallel form is also not
    // deterministic across OMP schedules, which test consumers rely on.
    // Per-thread engines + counter-based seed splitting would restore
    // parallelism while keeping determinism, but for test/utility tensor
    // sizes the serial cost (sub-ms for typical sizes; ~100ms for 100MB on
    // commodity hardware) is well below noise.
    for (size_t i = 0; i < A.size(); i++) {
        A.data()[i] = distribution(einsums::random_engine);
    }

    if constexpr (Normalize && sizeof...(MultiIndex) == 2) {
        for (int col = 0; col < A.dim(-1); col++) {
            RemoveComplexT<T> scale{1};
            RemoveComplexT<T> sumsq{0};

            auto column = A(All, col);
            // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
            // auto column = collapsed(All, col);
            sum_square(column, &scale, &sumsq);
            T value = scale * sqrt(sumsq);
            column /= value;
        }
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{1.1.0}
 *      Complex values are now clamped within the unit circle, rather than the unit square.
 * @endversion
 */
template <typename T = double, bool Normalize = false, std::integral... MultiIndex>
auto create_random_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(GlobalConfigMap::get_singleton().get_bool("row-major"), name,
                                                  detail::UnitCircleDistribution<T>(), index...);
    } else {
        return create_random_tensor<T, Normalize>(GlobalConfigMap::get_singleton().get_bool("row-major"), name,
                                                  std::uniform_real_distribution<T>(-1, 1), index...);
    }
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] row_major Whether the tensor should be row-major or column major.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionaddeddesc{2.0.0}
 *      Added row major parameter.
 * @endversion
 */
template <typename T = double, bool Normalize = false, std::integral... MultiIndex>
auto create_random_tensor(bool row_major, std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(row_major, name, detail::UnitCircleDistribution<T>(), index...);
    } else {
        return create_random_tensor<T, Normalize>(row_major, name, std::uniform_real_distribution<T>(-1, 1), index...);
    }
}

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
 * @param[in] row_major Whether the tensor should be row-major or column-major.
 * @param[in] name The name of the new tensor.
 * @param[in] dist The random distribution to use for generating the random numbers.
 * @param[in] dims The dimensions of the new tensor.
 * @return A new tensor filled with random data
 *
 * @versionaddeddesc{2.0.0}
 *      Added row major parameter.
 * @endversion
 */
template <typename T = double, bool Normalize = false, typename Distribution, Container Dims>
auto create_random_tensor(bool row_major, std::string const &name, Distribution &&dist, Dims const &dims) -> RuntimeTensor<T> {
    EINSUMS_LOG_TRACE("creating random runtime tensor {}, {}", name, dims);

    RuntimeTensor<T> A(name, dims, row_major);

    // Serial fill, with the same rationale as the templated overload above:
    // einsums::random_engine + distribution share state and racing them from
    // an OMP region produces 1600+ TSan reports plus non-deterministic data.
    for (size_t i = 0; i < A.size(); i++) {
        A.data()[i] = dist(einsums::random_engine);
    }

    if constexpr (Normalize) {
        if (dims.size() == 2) {
            for (int col = 0; col < A.dim(-1); col++) {
                RemoveComplexT<T> scale{1};
                RemoveComplexT<T> sumsq{0};

                auto column = A(All, col);
                // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
                // auto column = collapsed(All, col);
                sum_square(column, &scale, &sumsq);
                T value = scale * sqrt(sumsq);
                column /= value;
            }
        }
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{1.1.0}
 *      Complex values are now clamped within the unit circle, rather than the unit square.
 * @endversion
 */
template <typename T = double, bool Normalize = false, Container Indices>
auto create_random_tensor(std::string const &name, Indices const &index) -> RuntimeTensor<T> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(GlobalConfigMap::get_singleton().get_bool("row-major"), name,
                                                  detail::UnitCircleDistribution<T>(), index);
    } else {
        return create_random_tensor<T, Normalize>(GlobalConfigMap::get_singleton().get_bool("row-major"), name,
                                                  std::uniform_real_distribution<T>(-1, 1), index);
    }
}

/**
 * @brief Python-bindable runtime-rank create_random_tensor.
 *
 * Concrete-typed overload taking ``std::vector<size_t>`` so the
 * einsums-pybind codegen can emit a direct binding. Forwards to the
 * templated container overload above. Exposed across the four bound
 * dtypes (float, double, complex<float>, complex<double>).
 */
template <typename T = double>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("create_random_tensor", double) APIARY_INSTANTIATE_AS("create_random_tensor", float)
    APIARY_INSTANTIATE_AS("create_random_tensor", std::complex<double>)
        APIARY_INSTANTIATE_AS("create_random_tensor", std::complex<float>) auto create_random_tensor(std::string const         &name,
                                                                                                     std::vector<size_t> const &dims)
            -> RuntimeTensor<T> {
    return create_random_tensor<T, false>(name, dims);
}

/**
 * @brief Create a new tensor with \p name and \p index filled with random data.
 *
 * Just a simple factory function for creating new tensor with initial data. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations. The data generated will
 * be between -1 and 1 for reals.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam Normalize Should the resulting random data be normalized. Defaults to false.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] row_major Whether the tensor should be row-major or column-major.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionaddeddesc{2.0.0}
 *      Added row major parameter.
 * @endversion
 */
template <typename T = double, bool Normalize = false, Container Indices>
auto create_random_tensor(bool row_major, std::string const &name, Indices const &index) -> RuntimeTensor<T> {
    if constexpr (IsComplexV<T>) {
        return create_random_tensor<T, Normalize>(row_major, name, detail::UnitCircleDistribution<T>(), index);
    } else {
        return create_random_tensor<T, Normalize>(row_major, name, std::uniform_real_distribution<T>(-1, 1), index);
    }
}

} // namespace einsums
