//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <complex>

/// The tensor types the library precompiles ComputeGraph's per-tensor-type
/// templates for: dense tensors and views of rank 1 to 4 and the runtime-rank
/// tensor and view, over the four element types. Each header that declares such a
/// template ``extern`` for these types, and the source that instantiates it, applies
/// @p X to each type through this one list, so the two cannot drift apart. Any other
/// type instantiates the template where it is used.
#define EINSUMS_CG_COMMON_TENSOR_TYPES_FOR(X, T)                                                                                           \
    X(::einsums::Tensor<T, 1>)                                                                                                             \
    X(::einsums::Tensor<T, 2>)                                                                                                             \
    X(::einsums::Tensor<T, 3>)                                                                                                             \
    X(::einsums::Tensor<T, 4>)                                                                                                             \
    X(::einsums::TensorView<T, 1>)                                                                                                         \
    X(::einsums::TensorView<T, 2>)                                                                                                         \
    X(::einsums::TensorView<T, 3>)                                                                                                         \
    X(::einsums::TensorView<T, 4>)                                                                                                         \
    X(::einsums::RuntimeTensor<T>)                                                                                                         \
    X(::einsums::RuntimeTensorView<T>)
#define EINSUMS_CG_COMMON_TENSOR_TYPES(X)                                                                                                  \
    EINSUMS_CG_COMMON_TENSOR_TYPES_FOR(X, float)                                                                                           \
    EINSUMS_CG_COMMON_TENSOR_TYPES_FOR(X, double)                                                                                          \
    EINSUMS_CG_COMMON_TENSOR_TYPES_FOR(X, std::complex<float>)                                                                             \
    EINSUMS_CG_COMMON_TENSOR_TYPES_FOR(X, std::complex<double>)

/// The element types the library precompiles ComputeGraph's per-element-type
/// templates for, such as @ref Graph's tensor factories. @p X is applied to each.
#define EINSUMS_CG_ELEMENT_TYPES(X)                                                                                                        \
    X(float)                                                                                                                               \
    X(double)                                                                                                                              \
    X(std::complex<float>)                                                                                                                 \
    X(std::complex<double>)

/// The real element types, for the operations that have no complex form (sqrt, max, the sandwich
/// kernels). @p X is applied to each.
#define EINSUMS_CG_REAL_ELEMENT_TYPES(X)                                                                                                   \
    X(float)                                                                                                                               \
    X(double)
