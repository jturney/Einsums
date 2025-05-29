//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace einsums {

// EINSUMS_TENSOR_DEFINE_RANK(BlockTensor, 2)
// EINSUMS_TENSOR_DEFINE_RANK(BlockTensor, 3)
// EINSUMS_TENSOR_DEFINE_RANK(BlockTensor, 4)
//
// EINSUMS_TENSOR_DEFINE(DiskTensor)
// EINSUMS_TENSOR_DEFINE_DISK_VIEW(DiskView)
//
// EINSUMS_TENSOR_DEFINE_RANK(Tensor, 0)
// EINSUMS_TENSOR_DEFINE(Tensor)
// EINSUMS_TENSOR_DEFINE(TensorView)
//
// EINSUMS_TENSOR_DEFINE(TiledTensor)
// EINSUMS_TENSOR_DEFINE(TiledTensorView)
//
// #ifndef EINSUMS_WINDOWS
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensor, float);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensor, double);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensor, std::complex<float>);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensor, std::complex<double>);
//
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensorView, float);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensorView, double);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensorView, std::complex<float>);
// EINSUMS_TENSOR_DEFINE_T(RuntimeTensorView, std::complex<double>);
// #endif

} // namespace einsums