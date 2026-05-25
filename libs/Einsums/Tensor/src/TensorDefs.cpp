//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/GPU/DeviceVector.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#include <H5Lpublic.h>
#include <complex>
#include <memory>
#include <string>

namespace einsums {

TENSOR_DEFINE_RANK(BlockTensor, 2)
TENSOR_DEFINE_RANK(BlockTensor, 3)
TENSOR_DEFINE_RANK(BlockTensor, 4)

TENSOR_DEFINE(DiskTensor)
TENSOR_DEFINE(DiskView)

TENSOR_DEFINE_ALLOC_RANK(GeneralTensor, 0, std::allocator)
TENSOR_ALLOC_DEFINE(GeneralTensor, std::allocator)
TENSOR_ALLOC_DEFINE(GeneralTensor, BufferAllocator)
TENSOR_DEFINE(TensorView)

TENSOR_DEFINE(TiledTensor)
TENSOR_DEFINE(TiledTensorView)

#ifndef EINSUMS_WINDOWS
template class GeneralRuntimeTensor<float, std::allocator<float>>;
template class GeneralRuntimeTensor<double, std::allocator<double>>;
template class GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>;
template class GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>;

template class GeneralRuntimeTensor<float, BufferAllocator<float>>;
template class GeneralRuntimeTensor<double, BufferAllocator<double>>;
template class GeneralRuntimeTensor<std::complex<float>, BufferAllocator<std::complex<float>>>;
template class GeneralRuntimeTensor<std::complex<double>, BufferAllocator<std::complex<double>>>;

// NB: no explicit instantiations for gpu::DeviceAllocator — the device
// runtime variant has host-only members gated with static_assert /
// requires, so explicit class instantiation would force-instantiate
// those bodies and fail. Code that uses RuntimeGPUTensor<T> picks it
// up via implicit instantiation, mirroring GeneralTensor's pattern.

template class RuntimeTensorView<float>;
template class RuntimeTensorView<double>;
template class RuntimeTensorView<std::complex<float>>;
template class RuntimeTensorView<std::complex<double>>;

template class TiledRuntimeTensor<float>;
template class TiledRuntimeTensor<double>;
template class TiledRuntimeTensor<std::complex<float>>;
template class TiledRuntimeTensor<std::complex<double>>;

template class TiledRuntimeTensorView<float>;
template class TiledRuntimeTensorView<double>;
template class TiledRuntimeTensorView<std::complex<float>>;
template class TiledRuntimeTensorView<std::complex<double>>;
#endif

namespace {
bool verify_path(std::string const &path) {
    if (path.size() == 0) {
        return true;
    }

    if (path[0] != '/' && path[0] != '.') {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "The format of the disk tensor name \"{}\" was invalid! It must be formatted as a path.", path);
    }

    return true;
}
} // namespace

namespace detail {

bool verify_exists(hid_t loc_id, std::string const &path, hid_t lapl_id) {
    if (!verify_path(path)) {
        return false;
    }
    if (path.length() == 0) {
        return false;
    }

    std::string temp_path;

    temp_path.reserve(path.length());

    for (auto ch : path) {
        if (ch == '/' && temp_path.length() > 0) {
            auto res = H5Lexists(loc_id, temp_path.c_str(), lapl_id);

            if (res <= 0) {
                return false;
            }
        }
        temp_path.push_back(ch);
    }

    return H5Lexists(loc_id, temp_path.c_str(), lapl_id) > 0;
}
} // namespace detail

} // namespace einsums