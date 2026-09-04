//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file PooledTensor.hpp
 *
 * Tensor factories that carve their storage from a @ref einsums::MemoryPool.
 *
 * These live here rather than on @c MemoryPool itself because BufferAllocator
 * sits below Tensor in the module graph: the pool knows how to place *a* tensor
 * (@c MemoryPool::place), and this header names the concrete tensor types.
 *
 * A pooled tensor is an ordinary tensor to every consumer. It differs only in
 * where its bytes come from and in what happens when it dies: the keepalive
 * token it carries returns the carve to the pool, from whatever thread the
 * destructor runs on.
 */

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <complex>
#include <concepts>
#include <memory>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Runtime-rank tensor carved from @p pool, contents uninitialized.
 *
 * @versionadded{2.1.0}
 */
template <typename T, typename Alloc = std::allocator<T>>
[[nodiscard]] GeneralRuntimeTensor<T, Alloc> pool_empty(MemoryPool &pool, std::string name, std::vector<size_t> const &dims) {
    return pool.empty_as<GeneralRuntimeTensor<T, Alloc>>(std::move(name), dims);
}

/**
 * @brief Runtime-rank tensor carved from @p pool and zeroed.
 *
 * @versionadded{2.1.0}
 */
template <typename T, typename Alloc = std::allocator<T>>
[[nodiscard]] GeneralRuntimeTensor<T, Alloc> pool_zeros(MemoryPool &pool, std::string name, std::vector<size_t> const &dims) {
    return pool.zeros_as<GeneralRuntimeTensor<T, Alloc>>(std::move(name), dims);
}

/**
 * @brief Compile-time-rank tensor carved from @p pool, contents uninitialized.
 *
 * @versionadded{2.1.0}
 */
template <typename T, size_t Rank, std::integral... Dims>
    requires(sizeof...(Dims) == Rank)
[[nodiscard]] Tensor<T, Rank> pool_tensor(MemoryPool &pool, std::string name, Dims... dims) {
    return pool.empty_ranked<Tensor<T, Rank>>(std::move(name), dims...);
}

/**
 * @brief Compile-time-rank tensor carved from @p pool and zeroed.
 *
 * @versionadded{2.1.0}
 */
template <typename T, size_t Rank, std::integral... Dims>
    requires(sizeof...(Dims) == Rank)
[[nodiscard]] Tensor<T, Rank> pool_zero_tensor(MemoryPool &pool, std::string name, Dims... dims) {
    return pool.zeros_ranked<Tensor<T, Rank>>(std::move(name), dims...);
}

namespace detail {

/// Build a pooled runtime tensor on the heap.
///
/// The Python surface goes through a @c unique_ptr, not a value: these tensor
/// types have no move constructor, so pybind11 would fall back to the
/// deep-copying copy constructor and quietly hand Python an un-pooled tensor
/// twice the size. A holder also pins ownership regardless of the return-value
/// policy the binding is emitted with - a bare pointer defaults to
/// @c automatic_reference, which would leak both the tensor and its carve.
template <typename TensorT>
std::unique_ptr<TensorT> new_pooled(MemoryPool &pool, std::string name, std::vector<size_t> const &dims, bool zero) {
    auto owned = std::make_unique<TensorT>(typename TensorT::DeferredAlloc{}, std::move(name), dims);
    pool.place(*owned, zero);
    return owned;
}

} // namespace detail

/**
 * @brief Python entry point for @ref pool_empty.
 *
 * Bound as ``einsums.pool_empty(pool, name, dims, dtype)`` and reachable as
 * ``pool.empty(...)`` after the Python layer attaches it to the class.
 *
 * @versionadded{2.1.0}
 */
template <typename T, typename Alloc = std::allocator<T>>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("pool_empty", float, std::allocator<float>)
APIARY_INSTANTIATE_AS("pool_empty", double, std::allocator<double>)
APIARY_INSTANTIATE_AS("pool_empty", std::complex<float>, std::allocator<std::complex<float>>)
APIARY_INSTANTIATE_AS("pool_empty", std::complex<double>,
                      std::allocator<std::complex<double>>) std::unique_ptr<GeneralRuntimeTensor<T, Alloc>>
pool_new_empty(MemoryPool &pool, std::string name, std::vector<size_t> dims) {
    return detail::new_pooled<GeneralRuntimeTensor<T, Alloc>>(pool, std::move(name), dims, /*zero=*/false);
}

/**
 * @brief Python entry point for @ref pool_zeros.
 *
 * @versionadded{2.1.0}
 */
template <typename T, typename Alloc = std::allocator<T>>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("pool_zeros", float, std::allocator<float>)
APIARY_INSTANTIATE_AS("pool_zeros", double, std::allocator<double>)
APIARY_INSTANTIATE_AS("pool_zeros", std::complex<float>, std::allocator<std::complex<float>>)
APIARY_INSTANTIATE_AS("pool_zeros", std::complex<double>,
                      std::allocator<std::complex<double>>) std::unique_ptr<GeneralRuntimeTensor<T, Alloc>>
pool_new_zeros(MemoryPool &pool, std::string name, std::vector<size_t> dims) {
    return detail::new_pooled<GeneralRuntimeTensor<T, Alloc>>(pool, std::move(name), dims, /*zero=*/true);
}

EINSUMS_NAMESPACE_END()
