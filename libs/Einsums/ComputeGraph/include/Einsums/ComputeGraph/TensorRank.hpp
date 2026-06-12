//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums::compute_graph {

// True when the tensor type carries its rank as a static compile-time
// constant (the GeneralTensor / TensorView family). False for runtime-rank
// tensors (RuntimeTensor / future BlockRuntimeTensor / TiledRuntimeTensor),
// which expose ``Rank == einsums::dynamic_rank`` as a sentinel.
template <typename T>
concept HasCompileTimeRank = requires {
    { std::remove_cvref_t<T>::Rank } -> std::convertible_to<std::size_t>;
} && (std::remove_cvref_t<T>::Rank != einsums::dynamic_rank);

// A tensor whose rank is only known at runtime via a member ``.rank()``.
// Used as the dispatch tag for cg::einsum and other ComputeGraph operations
// when they need a code path that doesn't bake rank into the type.
template <typename T>
concept RuntimeRankTensorConcept = BasicTensorConcept<T> && (!HasCompileTimeRank<T>)&&requires(T const &t) {
    { t.rank() } -> std::convertible_to<std::size_t>;
};

namespace detail {

// Return the rank of a tensor, sourced at compile time when the tensor
// type carries a static ``::Rank`` (the GeneralTensor/TensorView family)
// and at runtime via ``.rank()`` otherwise (the RuntimeTensor family).
//
// Used by per-rank dispatch so the same function-template body works
// against both typed tensors (Tensor<T,R>, perf-identical to the previous
// ``if constexpr (AType::Rank == ...)`` form because the comparison still
// folds at compile time) and runtime-rank tensors (RuntimeTensor<T, Alloc>,
// where the comparison becomes a real but negligible branch dwarfed by the
// contraction work).
template <typename T>
constexpr std::size_t tensor_rank(T const &t) noexcept {
    using Clean = std::remove_cvref_t<T>;
    if constexpr (requires { Clean::Rank; }) {
        if constexpr (Clean::Rank == einsums::dynamic_rank) {
            return t.rank(); // dynamic-rank sentinel, defer to runtime
        } else {
            return Clean::Rank;
        }
    } else {
        return t.rank();
    }
}

} // namespace detail

} // namespace einsums::compute_graph
