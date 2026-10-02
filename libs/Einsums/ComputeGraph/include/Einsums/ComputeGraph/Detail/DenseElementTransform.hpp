//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Whether @p dims and @p strides lay the elements out as one gap-free block.
 *
 * True when the strides, taken in increasing order over the dimensions longer than one, are 1,
 * then the product of the extents before them: a packed layout in any dimension order, so the
 * elements are exactly the offsets 0 through size-1, each once. Any padding, overlap or
 * reversed stride makes it false.
 */
template <size_t Rank, typename Dims, typename Strides>
bool is_dense_block(Dims const &dims, Strides const &strides) {
    std::array<size_t, Rank> order{};
    size_t                   used = 0;
    for (size_t d = 0; d < Rank; ++d) {
        if (dims[d] > 1) {
            order[used++] = d;
        }
    }
    std::sort(order.begin(), order.begin() + used, [&](size_t a, size_t b) { return strides[a] < strides[b]; });

    size_t expected = 1;
    for (size_t k = 0; k < used; ++k) {
        size_t const d = order[k];
        if (static_cast<size_t>(strides[d]) != expected) {
            return false;
        }
        expected *= dims[d];
    }
    return true;
}

/**
 * @brief Apply @p unary_op to every element of a dense tensor, in place.
 *
 * ComputeGraph's own copy of the kernel behind ``tensor_algebra::element_transform``, which keeps its
 * version for eager code. The copy is what lets ComputeGraph and the headers it includes stay free of
 * the TensorAlgebra module.
 *
 * A tensor whose elements fill one gap-free block, in any dimension order, is walked as a flat
 * array, a loop the compiler can vectorize. Anything else (a strided view) is visited in C's
 * logical order and located through its strides, so a view is handled like an owning tensor.
 */
template <CoreTensorConcept CType, typename UnaryOperator>
    requires BasicTensorConcept<CType> && RankTensorConcept<CType>
void dense_element_transform(CType *C, UnaryOperator unary_op) {
    WAGGLE_ZONE_FUNC();
    using T               = typename CType::ValueType;
    constexpr size_t Rank = CType::Rank;

    Stride<Rank> index_strides;
    size_t const elements = dims_to_strides(C->dims(), index_strides);

    if (is_dense_block<Rank>(C->dims(), C->strides())) {
        T *const data = C->data();
        EINSUMS_OMP_PARALLEL_FOR_IF(elements >= ::einsums::hardware::omp_min_parallel_elements())
        for (size_t item = 0; item < elements; item++) {
            data[item] = unary_op(data[item]);
        }
        return;
    }

    EINSUMS_OMP_PARALLEL_FOR_IF(elements >= ::einsums::hardware::omp_min_parallel_elements())
    for (size_t item = 0; item < elements; item++) {
        size_t offset;
        sentinel_to_sentinels(item, index_strides, C->strides(), offset);

        T &target = C->data()[offset];
        target    = unary_op(target);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
