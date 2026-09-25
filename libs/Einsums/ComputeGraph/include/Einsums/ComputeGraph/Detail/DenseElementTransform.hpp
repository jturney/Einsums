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

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Apply @p unary_op to every element of a dense tensor, in place.
 *
 * ComputeGraph's own copy of the kernel behind ``tensor_algebra::element_transform``, which keeps its
 * version for eager code. The copy is what lets ComputeGraph and the headers it includes stay free of
 * the TensorAlgebra module.
 *
 * Elements are visited in C's logical order and located through its strides, so a view is handled
 * like an owning tensor.
 */
template <CoreTensorConcept CType, typename UnaryOperator>
    requires BasicTensorConcept<CType> && RankTensorConcept<CType>
void dense_element_transform(CType *C, UnaryOperator unary_op) {
    LabeledSection0();
    using T               = typename CType::ValueType;
    constexpr size_t Rank = CType::Rank;

    Stride<Rank> index_strides;
    size_t const elements = dims_to_strides(C->dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR_IF(elements >= ::einsums::hardware::omp_min_parallel_elements())
    for (size_t item = 0; item < elements; item++) {
        size_t offset;
        sentinel_to_sentinels(item, index_strides, C->strides(), offset);

        T &target = C->data()[offset];
        target    = unary_op(target);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
