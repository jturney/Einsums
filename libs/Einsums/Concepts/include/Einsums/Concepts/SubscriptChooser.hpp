//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <type_traits>
#include <utility>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Subscripts into a tensor, selecting between the subscript method or the function call syntax.
 *
 * @tparam TensorType The type of tensor that will be subscripted.
 * @tparam MultiArgs The data types for the indices.
 *
 * @param[in] tensor The tensor to subscript.
 * @param[in] args The indices.
 *
 * @return The value at the location specified by the subscript, or a reference to that value.
 *
 * @versionadded{1.0.0}
 */
template <RankTensorConcept TensorType, typename... MultiArgs>
    requires requires {
        requires std::remove_cvref_t<TensorType>::Rank == sizeof...(MultiArgs);
        requires(std::is_integral_v<std::remove_cvref_t<MultiArgs>> && ...);
        requires FastSubscriptableConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, MultiArgs &&...args) -> decltype(tensor.subscript(std::forward<MultiArgs>(args)...)) {
    return tensor.subscript(std::forward<MultiArgs>(args)...);
}

/**
 * @brief Subscripts into a tensor, selecting between the subscript method or the function call syntax.
 *
 * @tparam TensorType The type of tensor that will be subscripted.
 * @tparam MultiArgs The data types for the indices.
 *
 * @param[in] tensor The tensor to subscript.
 * @param[in] args The indices.
 *
 * @return The value at the location specified by the subscript, or a reference to that value.
 *
 * @versionadded{1.0.0}
 */
template <RankTensorConcept TensorType, typename... MultiArgs>
    requires requires {
        requires std::remove_cvref_t<TensorType>::Rank == sizeof...(MultiArgs);
        requires(std::is_integral_v<std::remove_cvref_t<MultiArgs>> && ...);
        requires !FastSubscriptableConcept<TensorType>;
        requires FunctionTensorConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, MultiArgs &&...args) -> decltype(tensor(std::forward<MultiArgs>(args)...)) {
    return tensor(std::forward<MultiArgs>(args)...);
}

/**
 * @brief Subscripts into a tensor, selecting between the subscript method or the function call syntax.
 *
 * @tparam TensorType The type of tensor that will be subscripted.
 * @tparam ContainerType The data type for the index container.
 *
 * @param[in] tensor The tensor to subscript.
 * @param[in] args The indices.
 *
 * @return The value at the location specified by the subscript, or a reference to that value.
 *
 * @versionadded{1.0.0}
 */
template <RankTensorConcept TensorType, typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires FastSubscriptableConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, ContainerType const &args) -> decltype(tensor.subscript(args)) {
    return tensor.subscript(args);
}

/**
 * @brief Subscripts into a tensor, selecting between the subscript method or the function call syntax.
 *
 * @tparam TensorType The type of tensor that will be subscripted.
 * @tparam ContainerType The data type for the index container.
 *
 * @param[in] tensor The tensor to subscript.
 * @param[in] args The indices.
 *
 * @return The value at the location specified by the subscript, or a reference to that value.
 *
 * @versionadded{1.0.0}
 */
template <RankTensorConcept TensorType, typename ContainerType>
    requires requires {
        requires !std::is_integral_v<ContainerType>;
        requires !FastSubscriptableConcept<TensorType>;
        requires FunctionTensorConcept<TensorType>;
    }
inline auto subscript_tensor(TensorType &&tensor, ContainerType const &args) -> decltype(tensor(args)) {
    return tensor(args);
}

EINSUMS_NAMESPACE_END()