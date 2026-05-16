//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums {

namespace detail {

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct AdditionOp {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct SubtractionOp {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct MultiplicationOp {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct DivisionOp {};

template <CanBeComplex T, typename... MultiIndex>
constexpr T compute_arithmetic(T scalar, MultiIndex... /*inds*/) {
    return scalar;
}

template <typename T, TensorConcept TensorType, typename... MultiIndex>
inline T compute_arithmetic(TensorType const *tensor, MultiIndex... inds) {
    return (*tensor)(inds...);
}

template <typename T, typename Op, typename Left, typename Right, typename... MultiIndex>
T compute_arithmetic(std::tuple<Op, Left, Right> const *input, MultiIndex... inds) {
    if constexpr (std::is_same_v<Op, AdditionOp>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) + compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, SubtractionOp>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) - compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, MultiplicationOp>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) * compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, DivisionOp>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) / compute_arithmetic<T>(std::get<2>(*input), inds...);
    }
}

template <typename T, typename Operand, typename... MultiIndex>
T compute_arithmetic(std::tuple<SubtractionOp, Operand> const *input, MultiIndex... inds) {
    return -compute_arithmetic<T>(std::get<1>(*input), inds...);
}

} // namespace detail

/**
 * @struct ArithmeticTensor
 *
 * This struct allows for lazy evaluation of simple arithmetic expressions without the need to create
 * several intermediate tensors. The goal is to have these be turned into simple arithmetic expressions
 * on the elements of the input tensors at compile time. Then, when an assignment is performed, the
 * elements of the tensors are looped and placed through this arithmetic expression.
 *
 * @tparam T The underlying type.
 * @tparam Rank The rank of the tensors.
 * @tparam Args The specific set of operations needed to perform the arithmetic operations.
 */
template <typename T, size_t rank, typename... Args>
struct ArithmeticTensor : tensor_base::CoreTensor {
  protected:
    /**
     * @property _tuple
     *
     * @brief The underlying syntax tree of the operation.
     *
     * This contains the operations as well as their arguments.
     */
    std::tuple<Args...> _tuple;

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed ArithmeticTensor)"};

  public:
    /**
     * @typedef ValueType
     *
     * @brief The data type of the elements represented by this matrix.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef tuple_type
     *
     * @brief Type of the tuple that holds the data needed to perform the arithmetic operations.
     */
    using tuple_type = std::tuple<Args...>; // NOLINT(readability-identifier-naming)

    /**
     * Construct a new ArithmeticTensor using the given tuple and the dimensions.
     *
     * @param input The tuple that defines the arithmetic operation.
     * @param dims The dimensions of the tensor.
     */
    ArithmeticTensor(std::tuple<Args...> const &input, Dim<Rank> dims) : _tuple{input}, _dims{dims} { ; }

    /**
     * @brief Evaluate the tensor at a given index.
     */
    template <typename... MultiIndex>
    T operator()(MultiIndex... inds) const {
        return detail::compute_arithmetic<T>(&_tuple, inds...);
    }

    /**
     * @brief Gets the tuple that defines the arithmetic operation.
     */
    std::tuple<Args...> const *get_tuple() const { return &_tuple; }

    /**
     * @brief Gets the dimensions of the tensor.
     *
     * @return The dimensions of the tensor.
     */
    Dim<Rank> dims() const { return _dims; }

    /**
     * @brief Gets the dimension of a tensor along a given axis.
     *
     * @param d The axis to query.
     *
     * @return The dimension along the axis being queried.
     */
    [[nodiscard]] size_t dim(int d) const { return _dims[d]; }

    /**
     * @brief Gets the name of the tensor.
     *
     * @return The name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * @brief Sets the name of the tensor.
     *
     * @param new_name The new name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; }

    /**
     * Indicates that the tensor is contiguous.
     */
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return false; }
};

} // namespace einsums

#ifndef DOXYGEN
#    define OPERATOR(op, name)                                                                                                             \
        template <typename T, size_t Rank, typename... Args1, typename... Args2>                                                           \
        auto operator op(const einsums::ArithmeticTensor<T, Rank, Args1...> &left,                                                         \
                         const einsums::ArithmeticTensor<T, Rank, Args2...> &right)                                                        \
            ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, const std::tuple<Args2...> *> {                       \
            return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, const std::tuple<Args2...> *>(                   \
                std::make_tuple(name(), left.get_tuple(), right.get_tuple()), left.dims());                                                \
        }                                                                                                                                  \
        template <einsums::CoreBasicTensorConcept LeftType, typename T, size_t Rank, typename... RightArgs>                                \
            requires requires {                                                                                                            \
                requires std::is_same_v<typename LeftType::ValueType, T>;                                                                  \
                requires LeftType::Rank == Rank;                                                                                           \
            }                                                                                                                              \
        auto operator op(const LeftType &left, const einsums::ArithmeticTensor<T, Rank, RightArgs...> &right)                              \
            ->einsums::ArithmeticTensor<T, Rank, name, const LeftType *, const std::tuple<RightArgs...> *> {                               \
            return einsums::ArithmeticTensor<T, Rank, name, const LeftType *, const std::tuple<RightArgs...> *>(                           \
                std::make_tuple(name(), &left, right.get_tuple()), left.dims());                                                           \
        }                                                                                                                                  \
        template <einsums::CoreBasicTensorConcept RightType, typename T, size_t Rank, typename... LeftArgs>                                \
            requires requires {                                                                                                            \
                requires std::is_same_v<typename RightType::ValueType, T>;                                                                 \
                requires RightType::Rank == Rank;                                                                                          \
            }                                                                                                                              \
        auto operator op(const einsums::ArithmeticTensor<T, Rank, LeftArgs...> &left, const RightType &right)                              \
            ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<LeftArgs...> *, const RightType *> {                               \
            return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<LeftArgs...> *, const RightType *>(                           \
                std::make_tuple(name(), left.get_tuple(), &right), left.dims());                                                           \
        }                                                                                                                                  \
        template <einsums::CoreBasicTensorConcept LeftType, einsums::CoreBasicTensorConcept RightType>                                     \
            requires requires {                                                                                                            \
                requires std::is_same_v<typename LeftType::ValueType, typename RightType::ValueType>;                                      \
                requires LeftType::Rank == RightType::Rank;                                                                                \
            }                                                                                                                              \
        auto operator op(const LeftType &left, const RightType &right)                                                                     \
            ->einsums::ArithmeticTensor<typename LeftType::ValueType, LeftType::Rank, name, const LeftType *, const RightType *> {         \
            return einsums::ArithmeticTensor<typename LeftType::ValueType, LeftType::Rank, name, const LeftType *, const RightType *>(     \
                std::make_tuple(name(), &left, &right), left.dims());                                                                      \
        }                                                                                                                                  \
        template <typename T, size_t Rank, typename... Args1>                                                                              \
            requires(!einsums::TensorConcept<T>)                                                                                           \
        auto operator op(const einsums::ArithmeticTensor<T, Rank, Args1...> &left, T &&right)                                              \
            ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, T> {                                                  \
            return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, T>(                                              \
                std::make_tuple(name(), left.get_tuple(), right), left.dims());                                                            \
        }                                                                                                                                  \
        template <einsums::CoreBasicTensorConcept LeftType, typename T>                                                                    \
            requires(!einsums::TensorConcept<T>)                                                                                           \
        auto operator op(const LeftType &left, T &&right)                                                                                  \
            ->einsums::ArithmeticTensor<typename LeftType::ValueType, LeftType::Rank, name, const LeftType *,                              \
                                        typename LeftType::ValueType> {                                                                    \
            return einsums::ArithmeticTensor<typename LeftType::ValueType, LeftType::Rank, name, const LeftType *,                         \
                                             typename LeftType::ValueType>(std::make_tuple(name(), &left, right), left.dims());            \
        }                                                                                                                                  \
        template <typename T, size_t Rank, typename... Args2>                                                                              \
            requires(!einsums::TensorConcept<T>)                                                                                           \
        auto operator op(T &&left, const einsums::ArithmeticTensor<T, Rank, Args2...> &right)                                              \
            ->einsums::ArithmeticTensor<T, Rank, name, T, const std::tuple<Args2...> *> {                                                  \
            return einsums::ArithmeticTensor<T, Rank, name, T, const std::tuple<Args2...> *>(                                              \
                std::make_tuple(name(), left, right.get_tuple()), right.dims());                                                           \
        }                                                                                                                                  \
        template <einsums::CoreBasicTensorConcept RightType, typename T>                                                                   \
            requires(!einsums::TensorConcept<T>)                                                                                           \
        auto operator op(T &&left, const RightType &right)                                                                                 \
            ->einsums::ArithmeticTensor<typename RightType::ValueType, RightType::Rank, name, T, const RightType *> {                      \
            return einsums::ArithmeticTensor<typename RightType::ValueType, RightType::Rank, name, T, const RightType *>(                  \
                std::make_tuple(name(), left, &right), right.dims());                                                                      \
        }

OPERATOR(+, einsums::detail::AdditionOp)
OPERATOR(-, einsums::detail::SubtractionOp)
OPERATOR(*, einsums::detail::MultiplicationOp)
OPERATOR(/, einsums::detail::DivisionOp)

#    undef OPERATOR

template <typename T, size_t Rank, typename... Args>
auto operator-(einsums::ArithmeticTensor<T, Rank, Args...> const &&tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, std::tuple<Args...> const *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, std::tuple<Args...> const *>(
        std::make_tuple(einsums::detail::SubtractionOp(), tensor.get_tuple()));
}

template <einsums::CoreTensorConcept TensorType>
auto operator-(TensorType const &tensor)
    -> einsums::ArithmeticTensor<typename TensorType::ValueType, TensorType::Rank, einsums::detail::SubtractionOp, TensorType const *> {
    return einsums::ArithmeticTensor<typename TensorType::ValueType, TensorType::Rank, einsums::detail::SubtractionOp, TensorType const *>(
        std::make_tuple(einsums::detail::SubtractionOp(), &tensor), tensor.dims());
}

template <typename T, size_t Rank, typename... Args>
auto operator-(einsums::ArithmeticTensor<T, Rank, Args...> const &tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, std::tuple<Args...> const *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, std::tuple<Args...> const *>(
        std::make_tuple(einsums::detail::SubtractionOp(), tensor.get_tuple()), tensor.dims());
}
#endif