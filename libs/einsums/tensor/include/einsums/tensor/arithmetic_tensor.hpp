//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/concepts/tensor.hpp>

#include <cstddef>

namespace einsums {
namespace detail {

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct addition_op {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct subtraction_op {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct multiplication_op {};

/**
 * @struct AdditionOp
 *
 * @brief Represents the addition between two tensors or between a tensor and a scalar.
 */
struct division_op {};

template <typename T, typename... MultiIndex>
EINSUMS_HOST_DEVICE inline T compute_arithmetic(T scalar, MultiIndex... inds) {
    return scalar;
}

template <typename T1, template <typename, size_t> typename TensorType, typename T, size_t Rank, typename... MultiIndex>
EINSUMS_HOST_DEVICE inline T compute_arithmetic(const TensorType<T, Rank> *tensor, MultiIndex... inds) {
    return (*tensor)(inds...);
}

template <typename T, typename Op, typename Left, typename Right, typename... MultiIndex>
EINSUMS_HOST_DEVICE inline T compute_arithmetic(const std::tuple<Op, Left, Right> *input, MultiIndex... inds) {
    if constexpr (std::is_same_v<Op, addition_op>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) + compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, subtraction_op>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) - compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, multiplication_op>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) * compute_arithmetic<T>(std::get<2>(*input), inds...);
    } else if constexpr (std::is_same_v<Op, division_op>) {
        return compute_arithmetic<T>(std::get<1>(*input), inds...) / compute_arithmetic<T>(std::get<2>(*input), inds...);
    }
}

template <typename T, typename Operand, typename... MultiIndex>
EINSUMS_HOST_DEVICE inline T compute_arithmetic(const std::tuple<subtraction_op, Operand> *input, MultiIndex... inds) {
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
template <typename T, size_t Rank, typename... Args>
struct ArithmeticTensor : public virtual tensor_base::tensor<T, Rank>, virtual tensor_base::core_tensor {
  protected:
    std::tuple<Args...> _tuple;
    einsums::dim<Rank>  _dims;
    std::string         _name{"(unnamed ArithmeticTensor)"};

  public:
    using tuple_type = std::tuple<Args...>;

    ArithmeticTensor(const std::tuple<Args...> &input, einsums::dim<Rank> dims) : _tuple{input}, _dims{dims} { ; }

    template <typename... MultiIndex>
    T operator()(MultiIndex... inds) const {
        return detail::compute_arithmetic<T>(&_tuple, inds...);
    }

    const std::tuple<Args...> *get_tuple() const { return &_tuple; }

    einsums::dim<Rank> dims() const override { return _dims; }

    size_t dim(int d) const override { return _dims[d]; }

    const std::string &name() const override { return _name; }

    void set_name(const std::string &new_name) override { _name = new_name; }
};

} // namespace einsums

#define OPERATOR(op, name)                                                                                                                 \
    template <typename T, size_t Rank, typename... Args1, typename... Args2>                                                               \
    auto operator op(const einsums::ArithmeticTensor<T, Rank, Args1...> &left, const einsums::ArithmeticTensor<T, Rank, Args2...> &right)  \
        ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, const std::tuple<Args2...> *> {                           \
        return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, const std::tuple<Args2...> *>(                       \
            std::make_tuple(name(), left.get_tuple(), right.get_tuple()), left.dims());                                                    \
    }                                                                                                                                      \
    template <template <typename, size_t> typename LeftType, typename T, size_t Rank, typename... RightArgs>                               \
    auto operator op(const LeftType<T, Rank> &left, const einsums::ArithmeticTensor<T, Rank, RightArgs...> &right)                         \
        ->einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, const std::tuple<RightArgs...> *> {                          \
        return einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, const std::tuple<RightArgs...> *>(                      \
            std::make_tuple(name(), &left, right.get_tuple()), left.dims());                                                               \
    }                                                                                                                                      \
    template <template <typename, size_t> typename RightType, typename T, size_t Rank, typename... LeftArgs>                               \
    auto operator op(const einsums::ArithmeticTensor<T, Rank, LeftArgs...> &left, const RightType<T, Rank> &right)                         \
        ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<LeftArgs...> *, const RightType<T, Rank> *> {                          \
        return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<LeftArgs...> *, const RightType<T, Rank> *>(                      \
            std::make_tuple(name(), left.get_tuple(), &right), left.dims());                                                               \
    }                                                                                                                                      \
    template <template <typename, size_t> typename LeftType, template <typename, size_t> typename RightType, typename T, size_t Rank>      \
    auto operator op(const LeftType<T, Rank> &left, const RightType<T, Rank> &right)                                                       \
        ->einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, const RightType<T, Rank> *> {                                \
        return einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, const RightType<T, Rank> *>(                            \
            std::make_tuple(name(), &left, &right), left.dims());                                                                          \
    }                                                                                                                                      \
    template <typename T, size_t Rank, typename... Args1>                                                                                  \
    auto operator op(const einsums::ArithmeticTensor<T, Rank, Args1...> &left, T &&right)                                                  \
        ->einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, T> {                                                      \
        return einsums::ArithmeticTensor<T, Rank, name, const std::tuple<Args1...> *, T>(std::make_tuple(name(), left.get_tuple(), right), \
                                                                                         left.dims());                                     \
    }                                                                                                                                      \
    template <template <typename, size_t> typename LeftType, typename T, size_t Rank>                                                      \
    auto operator op(const LeftType<T, Rank> &left, T &&right)->einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, T> {   \
        return einsums::ArithmeticTensor<T, Rank, name, const LeftType<T, Rank> *, T>(std::make_tuple(name(), &left, right), left.dims()); \
    }                                                                                                                                      \
    template <typename T, size_t Rank, typename... Args2>                                                                                  \
    auto operator op(T &&left, const einsums::ArithmeticTensor<T, Rank, Args2...> &right)                                                  \
        ->einsums::ArithmeticTensor<T, Rank, name, T, const std::tuple<Args2...> *> {                                                      \
        return einsums::ArithmeticTensor<T, Rank, name, T, const std::tuple<Args2...> *>(std::make_tuple(name(), left, right.get_tuple()), \
                                                                                         right.dims());                                    \
    }                                                                                                                                      \
    template <template <typename, size_t> typename RightType, typename T, size_t Rank>                                                     \
    auto operator op(T &&left, const RightType<T, Rank> &right)->einsums::ArithmeticTensor<T, Rank, name, T, const RightType<T, Rank> *> { \
        return einsums::ArithmeticTensor<T, Rank, name, T, const RightType<T, Rank> *>(std::make_tuple(name(), left, &right),              \
                                                                                       right.dims());                                      \
    }

OPERATOR(+, einsums::detail::addition_op)
OPERATOR(-, einsums::detail::subtraction_op)
OPERATOR(*, einsums::detail::multiplication_op)
OPERATOR(/, einsums::detail::division_op)

#undef OPERATOR

template <typename T, size_t Rank, typename... Args>
auto operator-(const einsums::ArithmeticTensor<T, Rank, Args...> &&tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const std::tuple<Args...> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const std::tuple<Args...> *>(
        std::make_tuple(einsums::detail::subtraction_op(), tensor.get_tuple()));
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto operator-(const TensorType<T, Rank> &tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const TensorType<T, Rank> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const TensorType<T, Rank> *>(
        std::make_tuple(einsums::detail::subtraction_op(), &tensor), tensor.dims());
}

template <typename T, size_t Rank, typename... Args>
auto operator-(const einsums::ArithmeticTensor<T, Rank, Args...> &tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const std::tuple<Args...> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::subtraction_op, const std::tuple<Args...> *>(
        std::make_tuple(einsums::detail::subtraction_op(), tensor.get_tuple()), tensor.dims());
}