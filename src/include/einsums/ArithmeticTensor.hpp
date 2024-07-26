#pragma once

#include "einsums/_Common.hpp"

#include "einsums/utility/TensorTraits.hpp"

#include <cstddef>
#include <type_traits>

namespace einsums {

namespace detail {

// Forward declarations.
template <typename T, size_t Rank>
struct TensorBase;

struct AdditionOp {};
struct SubtractionOp {};
struct MultiplicationOp {};
struct DivisionOp {};

#ifndef __HIP__
#    define __host__
#    define __device__
#endif

template <typename T, typename... MultiIndex>
    requires(std::is_arithmetic_v<T>)
__host__ __device__ inline T compute_arithmetic(T scalar, MultiIndex... inds) {
    return scalar;
}

template <typename T1, template <typename, size_t> typename TensorType, typename T, size_t Rank, typename... MultiIndex>
__host__ __device__ inline T compute_arithmetic(const TensorType<T, Rank> *tensor, MultiIndex... inds) {
    return (*tensor)(inds...);
}

template <typename T, typename Op, typename Left, typename Right, typename... MultiIndex>
__host__ __device__ inline T compute_arithmetic(const std::tuple<Op, Left, Right> *input, MultiIndex... inds) {
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
__host__ __device__ inline T compute_arithmetic(const std::tuple<SubtractionOp, Operand> *input, MultiIndex... inds) {
    return -compute_arithmetic<T>(std::get<1>(*input), inds...);
}

} // namespace detail

template <typename T, size_t Rank, typename... Args>
struct ArithmeticTensor : public virtual tensor_props::TensorBase<T, Rank> {
  protected:
    std::tuple<Args...> _tuple;
    Dim<Rank>           _dims;

  public:
    using tuple_type = std::tuple<Args...>;

    ArithmeticTensor(const std::tuple<Args...> &input, Dim<Rank> dims) : _tuple{input}, _dims{dims} { ; }

    template <typename... MultiIndex>
    __host__ __device__ T operator()(MultiIndex... inds) const {
        return detail::compute_arithmetic<T>(&_tuple, inds...);
    }

    const std::tuple<Args...> *get_tuple() const { return &_tuple; }

    Dim<Rank> dims() const override { return _dims; }

    size_t dim(int d) const override { return _dims[d]; }
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

OPERATOR(+, einsums::detail::AdditionOp)
OPERATOR(-, einsums::detail::SubtractionOp)
OPERATOR(*, einsums::detail::MultiplicationOp)
OPERATOR(/, einsums::detail::DivisionOp)

#undef OPERATOR

template <typename T, size_t Rank, typename... Args>
auto operator-(const einsums::ArithmeticTensor<T, Rank, Args...> &&tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const std::tuple<Args...> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const std::tuple<Args...> *>(
        std::make_tuple(einsums::detail::SubtractionOp(), tensor.get_tuple()));
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto operator-(const TensorType<T, Rank> &tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const TensorType<T, Rank> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const TensorType<T, Rank> *>(
        std::make_tuple(einsums::detail::SubtractionOp(), &tensor), tensor.dims());
}

template <typename T, size_t Rank, typename... Args>
auto operator-(const einsums::ArithmeticTensor<T, Rank, Args...> &tensor)
    -> einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const std::tuple<Args...> *> {
    return einsums::ArithmeticTensor<T, Rank, einsums::detail::SubtractionOp, const std::tuple<Args...> *>(
        std::make_tuple(einsums::detail::SubtractionOp(), tensor.get_tuple()), tensor.dims());
}