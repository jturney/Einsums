//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <fmt/format.h>

#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace einsums {

/**
 * @class RuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 *
 * This kind of tensor is unable to be used in many of the
 *
 * @tparam T The data type stored by the tensor.
 */
template <typename T>
struct RuntimeTensor : public tensor_base::CoreTensor, tensor_base::RuntimeTensorNoType, design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef Vector
     *
     * @brief Represents how the data is stored in the tensor.
     */
    using Vector = VectorData<T>;

    /**
     * @typedef ValueType
     *
     * @brief Represents the data type stored in the tensor.
     */
    using ValueType = T;

    RuntimeTensor() = default;

    /**
     * @brief Default copy constructor.
     */
    RuntimeTensor(RuntimeTensor<T> const &copy) = default;

    /**
     * @brief Create a new runtime tensor with the given name and dimensions.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    RuntimeTensor(std::string name, std::vector<size_t> const &dims) : _rank{dims.size()}, _name{name}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    /**
     * @brief Create a new runtime tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    explicit RuntimeTensor(std::vector<size_t> const &dims) : _rank{dims.size()}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions using an initializer list.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor as an initializer list.
     */
    RuntimeTensor(std::string name, std::initializer_list<size_t> dims) : RuntimeTensor(name, std::vector<size_t>(dims)) {}

    /**
     * @brief Create a new runtime tensor with the given dimensions using an initializer list.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     */
    explicit RuntimeTensor(std::initializer_list<size_t> dims) : RuntimeTensor(std::vector<size_t>(dims)) {}

    /**
     * @brief Copy a tensor into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor to copy.
     */
    template <size_t Rank>
    RuntimeTensor(Tensor<T, Rank> const &copy) : _rank{Rank}, _dims(Rank), _strides(Rank), _name{copy.name()} {
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]    = copy.dim(i);
            _strides[i] = copy.stride(i);
        }

        _data.resize(copy.size());

        std::memcpy(_data.data(), copy.data(), copy.size() * sizeof(T));
    }

    /**
     * @brief Copy a tensor view into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor view to copy.
     */
    template <size_t Rank>
    RuntimeTensor(TensorView<T, Rank> const &copy) : _rank{Rank}, _dims(Rank), _strides(Rank) {
        size_t size = 1;

        for (int i = Rank - 1; i >= 0; i--) {
            _strides[i] = size;
            _dims[i]    = (size_t)copy.dim(i);
            size *= _dims[i];
        }

        _data.resize(size);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {
            size_t hold = sentinel, ord = 0;
            for (int i = 0; i < Rank; i++) {
                ord += copy.stride(i) * (hold / _strides[i]);
                hold %= _strides[i];
            }
            _data[sentinel] = copy.data()[ord];
        }
    }

    // HIP clang doesn't like it when this is defaulted.
    virtual ~RuntimeTensor() {}

    /**
     * @brief Set all of the data in the tensor to zero.
     */
    virtual void zero() { std::memset(_data.data(), 0, _data.size() * sizeof(T)); }

    /**
     * @brief Set all of the data in the tensor to the same value.
     *
     * @param val The value to fill the tensor with.
     */
    virtual void set_all(T val) { std::fill(_data.begin(), _data.end(), val); }

    /**
     * @brief Get the pointer to the stored data.
     */
    T *data() { return _data.data(); }

    /**
     * @copydoc data()
     */
    T const *data() const { return _data.data(); }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(Storage const &index) {
        return &(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(Storage const &index) const {
        return &(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(Storage const &index) {
        if (index.size() < rank()) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(Storage const &index) const {
        if (index.size() < rank()) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <typename... Args>
    T *data(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <typename... Args>
    T const *data(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If fewer indices than necessary
     * are passed, it will throw an error. This will hopefully change in the future to allow for
     * the creation of views. It will still throw an error when too many arguments are passed.
     *
     * @param args The index to use for the subscript.
     *
     * @todo std::variant can't handle references. We may be able to make our own, but for right now,
     * this will not be able to handle the wrong number of arguments.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    T &operator()(Args... args) {
        if (sizeof...(Args) < rank()) {
            EINSUMS_THROW_EXCEPTION(todo_error,
                                    "Not yet implemented: can not handle fewer integral indices than rank in (non-const) runtime"
                                    "tensor.");
        } else if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If too few indices are passed,
     * it will create a view.
     *
     * @param args The index to use for the subscript.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    std::variant<T, RuntimeTensorView<T>> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};

        if (sizeof...(Args) < rank()) {
            std::vector<Range> slices(sizeof...(Args));

            for (int i = 0; i < sizeof...(Args); i++) {
                slices[i] = Range{-1, index[i]};
            }
            return std::variant<T, RuntimeTensorView<T>>((*this)(slices));
        } else {
            return std::variant<T, RuntimeTensorView<T>>(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
        }
    }

    /**
     * @brief Subscripts into the tensor with ranges.
     *
     * This function creates a view based on the ranges passed in.
     *
     * @param args The indices to use. Can be integers, ranges, or All.
     */
    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    RuntimeTensorView<T> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::tuple<Args...> arg_tuple = std::make_tuple(args...);
        std::vector<Range>  slices(sizeof...(Args));

        for_sequence<sizeof...(Args)>([&](auto n) {
            using Arg = std::tuple_element_t<n, std::tuple<Args...>>;
            if constexpr (std::is_same_v<Arg, AllT>) {
                slices[n] = Range{0, this->dim(n)};
            } else if constexpr (std::is_same_v<Arg, Range>) {
                slices[n] = std::get<n>(arg_tuple);
            } else if constexpr (std::is_integral_v<Arg>) {
                auto index = std::get<n>(arg_tuple);

                if (index < 0) {
                    index += this->dim(n);
                }

                slices[n] = Range{-1, index};
            }
        });

        return (*this)(slices);
    }

    /*
     * Special cases:
     *    Range{a, a + 1}: Keep the axis in the view. It will have dimension 1 and only have the a'th element. a can not be negative.
     *    Range{-1, a}: Remove the axis from the view. It will still affect the offset. a can not be negative.
     */
    /**
     * @brief Create a view with the specified parameters.
     *
     * Internal function only.
     *
     * @param slices A list of slices to use when specifying the parameters of the view.
     */
    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    /**
     * @brief Create a view with the specified parameters.
     *
     * Internal function only.
     *
     * @param slices A list of slices to use when specifying the parameters of the view.
     */
    RuntimeTensorView<T> const operator()(std::vector<Range> const &slices) const {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <size_t Rank>
    RuntimeTensor<T> &operator=(Tensor<T, Rank> const &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }
        std::memcpy(_data.data(), other.data(), other.size() * sizeof(T));

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensor<T> &operator=(Tensor<TOther, Rank> const &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < _data.size(); i++) {
            _data[i] = (T)other.data()[i];
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensor<T> &operator=(TensorView<TOther, Rank> const &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            std::array<size_t, Rank> index;

            einsums::sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = (T)subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensor<T> &operator=(RuntimeTensor<T> const &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            _data[sentinel] = other.data()[sentinel];
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    virtual RuntimeTensor<T> &operator=(RuntimeTensorView<T> const &other) {
        if (_dims != other.dims() || _rank != other.rank()) {
            if (_rank != other.rank()) {
                _rank = other.rank();
                _dims.resize(other.rank());
                _strides.resize(other.rank());
            }
            _data.resize(other.size());
            for (int i = 0; i < other.rank(); i++) {
                _dims[i] = other.dim(i);
            }
            size_t stride = 1;
            for (int i = _rank - 1; i >= 0; i--) {
                _strides[i] = stride;
                stride *= _dims[i];
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            thread_local std::vector<size_t> index(_rank);

            sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = other(index);
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther>
    RuntimeTensor<T> &operator=(RuntimeTensor<TOther> const &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            _data[sentinel] = other.data()[sentinel];
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther>
    RuntimeTensor<T> &operator=(RuntimeTensorView<TOther> const &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            thread_local std::vector<size_t> index(_rank);

            sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = other(index);
        }

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual RuntimeTensor<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

#ifndef DOXYGEN
#    define OPERATOR(OP, NAME)                                                                                                             \
        template <typename TOther>                                                                                                         \
        auto operator OP(const TOther &b)->RuntimeTensor<T> & {                                                                            \
            size_t elements  = size();                                                                                                     \
            T     *this_data = data();                                                                                                     \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                EINSUMS_OMP_PARALLEL_FOR_SIMD                                                                                              \
                for (size_t i = 0; i < elements; i++) {                                                                                    \
                    this_data[i] OP(T)(RemoveComplexT<T>) b;                                                                               \
                }                                                                                                                          \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                EINSUMS_OMP_PARALLEL_FOR                                                                                                   \
                for (size_t i = 0; i < elements; i++) {                                                                                    \
                    this_data[i] OP(T) b.real();                                                                                           \
                }                                                                                                                          \
            } else {                                                                                                                       \
                EINSUMS_OMP_PARALLEL_FOR                                                                                                   \
                for (size_t i = 0; i < elements; i++) {                                                                                    \
                    this_data[i] OP(T) b;                                                                                                  \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
        template <typename TOther>                                                                                                         \
        auto operator OP(const RuntimeTensor<TOther> &b)->RuntimeTensor<T> & {                                                             \
            if (size() != b.size()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(dimension_error, "tensors differ in size : {} {}", size(), b.size());                              \
            }                                                                                                                              \
            T            *this_data = this->data();                                                                                        \
            const TOther *b_data    = b.data();                                                                                            \
            size_t        elements  = size();                                                                                              \
            EINSUMS_OMP_PARALLEL_FOR_SIMD                                                                                                  \
            for (size_t sentinel = 0; sentinel < elements; sentinel++) {                                                                   \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    this_data[sentinel] OP(T)(RemoveComplexT<T>) b_data[sentinel];                                                         \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this_data[sentinel] OP(T) b_data[sentinel].real();                                                                     \
                } else {                                                                                                                   \
                    this_data[sentinel] OP(T) b_data[sentinel];                                                                            \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        template <typename TOther>                                                                                                         \
        auto operator OP(const RuntimeTensorView<TOther> &b)->RuntimeTensor<T> & {                                                         \
            if (b.rank() != rank()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error,                                                                               \
                                        "Can not perform the operation with runtime tensor and view of different ranks!");                 \
            }                                                                                                                              \
            if (b.dims() != dims()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(dimension_error,                                                                                   \
                                        "Can not perform the operation with runtime tensor and view of different dimensions!");            \
            }                                                                                                                              \
            size_t elements = size();                                                                                                      \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < elements; sentinel++) {                                                                   \
                thread_local std::vector<size_t> index(rank());                                                                            \
                sentinel_to_indices(sentinel, this->_strides, index);                                                                      \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    (*this)(index) OP(T)(RemoveComplexT<T>) b(index);                                                                      \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    (*this)(index) OP(T) b(index).real();                                                                                  \
                } else {                                                                                                                   \
                    (*this)(index) OP(T) b(index);                                                                                         \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#    undef OPERATOR
#endif

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensor into a rank-{} TensorView!", rank(), Rank);
        }
        Dim<Rank>    dims;
        Stride<Rank> strides;
        for (int i = Rank - 1; i >= 0; i--) {
            dims[i]    = _dims[i];
            strides[i] = _strides[i];
        }
        return TensorView<T, Rank>(_data.data(), dims, strides);
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensor into a rank-{} TensorView!", rank(), Rank);
        }
        Dim<Rank>    dims;
        Stride<Rank> strides;
        for (int i = Rank - 1; i >= 0; i--) {
            dims[i]    = _dims[i];
            strides[i] = _strides[i];
        }
        return TensorView<T, Rank>(const_cast<T const *>(_data.data()), dims, strides);
    }

    /**
     * @brief Get the length of the tensor along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    virtual size_t dim(int d) const {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }

    /**
     * @brief Get the dimensions of the tensor.
     */
    virtual std::vector<size_t> dims() const noexcept { return _dims; }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    virtual Vector const &vector_data() const { return _data; }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    virtual Vector &vector_data() { return _data; }

    /**
     * @brief Get the stride along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    virtual auto stride(int d) const -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides.at(d);
    }

    /**
     * @brief Return the strides of the tensor.
     */
    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    /**
     * @brief Create a rank-1 view of the tensor.
     */
    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t              size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        std::vector<size_t> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    virtual auto size() const -> size_t { return _data.size(); }

    /**
     * @brief Returns whether the tensor sees all of the underlying data.
     *
     * This type of tensor will always see all of its underlying data, so this will always be true.
     */
    virtual bool full_view_of_underlying() const noexcept { return true; }

    /**
     * @brief Get the rank of the tensor.
     */
    virtual size_t rank() const noexcept { return this->_rank; }

    /**
     * @brief Set the name of the tensor.
     *
     * @param new_name The new name of the tensor.
     */
    virtual void set_name(std::string const &new_name) { this->_name = new_name; }

    /**
     * @brief Get the name of the tensor.
     */
    virtual std::string const &name() const noexcept { return this->_name; }

  protected:
    /**
     * @property _data
     *
     * @brief The vector containing the data stored by the tensor.
     */
    Vector _data;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     */
    /**
     * @property _strides
     *
     * @brief The strides of the tensor.
     */
    std::vector<size_t> _dims, _strides;

    /**
     * @property _rank
     *
     * @brief The rank of the tensor.
     */
    size_t _rank{0};

    template <typename TOther>
    friend class RuntimeTensorView;

    template <typename TOther>
    friend class RuntimeTensor;
};

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
struct RuntimeTensorView : public tensor_base::CoreTensor,
                           public tensor_base::RuntimeTensorNoType,
                           public tensor_base::RuntimeTensorViewNoType,
                           public design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef ValueType
     *
     * @brief The data type stored by the tensor.
     */
    using ValueType = T;

    RuntimeTensorView() = default;

    /**
     * @brief Default copy constructor.
     *
     * @param copy The tensor to copy.
     */
    RuntimeTensorView(RuntimeTensorView<T> const &copy) = default;

    /**
     * @brief Creates a new view based on another view.
     *
     * This view and the other view will share the same data pointer.
     *
     * @param view The tensor to view.
     */
    RuntimeTensorView(RuntimeTensor<T> &view)
        : _data{view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor.
     *
     * @param view The tensor to view.
     */
    RuntimeTensorView(RuntimeTensor<T> const &view)
        : _data{(T *)view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    RuntimeTensorView(RuntimeTensor<T> const &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    RuntimeTensorView(RuntimeTensor<T> &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    RuntimeTensorView(RuntimeTensorView<T> &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    RuntimeTensorView(RuntimeTensorView<T> const &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    RuntimeTensorView(RuntimeTensor<T> &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    RuntimeTensorView(RuntimeTensor<T> const &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    RuntimeTensorView(RuntimeTensorView<T> &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    RuntimeTensorView(RuntimeTensorView<T> const &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> const &copy)
        : _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _data = const_cast<T *>(copy.data());

        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(Tensor<T, Rank> &copy)
        : _data{copy.data()}, _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{true}, _index_strides(Rank), _size{copy.size()} {
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = copy.stride(i);
        }
    }

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(Tensor<T, Rank> const &copy)
        : _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{true}, _index_strides(Rank), _size{copy.size()} {
        _data = const_cast<T *>(copy.data());
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = copy.stride(i);
        }
    }

    // HIP clang doesn't like it when this is defaulted.
    virtual ~RuntimeTensorView() {}

    /**
     * @brief Set all the entries in the tensor to zero.
     */
    virtual void zero() {
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0;

            for (int i = 0; i < _rank; i++) {
                ord += _strides[i] * (hold / _index_strides[i]);
                hold %= _index_strides[i];
            }

            _data[ord] = T{0.0};
        }
    }

    /**
     * @brief Fill the tensor with the specified value.
     *
     * @param val The value to fill the tensor with.
     */
    virtual void set_all(T val) {
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0;

            for (int i = 0; i < _rank; i++) {
                ord += _strides[i] * (hold / _index_strides[i]);
                hold %= _index_strides[i];
            }

            _data[ord] = val;
        }
    }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    T *data() { return _data; }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    T const *data() const { return _data; }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(Storage const &index) {
        return &(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(Storage const &index) const {
        return &(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does boudns checking.
     *
     * @param index The index to use for subscripting.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(Storage const &index) {
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does boudns checking.
     *
     * @param index The index to use for subscripting.
     */
    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(Storage const &index) const {
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * This version is mainly for rank-1 tensors. The index is treated as the first index
     * if it is not rank-1.
     *
     * @param index The index for the starting point.
     */
    T *data(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * This version is mainly for rank-1 tensors. The index is treated as the first index
     * if it is not rank-1.
     *
     * @param index The index for the starting point.
     */
    T const *data(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <typename... Args>
    T *data(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <typename... Args>
    T const *data(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, an error will be thrown. In the future, this may create a view
     * in this case instead. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     *
     * TODO: std::variant can't handle references. We might be able to make our own variant that can.
     * This new variant may also be able to replace HostDevReference.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    T &operator()(Args... args) {
        if (sizeof...(Args) < rank()) {
            EINSUMS_THROW_EXCEPTION(todo_error,
                                    "Not yet implemented: can not handle fewer integral indices than rank in (non-const) runtime tensor.");
        } else if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, a view will be created. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    std::variant<T, RuntimeTensorView<T>> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};

        if (sizeof...(Args) < rank()) {
            std::vector<Range> slices(sizeof...(Args));

            for (int i = 0; i < sizeof...(Args); i++) {
                slices[i] = Range{-1, index[i]};
            }
            return std::variant<T, RuntimeTensorView<T>>((*this)(slices));
        } else {
            return std::variant<T, RuntimeTensorView<T>>(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
        }
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    const RuntimeTensorView<T> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::tuple<Args...> arg_tuple = std::make_tuple(args...);
        std::vector<Range>  slices(sizeof...(Args));

        for_sequence<sizeof...(Args)>([&](auto n) {
            using Arg = std::tuple_element_t<n, std::tuple<Args...>>;
            if constexpr (std::is_same_v<Arg, AllT>) {
                slices[n] = Range{0, this->dim(n)};
            } else if constexpr (std::is_same_v<Arg, Range>) {
                slices[n] = std::get<n>(arg_tuple);
            } else if constexpr (std::is_integral_v<Arg>) {
                auto index = std::get<n>(arg_tuple);

                if (index < 0) {
                    index += this->dim(n);
                }

                slices[n] = Range{-1, index};
            }
        });

        return (*this)(slices);
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    RuntimeTensorView<T> operator()(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::tuple<Args...> arg_tuple = std::make_tuple(args...);
        std::vector<Range>  slices(sizeof...(Args));

        for_sequence<sizeof...(Args)>([&](auto n) {
            using Arg = std::tuple_element_t<n, std::tuple<Args...>>;
            if constexpr (std::is_same_v<Arg, AllT>) {
                slices[n] = Range{0, this->dim(n)};
            } else if constexpr (std::is_same_v<Arg, Range>) {
                slices[n] = std::get<n>(arg_tuple);
            } else if constexpr (std::is_integral_v<Arg>) {
                auto index = std::get<n>(arg_tuple);

                if (index < 0) {
                    index += this->dim(n);
                }

                slices[n] = Range{-1, index};
            }
        });

        return (*this)(slices);
    }

    /**
     * @brief Index into the tensor.
     *
     * This version is more intended for rank-1 tensors.
     *
     * @param index The index for the tensor.
     */
    T &operator()(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return _data[index * _strides[0]];
    }

    /**
     * @brief Index into the tensor.
     *
     * This version is more intended for rank-1 tensors.
     *
     * @param index The index for the tensor.
     */
    T const &operator()(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return _data[index * _strides[0]];
    }

    /*
     * Special cases:
     *    Rank{a, a}: Keep the axis in the view. It will have dimension 1 and only have the a'th element. a can not be negative.
     *    Rank{-1, a}: Remove the axis from the view. It will still affect the offset. a can not be negative.
     */
    /**
     * @brief Create a tensor view with the given parameters.
     *
     * This is an internal function only.
     *
     * @param slices The slices to use for the creation of the tensor view.
     */
    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    /**
     * @brief Create a tensor view with the given parameters.
     *
     * This is an internal function only.
     *
     * @param slices The slices to use for the creation of the tensor view.
     */
    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) const {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(Tensor<TOther, Rank> const &other) {
        if (_rank != Rank) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t                   hold = sentinel, ord = 0;
            std::array<size_t, Rank> index;

            for (int i = 0; i < Rank; i++) {
                size_t ind = hold / _index_strides[i];
                index[i]   = ind;
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(TensorView<TOther, Rank> const &other) {
        if (_rank != Rank) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a tensor view to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a tensor view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t                   hold = sentinel, ord = 0;
            std::array<size_t, Rank> index;

            for (int i = 0; i < Rank; i++) {
                size_t ind = hold / _index_strides[i];
                index[i]   = ind;
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensorView<T> &operator=(RuntimeTensor<T> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[sentinel];
        }

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensorView<T> &operator=(RuntimeTensorView<T> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                other_ord += ind * other.stride(i);
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[other_ord];
        }

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther>
    RuntimeTensorView<T> &operator=(RuntimeTensor<TOther> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[sentinel];
        }

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther>
    RuntimeTensorView<T> &operator=(RuntimeTensorView<TOther> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                other_ord += ind * other.stride(i);
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[other_ord];
        }

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual RuntimeTensorView<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

#ifndef DOXYGEN
#    define OPERATOR(OP, NAME)                                                                                                             \
        template <typename TOther>                                                                                                         \
        auto operator OP(const TOther &b)->RuntimeTensorView<T> & {                                                                        \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                      \
                size_t hold = sentinel, ord = 0;                                                                                           \
                                                                                                                                           \
                for (int i = 0; i < _rank; i++) {                                                                                          \
                    ord += _strides[i] * (hold / _index_strides[i]);                                                                       \
                    hold %= _index_strides[i];                                                                                             \
                }                                                                                                                          \
                                                                                                                                           \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    this->_data[ord] OP(T)(RemoveComplexT<T>) b;                                                                           \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this->_data[ord] OP(T) b.real();                                                                                       \
                } else {                                                                                                                   \
                    this->_data[ord] OP(T) b;                                                                                              \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        template <typename TOther>                                                                                                         \
        auto operator OP(const RuntimeTensor<TOther> &b)->RuntimeTensorView<T> & {                                                         \
            if (b.rank() != rank()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform the operation with runtime views of different ranks!");      \
            }                                                                                                                              \
            if (b.dims() != dims()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform the operation with runtime views of different dimensions!");     \
            }                                                                                                                              \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                      \
                size_t hold = sentinel, ord = 0, b_ord = 0;                                                                                \
                                                                                                                                           \
                for (int i = 0; i < _rank; i++) {                                                                                          \
                    ord += _strides[i] * (hold / _index_strides[i]);                                                                       \
                    b_ord += b.stride(i) * (hold / _index_strides[i]);                                                                     \
                    hold %= _index_strides[i];                                                                                             \
                }                                                                                                                          \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    this->_data[ord] OP(T)(RemoveComplexT<T>) b.data()[b_ord];                                                             \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this->_data[ord] OP(T) b.data()[b_ord].real();                                                                         \
                } else {                                                                                                                   \
                    this->_data[ord] OP(T) b.data()[b_ord];                                                                                \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        template <typename TOther>                                                                                                         \
        auto operator OP(const RuntimeTensorView<TOther> &b)->RuntimeTensorView<T> & {                                                     \
            if (b.rank() != rank()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform the operation with runtime views of different ranks!");      \
            }                                                                                                                              \
            if (b.dims() != dims()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform the operation with runtime views of different dimensions!");     \
            }                                                                                                                              \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                      \
                size_t hold = sentinel, ord = 0, b_ord = 0;                                                                                \
                                                                                                                                           \
                for (int i = 0; i < _rank; i++) {                                                                                          \
                    ord += _strides[i] * (hold / _index_strides[i]);                                                                       \
                    b_ord += b.stride(i) * (hold / _index_strides[i]);                                                                     \
                    hold %= _index_strides[i];                                                                                             \
                }                                                                                                                          \
                                                                                                                                           \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    this->_data[ord] OP(T)(RemoveComplexT<T>) b.data()[b_ord];                                                             \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this->_data[ord] OP(T) b.data()[b_ord].real();                                                                         \
                } else {                                                                                                                   \
                    this->_data[ord] OP(T) b.data()[b_ord];                                                                                \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#    undef OPERATOR
#endif

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }
        Dim<Rank>    dims;
        Stride<Rank> strides;
        for (int i = Rank - 1; i >= 0; i--) {
            dims[i]    = _dims[i];
            strides[i] = _strides[i];
        }
        return TensorView<T, Rank>(_data, dims, strides);
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }
        Dim<Rank>    dims;
        Stride<Rank> strides;
        for (int i = Rank - 1; i >= 0; i--) {
            dims[i]    = _dims[i];
            strides[i] = _strides[i];
        }
        return TensorView<T, Rank>(const_cast<T const *>(_data), dims, strides);
    }

    /**
     * @brief Get the length of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    virtual auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }

    /**
     * @brief Gets the dimensions of the tensor.
     */
    virtual auto dims() const noexcept -> std::vector<size_t> { return _dims; }

    /**
     * @brief Gets the stride of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    virtual auto stride(int d) const -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides.at(d);
    }

    /**
     * @brief Gets the strides of the tensor.
     */
    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    /**
     * @brief Gets the rank-1 veiw of the tensor.
     *
     * This does not work well for tensor views due to the variation in strides.
     */
    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t              size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        std::vector<size_t> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    virtual auto size() const noexcept -> size_t { return _size; }

    /**
     * @brief Checks whether the tensor sees all of the underlying data.
     */
    virtual bool full_view_of_underlying() const noexcept { return _full_view; }

    /**
     * @brief Returns the name of the tensor.
     */
    virtual std::string const &name() const { return _name; };

    /**
     * @brief Sets the name of the tensor.
     *
     * @param new_name The new name for the tensor.
     */
    virtual void set_name(std::string const &new_name) { _name = new_name; };

    /**
     * @brief Gets the rank of the tensor.
     */
    virtual size_t rank() const noexcept { return _rank; }

  protected:
    /**
     * @property _data
     *
     * @brief The pointer to the stored data.
     */
    T *_data;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed view)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor view.
     */
    /**
     * @property _strides
     *
     * @brief The strides of the tensor view.
     */
    /**
     * @property _index_strides
     *
     * @brief The strides as determined by the dimensions.
     *
     * This set of strides is used for converting a linear offset into a set of indices.
     */
    std::vector<size_t> _dims, _strides, _index_strides;

    /**
     * @property _rank
     *
     * @brief The rank of the tensor.
     */
    /**
     * @property _size
     *
     * @brief The number of elements of the tensor.
     */
    /**
     * @property _alloc_size
     *
     * @brief The number of elements including those that are skipped by the strides.
     */
    size_t _rank{0}, _size{0}, _alloc_size{0};

    /**
     * @property _full_view
     *
     * @brief Indicates whether the tensor sees all of the underlying data.
     */
    bool _full_view{false};
};

#ifndef DOXYGEN
template <einsums::FileOrOStream Output, einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void fprintln(Output &fp, AType const &A, einsums::TensorPrintOptions options = {}) {
    using namespace einsums;
    using T          = typename AType::ValueType;
    std::size_t Rank = A.rank();

    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};

        if constexpr (CoreTensorConcept<AType>) {
            if constexpr (!TensorViewConcept<AType>)
                fprintln(fp, "Type: In Core Tensor");
            else
                fprintln(fp, "Type: In Core Tensor View");
#    if defined(EINSUMS_COMPUTE_CODE)
        } else if constexpr (DeviceTensorConcept<AType>) {
            if constexpr (!TensorViewConcept<AType>)
                fprintln(fp, "Type: Device Tensor");
            else
                fprintln(fp, "Type: Device Tensor View");
#    endif
        } else if constexpr (DiskTensorConcept<AType>) {
            fprintln(fp, "Type: Disk Tensor");
        } else {
            fprintln(fp, "Type: {}", type_name<AType>());
        }

        fprintln(fp, "Data Type: {}", type_name<typename AType::ValueType>());

        if (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(fp, "Dims{{{}}}", oss.str().c_str());
        }

        if constexpr (einsums::BasicTensorConcept<AType>) {
            if (Rank > 0) {
                std::ostringstream oss;
                for (size_t i = 0; i < Rank; i++) {
                    oss << A.stride(i) << " ";
                }
                fprintln(fp, "Strides{{{}}}", oss.str());
            }
        }

        if (options.full_output) {
            fprintln(fp);

            if (Rank == 0) {
                T value = std::get<std::remove_cvref_t<T>>(A());

                std::ostringstream oss;
                oss << "              ";
                if constexpr (std::is_floating_point_v<T>) {
                    if (std::abs(value) < 1.0E-4) {
                        oss << fmt::format("{:14.4e} ", value);
                    } else {
                        oss << fmt::format("{:14.8f} ", value);
                    }
                } else if constexpr (IsComplexV<T>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(fp, "{}", oss.str());
                fprintln(fp);
#    if !defined(EINSUMS_COMPUTE_CODE)
            } else if constexpr (einsums::CoreTensorConcept<AType>) {
#    else
            } else if constexpr ((einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>)) {
#    endif
                if (Rank > 1) {
                    auto                final_dim = A.dim(A.rank() - 1);
                    auto                ndigits   = detail::ndigits(final_dim);
                    std::vector<size_t> index_strides;
                    dims_to_strides(A.dims(), index_strides);
                    size_t              size = A.size();
                    std::vector<size_t> indices(A.rank());

                    for (size_t sentinel = 0; sentinel < size; sentinel++) {

                        sentinel_to_indices(sentinel, index_strides, indices);

                        std::ostringstream oss;
                        for (int j = 0; j < final_dim; j++) {
                            if (j % options.width == 0) {
                                std::ostringstream tmp;
                                tmp << fmt::format("{}", fmt::join(indices, ", "));
                                if (final_dim >= j + options.width)
                                    oss << fmt::format("{:<14}", fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits,
                                                                             j + options.width - 1, ndigits));
                                else
                                    oss << fmt::format("{:<14}",
                                                       fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, final_dim - 1, ndigits));
                            }
                            T value = A(indices);
                            if (std::abs(value) > 1.0E+10) {
                                if constexpr (std::is_floating_point_v<T>)
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                                else if constexpr (IsComplexV<T>)
                                    oss << "\x1b[0;37;41m(" << fmt::format("{:14.8f} ", value.real()) << " + "
                                        << fmt::format("{:14.8f}i)", value.imag()) << "\x1b[0m";
                                else
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14d} ", value) << "\x1b[0m";
                            } else {
                                if constexpr (std::is_floating_point_v<T>) {
                                    if (std::abs(value) < 1.0E-4) {
                                        oss << fmt::format("{:14.4e} ", value);
                                    } else {
                                        oss << fmt::format("{:14.8f} ", value);
                                    }
                                } else if constexpr (IsComplexV<T>) {
                                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                                } else
                                    oss << fmt::format("{:14} ", value);
                            }
                            if (j % options.width == options.width - 1 && j != final_dim - 1) {
                                oss << "\n";
                            }
                        }
                        fprintln(fp, "{}", oss.str());
                        fprintln(fp);
                    }
                } else if (Rank == 1) {
                    size_t size = A.size();

                    for (size_t sentinel = 0; sentinel < size; sentinel++) {
                        std::ostringstream oss;
                        oss << "(";
                        oss << fmt::format("{}", sentinel);
                        oss << "): ";

                        T value = std::get<T>(A());
                        if (std::abs(value) > 1.0E+5) {
                            if constexpr (std::is_floating_point_v<T>)
                                oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "{:14.8f} ", value);
                            else if constexpr (IsComplexV<T>) {
                                oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "({:14.8f} + {:14.8f})",
                                                   value.real(), value.imag());
                            } else
                                oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "{:14} ", value);
                        } else {
                            if constexpr (std::is_floating_point_v<T>)
                                if (std::abs(value) < 1.0E-4) {
                                    oss << fmt::format("{:14.4e} ", value);
                                } else {
                                    oss << fmt::format("{:14.8f} ", value);
                                }
                            else if constexpr (IsComplexV<T>) {
                                oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                            } else
                                oss << fmt::format("{:14} ", value);
                        }

                        fprintln(fp, "{}", oss.str());
                    }
                }
            }
        }
    }
    fprintln(fp);
}

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void println(AType const &A, einsums::TensorPrintOptions options = {}) {
    fprintln(std::cout, A, options);
}

#endif

#if !defined(EINSUMS_WINDOWS) && !defined(DOXYGEN)
EINSUMS_TENSOR_EXPORT_T(RuntimeTensor, float);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensor, double);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensor, std::complex<float>);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensor, std::complex<double>);

EINSUMS_TENSOR_EXPORT_T(RuntimeTensorView, float);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensorView, double);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensorView, std::complex<float>);
EINSUMS_TENSOR_EXPORT_T(RuntimeTensorView, std::complex<double>);
#endif
} // namespace einsums