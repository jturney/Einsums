//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>
#include <Einsums/Utilities/Tuple.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(EINSUMS_COMPUTE_CODE)
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {
#ifndef DOXYGEN
// Forward declaration of the Tensor printing function.
template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options = {});

template <FileOrOStream Output, RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options = {});
#endif

/**
 * @brief Represents a general tensor
 *
 * @tparam T data type of the underlying tensor data
 * @tparam Rank the rank of the tensor
 */
template <typename T, size_t rank>
struct Tensor : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef Vector
     *
     * This represents the internal storage method of the tensor.
     */
    using Vector = VectorData<T>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    Tensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    Tensor(Tensor const &) = default;

    /**
     * @brief Destroy the Tensor object.
     */
    ~Tensor() = default;

    /**
     * @brief Default move constructor.
     */
    Tensor(Tensor &&) = default;

    /**
     * @brief Default move assignment.
     */
    Tensor &operator=(Tensor &&other) = default;

    /**
     * @brief Construct a new Tensor object with the given name and dimensions.
     *
     * Constructs a new Tensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = Tensor("A", 3, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <typename... Dims>
    Tensor(std::string name, Dims... dims) : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        size_t size = dims_to_strides(_dims, _strides);

        // Resize the data structure
        _data.resize(size);
    }

    /**
     * @brief Construct a new Tensor object. Moving \p existingTensor data to the new tensor.
     *
     * This constructor is useful for reshaping a tensor. It does not modify the underlying
     * tensor data. It only creates new mapping arrays for how the data is viewed.
     *
     * @code
     * auto A = Tensor("A", 27); // Creates a rank-1 tensor of 27 elements
     * auto B = Tensor(std::move(A), "B", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     * // At this point A is no longer valid.
     * @endcode
     *
     * Supports using -1 for one of the ranks to automatically compute the dimension of it.
     *
     * @code
     * auto A = Tensor("A", 27);
     * auto B = Tensor(std::move(A), "B", 3, -1, 3); // Automatically determines that -1 should be 3.
     * @endcode
     *
     * @tparam OtherRank The rank of \p existingTensor can be different from the rank of the new tensor
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param existingTensor The existing tensor that holds the tensor data.
     * @param name The name of the new tensor
     * @param dims The dimensionality of each rank of the new tensor.
     */
    template <size_t OtherRank, typename... Dims>
    explicit Tensor(Tensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
        : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...}, _data(std::move(existingTensor._data)) {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we
        // compute this dimensionality of this "0" index for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
        }

        if (nfound == 1) {
            size_t size{1};
            for (auto [i, dim] : enumerate(_dims)) {
                if (i != location) {
                    size *= dim;
                }
            }
            if (size > existingTensor.size()) {
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Size of new tensor is larger than the parent tensor.");
            }
            _dims[location] = existingTensor.size() / size;
        }

        size_t size = dims_to_strides(_dims, _strides);

        // Check size
        if (_data.size() != size) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Provided dims to not match size of parent tensor");
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit Tensor(Dim<Rank> dims) : _dims{std::move(dims)} {
        size_t size = dims_to_strides(_dims, _strides);

        // Resize the data structure
        _data.resize(size);
    }

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    Tensor(TensorView<T, rank> const &other) : _name{other.name()}, _dims{other.dims()} {
        size_t size = dims_to_strides(_dims, _strides);

        // Resize the data structure
        _data.resize(size);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            thread_local std::array<size_t, Rank> index;

            sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = subscript_tensor(other, index);
        }
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    void resize(Dim<Rank> dims) {
        if (_dims == dims) {
            return;
        }

        _dims = dims;

        size_t size = dims_to_strides(_dims, _strides);

        // Resize the data structure
        _data.resize(size);
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    template <typename... Dims>
        requires((std::is_arithmetic_v<Dims> && ... && (sizeof...(Dims) == Rank)))
    void resize(Dims... dims) {
        resize(Dim<Rank>{static_cast<size_t>(dims)...});
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() { memset(_data.data(), 0, sizeof(T) * _data.size()); }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) { std::fill(_data.begin(), _data.end(), value); }

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    T *data() { return _data.data(); }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    T const *data() const { return _data.data(); }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     * @code
     * auto A = Tensor("A", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     *
     * double* A_pointer = A.data(1, 2, 3); // Returns the pointer to element (1, 2, 3) in A.
     * @endcode
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to ptrdiff_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
            requires NoneOfType<Dim<Rank>, MultiIndex...>;
            requires NoneOfType<Offset<Rank>, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN)
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = std::array{static_cast<ptrdiff_t>(index)...};

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index_list);

        return &_data[ordinal];
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     *              Negative indices are taken to be an offset from the end of the axis.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
            requires NoneOfType<Dim<Rank>, MultiIndex...>;
            requires NoneOfType<Offset<Rank>, MultiIndex...>;
            requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...);
        }
    auto operator()(MultiIndex &&...index) const -> T const & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, std::forward<MultiIndex>(index)...);

        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam int_type The type of integer used for the indices.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return const T&
     */
    template <typename int_type>
        requires requires { requires std::is_integral_v<int_type>; }
    auto subscript(std::array<int_type, Rank> const &index) const -> T const & {
        size_t ordinal = indices_to_sentinel(_strides, index);
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to size_t.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
            requires NoneOfType<Dim<Rank>, MultiIndex...>;
            requires NoneOfType<Offset<Rank>, MultiIndex...>;
            requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...);
        }
    auto subscript(MultiIndex &&...index) const -> T const & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);

        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     *              Negative indices are taken to be an offset from the end of the axis.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
            requires NoneOfType<Dim<Rank>, MultiIndex...>;
            requires NoneOfType<Offset<Rank>, MultiIndex...>;
            requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...);
        }
    auto operator()(MultiIndex &&...index) -> T & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, std::forward<MultiIndex>(index)...);
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam int_type The type of integer used for the indices.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return T&
     */
    template <typename int_type>
        requires requires { requires std::is_integral_v<int_type>; }
    auto subscript(std::array<int_type, Rank> const &index) -> T & {
        size_t ordinal = indices_to_sentinel(_strides, index);
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
            requires NoneOfType<Dim<Rank>, MultiIndex...>;
            requires NoneOfType<Offset<Rank>, MultiIndex...>;
            requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...);
        }
    auto subscript(MultiIndex &&...index) -> T & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when some of the indices are All or Range. It then constructs a view of the tensor with those properties.
     *
     * @tparam MultiIndex Data type of the indices.
     * @param index The indices.
     * @return A tensor view with the appropriate starting point and dimensions.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto operator()(MultiIndex... index) -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");
        auto const &indices = std::forward_as_tuple(index...);

        Offset<Rank>                                                                         offsets;
        Stride<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> strides{};
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, ptrdiff_t>) {
                auto tmp = static_cast<ptrdiff_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                offsets[i] = tmp;
            } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                strides[counter] = _strides[i];
                dims[counter]    = _dims[i];
                counter++;
            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range       = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = _dims[i] + range[1];
                    range[1]  = temp;
                }
                dims[counter]    = range[1] - range[0];
                strides[counter] = _strides[i];
                counter++;
            }
        });

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{*this, std::move(dims), offsets,
                                                                                                           strides};
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when some of the indices are All or Range. It then constructs a view of the tensor with those properties.
     *
     * @tparam MultiIndex Data type of the indices.
     * @param index The indices.
     * @return A tensor view with the appropriate starting point and dimensions.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto operator()(MultiIndex... index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        auto const &indices = std::forward_as_tuple(index...);

        Offset<Rank>                                                                         offsets;
        Stride<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> strides{};
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, ptrdiff_t>) {
                auto tmp = static_cast<ptrdiff_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                offsets[i] = tmp;
            } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                strides[counter] = _strides[i];
                dims[counter]    = _dims[i];
                counter++;
            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range       = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = _dims[i] + range[1];
                    range[1]  = temp;
                }
                dims[counter]    = range[1] - range[0];
                strides[counter] = _strides[i];
                counter++;
            }
        });

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{*this, std::move(dims), offsets,
                                                                                                           strides};
    }

    /**
     * Subscripts into the tensor, creating a view based on the given ranges
     *
     * @tparam MultiIndex The types of the Ranges.
     * @param index The ranges.
     * @return The view based on these ranges.
     */
    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> TensorView<T, Rank> {
        Dim<Rank>    dims{};
        Offset<Rank> offset{};
        Stride<Rank> stride = _strides;

        auto ranges = arguments::get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
            auto range = ranges[r];
            offset[r]  = range[0];
            if (range[1] < 0) {
                auto temp = _dims[r] + range[1];
                range[1]  = temp;
            }
            dims[r] = range[1] - range[0];
        }

        return TensorView<T, Rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
    }

    /**
     * @copydoc Tensor<T, Rank>::operator(MultiIndex...) -> T&
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    T &operator()(Container const &index) {
        if (index.size() < Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index);
        return _data[ordinal];
    }

    /**
     * @copydoc Tensor<T, Rank>::operator(MultiIndex...) -> T&
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    const T &operator()(Container const &index) const {
        if (index.size() < Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index);
        return _data[ordinal];
    }

    /**
     * Copy the data from one tensor into this.
     *
     * @param other The tensor to copy.
     */
    auto operator=(Tensor const &other) -> Tensor & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _dims = other._dims;

            size_t size = dims_to_strides(_dims, _strides);

            // Resize the data structure
            _data.resize(size);
        }

        std::copy(other._data.begin(), other._data.end(), _data.begin());

        return *this;
    }

    /**
     * Cast the data from one tensor while copying its data into this tensor.
     *
     * @param other The tensor to cast and copy.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(Tensor<TOther, Rank> const &other) -> Tensor & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0) {
                realloc = true;
            } else if (dim(i) != other.dim(i)) {
                if constexpr (Rank != 1) {
                    EINSUMS_THROW_EXCEPTION(dimension_error, "dimensions do not match (this){} (other){}", dim(i), other.dim(i));
                } else {
                    realloc = true;
                }
            }
        }

        if (realloc) {
            _dims = other._dims;

            size_t size = dims_to_strides(_dims, _strides);

            // Resize the data structure
            _data.resize(size);
        }

        size_t size = this->size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            _data[sentinel] = other.data()[sentinel];
        }

        return *this;
    }

    /**
     * Copy the data from a tensor of a different kind into this one.
     */
    template <TensorConcept OtherTensor>
        requires requires {
            requires !BasicTensorConcept<OtherTensor>;
            requires SameRank<Tensor, OtherTensor>;
            requires CoreTensorConcept<OtherTensor>;
        }
    auto operator=(OtherTensor const &other) -> Tensor & {
        size_t size = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            thread_local std::array<size_t, Rank> index;
            sentinel_to_indices(sentinel, _strides, index);
            _data[sentinel] = subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * Cast the data from a tensor view while copying into this tensor.
     */
    template <typename TOther>
    auto operator=(TensorView<TOther, Rank> const &other) -> Tensor & {
        size_t size = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            thread_local std::array<size_t, Rank> index;
            sentinel_to_indices(sentinel, _strides, index);
            _data[sentinel] = other.subscript(index);
        }

        return *this;
    }

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * Copy the data from the device into this tensor.
     */
    auto operator=(DeviceTensor<T, Rank> const &other) -> Tensor<T, Rank> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _dims = other.dims();

            size_t size = dims_to_strides(_dims, _strides);

            // Resize the data structure
            _data.resize(size);
        }

        hip_catch(hipMemcpy(_data.data(), other.gpu_data(), _strides[0] * _dims[0] * sizeof(T), hipMemcpyDeviceToHost));

        return *this;
    }
#endif

    /**
     * Fill this tensor with a value.
     */
    auto operator=(T const &fill_value) -> Tensor & {
        set_all(fill_value);
        return *this;
    }

#ifndef DOXYGEN
#    define OPERATOR(OP)                                                                                                                   \
        auto operator OP(const T &b)->Tensor<T, Rank> & {                                                                                  \
            const size_t elements = this->size();                                                                                          \
            T           *array    = this->data();                                                                                          \
            EINSUMS_OMP_PARALLEL_FOR_SIMD                                                                                                  \
            for (size_t i = 0; i < elements; i++) {                                                                                        \
                array[i] OP b;                                                                                                             \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        auto operator OP(const Tensor<T, Rank> &b)->Tensor<T, Rank> & {                                                                    \
            if (size() != b.size()) {                                                                                                      \
                EINSUMS_THROW_EXCEPTION(dimension_error, "tensors differ in size : {} {}", size(), b.size());                              \
            }                                                                                                                              \
            EINSUMS_OMP_PARALLEL {                                                                                                         \
                auto tid       = omp_get_thread_num();                                                                                     \
                auto chunksize = _data.size() / omp_get_num_threads();                                                                     \
                auto abegin    = _data.begin() + chunksize * tid;                                                                          \
                auto bbegin    = b._data.begin() + chunksize * tid;                                                                        \
                auto aend      = (tid == omp_get_num_threads() - 1) ? _data.end() : abegin + chunksize;                                    \
                auto j         = bbegin;                                                                                                   \
                EINSUMS_OMP_SIMD for (auto i = abegin; i < aend; i++) {                                                                    \
                    (*i) OP(*j);                                                                                                           \
                    j++;                                                                                                                   \
                }                                                                                                                          \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#    undef OPERATOR
#endif

    /**
     * Get the dimension of the tensor along a given axis.
     */
    size_t dim(int d) const {
        // Add support for negative indices.
        if (d < 0) {
            d += Rank;
        }
        return _dims[d];
    }

    /**
     * Get all the dimensions of the tensor.
     */
    Dim<Rank> dims() const { return _dims; }

    /**
     * Get the internal vector containing the tensor's data.
     */
    auto vector_data() const -> Vector const & { return _data; }

    /// @copydoc Tensor<T,Rank>::vector_data() const
    auto vector_data() -> Vector & { return _data; }

    /**
     * Get the stride along a given axis.
     */
    size_t stride(int d) const {
        if (d < 0) {
            d += Rank;
        }
        return _strides[d];
    }

    /**
     * Get the strides of this tensor.
     */
    Stride<Rank> strides() const { return _strides; }

    /**
     * Flatten out the tensor.
     */
    auto to_rank_1_view() const -> TensorView<T, 1> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return TensorView<T, 1>{*this, dim};
    }

    /**
     * Returns the linear size of the tensor.
     */
    [[nodiscard]] size_t size() const { return _dims[0] * _strides[0]; }

    /**
     * Indicates that the tensor is contiguous.
     */
    bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; };

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; };

  private:
    std::string  _name{"(unnamed)"};
    Dim<Rank>    _dims;
    Stride<Rank> _strides;
    Vector       _data;

    template <typename T_, size_t Rank_>
    friend struct TensorView;

    template <typename T_, size_t OtherRank>
    friend struct Tensor;
};

/**
 * @struct Tensor<T, 0>
 *
 * @brief Represents a zero-rank tensor. It has a special implementation since it is essentially a scalar.
 *
 * @tparam T The data type being stored.
 */
template <typename T>
struct Tensor<T, 0> final : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {

    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = 0;

    /**
     * Default constructor
     */
    Tensor() = default;

    /**
     * Default copy constructor
     */
    Tensor(Tensor const &) = default;

    /**
     * Default move constructor
     */
    Tensor(Tensor &&) noexcept = default;

    /**
     * Default destructor
     */
    ~Tensor() = default;

    /**
     * Create a new zero-rank tensor with the given name.
     */
    explicit Tensor(std::string name) : _name{std::move(name)} {};

    /**
     * Create a new zero-rank tensor with the given dimensions. Since it is zero-rank,
     * the dimensions will be empty, and are ignored.
     */
    explicit Tensor(Dim<0> _ignore) {}

    /**
     * Get the pointer to the data stored by this tensor.
     */
    T *data() { return &_data; }

    /**
     * @copydoc Tensor<T,0>::data()
     */
    T const *data() const { return &_data; }

    /**
     * Copy assignment.
     */
    auto operator=(Tensor const &other) -> Tensor & {
        _data = other._data;
        return *this;
    }

    /**
     * Set the value of the tensor to the value passed in.
     */
    auto operator=(T const &other) -> Tensor & {
        _data = other;
        return *this;
    }

#ifndef DOXYGEN
#    if defined(OPERATOR)
#        undef OPERATOR
#    endif
#    define OPERATOR(OP)                                                                                                                   \
        auto operator OP(const T &other)->Tensor & {                                                                                       \
            _data OP other;                                                                                                                \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#    undef OPERATOR
#endif

    /**
     * Cast the tensor to a scalar.
     */
    operator T() const { return _data; } // NOLINT

    /**
     * Cast the tensor to a scalar.
     */
    operator T &() { return _data; } // NOLINT

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * Get the dimension of the tensor. Always returns 1.
     */
    size_t dim(int) const { return 1; }

    /**
     * Get the dimensions of the tensor. The result is empty.
     */
    Dim<0> dims() const { return Dim{}; }

    /**
     * Indicates that the tensor is contiguous.
     */
    bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the stride of the tensor. Always returns 1.
     */
    size_t stride(int d) const { return 0; }

    /**
     * Get the strides of the tensor. The result is empty.
     */
    Stride<0> strides() const { return Stride{}; }

  private:
    /**
     * @var _name
     *
     * The name of the tensor used for printing.
     */
    std::string _name{"(Unnamed)"};

    /**
     * The value stored by the tensor.
     */
    T _data{};
};

/**
 * @struct TensorView
 *
 * @brief Represents a view of a tensor, which may have different dimensions and start at a different index.
 *
 * @tparam T The data type being stored.
 * @tparam Rank The rank of the view.
 */
template <typename T, size_t rank>
struct TensorView final : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor view.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef underlying_type
     *
     * @brief The type of tensor this view views.
     */
    using underlying_type = Tensor<T, Rank>;

    TensorView() = delete;

    /**
     * Default copy constructor.
     */
    TensorView(TensorView const &) = default;

    /**
     * Default destructor.
     */
    ~TensorView() = default;

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, typename... Args>
    explicit TensorView(Tensor<T, OtherRank> const &other, Dim<Rank> const &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<Tensor<T, OtherRank> &>(other), args...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, typename... Args>
    explicit TensorView(Tensor<T, OtherRank> &other, Dim<Rank> const &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> &other, Dim<Rank> const &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> const &other, Dim<Rank> const &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<TensorView<T, OtherRank> &>(other), args...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, typename... Args>
    explicit TensorView(std::string name, Tensor<T, OtherRank> &other, Dim<Rank> const &dim, Args &&...args)
        : _name{std::move(name)}, _dims{dim} {
        common_initialization(other, args...);
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims) : _dims(dims), _full_view_of_underlying{true}, _data{const_cast<T *>(data)} {
        dims_to_strides(dims, _strides);
        dims_to_strides(dims, _index_strides);
        _source_dims = dims;
        _offsets.fill(0);
        _offset_ordinal = 0;
    }

    /**
     * Wrap a pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims) : _dims(dims), _full_view_of_underlying{true}, _data{const_cast<T *>(data)} {
        dims_to_strides(dims, _strides);
        dims_to_strides(dims, _index_strides);
        _source_dims = dims;
        _offsets.fill(0);
        _offset_ordinal = 0;
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     * @param strides The strides for the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims, Stride<Rank> const &strides)
        : _dims(dims), _strides(strides), _data{const_cast<T *>(data)} {
        dims_to_strides(dims, _index_strides);
        _full_view_of_underlying = (strides == _index_strides);
        _source_dims             = dims; // Must have size Rank
        _offsets.fill(0);                // No offset given, taken to be zero
        _offset_ordinal = 0;             // Follows
        // Check access is in increasing stride
        for (size_t i = 0; i < Rank - 1; i++) {
            if (_strides[i] < _strides[i + 1]) {
                EINSUMS_THROW_EXCEPTION(dimension_error,
                                        "Strides must be in decreasing order - else unexpected permute behaviour will ensue");
            }
        }
        if (!_full_view_of_underlying) { // Fuses Indices if Rank < OtherRank (as OtherRank is not known)
            size_t current_stride = 1;
            for (int i = Rank - 1; i >= 1; i--) {
                _source_dims[i] = _strides[i - 1] / current_stride;
                current_stride *= _source_dims[i];
            }
            _source_dims[0] = (_strides[0] * _dims[0]) / current_stride; // stride[0] must be the largest
        }
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     * @param strides The strides for the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims, Stride<Rank> const &strides)
        : _dims(dims), _strides(strides), _data{const_cast<T *>(data)} {
        dims_to_strides(dims, _index_strides);
        _full_view_of_underlying = (strides == _index_strides);
        _source_dims             = dims;
        _offsets.fill(0);
        _offset_ordinal = 0;
        // Check access is in increasing stride
        for (size_t i = 0; i < Rank - 1; i++) {
            if (_strides[i] < _strides[i + 1]) {
                EINSUMS_THROW_EXCEPTION(dimension_error,
                                        "Strides must be in decreasing order - Else unexpected permute behaviour will pursue");
            }
        }
        if (!_full_view_of_underlying) { // Fuses Indices if Rank < OtherRank (as OtherRank is not known)
            size_t current_stride = 1;
            for (int i = dims.size() - 1; i >= 1; i--) {
                _source_dims[i] = strides[i - 1] / current_stride;
                current_stride *= _source_dims[i];
            }
            _source_dims[0] = (_strides[0] * _dims[0]) / current_stride;
        }
    }

    /**
     * Copy data from a pointer into this view.
     *
     * @attention This is an expert function only. If you are using it, you must know what you are doing!
     */
    auto operator=(T const *other) -> TensorView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        size_t elements = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < elements; item++) {
            size_t new_item;

            sentinel_to_sentinels(item, _index_strides, _strides, new_item);

            this->_data[new_item + _offset_ordinal] = other[item];
        }

        return *this;
    }

    /**
     * Copy assignment.
     */
    auto operator=(TensorView const &other) -> TensorView & {
        if (this == &other)
            return *this;

        if (this->size() != other.size() || this->dims() != other.dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform copy assignment. The tensor views have different dimensions.");
        }

        size_t elements = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < elements; item++) {
            size_t out_item, in_item;

            sentinel_to_sentinels(item, _index_strides, _strides, out_item, other._strides, in_item);

            this->_data[out_item + _offset_ordinal] = other.data()[in_item];
        }

        return *this;
    }

    /**
     * Copy the data from another tensor into this view.
     */
    template <template <typename, size_t> typename AType>
        requires CoreRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(AType<T, Rank> const &other) -> TensorView & {
        if constexpr (std::is_same_v<AType<T, Rank>, TensorView>) {
            if (this == &other)
                return *this;
        }

        if (this->size() != other.size() || this->dims() != other.dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform copy assignment. The tensor views have different dimensions.");
        }

        size_t elements = this->size();

        if (other.full_view_of_underlying()) {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t item = 0; item < elements; item++) {
                size_t out_item, in_item;

                sentinel_to_sentinels(item, _index_strides, _strides, out_item);

                this->_data[out_item + _offset_ordinal] = other.data()[item];
            }
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t item = 0; item < elements; item++) {
                size_t out_item, in_item;

                sentinel_to_sentinels(item, _index_strides, _strides, out_item, other._strides, in_item);

                this->_data[out_item + _offset_ordinal] = other.data()[in_item];
            }
        }

        return *this;
    }

    /**
     * Fill this view with a single value.
     */
    auto operator=(T const &fill_value) -> TensorView & {
        size_t elements = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < elements; item++) {
            size_t out_item;

            sentinel_to_sentinels(item, _index_strides, _strides, out_item);

            this->_data[out_item + _offset_ordinal] = fill_value;
        }
        return *this;
    }

    void zero() { *this = T{0.0}; }

#ifndef DOXYGEN
#    if defined(OPERATOR)
#        undef OPERATOR
#    endif
#    define OPERATOR(OP)                                                                                                                   \
        auto operator OP(const T &value)->TensorView & {                                                                                   \
            size_t elements = this->size();                                                                                                \
                                                                                                                                           \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t item = 0; item < elements; item++) {                                                                               \
                size_t out_item;                                                                                                           \
                                                                                                                                           \
                sentinel_to_sentinels(item, _index_strides, _strides, out_item);                                                           \
                                                                                                                                           \
                this->_data[out_item + _offset_ordinal] OP value;                                                                          \
            }                                                                                                                              \
                                                                                                                                           \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#    undef OPERATOR
#endif

    /**
     * Get a pointer to the data.
     */
    T *data() {
        auto offset_data = &_data[_offset_ordinal];
        return offset_data;
    }

    /**
     * @copydoc TensorView<T,Rank>::data()
     */
    T const *data() const {
        auto offset_data = &_data[_offset_ordinal];
        return static_cast<T const *>(offset_data);
    }

    /**
     * Get a pointer to the data without the offset included.
     */
    T *full_data() { return _data; }

    /**
     * @copydoc TensorView<T,Rank>::data()
     */
    T const *full_data() const { return static_cast<T const *>(_data); }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index The index for the offset.
     */
    template <typename... MultiIndex>
    auto data(MultiIndex... index) const -> T * {
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = std::array{static_cast<ptrdiff_t>(index)...};

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index_list);
        return &_data[ordinal + _offset_ordinal];
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    auto data_array(std::array<size_t, Rank> const &index_list) const -> T * {
        size_t ordinal = indices_to_sentinel(_strides, index_list);
        return &_data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex &&...index) const -> T const & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, std::forward<MultiIndex>(index)...);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex &&...index) -> T & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, std::forward<MultiIndex>(index)...);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) const -> T const & {
        static_assert(sizeof...(MultiIndex) == Rank);
        size_t ordinal = indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) -> T & {
        static_assert(sizeof...(MultiIndex) == Rank);

        size_t ordinal = indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto subscript(std::array<int_type, Rank> const &index) const -> T const & {
        size_t ordinal = indices_to_sentinel(_strides, index);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto subscript(std::array<int_type, Rank> const &index) -> T & {
        size_t ordinal = indices_to_sentinel(_strides, index);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    T &operator()(Container const &index) {
        if (index.size() < Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    const T &operator()(Container const &index) const {
        if (index.size() < Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        size_t ordinal = indices_to_sentinel_negative_check(_strides, _dims, index);
        return _data[ordinal + _offset_ordinal];
    }

    /**
     * Get the dimension of the view along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    /**
     * Get the dimension of the original tensor along a given axis.
     */
    size_t source_dim(int d) const {
        if (d < 0)
            d += Rank;
        return _source_dims[d];
    }

    /**
     * Get the dimensions of the view.
     */
    Dim<Rank> dims() const { return _dims; }

    /**
     * Get the dimensions of the original tensor.
     */
    Dim<Rank> source_dims() const { return _source_dims; }

    /**
     * Get the name of the view.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the view.
     *
     * @param name The new name for the view.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * Get the stride of the view along a given axis.
     *
     * @param d The axis to query.
     */
    size_t stride(int d) const noexcept {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    /**
     * Get the strides of the tensor.
     */
    Stride<Rank> strides() const noexcept { return _strides; }

    /**
     * Get the offset of the view along a given axis.
     */
    size_t offset(int d) const noexcept {
        if (d < 0)
            d += Rank;
        return _offsets[d];
    }

    /**
     * Get the offsets of the tensor.
     */
    Offset<Rank> offsets() const noexcept { return _offsets; }

    /**
     * Flatten the view.
     *
     * @warning Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!
     */
    auto to_rank_1_view() const -> TensorView<T, 1> {
        if constexpr (Rank == 1) {
            return *this;
        } else {
            if (_strides[Rank - 1] != 1) {
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
            }
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
            Dim<1> dim{size};

#if defined(EINSUMS_SHOW_WARNING)
            println("Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!");
#endif

            return TensorView<T, 1>{*this, dim, Stride{1}};
        }
    }

    /**
     * Check whether the view has all elements of the tensor it is viewing.
     */
    bool full_view_of_underlying() const noexcept { return _full_view_of_underlying; }

    /**
     * Get the number of elements in the view.
     */
    size_t size() const { return _dims[0] * _index_strides[0]; }

  private:
    /**
     * Initialize the view using a pointer.
     */
    auto common_initialization(T const *other) {
        _data = const_cast<T *>(other);

        dims_to_strides(_dims, _index_strides);
        dims_to_strides(_dims, _strides);
        _source_dims = _dims;
        _offsets.fill(0);
        _offset_ordinal = 0;

        // At this time we'll assume we have full view of the underlying tensor since we were only provided
        // pointer.
        _full_view_of_underlying = true;
    }

    /**
     * Initialize a view using a tensor.
     */
    template <TensorConcept TensorType, typename... Args>
        requires(std::is_same_v<T, typename TensorType::ValueType>)
    auto common_initialization(TensorType &other, Args &&...args) -> void {
        constexpr size_t OtherRank = TensorType::Rank;

        static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller that the Tensor being viewed.");

        // set_mutex(other.get_mutex());
        Stride<Rank>      default_strides{};
        Offset<OtherRank> default_offsets{};
        Offset<OtherRank> temp_offsets{};
        Stride<Rank>      error_strides{};
        error_strides[0] = -1;

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we compute this
        // dimensionality for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
        }

        if (nfound == 1 && Rank == 1) {
            default_offsets.fill(0);
            default_strides.fill(1);

            auto offsets = arguments::get(default_offsets, args...);
            auto strides = arguments::get(default_strides, args...);

            // Perform this with integer arithmetic. There is a chance that with
            // sizes greater than 2^52 that the division becomes inaccurate with
            // floating points. Integer divisions should never become inaccurate.
            // In floating point, it would be ceil((size - offset) / stride)
            ptrdiff_t numerator   = other.size() - offsets[0];
            size_t    denominator = strides[0];

            _dims[location] = numerator / denominator;

            if (numerator % denominator != 0) {
                _dims[location] += 1;
            }
        }

        if (nfound == 1 && Rank > 1) {
            EINSUMS_THROW_EXCEPTION(todo_error, "Haven't coded up this case yet.");
        }

        // If the Ranks are the same then use "other"s stride information
        if constexpr (Rank == OtherRank) {
            default_strides = other._strides;
            // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
            // provide the information
        } else {
            if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies()) ==
                std::accumulate(other._dims.begin(), other._dims.end(), 1.0, std::multiplies())) {
                dims_to_strides(_dims, default_strides);
            } else {
                // Stride information cannot be automatically deduced.  It must be provided.
                default_strides = arguments::get(error_strides, args...);
                if (default_strides[0] == static_cast<size_t>(-1)) {
                    EINSUMS_THROW_EXCEPTION(bad_logic, "Unable to automatically deduce stride information. Stride must be passed in.");
                }
            }
        }

        default_offsets.fill(0);

        // Use default_* unless the caller provides one to use.
        _strides       = arguments::get(default_strides, args...);
        temp_offsets   = arguments::get(default_offsets, args...);
        size_t ordinal = indices_to_sentinel(other._strides, temp_offsets);

        // Find source dimensions
        if constexpr (std::is_same_v<TensorType, TensorView<T, Rank>>) {
            _source_dims    = other._source_dims;
            _offsets        = temp_offsets;
            _offset_ordinal = ordinal;
            ordinal         = 0;
        } else if constexpr (std::is_same_v<TensorType, Tensor<T, Rank>>) {
            _source_dims    = other._dims;
            _offsets        = temp_offsets;
            _offset_ordinal = ordinal;
            ordinal         = 0;
        } else {
            // In different Ranks, source dimensions can be deduced from the strides.
            // In decreasing strides when matching the correct dimension is reached (dim[i] = 1 is neglected)
            // Where dimenions less than the outermost are skipped these are incorporated into the dimension inner to it
            // If there are no further inner strides, the inner stride is non-zero to account for missing dimensionality.
            size_t current_stride    = 1;
            size_t tensor_index      = 0;
            size_t cumulative_stride = 1;
            for (int i = 0; i < Rank; i++) {
                _source_dims[i]   = 0;
                cumulative_stride = 1;
                current_stride    = _strides[i];
                while (_source_dims[i] == 0) {
                    cumulative_stride *= other._dims[tensor_index];
                    if (other._strides[tensor_index] == current_stride) {
                        _source_dims[i] = cumulative_stride;
                        _offsets[i]     = temp_offsets[tensor_index];
                    }
                    tensor_index++;
                }
                if (_source_dims[i] == 0) {
                    EINSUMS_THROW_EXCEPTION(bad_logic,
                                            "Unable to deduce source dimensions. Stride does not follow source tensor dimensions.");
                }
            }
            _offset_ordinal = indices_to_sentinel(_strides, _offsets); // Only counts offsets that are in the view
            ordinal -= _offset_ordinal;
        }

        // Determine the ordinal using the offsets provided (if any) and the strides of the parent
        _data = &(other._data[ordinal]);

        // Calculate the index strides.
        dims_to_strides(_dims, _index_strides);
    }

    /**
     * @var _name
     *
     * The name of the view used for printing.
     */
    std::string _name{"(Unnamed View)"};

    /**
     * @var _dims
     *
     * The dimensions of the view.
     */
    /**
     * @var _source_dims
     *
     * The dimensions of the source tensor.
     */
    Dim<Rank> _dims, _source_dims;

    /**
     * @var _strides
     *
     * The strides of the tensor.
     */
    /**
     * @var _index_strides
     *
     * These are strides used for iterating over indices in the tensor.
     * These are based on the dimensions of the view, while _strides are based on the
     * dimensions of the tensor being viewed.
     */
    Stride<Rank> _strides, _index_strides;

    /**
     * @var _offsets
     *
     * These are offsets used to access data from a midpoint in the tensor.
     */
    /**
     * @var _offset_ordinal
     *
     * This is the value at which the offset is currently set. Found by multiplying the offsets by the strides.
     */
    Offset<Rank> _offsets;
    size_t       _offset_ordinal{0};

    /**
     * @var _full_view_of_underlying
     *
     * Whether the view is viewing the whole tensor.
     */
    bool _full_view_of_underlying{false};

    /**
     * @var _data
     *
     * Pointer to the data of the tensor.
     */
    T *_data;

    template <typename T_, size_t Rank_>
    friend struct Tensor;

    template <typename T_, size_t OtherRank_>
    friend struct TensorView;
};
} // namespace einsums

// Include HDF5 interface
#include "H5.hpp"

// Tensor IO interface
namespace einsums {

/**
 * Write a tensor to the disk.
 */
template <size_t Rank, typename T, class... Args>
void write(h5::fd_t const &fd, Tensor<T, Rank> const &ref, Args &&...args) {
    // Can these h5 parameters be moved into the Tensor class?
    h5::current_dims current_dims_default;
    h5::max_dims     max_dims_default;
    h5::count        count_default;
    h5::offset       offset_default{0, 0, 0, 0, 0, 0, 0};
    h5::stride       stride_default{1, 1, 1, 1, 1, 1, 1};
    h5::block        block_default{1, 1, 1, 1, 1, 1, 1};

    current_dims_default.rank = Rank;
    max_dims_default.rank     = Rank;
    count_default.rank        = Rank;
    offset_default.rank       = Rank;
    stride_default.rank       = Rank;
    block_default.rank        = Rank;

    for (int i = 0; i < Rank; i++) {
        current_dims_default[i] = ref.dim(i);
        max_dims_default[i]     = ref.dim(i);
        count_default[i]        = ref.dim(i);
    }

    h5::current_dims const &current_dims = h5::arg::get(current_dims_default, args...);
    h5::max_dims const     &max_dims     = h5::arg::get(max_dims_default, args...);
    h5::count const        &count        = h5::arg::get(count_default, args...);
    h5::offset const       &offset       = h5::arg::get(offset_default, args...);
    h5::stride const       &stride       = h5::arg::get(stride_default, args...);
    // h5::block const        &block        = h5::arg::get(block_default, args...);

    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), current_dims, max_dims);
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write(ds, ref, count, offset, stride);
}

/**
 * Write a zero-rank tensor to the disk.
 */
template <typename T>
void write(h5::fd_t const &fd, Tensor<T, 0> const &ref) {
    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), h5::current_dims{1});
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write<T>(ds, ref.data(), h5::count{1});
}

/**
 * Write a tensor view to the disk.
 */
template <size_t Rank, typename T, class... Args>
void write(h5::fd_t const &fd, TensorView<T, Rank> const &ref, Args &&...args) {
    h5::count  count_default{1, 1, 1, 1, 1, 1, 1};
    h5::offset offset_default{0, 0, 0, 0, 0, 0, 0};
    h5::stride stride_default{1, 1, 1, 1, 1, 1, 1};
    h5::offset view_offset{0, 0, 0, 0, 0, 0, 0};
    h5::offset disk_offset{0, 0, 0, 0, 0, 0, 0};

    count_default.rank  = Rank;
    offset_default.rank = Rank;
    stride_default.rank = Rank;
    view_offset.rank    = Rank;
    disk_offset.rank    = Rank;

    if (ref.stride(Rank - 1) != 1) {
        EINSUMS_THROW_EXCEPTION(bad_logic, "Final dimension of TensorView must be contiguous to write.");
    }

    h5::offset const &offset = h5::arg::get(offset_default, args...);
    // h5::stride const &stride = h5::arg::get(stride_default, args...);

    count_default[Rank - 1] = ref.dim(Rank - 1);

    // Does the entry exist on disk?
    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) > 0) {
        ds = h5::open(fd, ref.name().c_str());
    } else {
        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            chunk_temp[i] = ref.dim(i);
        }

        ds = h5::create<T>(fd, ref.name().c_str(), h5::current_dims{ref.dims()},
                           h5::chunk{chunk_temp} /*| h5::gzip{9} | h5::fill_value<T>(0.0)*/);
    }

    Stride<Rank - 1> index_strides;
    size_t elements = dims_to_strides(Dim<Rank - 1>(ref.dims().cbegin(), std::next(ref.dims().cbegin(), Rank - 1)), index_strides);

    for (size_t item = 0; item < elements; item++) {
        std::array<int64_t, Rank - 1> combination;
        sentinel_to_indices(item, index_strides, combination);
        // We generate all the cartesian products for all the dimensions except the final dimension
        // We call write on that final dimension.
        detail::add_elements<Rank - 1>(view_offset, offset_default, combination);
        detail::add_elements<Rank - 1>(disk_offset, offset, combination);

        // Get the data pointer from the view
        T *data = ref.data_array(view_offset);
        h5::write<T>(ds, data, count_default, disk_offset);
    }
}

/**
 * Read data from a disk and put it into a tensor.
 *
 * @todo This needs to be expanded to handle the various h5 parameters like above.
 */
template <typename T, size_t Rank>
auto read(h5::fd_t const &fd, std::string const &name) -> Tensor<T, Rank> {
    try {
        auto temp = h5::read<einsums::Tensor<T, Rank>>(fd, name);
        temp.set_name(name);
        return temp;
    } catch (std::exception &e) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", e.what());
    }
}

/**
 * Read data from a disk and put it into a zero-rank tensor.
 */
template <typename T>
auto read(h5::fd_t const &fd, std::string const &name) -> Tensor<T, 0> {
    try {
        T            temp{0};
        Tensor<T, 0> tensor{name};
        h5::read<T>(fd, name, &temp, h5::count{1});
        tensor = temp;
        return tensor;
    } catch (std::exception &e) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", e.what());
    }
}

/**
 * Function that zeros a tensor.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
void zero(TensorType<T, Rank> &A) {
    A.zero();
}

#ifndef DOXYGEN
#    ifdef __cpp_deduction_guides
template <typename... Args>
Tensor(std::string const &, Args...) -> Tensor<double, sizeof...(Args)>;

template <typename T, size_t OtherRank, typename... Dims>
explicit Tensor(Tensor<T, OtherRank> &&otherTensor, std::string name, Dims... dims) -> Tensor<T, sizeof...(dims)>;

template <size_t Rank, typename... Args>
explicit Tensor(Dim<Rank> const &, Args...) -> Tensor<double, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(Tensor<T, OtherRank> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(Tensor<T, OtherRank> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(std::string, Tensor<T, OtherRank> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

// Supposedly C++20 will allow template deduction guides for template aliases. i.e. Dim, Stride, Offset, Count, Range.
// Clang has no support for class template argument deduction for alias templates. P1814R0
#    endif
#endif

// Useful factories

/**
 * @brief Create a new tensor with \p name and \p args .
 *
 * Just a simple factory function for creating new tensors. Defaults to using double for the
 * underlying data and automatically determines rank of the tensor from args.
 *
 * A \p name for the tensor is required. \p name is used when printing and performing disk I/O.
 *
 * By default, the allocated tensor data is not initialized to zero. This was a performance
 * decision. In many cases the next step after creating a tensor is to load or store data into
 * it...why waste the CPU cycles zeroing something that will immediately get set to something
 * else.  If you wish to explicitly zero the contents of your tensor use the zero function.
 *
 * @code
 * auto a = create_tensor("a", 3, 3);           // auto -> Tensor<double, 2>
 * auto b = create_tensor<float>("b", 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying tensor. Defaults to double.
 * @tparam Args The datatype of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param name The name of the new tensor.
 * @param args The arguments needed to construct the tensor.
 * @return A new tensor. By default, memory is not initialized to anything. It may be filled with garbage.
 */
template <typename Type = double, typename... Args>
auto create_tensor(std::string const &name, Args... args) {
    EINSUMS_LOG_TRACE("creating tensor {}, {}", name, std::forward_as_tuple(args...));
    return Tensor<Type, sizeof...(Args)>{name, args...};
}

/**
 * @brief Create a new tensor with \p name and \p args .
 *
 * Just a simple factory function for creating new tensors. Defaults to using double for the
 * underlying data and automatically determines rank of the tensor from args.
 *
 * A \p name for the tensor is required. \p name is used when printing and performing disk I/O.
 *
 * By default, the allocated tensor data is not initialized to zero. This was a performance
 * decision. In many cases the next step after creating a tensor is to load or store data into
 * it...why waste the CPU cycles zeroing something that will immediately get set to something
 * else.  If you wish to explicitly zero the contents of your tensor use the zero function.
 *
 * @code
 * auto a = create_tensor(3, 3);           // auto -> Tensor<double, 2>
 * auto b = create_tensor<float>(4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying tensor. Defaults to double.
 * @tparam Args The datatype of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param args The arguments needed to construct the tensor.
 * @return A new tensor. By default, memory is not initialized to anything. It may be filled with garbage.
 */
template <typename Type = double, std::integral... Args>
auto create_tensor(Args... args) {
    return Tensor<Type, sizeof...(Args)>{"Temporary", args...};
}

namespace detail {

/**
 * Count the number of digits in a number.
 */
template <std::integral T>
auto ndigits(T number) -> int {
    int digits{0};
    if (number < 0)
        digits = 1; // Remove this line if '-' counts as a digit
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}
} // namespace detail

#ifndef DOXYGEN
template <FileOrOStream Output, RankTensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options) {
    using T                    = typename AType::ValueType;
    constexpr std::size_t Rank = AType::Rank;

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

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(fp, "Dims{{{}}}", oss.str().c_str());
        }

        if constexpr (Rank > 0 && einsums::BasicTensorConcept<AType>) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.stride(i) << " ";
            }
            fprintln(fp, "Strides{{{}}}", oss.str());
        }

        if (options.full_output) {
            fprintln(fp);

            if constexpr (Rank == 0) {
                T value = A;

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
            } else if constexpr (Rank > 1 && einsums::CoreTensorConcept<AType>) {
#    else
            } else if constexpr (Rank > 1 && (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>)) {
#    endif

                Stride<Rank - 1> index_strides;
                size_t           elements = dims_to_strides(einsums::slice_array<Rank - 1>(A.dims()), index_strides);

                auto final_dim = A.dim(Rank - 1);
                auto ndigits   = detail::ndigits(final_dim);

                for (size_t item = 0; item < elements; item++) {
                    std::array<int64_t, Rank - 1> target_combination;

                    sentinel_to_indices(item, index_strides, target_combination);

                    std::ostringstream oss;
                    for (int j = 0; j < final_dim; j++) {
                        if (j % options.width == 0) {
                            std::ostringstream tmp;
                            tmp << fmt::format("{}", fmt::join(target_combination, ", "));
                            if (final_dim >= j + options.width)
                                oss << fmt::format(
                                    "{:<14}", fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, j + options.width - 1, ndigits));
                            else
                                oss << fmt::format("{:<14}",
                                                   fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, final_dim - 1, ndigits));
                        }
                        auto new_tuple = std::tuple_cat(target_combination, std::tuple(j));
                        T    value     = std::apply(A, new_tuple);
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
#    if !defined(EINSUMS_COMPUTE_CODE)
            } else if constexpr (Rank == 1 && einsums::CoreTensorConcept<AType>) {
#    else
            } else if constexpr (Rank == 1 && (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>)) {
#    endif
                size_t elements = A.size();

                for (size_t item = 0; item < elements; item++) {
                    std::ostringstream oss;
                    oss << "(";
                    oss << fmt::format("{}, ", item);
                    oss << "): ";

                    T value = A(item);
                    if (std::abs(value) > 1.0E+5) {
                        if constexpr (std::is_floating_point_v<T>)
                            oss << fmt::format(fg(fmt::color::white) | bg(fmt::color::red), "{:14.8f} ", value);
                        else if constexpr (IsComplexV<T>) {
                            oss << fmt::format(fg(color::white) | bg(color::red), "({:14.8f} + {:14.8f})", value.real(), value.imag());
                        } else
                            oss << fmt::format(fg(color::white) | bg(color::red), "{:14} ", value);
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
    fprintln(fp);
}

template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options) {
    fprintln(std::cout, A, options);
}

EINSUMS_TENSOR_EXPORT_RANK(Tensor, 0)
EINSUMS_TENSOR_EXPORT(Tensor)
EINSUMS_TENSOR_EXPORT(TensorView)
#endif

} // namespace einsums
