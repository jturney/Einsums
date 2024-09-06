//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/concepts/complex.hpp>
#include <einsums/concepts/file.hpp>
#include <einsums/concepts/tensor.hpp>
#include <einsums/errors/exception.hpp>
#include <einsums/errors/throw_exception.hpp>
#include <einsums/iterator/enumerate.hpp>
#include <einsums/print.hpp>
#include <einsums/profile/section.hpp>
#include <einsums/tensor/tensor_fwd.hpp>
#include <einsums/util/are_all_convertible.hpp>
#include <einsums/util/arguments.hpp>
#include <einsums/util/type_name.hpp>

#include <range/v3/range_fwd.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/iota.hpp>

// #include "einsums/ArithmeticTensor.hpp"
// #include "einsums/Exception.hpp"
// #include "einsums/OpenMP.h"
// #include "einsums/STL.hpp"
// #include "einsums/State.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
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

#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums {

// Forward declarations
template <typename T, std::size_t Rank>
struct tensor_view;

namespace disk {

template <typename T, std::size_t ViewRank, std::size_t Rank>
struct view;

template <typename T, std::size_t Rank>
struct tensor;

} // namespace disk
} // namespace einsums

namespace einsums {

// Forward declaration of the Tensor printing function.
template <einsums::TensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void println(const AType &A, tensor_print_options options = {});

template <einsums::TensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(std::FILE *fp, const AType &A, tensor_print_options options = {});

template <einsums::TensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(std::ostream &os, const AType &A, tensor_print_options options = {});

namespace detail {

/**
 * @brief Get the dim ranges object
 *
 * @tparam TensorType
 * @tparam I
 * @param tensor The tensor object to query
 * @return A tuple containing the dimension ranges compatible with range-v3 cartesian_product function.
 */
template <typename TensorType, std::size_t... I>
auto get_dim_ranges(const TensorType &tensor, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(I))...};
}

/**
 * @brief Adds elements from two sources into the target.
 *
 * Useful in adding offsets to a set of indices.
 *
 * @tparam N The rank of the data.
 * @tparam Target
 * @tparam Source1
 * @tparam Source2
 * @param target The output
 * @param source1 The first source
 * @param source2 The second source
 */
template <size_t N, typename Target, typename Source1, typename Source2>
void add_elements(Target &target, const Source1 &source1, const Source2 &source2) {
    if constexpr (N > 1) {
        add_elements<N - 1>(target, source1, source2);
    }
    target[N - 1] = source1[N - 1] + std::get<N - 1>(source2);
}

} // namespace detail

/**
 * @brief Find the ranges for each dimension of a tensor.
 *
 * The returned tuple is compatible with ranges-v3 cartesian_product function.
 *
 * @tparam N
 * @tparam TensorType
 * @param tensor Tensor to query
 * @return Tuple containing the range for each dimension of the tensor.
 */
template <int N, typename TensorType>
auto get_dim_ranges(const TensorType &tensor) {
    return detail::get_dim_ranges(tensor, std::make_index_sequence<N>{});
}

// template <typename T>
// using VectorData = std::vector<T, AlignedAllocator<T, 64>>;

// TODO: Need to provide the aligned_allocator variant.
template <typename T>
using vector_type = std::vector<T>;

/**
 * @brief Represents a general tensor
 *
 * @tparam T data type of the underlying tensor data
 * @tparam Rank the rank of the tensor
 */
template <typename T, std::size_t Rank>
struct tensor : virtual tensor_base::core_tensor,
                virtual tensor_base::basic_tensor<T, Rank>,
                virtual tensor_base::lockable_tensor,
                virtual tensor_base::algebra_optimized_tensor {

    using value_type = T;
    using vector     = vector_type<T>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    tensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    tensor(const tensor &) = default;

    /**
     * @brief Destroy the Tensor object.
     */
    ~tensor() override = default;

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
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to std::size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <typename... Dims>
    explicit tensor(std::string name, Dims... dims) : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        std::size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

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
     * Supports using -1 for one of the ranks to automatically compute the dimensional of it.
     *
     * @code
     * auto A = Tensor("A", 27);
     * auto B = Tensor(std::move(A), "B", 3, -1, 3); // Automatically determines that -1 should be 3.
     * @endcode
     *
     * @tparam OtherRank The rank of \p existingTensor can be different than the rank of the new tensor
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to std::size_t.
     * @param existingTensor The existing tensor that holds the tensor data.
     * @param name The name of the new tensor
     * @param dims The dimensionality of each rank of the new tensor.
     */
    template <size_t OtherRank, typename... Dims>
    explicit tensor(tensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
        : _data(std::move(existingTensor._data)), _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

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
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "More than one -1 was provided.");
        }

        if (nfound == 1) {
            std::size_t size{1};
            for (auto [i, dim] : enumerate(_dims)) {
                if (i != location) {
                    size *= dim;
                }
            }
            if (size > existingTensor.size()) {
                EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Size of new tensor is larger than the parent tensor.");
            }
            _dims[location] = existingTensor.size() / size;
        }

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

        // Check size
        if (_data.size() != size) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Provided dims to not match size of parent tensor");
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit tensor(einsums::dim<Rank> dims) : _dims{std::move(dims)} {
        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        _data.resize(size);
    }

    /**
     * @brief Construct a new Tensor object from a tensor_view.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    tensor(const tensor_view<T, Rank> &other) : _name{other._name}, _dims{other._dims} {
        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        _data.resize(size);

        // TODO: Attempt to thread this.
        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T  value        = std::apply(other, target_combination);
            target_value    = value;
        }
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    void resize(dim<Rank> dims) {
        if (_dims == dims) {
            return;
        }

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        _dims = dims;

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        _data.resize(size);
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    template <typename... Dims>
    auto resize(Dims... dims) -> void
        requires((std::is_arithmetic_v<Dims> && ... && (sizeof...(Dims) == Rank)))
    {
        resize(einsums::dim<Rank>{static_cast<size_t>(dims)...});
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        // #pragma omp parallel
        //         {
        //             auto tid       = omp_get_thread_num();
        //             auto chunksize = _data.size() / omp_get_num_threads();
        //             auto begin     = _data.begin() + chunksize * tid;
        //             auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
        //             memset(&(*begin), 0, end - begin);
        //         }
        memset(_data.data(), 0, sizeof(T) * _data.size());
    }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        // #pragma omp parallel
        //         {
        //             auto tid       = omp_get_thread_num();
        //             auto chunksize = _data.size() / omp_get_num_threads();
        //             auto begin     = _data.begin() + chunksize * tid;
        //             auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
        //             std::fill(begin, end, value);
        //         }
        std::fill(_data.begin(), _data.end(), value);
    }

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    auto data() -> T * override { return _data.data(); }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    auto data() const -> const T * override { return _data.data(); }

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
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
            requires NoneOfType<dim<Rank>, MultiIndex...>;
            requires NoneOfType<offset<Rank>, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return &_data[ordinal];
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
            requires NoneOfType<dim<Rank>, MultiIndex...>;
            requires NoneOfType<offset<Rank>, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> const T & {

        assert(sizeof...(MultiIndex) == _dims.size());

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
            requires NoneOfType<dim<Rank>, MultiIndex...>;
            requires NoneOfType<offset<Rank>, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T & {

        assert(sizeof...(MultiIndex) == _dims.size());

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return _data[ordinal];
    }

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<all_t, MultiIndex...>; }
    auto operator()(MultiIndex... index)
        -> tensor_view<T, util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>()> {
        using namespace util::arguments;

        // Construct a tensor_view using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === tensor_view{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");
        const auto &indices = std::forward_as_tuple(index...);

        einsums::offset<Rank>                                                                          offsets;
        einsums::stride<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()> strides{};
        einsums::dim<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                offsets[i] = tmp;
            } else if constexpr (std::is_same_v<all_t, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                strides[counter] = _strides[i];
                dims[counter]    = _dims[i];
                counter++;

            } else if constexpr (std::is_same_v<range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
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

        return tensor_view<T, count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()>{*this, std::move(dims),
                                                                                                             offsets, strides};
    }

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<all_t, MultiIndex...>; }
    auto operator()(MultiIndex... index) const -> const
        tensor_view<T, util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>()> {
        using namespace util::arguments;
        // Construct a tensor_view using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === tensor_view{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");
        const auto &indices = std::forward_as_tuple(index...);

        einsums::offset<Rank>                                                                          offsets;
        einsums::stride<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()> strides{};
        einsums::dim<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                offsets[i] = tmp;
            } else if constexpr (std::is_same_v<all_t, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                strides[counter] = _strides[i];
                dims[counter]    = _dims[i];
                counter++;

            } else if constexpr (std::is_same_v<range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
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

        return tensor_view<T, count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()>{*this, std::move(dims),
                                                                                                             offsets, strides};
    }

    template <typename... MultiIndex>
        requires NumOfType<range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> tensor_view<T, Rank> {
        einsums::dim<Rank>    dims{};
        einsums::offset<Rank> offset{};
        einsums::stride<Rank> stride = _strides;

        auto ranges = util::arguments::get_array_from_tuple<std::array<range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
            auto range = ranges[r];
            offset[r]  = range[0];
            if (range[1] < 0) {
                auto temp = _dims[r] + range[1];
                range[1]  = temp;
            }
            dims[r] = range[1] - range[0];
        }

        return tensor_view<T, Rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
    }

    auto operator=(const tensor<T, Rank> &other) -> tensor<T, Rank> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            struct Stride {
                std::size_t value{1};
                Stride() = default;
                auto operator()(size_t dim) -> std::size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other._dims;

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
            std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        std::copy(other._data.begin(), other._data.end(), _data.begin());

        return *this;
    }

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const tensor<TOther, Rank> &other) -> tensor<T, Rank> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0) {
                realloc = true;
            } else if (dim(i) != other.dim(i)) {
                const std::string str = fmt::format("dimensions do not match (this){} (other){}", dim(i), other.dim(i));
                if constexpr (Rank != 1) {
                    EINSUMS_THROW_EXCEPTION(error::bad_parameter, fmt::runtime(str));
                } else {
                    realloc = true;
                }
            }
        }

        if (realloc) {
            struct Stride {
                std::size_t value{1};
                Stride() = default;
                auto operator()(size_t dim) -> std::size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other._dims;

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
            std::size_t size = _strides.empty() ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T  value        = std::apply(other, target_combination);
            target_value    = value;
        }

        return *this;
    }

    template <TensorConcept OtherTensor>
        requires requires {
            requires !BasicTensorConcept<OtherTensor>;
            requires SameRank<tensor<T, Rank>, OtherTensor>;
            requires CoreTensorConcept<OtherTensor>;
        }
    auto operator=(const OtherTensor &other) -> tensor<T, Rank> & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T  value        = std::apply(other, target_combination);
            target_value    = value;
        }

        return *this;
    }

    template <typename TOther>
    auto operator=(const tensor_view<TOther, Rank> &other) -> tensor<T, Rank> & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T  value        = std::apply(other, target_combination);
            target_value    = value;
        }

        return *this;
    }

#if defined(EINSUMS_COMPUTE_CODE)
    auto operator=(const device_tensor<T, Rank> &other) -> tensor<T, Rank> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            struct Stride {
                std::size_t value{1};
                Stride() = default;
                auto operator()(size_t dim) -> std::size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other.dims();

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
            std::size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        gpu::hip_catch(hipMemcpy(_data.data(), other.gpu_data(), _strides[0] * _dims[0] * sizeof(T), hipMemcpyDeviceToHost));

        return *this;
    }
#endif

    auto operator=(const T &fill_value) -> tensor & {
        set_all(fill_value);
        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b)->tensor<T, Rank> & {                                                                                      \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto begin     = _data.begin() + chunksize * tid;                                                                              \
            auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;                                         \
            EINSUMS_OMP_SIMD for (auto i = begin; i < end; i++) {                                                                          \
                (*i) OP b;                                                                                                                 \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const tensor<T, Rank> &b)->tensor<T, Rank> & {                                                                        \
        if (size() != b.size()) {                                                                                                          \
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "tensors differ in size : {} {}", size(), b.size());                             \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto abegin    = _data.begin() + chunksize * tid;                                                                              \
            auto bbegin    = b._data.begin() + chunksize * tid;                                                                            \
            auto aend      = (tid == omp_get_num_threads() - 1) ? _data.end() : abegin + chunksize;                                        \
            auto j         = bbegin;                                                                                                       \
            EINSUMS_OMP_SIMD for (auto i = abegin; i < aend; i++) {                                                                        \
                (*i) OP(*j++);                                                                                                             \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    auto dim(int d) const -> std::size_t override {
        // Add support for negative indices.
        if (d < 0) {
            d += Rank;
        }
        return _dims[d];
    }
    auto dims() const -> einsums::dim<Rank> override { return _dims; }

    auto vector_data() const -> const vector & { return _data; }
    auto vector_data() -> vector & { return _data; }

    [[nodiscard]] auto stride(int d) const noexcept -> std::size_t override {
        if (d < 0) {
            d += Rank;
        }
        return _strides[d];
    }

    auto strides() const noexcept -> einsums::stride<Rank> override { return _strides; }

    auto to_rank_1_view() const -> tensor_view<T, 1> {
        std::size_t     size = _strides.empty() ? 0 : _strides[0] * _dims[0];
        einsums::dim<1> dim{size};

        return tensor_view<T, 1>{*this, dim};
    }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    auto full_view_of_underlying() const noexcept -> bool override { return true; }

    const std::string &name() const override { return _name; };

    void set_name(const std::string &new_name) override { _name = new_name; };

  private:
    std::string           _name{"(unnamed)"};
    einsums::dim<Rank>    _dims;
    einsums::stride<Rank> _strides;
    vector                _data;

    template <typename T_, std::size_t Rank_>
    friend struct tensor_view;

    template <typename T_, std::size_t OtherRank>
    friend struct Tensor;
}; // namespace einsums

template <typename T>
struct tensor<T, 0> : public virtual tensor_base::core_tensor,
                      virtual tensor_base::basic_tensor<T, 0>,
                      virtual tensor_base::lockable_tensor,
                      virtual tensor_base::algebra_optimized_tensor {

    tensor()                   = default;
    tensor(const tensor &)     = default;
    tensor(tensor &&) noexcept = default;
    ~tensor() override         = default;

    explicit tensor(std::string name) : _name{std::move(name)} {};

    explicit tensor(einsums::dim<0> _ignore) {}

    auto               data() -> T               *override { return &_data; }
    [[nodiscard]] auto data() const -> const T * override { return &_data; }

    auto operator=(const tensor<T, 0> &other) -> tensor<T, 0> & {
        _data = other._data;
        return *this;
    }

    auto operator=(const T &other) -> tensor<T, 0> & {
        _data = other;
        return *this;
    }

#if defined(OPERATOR)
#    undef OPERATOR
#endif
#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &other)->tensor<T, 0> & {                                                                                     \
        _data OP other;                                                                                                                    \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    operator T() const { return _data; } // NOLINT
    operator T &() { return _data; }     // NOLINT

    [[nodiscard]] auto name() const -> const std::string & override { return _name; }
    void               set_name(const std::string &name) override { _name = name; }

    [[nodiscard]] auto dim(int) const -> std::size_t override { return 1; }

    [[nodiscard]] auto dims() const -> einsums::dim<0> override { return einsums::dim<0>{}; }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool override { return true; }

    std::size_t stride(int d) const override { return 0; }

    einsums::stride<0> strides() const override { return einsums::stride<0>(); }

  private:
    std::string _name{"(Unnamed)"};
    T           _data{};
};

template <typename T, std::size_t Rank>
struct tensor_view final : public virtual tensor_base::core_tensor,
                           virtual tensor_base::basic_tensor<T, Rank>,
                           virtual tensor_base::tensor_view<T, Rank, tensor<T, Rank>>,
                           virtual tensor_base::lockable_tensor,
                           virtual tensor_base::algebra_optimized_tensor {

    tensor_view()                    = delete;
    tensor_view(const tensor_view &) = default;
    ~tensor_view() override          = default;

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and tensor_view).  The
    // call to common_initialization is able to perform an enable_if check.
    template <size_t OtherRank, typename... Args>
    explicit tensor_view(const tensor<T, OtherRank> &other, const dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 1");
        common_initialization(const_cast<tensor<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit tensor_view(tensor<T, OtherRank> &other, const dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 2");
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit tensor_view(tensor_view<T, OtherRank> &other, const dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 3");
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit tensor_view(const tensor_view<T, OtherRank> &other, const dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 4");
        common_initialization(const_cast<tensor_view<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit tensor_view(std::string name, tensor<T, OtherRank> &other, const dim<Rank> &dim, Args &&...args)
        : _name{std::move(name)}, _dims{dim} {
        // println(" here 5");
        common_initialization(other, args...);
    }

    // template <typename... Args>
    // explicit tensor_view(const double *other, const Dim<Rank> &dim) : _name{"Raw Array"}, _dims{dim} {
    //     common_initialization(other);
    // }

    auto operator=(const T *other) -> tensor_view & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        auto        target_dims = get_dim_ranges<Rank>(*this);
        std::size_t item{0};

        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target = std::apply(*this, target_combination);
            target    = other[item];
            item++;
        }

        return *this;
    }

    auto operator=(const tensor_view<T, Rank> &other) -> tensor_view & {
        if (this == &other)
            return *this;

        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for default(none) shared(view, other)
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target    = std::apply(other, *target_combination);
        }

        return *this;
    }

    template <template <typename, std::size_t> typename AType>
        requires CoreRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &other) -> tensor_view & {
        if constexpr (std::is_same_v<AType<T, Rank>, tensor_view<T, Rank>>) {
            if (this == &other)
                return *this;
        }

        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);

        // #pragma omp parallel for default(none) shared(view, other)
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target    = std::apply(other, *target_combination);
        }

        return *this;
    }

    auto operator=(const T &fill_value) -> tensor_view & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for default(none) shared(view, fill_value)
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target    = fill_value;
        }

        return *this;
    }

#if defined(OPERATOR)
#    undef OPERATOR)
#endif
#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &value)->tensor_view & {                                                                                      \
        auto target_dims = get_dim_ranges<Rank>(*this);                                                                                    \
        auto view        = std::apply(ranges::views::cartesian_product, target_dims);                                                      \
        EINSUMS_OMP_PARALLEL_FOR for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {    \
            T        &target = std::apply(*this, *target_combination);                                                                     \
            target OP value;                                                                                                               \
        }                                                                                                                                  \
                                                                                                                                           \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    auto data() -> T * override { return &_data[0]; }
    auto data() const -> const T * override { return static_cast<const T *>(&_data[0]); }
    template <typename... MultiIndex>
    auto data(MultiIndex... index) const -> T * {
        assert(sizeof...(MultiIndex) <= _dims.size());
        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return &_data[ordinal];
    }

    auto data_array(const std::array<size_t, Rank> &index_list) const -> T * {
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return &_data[ordinal];
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> const T & {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return _data[ordinal];
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        std::size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), std::size_t{0});
        return _data[ordinal];
    }

    [[nodiscard]] auto dim(int d) const -> std::size_t override {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }
    auto dims() const -> einsums::dim<Rank> override { return _dims; }

    [[nodiscard]] auto name() const -> const std::string & override { return _name; }
    void               set_name(const std::string &name) override { _name = name; }

    [[nodiscard]] auto stride(int d) const noexcept -> std::size_t override {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> einsums::stride<Rank> override { return _strides; }

    auto to_rank_1_view() const -> tensor_view<T, 1> {
        if constexpr (Rank == 1) {
            return *this;
        } else {
            if (_strides[Rank - 1] != 1) {
                EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Creating a Rank-1 tensor_view for this Tensor(View) is not supported.");
            }
            std::size_t     size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
            einsums::dim<1> dim{size};

#if defined(EINSUMS_SHOW_WARNING)
            println("Creating a Rank-1 tensor_view of an existing tensor_view may not work. Be careful!");
#endif

            return tensor_view<T, 1>{*this, dim, einsums::stride<1>{1}};
        }
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool override { return _full_view_of_underlying; }

    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

  private:
    auto common_initialization(const T *other) {
        _data = const_cast<T *>(other);

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        // At this time we'll assume we have full view of the underlying tensor since we were only provided
        // pointer.
        _full_view_of_underlying = true;
    }

    template <TensorConcept TensorType, typename... Args>
        requires(std::is_same_v<T, typename TensorType::value_type>)
    auto common_initialization(TensorType &other, Args &&...args) -> void {
        constexpr std::size_t OtherRank = TensorType::rank;

        static_assert(Rank <= OtherRank, "A tensor_view must be the same Rank or smaller that the Tensor being viewed.");

        set_mutex(other.get_mutex());

        einsums::stride<Rank>      default_strides{};
        einsums::offset<OtherRank> default_offsets{};
        einsums::stride<Rank>      error_strides{};
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
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "More than one -1 was provided.");
        }

        if (nfound == 1 && Rank == 1) {
            default_offsets.fill(0);
            default_strides.fill(1);

            auto offsets = util::arguments::get(default_offsets, args...);
            auto strides = util::arguments::get(default_strides, args...);

            _dims[location] = static_cast<std::int64_t>(std::ceil((other.size() - offsets[0]) / static_cast<float>(strides[0])));
        }

        if (nfound == 1 && Rank > 1) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Haven't coded up this case yet.");
        }

        // If the Ranks are the same then use "other"s stride information
        if constexpr (Rank == OtherRank) {
            default_strides = other._strides;
            // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
            // provide the information
        } else {
            if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies<>()) ==
                std::accumulate(other._dims.begin(), other._dims.end(), 1.0, std::multiplies<>())) {
                struct Stride {
                    std::size_t value{1};
                    Stride() = default;
                    auto operator()(size_t dim) -> std::size_t {
                        auto old_value = value;
                        value *= dim;
                        return old_value;
                    }
                };

                // Row-major order of dimensions
                std::transform(_dims.rbegin(), _dims.rend(), default_strides.rbegin(), Stride());
                std::size_t size = default_strides.empty() ? 0 : default_strides[0] * _dims[0];
            } else {
                // Stride information cannot be automatically deduced.  It must be provided.
                default_strides = util::arguments::get(error_strides, args...);
                if (default_strides[0] == static_cast<size_t>(-1)) {
                    EINSUMS_THROW_EXCEPTION(error::bad_parameter,
                                            "Unable to automatically deduce stride information. Stride must be passed in.");
                }
            }
        }

        default_offsets.fill(0);

        // Use default_* unless the caller provides one to use.
        _strides                                  = util::arguments::get(default_strides, args...);
        const einsums::offset<OtherRank> &offsets = util::arguments::get(default_offsets, args...);

        // Determine the ordinal using the offsets provided (if any) and the strides of the parent
        std::size_t ordinal = std::inner_product(offsets.begin(), offsets.end(), other._strides.begin(), std::size_t{0});
        _data               = &(other._data[ordinal]);
    }

    std::string           _name{"(Unnamed View)"};
    einsums::dim<Rank>    _dims;
    einsums::stride<Rank> _strides;

    bool _full_view_of_underlying{false};

    T *_data;

    template <typename T_, std::size_t Rank_>
    friend struct tensor;

    template <typename T_, std::size_t OtherRank_>
    friend struct tensor_view;
};

} // namespace einsums

// Include HDF5 interface
#include "H5.hpp"

// Tensor IO interface
namespace einsums {

template <size_t Rank, typename T, class... Args>
void write(const h5::fd_t &fd, const tensor<T, Rank> &ref, Args &&...args) {
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

    const h5::current_dims &current_dims = h5::arg::get(current_dims_default, args...);
    const h5::max_dims     &max_dims     = h5::arg::get(max_dims_default, args...);
    const h5::count        &count        = h5::arg::get(count_default, args...);
    const h5::offset       &offset       = h5::arg::get(offset_default, args...);
    const h5::stride       &stride       = h5::arg::get(stride_default, args...);
    const h5::block        &block        = h5::arg::get(block_default, args...);

    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), current_dims, max_dims);
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write(ds, ref, count, offset, stride);
}

template <typename T>
void write(const h5::fd_t &fd, const tensor<T, 0> &ref) {
    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), h5::current_dims{1});
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write<T>(ds, ref.data(), h5::count{1});
}

template <size_t Rank, typename T, class... Args>
void write(const h5::fd_t &fd, const tensor_view<T, Rank> &ref, Args &&...args) {
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
        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Final dimension of tensor_view must be contiguous to write.");
    }

    const h5::offset &offset = h5::arg::get(offset_default, args...);
    const h5::stride &stride = h5::arg::get(stride_default, args...);

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

    auto dims = get_dim_ranges<Rank - 1>(ref);

    for (auto combination : std::apply(ranges::views::cartesian_product, dims)) {
        // We generate all the cartesian products for all the dimensions except the final dimension
        // We call write on that final dimension.
        detail::add_elements<Rank - 1>(view_offset, offset_default, combination);
        detail::add_elements<Rank - 1>(disk_offset, offset, combination);

        // Get the data pointer from the view
        T *data = ref.data_array(view_offset);
        h5::write<T>(ds, data, count_default, disk_offset);
    }
}

// This needs to be expanded to handle the various h5 parameters like above.
template <typename T, std::size_t Rank>
auto read(const h5::fd_t &fd, const std::string &name) -> tensor<T, Rank> {
    try {
        auto temp = h5::read<einsums::tensor<T, Rank>>(fd, name);
        temp.set_name(name);
        return temp;
    } catch (std::exception &e) {
        EINSUMS_THROW_EXCEPTION(error::disk_error,
                                "Unable to open disk tensor '{}'\n"
                                "{}",
                                name, e.what());
    }
}

template <typename T>
auto read(const h5::fd_t &fd, const std::string &name) -> tensor<T, 0> {
    try {
        T            temp{0};
        tensor<T, 0> tensor{name};
        h5::read<T>(fd, name, &temp, h5::count{1});
        tensor = temp;
        return tensor;
    } catch (std::exception &e) {
        EINSUMS_THROW_EXCEPTION(error::disk_error,
                                "Unable to open disk tensor '{}'\n"
                                "{}",
                                name, e.what());
    }
}

template <template <typename, std::size_t> typename TensorType, typename T, std::size_t Rank>
void zero(TensorType<T, Rank> &A) {
    A.zero();
}

namespace disk {
template <typename T, std::size_t Rank>
struct tensor final : public virtual tensor_base::disk_tensor, virtual tensor_base::tensor<T, Rank>, virtual tensor_base::lockable_tensor {
    tensor()                   = default;
    tensor(const tensor &)     = default;
    tensor(tensor &&) noexcept = default;
    ~tensor() override         = default;

    template <typename... Dims>
    explicit tensor(h5::fd_t &file, std::string name, chunk<sizeof...(Dims)> chunk, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error,
                                        "Unable to open disk tensor '{}'\n"
                                        "{}",
                                        name, e.what());
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error,
                                        "Unable to open disk tensor '{}'\n"
                                        "{}",
                                        name, e.what());
            }
        }
    }

    template <typename... Dims, typename = std::enable_if_t<util::are_all_convertible<size_t, Dims...>::value>>
    explicit tensor(h5::fd_t &file, std::string name, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr std::size_t chunk_min{64};
            if (_dims[i] < chunk_min)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = chunk_min;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error,
                                        "Unable to open disk tensor '{}'\n"
                                        "{}",
                                        name, e.what());
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk_temp} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error,
                                        "Unable to open disk tensor '{}'\n"
                                        "{}",
                                        name, e.what());
            }
        }
    }

    // Constructs a disk_tensor shaped like the provided Tensor. Data from the provided tensor
    // is NOT saved.
    explicit tensor(h5::fd_t &file, const einsums::tensor<T, Rank> &tensor) : _file{file}, _name{tensor.name()} {
        // Save dimension information from the provided tensor.
        h5::current_dims cdims;
        for (int i = 0; i < Rank; i++) {
            _dims[i] = tensor.dim(i);
            cdims[i] = _dims[i];
        }

        struct Stride {
            std::size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> std::size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr std::size_t chunk_min{64};
            if (_dims[i] < chunk_min)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = chunk_min;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error, "Unable to open disk tensor '%s'", _name.c_str());
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, cdims, h5::chunk{chunk_temp} /*| h5::gzip{9} | h5::fill_value<T>(0.0)*/);
            } catch (std::exception &e) {
                EINSUMS_THROW_EXCEPTION(error::disk_error, "Unable to create disk tensor '%s'", _name.c_str());
            }
        }
    }

    // Provides ability to store another tensor to a part of a disk tensor.

    [[nodiscard]] auto dim(int d) const -> std::size_t override { return _dims[d]; }
    auto               dims() const -> einsums::dim<Rank> override { return _dims; }

    [[nodiscard]] auto existed() const -> bool { return _existed; }

    [[nodiscard]] auto disk() -> h5::ds_t & { return _disk; }

    // void _write(Tensor<T, Rank> &data) { h5::write(disk(), data); }

    [[nodiscard]] auto name() const -> const std::string & override { return _name; }

    void set_name(const std::string &new_name) override { _name = new_name; }

    [[nodiscard]] auto stride(int d) const noexcept -> std::size_t { return _strides[d]; }

    // This creates a Disk object with its Rank being equal to the number of All{} parameters
    // Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index)
        -> std::enable_if_t<
            util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>() != 0,
            disk::view<T, util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>(),
                       Rank>> {
        using namespace util::arguments;

        // Get positions of All
        auto all_positions =
            get_array_from_tuple<std::array<int, count_of_type<all_t, MultiIndex...>()>>(positions_of_type<all_t, MultiIndex...>());
        auto index_positions =
            get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(positions_of_type<size_t, MultiIndex...>());
        auto range_positions =
            get_array_from_tuple<std::array<int, count_of_type<range, MultiIndex...>()>>(positions_of_type<range, MultiIndex...>());

        const auto &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        einsums::offset<Rank> offsets{};
        einsums::stride<Rank> strides{};
        einsums::count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        einsums::dim<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<range>(indices, value)[0];
            counts[value]  = get_from_tuple<range>(indices, value)[1] - get_from_tuple<range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return view<T, count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                            offsets, strides);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const
        -> std::enable_if_t<
            util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>() != 0,
            const view<T, util::arguments::count_of_type<all_t, MultiIndex...>() + util::arguments::count_of_type<range, MultiIndex...>(),
                       Rank>> {
        using namespace util::arguments;

        // Get positions of All
        auto all_positions =
            get_array_from_tuple<std::array<int, count_of_type<all_t, MultiIndex...>()>>(positions_of_type<all_t, MultiIndex...>());
        auto index_positions =
            get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(positions_of_type<size_t, MultiIndex...>());
        auto range_positions =
            get_array_from_tuple<std::array<int, count_of_type<range, MultiIndex...>()>>(positions_of_type<range, MultiIndex...>());

        const auto &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        einsums::offset<Rank> offsets{};
        einsums::stride<Rank> strides{};
        einsums::count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        einsums::dim<count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<range>(indices, value)[0];
            counts[value]  = get_from_tuple<range>(indices, value)[1] - get_from_tuple<range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return view<T, count_of_type<all_t, MultiIndex...>() + count_of_type<range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                            offsets, strides);
    }

  private:
    h5::fd_t &_file;

    std::string           _name;
    einsums::dim<Rank>    _dims;
    einsums::stride<Rank> _strides;

    h5::ds_t _disk;

    // Did the entry already exist on disk? Doesn't indicate validity of the data just the existence of the entry.
    bool _existed{false};
};

template <typename T, std::size_t ViewRank, std::size_t Rank>
struct view final : public virtual tensor_base::disk_tensor,
                    virtual tensor_base::tensor_view<T, ViewRank, tensor<T, Rank>>,
                    virtual tensor_base::lockable_tensor {
    view(tensor<T, Rank> &parent, const dim<ViewRank> &dims, const count<Rank> &counts, const offset<Rank> &offsets,
         const stride<Rank> &strides)
        : _parent(parent), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides), _tensor{_dims} {
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    };
    view(const tensor<T, Rank> &parent, const dim<ViewRank> &dims, const count<Rank> &counts, const offset<Rank> &offsets,
         const stride<Rank> &strides)
        : _parent(const_cast<tensor<T, Rank> &>(parent)), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides),
          _tensor{_dims} {
        profile::section const section("DiskView constructor");
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
        set_read_only(true);
    };
    view(const view &)     = default;
    view(view &&) noexcept = default;

    ~view() override { put(); }

    void set_read_only(bool readOnly) { _readOnly = readOnly; }

    auto operator=(const T *other) -> view & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        // Save the data to disk.
        h5::write<T>(_parent.disk(), other, h5::count{_counts}, h5::offset{_offsets});

        return *this;
    }

    // TODO: I'm not entirely sure if a tensor_view can be sent to the disk.
    template <template <typename, std::size_t> typename TType>
    auto operator=(const TType<T, ViewRank> &other) -> view & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(error::permission_denied, "Attempting to write data to a read only disk view.");
        }

        // Check dims
        for (int i = 0; i < ViewRank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(error::bad_parameter,
                                        fmt::format("dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i)));
            }
        }

        // Performing the write here will cause a double write to occur. The destructor above will call put to save
        // the data to disk.
        // Sync the data to disk and into our internal tensor.
        // h5::write<T>(_parent.disk(), other.data(), h5::count{_counts}, h5::offset{_offsets});
        _tensor = other;

        return *this;
    }

    // Does not perform a disk read. That was handled by the constructor.
    auto get() -> einsums::tensor<T, ViewRank> & { return _tensor; }

    void put() {
        if (!_readOnly)
            h5::write<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> const T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    [[nodiscard]] auto dim(int d) const -> std::size_t override { return _tensor.dim(d); }
    auto               dims() const -> einsums::dim<ViewRank> override { return _tensor.dims(); }

    const std::string &name() const override { return _name; }
    void               set_name(const std::string &new_name) override { _name = new_name; }

    operator einsums::tensor<T, ViewRank> &() const { return _tensor; }       // NOLINT
    operator const einsums::tensor<T, ViewRank> &() const { return _tensor; } // NOLINT

    void zero() { _tensor.zero(); }
    void set_all(T value) { _tensor.set_all(value); }

    bool full_view_of_underlying() const override {
        std::size_t prod = 1;
        for (int i = 0; i < ViewRank; i++) {
            prod *= _dims[i];
        }

        std::size_t prod2 = 1;
        for (int i = 0; i < Rank; i++) {
            prod2 *= _parent.dim(i);
        }

        return prod == prod2;
    }

  private:
    disk::tensor<T, Rank> &_parent;
    einsums::dim<ViewRank> _dims;
    einsums::count<Rank>   _counts;
    einsums::offset<Rank>  _offsets;
    einsums::stride<Rank>  _strides;
    tensor<T, ViewRank>    _tensor;
    std::string            _name{"(unnamed)"};

    bool _readOnly{false};
};
} // namespace disk

#ifdef __cpp_deduction_guides
template <typename... Args>
tensor(const std::string &, Args...) -> tensor<double, sizeof...(Args)>;
template <typename T, std::size_t OtherRank, typename... Dims>
explicit tensor(tensor<T, OtherRank> &&otherTensor, std::string name, Dims... dims) -> tensor<T, sizeof...(dims)>;
template <size_t Rank, typename... Args>
explicit tensor(const dim<Rank> &, Args...) -> tensor<double, Rank>;

template <typename T, std::size_t Rank, std::size_t OtherRank, typename... Args>
tensor_view(tensor<T, OtherRank> &, const dim<Rank> &, Args...) -> tensor_view<T, Rank>;
template <typename T, std::size_t Rank, std::size_t OtherRank, typename... Args>
tensor_view(const tensor<T, OtherRank> &, const dim<Rank> &, Args...) -> tensor_view<T, Rank>;
template <typename T, std::size_t Rank, std::size_t OtherRank, typename... Args>
tensor_view(tensor_view<T, OtherRank> &, const dim<Rank> &, Args...) -> tensor_view<T, Rank>;
template <typename T, std::size_t Rank, std::size_t OtherRank, typename... Args>
tensor_view(const tensor_view<T, OtherRank> &, const dim<Rank> &, Args...) -> tensor_view<T, Rank>;
template <typename T, std::size_t Rank, std::size_t OtherRank, typename... Args>
tensor_view(std::string, tensor<T, OtherRank> &, const dim<Rank> &, Args...) -> tensor_view<T, Rank>;

namespace disk {
template <typename... Dims>
tensor(h5::fd_t &file, std::string name, Dims... dims)->disk::tensor<double, sizeof...(Dims)>;
template <typename... Dims>
tensor(h5::fd_t &file, std::string name, chunk<sizeof...(Dims)> chunk, Dims... dims)->disk::tensor<double, sizeof...(Dims)>;
} // namespace disk

// Supposedly C++20 will allow template deduction guides for template aliases. i.e. Dim, Stride, Offset, Count, Range.
// Clang has no support for class template argument deduction for alias templates. P1814R0
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
 * @return A new tensor. By default memory is not initialized to anything. It may be filled with garbage.
 */
template <typename Type = double, typename... Args>
auto create_tensor(const std::string name, Args... args) {
    return tensor<Type, sizeof...(Args)>{name, args...};
}

template <typename Type = double, std::integral... Args>
auto create_tensor(Args... args) {
    return tensor<Type, sizeof...(Args)>{"Temporary", args...};
}

/**
 * @brief Create a disk tensor object.
 *
 * Creates a new tensor that lives on disk. This function does not create any tensor in memory
 * but the tensor is "created" on the HDF5 @p file handle.
 *
 * @code
 * auto a = create_disk_tensor(handle, "a", 3, 3);           // auto -> disk_tensor<double, 2>
 * auto b = create_disk_tensor<float>(handle, "b", 4, 5, 6); // auto -> disk_tensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying disk tensor. Defaults to double.
 * @tparam Args The datatypes of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param file The HDF5 file descriptor
 * @param name The name of the tensor. Stored in the file under this name.
 * @param args The arguments needed to constructor the tensor
 * @return A new disk tensor.
 */
template <typename Type = double, typename... Args>
auto create_disk_tensor(h5::fd_t &file, const std::string name, Args... args) -> disk::tensor<Type, sizeof...(Args)> {
    return disk::tensor<Type, sizeof...(Args)>{file, name, args...};
}

/**
 * @brief Create a disk tensor object.
 *
 * Creates a new tensor that lives on disk. This function does not create any tensor in memory
 * but the tensor is "created" on the HDF5 @p file handle. Data from the provided tensor is NOT
 * saved.
 *
 * @code
 * auto mem_a = create_tensor("a", 3, 3");           // auto -> Tensor<double, 2>
 * auto a = create_disk_tensor_like(handle, mem_a);  // auto -> disk_tensor<double, 2>
 * @endcode
 *
 * @tparam Type The datatype of the underlying disk tensor.
 * @tparam Rank The datatypes of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param file The HDF5 file descriptor
 * @param tensor The tensor to reference for size and name.
 * @return A new disk tensor.
 */
template <typename T, std::size_t Rank>
auto create_disk_tensor_like(h5::fd_t &file, const tensor<T, Rank> &tensor) -> disk::tensor<T, Rank> {
    return disk::tensor(file, tensor);
}

namespace detail {
template <typename T>
inline auto ndigits(T number) -> int {
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

template <FileOrOstream Output, TensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(Output fp, const AType &A, tensor_print_options options) {
    using T                    = typename AType::value_type;
    constexpr std::size_t Rank = AType::rank;

    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};

        if constexpr (CoreTensorConcept<AType>) {
            if constexpr (!TensorViewConcept<AType>)
                fprintln(fp, "Type: In Core Tensor");
            else
                fprintln(fp, "Type: In Core Tensor View");
#if defined(EINSUMS_COMPUTE_CODE)
        } else if constexpr (DeviceTensorConcept<AType>) {
            if constexpr (!DeviceTensorViewConcept<AType>)
                fprintln(fp, "Type: Device Tensor");
            else
                fprintln(fp, "Type: Device Tensor View");
#endif
        } else if constexpr (DiskTensorConcept<AType>) {
            fprintln(fp, "Type: Disk Tensor");
        } else {
            fprintln(fp, "Type: {}", util::type_name<AType>());
        }

        fprintln(fp, "Data Type: {}", util::type_name<typename AType::value_type>());

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
                } else if constexpr (is_complex_v<T>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(fp, "{}", oss.str());
                fprintln(fp);
#if !defined(EINSUMS_COMPUTE_CODE)
            } else if constexpr (Rank > 1 && einsums::CoreTensorConcept<AType>) {
#else
            } else if constexpr (Rank > 1 && (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>)) {
#endif
                auto target_dims = einsums::get_dim_ranges<Rank - 1>(A);
                auto final_dim   = A.dim(Rank - 1);
                auto ndigits     = detail::ndigits(final_dim);

                for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
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
                        auto new_tuple = std::tuple_cat(target_combination.base(), std::tuple(j));
                        T    value     = std::apply(A, new_tuple);
                        if (std::abs(value) > 1.0E+10) {
                            if constexpr (std::is_floating_point_v<T>)
                                oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                            else if constexpr (is_complex_v<T>)
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
                            } else if constexpr (is_complex_v<T>) {
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
#if !defined(EINSUMS_COMPUTE_CODE)
            } else if constexpr (Rank == 1 && einsums::CoreTensorConcept<AType>) {
#else
            } else if constexpr (Rank == 1 && (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>)) {
#endif
                auto target_dims = einsums::get_dim_ranges<Rank>(A);

                for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
                    std::ostringstream oss;
                    oss << "(";
                    oss << fmt::format("{}", fmt::join(target_combination, ", "));
                    oss << "): ";

                    T value = std::apply(A, target_combination);
                    if (std::abs(value) > 1.0E+5) {
                        if constexpr (std::is_floating_point_v<T>)
                            oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "{:14.8f} ", value);
                        else if constexpr (is_complex_v<T>) {
                            oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "({:14.8f} + {:14.8f})", value.real(),
                                               value.imag());
                        } else
                            oss << fmt::format(fmt::fg(fmt::color::white) | fmt::bg(fmt::color::red), "{:14} ", value);
                    } else {
                        if constexpr (std::is_floating_point_v<T>)
                            if (std::abs(value) < 1.0E-4) {
                                oss << fmt::format("{:14.4e} ", value);
                            } else {
                                oss << fmt::format("{:14.8f} ", value);
                            }
                        else if constexpr (is_complex_v<T>) {
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

template <einsums::TensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void println(const AType &A, tensor_print_options options) {
    fprintln(stdout, A, options);
}

} // namespace einsums
