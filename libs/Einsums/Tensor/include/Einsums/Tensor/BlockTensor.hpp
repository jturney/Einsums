//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file BlockTensor.hpp
 */

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

/// @todo
#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/Tensor/DeviceTensor.hpp>
#endif

#include <concepts>
#include <functional>
#include <string>

namespace einsums {
namespace tensor_base {

/**
 * @struct BlockTensor
 *
 * Represents a block-diagonal tensor.
 *
 * @tparam T The data type stored in this tensor.
 * @tparam Rank The rank of the tensor
 * @tparam TensorType The underlying type for the tensors.
 */
template <typename T, size_t rank, typename TensorType>
struct BlockTensor : public BlockTensorNoExtra, public design_pats::Lockable<std::recursive_mutex>, AlgebraOptimizedTensor {
  public:
    /**
     * @typedef StoredType
     *
     * @brief The kind of tensor that is used by this structured tensor to store data.
     */
    using StoredType = TensorType;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef ValueType
     *
     * @brief The type of data stored in the tensor.
     */
    using ValueType = T;

    /**
     * @name Constructors
     * @{
     */
    /**
     * Construct a new BlockTensor object.
     */
    BlockTensor() = default;

    /**
     * Copy constructs a new BlockTensor object.
     */
    BlockTensor(BlockTensor const &other) : _ranges{other._ranges}, _dims{other._dims}, _blocks{}, _dim{other._dim} {
        _blocks.reserve(other._blocks.size());
        for (int i = 0; i < other._blocks.size(); i++) {
            _blocks.emplace_back((TensorType const &)other._blocks[i]);
        }

        update_dims();
    }

    /**
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     * The values in \p block_dims must be castable to size_t.
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling BlockTensor.zero() or zero(BlockTensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <std::convertible_to<size_t>... Dims>
    explicit BlockTensor(std::string name, Dims... block_dims)
        : _name{std::move(name)}, _dim{(static_cast<size_t>(block_dims) + ... + 0)}, _blocks(), _ranges(), _dims(sizeof...(Dims)) {
        auto dim_array   = Dim<sizeof...(Dims)>{block_dims...};
        auto _block_dims = Dim<Rank>();
        _blocks.reserve(sizeof...(Dims));

        for (int i = 0; i < sizeof...(Dims); i++) {
            _block_dims.fill(dim_array[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     * The values in \p block_dims need to be castable to size_t.
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", std::array<int>{2, 3});
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam ArrayArg A container type that stores the dimensions. For instance, std::array or einsums::Dim.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockTensor(std::string name, ArrayArg const &block_dims)
        : _name{std::move(name)}, _dim{0}, _blocks(), _ranges(), _dims(block_dims.cbegin(), block_dims.cend()) {

        auto _block_dims = Dim<Rank>();
        _blocks.reserve(block_dims.size());

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     * The values in \p block_dims need to be castable to size_t.
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", {2, 3});
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam IntType The type of the size values. This should be determined automatically.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <std::convertible_to<size_t> IntType>
    explicit BlockTensor(std::string name, std::initializer_list<IntType> block_dims)
        : _name{std::move(name)}, _dim{0}, _blocks(), _ranges(), _dims(block_dims.begin(), block_dims.end()) {

        auto _block_dims = Dim<Rank>();
        _blocks.reserve(block_dims.size());

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * Constructs a new BlockTensor object using the information provided in \p block_dims .
     * The values in \p block_dims need to be castable to size_t.
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>(std::array<int>{2, 3});
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam ArrayArg A container type that stores the dimensions. For instance, std::array or einsums::Dim.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockTensor(ArrayArg const &block_dims) : _blocks(), _ranges(), _dims(block_dims.cbegin(), block_dims.cend()) {
        auto _block_dims = Dim<Rank>();

        _blocks.reserve(block_dims.size());

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(_block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * Constructs a new BlockTensor object using the information provided in \p block_dims .
     * The values in \p block_dims need to be castable to size_t.
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>({2, 3});
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam ArrayArg A container type that stores the dimensions. For instance, std::array or einsums::Dim.
     * @param block_dims The size of each block.
     */
    template <std::convertible_to<size_t> IntType>
    explicit BlockTensor(std::initializer_list<IntType> block_dims) : _blocks(), _ranges(), _dims(block_dims.begin(), block_dims.end()) {
        auto _block_dims = Dim<Rank>();

        _blocks.reserve(block_dims.size());

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(_block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * Destroy the BlockTensor object.
     */
    ~BlockTensor() = default;
    // End constructor group
    /**
     * @}
     */

    /**
     * @brief Find the block which can be indexed by the given \p index.
     *
     * This finds the block for which the \p index is within its range. Since these
     * tensors are always square/hypercubic, the index is valid for all axes.
     * This index should be greater than or equal to zero and less than the dimension
     * of the tensor along any axis.
     *
     * @param index The index to test.
     * @return The index of the block containing the given index.
     * @throws std::out_of_range Throws this when the index it outside of the tensor.
     */
    int block_of(size_t index) const {
        for (int i = 0; i < _ranges.size(); i++) {
            if (_ranges[i][0] <= index && _ranges[i][1] > index) {
                return i;
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range!");
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].zero();
        }
    }

    /**
     * @brief Set the all entries in the blocks to the given value.
     *
     * This does not set the value in the unoccupied blocks.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].set_all(value);
        }
    }

    /**
     * @brief Return the selected block with an integer ID.
     *
     * @param id The index of the block in the list of blocks.
     * @return The block requested.
     * @throws std::out_of_range if \p id is outside of the list of blocks.
     */
    TensorType const &block(int id) const { return _blocks.at(id); }

    /// @copydoc block(int) const
    TensorType &block(int id) { return _blocks.at(id); }

    /**
     * @brief Return the first block with the given name.
     *
     * @param name The name of the block to find.
     * @return The requested block.
     * @throws std::out_of_range if no block with the given name is contained in this tensor.
     */
    TensorType const &block(std::string const &name) const {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}: no blocks in tensor", name);
        }
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}: no blocks with given name", name);
    }

    /**
     * @copydoc block(std::string const &) const
     */
    TensorType &block(std::string const &name) {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}: no blocks in tensor", name);
        }
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}: no blocks with given name", name);
    }

    /**
     * @brief Add a block to the end of the list of blocks.
     *
     * @param value The tensor to push.
     * @throws dimension_error if the tensor being pushed is not square.
     */
    void push_block(TensorType value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    dimension_error, "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        update_dims();
    }

    /**
     * @brief Add a block to the specified position in the list of blocks.
     *
     * @param pos The position to insert at.
     * @param value The tensor to insert.
     * @throws dimension_error if the tensor being pushed is not square.
     */
    void insert_block(int pos, TensorType value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    dimension_error, "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        update_dims();
    }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location, or nullptr if it is outside of the blocks.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN)
        assert(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return &(_blocks[block].data()[ordinal]);
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
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> T const & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return 0; // The indices point outside of all the blocks.
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks.at(block).strides().begin(), size_t{0});
        return _blocks.at(block).data()[ordinal];
    }

    /**
     * @copydoc BlockTensor<T, Rank>::operator(MultiIndex... index) const
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return subscript_tensor(_blocks.at(block), index_list);
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam Container A container type, such as std::array.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return T& A reference to the value at that index.
     *
     * @throws not_enough_args Throws this if the container doesn't have enough indices.
     * @throws too_many_args Throws this if the container has too many indices.
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
        if (index.size() < Rank) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        std::array<std::int64_t, Rank> index_list{};

        for (int i = 0; i < Rank; i++) {
            index_list[i] = index[i];
        }

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return subscript_tensor(_blocks.at(block), index_list);
    }

    /**
     * @copydoc BlockTensor<T, Rank>::operator()(Container const &index)
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
        if (index.size() < Rank) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > Rank) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }

        std::array<std::int64_t, Rank> index_list{};

        for (int i = 0; i < Rank; i++) {
            index_list[i] = index[i];
        }

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return subscript_tensor(_blocks.at(block), index_list);
    }

    /**
     * @brief Return the block with the given index. Equivalent to block(index)
     */
    TensorType const &operator[](size_t index) const { return this->block(index); }

    /**
     * @copydoc operator[](size_t) const
     */
    TensorType &operator[](size_t index) { return this->block(index); }

    /**
     * @copydoc operator[](size_t) const
     */
    TensorType const &operator[](std::string const &name) const { return this->block(name); }

    /**
     * @copydoc operator[](size_t) const
     */
    TensorType &operator[](std::string const &name) { return this->block(name); }

    /**
     * @brief Copy assignment.
     */
    auto operator=(BlockTensor<T, Rank, TensorType> const &other) -> BlockTensor<T, Rank, TensorType> & {

        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        assert(_dims.size() > 0 || _dim == 0);
        assert(_dims.size() == _ranges.size());
        assert(_dims.size() == _blocks.size());

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

    /**
     * @brief Copy assignment with a cast.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(BlockTensor<TOther, Rank, TensorType> const &other) -> BlockTensor<T, Rank, TensorType> & {
        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

#ifndef DOXYGEN
#    define OPERATOR(OP)                                                                                                                   \
        auto operator OP(const T &b)->BlockTensor<T, Rank, TensorType> & {                                                                 \
            for (int i = 0; i < _blocks.size(); i++) {                                                                                     \
                if (block_dim(i) == 0) {                                                                                                   \
                    continue;                                                                                                              \
                }                                                                                                                          \
                _blocks[i] OP b;                                                                                                           \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        auto operator OP(const BlockTensor<T, Rank, TensorType> &b)->BlockTensor<T, Rank, TensorType> & {                                  \
            if (_blocks.size() != b._blocks.size()) {                                                                                      \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "tensors differ in number of blocks : {} {}", _blocks.size(),                 \
                                        b._blocks.size());                                                                                 \
            }                                                                                                                              \
            for (int i = 0; i < _blocks.size(); i++) {                                                                                     \
                if (_blocks[i].size() != b._blocks[i].size()) {                                                                            \
                    EINSUMS_THROW_EXCEPTION(dimension_error, "tensor blocks differ in size : {} {}", _blocks[i].size(),                    \
                                            b._blocks[i].size());                                                                          \
                }                                                                                                                          \
            }                                                                                                                              \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (int i = 0; i < _blocks.size(); i++) {                                                                                     \
                if (block_dim(i) == 0) {                                                                                                   \
                    continue;                                                                                                              \
                }                                                                                                                          \
                _blocks[i] OP b._blocks[i];                                                                                                \
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
     * @brief Convert block tensor into a normal tensor.
     */
    explicit operator TensorType() const {
        Dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (this->block_dim(i) == 0) {
                continue;
            }
            std::array<Range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    /**
     * @copydoc BlockTensor<T, Rank>::operatorTensorType() const
     */
    explicit operator TensorType() {
        update_dims();
        Dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (this->block_dim(i) == 0) {
                continue;
            }
            std::array<Range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    /**
     * @brief Return the number of blocks.
     */
    size_t num_blocks() const { return _blocks.size(); }

    /**
     * @brief Return the dimensions of each block.
     */
    [[nodiscard]] auto block_dims() const -> std::vector<size_t> const { return _dims; }

    /**
     * @brief Return a list containing the ranges for each block.
     */
    std::vector<Range> ranges() const { return _ranges; }

    /**
     * @brief Return the range for a given block.
     */
    Range block_range(int i) const { return _ranges.at(i); }

    /**
     * @brief Return the dimensions of the given block.
     */
    Dim<Rank> block_dims(size_t block) const { return _blocks.at(block).dims(); }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(size_t block, int ind = 0) const { return _blocks.at(block).dim(ind); }

    /**
     * @brief Return the dimensions of a given block.
     */
    Dim<Rank> block_dims(std::string const &name) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dims();
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}", name);
    }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(std::string const &name, int ind = 0) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dim(ind);
            }
        }

        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Could not find block with the name {}", name);
    }

    /**
     * @brief Return the dimensions of this tensor.
     */
    virtual Dim<Rank> dims() const {
        Dim<Rank> out;
        out.fill(_dim);
        return out;
    }

    /**
     * @brief Return the dimension of this tensor along an axis.
     *
     * Because the tensor is square, the argument is ignored.
     */
    virtual size_t dim(int dim) const { return _dim; }

    /**
     * @brief Return the dimension of this tensor along the first axis.
     */
    virtual size_t dim() const { return _dim; }

    /**
     * @brief Return the dimensions of each of the blocks.
     */
    std::vector<size_t> vector_dims() const {
        std::vector<size_t> out(num_blocks());

        for (int i = 0; i < out.size(); i++) {
            out[i] = _blocks[i].dim(0);
        }

        return out;
    }

    /**
     * @brief Returns the list of tensors.
     */
    auto vector_data() const -> std::vector<TensorType> const & { return _blocks; }

    /**
     * @copydoc vector_data() const
     */
    auto vector_data() -> std::vector<TensorType> & { return _blocks; }

    /**
     * @brief Gets the name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * @brief Sets the name of the tensor.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * @brief Gets the name of a block.
     */
    [[nodiscard]] auto name(int i) const -> std::string const & { return _blocks[i].name(); }

    /**
     * @brief Sets the name of a block.
     */
    void set_name(int i, std::string const &name) { _blocks[i].set_name(name); }

    /**
     * @brief Returns the strides of a given block.
     */
    auto strides(int i) const noexcept -> auto const & { return _blocks[i].strides(); }

    /**
     * Returns the size of the tensor. That is, the product of the dimensions.
     */
    [[nodiscard]] size_t size() const {
        size_t product = 1;
        for (int i = 0; i < Rank; i++) {
            product *= this->_dim;
        }

        return product;
    }

    /**
     * @brief Returns true if all of the elements are viewed by this tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return true; }

    /**
     * @brief Lock the tensor.
     */
    void lock() const { Lockable::lock(); }

    /**
     * @brief Unlock the tensor.
     */
    void unlock() const { Lockable::unlock(); }

    /**
     * @brief Try to lock the tensor, returning true if successful.
     */
    bool try_lock() const { return Lockable::try_lock(); }

    /**
     * Lock a specific block.
     */
    virtual void lock(int block) const {
        if constexpr (einsums::IsLockableV<TensorType>) {
            _blocks.at(block).lock();
        }
    }

    /**
     * Try to lock a specific block.
     */
    virtual bool try_lock(int block) const {
        if constexpr (einsums::IsLockableV<TensorType>) {
            return _blocks.at(block).try_lock();
        } else {
            return true;
        }
    }

    /**
     * Unlock a specific block.
     */
    virtual void unlock(int block) const {
        if constexpr (einsums::IsLockableV<TensorType>) {
            _blocks.at(block).unlock();
        }
    }

  protected:
    /**
     * @var _name
     *
     * @brief This is the name of the tensor used for printing.
     */
    std::string _name{"(Unnamed)"};

    /**
     * @var _dim
     *
     * @brief This is the dimension of the tensor along one axis.
     */
    size_t _dim{0}; // Only allowing square tensors.

    /**
     * @var _blocks
     *
     * @brief This stores the blocks of the tensor.
     */
    std::vector<TensorType> _blocks{};

    /**
     * @var _ranges
     *
     * @brief This stores the starting and ending indices for each of the blocks in relation
     * to the whole tensor.
     */
    std::vector<Range> _ranges{};

    /**
     * @var _dims
     *
     * @brief This stores the dimensions of each of the blocks along an axis.
     */
    std::vector<size_t> _dims;

    template <typename T_, size_t OtherRank, typename OtherTensorType>
    friend struct BlockTensor;

    /**
     * @var _zero_value
     *
     * @brief This holds the value outside of the blocks of the tensor.
     *
     * This value is updated to be zero whenever it is accessed, meaning it
     * essentially cannot be changed. However, if it is changed, it won't be
     * reset until the next time it is returned.
     */
    T _zero_value{0.0};

    // template <typename T_, size_t Rank_>
    // friend struct BlockTensorViewBase;

    /**
     * Go through the blocks of the tensor and make sure that the tensor's dimensions
     * matches the sum of the dimensions of the blocks. Also make sure that the stored
     * dimensions match the actual dimensions.
     */
    void update_dims() {
        if (_dims.size() != _blocks.size()) {
            _dims.resize(_blocks.size());
        }
        if (_ranges.size() != _blocks.size()) {
            _ranges.resize(_blocks.size());
        }

        size_t sum = 0;

        for (int i = 0; i < _blocks.size(); i++) {
            _dims[i]   = _blocks[i].dim(0);
            _ranges[i] = Range{sum, sum + _dims[i]};
            sum += _dims[i];
        }

        _dim = sum;
    }
};
} // namespace tensor_base

/**
 * @struct BlockTensor
 *
 * Represents a block-diagonal tensor in core memory.
 *
 * @tparam T The type of data stored in the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BlockTensor : public tensor_base::BlockTensor<T, Rank, Tensor<T, Rank>>, tensor_base::CoreTensor {
    /**
     * @brief Construct a new BlockTensor object. Default constructor.
     */
    BlockTensor() = default;

    /**
     * @brief Construct a new BlockTensor object. Default copy constructor
     */
    BlockTensor(BlockTensor const &) = default;

    /**
     * @brief Destroy the BlockTensor object.
     */
    ~BlockTensor() = default;

    /**
     * @brief Construct a new BlockTensor object with the given name and dimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed BlockTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockTensor(std::string name, Dims... block_dims) : tensor_base::BlockTensor<T, Rank, Tensor<T, Rank>>(name, block_dims...) {}

    /**
     * @brief Construct a new BlockTensor object with the given name and dimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed BlockTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockTensor(std::string name, ArrayArg const &block_dims)
        : tensor_base::BlockTensor<T, Rank, Tensor<T, Rank>>(name, block_dims) {}

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockTensor(Dim<Dims> block_dims) : tensor_base::BlockTensor<T, Rank, Tensor<T, Rank>>(block_dims) {}

    operator TiledTensorView<T, Rank>() {
        std::array<std::vector<size_t>, Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = this->_dims;
        }

        TiledTensorView<T, Rank> out{"block view", block_dims};

        std::array<int, Rank> index;

        for (int i = 0; i < this->_blocks.size(); i++) {
            index.fill(i);
            out.insert_tile(index, TensorView<T, Rank>(this->_blocks[i], this->_blocks[i].dims()));
        }

        return out;
    }

    operator TiledTensorView<T, Rank> const() const {
        std::array<std::vector<size_t>, Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = this->_dims;
        }

        TiledTensorView<T, Rank> out{"block view", block_dims};

        std::array<int, Rank> index;

        for (int i = 0; i < this->_blocks.size(); i++) {
            index.fill(i);
            out.insert_tile(index, TensorView<T, Rank>(this->_blocks[i], this->_blocks[i].dims()));
        }

        return out;
    }
};

#ifdef EINSUMS_COMPUTE_CODE
/**
 * @struct BlockDeviceTensor
 *
 * Represents a block-diagonal tensor stored on the device.
 *
 * @tparam T The type of data stored. Automatic conversion of complex types.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BlockDeviceTensor : public tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>, tensor_base::DeviceTensorBase {
  public:
    /**
     * @typedef host_datatype
     *
     * @brief The type of data stored as seen by the host.
     *
     * @sa tensor_props::DeviceTypedTensor::host_datatype
     */
    using host_datatype = typename tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @typedef dev_datatype
     *
     * @brief The type of data stored as seen by the device.
     *
     * @sa tensor_props::DeviceTypedTensor::dev_datatype
     */
    using dev_datatype = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default constructor.
     */
    BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default copy constructor
     */
    BlockDeviceTensor(BlockDeviceTensor const &) = default;

    /**
     * @brief Destroy the BlockDeviceTensor object.
     */
    ~BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... block_dims)
        : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name) {

        this->_blocks.reserve(sizeof...(Dims));

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>::push_block(einsums::DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, ArrayArg const &block_dims)
        : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name) {

        this->_blocks.reserve(block_dims.size());
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>::push_block(einsums::DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param mode The storage mode.
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(detail::HostToDeviceMode mode, Dim<Dims> block_dims)
        : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>() {
        this->_blocks.reserve(Dims);
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            this->push_block(einsums::DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
        requires(NoneOfType<detail::HostToDeviceMode, Dims...>)
    explicit BlockDeviceTensor(std::string name, Dims... block_dims)
        : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name) {
        this->_blocks.reserve(sizeof...(Dims));

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>::push_block(
                einsums::DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, ArrayArg const &block_dims)
        : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name) {
        this->_blocks.reserve(block_dims.size());

        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>::push_block(
                einsums::DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(Dim<Dims> block_dims) : tensor_base::BlockTensor<T, Rank, einsums::DeviceTensor<T, Rank>>() {
        this->_blocks.reserve(Dims);

        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            this->push_block(einsums::DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto gpu_data(MultiIndex... index) -> T * {
#    if !defined(DOXYGEN)
        assert(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = this->_dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < this->_ranges.size(); j++) {
                    if (this->_ranges[j][0] <= _index && _index < this->_ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (this->_ranges[block][0] <= _index && _index < this->_ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= this->_ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), this->_blocks[block].strides().begin(), size_t{0});
        return &(this->_blocks[block].gpu_data()[ordinal]);
#    endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This is different from the normal subscript since there needs to be a way to
     * access data on the device. C++ references do not handle data synchronization
     * in this way, so it needs to be done differently.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> HostDevReference<T> {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = this->_dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < this->_ranges.size(); j++) {
                    if (this->_ranges[j][0] <= _index && _index < this->_ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (this->_ranges.at(block)[0] <= _index && _index < this->_ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= this->_ranges.at(block)[0];
            } else {
                return HostDevReference<T>();
            }
        }

        return subscript_tensor(this->_blocks.at(block), index_list);
    }

    operator TiledDeviceTensorView<T, Rank>() {
        std::array<std::vector<size_t>, Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = this->_dims;
        }

        TiledDeviceTensorView<T, Rank> out("block view", block_dims);

        std::array<int, Rank> index;

        for (int i = 0; i < this->_blocks.size(); i++) {
            index.fill(i);
            out.insert_tile(index, DeviceTensorView<T, Rank>(this->_blocks[i], this->_blocks[i].dims()));
        }

        return out;
    }

    operator TiledDeviceTensorView<T, Rank> const() const {
        std::array<std::vector<size_t>, Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = this->_dims;
        }

        TiledDeviceTensorView<T, Rank> out("block view", block_dims);

        std::array<int, Rank> index;

        for (int i = 0; i < this->_blocks.size(); i++) {
            index.fill(i);
            out.insert_tile(index, DeviceTensorView<T, Rank>(this->_blocks[i], this->_blocks[i].dims()));
        }

        return out;
    }
};
#endif

#ifndef DOXYGEN
template <einsums::BlockTensorConcept AType>
void println(AType const &A, TensorPrintOptions options = {}) {
    println("Name: {}", A.name());
    {
        print::Indent const indent{};
        println("Block Tensor");
        println("Data Type: {}", type_name<typename AType::ValueType>());

        for (int i = 0; i < A.num_blocks(); i++) {
            println(A[i], options);
        }
    }
}

template <einsums::BlockTensorConcept AType, typename... Args>
void fprintln(FILE *fp, AType const &A, TensorPrintOptions options = {}) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Block Tensor");
        fprintln(fp, "Data Type: {}", type_name<typename AType::ValueType>());

        for (int i = 0; i < A.num_blocks(); i++) {
            fprintln(fp, A[i], options);
        }
    }
}

template <einsums::BlockTensorConcept AType, typename... Args>
void fprintln(std::ostream &os, AType const &A, TensorPrintOptions options = {}) {
    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(os, "Block Tensor");
        fprintln(os, "Data Type: {}", type_name<typename AType::ValueType>());

        for (int i = 0; i < A.num_blocks(); i++) {
            fprintln(os, A[i], options);
        }
    }
}

EINSUMS_TENSOR_EXPORT_RANK(BlockTensor, 2)
EINSUMS_TENSOR_EXPORT_RANK(BlockTensor, 3)
EINSUMS_TENSOR_EXPORT_RANK(BlockTensor, 4)

#    ifdef EINSUMS_COMPUTE_CODE
// EINSUMS_TENSOR_EXPORT_RANK(BlockDeviceTensor, 2)
// EINSUMS_TENSOR_EXPORT_RANK(BlockDeviceTensor, 3)
// EINSUMS_TENSOR_EXPORT_RANK(BlockDeviceTensor, 4)
#    endif

#endif

} // namespace einsums