//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <string>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/Tensor/DeviceTensor.hpp>
#endif

#include <array>
#include <cmath>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace einsums {

namespace tensor_base {

/**
 * @struct TiledTensor
 *
 * Represents a tiled tensor. Is a lockable type.
 *
 * @tparam TensorType The underlying storage type.
 * @tparam T The type of data being stored.
 * @tparam Rank The tensor rank.
 */
template <typename T, size_t rank, typename TensorType>
struct TiledTensor : public TiledTensorNoExtra, design_pats::Lockable<std::recursive_mutex>, AlgebraOptimizedTensor {
  public:
    /**
     * @typedef map_type
     *
     * @brief The data type used to hold the subtensors.
     */
    using map_type = typename std::unordered_map<std::array<int, rank>, TensorType, einsums::hashes::container_hash<std::array<int, rank>>>;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef ValueType
     *
     * @brief Represents the type of data stored in this tensor.
     */
    using ValueType = T;

    /**
     * @typedef StoredType
     *
     * @brief The types of tensor this collected tensor stores.
     */
    using StoredType = TensorType;

    /**
     * Create a new empty tiled tensor.
     */
    TiledTensor() : _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{}, _grid_size{0} {}

    /**
     * Create a new empty tiled tensor with the given grid. If only one grid is given, the grid is applied to all dimensions.
     * Otherwise, the number of grids must match the rank.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <ContainerOrInitializer... Sizes>
        requires(!ContainerOrInitializer<typename Sizes::value_type> && ... && true)
    TiledTensor(std::string name, Sizes const &...sizes) : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
        static_assert(sizeof...(Sizes) == rank || sizeof...(Sizes) == 1);

        _size = 1;
        if constexpr (sizeof...(Sizes) == rank) {
            auto size_tuple = std::make_tuple(sizes...);
            for_sequence<rank>([&](auto i) {
                auto &size = std::get<i>(size_tuple);

                this->_tile_sizes[(int)i] = std::vector<int>(size.size());
                for (int j = 0; j < size.size(); j++) {
                    this->_tile_sizes[(int)i][j] = size[j];
                }
            });
        } else {
            for (int i = 0; i < rank; i++) {
                _tile_sizes[i] = std::vector<int>(sizes.size()...);

                for (int j = 0; j < _tile_sizes[i].size(); j++) {
                    _tile_sizes[i][j] = (sizes[j] + ...);
                }
            }
        }
        for (int i = 0; i < rank; i++) {
            _tile_offsets[i] = std::vector<int>();
            _tile_offsets[i].reserve(_tile_sizes[i].size());

            int sum = 0;
            for (int j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_offsets[i].push_back(sum);
                sum += _tile_sizes[i].at(j);
            }
            _dims[i] = sum;
            _size *= sum;
        }

        _grid_size = 1;

        for (int i = 0; i < rank; i++) {
            _grid_size *= _tile_offsets[i].size();
        }
    }

    /**
     * @brief Create a new empty tiled tensor with the given grid.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <Container ContainerType>
        requires(Container<typename ContainerType::value_type> && std::is_integral_v<typename ContainerType::value_type::value_type>)
    TiledTensor(std::string name, ContainerType const &sizes) : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
        if (sizes.size() != rank) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Wrong number of grid sizes passed to TiledTensor constructor!");
        }
        for (int i = 0; i < rank; i++) {
            _tile_sizes[i] = std::vector<int>(sizes[i].size());

            for (int j = 0; j < sizes[i].size(); j++) {
                _tile_sizes[i][j] = sizes[i][j];
            }
        }
        _size = 1;
        for (int i = 0; i < rank; i++) {
            _tile_offsets[i] = std::vector<int>();
            _tile_offsets[i].reserve(_tile_sizes[i].size());
            int sum = 0;
            for (int j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_offsets[i].push_back(sum);
                sum += _tile_sizes[i].at(j);
            }
            _dims[i] = sum;
            _size *= sum;
        }

        _grid_size = 1;

        for (int i = 0; i < rank; i++) {
            _grid_size *= _tile_offsets[i].size();
        }
    }

    /**
     * Copy a tiled tensor.
     *
     * @param other The tensor to be copied.
     */
    TiledTensor(TiledTensor<T, rank, TensorType> const &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims}, _grid_size{other._grid_size} {
        for (auto &pair : other._tiles) {
            _tiles.insert_or_assign(pair.first, pair.second);
        }
    }

    /**
     * Copy a tiled tensor from one tensor type to another.
     *
     * @param other The tensor to be copied and converted.
     */
    template <TRTensorConcept<rank, T> OtherTensor>
    TiledTensor(TiledTensor<T, rank, OtherTensor> const &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims}, _grid_size{other._grid_size} {
        for (auto &pair : other._tiles) {
            _tiles.insert_or_assign(pair.first, pair.second);
        }
    }

    virtual ~TiledTensor() = default;

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    TensorType &tile(MultiIndex... index) {
        std::array<int, rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            Dim<rank> dims{};

            for (int i = 0; i < rank; i++) {
                dims[i] = _tile_sizes[i].at(arr_index[i]);
            }

            add_tile(arr_index);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    const TensorType &tile(MultiIndex... index) const {
        std::array<int, rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, this will throw an error.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    const TensorType &tile(Storage index) const {
        std::array<int, rank> arr_index;

        for (int i = 0; i < rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    TensorType &tile(Storage index) {
        std::array<int, rank> arr_index{index};

        for (int i = 0; i < rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            Dim<rank> dims{};

            for (int i = 0; i < rank; i++) {
                dims[i] = _tile_sizes[i].at(arr_index[i]);
            }
            add_tile(arr_index);
        }

        return _tiles.at(arr_index);
    }

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    bool has_tile(MultiIndex... index) const {
        std::array<int, rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
                return false;
            }
        }

        return _tiles.count(arr_index) > 0;
    }

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    std::array<int, rank> tile_of(MultiIndex... index) const {
        std::array<int, rank> arr_index{static_cast<int>(index)...};
        std::array<int, rank> out{0};

        for (int i = 0; i < rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index not in the tensor!");
            }

            if (arr_index[i] >= _tile_offsets[i].at(_tile_offsets[i].size() - 1)) {
                out[i] = _tile_offsets[i].size() - 1;
            } else {
                for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                    if (arr_index[i] < _tile_offsets[i].at(j + 1) && arr_index[i] >= _tile_offsets[i].at(j)) {
                        out[i] = j;
                        break;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Returns whether a tile exists at a given position, and if it is filled.
     *
     * @param index The position to check for a tile.
     * @return True if there is a tile and it is initialized at this position. False if there is no tile or it is not initialized.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_tile(Storage index) const {
        std::array<int, rank> arr_index;

        for (int i = 0; i < rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
            if (arr_index[i] < 0) {
                index[i] += _tile_sizes[i].size();
            }

            if (arr_index[i] >= _tile_sizes[i].size() || arr_index[i] < 0) {
                return false;
            }
        }

        return _tiles.count(arr_index) > 0;
    }

    /**
     * Returns the tile coordinates of a given tensor index.
     *
     * @param index The tensor index to check.
     * @return The coordinates of the tile.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    std::array<int, rank> tile_of(Storage index) const {
        std::array<int, rank> arr_index;
        std::array<int, rank> out{0};

        for (int i = 0; i < rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index not in the tensor!");
            }

            if (arr_index[i] >= _tile_offsets[i].at(_tile_offsets[i].size() - 1)) {
                out[i] = _tile_offsets[i].size() - 1;
            } else {
                for (int j = 0; j < _tile_offsets[i].size() - 1; j++) {
                    if (arr_index[i] < _tile_offsets[i].at(j + 1) && arr_index[i] >= _tile_offsets[i].at(j)) {
                        out[i] = j;
                        break;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T operator()(MultiIndex... index) const {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }

        if (has_tile(coords)) {
            return subscript_tensor(tile(coords), array_ind);
        } else {
            return T{0.0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T at(MultiIndex... index) const {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }

        if (has_tile(coords)) {
            return subscript_tensor(tile(coords), array_ind);
        } else {
            return T{0.0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == rank)
    T &operator()(MultiIndex... index) {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int64_t, rank>{static_cast<int64_t>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }
        auto &out = tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T operator()(ContainerType const &index) const {
        if (index.size() < rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Not enough indices passed to Tensor!");
        } else if (index.size() > rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Too many indices passed to Tensor!");
        }
        auto coords = tile_of(index);

        std::array<std::int64_t, rank> array_ind;

        for (size_t i = 0; i < rank; i++) {
            array_ind[i] = index[i];
        }

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }

        if (has_tile(coords)) {
            return subscript_tensor(tile(coords), array_ind);
        } else {
            return T{0.0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T at(ContainerType const &index) const {
        if (index.size() < rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Not enough indices passed to Tensor!");
        } else if (index.size() > rank) {
            [[unlikely]] EINSUMS_THROW_EXCEPTION(std::out_of_range, "Too many indices passed to Tensor!");
        }
        auto coords = tile_of(index);

        std::array<std::int64_t, rank> array_ind;

        for (size_t i = 0; i < rank; i++) {
            array_ind[i] = index[i];
        }

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }

        if (has_tile(coords)) {
            return subscript_tensor(tile(coords), array_ind);
        } else {
            return T{0.0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <typename ContainerType>
        requires requires {
            requires !std::is_integral_v<ContainerType>;
            requires !std::is_same_v<ContainerType, Dim<rank>>;
            requires !std::is_same_v<ContainerType, Stride<rank>>;
            requires !std::is_same_v<ContainerType, Offset<rank>>;
            requires !std::is_same_v<ContainerType, Range>;
        }
    T &operator()(ContainerType const &index) {
        if (index.size() < rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to Tensor!");
        } else if (index.size() > rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to Tensor!");
        }
        auto coords = tile_of(index);

        std::array<std::int64_t, rank> array_ind;

        for (size_t i = 0; i < rank; i++) {
            array_ind[i] = index[i];
        }

        // Find the index in the tile.
        for (int i = 0; i < rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }
        auto &out = tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * Sets all entries in the tensor to zero. This clears all tiles. There will be no more tiles after this.
     */
    void zero() { _tiles.clear(); }

    /**
     * Sets all entries in the tensor to zero. This keeps all tiles, just calls zero on the tensors.
     */
    void zero_no_clear() {
        for (auto &tile : _tiles) {
            tile.second.zero();
        }
    }

    /**
     * Sets all entries to the given value. Initializes all tiles, unless zero is given.
     * If zero is passed, calls @ref zero
     *
     * @param value The value to broadcast.
     */
    void set_all(T value) {
        if (value == T{0}) {
            zero();
            return;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles.at(tile_index).set_all(value);
        }
    }

    /**
     * Sets all entries to the given value. If a tile does not exist, it is ignored.
     *
     * @param value The value to broadcast.
     */
    void set_all_existing(T value) {
        for (auto &tile : _tiles) {
            tile.second.set_all(value);
        }
    }

    /**
     * @brief Copy assignment.
     *
     * @param copy The tensor to copy.
     */
    TiledTensor<T, rank, TensorType> &operator=(TiledTensor<T, rank, TensorType> const &copy) {
        zero();
        _tile_sizes   = copy._tile_sizes;
        _tile_offsets = copy._tile_offsets;
        _dims         = copy._dims;
        _name         = copy._name;
        _size         = copy._size;
        _grid_size    = copy._grid_size;

        for (auto const &tile : copy._tiles) {
            add_tile(tile.first);
            _tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    /**
     * @brief Copy assignment from a different kind of tiled tensor.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledTensor<T, rank, TensorType>, TensorOther>)
    TiledTensor<T, rank, TensorType> &operator=(TensorOther const &copy) {
        zero();
        _tile_sizes   = copy._tile_sizes;
        _tile_offsets = copy._tile_offsets;
        _dims         = copy._dims;
        _name         = copy._name;
        _size         = copy._size;
        _grid_size    = copy._grid_size;

        for (auto const &tile : copy._tiles) {
            add_tile(tile.first);
            _tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    /**
     * @brief Set all occupied tensors with the given value.
     *
     * @param value The value to fill the tensors with.
     */
    TiledTensor &operator=(T value) {
        set_all(value);
        return *this;
    }

    /**
     * @brief Add a scalar to every tensor.
     *
     * @param value The value to add.
     */
    TiledTensor &operator+=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles.at(tile_index) += value;
        }
        return *this;
    }

    /**
     * @brief Subtract a scalar from every tensor.
     *
     * @param value The value to subtract.
     */
    TiledTensor &operator-=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles.at(tile_index) -= value;
        }
        return *this;
    }

    /**
     * @brief Multiply every tensor by a scalar.
     *
     * @param value The value to multiply.
     */
    TiledTensor &operator*=(T value) {
        if (value == T{0}) {
            zero();
            return *this;
        }
        for (auto &tile : _tiles) {
            tile.second *= value;
        }
        return *this;
    }

    /**
     * @brief Divide every tensor by a scalar.
     *
     * @param value The value to divide by.
     */
    TiledTensor &operator/=(T value) {
        for (auto &tile : _tiles) {
            tile.second /= value;
        }
        return *this;
    }

    /**
     * @brief Perform in-place addition between two tensors.
     *
     * @param other The tensor to add to this one.
     */
    TiledTensor &operator+=(TiledTensor const &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
        }

        for (auto const &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles.at(tile.first) += tile.second;
            } else {
                add_tile(tile.first);
                _tiles.at(tile.first) = TensorType(tile.second);
            }
        }

        return *this;
    }

    /**
     * @brief Perform in-place subtraction between two tensors.
     *
     * @param other The tensor to subtract from this one.
     */
    TiledTensor &operator-=(TiledTensor const &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
        }

        for (auto const &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles.at(tile.first) -= tile.second;
            } else {
                add_tile(tile.first);
                _tiles.at(tile.first) = TensorType(tile.second);
                _tiles.at(tile.first) *= -1;
            }
        }

        return *this;
    }

    /**
     * @brief Perform in-place multiplication between two tensors.
     *
     * @param other The tensor to multiply this one by.
     */
    TiledTensor &operator*=(TiledTensor const &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
        }

        for (auto const &tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second *= other._tiles.at(tile.first);
            } else {
                _tiles.erase(tile.first);
            }
        }

        return *this;
    }

    /**
     * @brief Perform in-place division between two tensors.
     *
     * If a block is zero in the divisor, then the corresponding block will also be zeroed in the output,
     * rather than setting it to NaN or infinity.
     *
     * @param other The tensor to division this one by.
     */
    TiledTensor &operator/=(TiledTensor const &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors do not have the same layouts.");
        }

        for (auto const &tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second /= other._tiles.at(tile.first);
            } else {
                tile.second /= T{0};
            }
        }

        return *this;
    }

    /**
     * Returns the tile offsets.
     */
    std::array<std::vector<int>, rank> tile_offsets() const { return _tile_offsets; }

    /**
     * Returns the tile offsets along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_offset(int i = 0) const { return _tile_offsets.at(i); }

    /**
     * Returns the tile sizes.
     */
    std::array<std::vector<int>, rank> tile_sizes() const { return _tile_sizes; }

    /**
     * Returns the tile sizes along a given dimension.
     *
     * @param i The axis to retrieve.
     *
     */
    std::vector<int> tile_size(int i = 0) const { return _tile_sizes.at(i); }

    /**
     * Get a reference to the tile map.
     */
    map_type const &tiles() const { return _tiles; }

    /**
     * Get a reference to the tile map.
     */
    map_type &tiles() { return _tiles; }

    /**
     * Get the name.
     */
    virtual std::string const &name() const { return _name; }

    /**
     * Sets the name.
     *
     * @param val The new name.
     */
    virtual void set_name(std::string const &val) { _name = val; }

    /**
     * Gets the size of the tensor.
     */
    size_t size() const { return _size; }

    /**
     * Gets the number of possible tiles, empty and filled.
     */
    size_t grid_size() const { return _grid_size; }

    /**
     * Gets the number of possible tiles along an axis, empty and filled.
     */
    size_t grid_size(int i) const { return _tile_sizes[i].size(); }

    /**
     * Gets the number of filled tiles.
     */
    size_t num_filled() const { return _tiles.size(); }

    /**
     * @brief Indicates whether the tensor sees all of the underlying elements, or could if all blocks were filled.
     */
    virtual bool full_view_of_underlying() const { return true; }

    /**
     * @brief Get the dimension along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const { return _dims.at(d); }

    /**
     * @brief Get the dimensions
     */
    Dim<rank> dims() const { return _dims; }

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a list of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <std::integral... Index>
        requires(sizeof...(Index) == rank)
    bool has_zero_size(Index... index) const {
        std::array<int, rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < rank; i++) {
            if (_tile_sizes[i].at(arr_index[i]) == 0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a container of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <typename Storage>
        requires(!std::integral<Storage>)
    bool has_zero_size(Storage const &index) const {
        for (int i = 0; i < rank; i++) {
            if (_tile_sizes[i].at(index[i]) == 0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Convert to the underlying tensor type.
     */
    operator TensorType() const {
        TensorType out(_dims);
        out.set_name(name());

        Stride<rank> tile_strides;

        size_t tiles = 1;

        for (ptrdiff_t i = rank - 1; i >= 0; i--) {
            tile_strides[i] = tiles;
            tiles *= grid_size(i);
        }

        for (size_t tile = 0; tile < tiles; tile++) {
            std::array<int64_t, rank> tile_index;

            sentinel_to_indices(tile, tile_strides, tile_index);

            if (!this->has_tile(tile_index) || this->has_zero_size(tile_index)) {
                continue;
            } else {
                // Calculate the view ranges.
                thread_local std::array<Range, rank> ranges;

                for (size_t i = 0; i < rank; i++) {
                    ranges[i] =
                        Range{this->tile_offset(i)[tile_index[i]], this->tile_offset(i)[tile_index[i]] + this->tile_size(i)[tile_index[i]]};
                }

                // Create the view.
                auto tile_view = std::apply(out, ranges);

                // Assign.
                tile_view = this->tile(tile_index);
            }
        }

        return out;
    }

  protected:
    /**
     * @property _tile_offsets
     *
     * @brief A list containing the positions along the axes that each tile starts.
     */
    /**
     * @property _tile_sizes
     *
     * @brief A list of the lengths of the tiles along the axes.
     */
    std::array<std::vector<int>, rank> _tile_offsets, _tile_sizes;

    /**
     * @property _tiles
     *
     * @brief The map containing the tiles.
     */
    map_type _tiles;

    /**
     * @property _dims
     *
     * @brief The overall dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @property _size
     *
     * @brief The total number of elements in the tensor, including the ignored zeros.
     */
    /**
     * @property _grid_size
     *
     * @brief The number of possible tile positions within the grid.
     */
    size_t _size, _grid_size;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed)"};

    /**
     * Add a tile using the underlying type's preferred method. Also gives it a name.
     *
     * @param pos The position in the grid to place the tile.
     */
    virtual void add_tile(std::array<int, rank> const &pos) = 0;

    template <typename TOther, size_t RankOther, typename TensorTypeOther>
    friend struct TiledTensorBase;
};

} // namespace tensor_base

/**
 * @struct TiledTensor
 *
 * @brief Holds a tile-wise sparse tensor.
 *
 * Tensors of this class have large blocks that are rigorously zero. These blocks need to line up on a grid.
 */
template <typename T, size_t Rank>
struct TiledTensor final : public tensor_base::TiledTensor<T, Rank, einsums::Tensor<T, Rank>>, tensor_base::CoreTensor {
  protected:
    /**
     * @brief Construct a new tile in the set of tiles at the given position.
     *
     * @param pos The position of the tile to create.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
        std::string tile_name = this->_name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = this->_tile_sizes[i].at(pos[i]);
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        this->_tiles.emplace(pos, dims);
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledTensor() = default;

    /**
     * @brief Create a new tiled tensor with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
    TiledTensor(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * @param other The tensor to copy.
     */
    TiledTensor(TiledTensor<T, Rank> const &other) : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(other) {}

    /**
     * @brief Copy cast constructor.
     *
     * This constructor copies data from a tiled tensor that is not just a TiledTensor. This includes
     * TiledTensorView, TiledDeviceTensor, and others.
     *
     * @param other The tensor to copy.
     */
    template <TiledTensorConcept OtherTensor>
        requires(SameUnderlyingAndRank<TiledTensor<T, Rank>, OtherTensor>)
    TiledTensor(OtherTensor const &other) : tensor_base::TiledTensor<T, Rank, Tensor<T, Rank>>(other) {}

    ~TiledTensor() = default;

    /**
     * @brief Copy assignment.
     *
     * This copies the data from the other tensor into this one. The other tensor should be
     * some sort of tiled tensor as well.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledTensor<T, Rank>, TensorOther>)
    TiledTensor<T, Rank> &operator=(TensorOther const &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (auto const &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }
};

/**
 * @struct TiledTensorView
 *
 * @brief Tensors of this class hold views of the tiles of a tiled tensor.
 *
 * Since views of block tensors are not guaranteed to be truly block diagonal, TiledTensorViews also hold
 * views of BlockTensors when the view is not hypersquare.
 */
template <typename T, size_t Rank>
struct TiledTensorView final : public tensor_base::TiledTensor<T, Rank, einsums::TensorView<T, Rank>>, tensor_base::CoreTensor {
  private:
    /**
     * @property _full_view_of_underlying
     *
     * @brief Indicates whether this view can see all of the elements in the base tensor.
     */
    bool _full_view_of_underlying{false};

    /**
     * @brief Tries to add a tile to the tensor, but it can't.
     *
     * As of the current version, modification of the underlying structure of a TiledTensorView is
     * not allowed. This is because currently, TiledTensorViews are kind of scuffed in that
     * their structure is desynchronized from the tensor they view. This may change in the future.
     */
    void add_tile(std::array<int, Rank> const &) override {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "Can't add a tile to a TiledTensorView!");
    }

  public:
    /**
     * @typedef underlying_type
     *
     * @brief Represents the kind of tensor this object views.
     */
    using underlying_type = TiledTensor<T, Rank>;

    TiledTensorView() = default;

    /**
     * @brief Create an empty view with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
    TiledTensorView(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, einsums::TensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * On exit, the structure of the copied tensor is copied. This means that the new tensor will view the same tensor
     * as the tensor being copied. Modifications to one will modify the other.
     *
     * @param other The tensor view to copy.
     */
    TiledTensorView(TiledTensorView<T, Rank> const &other) = default;

    ~TiledTensorView() = default;

    /**
     * @brief Checks to see if the view sees all of the data in the tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    /**
     * @brief Add a tile to the view.
     *
     * This does not add a tile to the viewed tensor, only to the view.
     */
    void insert_tile(std::array<int, Rank> pos, einsums::TensorView<T, Rank> &&view) {
        std::lock_guard lock(*this);
        this->_tiles.emplace(pos, view);
    }
};

#ifdef EINSUMS_COMPUTE_CODE
template <typename T, size_t Rank>
struct TiledDeviceTensor final : public tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>, tensor_base::DeviceTensorBase {
  private:
    /**
     * @property _mode
     *
     * @brief The storage mode to use for the subtensors.
     */
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    /**
     * @brief Create a new uninitialized tensor at the given position.
     *
     * @param pos The position to put the new tensor.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
        std::lock_guard(*this);
        std::string tile_name = this->_name + " - (";
        Dim<Rank>   dims{};

        for (int i = 0; i < Rank; i++) {
            tile_name += std::to_string(pos[i]);
            dims[i] = this->_tile_sizes[i].at(pos[i]);
            if (i != Rank - 1) {
                tile_name += ", ";
            }
        }
        tile_name += ")";

        this->_tiles.emplace(pos, einsums::DeviceTensor<T, Rank>(dims, _mode));
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledDeviceTensor() = default;

    /**
     * @brief Create a new tiled tensor with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, detail::HostToDeviceMode mode, Sizes &&...sizes)
        : _mode{mode}, tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Create a new tiled tensor with the given name and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * @param other The tensor to copy.
     */
    TiledDeviceTensor(TiledDeviceTensor<T, Rank> const &other) = default;

    /**
     * @brief Copy the data from another tiled tensor into this one.
     *
     * The parameter should be some type of tiled tensor or tiled tensor view.
     *
     * @param other The tensor to copy.
     */
    template <RankTiledTensor<Rank, T> OtherType>
    TiledDeviceTensor(OtherType const &other) : tensor_base::TiledTensor<T, Rank, einsums::DeviceTensor<T, Rank>>(other) {}

    ~TiledDeviceTensor() = default;

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i].at(coords[i]);
        }

        if (this->has_tile(coords)) {
            return subscript_tensor(this->tile(coords), array_ind);
        } else {
            return T{0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    HostDevReference<T> operator()(MultiIndex... index) {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets[i].at(coords[i]);
        }
        auto &out = this->tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * @brief Indexes into the tensor.
     *
     * If the appropriate tile does not exist, this will return zero.
     *
     * @param index The index for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * @brief Indexes into the tensor.
     *
     * If the appropriate tile does not exist, it will be created, zeroed, and the value returned.
     *
     * @param index The index to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) -> HostDevReference<T> {
        return std::apply(*this, index);
    }

    /**
     * @brief Copy assignment.
     *
     * Copies the data from another tiled tensor into this one.
     *
     * @param copy The tensor to copy.
     */
    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledDeviceTensor<T, Rank>, TensorOther>)
    TiledDeviceTensor<T, Rank> &operator=(TensorOther const &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (auto const &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    /**
     * @brief Cast to normal tensor.
     */
    operator einsums::Tensor<T, Rank>() const { return (einsums::Tensor<T, Rank>)(einsums::DeviceTensor<T, Rank>)*this; }
};

template <typename T, size_t Rank>
struct TiledDeviceTensorView final : public tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>, tensor_base::DeviceTensorBase {
  public:
    /**
     * @typedef underlying_type
     *
     * @brief The type of tensor this object views.
     */
    using underlying_type = TiledDeviceTensor<T, Rank>;

    TiledDeviceTensorView() = default;

    /**
     * @brief Construct a new tensor view with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, detail::HostToDeviceMode mode, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Construct a new tensor view with the given name, storage mode, and grid specification.
     *
     * The sizes should be collections of integers. There should either be only one collection, or there should be
     * as many collections as the rank of the tensor. If there are as many collections as the rank of the tensor,
     * then each collection will be used to split up the respective axis as a grid. If there is only one collection, then
     * it will be applied to all of the axes, making a square tensor whose diagonal tiles are square as well. Obviously,
     * if the tensor is only one-dimensional, these two behaviors are the same.
     *
     * @param name The name of the tensor.
     * @param mode The storage mode for the tensors.
     * @param sizes The grids for the axes. There must either only be one, or the number must be the same as the rank.
     */
    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, Sizes &&...sizes)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(name, std::forward<Sizes>(sizes)...) {}

    /**
     * @brief Copy constructor.
     *
     * Only the internal structure is truly copied. The views contained in this tensor view will still point to the
     * same place as the copied tensor, meaning updates to one will update the other.
     *
     * @param other The tensor view to copy.
     */
    TiledDeviceTensorView(TiledDeviceTensorView<T, Rank> const &other) = default;

    /**
     * @brief Create a copy of the core tiled tensor on the GPU.
     *
     * @param other The tensor to copy.
     */
    TiledDeviceTensorView(TiledTensor<T, Rank> &other)
        : tensor_base::TiledTensor<T, Rank, DeviceTensorView<T, Rank>>(other.name(), other.tile_sizes()) {
        for (auto &tile : other.tiles()) {
            this->_tiles.emplace(tile.first, tile.second);
        }
    }

    ~TiledDeviceTensorView() = default;

    /**
     * @brief Indicates whether the view can see all of the data of the original tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    /**
     * @brief Add a tile to the structure of this tensor.
     *
     * @param pos The position to put the view.
     * @param view The view to add to the tensor.
     */
    void insert_tile(std::array<int, Rank> pos, DeviceTensorView<T, Rank> &&view) {
        std::lock_guard lock(*this);
        this->_tiles.emplace(pos, view);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets.at(i).at(coords[i]);
        }

        if (this->has_tile(coords)) {
            return subscript_tensor(this->tile(coords), array_ind);
        } else {
            return T{0};
        }
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    HostDevReference<T> operator()(MultiIndex... index) {
        auto coords = this->tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += this->_dims[i];
            }
            array_ind[i] -= this->_tile_offsets.at(i).at(coords[i]);
        }
        auto &out = this->tile(coords);

        return subscript_tensor(out, array_ind);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, this will return zero.
     *
     * @param index The index to evaluate.
     * @return The value at the position.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * Indexes into the tensor. If the index points to a tile that is not initialized, it will create the tile and return a value for it.
     *
     * @param index The index to evaluate.
     * @return A reference to the position.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) -> HostDevReference<T> {
        return std::apply(*this, index);
    }

  private:
    /**
     * @property _full_view_of_underlying
     *
     * @brief Indicates whether the view can see all of the elements of the underlying tensor.
     */
    bool _full_view_of_underlying{false};

    /**
     * @brief Tries to add a tile to the tensor, but it can't.
     *
     * As of the current version, modification of the underlying structure of a TiledTensorView is
     * not allowed. This is because currently, TiledTensorViews are kind of scuffed in that
     * their structure is desynchronized from the tensor they view. This may change in the future.
     */
    void add_tile(std::array<int, Rank> const &pos) override {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "Can't add a tile to a TiledDeviceTensorView!");
    }
};

// EINSUMS_TENSOR_EXPORT(TiledDeviceTensor)
// EINSUMS_TENSOR_EXPORT(TiledDeviceTensorView)

#endif

EINSUMS_TENSOR_EXPORT(TiledTensor)
EINSUMS_TENSOR_EXPORT(TiledTensorView)

/**
 * Prints a TiledTensor to standard output.
 */
template <einsums::TiledTensorConcept TensorType>
void println(TensorType const &A, TensorPrintOptions options = {}) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    println("Name: {}", A.name());
    {
        print::Indent const indent{};
        println("Tiled Tensor");
        println("Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            println("Dims{{{}}}", oss.str().c_str());
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                println(A.tile(tile_index), options);
            }
        }
    }
}

/**
 * Prints a TiledTensor to a file pointer.
 */
template <einsums::TiledTensorConcept TensorType>
void fprintln(FILE *fp, TensorType const &A, TensorPrintOptions options = {}) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Tiled Tensor");
        fprintln(fp, "Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(fp, "Dims{{{}}}", oss.str().c_str());
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                fprintln(fp, A.tile(tile_index)), options;
            }
        }
    }
}

/**
 * Prints a TiledTensor to an output stream.
 */
template <einsums::TiledTensorConcept TensorType>
void fprintln(std::ostream &os, TensorType const &A, TensorPrintOptions options = {}) {
    using T               = typename TensorType::ValueType;
    constexpr size_t Rank = TensorType::Rank;
    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(os, "Tiled Tensor");
        fprintln(os, "Data Type: {}", type_name<T>());

        if constexpr (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(os, "Dims{{{}}}", oss.str().c_str());
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= A.tile_offset(i).size();
        }

        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % A.tile_offset(j).size();
                remaining /= A.tile_offset(j).size();
            }

            if (A.has_tile(tile_index)) {
                fprintln(os, A.tile(tile_index), options);
            }
        }
    }
}

} // namespace einsums