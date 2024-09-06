//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/tensor/detail/hash.hpp>
#include <einsums/tensor/tensor.hpp>

namespace einsums {
namespace tensor_base {

/**
 * @struct TiledTensorBase
 *
 * Represents a tiled tensor. Is a lockable type.
 *
 * @tparam TensorType The underlying storage type.
 * @tparam T The type of data being stored.
 * @tparam Rank The tensor rank.
 */
template <typename T, size_t Rank, typename TensorType>
struct tiled_tensor : public virtual collected_tensor<T, Rank, TensorType>,
                      virtual tiled_tensor_no_extra,
                      virtual lockable_tensor,
                      virtual algebra_optimized_tensor {
  public:
    using map_type = typename std::unordered_map<std::array<int, Rank>, TensorType, einsums::hashes::container_hash<std::array<int, Rank>>>;

  protected:
    std::array<std::vector<int>, Rank> _tile_offsets, _tile_sizes;

    map_type _tiles;

    einsums::dim<Rank> _dims;

    size_t _size, _grid_size;

    std::string _name{"(unnamed)"};

    /**
     * Add a tile using the underlying type's preferred method. Also gives it a name.
     */
    virtual void add_tile(std::array<int, Rank> pos) = 0;

    template <typename TOther, size_t RankOther, typename TensorTypeOther>
    friend struct tiled_tensor;

  public:
    /**
     * Create a new empty tiled tensor.
     */
    tiled_tensor() : _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{}, _grid_size{0} {}

    /**
     * Create a new empty tiled tensor with the given grid. If only one grid is given, the grid is applied to all dimensions.
     * Otherwise, the number of grids must match the rank.
     *
     * @param name The name of the tensor.
     * @param sizes The grids to apply.
     */
    template <typename... Sizes>
        requires(!std::is_same_v<std::array<std::vector<int>, Rank>, Sizes> && ... && true)
    tiled_tensor(std::string name, Sizes... sizes) : _name(name), _tile_offsets(), _tile_sizes(), _tiles(), _size(0), _dims{} {
        static_assert(sizeof...(Sizes) == Rank || sizeof...(Sizes) == 1);

        _size = 1;
        if constexpr (sizeof...(Sizes) == Rank &&
                      !std::is_same_v<std::array<std::vector<int>, Rank>, decltype(std::get<0>(std::tuple(sizes...)))>) {
            _tile_sizes = std::array<std::vector<int>, Rank>{std::vector<int>(sizes.begin(), sizes.end())...};
        } else {
            for (int i = 0; i < Rank; i++) {
                _tile_sizes[i] = std::vector<int>(sizes.begin()..., sizes.end()...);
            }
        }
        for (int i = 0; i < Rank; i++) {
            _tile_offsets[i] = std::vector<int>();
            int sum          = 0;
            for (int j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_offsets[i].push_back(sum);
                sum += _tile_sizes[i].at(j);
            }
            _dims[i] = sum;
            _size *= sum;
        }

        _grid_size = 1;

        for (int i = 0; i < Rank; i++) {
            _grid_size *= _tile_offsets[i].size();
        }
    }

    tiled_tensor(std::string name, const std::array<std::vector<int>, Rank> &sizes)
        : _name(name), _tile_offsets(), _tile_sizes(sizes), _tiles(), _size(0), _dims{} {
        _size = 1;
        for (int i = 0; i < Rank; i++) {
            _tile_offsets[i] = std::vector<int>();
            int sum          = 0;
            for (int j = 0; j < _tile_sizes[i].size(); j++) {
                _tile_offsets[i].push_back(sum);
                sum += _tile_sizes[i].at(j);
            }
            _dims[i] = sum;
            _size *= sum;
        }

        _grid_size = 1;

        for (int i = 0; i < Rank; i++) {
            _grid_size *= _tile_offsets[i].size();
        }
    }

    /**
     * Copy a tiled tensor.
     *
     * @param other The tensor to be copied.
     */
    tiled_tensor(const tiled_tensor<T, Rank, TensorType> &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims}, _grid_size{other._grid_size} {
        for (auto &pair : other._tiles) {
            _tiles[pair.first] = TensorType(pair.second);
        }
    }

    /**
     * Copy a tiled tensor from one tensor type to another.
     *
     * @param other The tensor to be copied and converted.
     */
    template <TRTensorConcept<Rank, T> OtherTensor>
    tiled_tensor(const tiled_tensor<T, Rank, OtherTensor> &other)
        : _tile_offsets(other._tile_offsets), _tile_sizes(other._tile_sizes), _name(other._name), _size(other._size), _tiles(),
          _dims{other._dims}, _grid_size{other._grid_size} {
        for (auto &pair : other._tiles) {
            _tiles[pair.first] = TensorType(pair.second);
        }
    }

    ~tiled_tensor() = default;

    /**
     * Returns the tile with given coordinates. If the tile is not filled, it will be created.
     *
     * @param index The index of the tile.
     * @return The tile at the given index.
     */
    template <std::integral... MultiIndex>
        requires(sizeof...(MultiIndex) == Rank)
    TensorType &tile(MultiIndex... index) {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            einsums::dim<Rank> dims{};

            for (int i = 0; i < Rank; i++) {
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
        requires(sizeof...(MultiIndex) == Rank)
    const TensorType &tile(MultiIndex... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < Rank; i++) {
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
        std::array<int, Rank> arr_index;

        for (int i = 0; i < Rank; i++) {
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
        std::array<int, Rank> arr_index{index};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _tile_sizes[i].size();
            }

            assert(arr_index[i] < _tile_sizes[i].size() && arr_index[i] >= 0);
        }

        if (!has_tile(arr_index)) {
            einsums::dim<Rank> dims{};

            for (int i = 0; i < Rank; i++) {
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
        requires(sizeof...(MultiIndex) == Rank)
    bool has_tile(MultiIndex... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < Rank; i++) {
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
        requires(sizeof...(MultiIndex) == Rank)
    std::array<int, Rank> tile_of(MultiIndex... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};
        std::array<int, Rank> out{0};

        for (int i = 0; i < Rank; i++) {
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Index not in the tensor!");
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
        std::array<int, Rank> arr_index;

        for (int i = 0; i < Rank; i++) {
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
    std::array<int, Rank> tile_of(Storage index) const {
        std::array<int, Rank> arr_index;
        std::array<int, Rank> out{0};

        for (int i = 0; i < Rank; i++) {
            arr_index[i] = static_cast<int>(index[i]);
            if (arr_index[i] < 0) {
                arr_index[i] += _dims[i];
            }

            if (arr_index[i] < 0 || arr_index[i] >= _dims[i]) {
                EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Index not in the tensor!");
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
        requires(sizeof...(MultiIndex) == Rank)
    T operator()(MultiIndex... index) const {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }

        if (has_tile(coords)) {
            return std::apply(tile(coords), array_ind);
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
        requires(sizeof...(MultiIndex) == Rank)
    T &operator()(MultiIndex... index) {
        auto coords = tile_of(index...);

        auto array_ind = std::array<int, Rank>{static_cast<int>(index)...};

        // Find the index in the tile.
        for (int i = 0; i < Rank; i++) {
            if (array_ind[i] < 0) {
                array_ind[i] += _dims[i];
            }
            array_ind[i] -= _tile_offsets[i].at(coords[i]);
        }
        auto &out = tile(coords);

        return std::apply(out, array_ind);
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
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
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

    tiled_tensor<T, Rank, TensorType> &operator=(const tiled_tensor<T, Rank, TensorType> &copy) {
        zero();
        _tile_sizes   = copy._tile_sizes;
        _tile_offsets = copy._tile_offsets;
        _dims         = copy._dims;
        _name         = copy._name;
        _size         = copy._size;
        _grid_size    = copy._grid_size;

        for (const auto &tile : copy._tiles) {
            add_tile(tile.first);
            _tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<tiled_tensor<T, Rank, TensorType>, TensorOther>)
    tiled_tensor<T, Rank, TensorType> &operator=(const TensorOther &copy) {
        zero();
        _tile_sizes   = copy._tile_sizes;
        _tile_offsets = copy._tile_offsets;
        _dims         = copy._dims;
        _name         = copy._name;
        _size         = copy._size;
        _grid_size    = copy._grid_size;

        for (const auto &tile : copy._tiles) {
            add_tile(tile.first);
            _tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    tiled_tensor &operator=(T value) {
        set_all(value);
        return *this;
    }

    tiled_tensor &operator+=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles.at(tile_index) += value;
        }
        return *this;
    }

    tiled_tensor &operator-=(T value) {
        if (value == T{0}) {
            return *this;
        }

        // Find the number of tiles.
        long num_tiles = 1;
        for (int i = 0; i < Rank; i++) {
            num_tiles *= _tile_offsets[i].size();
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (long i = 0; i < num_tiles; i++) {
            std::array<int, Rank> tile_index{};
            long                  remaining = i;

            // Turn sentinel into an index.
            for (int j = 0; j < Rank; j++) {
                tile_index[j] = remaining % _tile_offsets[j].size();
                remaining /= _tile_offsets[j].size();
            }

            // Set the tile index.
            _tiles.at(tile_index) -= value;
        }
        return *this;
    }

    tiled_tensor &operator*=(T value) {
        if (value == T{0}) {
            zero();
            return *this;
        }
        for (auto &tile : _tiles) {
            tile.second *= value;
        }
        return *this;
    }

    tiled_tensor &operator/=(T value) {
        for (auto &tile : _tiles) {
            tile.second /= value;
        }
        return *this;
    }

    tiled_tensor &operator+=(const tiled_tensor &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : other._tiles) {
            if (has_tile(tile.first)) {
                _tiles.at(tile.first) += tile.second;
            } else {
                add_tile(tile.first);
                _tiles.at(tile.first) = TensorType(tile.second);
            }
        }

        return *this;
    }

    tiled_tensor &operator-=(const tiled_tensor &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : other._tiles) {
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

    tiled_tensor &operator*=(const tiled_tensor &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : _tiles) {
            if (other.has_tile(tile.first)) {
                tile.second *= other._tiles.at(tile.first);
            } else {
                _tiles.erase(tile.first);
            }
        }

        return *this;
    }

    tiled_tensor &operator/=(const tiled_tensor &other) {
        if (_tile_sizes != other._tile_sizes) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Tiled tensors do not have the same layouts.");
        }

        for (const auto &tile : _tiles) {
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
    std::array<std::vector<int>, Rank> tile_offsets() const { return _tile_offsets; }

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
    std::array<std::vector<int>, Rank> tile_sizes() const { return _tile_sizes; }

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
    const map_type &tiles() const { return _tiles; }

    /**
     * Get a reference to the tile map.
     */
    map_type &tiles() { return _tiles; }

    /**
     * Get the name.
     */
    virtual const std::string &name() const override { return _name; }

    /**
     * Sets the name.
     *
     * @param val The new name.
     */
    virtual void set_name(const std::string &val) override { _name = val; }

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

    virtual bool full_view_of_underlying() const override { return true; }

    /**
     * @brief Get the dimensions.
     */
    virtual size_t dim(int d) const override { return _dims.at(d); }

    virtual einsums::dim<Rank> dims() const override { return _dims; }

    /**
     * Check to see if the given tile has zero size.
     *
     * @param index The index of the tile, as a list of integers.
     * @return True if the tile has at least one dimension of zero, leading to a size of zero, or false if there are no zero dimensions.
     */
    template <std::integral... Index>
        requires(sizeof...(Index) == Rank)
    bool has_zero_size(Index... index) const {
        std::array<int, Rank> arr_index{static_cast<int>(index)...};

        for (int i = 0; i < Rank; i++) {
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
    bool has_zero_size(const Storage &index) const {
        for (int i = 0; i < Rank; i++) {
            if (_tile_sizes[i].at(index[i]) == 0) {
                return true;
            }
        }

        return false;
    }

    /**
     * Convert.
     */
    operator TensorType() const {
        TensorType out(_dims);
        out.set_name(name());
        out.zero();

        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            std::apply(out, target_combination) = std::apply(*this, target_combination);
        }

        return out;
    }
};

} // namespace tensor_base

template <typename T, size_t Rank>
struct tiled_tensor final : public virtual tensor_base::tiled_tensor<T, Rank, tensor<T, Rank>>, virtual tensor_base::core_tensor {
  protected:
    void add_tile(std::array<int, Rank> pos) override {
        std::string        tile_name = this->_name + " - (";
        einsums::dim<Rank> dims{};

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
    tiled_tensor() = default;

    template <typename... Sizes>
    tiled_tensor(std::string name, Sizes... sizes) : tensor_base::tiled_tensor<T, Rank, tensor<T, Rank>>(name, sizes...) {}

    tiled_tensor(std::string name, std::initializer_list<int> init) : tensor_base::tiled_tensor<T, Rank, tensor<T, Rank>>(name, init) {}

    tiled_tensor(const tiled_tensor<T, Rank> &other) : tensor_base::tiled_tensor<T, Rank, tensor<T, Rank>>(other) {}

    template <TiledTensorConcept OtherTensor>
        requires(SameUnderlyingAndRank<tiled_tensor<T, Rank>, OtherTensor>)
    tiled_tensor(const OtherTensor &other) : tensor_base::tiled_tensor<T, Rank, tensor<T, Rank>>(other) {}

    ~tiled_tensor() = default;

    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<tiled_tensor<T, Rank>, TensorOther>)
    tiled_tensor<T, Rank> &operator=(const TensorOther &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (const auto &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    size_t dim(int d) const override { return this->_dims[d]; }
};

template <typename T, size_t Rank>
struct tiled_tensor_view final : public virtual tensor_base::tiled_tensor<T, Rank, tensor_view<T, Rank>>,
                                 virtual tensor_base::tensor_view<T, Rank, tiled_tensor<T, Rank>>,
                                 virtual tensor_base::core_tensor {
  private:
    bool _full_view_of_underlying{false};

    void add_tile(std::array<int, Rank> pos) override {
        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Can't add a tile to a TiledTensorView!");
    }

  public:
    tiled_tensor_view() = default;

    template <typename... Sizes>
    tiled_tensor_view(std::string name, Sizes... sizes) : tensor_base::tiled_tensor<T, Rank, tensor_view<T, Rank>>(name, sizes...) {}

    tiled_tensor_view(const tiled_tensor_view<T, Rank> &other) = default;

    ~tiled_tensor_view() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, tensor_view<T, Rank> &view) {
        std::lock_guard(*this);
        this->_tiles[pos] = view;
    }

    size_t dim(int d) const override { return this->_dims[d]; }
};

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct TiledDeviceTensor final : public virtual tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>,
                                 virtual tensor_props::DeviceTensorBase {
  private:
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override {
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

        this->_tiles.emplace(pos, DeviceTensor<T, Rank>(dims, _mode));
        this->_tiles.at(pos).set_name(tile_name);
        this->_tiles.at(pos).zero();
    }

  public:
    TiledDeviceTensor() = default;

    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, detail::HostToDeviceMode mode, Sizes... sizes)
        : _mode{mode}, tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>(name, sizes...) {}

    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensor(std::string name, Sizes... sizes) : tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>(name, sizes...) {}

    TiledDeviceTensor(const TiledDeviceTensor<T, Rank> &other) = default;

    template <RankTiledTensor<Rank, T> OtherType>
    TiledDeviceTensor(const OtherType &other) : tensor_props::TiledTensorBase<T, Rank, DeviceTensor<T, Rank>>(other) {}

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
            return std::apply(this->tile(coords), array_ind);
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

        return std::apply(out, array_ind);
    }

    template <TiledTensorConcept TensorOther>
        requires(SameUnderlyingAndRank<TiledDeviceTensor<T, Rank>, TensorOther>)
    TiledDeviceTensor<T, Rank> &operator=(const TensorOther &copy) {
        this->zero();
        this->_tile_sizes   = copy.tile_sizes();
        this->_tile_offsets = copy.tile_offsets();
        this->_dims         = copy.dims();
        this->_name         = copy.name();
        this->_size         = copy.size();
        this->_grid_size    = copy.grid_size();

        for (const auto &tile : copy.tiles()) {
            add_tile(tile.first);
            this->_tiles.at(tile.first) = tile.second;
        }

        return *this;
    }

    operator Tensor<T, Rank>() const { return (Tensor<T, Rank>)(DeviceTensor<T, Rank>)*this; }

    size_t dim(int d) const override { return this->_dims[d]; }
};

template <typename T, size_t Rank>
struct TiledDeviceTensorView final : public virtual tensor_props::TiledTensorBase<T, Rank, DeviceTensorView<T, Rank>>,
                                     virtual tensor_props::DeviceTensorBase,
                                     virtual tensor_props::TensorViewBase<T, Rank, TiledDeviceTensor<T, Rank>> {
  private:
    bool                     _full_view_of_underlying{false};
    detail::HostToDeviceMode _mode{detail::DEV_ONLY};

    void add_tile(std::array<int, Rank> pos) override { throw EINSUMSEXCEPTION("Can't add a tile to a TiledDeviceTensorView!"); }

  public:
    TiledDeviceTensorView() = default;

    template <typename... Sizes>
        requires(!(std::is_same_v<Sizes, detail::HostToDeviceMode> || ...))
    TiledDeviceTensorView(std::string name, detail::HostToDeviceMode mode, Sizes... sizes)
        : _mode{mode}, tensor_props::TiledTensorBase<T, Rank, DeviceTensorView<T, Rank>>(name, sizes...) {}

    TiledDeviceTensorView(const TiledDeviceTensorView<T, Rank> &other) = default;

    TiledDeviceTensorView(TiledTensor<T, Rank> &other)
        : tensor_props::TiledTensorBase<T, Rank, DeviceTensorView<T, Rank>>(other.name(), other.tile_sizes()) {
        for (auto &tile : other.tiles()) {
            this->_tiles.emplace(tile.first, tile.second);
        }
    }

    ~TiledDeviceTensorView() = default;

    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view_of_underlying; }

    void insert_tile(std::array<int, Rank> pos, DeviceTensorView<T, Rank> &view) { this->_tiles[pos] = view; }

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
            return std::apply(this->tile(coords), array_ind);
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

        return std::apply(out, array_ind);
    }

    size_t dim(int d) const override { return this->_dims[d]; }
};

#endif

template <FileOrOstream Output, TiledTensorConcept TensorType>
void fprintln(Output fp, const TensorType &A, tensor_print_options options = {}) {
    using T               = typename TensorType::data_type;
    constexpr size_t Rank = TensorType::rank;
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Tiled Tensor");
        fprintln(fp, "Data Type: {}", util::type_name<T>());

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

template <einsums::TiledTensorConcept TensorType>
void println(const TensorType &A, tensor_print_options options = {}) {
    fprintln(stdout, A, options);
}

} // namespace einsums