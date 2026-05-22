//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace einsums {

/**
 * @struct TiledRuntimeTensor
 *
 * @brief A runtime-rank, tile-wise sparse tensor.
 *
 * This is the runtime-rank analogue of @ref TiledTensor. Where @ref TiledTensor
 * carries its rank as a compile-time template parameter (and therefore keys its
 * tiles by ``std::array<int, Rank>``), TiledRuntimeTensor stores its rank at
 * runtime and keys tiles by ``std::vector<int>``. Each populated tile is a dense
 * @ref RuntimeTensor (which reuses ``detail::TensorImpl`` for its runtime-rank
 * dims/strides machinery); absent tiles are rigorously zero and are not stored.
 *
 * Like @ref TiledTensor this models a *general* tiled layout: a tile may sit
 * anywhere on the grid (not only the diagonal) and may be rectangular. The
 * block-diagonal, square-tile special case is what @ref BlockTensor captures;
 * the off-diagonal / rectangular cases (e.g. a non-totally-symmetric operator
 * coupling different symmetry sectors) require this general form.
 *
 * @par ComputeGraph integration (Tier A)
 * This type satisfies @ref CoreBasicTensorConcept so it can be captured into a
 * ComputeGraph for *lifecycle orchestration* (materialize / release / zero) and
 * passed around as an operand handle. Because a multi-tile tensor has no single
 * contiguous backing buffer, @ref data() returns ``nullptr`` for the multi-tile
 * case (and the single tile's pointer in the degenerate one-cell grid), and the
 * type advertises itself via @ref is_tiled_tensor() so the graph can flag the
 * handle (``TensorHandle::is_tiled``) and keep it out of dense buffer-level code
 * paths. Tiled *contraction inside the graph* is intentionally not provided here.
 *
 * @tparam T The data type stored in each tile.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
struct TiledRuntimeTensor : public tensor_base::CoreTensor,
                            public tensor_base::TiledTensorNoExtra,
                            public design_pats::Lockable<std::recursive_mutex> {
  public:
    /// The dense tensor type stored for each populated tile.
    using StoredType = RuntimeTensor<T>;

    /// The element type stored in the tensor.
    using ValueType = T;

    /// Map from grid coordinate (one int per axis) to the tile living there.
    // NOLINTNEXTLINE(readability-identifier-naming)
    using map_type = std::unordered_map<std::vector<int>, StoredType, einsums::hashes::ContainerHash<std::vector<int>>>;

    /// Compile-time rank sentinel; the real rank is only known at runtime via
    /// @ref rank(). See @ref einsums::dynamic_rank for what this triggers in
    /// compile-time rank checks.
    static constexpr int Rank = dynamic_rank;

    TiledRuntimeTensor() = default;

    /**
     * @brief Construct an empty tiled tensor from a tile grid.
     *
     * No tiles are populated yet; use @ref add_tile / @ref tile to declare the
     * sparsity pattern. The grid is given as one vector of tile sizes per axis,
     * so @c tile_sizes.size() determines the rank.
     *
     * @param name       The name of the tensor.
     * @param tile_sizes One vector per axis giving that axis's tile extents.
     * @param row_major  Memory layout for the *global* shape metadata and for
     *                   tiles created lazily. Defaults to the library setting.
     */
    TiledRuntimeTensor(std::string name, std::vector<std::vector<int>> tile_sizes, bool row_major)
        : _name(std::move(name)), _tile_sizes(std::move(tile_sizes)), _row_major(row_major) {
        rebuild_grid();
    }

    TiledRuntimeTensor(std::string name, std::vector<std::vector<int>> tile_sizes)
        : TiledRuntimeTensor(std::move(name), std::move(tile_sizes), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    TiledRuntimeTensor(TiledRuntimeTensor const &)            = default;
    TiledRuntimeTensor(TiledRuntimeTensor &&)                 = default;
    TiledRuntimeTensor &operator=(TiledRuntimeTensor const &) = default;
    TiledRuntimeTensor &operator=(TiledRuntimeTensor &&)      = default;
    virtual ~TiledRuntimeTensor()                             = default;

    // ── Shape / metadata ─────────────────────────────────────────────

    /// Runtime rank (number of grid axes).
    [[nodiscard]] size_t rank() const noexcept { return _dims.size(); }

    /// Global extent along axis @p d (sum of that axis's tile sizes). Supports
    /// Python-style negative indexing.
    [[nodiscard]] size_t dim(int d) const {
        int const r = static_cast<int>(_dims.size());
        if (d < 0) {
            d += r;
        }
        if (d < 0 || d >= r) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "TiledRuntimeTensor::dim: axis index out of range");
        }
        return _dims[d];
    }

    /// Global shape (one extent per axis).
    [[nodiscard]] std::vector<size_t> dims() const { return _dims; }

    /// Row/column-major stride of the *global* bounding shape along axis @p d.
    /// Advisory only — a tiled tensor has no single contiguous buffer; these
    /// describe the dense shape the tiles tile over.
    [[nodiscard]] size_t stride(int d) const {
        int const r = static_cast<int>(_strides.size());
        if (d < 0) {
            d += r;
        }
        if (d < 0 || d >= r) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "TiledRuntimeTensor::stride: axis index out of range");
        }
        return _strides[d];
    }

    /// Global strides (see @ref stride).
    [[nodiscard]] std::vector<size_t> strides() const { return _strides; }

    /// Number of elements in the dense global bounding shape (product of dims).
    [[nodiscard]] size_t size() const noexcept { return _size; }

    /// A tiled tensor is always considered a full view of itself.
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return true; }

    [[nodiscard]] std::string const &name() const noexcept { return _name; }
    void                             set_name(std::string const &new_name) { _name = new_name; }

    [[nodiscard]] bool is_row_major() const noexcept { return _row_major; }

    /**
     * @brief Raw data pointer.
     *
     * A multi-tile tensor has no single contiguous buffer, so this returns
     * ``nullptr``. In the degenerate single-cell grid (one possible tile) it
     * returns that tile's buffer if the tile exists, matching the dense case.
     * See the class note on ComputeGraph integration.
     */
    [[nodiscard]] T *data() noexcept {
        if (grid_size() == 1 && !_tiles.empty()) {
            return _tiles.begin()->second.data();
        }
        return nullptr;
    }
    [[nodiscard]] T const *data() const noexcept {
        if (grid_size() == 1 && !_tiles.empty()) {
            return _tiles.begin()->second.data();
        }
        return nullptr;
    }

    /// Advertises this type as tiled so ComputeGraph can flag its handle and
    /// keep it out of dense buffer-level passes. Probed via ``if constexpr``.
    [[nodiscard]] bool is_tiled_tensor() const noexcept { return true; }

    // ── Grid / tile management ───────────────────────────────────────

    /// Number of grid cells (product of the per-axis tile counts). This is the
    /// number of *possible* tiles, not the number populated.
    [[nodiscard]] size_t grid_size() const noexcept {
        size_t g = 1;
        for (auto const &axis : _tile_sizes) {
            g *= axis.size();
        }
        return g;
    }

    /// Number of populated (non-zero) tiles.
    [[nodiscard]] size_t num_filled_tiles() const noexcept { return _tiles.size(); }

    /// The per-axis tile sizes (the grid).
    [[nodiscard]] std::vector<std::vector<int>> const &tile_sizes() const noexcept { return _tile_sizes; }

    /// The per-axis tile offsets (running sums of the tile sizes).
    [[nodiscard]] std::vector<std::vector<int>> const &tile_offsets() const noexcept { return _tile_offsets; }

    /// Read-only access to the populated tiles.
    [[nodiscard]] map_type const &tiles() const noexcept { return _tiles; }

    /// Mutable access to the populated tiles.
    [[nodiscard]] map_type &tiles() noexcept { return _tiles; }

    /// True if a tile exists at the given grid coordinate.
    [[nodiscard]] bool has_tile(std::vector<int> const &coord) const { return _tiles.find(normalize(coord)) != _tiles.end(); }

    /**
     * @brief Declare a tile at @p coord (deferred — no storage allocated).
     *
     * Creates a shell @ref RuntimeTensor with the right dims for that grid cell
     * but no backing data. The tile is allocated later by @ref materialize (or
     * by the ComputeGraph MaterializationPass). No-op if the tile already
     * exists.
     *
     * @param coord The grid coordinate, one index per axis.
     */
    void add_tile(std::vector<int> const &coord) {
        std::vector<int> key = normalize(coord);
        if (_tiles.find(key) != _tiles.end()) {
            return;
        }
        std::vector<size_t> tile_dims(key.size());
        std::string         tile_name = _name + " - (";
        for (size_t i = 0; i < key.size(); ++i) {
            tile_dims[i] = static_cast<size_t>(_tile_sizes[i].at(key[i]));
            tile_name += std::to_string(key[i]);
            if (i + 1 != key.size()) {
                tile_name += ", ";
            }
        }
        tile_name += ")";
        auto [it, inserted] = _tiles.emplace(key, StoredType(typename StoredType::DeferredAlloc{}, std::move(tile_name), tile_dims));
        // RuntimeTensor's copy ctor (used by emplace) repoints _impl at its own
        // (empty) data buffer, dropping the deferred sentinel pointer the
        // DeferredAlloc ctor installed. With a null pointer TensorImpl::dim()
        // guards to 0, so the tile would report empty dims until materialized.
        // Restore the sentinel so the deferred tile keeps its shape; dims were
        // preserved by the copy, only the pointer needs re-establishing.
        it->second.set_data(reinterpret_cast<T *>(0x1));
    }

    /**
     * @brief Access the tile at @p coord, creating it (deferred) if absent.
     *
     * @param coord The grid coordinate, one index per axis.
     * @return Reference to the tile.
     */
    StoredType &tile(std::vector<int> const &coord) {
        std::vector<int> key = normalize(coord);
        if (_tiles.find(key) == _tiles.end()) {
            add_tile(key);
        }
        return _tiles.at(key);
    }

    /// Access an existing tile; throws if it is not populated.
    [[nodiscard]] StoredType const &tile(std::vector<int> const &coord) const { return _tiles.at(normalize(coord)); }

    // ── Deferred-allocation lifecycle ────────────────────────────────
    //
    // Mirrors RuntimeTensor's API so ComputeGraph's make_handle SFINAE probes
    // pick up the same capabilities. These operate tile-by-tile.

    /// Allocate backing storage for every populated tile. Idempotent.
    void materialize() {
        for (auto &[coord, t] : _tiles) {
            t.materialize();
        }
    }

    /// True iff every populated tile is materialized (vacuously true when there
    /// are no tiles). Drives the handle's initial AllocState in make_handle.
    [[nodiscard]] bool is_materialized() const {
        for (auto const &[coord, t] : _tiles) {
            if (!t.is_materialized()) {
                return false;
            }
        }
        return true;
    }

    /// Release backing storage for every populated tile, returning to the
    /// deferred state. The sparsity pattern (which tiles exist) is preserved.
    void release() {
        for (auto &[coord, t] : _tiles) {
            t.release();
        }
    }

    /// Zero every populated tile.
    void zero() {
        for (auto &[coord, t] : _tiles) {
            t.zero();
        }
    }

    /// Set every element of every populated tile to @p val.
    void set_all(T val) {
        for (auto &[coord, t] : _tiles) {
            t.set_all(val);
        }
    }

    /// Lazily-created liveness token for ComputeGraph's runtime validator (see
    /// TensorHandle::make_handle). Mirrors RuntimeTensor::liveness_token.
    [[nodiscard]] std::weak_ptr<void> liveness_token() const {
        if (!_life_token) {
            _life_token = std::make_shared<char>();
        }
        return _life_token;
    }

  protected:
    /// Normalize a grid coordinate: validate length, resolve negative indices,
    /// bounds-check against the grid.
    [[nodiscard]] std::vector<int> normalize(std::vector<int> const &coord) const {
        if (coord.size() != _tile_sizes.size()) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "TiledRuntimeTensor: tile coordinate rank does not match tensor rank");
        }
        std::vector<int> out(coord);
        for (size_t i = 0; i < out.size(); ++i) {
            int const n = static_cast<int>(_tile_sizes[i].size());
            if (out[i] < 0) {
                out[i] += n;
            }
            if (out[i] < 0 || out[i] >= n) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "TiledRuntimeTensor: tile coordinate out of range");
            }
        }
        return out;
    }

    /// Recompute global dims, offsets, strides and size from _tile_sizes.
    void rebuild_grid() {
        size_t const r = _tile_sizes.size();
        _tile_offsets.assign(r, {});
        _dims.assign(r, 0);
        _size = (r == 0) ? 0 : 1;
        for (size_t i = 0; i < r; ++i) {
            _tile_offsets[i].reserve(_tile_sizes[i].size());
            int sum = 0;
            for (int s : _tile_sizes[i]) {
                _tile_offsets[i].push_back(sum);
                sum += s;
            }
            _dims[i] = static_cast<size_t>(sum);
            _size *= _dims[i];
        }
        compute_strides();
    }

    /// Row/column-major strides over the global bounding shape.
    void compute_strides() {
        size_t const r = _dims.size();
        _strides.assign(r, 0);
        if (r == 0) {
            return;
        }
        if (_row_major) {
            _strides[r - 1] = 1;
            for (int i = static_cast<int>(r) - 2; i >= 0; --i) {
                _strides[i] = _strides[i + 1] * _dims[i + 1];
            }
        } else {
            _strides[0] = 1;
            for (size_t i = 1; i < r; ++i) {
                _strides[i] = _strides[i - 1] * _dims[i - 1];
            }
        }
    }

    std::string                   _name{"(unnamed tiled)"};
    std::vector<std::vector<int>> _tile_sizes{};
    std::vector<std::vector<int>> _tile_offsets{};
    std::vector<size_t>           _dims{};
    std::vector<size_t>           _strides{};
    size_t                        _size{0};
    bool                          _row_major{true};
    map_type                      _tiles{};

    mutable std::shared_ptr<void> _life_token;
};

} // namespace einsums
