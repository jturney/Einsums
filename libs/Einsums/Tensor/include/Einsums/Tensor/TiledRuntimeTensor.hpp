//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <complex>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace einsums {

template <typename T>
struct TiledRuntimeTensorView;

/**
 * @struct IndexSpace
 *
 * @brief A per-irrep selection of indices along one axis of a tiled tensor.
 *
 * Each axis of a @ref TiledRuntimeTensor is partitioned into per-irrep blocks.
 * An IndexSpace picks a half-open ``[start, stop)`` sub-range out of each
 * irrep's block, so a single IndexSpace can describe e.g. "the occupied
 * orbitals" (``[0, nocc_h)`` in every irrep ``h``) or "the virtuals"
 * (``[nocc_h, nmo_h)``). Slicing a tiled tensor with one IndexSpace per axis
 * (``A.view({o, v, o, v})``) yields a @ref TiledRuntimeTensorView.
 *
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_RENAME("IndexSpace") IndexSpace {
    /// Per-irrep half-open ``[start, stop)`` ranges; one entry per irrep on the
    /// axis this space applies to.
    std::vector<std::pair<int, int>> ranges;

    IndexSpace() = default;

    /// Build from explicit per-irrep ``(start, stop)`` ranges.
    APIARY_EXPOSE explicit IndexSpace(std::vector<std::pair<int, int>> r) : ranges(std::move(r)) {}

    /// Number of irreps this space covers.
    [[nodiscard]] APIARY_EXPOSE int nirrep() const { return static_cast<int>(ranges.size()); }

    /// First index selected in irrep @p h.
    [[nodiscard]] APIARY_EXPOSE int start(int h) const { return ranges.at(static_cast<size_t>(h)).first; }

    /// One past the last index selected in irrep @p h.
    [[nodiscard]] APIARY_EXPOSE int stop(int h) const { return ranges.at(static_cast<size_t>(h)).second; }

    /// Number of indices selected in irrep @p h.
    [[nodiscard]] APIARY_EXPOSE int length(int h) const {
        auto const &p = ranges.at(static_cast<size_t>(h));
        return p.second - p.first;
    }
};

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
struct
    // clang-format off
APIARY_EXPOSE
// No buffer protocol: a multi-tile tensor has no single contiguous buffer.
// Python fills/reads it per tile via tile_view(), which yields a numpy-backed
// RuntimeTensorView over one tile.
APIARY_INSTANTIATE_AS("TiledRuntimeTensorF", TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorD", TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorC", TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorZ", TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    TiledRuntimeTensor : public tensor_base::CoreTensor,
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

    APIARY_EXPOSE TiledRuntimeTensor(std::string name, std::vector<std::vector<int>> tile_sizes)
        : TiledRuntimeTensor(std::move(name), std::move(tile_sizes), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    TiledRuntimeTensor(TiledRuntimeTensor const &)            = default;
    TiledRuntimeTensor(TiledRuntimeTensor &&)                 = default;
    TiledRuntimeTensor &operator=(TiledRuntimeTensor const &) = default;
    TiledRuntimeTensor &operator=(TiledRuntimeTensor &&)      = default;
    virtual ~TiledRuntimeTensor()                             = default;

    // ── Shape / metadata ─────────────────────────────────────────────

    /// Runtime rank (number of grid axes).
    [[nodiscard]] APIARY_EXPOSE size_t rank() const noexcept { return _dims.size(); }

    /// Global extent along axis @p d (sum of that axis's tile sizes). Supports
    /// Python-style negative indexing.
    [[nodiscard]] APIARY_EXPOSE size_t dim(int d) const {
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
    [[nodiscard]] APIARY_EXPOSE std::vector<size_t> dims() const { return _dims; }

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

    [[nodiscard]] APIARY_GETTER("name") std::string const &name() const noexcept { return _name; }
    APIARY_SETTER("name") void set_name(std::string const &new_name) { _name = new_name; }

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
    [[nodiscard]] APIARY_EXPOSE size_t grid_size() const noexcept {
        size_t g = 1;
        for (auto const &axis : _tile_sizes) {
            g *= axis.size();
        }
        return g;
    }

    /// Number of populated (non-zero) tiles.
    [[nodiscard]] APIARY_EXPOSE size_t num_filled_tiles() const noexcept { return _tiles.size(); }

    /// The per-axis tile sizes (the grid).
    [[nodiscard]] APIARY_EXPOSE std::vector<std::vector<int>> const &tile_sizes() const noexcept { return _tile_sizes; }

    /// The per-axis tile offsets (running sums of the tile sizes).
    [[nodiscard]] APIARY_EXPOSE std::vector<std::vector<int>> const &tile_offsets() const noexcept { return _tile_offsets; }

    /// Read-only access to the populated tiles.
    [[nodiscard]] map_type const &tiles() const noexcept { return _tiles; }

    /// Mutable access to the populated tiles.
    [[nodiscard]] map_type &tiles() noexcept { return _tiles; }

    /// True if a tile exists at the given grid coordinate.
    [[nodiscard]] APIARY_EXPOSE bool has_tile(std::vector<int> const &coord) const { return _tiles.find(normalize(coord)) != _tiles.end(); }

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
    APIARY_EXPOSE void add_tile(std::vector<int> const &coord) {
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

    /// EXPERIMENTAL (zero-copy bridge). Add a tile at @p coord that ALIASES the
    /// external buffer @p ptr instead of owning a copy. The tile takes the grid
    /// dims for @p coord and the given layout (row_major matches psi4's
    /// contiguous irrep blocks). @p ptr must outlive this tensor. This lets a
    /// psi4 Matrix/Vector be wrapped as a tiled Einsums tensor with no copy —
    /// the precursor to making such tensors the storage backend.
    void add_alias_tile(std::vector<int> const &coord, T *ptr, bool row_major) {
        std::vector<int> const key = normalize(coord);
        add_tile(key);                           // create a deferred tile (fixes its dims)
        _tiles.at(key).alias_to(ptr, row_major); // re-point it at the external buffer
    }

    /// Python-facing tile accessor: materialize the tile at @p coord (creating
    /// it if absent) and return a numpy-backed view of it. KEEP_ALIVE(0,1) ties
    /// this tensor's lifetime to the returned view, so the tile's storage
    /// outlives any numpy array Python builds from it.
    APIARY_EXPOSE APIARY_KEEP_ALIVE(0, 1) RuntimeTensorView<T> tile_view(std::vector<int> const &coord) {
        auto &t = tile(coord);
        t.materialize();
        return RuntimeTensorView<T>(t);
    }

    /// Create a non-owning @ref TiledRuntimeTensorView restricted to one
    /// @ref IndexSpace per axis (e.g. ``A.view({o, v, o, v})``). The view keeps
    /// this tensor alive; defined out-of-line below TiledRuntimeTensorView.
    APIARY_EXPOSE APIARY_KEEP_ALIVE(0, 1) TiledRuntimeTensorView<T> view(std::vector<IndexSpace> const &spaces);

    // ── Deferred-allocation lifecycle ────────────────────────────────
    //
    // Mirrors RuntimeTensor's API so ComputeGraph's make_handle SFINAE probes
    // pick up the same capabilities. These operate tile-by-tile.

    /// Allocate backing storage for every populated tile. Idempotent.
    APIARY_EXPOSE void materialize() {
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
    APIARY_EXPOSE void release() {
        for (auto &[coord, t] : _tiles) {
            t.release();
        }
    }

    /// Zero every populated tile.
    APIARY_EXPOSE void zero() {
        for (auto &[coord, t] : _tiles) {
            t.zero();
        }
    }

    /// Set every element of every populated tile to @p val.
    APIARY_EXPOSE void set_all(T val) {
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

// Explicit instantiations live in TensorDefs.cpp (built into libEinsums with
// default visibility via EINSUMS_EXPORT). The pybind codegen TU is compiled
// with hidden visibility, so it must NOT re-instantiate these — extern template
// makes it link against libEinsums's copies instead.
#if !defined(EINSUMS_WINDOWS)
extern template class EINSUMS_EXPORT TiledRuntimeTensor<float>;
extern template class EINSUMS_EXPORT TiledRuntimeTensor<double>;
extern template class EINSUMS_EXPORT TiledRuntimeTensor<std::complex<float>>;
extern template class EINSUMS_EXPORT TiledRuntimeTensor<std::complex<double>>;
#endif

/**
 * @struct TiledRuntimeTensorView
 *
 * @brief A non-owning, per-irrep-sliced view of a @ref TiledRuntimeTensor.
 *
 * Produced by @ref TiledRuntimeTensor::view with one @ref IndexSpace per axis.
 * It does not own data: each populated tile is exposed as a
 * @ref RuntimeTensorView sub-block of the parent tile (via
 * @ref RuntimeTensor::at_view), so a view like ``A[o, v, o, v]`` over an MO
 * tensor selects the occ/vir sub-blocks of every symmetry-allowed tile with no
 * copies. The parent must outlive the view.
 *
 * @tparam T The data type stored in each tile.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
struct
    // clang-format off
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("TiledRuntimeTensorViewF", TiledRuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorViewD", TiledRuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorViewC", TiledRuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("TiledRuntimeTensorViewZ", TiledRuntimeTensorView<std::complex<double>>)
    // clang-format on
    TiledRuntimeTensorView : public tensor_base::CoreTensor,
                             public tensor_base::TiledTensorNoExtra {
  public:
    using ValueType = T;

    /// Compile-time rank sentinel; the real rank is the number of axes (== the
    /// number of IndexSpaces). See @ref einsums::dynamic_rank.
    static constexpr int Rank = dynamic_rank;

    TiledRuntimeTensorView() = default;

    /// Build a view of @p parent selecting @p spaces[k] along axis @p k. One
    /// IndexSpace per axis, each covering every irrep on that axis. Computes the
    /// reduced per-axis tile sizes and global dims; no data is touched.
    TiledRuntimeTensorView(TiledRuntimeTensor<T> &parent, std::vector<IndexSpace> spaces, std::string name)
        : _parent(&parent), _spaces(std::move(spaces)), _name(std::move(name)) {
        auto const &parent_sizes = parent.tile_sizes();
        if (_spaces.size() != parent_sizes.size()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "TiledRuntimeTensorView: got {} index spaces for a rank-{} tensor",
                                    _spaces.size(), parent_sizes.size());
        }
        std::size_t const r = _spaces.size();
        _tile_sizes.resize(r);
        _dims.assign(r, 0);
        for (std::size_t k = 0; k < r; ++k) {
            auto const &axis = parent_sizes[k];
            if (_spaces[k].ranges.size() != axis.size()) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "TiledRuntimeTensorView: axis {} index space covers {} irreps but the tensor has {}", k,
                                        _spaces[k].ranges.size(), axis.size());
            }
            _tile_sizes[k].resize(axis.size());
            for (std::size_t h = 0; h < axis.size(); ++h) {
                auto const &rng = _spaces[k].ranges[h];
                if (rng.first < 0 || rng.second > axis[h] || rng.first > rng.second) {
                    EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                            "TiledRuntimeTensorView: axis {} irrep {} range [{}, {}) outside tile size {}", k, h, rng.first,
                                            rng.second, axis[h]);
                }
                int const len     = rng.second - rng.first;
                _tile_sizes[k][h] = len;
                _dims[k] += static_cast<std::size_t>(len);
            }
        }
    }

    [[nodiscard]] APIARY_EXPOSE std::size_t rank() const noexcept { return _spaces.size(); }

    [[nodiscard]] APIARY_EXPOSE std::vector<std::size_t> dims() const { return _dims; }

    [[nodiscard]] APIARY_EXPOSE std::size_t dim(int d) const {
        int const r = static_cast<int>(_dims.size());
        if (d < 0) {
            d += r;
        }
        return _dims.at(static_cast<std::size_t>(d));
    }

    /// Per-axis, per-irrep sizes of the selected sub-blocks.
    [[nodiscard]] APIARY_EXPOSE std::vector<std::vector<int>> const &tile_sizes() const noexcept { return _tile_sizes; }

    /// The view shares the parent's sparsity pattern (same populated coords).
    [[nodiscard]] APIARY_EXPOSE bool has_tile(std::vector<int> const &coord) const { return _parent->has_tile(coord); }

    [[nodiscard]] APIARY_EXPOSE std::size_t num_filled_tiles() const noexcept { return _parent->num_filled_tiles(); }

    [[nodiscard]] APIARY_GETTER("name") std::string const &name() const noexcept { return _name; }

    /// Sub-view of the parent's tile at @p coord, restricted to this view's
    /// per-axis ranges for that tile's irreps. The result is a live
    /// @ref RuntimeTensorView onto the parent tile's storage.
    APIARY_EXPOSE APIARY_KEEP_ALIVE(0, 1) RuntimeTensorView<T> tile_view(std::vector<int> const &coord) {
        auto &t = _parent->tile(coord);
        t.materialize();
        std::vector<SliceSpec> specs(_spaces.size());
        for (std::size_t k = 0; k < _spaces.size(); ++k) {
            auto const &rng = _spaces[k].ranges.at(static_cast<std::size_t>(coord[k]));
            specs[k]        = SliceSpec{SliceSpec::Kind::Range, 0, rng.first, rng.second, 1};
        }
        return t.at_view(specs);
    }

  private:
    TiledRuntimeTensor<T>        *_parent = nullptr;
    std::vector<IndexSpace>       _spaces;
    std::vector<std::size_t>      _dims;
    std::vector<std::vector<int>> _tile_sizes;
    std::string                   _name;
};

template <typename T>
TiledRuntimeTensorView<T> TiledRuntimeTensor<T>::view(std::vector<IndexSpace> const &spaces) {
    return TiledRuntimeTensorView<T>(*this, spaces, _name + " (view)");
}

#if !defined(EINSUMS_WINDOWS)
extern template class EINSUMS_EXPORT TiledRuntimeTensorView<float>;
extern template class EINSUMS_EXPORT TiledRuntimeTensorView<double>;
extern template class EINSUMS_EXPORT TiledRuntimeTensorView<std::complex<float>>;
extern template class EINSUMS_EXPORT TiledRuntimeTensorView<std::complex<double>>;
#endif

} // namespace einsums
