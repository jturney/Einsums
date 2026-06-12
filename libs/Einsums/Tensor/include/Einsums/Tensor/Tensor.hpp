//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/PendingInit.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>
#include <Einsums/Utilities/Tuple.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums {
// Forward declaration of the Tensor printing function.
template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options = {});

template <FileOrOStream Output, RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options = {});

/**
 * @brief Represents a general tensor
 *
 * @tparam T data type of the underlying tensor data
 * @tparam Rank the rank of the tensor
 */
template <typename T, size_t rank, typename Alloc>
struct GeneralTensor : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = T *;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = T const *;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = T &;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = T const &;

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
     * For device allocators, DeviceVector is used instead of std::vector.
     */
    using Vector = std::conditional_t<gpu::IsDeviceAllocatorV<Alloc>, gpu::DeviceVector<T>, std::vector<T, Alloc>>;

    using Allocator = Alloc;

    /// True if this tensor uses device (GPU) memory.
    static constexpr bool IsDeviceTensor = gpu::IsDeviceAllocatorV<Alloc>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    GeneralTensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    GeneralTensor(GeneralTensor const &other) : _name(other.name()), _data(other._data), _impl(other._impl) {
        _impl.set_data(_data.data());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
        if (other._symmetry)
            _symmetry = std::make_unique<SymmetryDescriptor>(*other._symmetry);
    }

    /**
     * @brief Construct a new Tensor object from one with a different allocator.
     *
     * Handles host↔device transfers automatically when copying between
     * host tensors and GPU tensors.
     */
    template <typename Alloc2>
    GeneralTensor(GeneralTensor<T, Rank, Alloc2> const &other) : _name(other.name()), _impl(other._impl) {
        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        constexpr bool this_is_device  = IsDeviceTensor;

        _data.resize(_impl.size());
        _impl.set_data(_data.data());

        if constexpr (this_is_device && !other_is_device) {
            // Host → Device
            gpu::memcpy_host_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else if constexpr (!this_is_device && other_is_device) {
            // Device → Host
            gpu::memcpy_device_to_host(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else if constexpr (this_is_device && other_is_device) {
            // Device → Device
            gpu::memcpy_device_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else {
            // Host → Host (same as before)
            std::memcpy(_data.data(), other.data(), _impl.size() * sizeof(T));
        }

        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
        if (other._symmetry)
            _symmetry = std::make_unique<SymmetryDescriptor>(*other._symmetry);
    }

    /**
     * @brief Destroy the Tensor object.
     */
    ~GeneralTensor() { ProfileMemFree(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T))); }

    /**
     * @brief Default move constructor.
     */
    GeneralTensor(GeneralTensor &&other) noexcept
        : _name{std::move(other._name)}, _data{std::move(other._data)}, _dim_array{std::move(other._dim_array)},
          _stride_array{std::move(other._stride_array)}, _impl{std::move(other._impl)}, _symmetry{std::move(other._symmetry)} {}

    /**
     * @brief Default move assignment.
     */
    GeneralTensor &operator=(GeneralTensor &&other) noexcept {
        _name         = std::move(other._name);
        _data         = std::move(other._data);
        _dim_array    = std::move(other._dim_array);
        _stride_array = std::move(other._stride_array);
        _symmetry     = std::move(other._symmetry);
        _impl         = std::move(other._impl);

        return *this;
    }

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
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    GeneralTensor(std::string name, Dims... dims)
        : _name{std::move(name)}, _impl(nullptr, std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...},
                                        GlobalConfigMap::get_singleton().get_bool("row-major")) {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        // Resize the data structure
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

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
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    GeneralTensor(bool row_major, std::string name, Dims... dims)
        : _name{std::move(name)}, _impl(nullptr, std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...}, row_major) {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        // Resize the data structure
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
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
    explicit GeneralTensor(GeneralTensor<T, OtherRank, Alloc> &&existingTensor, std::string name, Dims... dims)
        : _name{std::move(name)}, _data(std::move(existingTensor._data)) {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we
        // compute this dimensionality of this "0" index for them.

        auto _dims = std::array<ptrdiff_t, sizeof...(Dims)>{static_cast<ptrdiff_t>(dims)...};

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

        _impl = detail::TensorImpl<T>(_data.data(), _dims, existingTensor.impl().is_row_major());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        // Check size
        if (_data.size() != _impl.size()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Provided dims to not match size of parent tensor");
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit GeneralTensor(Dim<Rank> dims) : _impl(nullptr, dims) {
        // Resize the data structure
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit GeneralTensor(bool row_major, Dim<Rank> dims) : _impl(nullptr, dims, row_major) {
        // Resize the data structure
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    GeneralTensor(TensorView<T, rank> const &other)
        : _name{other.name()}, _impl(nullptr, other.dims(), GlobalConfigMap::get_singleton().get_bool("row-major")) {
        // Resize the data structure
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor from the implementation of another.
     */
    GeneralTensor(detail::TensorImpl<T> const &other) : _impl(nullptr, other.dims()) {
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));

        _impl.set_data(_data.data());
        detail::copy_to(other, _impl);

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    // ── Deferred allocation (shell tensor) ───────────────────────────────

    /**
     * @brief Tag type for deferred allocation.
     *
     * A shell tensor has valid metadata (name, dims, strides, rank) but no
     * backing data storage. Call materialize() to allocate data after
     * optimization passes have decided placement and distribution.
     */
    struct DeferredAlloc {};

    /// Tag value for deferred allocation constructors.
    static constexpr DeferredAlloc deferred_alloc{};

    /**
     * @brief Construct a shell tensor with deferred allocation.
     *
     * Creates a tensor object with valid dimensions and strides but NO
     * data allocation. The tensor address is valid for registration with
     * CaptureContext. Call materialize() to allocate storage.
     *
     * @param tag   The DeferredAlloc tag.
     * @param name  Name of the tensor.
     * @param dims  Dimensions of each rank.
     */
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    GeneralTensor(DeferredAlloc, std::string name, Dims... dims)
        : _name{std::move(name)},
          // Use a sentinel non-null pointer so TensorImpl stores dims/strides correctly.
          // The pointer is never dereferenced; it just prevents TensorImpl::dim() from
          // returning 0, which it does when _ptr == nullptr.
          _impl(reinterpret_cast<T *>(0x1), std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...},
                GlobalConfigMap::get_singleton().get_bool("row-major")) {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");
        // Do not resize _data; storage is deferred until materialize().
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Allocate the backing data storage for a deferred tensor.
     *
     * After this call, data() returns a valid pointer and the tensor
     * can be used in computations. The call is idempotent and safe to repeat.
     */
    void materialize() {
        if (is_materialized())
            return;
        _data.resize(_impl.size());
        ProfileMemAlloc(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));
        _impl.set_data(_data.data());
    }

    /**
     * @brief Check if this tensor has allocated backing storage.
     * @return true if data() returns a valid pointer.
     */
    [[nodiscard]] bool is_materialized() const { return !_data.empty() || _impl.size() == 0; }

    /**
     * @brief Release the backing data storage, returning to deferred state.
     *
     * Frees memory immediately. The tensor retains its dimensions and strides
     * but data() returns a sentinel pointer. Call materialize() to re-allocate.
     * Used by FreeInsertion pass to release intermediates after their last consumer.
     */
    void release() {
        if (_data.empty())
            return;
        ProfileMemFree(static_cast<int64_t>(_data.size()) * static_cast<int64_t>(sizeof(T)));
        _data.clear();
        _data.shrink_to_fit();
        _impl.set_data(reinterpret_cast<T *>(0x1)); // sentinel; dims/strides preserved
    }

    /**
     * @brief Change the dimensions of a deferred tensor before allocation.
     *
     * Used by DistributionPlanning/Materialization to shrink a globally-declared
     * tensor to a local partition before allocating storage. Must only be called
     * on deferred (not yet materialized) tensors.
     *
     * @param dims New dimensions for the tensor.
     */
    void resize_deferred(Dim<Rank> dims) {
        assert(!is_materialized() && "resize_deferred() must be called before materialize()");

        detail::TensorImpl<T> new_impl(reinterpret_cast<T *>(0x1), dims, _impl.is_row_major());
        _impl = std::move(new_impl);

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /// @overload
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    void resize_deferred(Dims... dims) {
        resize_deferred(Dim<Rank>{static_cast<size_t>(dims)...});
    }

    // ── End deferred allocation ───────────────────────────────────────────

    // ── Distributed indexing ─────────────────────────────────────────────

    /**
     * @brief Set the global-to-local mapping for distributed tensors.
     *
     * Called by MaterializationPass after resize_deferred. Stores the global
     * dimensions and the starting offset for this rank's partition along each
     * dimension. Enables range() and global() methods.
     *
     * @param global_dims Global tensor dimensions (before partitioning).
     * @param offsets     Starting global index for this rank along each dimension.
     */
    void set_distribution(std::array<size_t, Rank> global_dims, std::array<size_t, Rank> offsets) {
        _global_dims           = global_dims;
        _local_offset          = offsets;
        _is_distributed_tensor = true;
    }

    /**
     * @brief Get the global index range [start, end) for this rank along dimension @p dim.
     *
     * For non-distributed tensors, returns {0, dim(dim)} (full range).
     * For distributed tensors, returns the partition assigned to this rank.
     *
     * Use this in fill lambdas to iterate only over local elements:
     * @code
     * auto [p_start, p_end] = T.range(0);
     * for (size_t p = p_start; p < p_end; p++) { ... }
     * @endcode
     */
    [[nodiscard]] std::pair<size_t, size_t> range(size_t dim) const {
        if (!_is_distributed_tensor) {
            return {0, this->dim(dim)};
        }
        return {_local_offset[dim], _local_offset[dim] + this->dim(dim)};
    }

    /**
     * @brief Access an element using GLOBAL indices.
     *
     * Subtracts the local offset before accessing the underlying data.
     * For non-distributed tensors, equivalent to operator().
     *
     * @code
     * auto [p_start, p_end] = T.range(0);
     * for (size_t p = p_start; p < p_end; p++)
     *     T.global(p, q) = value;  // p is a global index
     * @endcode
     */
    template <std::integral... Indices>
        requires(sizeof...(Indices) == Rank)
    T &global(Indices... indices) {
        if (!_is_distributed_tensor) {
            return (*this)(indices...);
        }
        std::array<size_t, Rank> idx{static_cast<size_t>(indices)...};
        for (size_t d = 0; d < Rank; d++) {
            idx[d] -= _local_offset[d];
        }
        return _apply_global(idx, std::make_index_sequence<Rank>{});
    }

    template <std::integral... Indices>
        requires(sizeof...(Indices) == Rank)
    T const &global(Indices... indices) const {
        if (!_is_distributed_tensor) {
            return (*this)(indices...);
        }
        std::array<size_t, Rank> idx{static_cast<size_t>(indices)...};
        for (size_t d = 0; d < Rank; d++) {
            idx[d] -= _local_offset[d];
        }
        return _apply_global(idx, std::make_index_sequence<Rank>{});
    }

    /// Check if this tensor has distribution metadata set.
    [[nodiscard]] bool is_distributed_tensor() const { return _is_distributed_tensor; }

    /// Global dimension along axis @p d (before partitioning).
    [[nodiscard]] size_t global_dim(size_t d) const { return _is_distributed_tensor ? _global_dims[d] : this->dim(d); }

  private:
    template <size_t... Is>
    // NOLINTNEXTLINE(readability-identifier-naming)
    T &_apply_global(std::array<size_t, Rank> const &idx, std::index_sequence<Is...>) {
        return (*this)(idx[Is]...);
    }
    template <size_t... Is>
    // NOLINTNEXTLINE(readability-identifier-naming)
    T const &_apply_global(std::array<size_t, Rank> const &idx, std::index_sequence<Is...>) const {
        return (*this)(idx[Is]...);
    }

    std::array<size_t, Rank> _global_dims{};
    std::array<size_t, Rank> _local_offset{};
    bool                     _is_distributed_tensor{false};

  public:
    // ── Local view (temporary slice for distributed computing) ───────────

    /**
     * @brief Saved state for restoring a tensor after a local view.
     */
    struct LocalViewState {
        T                       *saved_data{nullptr};
        std::array<size_t, Rank> saved_dims{};
        std::array<size_t, Rank> saved_strides{};
        bool                     active{false};
    };

    /**
     * @brief Temporarily restrict this tensor to a contiguous slice along one dimension.
     *
     * Modifies the data pointer and dims so that operations see only the local
     * partition. Call end_local_view() to restore the original state.
     *
     * Used by InputSlicing pass for distributed computation: each rank "views"
     * its portion of a pre-allocated tensor without copying.
     *
     * @param dim   Which dimension to slice along.
     * @param start First element index along that dimension.
     * @param count Number of elements in the slice.
     * @return Saved state to pass to end_local_view().
     */
    LocalViewState begin_local_view(size_t dim, size_t start, size_t count) {
        LocalViewState state;
        state.saved_data = _impl.data();
        for (size_t d = 0; d < Rank; d++) {
            state.saved_dims[d]    = _impl.dim(d);
            state.saved_strides[d] = _impl.stride(d);
        }
        state.active = true;

        // Offset the data pointer: move by start * stride[dim] elements
        T *new_data = _impl.data() + start * _impl.stride(dim);

        // Build new dims with the sliced dimension
        std::array<size_t, Rank> new_dims;
        std::array<size_t, Rank> old_strides;
        for (size_t d = 0; d < Rank; d++) {
            new_dims[d]    = _impl.dim(d);
            old_strides[d] = _impl.stride(d);
        }
        new_dims[dim] = count;

        // Rebuild impl with new data and dims, then OVERRIDE strides.
        // The strides must stay the same as the original tensor because the
        // underlying memory layout hasn't changed; we're just viewing a subset.
        bool                  row_major = _impl.is_row_major();
        detail::TensorImpl<T> new_impl(new_data, new_dims, row_major);
        // Override the computed strides with the original ones
        new_impl.set_strides(old_strides);
        _impl = std::move(new_impl);

        for (size_t d = 0; d < Rank; d++) {
            _dim_array[d]    = new_dims[d];
            _stride_array[d] = old_strides[d];
        }

        return state;
    }

    /**
     * @brief Restore a tensor to its original state after a local view.
     */
    void end_local_view(LocalViewState const &state) {
        if (!state.active)
            return;

        detail::TensorImpl<T> restored(state.saved_data, state.saved_dims, _impl.is_row_major());
        _impl = std::move(restored);

        for (size_t d = 0; d < Rank; d++) {
            _dim_array[d]    = _impl.dim(d);
            _stride_array[d] = _impl.stride(d);
        }
    }

    // ── End local view ───────────────────────────────────────────────────

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    void resize(Dim<Rank> dims) {
        if (std::equal(_impl.dims().cbegin(), _impl.dims().cend(), dims.cbegin())) {
            return;
        }

        // Build new impl to compute the required size, but don't commit yet.
        detail::TensorImpl<T> new_impl(nullptr, dims, _impl.is_row_major());

        // Resize data first; if this throws, _impl and _data remain consistent.
        _data.resize(new_impl.size());

        // Data resize succeeded, so now commit the new impl.
        _impl = std::move(new_impl);
        _impl.set_data(_data.data());

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    void resize(Dims... dims) {
        resize(Dim<Rank>{static_cast<size_t>(dims)...});
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        // _data.data() is allowed to be nullptr when _data.size() == 0
        // (e.g. zero-sized tensors used during construction or as views).
        // memset / device_memset are declared [[gnu::nonnull(1)]] in the
        // glibc/CUDA headers, so passing nullptr trips UBSan even though
        // the count is also 0 (which is otherwise defined). Skip the call
        // when there's nothing to write. Caught on the ASan/UBSan leg by
        // Tests.Unit.Modules.LinearAlgebra.pow + .TensorUtilities.BlockViews.
        if (_data.empty()) {
            return;
        }
        if constexpr (IsDeviceTensor) {
            gpu::device_memset(_data.data(), 0, sizeof(T) * _data.size());
        } else if constexpr (std::is_trivially_copyable_v<T>) {
            memset(_data.data(), 0, sizeof(T) * _data.size());
        } else {
            std::fill(_data.begin(), _data.end(), T{0.0});
        }
    }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        static_assert(!IsDeviceTensor, "set_all() is not supported for device tensors. Use a GPU kernel instead.");
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
    [[nodiscard]] Pointer data() { return _impl.data(); }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    [[nodiscard]] ConstPointer data() const { return _impl.data(); }

    /**
     * @brief Redirect the tensor's internal data pointer.
     *
     * Used by the ComputeGraph GPU executor to temporarily swap the tensor's
     * data buffer to a device shadow allocation. The caller is responsible for
     * restoring the original pointer after use.
     *
     * @warning This does NOT change ownership or lifetime of the underlying buffer.
     *          The original buffer must remain valid until the pointer is restored.
     *
     * @param[in] ptr New data pointer. Must point to a buffer of at least size() elements.
     */
    void set_data(Pointer ptr) { _impl.set_data(ptr); }

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
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    [[nodiscard]] auto data(MultiIndex &&...index) -> Pointer {
        EINSUMS_ASSERT(sizeof...(MultiIndex) <= Rank);

        return _impl.data(std::forward<MultiIndex>(index)...);
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
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    [[nodiscard]] auto operator()(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
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
    template <Container Index>
    [[nodiscard]] auto subscript(Index const &index) const -> ConstReference {
        return _impl.subscript_no_check(index);
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
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    [[nodiscard]] auto subscript(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
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
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
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
    template <Container Index>
    auto subscript(Index const &index) -> Reference {
        return _impl.subscript_no_check(index);
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
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
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
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>; }
    auto operator()(MultiIndex const &...index)
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{
            _impl.template subscript<true>(index...), data()};
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
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>; }
    auto operator()(MultiIndex const &...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(
            _impl.template subscript<true>(index...));
    }

    /**
     * @copydoc GeneralTensor<T, Rank, Alloc>::operator(MultiIndex...) -> T&
     */
    template <Container Index>
        requires requires {
            requires !std::is_same_v<Index, Offset<Rank>>;
            requires !std::is_same_v<Index, Range>;
        }
    Reference operator()(Index const &index) {
        return _impl.subscript(index);
    }

    /**
     * @copydoc GeneralTensor<T, Rank, Alloc>::operator(MultiIndex...) -> T&
     */
    template <Container Index>
        requires requires {
            requires !std::is_same_v<Index, Offset<Rank>>;
            requires !std::is_same_v<Index, Range>;
        }
    ConstReference operator()(Index const &index) const {
        return _impl.subscript(index);
    }

    /**
     * Copy the data from one tensor into this.
     *
     * @param other The tensor to copy.
     */
    auto operator=(GeneralTensor const &other) -> GeneralTensor & {
        if (&other == this)
            return *this;

        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = other.impl();

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        if constexpr (IsDeviceTensor) {
            gpu::memcpy_device_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else {
            detail::copy_to(other.impl(), _impl);
        }

        _symmetry = other._symmetry ? std::make_unique<SymmetryDescriptor>(*other._symmetry) : nullptr;

        return *this;
    }

    template <typename Alloc2>
    auto operator=(GeneralTensor<T, Rank, Alloc2> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = other.impl();

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        if constexpr (IsDeviceTensor && !other_is_device) {
            gpu::memcpy_host_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else if constexpr (!IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_host(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else if constexpr (IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else {
            detail::copy_to(other.impl(), _impl);
        }

        return *this;
    }

    /**
     * Cast the data from one tensor while copying its data into this tensor.
     *
     * @param other The tensor to cast and copy.
     */
    template <typename TOther, typename Alloc2>
        requires(!std::same_as<T, TOther>)
    auto operator=(GeneralTensor<TOther, Rank, Alloc2> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = detail::TensorImpl<T>(nullptr, other.dims(), other.impl().is_row_major());

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        detail::copy_to(other.impl(), _impl);

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        return *this;
    }

    /**
     * Copy the data from a tensor of a different kind into this one.
     */

    template <BasicTensorConcept OtherTensor>
    auto operator=(OtherTensor const &other) -> GeneralTensor & {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <TensorConcept OtherTensor>
        requires requires {
            requires !BasicTensorConcept<OtherTensor>;
            requires SameRank<GeneralTensor, OtherTensor>;
            requires CoreTensorConcept<OtherTensor>;
        }
    auto operator=(OtherTensor const &other) -> GeneralTensor & {
        static_assert(!IsDeviceTensor, "Element-wise assignment from a non-basic tensor is not supported for device tensors. "
                                       "Use bulk operations (memcpy, BLAS) or the ComputeGraph instead.");
        LabeledSection("operator=");
        size_t size = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            thread_local std::array<size_t, Rank> index;
            sentinel_to_indices(sentinel, _impl.strides(), index);
            _data[sentinel] = subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * Cast the data from a tensor view while copying into this tensor.
     */
    template <typename TOther>
    auto operator=(TensorView<TOther, Rank> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
    detail:
        copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Fill this tensor with a value.
     */
    auto operator=(T const &fill_value) -> GeneralTensor & {
        LabeledSection("operator= value");
        set_all(fill_value);
        return *this;
    }

    GeneralTensor &operator+=(T const &b) {
        detail::impl_scalar_add_contiguous(b, _impl);

        return *this;
    }

    GeneralTensor &operator-=(T const &b) {
        detail::impl_scalar_add_contiguous(-b, _impl);

        return *this;
    }

    GeneralTensor &operator*=(T const &b) {
        detail::impl_scal_contiguous(b, _impl);

        return *this;
    }

    GeneralTensor &operator/=(T const &b) {
        detail::impl_div_scalar_contiguous(b, _impl);

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator+=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy_contiguous(T{1}, other.impl(), _impl);
            } else {
                detail::impl_axpy_contiguous(T{1.0}, other.impl(), _impl);
            }
        } else {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy(T{1}, other.impl(), _impl);
            } else {
                detail::impl_axpy(T{1.0}, other.impl(), _impl);
            }
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator-=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy_contiguous(T{-1}, other.impl(), _impl);
            } else {
                detail::impl_axpy_contiguous(T{-1.0}, other.impl(), _impl);
            }
        } else {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy(T{-1}, other.impl(), _impl);
            } else {
                detail::impl_axpy(T{-1.0}, other.impl(), _impl);
            }
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator*=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            detail::impl_mult_contiguous(other.impl(), _impl);
        } else {
            detail::impl_mult(other.impl(), _impl);
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator/=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            detail::impl_div_contiguous(other.impl(), _impl);
        } else {
            detail::impl_div(other.impl(), _impl);
        }

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator+=(TOther const &other) {
        detail::add_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator-=(TOther const &other) {
        detail::sub_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator*=(TOther const &other) {
        detail::mult_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator/=(TOther const &other) {
        detail::div_assign(other.impl(), _impl);

        return *this;
    }

    /**
     * Get the dimension of the tensor along a given axis.
     */
    [[nodiscard]] size_t dim(int d) const { return _impl.dim(d); }

    /**
     * Get all the dimensions of the tensor.
     */
    [[nodiscard]] Dim<Rank> const &dims() const { return _dim_array; }

    /**
     * Get the internal vector containing the tensor's data.
     */
    [[nodiscard]] auto vector_data() const -> Vector const & { return _data; }

    /// @copydoc GeneralTensor<T,Rank, Alloc>::vector_data() const
    [[nodiscard]] auto vector_data() -> Vector & { return _data; }

    /**
     * Get the stride along a given axis.
     */
    [[nodiscard]] size_t stride(int d) const { return _impl.stride(d); }

    /**
     * Get the strides of this tensor.
     */
    [[nodiscard]] Stride<Rank> const &strides() const { return _stride_array; }

    /**
     * Flatten out the tensor.
     */
    [[nodiscard]] auto to_rank_1_view() const -> TensorView<T, 1> {
        Dim<1> dim{size()};

        return TensorView<T, 1>{*this, dim};
    }

    /**
     * Returns the linear size of the tensor.
     */
    [[nodiscard]] size_t size() const { return _impl.size(); }

    /**
     * Indicates that the tensor is contiguous.
     */
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; };

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; };

    /**
     * Get the implementation of the tensor.
     */
    [[nodiscard]] detail::TensorImpl<T> &impl() { return _impl; }

    /**
     * Get the implementation of the tensor.
     */
    [[nodiscard]] detail::TensorImpl<T> const &impl() const { return _impl; }

    [[nodiscard]] bool is_row_major() const { return _impl.is_row_major(); }

    [[nodiscard]] bool is_column_major() const { return _impl.is_column_major(); }

    // ── Symmetry metadata ──────────────────────────────────────────────────
    //
    // A tensor optionally carries a SymmetryDescriptor describing invariants
    // the caller guarantees hold for its data (e.g. ``T(i,j) = T(j,i)``).
    // The descriptor is metadata; storage remains dense. BLAS dispatch and
    // ComputeGraph passes read it to pick specialized kernels
    // (``syev`` over ``geev``, ``symm`` over ``gemm``, etc.).
    //
    // The descriptor is shared-owned so copies preserve the declared
    // symmetry without copying the (small) descriptor payload.

    /// Attach or replace the symmetry descriptor. Pass an empty descriptor
    /// to clear.
    void set_symmetry(SymmetryDescriptor desc) {
        if (desc.empty())
            _symmetry.reset();
        else
            _symmetry = std::make_unique<SymmetryDescriptor>(std::move(desc));
    }

    /// Current symmetry descriptor, or ``nullptr`` if none declared. The
    /// returned pointer is valid for the tensor's lifetime.
    [[nodiscard]] SymmetryDescriptor const *symmetry() const noexcept { return _symmetry.get(); }

    /// Clear any declared symmetry.
    void clear_symmetry() { _symmetry.reset(); }

    /// True iff a non-empty symmetry descriptor is attached.
    [[nodiscard]] bool has_symmetry() const { return _symmetry && !_symmetry->empty(); }

    [[nodiscard]] TensorView<T, Rank> transpose_view() { return TensorView<T, Rank>(_impl.transpose_view()); }

    [[nodiscard]] TensorView<T, Rank> const transpose_view() const { return TensorView<T, Rank>(_impl.transpose_view()); }

    [[nodiscard]] TensorView<T, Rank> to_row_major() { return TensorView<T, Rank>(_impl.to_row_major()); }

    [[nodiscard]] TensorView<T, Rank> const to_row_major() const { return TensorView<T, Rank>(_impl.to_row_major()); }

    [[nodiscard]] TensorView<T, Rank> to_column_major() { return TensorView<T, Rank>(_impl.to_column_major()); }

    [[nodiscard]] TensorView<T, Rank> const to_column_major() const { return TensorView<T, Rank>(_impl.to_column_major()); }

    [[nodiscard]] bool is_gemmable(size_t *lda = nullptr) const { return _impl.is_gemmable(lda); }

    [[nodiscard]] bool is_totally_vectorable(size_t *incx = nullptr) const { return _impl.is_totally_vectorable(incx); }

    template <std::integral... MultiIndex>
    [[nodiscard]] TensorView<T, Rank - sizeof...(MultiIndex) + 1> tie_indices(MultiIndex &&...index) {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    [[nodiscard]] TensorView<T, Rank - sizeof...(MultiIndex) + 1> const tie_indices(MultiIndex &&...index) const {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    /// Lazily-created token whose lifetime tracks this object. The graph's
    /// validator, see make_handle, holds a std::weak_ptr to it to detect
    /// destruction without dereferencing a possibly-freed tensor. Reading a
    /// destroyed object's memory, the old canary approach, is undefined
    /// behavior and unreliable. The token is created on first request, so
    /// tensors never captured into a graph pay nothing.
    [[nodiscard]] std::weak_ptr<void> liveness_token() const {
        if (!_life_token) {
            _life_token = std::make_shared<char>();
        }
        return _life_token;
    }

  private:
    mutable std::shared_ptr<void> _life_token;

    std::string _name{"(unnamed)"};

    Vector _data{};

    detail::TensorImpl<T> _impl{};

    Dim<Rank>    _dim_array;
    Stride<Rank> _stride_array;

    /// Optional declared symmetry, null when the tensor is treated as general.
    /// unique_ptr keeps copies independent, avoiding accidental aliasing through
    /// a refcount, and avoids atomic overhead. A null descriptor costs 8 bytes
    /// per tensor. The copy constructor and copy-assignment operator on
    /// GeneralTensor deep-clone the descriptor so the symmetry survives across
    /// copies.
    std::unique_ptr<SymmetryDescriptor> _symmetry{};

    /// Post-materialize init policy. Set at declaration time (e.g. by
    /// ``Workspace::declare_zero_tensor``) and consumed by ``make_handle``
    /// so the init metadata reaches a Graph that captures this tensor
    /// later. POD-ish so it's safe across copies / moves; see
    /// ``Einsums/Tensor/PendingInit.hpp``.
    PendingInit _pending_init{PendingInit::None};

    template <typename T_, size_t Rank_>
    friend struct TensorView;

    template <typename T_, size_t OtherRank, typename Alloc2>
    friend struct GeneralTensor;

  public:
    /// Pending post-materialize init kind. Defaults to ``None``.
    [[nodiscard]] PendingInit pending_init() const { return _pending_init; }

    /// Tag this tensor with a post-materialize init policy. Used by
    /// declaration helpers (Workspace / Graph) so capture-time handles
    /// created via ``make_handle`` pick up the same init behavior.
    void set_pending_init(PendingInit k) { _pending_init = k; }
};

/**
 * @struct Tensor<T, 0>
 *
 * @brief Represents a zero-rank tensor. It has a special implementation since it is essentially a scalar.
 *
 * @tparam T The data type being stored.
 */
template <typename T, typename Alloc>
struct GeneralTensor<T, 0, Alloc> final : tensor_base::CoreTensor,
                                          design_pats::Lockable<std::recursive_mutex>,
                                          tensor_base::AlgebraOptimizedTensor {

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
    GeneralTensor() = default;

    /**
     * Default copy constructor
     */
    GeneralTensor(GeneralTensor const &) = default;

    /**
     * Default move constructor
     */
    GeneralTensor(GeneralTensor &&) noexcept = default;

    /**
     * Default destructor
     */
    ~GeneralTensor() = default;

    /**
     * Create a new zero-rank tensor with the given name.
     */
    explicit GeneralTensor(std::string name) : _name{std::move(name)} {};

    /**
     * Create a new zero-rank tensor with the given dimensions. Since it is zero-rank,
     * the dimensions will be empty, and are ignored.
     */
    explicit GeneralTensor(Dim<0> _ignore) {}

    /**
     * Get the pointer to the data stored by this tensor.
     */
    [[nodiscard]] T *data() { return &_data; }

    /**
     * @copydoc GeneralTensor<T,0,Alloc>::data()
     */
    [[nodiscard]] T const *data() const { return &_data; }

    /**
     * Copy assignment.
     */
    auto operator=(GeneralTensor const &other) -> GeneralTensor & {
        _data = other._data;
        return *this;
    }

    /**
     * Set the value of the tensor to the value passed in.
     */
    auto operator=(T const &other) -> GeneralTensor & {
        _data = other;
        return *this;
    }

#if defined(OPERATOR)
#    undef OPERATOR
#endif
#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &other)->GeneralTensor & {                                                                                    \
        _data OP other;                                                                                                                    \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

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
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * Get the dimension of the tensor. Always returns 1.
     */
    [[nodiscard]] size_t dim(int) const { return 1; }

    /**
     * Get the dimensions of the tensor. The result is empty.
     */
    [[nodiscard]] Dim<0> dims() const { return Dim{}; }

    /**
     * Indicates that the tensor is contiguous.
     */
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the stride of the tensor. Always returns 1.
     */
    [[nodiscard]] size_t stride(int /*d*/) const { return 0; }

    /**
     * Get the strides of the tensor. The result is empty.
     */
    [[nodiscard]] Stride<0> strides() const { return Stride{}; }

    /// Lazily-created token whose lifetime tracks this object. The graph's
    /// validator, see make_handle, holds a std::weak_ptr to it to detect
    /// destruction without dereferencing a possibly-freed tensor. Reading a
    /// destroyed object's memory, the old canary approach, is undefined
    /// behavior and unreliable. The token is created on first request, so
    /// tensors never captured into a graph pay nothing.
    [[nodiscard]] std::weak_ptr<void> liveness_token() const {
        if (!_life_token) {
            _life_token = std::make_shared<char>();
        }
        return _life_token;
    }

  private:
    mutable std::shared_ptr<void> _life_token;

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
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = T *;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = T const *;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = T &;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = T const &;

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
    // NOLINTNEXTLINE(readability-identifier-naming)
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
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(GeneralTensor<T, OtherRank, Alloc> const &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(const_cast<GeneralTensor<T, OtherRank, Alloc> &>(other), dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(GeneralTensor<T, OtherRank, Alloc> &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> const &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(const_cast<TensorView<T, OtherRank> &>(other), dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(std::string name, GeneralTensor<T, OtherRank, Alloc> &other, Dim const &dims, Args &&...args)
        : _name{std::move(name)} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims, bool row_major)
        : _impl(const_cast<T *>(data), dims, row_major), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims, bool row_major)
        : _impl(const_cast<T *>(data), dims, row_major), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims)
        : _impl(const_cast<T *>(data), dims, GlobalConfigMap::get_singleton().get_bool("row-major")), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims)
        : _impl(const_cast<T *>(data), dims, GlobalConfigMap::get_singleton().get_bool("row-major")), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     * @param strides The strides for the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims, Stride<Rank> const &strides)
        : _impl(const_cast<T *>(data), dims, strides), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
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
        : _impl(const_cast<T *>(data), dims, strides), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    explicit TensorView(detail::TensorImpl<T> const &impl) : _impl(impl), _parent{const_cast<T *>(impl.data())} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }
    explicit TensorView(detail::TensorImpl<T> const &impl, T *parent) : _impl(impl), _parent{parent} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    explicit TensorView(detail::TensorImpl<T> const &impl, T const *parent) : _impl(impl), _parent{const_cast<T *>(parent)} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Copy data from a pointer into this view. It always assumes that the source array is stored in row-major
     * order.
     *
     * @attention This is an expert function only. If you are using it, you must know what you are doing!
     */
    auto operator=(T const *other) -> TensorView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        detail::TensorImpl<T> other_impl(const_cast<T *>(other), _impl.dims(), true);

        detail::copy_to(other_impl, _impl);

        return *this;
    }

    /**
     * Copy assignment.
     */
    auto operator=(TensorView const &other) -> TensorView & {
        if (this == &other)
            return *this;

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Copy the data from another tensor into this view.
     */
    template <typename AType>
        requires CoreRankTensor<AType, Rank, T>
    auto operator=(AType const &other) -> TensorView & {
        if constexpr (std::is_same_v<AType, TensorView>) {
            if (this == &other)
                return *this;
        }

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Fill this view with a single value.
     */
    auto operator=(T const &fill_value) -> TensorView & {
        detail::copy_to(fill_value, _impl);

        return *this;
    }

    void zero() { *this = T{0.0}; }

    void set_all(T value) { *this = value; }

    template <typename TOther>
    TensorView &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    /**
     * Get a pointer to the data.
     */
    Pointer data() { return _impl.data(); }

    /**
     * @copydoc TensorView<T,Rank>::data()
     */
    ConstPointer data() const { return _impl.data(); }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index The index for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto data(MultiIndex &&...index) const -> ConstPointer {
        return _impl.data(std::forward<MultiIndex>(index)...);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    auto data(Index const &index_list) const -> ConstPointer {
        return _impl.data(index_list);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index The index for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto data(MultiIndex &&...index) -> Pointer {
        return _impl.data(std::forward<MultiIndex>(index)...);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    auto data(Index const &index_list) -> Pointer {
        return _impl.data(index_list);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    [[deprecated("The data_array method will be removed in the future. Its functionality will be taken by data.")]] auto
    data_array(Index const &index_list) const -> Pointer {
        return const_cast<Pointer>(data(index_list));
    }

    Pointer full_data() noexcept { return _parent; }

    ConstPointer full_data() const noexcept { return _parent; }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, std::remove_cvref_t<MultiIndex>...> || AtLeastOneOfType<Range, std::remove_cvref_t<MultiIndex>...>)
    [[nodiscard]] auto operator()(MultiIndex &&...index) -> TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                                                              count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()> {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    [[nodiscard]] auto operator()(MultiIndex &&...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    [[nodiscard]] auto subscript(MultiIndex &&...index)
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    [[nodiscard]] auto subscript(MultiIndex &&...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
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
    [[nodiscard]] auto subscript(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);
        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
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
    [[nodiscard]] auto subscript(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
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
    [[nodiscard]] auto subscript(std::array<int_type, Rank> const &index) const -> ConstReference {
        return _impl.subscript_no_check(index);
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
    [[nodiscard]] auto subscript(std::array<int_type, Rank> const &index) -> Reference {
        return _impl.subscript_no_check(index);
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
    Reference operator()(Container const &index) {
        return _impl.subscript(index);
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
    ConstReference operator()(Container const &index) const {
        return _impl.subscript(index);
    }

    /**
     * Get the dimension of the view along a given axis.
     *
     * @param d The axis to query.
     */
    [[nodiscard]] size_t dim(int d) const { return _impl.dim(d); }

    /**
     * Get the dimension of the original tensor along a given axis.
     */
    [[nodiscard]] size_t source_dim(int d) const {
        int temp = d;
        if (temp < 0) {
            temp += Rank;
        }

        if (temp < 0 || std::cmp_greater_equal(temp, Rank)) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index to the source dims was out of range! Expected between {} and {}, got {}.",
                                    -(ptrdiff_t)Rank, Rank - 1, d);
        }

        return _source_dims[temp];
    }

    /**
     * Get the dimensions of the view.
     */
    [[nodiscard]] Dim<Rank> const &dims() const { return _dim_array; }

    /**
     * Get the dimensions of the original tensor.
     */
    [[nodiscard]] Dim<Rank> const &source_dims() const { return _source_dims; }

    /**
     * Get the name of the view.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

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
    [[nodiscard]] size_t stride(int d) const { return _impl.stride(d); }

    /**
     * Get the strides of the tensor.
     */
    [[nodiscard]] Stride<Rank> const &strides() const noexcept { return _stride_array; }

    /**
     * Get the offset of the view along a given axis.
     */
    [[nodiscard]] size_t offset(int d) const {
        int temp = d;
        if (temp < 0) {
            temp += Rank;
        }

        if (temp < 0 || std::cmp_greater_equal(temp, Rank)) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index to the offsets was out of range! Expected between {} and {}, got {}.",
                                    -(ptrdiff_t)Rank, Rank - 1, d);
        }

        return _offsets[temp];
    }

    /**
     * Get the offsets of the tensor.
     */
    [[nodiscard]] Offset<Rank> offsets() const noexcept { return _offsets; }

    /**
     * Flatten the view.
     *
     * @warning Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!
     */
    [[nodiscard]] auto to_rank_1_view() const -> TensorView<T, 1> {
        if constexpr (Rank == 1) {
            return *this;
        } else {
            if (std::min(_impl.stride(0), _impl.stride(-1)) != 1) {
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
            }
            size_t size = Rank == 0 ? 0 : std::max(stride(0) * dim(0), stride(-1) * dim(-1));
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
    [[nodiscard]] bool full_view_of_underlying() const noexcept { return _impl.is_contiguous(); }

    /**
     * Get the number of elements in the view.
     */
    [[nodiscard]] size_t size() const { return _impl.size(); }

    /**
     * Get the underlying implementation details.
     */
    [[nodiscard]] detail::TensorImpl<T> &impl() noexcept { return _impl; }

    /**
     * Get the underlying implementation details.
     */
    [[nodiscard]] detail::TensorImpl<T> const &impl() const noexcept { return _impl; }

    [[nodiscard]] bool is_row_major() const { return _impl.is_row_major(); }

    [[nodiscard]] bool is_column_major() const { return _impl.is_column_major(); }

    // Views don't inherit symmetry yet (would need reasoning about which
    // slice preserves the parent's invariants). Return null so the
    // dispatch fast-path falls through to the general kernel. Phase 2+
    // can populate a descriptor when the view is provably symmetric.
    [[nodiscard]] SymmetryDescriptor const *symmetry() const noexcept { return nullptr; }
    [[nodiscard]] bool                      has_symmetry() const noexcept { return false; }

    [[nodiscard]] TensorView<T, Rank> transpose_view() { return TensorView<T, Rank>(_impl.transpose_view()); }

    [[nodiscard]] TensorView<T, Rank> const transpose_view() const { return TensorView<T, Rank>(_impl.transpose_view()); }

    [[nodiscard]] TensorView<T, Rank> to_row_major() { return TensorView<T, Rank>(_impl.to_row_major()); }

    [[nodiscard]] TensorView<T, Rank> const to_row_major() const { return TensorView<T, Rank>(_impl.to_row_major()); }

    [[nodiscard]] TensorView<T, Rank> to_column_major() { return TensorView<T, Rank>(_impl.to_column_major()); }

    [[nodiscard]] TensorView<T, Rank> const to_column_major() const { return TensorView<T, Rank>(_impl.to_column_major()); }

    [[nodiscard]] bool is_gemmable(size_t *lda = nullptr) const { return _impl.is_gemmable(lda); }

    [[nodiscard]] bool is_totally_vectorable(size_t *incx = nullptr) const { return _impl.is_totally_vectorable(incx); }

    template <std::integral... MultiIndex>
    [[nodiscard]] TensorView<T, Rank - sizeof...(MultiIndex) + 1> tie_indices(MultiIndex &&...index) {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    [[nodiscard]] TensorView<T, Rank - sizeof...(MultiIndex) + 1> const tie_indices(MultiIndex &&...index) const {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

  private:
    /**
     * Initialize a view using a tensor.
     */
    template <TensorConcept TensorType, typename... Args>
        requires(std::is_same_v<T, typename TensorType::ValueType>)
    auto common_initialization(TensorType &other, Dim<Rank> const &dims, Args &&...args) -> void {
        constexpr size_t OtherRank = TensorType::Rank;

        static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller that the Tensor being viewed.");

        _parent = other.data();

        // set_mutex(other.get_mutex());
        Stride<Rank>      default_strides{};
        Offset<OtherRank> default_offsets{};
        Offset<OtherRank> temp_offsets{};
        Stride<Rank>      error_strides{};
        error_strides[0]   = -1;
        Dim<Rank>    _dims = dims;
        Stride<Rank> _strides{};
        T           *_data;

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we compute this
        // dimensionality for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(dims)) {
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
            default_strides = other.strides();
            // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
            // provide the information
        } else {
            if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies()) ==
                std::accumulate(other.dims().begin(), other.dims().end(), 1.0, std::multiplies())) {
                dims_to_strides(_dims, default_strides, other.impl().is_row_major());
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
        size_t ordinal = indices_to_sentinel(other.strides(), temp_offsets);

        // Find source dimensions
        if constexpr (IsIncoreBasicTensorV<TensorType> && IsSameUnderlyingAndRankV<TensorType, TensorView<T, Rank>>) {
            _source_dims    = other.dims();
            _offsets        = temp_offsets;
            _offset_ordinal = ordinal;
            ordinal         = 0;
        } else {
            // In different Ranks, source dimensions can be deduced from the strides.
            // In decreasing strides when matching the correct dimension is reached (dim[i] = 1 is neglected)
            // Where dimenions less than the outermost are skipped these are incorporated into the dimension inner to it
            // If there are no further inner strides, the inner stride is non-zero to account for missing dimensionality.

            if (other.impl().is_column_major()) {
                size_t current_stride    = 1;
                size_t tensor_index      = 0;
                size_t cumulative_stride = 1;
                for (int i = 0; std::cmp_less(i, Rank); i++) {
                    _source_dims[i]   = 0;
                    cumulative_stride = 1;
                    current_stride    = _strides[i];
                    while (_source_dims[i] == 0) {
                        if (other.stride(tensor_index) == current_stride) {
                            _source_dims[i] = cumulative_stride;
                            _offsets[i]     = temp_offsets[tensor_index];
                        }
                        cumulative_stride *= other.dim(tensor_index);
                        tensor_index++;
                    }
                    if (_source_dims[i] == 0) {
                        EINSUMS_THROW_EXCEPTION(bad_logic,
                                                "Unable to deduce source dimensions. Stride does not follow source tensor dimensions.");
                    }
                }
            } else {
                size_t current_stride    = 1;
                size_t tensor_index      = 0;
                size_t cumulative_stride = 1;
                for (int i = 0; std::cmp_less(i, Rank); i++) {
                    _source_dims[i]   = 0;
                    cumulative_stride = 1;
                    current_stride    = _strides[i];
                    while (_source_dims[i] == 0) {
                        cumulative_stride *= other.dim(tensor_index);
                        if (other.stride(tensor_index) == current_stride) {
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
            }

            _offset_ordinal = indices_to_sentinel(_strides, _offsets); // Only counts offsets that are in the view
            ordinal -= _offset_ordinal;
        }

        if (Rank > 0 && _strides[Rank - 1] == 0) {
            _strides[Rank - 1] = 1;
        }

        for (ptrdiff_t i = static_cast<ptrdiff_t>(Rank) - 2; i >= 0; i--) {
            if (_strides[i] == 0) {
                _strides[i] = _strides[i + 1];
            }
        }

        // Determine the ordinal using the offsets provided (if any) and the strides of the parent
        _data = &(other.data()[_offset_ordinal]);

        _impl = detail::TensorImpl<T>(_data, _dims, _strides);

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @var _name
     *
     * The name of the view used for printing.
     */
    std::string _name{"(Unnamed View)"};

    /**
     * @var _impl
     *
     * The underlying implementation.
     */
    detail::TensorImpl<T> _impl;

    /**
     * @var _source_dims
     *
     * The dimensions of the source tensor.
     */
    Dim<Rank> _source_dims;

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
    Dim<Rank>    _dim_array;
    Stride<Rank> _stride_array;

    T *_parent{nullptr};

    template <typename T_, size_t Rank_, typename Alloc>
    friend struct GeneralTensor;

    template <typename T_, size_t OtherRank_>
    friend struct TensorView;
};

/**
 * Function that zeros a tensor.
 */
template <TensorConcept TensorType>
void zero(TensorType &A) {
    A.zero();
}

#ifdef __cpp_deduction_guides
template <typename... Args>
GeneralTensor(std::string const &, Args...) -> GeneralTensor<double, sizeof...(Args), std::allocator<double>>;

template <typename... Args>
GeneralTensor(bool, std::string const &, Args...) -> GeneralTensor<double, sizeof...(Args), std::allocator<double>>;

template <typename T, size_t OtherRank, typename Alloc, typename... Dims>
explicit GeneralTensor(GeneralTensor<T, OtherRank, Alloc> &&otherTensor, std::string name, Dims... dims)
    -> GeneralTensor<T, sizeof...(dims), Alloc>;

template <size_t Rank, typename... Args>
explicit GeneralTensor(Dim<Rank> const &, Args...) -> GeneralTensor<double, Rank, std::allocator<double>>;

template <size_t Rank, typename... Args>
explicit GeneralTensor(bool, Dim<Rank> const &, Args...) -> GeneralTensor<double, Rank, std::allocator<double>>;

template <typename T, size_t Rank, typename Alloc>
GeneralTensor(GeneralTensor<T, Rank, Alloc> const &) -> GeneralTensor<T, Rank, Alloc>;

template <typename T, size_t Rank, typename Alloc>
GeneralTensor(GeneralTensor<T, Rank, Alloc> &&) -> GeneralTensor<T, Rank, Alloc>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(GeneralTensor<T, OtherRank, Alloc> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(GeneralTensor<T, OtherRank, Alloc> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(std::string, GeneralTensor<T, OtherRank, Alloc> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

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
auto create_tensor(bool row_major, std::string const &name, Args... args) {
    EINSUMS_LOG_TRACE("creating tensor {}, {}", name, std::forward_as_tuple(args...));
    return Tensor<Type, sizeof...(Args)>{row_major, name, args...};
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

template <FileOrOStream Output, RankTensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent;
        if constexpr (!TensorViewConcept<AType>)
            fprintln(fp, "Type: In Core Tensor");
        else
            fprintln(fp, "Type: In Core Tensor View");
    }

    if constexpr (AType::Rank == 0) {
        {
            print::Indent const indent{};

            fprintln(fp, "Data Type: {}", type_name<typename AType::ValueType>());

            if (options.full_output) {
                fprintln(fp);

                typename AType::ValueType value = A;

                std::ostringstream oss;
                oss << "              ";
                if constexpr (std::is_floating_point_v<typename AType::ValueType>) {
                    if (std::abs(value) < 1.0E-4) {
                        oss << fmt::format("{:14.4e} ", value);
                    } else {
                        oss << fmt::format("{:14.8f} ", value);
                    }
                } else if constexpr (IsComplexV<typename AType::ValueType>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(fp, "{}", oss.str());
                fprintln(fp);
            }
        }
        fprintln(fp);
    } else {
        fprintln(fp, A.impl(), options);
    }
}

template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options) {
    fprintln(std::cout, A, options);
}

TENSOR_EXPORT_ALLOC_RANK(GeneralTensor, 0, std::allocator)
TENSOR_ALLOC_EXPORT(GeneralTensor, std::allocator)
TENSOR_ALLOC_EXPORT(GeneralTensor, BufferAllocator)

TENSOR_EXPORT(TensorView)

} // namespace einsums
