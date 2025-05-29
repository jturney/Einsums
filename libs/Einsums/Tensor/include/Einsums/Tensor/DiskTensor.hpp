//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/H5.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <string>

namespace einsums {

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t rank>
struct DiskTensor final : public tensor_base::DiskTensor, design_pats::Lockable<std::recursive_mutex> {

    /**
     * @typedef ValueType
     *
     * @brief The type of data stored by this tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of this tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * Default constructor.
     */
    DiskTensor() = default;

    /**
     * Default copy constructor.
     */
    DiskTensor(DiskTensor const &) = default;

    /**
     * Default move constructor.
     */
    DiskTensor(DiskTensor &&) noexcept = default;

    /**
     * Default destructor.
     */
    ~DiskTensor() = default;

    /**
     * Create a new disk tensor bound to a file.
     *
     * @param file The file to use for the storage.
     * @param name The name for the tensor.
     * @param chunk @todo No clue.
     * @param dims The dimensions of the tensor.
     */
    template <typename... Dims>
    explicit DiskTensor(h5::fd_t &file, std::string name, Chunk<sizeof...(Dims)> chunk, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {

        dims_to_strides(_dims, _strides);

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '{}'", _name);
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '{}': {}", _name, e.what());
                std::abort();
            }
        }
    }

    /**
     * Create a new disk tensor bound to a file.
     *
     * @param file The file to use for the storage.
     * @param name The name for the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <typename... Dims, typename = std::enable_if_t<are_all_convertible<size_t, Dims...>::value>>
    explicit DiskTensor(h5::fd_t &file, std::string name, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        dims_to_strides(_dims, _strides);

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr size_t chunk_min{64};
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
                println("Unable to open disk tensor '{}'", _name);
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk_temp} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '{}': {}", _name, e.what());
                std::abort();
            }
        }
    }

    /// Constructs a DiskTensor shaped like the provided Tensor. Data from the provided tensor
    /// is NOT saved.
    explicit DiskTensor(h5::fd_t &file, Tensor<T, Rank> const &tensor) : _file{file}, _name{tensor.name()} {
        // Save dimension information from the provided tensor.
        h5::current_dims cdims;
        for (int i = 0; i < Rank; i++) {
            _dims[i] = tensor.dim(i);
            cdims[i] = _dims[i];
        }

        dims_to_strides(_dims, _strides);

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr size_t chunk_min{64};
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
                println("Unable to open disk tensor '%s'", _name.c_str());
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, cdims, h5::chunk{chunk_temp} /*| h5::gzip{9} | h5::fill_value<T>(0.0)*/);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '%s'", _name.c_str());
                std::abort();
            }
        }
    }

    // Provides ability to store another tensor to a part of a disk tensor.

    /**
     * Get the dimension along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const { return _dims[d]; }

    /**
     * Get the dimensions.
     */
    Dim<Rank> dims() const { return _dims; }

    /**
     * Check whether the data already existed on disk.
     */
    [[nodiscard]] auto existed() const -> bool { return _existed; }

    /**
     * Get the disk object.
     */
    [[nodiscard]] auto disk() -> h5::ds_t & { return _disk; }

    // void _write(Tensor<T, Rank> &data) { h5::write(disk(), data); }

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; }

    /**
     * Get the stride along a given axis.
     */
    size_t stride(int d) const { return _strides[d]; }

    /**
     * @brief Get the array of strides for this tensor.
     */
    Stride<Rank> strides() const { return _strides; }

    /**
     * @brief Returns whether this tensor is viewing the entirety of the data.
     *
     * For this kind of tensor, this will always return true.
     */
    constexpr bool full_view_of_underlying() { return true; }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index)
        -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> {
        // Get positions of All
        auto all_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            counts[value]  = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                               offsets, strides);
    }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) const
        -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> const {
        // Get positions of All
        auto all_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            counts[value]  = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                               offsets, strides);
    }

  private:
    /**
     * @var _file
     *
     * Holds a reference to the file containing the tensor data.
     */
    h5::fd_t &_file;

    /**
     * @var _name
     *
     * The name of the tensor used for printing.
     */
    std::string _name;

    /**
     * @var _dims
     *
     * Holds the dimensions of the tensor.
     */
    Dim<Rank> _dims;

    /**
     * @var _strides
     *
     * Holds the strides of the tensor.
     */
    Stride<Rank> _strides;

    /**
     * @var _disk
     *
     * Holds a reference to the disk manager.
     */
    h5::ds_t _disk;

    /** @var _existed
     *
     * Did the entry already exist on disk? Doesn't indicate validity of the data just the existence of the entry.
     */
    bool _existed{false};
};

/**
 * @struct DiskView
 *
 * @brief Represents a view of a DiskTensor.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam ViewRank The rank of the view.
 * @tparam Rank The rank of the DiskTensor being viewed.
 */
template <typename T, size_t ViewRank, size_t rank>
struct DiskView final : tensor_base::DiskTensor, design_pats::Lockable<std::recursive_mutex> {
    /**
     * @typedef ValueType
     *
     * @brief Holds the type of data stored by this tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the view.
     */
    constexpr static size_t Rank = ViewRank;

    /**
     * @typedef underlying_type
     *
     * @brief Holds the tensor type that this object views. It will be a DiskTensor in this case.
     */
    using underlying_type = einsums::DiskTensor<T, rank>;

    /**
     * Construct a view of a tensor with the given dimensions, counts, strides, and offsets.
     */
    DiskView(einsums::DiskTensor<T, rank> &parent, Dim<ViewRank> const &dims, Count<rank> const &counts, Offset<rank> const &offsets,
             Stride<rank> const &strides)
        : _parent(parent), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides), _tensor{_dims} {
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    };

    /**
     * Construct a view of a tensor with the given dimensions, counts, strides, and offsets.
     */
    DiskView(einsums::DiskTensor<T, rank> const &parent, Dim<ViewRank> const &dims, Count<rank> const &counts, Offset<rank> const &offsets,
             Stride<rank> const &strides)
        : _parent(const_cast<einsums::DiskTensor<T, rank> &>(parent)), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides),
          _tensor{_dims} {
        // Section const section("DiskView constructor");
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
        set_read_only(true);
    };

    /**
     * Default copy constructor
     */
    DiskView(DiskView const &) = default;

    /**
     * Default move constructor
     */
    DiskView(DiskView &&) noexcept = default;

    /**
     * Destructor.
     */
    ~DiskView() { put(); }

    /**
     * Make the tensor view read only.
     */
    void set_read_only(bool readOnly) { _readOnly = readOnly; }

    /**
     * Copy data from a pointer to the view.
     *
     * @attention This is an expert method only. If you are using this, then you must know what you are doing!
     */
    auto operator=(T const *other) -> DiskView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        // Save the data to disk.
        h5::write<T>(_parent.disk(), other, h5::count{_counts}, h5::offset{_offsets});

        return *this;
    }

    /**
     * Copy a tensor into disk.
     *
     * @todo I'm not entirely sure if a TensorView can be sent to the disk.
     */
    template <template <typename, size_t> typename TType>
    auto operator=(TType<T, ViewRank> const &other) -> DiskView & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(access_denied, "Attempting to write data to a read only disk view.");
        }

        // Check dims
        for (int i = 0; i < ViewRank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i));
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
    /**
     * Gets the underlying tensor holding the data. Does not read the tensor.
     */
    auto get() -> Tensor<T, ViewRank> & { return _tensor; }

    /**
     * Push any changes to the view to the disk.
     */
    void put() {
        if (!_readOnly)
            h5::write<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> T const & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    /**
     * Get the dimension along a given axis.
     */
    size_t dim(int d) const { return _tensor.dim(d); }

    /**
     * Get all the dimensions of the view.
     */
    Dim<ViewRank> dims() const { return _tensor.dims(); }

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; }

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator Tensor<T, ViewRank> &() { return _tensor; } // NOLINT

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator Tensor<T, ViewRank> const &() const { return _tensor; } // NOLINT

    /**
     * Set all of the values of the tensor to zero.
     */
    void zero() { _tensor.zero(); }

    /**
     * Set all of the values of the tensor to the passed value.
     */
    void set_all(T value) { _tensor.set_all(value); }

    /**
     * Gets whether the view is showing the whole tensor.
     */
    bool full_view_of_underlying() const {
        size_t prod = 1;
        for (int i = 0; i < ViewRank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= _parent.dim(i);
        }

        return prod == prod2;
    }

  private:
    /**
     * @var _parent
     *
     * This is the tensor that the view is viewing.
     */
    einsums::DiskTensor<T, rank> &_parent;

    /**
     * @var _dims
     *
     * Holds the dimensions of the tensor.
     */
    Dim<ViewRank> _dims;

    /**
     * @var _counts
     *
     * @todo No clue
     */
    Count<rank> _counts;

    /**
     * @var _offsets
     *
     * Holds where in the parent this view starts.
     */
    Offset<rank> _offsets;

    /**
     * @var _strides
     *
     * Holds the strides of the parent.
     */
    Stride<rank> _strides;

    /**
     * @var _tensor
     *
     * This is the in-core representation of the view.
     */
    Tensor<T, ViewRank> _tensor;

    /**
     * @var _name
     *
     * This is the name of the view used for printing.
     */
    std::string _name{"(unnamed)"};

    /**
     * @var _readOnly
     *
     * Indicates whether the view is read-only or read-write.
     */
    bool _readOnly{false};

    // std::unique_ptr<Tensor<ViewRank, T>> _tensor;
};

#ifdef __cpp_deduction_guides
template <typename... Dims>
DiskTensor(h5::fd_t &file, std::string name, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;

template <typename... Dims>
DiskTensor(h5::fd_t &file, std::string name, Chunk<sizeof...(Dims)> chunk, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;
#endif

/**
 * @brief Create a disk tensor object.
 *
 * Creates a new tensor that lives on disk. This function does not create any tensor in memory
 * but the tensor is "created" on the HDF5 @p file handle.
 *
 * @code
 * auto a = create_disk_tensor(handle, "a", 3, 3);           // auto -> DiskTensor<double, 2>
 * auto b = create_disk_tensor<float>(handle, "b", 4, 5, 6); // auto -> DiskTensor<float, 3>
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
auto create_disk_tensor(h5::fd_t &file, std::string const &name, Args... args) -> DiskTensor<Type, sizeof...(Args)> {
    return DiskTensor<Type, sizeof...(Args)>{file, name, args...};
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
 * auto a = create_disk_tensor_like(handle, mem_a);  // auto -> DiskTensor<double, 2>
 * @endcode
 *
 * @tparam Type The datatype of the underlying disk tensor.
 * @tparam Rank The datatypes of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param file The HDF5 file descriptor
 * @param tensor The tensor to reference for size and name.
 * @return A new disk tensor.
 */
template <typename T, size_t Rank>
auto create_disk_tensor_like(h5::fd_t &file, Tensor<T, Rank> const &tensor) -> DiskTensor<T, Rank> {
    return DiskTensor(file, tensor);
}

#ifndef DOXYGEN

EINSUMS_TENSOR_EXPORT(DiskTensor)

EINSUMS_TENSOR_EXPORT_DISK_VIEW(DiskView)

#endif

} // namespace einsums