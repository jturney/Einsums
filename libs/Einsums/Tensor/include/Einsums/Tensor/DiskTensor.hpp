//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/Utilities.hpp>

#include <H5Dpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace einsums {

namespace detail {

EINSUMS_EXPORT bool verify_exists(hid_t loc_id, std::string const &path, hid_t lapl_id);

}

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
    DiskTensor() = delete;

    /**
     * Default copy constructor.
     */
    DiskTensor(DiskTensor const &other) : _file{other._file}, _name{other.name()}, _dims{other.dims()}, _size{other.size()} {
        _dataspace = H5Scopy(other._dataspace);

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }
        _existed = false;

        _creation_props = H5Pcopy(other._creation_props);

        if (_name.size() == 0) {
            // Create temporary names.
            auto new_temp1 = make_temp_path();
            auto new_temp2 = make_temp_path();

            // Link the temporary dataset into a temporary location.
            auto err = H5Olink(other._dataset, _file, new_temp1.c_str(), detail::Einsums_Tensor_vars::get_singleton().link_property_list,
                               H5P_DEFAULT);

            if (err < 0) {
                H5Sclose(_dataspace);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not link anonymous tensor into file for copying!");
            }

            H5Dflush(other._dataset);

            // Copy.
            err = H5Ocopy(_file, new_temp1.c_str(), _file, new_temp2.c_str(), _creation_props,
                          detail::Einsums_Tensor_vars::get_singleton().link_property_list);
            if (err < 0) {
                H5Ldelete(_file, new_temp1.c_str(), H5P_DEFAULT);
                H5Ldelete(_file, new_temp2.c_str(), H5P_DEFAULT);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Something went wrong when copying anonymous disk tensors!");
            }

            // Unlink the datasets so they are deleted from the file when the tensors are freed.
            H5Ldelete(_file, new_temp1.c_str(), H5P_DEFAULT);
            H5Ldelete(_file, new_temp2.c_str(), H5P_DEFAULT);
        } else {
            // Create temporary name.
            auto new_temp = make_temp_path();

            H5Dflush(other._dataset);

            // Copy.
            auto err = H5Ocopy(_file, other.name().c_str(), _file, new_temp.c_str(), _creation_props,
                               detail::Einsums_Tensor_vars::get_singleton().link_property_list);
            if (err < 0) {
                H5Ldelete(_file, new_temp.c_str(), H5P_DEFAULT);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Something went wrong when copying disk tensors!");
            }

            // Unlink the datasets so they are deleted from the file when the tensors are freed.
            H5Ldelete(_file, new_temp.c_str(), H5P_DEFAULT);
        }
    }

    /**
     * Default move constructor.
     */
    DiskTensor(DiskTensor &&other) noexcept
        : _tensor(std::move(other._tensor)), _constructed(other._constructed), _creation_props(other._creation_props),
          _data_type(other._data_type), _dataset(other._dataset), _dataspace(other._dataspace), _dims(std::move(other._dims)),
          _existed(other._existed), _file(other._file), _size(other._size), _strides(std::move(other._strides)) {
        other._constructed    = false;
        other._creation_props = H5I_INVALID_HID;
        other._dataset        = H5I_INVALID_HID;
        other._dataspace      = H5I_INVALID_HID;
        other._existed        = false;
        other._file           = H5I_INVALID_HID;
        other._size           = 0;
    }

    /**
     * Default destructor.
     */
    ~DiskTensor() { // NOLINT(bugprone-exception-escape)
        if (_dataset != H5I_INVALID_HID) {
            put();
            H5Dclose(_dataset);
        }

        if (_dataspace != H5I_INVALID_HID) {
            H5Sclose(_dataspace);
        }

        if (_creation_props != H5I_INVALID_HID) {
            H5Pclose(_creation_props);
        }
    }

    /**
     * Create a new disk tensor bound to a file.
     *
     * @param file The file to use for the storage. Can also be a parent object.
     * @param name The name for the tensor.
     * @param dims The dimensions of the tensor.
     */
    explicit DiskTensor(hid_t file, std::string name, Dim<Rank> dims, int deflate_level = -1)
        : _file{file}, _name{std::move(name)}, _dims{dims} {

        _size = dims_to_strides(_dims, _strides, true);

        std::array<hsize_t, Rank> max_dims;
        max_dims.fill(H5S_UNLIMITED);

        _dataspace = H5Screate_simple(Rank, reinterpret_cast<hsize_t *>(_dims.data()), max_dims.data());

        if (_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Dataspace creation failed!");
        }

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        // Check to see if the data set exists
        if (detail::verify_exists(file, _name, H5P_DEFAULT)) {
            _existed = true;
            _dataset = H5Dopen(file, _name.c_str(), H5P_DEFAULT);
        } else {
            _existed        = false;
            _creation_props = H5Pcreate(H5P_DATASET_CREATE);

            if (_creation_props == H5I_INVALID_HID) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create creation property list!");
            }

            auto err = H5Pset_layout(_creation_props, H5D_CHUNKED);

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not set layout to chunked!");
            }

            std::array<size_t, rank> chunk_dims;

            chunk_dims.fill(1);

            size_t prod = 1;

            // Maximum chunk size is 2^32 - 1, but chunks can not be bigger than the data set in any dimension.
            for (int i = rank - 1; i >= 0; i--) {
                if (prod * _dims[i] >= 0xffffffffUL / sizeof(T)) {
                    chunk_dims[i] = 0xffffffffUL / sizeof(T) / prod;
                    break;
                } else {
                    prod *= _dims[i];
                    chunk_dims[i] = _dims[i];
                }
            }

            err = H5Pset_chunk(_creation_props, rank, reinterpret_cast<hsize_t const *>(chunk_dims.data()));

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not set up the chunk properties!");
            }

            err = H5Pset_chunk_opts(_creation_props, H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS);

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not set up the chunk options!");
            }

            if (deflate_level < 0 && _size > 0xffffffff) {
                deflate_level = 1;
            }

            if (deflate_level > 0) {
                err = H5Pset_deflate(_creation_props, deflate_level);

                if (err < 0) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not set up compression options!");
                }
            }

            if (_name.size() > 0) {
                _dataset = H5Dcreate(_file, _name.c_str(), _data_type, _dataspace,
                                     detail::Einsums_Tensor_vars::get_singleton().link_property_list, _creation_props, H5P_DEFAULT);
            } else {
                _dataset = H5Dcreate_anon(_file, _data_type, _dataspace, _creation_props, H5P_DEFAULT);
            }
        }

        if (_dataset == H5I_INVALID_HID) {
            H5Sclose(_dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not open data set!");
        }
    }

    /**
     * Create a new disk tensor bound to a file.
     *
     * @param file The file to use for the storage. Can also be a parent object.
     * @param name The name for the tensor.
     * @param chunk @todo No clue.
     * @param dims The dimensions of the tensor.
     */
    template <std::integral... Dims>
    explicit DiskTensor(hid_t file, std::string const &name, Dims... dims) : DiskTensor(file, name, Dim{dims...}) {}

    /**
     * Create a new disk tensor bound to a file.
     *
     * @param file The file to use for the storage. Can also be a parent object.
     * @param name The name for the tensor.
     * @param chunk @todo No clue.
     * @param dims The dimensions of the tensor.
     */
    template <typename... Dims>
    explicit DiskTensor(std::string const &name, Dims &&...dims)
        : DiskTensor(detail::Einsums_Tensor_vars::get_singleton().hdf5_file, name, std::forward<Dims>(dims)...) {}

    // Provides ability to store another tensor to a part of a disk tensor.

    template <std::integral... Dims>
    void resize(Dims... dims) {
        resize(Dim<Rank>{dims...});
    }

    void resize(Dim<Rank> const &new_dims) {
        put();
        _constructed = false;
        H5Dflush(_dataset);
        auto err = H5Dset_extent(_dataset, reinterpret_cast<hsize_t const *>(new_dims.data()));
        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not extend the disk tensor!");
        }
        H5Dflush(_dataset);
        H5Sclose(_dataspace);
        _dataspace = H5Dget_space(_dataset);
        _dims      = new_dims;
        _size      = dims_to_strides(_dims, _strides, true);

        err = H5Sselect_all(_dataspace);

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not select the extended dataspace!");
        }

        Dim<Rank> test_dims, max_dims;

        auto test_rank = H5Sget_simple_extent_dims(_dataspace, reinterpret_cast<hsize_t *>(test_dims.data()),
                                                   reinterpret_cast<hsize_t *>(max_dims.data()));

        if (std::cmp_not_equal(test_rank, Rank)) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Something went wrong when creating the resized data space!");
        }

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (test_dims[i] != _dims[i]) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "The new dimensions did not get set properly during resize!");
            }
        }
    }

    DiskTensor &operator=(DiskTensor const &other) {
        _dims        = other._dims;
        _strides     = other._strides;
        _size        = other._size;
        _existed     = true;
        _constructed = false;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        if (_file == H5I_INVALID_HID) {
            _file = detail::Einsums_Tensor_vars::get_singleton().hdf5_file;
        }

        _creation_props = H5Pcopy(other._creation_props);

        if (_name.size() == 0) {
            // Create temporary names.
            auto new_temp1 = make_temp_path();
            auto new_temp2 = make_temp_path();

            // Link the temporary dataset into a temporary location.
            auto err = H5Olink(other._dataset, _file, new_temp1.c_str(), detail::Einsums_Tensor_vars::get_singleton().link_property_list,
                               H5P_DEFAULT);

            if (err < 0) {
                H5Sclose(_dataspace);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not link anonymous tensor into file for copying!");
            }

            H5Dflush(other._dataset);

            // Copy.
            err = H5Ocopy(_file, new_temp1.c_str(), _file, new_temp2.c_str(), _creation_props,
                          detail::Einsums_Tensor_vars::get_singleton().link_property_list);
            if (err < 0) {
                H5Ldelete(_file, new_temp1.c_str(), H5P_DEFAULT);
                H5Ldelete(_file, new_temp2.c_str(), H5P_DEFAULT);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Something went wrong when copying anonymous disk tensors!");
            }

            // Unlink the datasets so they are deleted from the file when the tensors are freed.
            H5Ldelete(_file, new_temp1.c_str(), H5P_DEFAULT);
            H5Ldelete(_file, new_temp2.c_str(), H5P_DEFAULT);
        } else {
            // Create temporary name.
            auto new_temp = make_temp_path();

            H5Dflush(other._dataset);

            // Copy.
            auto err = H5Ocopy(_file, other.name().c_str(), _file, new_temp.c_str(), _creation_props,
                               detail::Einsums_Tensor_vars::get_singleton().link_property_list);
            if (err < 0) {
                H5Ldelete(_file, new_temp.c_str(), H5P_DEFAULT);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Something went wrong when copying disk tensors!");
            }

            // Unlink the datasets so they are deleted from the file when the tensors are freed.
            H5Ldelete(_file, new_temp.c_str(), H5P_DEFAULT);
        }

        _dataspace = H5Dget_space(_dataset);
        return *this;
    }

    /**
     * Get the dimension along a given axis.
     *
     * @param d The axis to query.
     */
    [[nodiscard]] size_t dim(int d) const { return _dims[d]; }

    /**
     * Get the dimensions.
     */
    [[nodiscard]] Dim<Rank> dims() const { return _dims; }

    /**
     * Check whether the data already existed on disk.
     */
    [[nodiscard]] auto existed() const -> bool { return _existed; }

    /**
     * Get the parent object/file.
     */
    [[nodiscard]] hid_t file() const { return _file; }

    /**
     * Get the dataspace.
     */
    [[nodiscard]] hid_t dataspace() const { return _dataspace; }

    /**
     * Get the dataset.
     */
    [[nodiscard]] hid_t dataset() const { return _dataset; }

    // void _write(Tensor<T, Rank> &data) { h5::write(disk(), data); }

    /**
     * Get the name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     *
     * @versionchangeddesc{2.0.0}
     *      When changing the name of a tensor, links are now moved within the file.
     * @endversion
     */
    void set_name(std::string const &new_name) {
        herr_t err;
        if (_name.size() == 0) {
            err = H5Olink(_dataset, _file, new_name.c_str(), detail::Einsums_Tensor_vars::get_singleton().link_property_list, H5P_DEFAULT);
        } else {
            err = H5Lmove(_file, _name.c_str(), _file, new_name.c_str(), detail::Einsums_Tensor_vars::get_singleton().link_property_list,
                          H5P_DEFAULT);
        }

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Error when changing the name of a disk tensor! Could not create new link!");
        }
        _name = new_name;
    }

    /**
     * Get the stride along a given axis.
     */
    [[nodiscard]] size_t stride(int d) const { return _strides[d]; }

    /**
     * @brief Get the array of strides for this tensor.
     */
    [[nodiscard]] Stride<Rank> strides() const { return _strides; }

    /**
     * Get the size of the tensor.
     */
    [[nodiscard]] size_t size() const { return _size; }

    /**
     * @brief Returns whether this tensor is viewing the entirety of the data.
     *
     * For this kind of tensor, this will always return true.
     */
    [[nodiscard]] constexpr bool full_view_of_underlying() { return true; }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
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
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Dget_space(_dataset);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        auto err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), nullptr,
                                       reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) const
        -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        auto all_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Dget_space(_dataset);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        auto err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), nullptr,
                                       reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    template <CoreBasicTensorConcept TensorType>
        requires(RankTensorConcept<TensorType, rank>)
    void write(TensorType const &tensor) {

        std::array<size_t, rank> dims, counts;

        counts.fill(1);

        hid_t mem_dataspace = H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(tensor.dims().data()),
                                               reinterpret_cast<hsize_t const *>(tensor.dims().data()));

        if (mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
        }

        herr_t err;

        if (tensor.is_row_major()) {
            err = H5Dwrite(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, tensor.data());
        } else {
            auto lock = std::lock_guard(*this);
            if (!_constructed) {
                _tensor      = BufferTensor<T, Rank>{true, _dims};
                _constructed = true;
            }

            _tensor = tensor;

            err = H5Dwrite(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());
        }

        H5Sclose(mem_dataspace);

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not write data to HDF5 file!");
        }
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] BufferTensor<T, rank> &get() {
        auto                     lock = std::lock_guard(*this);
        std::array<size_t, rank> counts;

        counts.fill(1);

        hid_t mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims().data()), reinterpret_cast<hsize_t const *>(dims().data()));

        if (mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
        }
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;

            auto err = H5Dread(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

            if (err < 0) {
                H5Sclose(mem_dataspace);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
            }
        }

        H5Sclose(mem_dataspace);
        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] BufferTensor<T, rank> const &get() const {
        auto                     lock = std::lock_guard(*this);
        std::array<size_t, rank> counts;

        counts.fill(1);

        hid_t mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims().data()), reinterpret_cast<hsize_t const *>(dims().data()));

        if (mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
        }
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;

            auto err = H5Dread(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

            if (err < 0) {
                H5Sclose(mem_dataspace);
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
            }
        }
        H5Sclose(mem_dataspace);
        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] BufferTensor<T, rank> &get_update() {
        auto                     lock = std::lock_guard(*this);
        std::array<size_t, rank> counts;

        counts.fill(1);

        hid_t mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims().data()), reinterpret_cast<hsize_t const *>(dims().data()));

        if (mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
        }
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;
        }
        auto err = H5Dread(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

        if (err < 0) {
            H5Sclose(mem_dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
        }

        H5Sclose(mem_dataspace);
        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] BufferTensor<T, rank> const &get_update() const {
        auto                     lock = std::lock_guard(*this);
        std::array<size_t, rank> counts;

        counts.fill(1);

        hid_t mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims().data()), reinterpret_cast<hsize_t const *>(dims().data()));

        if (mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
        }
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;
        }

        auto err = H5Dread(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

        if (err < 0) {
            H5Sclose(mem_dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
        }

        H5Sclose(mem_dataspace);
        return _tensor;
    }

    /**
     * Writes the buffered tensor's data to the disk then destroys the buffered tensor.
     * There is no const version.
     */
    void unget() {
        if (_constructed) {
            put();

            _tensor.~BufferTensor<T, rank>();
            _constructed = false;
        }
    }

    /**
     * Push any changes to the view to the disk. There is no const version.
     */
    void put() {
        if (_constructed) {
            std::array<size_t, rank> counts;

            counts.fill(1);

            hid_t mem_dataspace =
                H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims().data()), reinterpret_cast<hsize_t const *>(dims().data()));

            if (mem_dataspace == H5I_INVALID_HID) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create memory dataspace!");
            }
            H5Dwrite(_dataset, _data_type, mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());
            H5Sclose(mem_dataspace);
        }
    }

    void unlink() const { H5Ldelete(_file, _name.c_str(), H5P_DEFAULT); }

  private:
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

    hid_t _file{H5I_INVALID_HID}, _dataspace{H5I_INVALID_HID}, _dataset{H5I_INVALID_HID}, _data_type{H5I_INVALID_HID},
        _creation_props{H5I_INVALID_HID};

    /**
     * @var _size
     *
     * Holds the size of the tensor.
     */
    size_t _size;

    /** @var _existed
     *
     * Did the entry already exist on disk? Doesn't indicate validity of the data just the existence of the entry.
     */
    bool _existed{false};

    mutable bool _constructed{false};

    mutable BufferTensor<T, rank> _tensor;
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
template <typename T, size_t rank>
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
    constexpr static size_t Rank = rank;

    /**
     * @typedef underlying_type
     *
     * @brief Holds the tensor type that this object views. It will be a DiskTensor in this case.
     */
    using underlying_type = einsums::DiskTensor<T, rank>; // NOLINT(readability-identifier-naming)

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskTensor<T, BaseRank> &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2);

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskTensor<T, BaseRank> const &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2);

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        set_read_only(true);
        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskView<T, BaseRank> &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2) && parent.full_view_of_underlying();

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }
        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskView<T, BaseRank> const &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2) && parent.full_view_of_underlying();

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        set_read_only(true);

        _size = prod;
    }

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
    ~DiskView() {
        put();

        if (_dataspace != H5I_INVALID_HID) {
            H5Sclose(_dataspace);
            _dataspace = H5I_INVALID_HID;
        }

        if (_mem_dataspace != H5I_INVALID_HID && _mem_dataspace != H5S_ALL) {
            H5Sclose(_mem_dataspace);
            _mem_dataspace = H5S_ALL;
        }
    }

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
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(access_denied, "Attempting to write data to a read only disk view.");
        }

        (void)get();

        std::memcpy(_tensor.data(), other, _tensor.size() * sizeof(T));

        return *this;
    }

    /**
     * Copy a tensor into disk.
     */
    template <TensorConcept TType>
        requires SameUnderlyingAndRank<TType, DiskView>
    auto operator=(TType const &other) -> DiskView & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(access_denied, "Attempting to write data to a read only disk view.");
        }

        // Check dims
        for (int i = 0; i < rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i));
            }
        }

        // Performing the write here will cause a double write to occur. The destructor above will call put to save
        // the data to disk.
        // Sync the data to disk and into our internal tensor.
        // h5::write<T>(_parent.disk(), other.data(), h5::count{_counts}, h5::offset{_offsets});
        get() = other;

        return *this;
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] auto get() -> BufferTensor<T, rank> & {
        auto lock = std::lock_guard(*this);
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;

            auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
            }
        }
        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] auto get() const -> BufferTensor<T, rank> const & {
        auto lock = std::lock_guard(*this);
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;

            auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
            }
        }
        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] auto get_update() -> BufferTensor<T, rank> & {
        auto lock = std::lock_guard(*this);
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;
        }

        auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
        }

        return _tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] auto get_update() const -> BufferTensor<T, rank> const & {
        auto lock = std::lock_guard(*this);
        if (!_constructed) {
            _tensor      = BufferTensor<T, Rank>{true, _dims};
            _constructed = true;
        }

        auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
        }

        return _tensor;
    }

    /**
     * Writes the buffered tensor's data to the disk then destroys the buffered tensor.
     * There is no const version.
     */
    void unget() {
        if (_constructed) {
            put();

            _tensor.~BufferTensor<T, rank>();
            _constructed = false;
        }
    }

    /**
     * Push any changes to the view to the disk.
     */
    void put() {
        if (!_readOnly && _constructed) {
            H5Dwrite(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor.data());
        }
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
        requires(NoneOfType<AllT, MultiIndex...> && NoneOfType<Range, MultiIndex...>)
    auto operator()(MultiIndex &&...index) const -> T const & {
        return get()(std::forward<MultiIndex>(index)...);
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
        requires(NoneOfType<AllT, MultiIndex...> && NoneOfType<Range, MultiIndex...>)
    auto operator()(MultiIndex &&...index) -> T & {
        return get()(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
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
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Scopy(_dataspace);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        int parent_rank = H5Sget_simple_extent_ndims(_dataspace);

        if (parent_rank < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        std::vector<hsize_t> start_vec(parent_rank), count_vec(parent_rank), block_vec(parent_rank);

        auto err = H5Sget_regular_hyperslab(dataspace, start_vec.data(), nullptr, count_vec.data(), block_vec.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        for (int i = 0, j = 0; i < parent_rank && j < rank; i++) {
            if (block_vec[i] != 1) {
                start_vec[i] += offsets[j];
                j++;
            }

            while (j < rank && block[j] == 1) {
                j++;
            }
        }

        err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), nullptr,
                                  reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) const
        -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
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
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Scopy(_dataspace);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        int parent_rank = H5Sget_simple_extent_ndims(_dataspace);

        if (parent_rank < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        std::vector<hsize_t> start_vec(parent_rank), count_vec(parent_rank), block_vec(parent_rank);

        auto err = H5Sget_regular_hyperslab(dataspace, start_vec.data(), nullptr, count_vec.data(), block_vec.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        for (int i = 0, j = 0; i < parent_rank && j < rank; i++) {
            if (block_vec[i] != 1) {
                start_vec[i] += offsets[j];
                j++;
            }

            while (j < rank && block[j] == 1) {
                j++;
            }
        }

        err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), nullptr,
                                  reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    /**
     * Get the dimension along a given axis.
     */
    [[nodiscard]] size_t dim(int d) const {
        if (d < 0) {
            d += rank;
        }
        return _dims.at(d);
    }

    /**
     * Get all the dimensions of the view.
     */
    [[nodiscard]] Dim<rank> dims() const { return _dims; }

    /**
     * Get the size of the view.
     */
    [[nodiscard]] size_t size() const { return _size; }

    /**
     * Get the name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; }

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator BufferTensor<T, rank> &() { return get(); } // NOLINT

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator BufferTensor<T, rank> const &() const { return get(); } // NOLINT

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
    [[nodiscard]] bool full_view_of_underlying() const { return _full_view; }

  private:
    /**
     * @var _dataspace
     *
     * The dataspace that contains the parameters for the view.
     */
    hid_t _dataspace{H5I_INVALID_HID};

    /**
     * @var _mem_dataspace
     *
     * The dataspace that specifies the parameters for the core tensor.
     */
    hid_t _mem_dataspace{H5S_ALL};

    /**
     * @var _dataset
     *
     * The data set that this tensor is viewing.
     */
    hid_t _dataset{H5I_INVALID_HID};

    /**
     * @var _data_type
     *
     * The data type identifier.
     */
    hid_t _data_type{H5I_INVALID_HID};

    /**
     * @var _dims
     *
     * Holds the dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @var _size
     *
     * The size of the tensor view.
     */
    size_t _size;

    /**
     * @var _tensor
     *
     * This is the in-core representation of the view.
     */
    mutable BufferTensor<T, rank> _tensor;

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

    /**
     * @var _full_view
     *
     * Indicates whether this view sees the entire data space that it was built on.
     */
    bool _full_view{false};

    mutable bool _constructed{false};

    // std::unique_ptr<Tensor<ViewRank, T>> _tensor;
};

#ifdef __cpp_deduction_guides
template <typename... Dims>
DiskTensor(hid_t file, std::string name, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;

template <typename... Dims>
DiskTensor(std::string name, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;
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
auto create_disk_tensor(hid_t file, std::string const &name, Args &&...args) -> DiskTensor<Type, sizeof...(Args)> {
    return DiskTensor<Type, sizeof...(Args)>{file, name, std::forward<Args>(args)...};
}

template <typename Type = double, typename... Args>
auto create_disk_tensor(std::string const &name, Args &&...args) -> DiskTensor<Type, sizeof...(Args)> {
    return DiskTensor<Type, sizeof...(Args)>{detail::Einsums_Tensor_vars::get_singleton().hdf5_file, name, std::forward<Args>(args)...};
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
auto create_disk_tensor_like(hid_t file, Tensor<T, Rank> const &tensor) -> DiskTensor<T, Rank> {
    return DiskTensor(file, tensor.name(), tensor.dims());
}

template <typename T, size_t Rank>
auto create_disk_tensor_like(Tensor<T, Rank> const &tensor) -> DiskTensor<T, Rank> {
    return DiskTensor(detail::Einsums_Tensor_vars::get_singleton().hdf5_file, tensor.name(), tensor.dims());
}

#ifndef DOXYGEN

TENSOR_EXPORT(DiskTensor)

TENSOR_EXPORT(DiskView)

#endif

} // namespace einsums