//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/tensor_base/common.hpp>

#include <complex>
#include <memory>
#include <mutex>
#include <type_traits>

#if defined(__HIP__)
#    include <hip/hip_complex.h>
#endif

namespace einsums::tensor_base {

/**
 * @struct typed_tensor
 *
 * @brief Represents a tensor that stores a data type.
 *
 * @tparam T The data type the tensor stores.
 */
template <typename T>
struct typed_tensor {
    /**
     * @typedef value_type
     *
     * @brief Gets the stored data type.
     */
    using value_type = T;

    typed_tensor()                     = default;
    typed_tensor(const typed_tensor &) = default;

    virtual ~typed_tensor() = default;
};

#if defined(__HIP__)
/**
 * @struct device_typed_tensor
 *
 * Represents a tensor that stores different data types on the host and the device.
 * By default, if the host stores complex<float> or complex<double>, then it converts
 * it to hipComplex or hipDoubleComplex. The two types should have the same storage size.
 * @tparam HostT The host data type.
 * @tparam DevT The device type.
 */
template <typename HostT, typename DevT = void>
    requires(std::is_void_v<DevT> || sizeof(HostT) == sizeof(DevT))
struct device_typed_tensor : virtual typed_tensor<HostT> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<DevT>,
                           std::conditional_t<std::is_same_v<HostT, std::complex<float>>, hipComplex,
                                              std::conditional_t<std::is_same_v<HostT, std::complex<double>>, hipDoubleComplex, HostT>>,
                           DevT>;

    using host_datatype = HostT;

    device_typed_tensor()                            = default;
    device_typed_tensor(const device_typed_tensor &) = default;

    ~device_typed_tensor() override = default;
};
#endif

/**
 * @struct rank_tensor
 *
 * @brief Base class for tensors with a rank. Used for querying the rank.
 *
 * @tparam Rank The rank of the tensor.
 */
template <size_t Rank>
struct rank_tensor {
    /**
     * @property rank
     *
     * @brief The rank of the tensor.
     */
    static constexpr size_t rank = Rank;

    rank_tensor()                    = default;
    rank_tensor(const rank_tensor &) = default;

    virtual ~rank_tensor() = default;

    virtual einsums::dim<Rank> dims() const = 0;

    virtual auto dim(int d) const -> size_t = 0;
};

/**
 * @struct tensor_no_extra
 *
 * @brief Specifies that a class is a tensor, just without any template parameters. Internal use only. Use TensorBase instead.
 *
 * Used for checking that a type is a tensor without regard for storage type.
 */
struct tensor_no_extra {
    tensor_no_extra()                        = default;
    tensor_no_extra(const tensor_no_extra &) = default;

    virtual ~tensor_no_extra() = default;
};

/**
 * @struct tensor
 *
 * @brief Base class for all tensors. Just says that a class is a tensor.
 *
 * Indicates a tensor with a rank and type. Some virtual methods need to be defined.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct tensor : public virtual tensor_no_extra, virtual typed_tensor<T>, virtual rank_tensor<Rank> {
    tensor()               = default;
    tensor(const tensor &) = default;

    ~tensor() override = default;

    virtual bool               full_view_of_underlying() const { return true; }
    virtual const std::string &name() const                          = 0;
    virtual void               set_name(const std::string &new_name) = 0;
};

/**
 * @struct lockable_tensor
 *
 * @brief Base class for lockable tensors. Works with all of the std:: locking functions. Uses a recursive mutex.
 */
struct lockable_tensor {
  protected:
    /**
     * @property _lock
     *
     * @brief The base mutex for locking the tensor.
     */
    mutable std::shared_ptr<std::recursive_mutex> _lock; // Make it mutable so that it can be modified even in const methods.

  public:
    virtual ~lockable_tensor() = default;

    lockable_tensor() { _lock = std::make_shared<std::recursive_mutex>(); }
    lockable_tensor(const lockable_tensor &) { _lock = std::make_shared<std::recursive_mutex>(); }

    /**
     * @brief Lock the tensor.
     */
    virtual void lock() const { _lock->lock(); }

    /**
     * @brief Try to lock the tensor. Returns false if a lock could not be obtained, or true if it could.
     */
    virtual bool try_lock() const { return _lock->try_lock(); }

    /**
     * @brief Unlock the tensor.
     */
    virtual void unlock() const { _lock->unlock(); }

    /**
     * @brief Get the mutex.
     */
    std::shared_ptr<std::recursive_mutex> get_mutex() const { return _lock; }

    /**
     * @brief Set the mutex.
     */
    void set_mutex(std::shared_ptr<std::recursive_mutex> mutex) { _lock = mutex; }
};

/*==================
 * Location-based.
 *==================*/

/**
 * @struct core_tensor
 *
 * @brief Represents a tensor only available to the core.
 */
struct core_tensor {
    core_tensor()                    = default;
    core_tensor(const core_tensor &) = default;

    virtual ~core_tensor() = default;
};

#ifdef __HIP__
/**
 * @struct device_tensor
 *
 * @brief Represents a tensor available to graphics hardware.
 */
struct device_tensor {
  public:
    device_tensor()                      = default;
    device_tensor(const device_tensor &) = default;

    virtual ~device_tensor() = default;
};
#endif

/**
 * @struct disk_tensor
 *
 * @brief Represents a tensor stored on disk.
 */
struct disk_tensor {
    disk_tensor()                    = default;
    disk_tensor(const disk_tensor &) = default;

    virtual ~disk_tensor() = default;
};

/*===================
 * Other properties.
 *===================*/

/**
 * @struct tensor_view_no_extra
 *
 * @brief Internal property that specifies that a tensor is a view.
 *
 * This specifies that a tensor is a view without needing to specify template parameters.
 * This struct is not intended to be used directly by the user, and is used as a base class
 * for TensorViewBase.
 */
struct tensor_view_no_extra {
    tensor_view_no_extra()                             = default;
    tensor_view_no_extra(const tensor_view_no_extra &) = default;

    virtual ~tensor_view_no_extra() = default;
};

/**
 * @struct tensor_view_only_viewed
 *
 * @brief Internal property that specifies that a tensor is a view.
 *
 * This specifies that a tensor is a view without needing to specify type and rank.
 * This struct is not intended to be used directly by the user, and is used as a base class
 * for TensorViewBase.
 *
 * @tparam Viewed The tensor type viewed by this tensor.
 */
template <typename Viewed>
struct tensor_view_only_viewed {
    tensor_view_only_viewed()                                = default;
    tensor_view_only_viewed(const tensor_view_only_viewed &) = default;

    virtual ~tensor_view_only_viewed() = default;
};

/**
 * @struct tensor_view
 *
 * @brief Represents a view of a different tensor.
 *
 * @tparam T The type stored by the underlying tensor.
 * @tparam Rank The rank of the view.
 * @tparam UnderlyingType The tensor type viewed by this view.
 */
template <typename T, size_t Rank, typename UnderlyingType>
struct tensor_view : public virtual tensor_view_no_extra, virtual tensor_view_only_viewed<UnderlyingType>, virtual tensor<T, Rank> {
  public:
    using underlying_type = UnderlyingType;

    tensor_view()                    = default;
    tensor_view(const tensor_view &) = default;

    ~tensor_view() override = default;

    bool full_view_of_underlying() const override = 0;
};

/**
 * @struct basic_tensor_no_extra
 *
 * @brief Represents a regular tensor. Internal use only. Use BasicTensorBase instead.
 *
 * Represents a regular tensor, but without template parameters. See BasicTensorBase for more details.
 */
struct basic_tensor_no_extra {
    basic_tensor_no_extra()                              = default;
    basic_tensor_no_extra(const basic_tensor_no_extra &) = default;

    virtual ~basic_tensor_no_extra() = default;
};

/**
 * @struct basic_tensor
 *
 * @brief Represents a regular tensor. It has no special layouts.
 *
 * Represents a tensor that is not block diagonal, tiled, or otherwise.
 * Just a plain vanilla tensor. These store memory in such a way that it
 * is possible to pass to BLAS and LAPACK.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct basic_tensor : virtual tensor<T, Rank>, virtual basic_tensor_no_extra {
    basic_tensor()                     = default;
    basic_tensor(const basic_tensor &) = default;

    ~basic_tensor() override = default;

    virtual T       *data()       = 0;
    virtual const T *data() const = 0;

    virtual size_t                stride(int d) const = 0;
    virtual einsums::stride<Rank> strides() const     = 0;
};

/**
 * @struct collected_tensor_no_extra
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 */
struct collected_tensor_no_extra {
    collected_tensor_no_extra()                                  = default;
    collected_tensor_no_extra(const collected_tensor_no_extra &) = default;

    virtual ~collected_tensor_no_extra() = default;
};

/**
 * @struct collected_tensor_only_stored
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 *
 * @tparam Stored The type of tensor stored.
 */
template <typename Stored>
struct collected_tensor_only_stored {
    collected_tensor_only_stored()                                     = default;
    collected_tensor_only_stored(const collected_tensor_only_stored &) = default;

    virtual ~collected_tensor_only_stored() = default;
};

/**
 * @struct collected_tensor
 *
 * @brief Specifies that a tensor is a collection of other tensors.
 *
 * Examples of tensor collections include BlockTensors and TiledTensors, which store
 * lists of tensors to save on memory.
 *
 * @tparam T The type of data stored by the tensors in the collection.
 * @tparam Rank The rank of the tensor.
 * @tparam TensorType The type of tensor stored by the collection.
 */
template <typename T, size_t Rank, typename TensorType>
struct collected_tensor : public virtual collected_tensor_no_extra,
                          virtual collected_tensor_only_stored<TensorType>,
                          virtual tensor<T, Rank> {
    using tensor_type = TensorType;

    collected_tensor()                         = default;
    collected_tensor(const collected_tensor &) = default;

    ~collected_tensor() override = default;
};

/**
 * @struct tiled_tensor_no_extra
 *
 * @brief Specifies that a tensor is a tiled tensor without needing to specify type parameters.
 *
 * Only used internally. Use TiledTensorBase in your code.
 */
struct tiled_tensor_no_extra {
    tiled_tensor_no_extra()                              = default;
    tiled_tensor_no_extra(const tiled_tensor_no_extra &) = default;

    virtual ~tiled_tensor_no_extra() = default;
};

// Large class. See TiledTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct tiled_tensor;

/**
 * @struct block_tensor_no_extra
 *
 * @brief Specifies that a tensor is a block tensor. Internal use only. Use BlockTensorBase instead.
 *
 * Specifies that a tensor is a block tensor without needing template parameters. Internal use only.
 * Use BlockTensorBase in your code.
 */
struct block_tensor_no_extra {
    block_tensor_no_extra()                              = default;
    block_tensor_no_extra(const block_tensor_no_extra &) = default;

    virtual ~block_tensor_no_extra() = default;
};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensorBase;

/**
 * @struct function_tensor_no_extra
 *
 * @brief Specifies that a tensor is a function tensor, but with no template parameters. Internal use only. Use FunctionTensorBase instead.
 *
 * Used for checking to see if a tensor is a function tensor. Internal use only. Use FunctionTensorBase instead.
 */
struct function_tensor_no_extra {
    function_tensor_no_extra()                                 = default;
    function_tensor_no_extra(const function_tensor_no_extra &) = default;

    virtual ~function_tensor_no_extra() = default;
};

// Large class. See FunctionTensor.hpp for code.
template <typename T, size_t Rank>
struct function_tensor;

/**
 * @struct algebra_optimized_tensor
 *
 * @brief Specifies that the tensor type can be used by einsum to select different routines other than the generic algorithm.
 */
struct algebra_optimized_tensor {
    algebra_optimized_tensor()                                 = default;
    algebra_optimized_tensor(const algebra_optimized_tensor &) = default;

    virtual ~algebra_optimized_tensor() = default;
};

} // namespace einsums::tensor_base