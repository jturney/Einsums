//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <vector>

namespace einsums {

/**
 * @struct TensorPrintOptions
 * @brief Represents options and default options for printing tensors.
 */
struct TensorPrintOptions {
    /**
     * @var width
     *
     * How many columns of tensor data are printed per line.
     */
    int width{7};

    /**
     * @var full_output
     *
     * Print the tensor data (true) or just name and data span information (false).
     */
    bool full_output{true};
};

namespace detail {

/**
 * @enum HostToDeviceMode
 *
 * @brief Enum that specifies how device tensors store data and make it available to the GPU.
 */
enum HostToDeviceMode { UNKNOWN, DEV_ONLY, MAPPED, PINNED };

} // namespace detail

#ifndef DOXYGEN
// Forward declarations of tensors.
template <typename T, size_t Rank>
struct Tensor;

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct TiledTensor;

#    if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct BlockDeviceTensor;

template <typename T, size_t Rank>
struct TiledDeviceTensor;

template <typename T, size_t Rank>
struct TiledDeviceTensorView;
#    endif

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t Rank>
struct TiledTensorView;

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView;

template <typename T, size_t Rank>
struct DiskTensor;

template <typename T>
struct RuntimeTensor;

template <typename T>
struct RuntimeTensorView;

namespace detail {
auto EINSUMS_EXPORT allocate_aligned_memory(size_t align, size_t size) -> void *;
void EINSUMS_EXPORT deallocate_aligned_memory(void *ptr) noexcept;
} // namespace detail

template <typename T, size_t Align = 32>
struct AlignedAllocator;

template <size_t Align>
struct AlignedAllocator<void, Align> {
  public:
    using pointer       = void *;
    using const_pointer = void const *;
    using value_type    = void;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template <typename T, size_t Align>
struct AlignedAllocator {
    using value_type      = T;
    using pointer         = T *;
    using const_pointer   = T const *;
    using reference       = T &;
    using const_reference = T const &;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(AlignedAllocator<U, Align> const &) noexcept {}

    [[nodiscard]] auto max_size() const noexcept -> size_type { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    auto address(reference x) const noexcept -> pointer { return std::addressof(x); }

    auto address(const_reference x) const noexcept -> const_pointer { return std::addressof(x); }

    auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) -> pointer {
        auto const alignment = static_cast<size_type>(Align);
        void      *ptr       = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { return detail::deallocate_aligned_memory(p); }

    template <class U, class... Args>
    void construct(U *p, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T>
using VectorData = std::vector<T, AlignedAllocator<T, 64>>;
#endif

} // namespace einsums

#if !defined(EINSUMS_WINDOWS)

/**
 * @def EINSUMS_TENSOR_EXPORT_T
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_T(tensortype, type) extern template class EINSUMS_EXPORT tensortype<type>;

/**
 * @def EINSUMS_TENSOR_DEFINE_T
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_T(tensortype, type) template class tensortype<type>;

/**
 * @def EINSUMS_TENSOR_EXPORT_TR
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_TR(tensortype, type, rank) extern template class EINSUMS_EXPORT tensortype<type, rank>;

/**
 * @def EINSUMS_TENSOR_EXPORT_RANK
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_RANK(tensortype, rank)                                                                                   \
        EINSUMS_TENSOR_EXPORT_TR(tensortype, float, rank)                                                                                  \
        EINSUMS_TENSOR_EXPORT_TR(tensortype, double, rank)                                                                                 \
        EINSUMS_TENSOR_EXPORT_TR(tensortype, std::complex<float>, rank)                                                                    \
        EINSUMS_TENSOR_EXPORT_TR(tensortype, std::complex<double>, rank)

/**
 * @def EINSUMS_TENSOR_EXPORT
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define EINSUMS_TENSOR_EXPORT(tensortype)                                                                                              \
        EINSUMS_TENSOR_EXPORT_RANK(tensortype, 1)                                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK(tensortype, 2)                                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK(tensortype, 3)                                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK(tensortype, 4)

/**
 * @def EINSUMS_TENSOR_DEFINE_TR
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_TR(tensortype, type, rank) template class tensortype<type, rank>;

/**
 * @def EINSUMS_TENSOR_DEFINE_RANK
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_RANK(tensortype, rank)                                                                                   \
        EINSUMS_TENSOR_DEFINE_TR(tensortype, float, rank)                                                                                  \
        EINSUMS_TENSOR_DEFINE_TR(tensortype, double, rank)                                                                                 \
        EINSUMS_TENSOR_DEFINE_TR(tensortype, std::complex<float>, rank)                                                                    \
        EINSUMS_TENSOR_DEFINE_TR(tensortype, std::complex<double>, rank)

/**
 * @def EINSUMS_TENSOR_DEFINE
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define EINSUMS_TENSOR_DEFINE(tensortype)                                                                                              \
        EINSUMS_TENSOR_DEFINE_RANK(tensortype, 1)                                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK(tensortype, 2)                                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK(tensortype, 3)                                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK(tensortype, 4)

/**
 * @def EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)                                                          \
        extern template class EINSUMS_EXPORT tensortype<type, view_rank, rank>;

/**
 * @def EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                              \
        EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                             \
        EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                            \
        EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                               \
        EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the base tensor.
 */
#    define EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)                                                                        \
        EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 1, rank)                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 2, rank)                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 3, rank)                                                                          \
        EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def EINSUMS_TENSOR_EXPORT_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define EINSUMS_TENSOR_EXPORT_DISK_VIEW(tensortype)                                                                                    \
        EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 1)                                                                               \
        EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 2)                                                                               \
        EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 3)                                                                               \
        EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 4)

/**
 * @def EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view
 * @param rank The rank of the tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank) template class tensortype<type, view_rank, rank>;

/**
 * @def EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                              \
        EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                             \
        EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                            \
        EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                               \
        EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the base tensor.
 */
#    define EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)                                                                        \
        EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 1, rank)                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 2, rank)                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 3, rank)                                                                          \
        EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def EINSUMS_TENSOR_DEFINE_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define EINSUMS_TENSOR_DEFINE_DISK_VIEW(tensortype)                                                                                    \
        EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 1)                                                                               \
        EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 2)                                                                               \
        EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 3)                                                                               \
        EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 4)

#else

#    define EINSUMS_TENSOR_EXPORT_TR(tensortype, type, rank)
#    define EINSUMS_TENSOR_EXPORT_RANK(tensortype, rank)
#    define EINSUMS_TENSOR_EXPORT(tensortype)
#    define EINSUMS_TENSOR_DEFINE_TR(tensortype, type, rank)
#    define EINSUMS_TENSOR_DEFINE_RANK(tensortype, rank)
#    define EINSUMS_TENSOR_DEFINE(tensortype)
#    define EINSUMS_TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define EINSUMS_TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define EINSUMS_TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)
#    define EINSUMS_TENSOR_EXPORT_DISK_VIEW(tensortype)
#    define EINSUMS_TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define EINSUMS_TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define EINSUMS_TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)
#    define EINSUMS_TENSOR_DEFINE_DISK_VIEW(tensortype)

#endif