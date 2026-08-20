//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/DeviceVector.hpp>

#include <cstddef>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

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

// Forward declarations of tensors.
template <typename T, size_t Rank, typename Alloc>
struct GeneralTensor;

/// The standard in-core tensor: GeneralTensor with the standard allocator.
/// The data is stored contiguously in memory.
template <typename T, size_t Rank>
using Tensor = GeneralTensor<T, Rank, std::allocator<T>>;

template <typename T, size_t Rank>
using BufferTensor = GeneralTensor<T, Rank, BufferAllocator<T>>;

/// GPU-resident tensor using the gpu:: abstraction layer.
/// On the mock backend, with no GPU, it uses std::malloc with the same
/// behavior, so it is testable anywhere.
template <typename T, size_t Rank>
using GPUTensor = GeneralTensor<T, Rank, gpu::DeviceAllocator<T>>;

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct TiledTensor;

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t Rank>
struct TiledTensorView;

template <typename T, size_t Rank>
struct DiskView;

template <typename T, size_t Rank>
struct DiskTensor;

template <typename T, typename Alloc>
struct GeneralRuntimeTensor;

/// Runtime-rank in-core tensor: GeneralRuntimeTensor with the standard
/// allocator. Mostly used for communication with the Python interface.
template <typename T>
using RuntimeTensor = GeneralRuntimeTensor<T, std::allocator<T>>;

template <typename T>
using BufferRuntimeTensor = GeneralRuntimeTensor<T, BufferAllocator<T>>;

template <typename T>
struct TiledRuntimeTensor;

/// GPU-resident runtime-rank tensor using the gpu:: abstraction layer.
///
/// Mirrors GPUTensor but with rank known only at runtime. It is not exposed to
/// Python. The ComputeGraph optimization passes, such as GPUPlacement, own
/// the host-to-device decision; users program in Python as if everything
/// runs on a single host. On the mock backend, with no GPU, it uses std::malloc.
template <typename T>
using RuntimeGPUTensor = GeneralRuntimeTensor<T, gpu::DeviceAllocator<T>>;

template <typename T>
struct RuntimeTensorView;

template <typename T>
using VectorData = BufferVector<T>;

/**
 * @typedef ShapeVector
 *
 * @brief Storage for a tensor's dimensions, strides and index coordinates.
 *
 * Deliberately NOT @c BufferVector, and the distinction is the point.
 * @c BufferAllocator enforces a global ceiling (@c --einsums:buffer-size,
 * default 64MB) whose purpose is to bound transient contraction WORKSPACE, so
 * that out-of-core algorithms can size their chunks against a declared budget
 * and a runaway temporary throws a named error instead of exhausting the
 * machine. LAPACK's @c work and @c iwork arrays are the model consumer, and
 * they still use @c BufferVector.
 *
 * Shape metadata is a different population. Every tensor and every VIEW holds
 * dims and strides, so the total scales with how many tensors are ALIVE rather
 * than with how large they are - and a deferred-execution graph manufactures
 * views by the hundred thousand and keeps them for the life of the graph.
 * Charging those against the workspace ceiling let metadata crowd out the
 * workspace the ceiling exists to bound: a captured DLPNO-CCSD iteration
 * (212,000 nodes at ethanol/cc-pVTZ) exhausted the then-4MB default outright, and
 * every coupled-cluster entry point had to raise it by two or three orders of
 * magnitude to run at all. The failures read as leaks and were not; the
 * accounting was balanced, the population was simply the wrong one.
 *
 * @versionadded{1.1.1}
 */
template <typename T>
using ShapeVector = std::vector<T>;

EINSUMS_NAMESPACE_END()

#if !defined(EINSUMS_WINDOWS)
/**
 * @def TENSOR_EXPORT_TR
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_TR(tensortype, type, rank) extern template struct EINSUMS_EXPORT tensortype<type, rank>;

/**
 * @def TENSOR_EXPORT_RANK
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_RANK(tensortype, rank)                                                                                           \
        TENSOR_EXPORT_TR(tensortype, float, rank)                                                                                          \
        TENSOR_EXPORT_TR(tensortype, double, rank)                                                                                         \
        TENSOR_EXPORT_TR(tensortype, std::complex<float>, rank)                                                                            \
        TENSOR_EXPORT_TR(tensortype, std::complex<double>, rank)

/**
 * @def TENSOR_EXPORT
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define TENSOR_EXPORT(tensortype)                                                                                                      \
        TENSOR_EXPORT_RANK(tensortype, 1)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 2)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 3)                                                                                                  \
        TENSOR_EXPORT_RANK(tensortype, 4)

/**
 * @def TENSOR_DEFINE_TR
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_TR(tensortype, type, rank) template struct tensortype<type, rank>;

/**
 * @def TENSOR_DEFINE_RANK
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_RANK(tensortype, rank)                                                                                           \
        TENSOR_DEFINE_TR(tensortype, float, rank)                                                                                          \
        TENSOR_DEFINE_TR(tensortype, double, rank)                                                                                         \
        TENSOR_DEFINE_TR(tensortype, std::complex<float>, rank)                                                                            \
        TENSOR_DEFINE_TR(tensortype, std::complex<double>, rank)

/**
 * @def TENSOR_DEFINE
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define TENSOR_DEFINE(tensortype)                                                                                                      \
        TENSOR_DEFINE_RANK(tensortype, 1)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 2)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 3)                                                                                                  \
        TENSOR_DEFINE_RANK(tensortype, 4)

/**
 * @def TENSOR_EXPORT_ALLOC_TR
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_ALLOC_TR(tensortype, type, rank, alloc) extern template struct EINSUMS_EXPORT tensortype<type, rank, alloc<type>>;

/**
 * @def TENSOR_EXPORT_ALLOC_RANK
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_EXPORT_ALLOC_RANK(tensortype, rank, alloc)                                                                              \
        TENSOR_EXPORT_ALLOC_TR(tensortype, float, rank, alloc)                                                                             \
        TENSOR_EXPORT_ALLOC_TR(tensortype, double, rank, alloc)                                                                            \
        TENSOR_EXPORT_ALLOC_TR(tensortype, std::complex<float>, rank, alloc)                                                               \
        TENSOR_EXPORT_ALLOC_TR(tensortype, std::complex<double>, rank, alloc)

/**
 * @def TENSOR_ALLOC_EXPORT
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define TENSOR_ALLOC_EXPORT(tensortype, alloc)                                                                                         \
        TENSOR_EXPORT_ALLOC_RANK(tensortype, 1, alloc)                                                                                     \
        TENSOR_EXPORT_ALLOC_RANK(tensortype, 2, alloc)                                                                                     \
        TENSOR_EXPORT_ALLOC_RANK(tensortype, 3, alloc)                                                                                     \
        TENSOR_EXPORT_ALLOC_RANK(tensortype, 4, alloc)

/**
 * @def TENSOR_DEFINE_TR
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_ALLOC_TR(tensortype, type, rank, alloc) template struct tensortype<type, rank, alloc<type>>;

/**
 * @def TENSOR_DEFINE_RANK
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_ALLOC_RANK(tensortype, rank, alloc)                                                                              \
        TENSOR_DEFINE_ALLOC_TR(tensortype, float, rank, alloc)                                                                             \
        TENSOR_DEFINE_ALLOC_TR(tensortype, double, rank, alloc)                                                                            \
        TENSOR_DEFINE_ALLOC_TR(tensortype, std::complex<float>, rank, alloc)                                                               \
        TENSOR_DEFINE_ALLOC_TR(tensortype, std::complex<double>, rank, alloc)

/**
 * @def TENSOR_DEFINE
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define TENSOR_ALLOC_DEFINE(tensortype, alloc)                                                                                         \
        TENSOR_DEFINE_ALLOC_RANK(tensortype, 1, alloc)                                                                                     \
        TENSOR_DEFINE_ALLOC_RANK(tensortype, 2, alloc)                                                                                     \
        TENSOR_DEFINE_ALLOC_RANK(tensortype, 3, alloc)                                                                                     \
        TENSOR_DEFINE_ALLOC_RANK(tensortype, 4, alloc)

/**
 * @def TENSOR_EXPORT_TR_DISK_VIEW
 *
 * Creates an exported template declaration for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to declare.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)                                                                  \
        extern template struct EINSUMS_EXPORT tensortype<type, view_rank, rank>;

/**
 * @def TENSOR_EXPORT_RANK_DISK_VIEW
 *
 * Creates exported template declarations for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to declare.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                                      \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                                     \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                                    \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                                       \
        TENSOR_EXPORT_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def TENSOR_EXPORT_RANK2_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)                                                                                \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 1, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 2, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 3, rank)                                                                                  \
        TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def TENSOR_EXPORT_DISK_VIEW
 *
 * Creates exported template declarations for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to declare.
 */
#    define TENSOR_EXPORT_DISK_VIEW(tensortype)                                                                                            \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 1)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 2)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 3)                                                                                       \
        TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, 4)

/**
 * @def TENSOR_DEFINE_TR_DISK_VIEW
 *
 * Creates an exported template definition for a tensor with the given type and rank.
 *
 * @param tensortype The kind of tensor to define.
 * @param type The type held by that tensor.
 * @param view_rank The rank of the view
 * @param rank The rank of the tensor.
 */
#    define TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank) template struct tensortype<type, view_rank, rank>;

/**
 * @def TENSOR_DEFINE_RANK_DISK_VIEW
 *
 * Creates exported template definitions for a tensor with the given rank, and for each stored type from
 * @c float , @c double , @c std::complex<float> , and @c std::complex<double> .
 *
 * @param tensortype The type of tensor to define.
 * @param view_rank The rank of the view.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)                                                                      \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, float, view_rank, rank)                                                                     \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, double, view_rank, rank)                                                                    \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<float>, view_rank, rank)                                                       \
        TENSOR_DEFINE_TR_DISK_VIEW(tensortype, std::complex<double>, view_rank, rank)

/**
 * @def TENSOR_DEFINE_RANK2_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 * @param rank The rank of the base tensor.
 */
#    define TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)                                                                                \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 1, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 2, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 3, rank)                                                                                  \
        TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, 4, rank)

/**
 * @def TENSOR_DEFINE_DISK_VIEW
 *
 * Creates exported template definitions for a tensor for each stored type from @c float , @c double ,
 * @c std::complex<float> , and @c std::complex<double> , and for all ranks and view ranks between 1 and 4 inclusive.
 *
 * @param tensortype The type of tensor to define.
 */
#    define TENSOR_DEFINE_DISK_VIEW(tensortype)                                                                                            \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 1)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 2)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 3)                                                                                       \
        TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, 4)

#else

#    define TENSOR_EXPORT_TR(tensortype, type, rank)
#    define TENSOR_EXPORT_RANK(tensortype, rank)
#    define TENSOR_EXPORT(tensortype)
#    define TENSOR_DEFINE_TR(tensortype, type, rank)
#    define TENSOR_DEFINE_RANK(tensortype, rank)
#    define TENSOR_DEFINE(tensortype)
#    define TENSOR_EXPORT_ALLOC_TR(tensortype, type, rank, alloc)
#    define TENSOR_EXPORT_ALLOC_RANK(tensortype, rank, alloc)
#    define TENSOR_ALLOC_EXPORT(tensortype, alloc)
#    define TENSOR_DEFINE_ALLOC_TR(tensortype, type, rank, alloc)
#    define TENSOR_DEFINE_ALLOC_RANK(tensortype, rank, alloc)
#    define TENSOR_ALLOC_DEFINE(tensortype, alloc)
#    define TENSOR_EXPORT_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define TENSOR_EXPORT_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define TENSOR_EXPORT_RANK2_DISK_VIEW(tensortype, rank)
#    define TENSOR_EXPORT_DISK_VIEW(tensortype)
#    define TENSOR_DEFINE_TR_DISK_VIEW(tensortype, type, view_rank, rank)
#    define TENSOR_DEFINE_RANK_DISK_VIEW(tensortype, view_rank, rank)
#    define TENSOR_DEFINE_RANK2_DISK_VIEW(tensortype, rank)
#    define TENSOR_DEFINE_DISK_VIEW(tensortype)

#endif