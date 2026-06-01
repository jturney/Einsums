//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>

#include <complex>
#include <string>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief Metadata for Scale nodes.
 *
 * Stores the scalar factor applied to the tensor: ``A *= factor``.
 * Used by ScaleAbsorption to fold scale(alpha) into a following op's prefactor.
 */
struct ScaleDescriptor {
    double factor{1.0}; ///< The scaling factor
};

/**
 * @brief Metadata for Permute nodes.
 *
 * Stores the alpha/beta prefactors: C = beta * C + alpha * permute(A).
 */
struct PermuteDescriptor {
    double                   alpha{1.0}; ///< Prefactor for the source tensor
    double                   beta{0.0};  ///< Prefactor for the destination tensor (0 = overwrite)
    std::vector<std::string> c_indices;  ///< Output index names (e.g., {"j","i"})
    std::vector<std::string> a_indices;  ///< Input index names (e.g., {"i","j"})
};

/**
 * @brief Data-type tag for BatchedGemmDescriptor.
 *
 * The descriptor is type-erased for pass introspection; the tag tells
 * the executor which `blas::gemm_batch<T>` variant to dispatch.
 */
enum class BlasScalar : std::uint8_t {
    Float,
    Double,
    ComplexFloat,
    ComplexDouble,
};

/**
 * @brief Metadata for BatchedGemm nodes produced by the GEMMBatching pass.
 *
 * A BatchedGemm collapses N independent Einsum nodes (each expressing
 * a rank-2 × rank-2 → rank-2 contraction with one link index and
 * matching M/N/K dimensions, alpha/beta prefactors, trans flags, and
 * data type) into a single `blas::gemm_batch` call. The @p inputs and
 * @p outputs on the parent @ref Node store the full 2N inputs (A_0,
 * B_0, A_1, B_1, …) and N outputs (C_0, …) in the original group
 * order; this descriptor carries the shared BLAS parameters.
 */
struct BatchedGemmDescriptor {
    int                  m{0};            ///< Rows of each C (and A if trans_a == 'N').
    int                  n{0};            ///< Cols of each C (and B if trans_b == 'N').
    int                  k{0};            ///< Link dimension.
    int                  lda{0};          ///< Leading dim of each A (row-major stride).
    int                  ldb{0};          ///< Leading dim of each B.
    int                  ldc{0};          ///< Leading dim of each C.
    char                 trans_a{'N'};    ///< BLAS transpose flag for A ('N' or 'T').
    char                 trans_b{'N'};    ///< BLAS transpose flag for B.
    std::complex<double> alpha{1.0, 0.0}; ///< A*B prefactor (full complex; imag part used for complex tensors).
    std::complex<double> beta{0.0, 0.0};  ///< C prefactor.
    int                  batch_count{0};  ///< Number of GEMMs fused into this call.
    BlasScalar           scalar{BlasScalar::Double};

    /// Strided-batched mode: when true, the batched executor reads a
    /// single base pointer per operand from the live slot and computes
    /// each matrix pointer as `base + i * batch_stride_* * sizeof(T)`.
    /// Matches the layout `cublasDgemmStridedBatched` requires on GPU.
    /// When false, the executor stores N per-slice extractors (the
    /// output of the GEMMBatching pass over independent 2D einsums).
    bool strided{false};

    /// Number of elements between consecutive batch slices of each
    /// operand. Only meaningful when @ref strided is true; for a 3D
    /// tensor of shape (B, M, K) with batch at axis 0, this is M*K.
    std::int64_t batch_stride_a{0};
    std::int64_t batch_stride_b{0};
    std::int64_t batch_stride_c{0};
};

/**
 * @brief Metadata for memory allocation/deallocation nodes.
 *
 * Marks the lifetime boundaries of a tensor in the graph. Used by
 * the MemoryPlanning pass to identify buffer reuse opportunities.
 * The actual allocation is managed by the graph (via ``owned_tensors_``).
 */
struct AllocDescriptor {
    TensorId    tensor_id{0};  ///< Which tensor this alloc/free refers to
    size_t      size_bytes{0}; ///< Size of the allocation in bytes
    std::string tensor_name;   ///< Name for debugging
};

/**
 * @brief Metadata for GPU memory transfer nodes (HostToDevice / DeviceToHost).
 *
 * Inserted by TransferInsertion and pruned by TransferElimination.
 * The executor lambda performs the actual gpu::memcpy_* call.
 */
struct TransferDescriptor {
    TensorId tensor_id{0};  ///< Which tensor is being transferred
    size_t   size_bytes{0}; ///< Number of bytes to transfer
};

/**
 * @brief Metadata for disk I/O nodes (DiskRead / DiskWrite).
 *
 * Stores the file path and dataset name for tensor serialization.
 */
struct DiskIODescriptor {
    std::string file_path;    ///< Path to the file (HDF5, binary, etc.)
    std::string dataset_name; ///< Dataset/key name within the file
    TensorId    tensor_id{0}; ///< Which tensor is being read/written
    size_t      size_bytes{0};
};

/**
 * @brief Metadata for Initialize nodes (zero fill, random fill, disk load).
 */
struct InitializeDescriptor {
    TensorId    tensor_id{0};
    InitKind    kind{InitKind::Zero};
    std::string source_path; ///< File path for FromDisk initialization
};

/**
 * @brief Metadata for distributed communication nodes (Allreduce, Broadcast, etc.).
 */
struct CommDescriptor {
    TensorId tensor_id{0};    ///< Tensor being communicated
    size_t   size_bytes{0};   ///< Size of the data in bytes
    int      root{0};         ///< Root rank (for Broadcast/Scatter)
    bool     use_nccl{false}; ///< True if tensor is GPU-resident and NCCL available
};

} // namespace einsums::compute_graph
