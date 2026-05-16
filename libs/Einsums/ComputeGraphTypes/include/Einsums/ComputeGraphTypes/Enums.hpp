//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <string_view>

namespace einsums {
namespace EINSUMS_PYBIND_MODULE("graph") compute_graph {

/**
 * @brief Identifies the kind of operation a node represents.
 *
 * Used by optimization passes for pattern matching. For example,
 * ScaleAbsorption looks for OpKind::Scale followed by a prefactor-bearing op.
 *
 * Categories:
 * - **TensorAlgebra**: Einsum, Permute, Transpose, ElementTransform, KhatriRao
 * - **BLAS-level**: Gemm, Gemv, Ger, Dot, Scale, Axpy, Axpby, DirectProduct
 * - **LAPACK-level**: SVD, QR, Syev, Heev, Geev, Gesv, Invert, Det, Pow, etc.
 * - **Other**: HPTTPermute, Custom
 */
enum class EINSUMS_PYBIND_EXPOSE OpKind : std::uint8_t {
    // TensorAlgebra operations
    Einsum,           ///< Tensor contraction via tensor_algebra::einsum()
    Permute,          ///< Index reordering via tensor_algebra::permute()
    Transpose,        ///< 2D transpose via tensor_algebra::transpose()
    ElementTransform, ///< Element-wise unary transform
    KhatriRao,        ///< Khatri-Rao product

    // LinearAlgebra - BLAS level
    BatchedGemm,   ///< Many independent GEMMs in one `gemm_batch` call
    Gemm,          ///< General matrix-matrix multiply (BLAS Level 3)
    Gemv,          ///< General matrix-vector multiply (BLAS Level 2)
    Ger,           ///< Rank-1 update (BLAS Level 2)
    Dot,           ///< Dot product (returns scalar)
    Scale,         ///< Scalar multiplication of entire tensor
    Axpy,          ///< Y += alpha * X
    Axpby,         ///< Y = alpha * X + beta * Y
    DirectProduct, ///< Element-wise (Hadamard) product

    // LinearAlgebra - LAPACK level
    SVD,           ///< Singular value decomposition
    SVD_DD,        ///< SVD with divide-and-conquer algorithm
    TruncatedSVD,  ///< Truncated SVD (keeping k singular values)
    QR,            ///< QR decomposition
    Syev,          ///< Symmetric eigendecomposition
    Heev,          ///< Hermitian eigendecomposition
    Geev,          ///< General eigendecomposition
    TruncatedSyev, ///< Truncated symmetric eigendecomposition
    Gesv,          ///< General linear system solver (AX = B)
    Getrf,         ///< LU factorization
    Getri,         ///< Inverse from LU factorization
    Invert,        ///< Matrix inverse
    Pseudoinverse, ///< Moore-Penrose pseudoinverse
    Det,           ///< Matrix determinant (returns scalar)
    Pow,           ///< Matrix power
    SymmGemm,      ///< Symmetric double multiply: C = B^T * A * B
    Norm,          ///< Tensor norm (returns scalar)
    SolveLyapunov, ///< Continuous Lyapunov equation solver

    // HPTT
    HPTTPermute, ///< High-performance tensor transpose

    // Control flow
    Conditional, ///< If-then-else branch with subgraphs
    Loop,        ///< While/for loop with body subgraph

    // Memory management
    Alloc, ///< Tensor allocation (marks lifetime start)
    Free,  ///< Tensor deallocation (marks lifetime end)

    // GPU memory transfers
    HostToDevice, ///< Transfer tensor from host to device memory
    DeviceToHost, ///< Transfer tensor from device to host memory

    // TaskPool data-parallel operations
    ParallelFor,    ///< Data-parallel for loop (delegates to TaskPool)
    ParallelReduce, ///< Data-parallel reduce (delegates to TaskPool)

    // Disk I/O
    DiskRead,  ///< Read tensor data from disk (HDF5, binary, etc.)
    DiskWrite, ///< Write tensor data to disk (checkpointing)

    // Deferred allocation
    Materialize, ///< Allocate storage for a deferred (shell) tensor
    Initialize,  ///< Fill tensor with initial values (zero, random, disk)

    // Aliasing / dataflow
    View,       ///< Non-owning slice/view of another tensor (zero-copy alias)
    WriteParam, ///< Write the value of a scalar tensor into a Pipeline parameter
    Trace,      ///< Diagonal sum of a square rank-2 tensor (returns scalar)

    // Distributed communication
    Allreduce, ///< Sum partial results across MPI ranks
    Broadcast, ///< Root sends data to all ranks
    Allgather, ///< Each rank contributes a piece, all receive the whole
    Scatter,   ///< Root distributes pieces to ranks
    Barrier,   ///< Synchronization point across ranks

    // User-defined
    Custom, ///< User-registered custom operation
};

/**
 * @brief Where a node should execute.
 *
 * Set by the GPUPlacement pass. Defaults to CPU; nodes promoted to GPU
 * will have their executors replaced with GPU-dispatching versions.
 */
enum class Target : uint8_t {
    CPU, ///< Execute on the host (default)
    GPU, ///< Execute on a GPU device
};

/// Allocation state of a tensor in the graph.
enum class AllocState : std::uint8_t {
    Materialized, ///< Normal: data allocated and ready
    Deferred,     ///< Shell tensor: dims known, no data until MaterializationPass
};

/// Ownership level of a tensor (determines declaration scope in code generation).
enum class TensorOwnership : std::uint8_t {
    Graph,     ///< Intermediate, scoped to a single graph stage
    Pipeline,  ///< Shared across stages within a pipeline
    Workspace, ///< Shared across pipelines within a workspace
};

/// Initialization strategy for deferred tensors.
enum class InitKind : std::uint8_t {
    None,     ///< No automatic initialization
    Zero,     ///< Fill with zeros after materialization
    Random,   ///< Fill with random values after materialization
    FromDisk, ///< Load from file after materialization
};

/**
 * @brief Where a tensor's data currently resides.
 *
 * Tracked per-tensor by the GPU optimization passes.
 */
enum class Residency : std::uint8_t {
    Host,    ///< Data is on the CPU (default for all tensors)
    Device,  ///< Data is on the GPU
    Both,    ///< Valid copies exist on both host and device
    Unknown, ///< Residency has not been determined yet
};

/**
 * @brief Convert an OpKind enum value to its string name.
 * @param[in] kind The operation kind.
 * @return A string_view (e.g., "Einsum", "Scale", "Gemm").
 */
inline std::string_view op_kind_name(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
        return "Einsum";
    case OpKind::Permute:
        return "Permute";
    case OpKind::Transpose:
        return "Transpose";
    case OpKind::ElementTransform:
        return "ElementTransform";
    case OpKind::KhatriRao:
        return "KhatriRao";
    case OpKind::BatchedGemm:
        return "BatchedGemm";
    case OpKind::Gemm:
        return "Gemm";
    case OpKind::Gemv:
        return "Gemv";
    case OpKind::Ger:
        return "Ger";
    case OpKind::Dot:
        return "Dot";
    case OpKind::Scale:
        return "Scale";
    case OpKind::Axpy:
        return "Axpy";
    case OpKind::Axpby:
        return "Axpby";
    case OpKind::DirectProduct:
        return "DirectProduct";
    case OpKind::SVD:
        return "SVD";
    case OpKind::SVD_DD:
        return "SVD_DD";
    case OpKind::TruncatedSVD:
        return "TruncatedSVD";
    case OpKind::QR:
        return "QR";
    case OpKind::Syev:
        return "Syev";
    case OpKind::Heev:
        return "Heev";
    case OpKind::Geev:
        return "Geev";
    case OpKind::TruncatedSyev:
        return "TruncatedSyev";
    case OpKind::Gesv:
        return "Gesv";
    case OpKind::Getrf:
        return "Getrf";
    case OpKind::Getri:
        return "Getri";
    case OpKind::Invert:
        return "Invert";
    case OpKind::Pseudoinverse:
        return "Pseudoinverse";
    case OpKind::Det:
        return "Det";
    case OpKind::Pow:
        return "Pow";
    case OpKind::SymmGemm:
        return "SymmGemm";
    case OpKind::Norm:
        return "Norm";
    case OpKind::SolveLyapunov:
        return "SolveLyapunov";
    case OpKind::HPTTPermute:
        return "HPTTPermute";
    case OpKind::Conditional:
        return "Conditional";
    case OpKind::Loop:
        return "Loop";
    case OpKind::Alloc:
        return "Alloc";
    case OpKind::Free:
        return "Free";
    case OpKind::ParallelFor:
        return "ParallelFor";
    case OpKind::ParallelReduce:
        return "ParallelReduce";
    case OpKind::HostToDevice:
        return "HostToDevice";
    case OpKind::DeviceToHost:
        return "DeviceToHost";
    case OpKind::DiskRead:
        return "DiskRead";
    case OpKind::DiskWrite:
        return "DiskWrite";
    case OpKind::Materialize:
        return "Materialize";
    case OpKind::Initialize:
        return "Initialize";
    case OpKind::View:
        return "View";
    case OpKind::WriteParam:
        return "WriteParam";
    case OpKind::Trace:
        return "Trace";
    case OpKind::Allreduce:
        return "Allreduce";
    case OpKind::Broadcast:
        return "Broadcast";
    case OpKind::Allgather:
        return "Allgather";
    case OpKind::Scatter:
        return "Scatter";
    case OpKind::Barrier:
        return "Barrier";
    case OpKind::Custom:
        return "Custom";
    }
    return "Unknown";
}

} // namespace EINSUMS_PYBIND_MODULE("graph")compute_graph
} // namespace einsums
