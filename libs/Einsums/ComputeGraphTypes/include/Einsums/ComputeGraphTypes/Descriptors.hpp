//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <complex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// Only the descriptors expressible at this tier: inert data over Enums.hpp, Ids.hpp and std types.
// Those that need a type from further up (Einsum, Axpby, Scale, Permute, ElementwiseBinary,
// Loop/Conditional, View) live in `Einsums/ComputeGraph/Node.hpp`.

/**
 * @brief Data-type tag for BatchedGemmDescriptor.
 *
 * Tells the executor which `blas::gemm_batch<T>` to call.
 */
enum class BlasScalar : std::uint8_t {
    Float,
    Double,
    ComplexFloat,
    ComplexDouble,
};

/**
 * @brief The @ref BlasScalar tag naming @p T.
 *
 * @p T is one of the four types `blas::gemm_batch` accepts.
 */
template <typename T>
constexpr BlasScalar blas_scalar_of() {
    if constexpr (std::is_same_v<T, float>) {
        return BlasScalar::Float;
    } else if constexpr (std::is_same_v<T, double>) {
        return BlasScalar::Double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return BlasScalar::ComplexFloat;
    } else {
        static_assert(std::is_same_v<T, std::complex<double>>, "blas_scalar_of: not a BLAS element type");
        return BlasScalar::ComplexDouble;
    }
}

/**
 * @brief Metadata for BatchedGemm nodes produced by the GEMMBatching pass.
 *
 * N independent matrix-matrix Einsums agreeing on shape, prefactors, trans flags and type, as one
 * `blas::gemm_batch` call. The node holds the 2N inputs (A_0, B_0, A_1, ...) and N outputs in
 * order; this holds the shared parameters.
 */
struct BatchedGemmDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "BatchedGemmDescriptor";
    int                               m{0};            ///< Rows of each C (and A if trans_a == 'N').
    int                               n{0};            ///< Cols of each C (and B if trans_b == 'N').
    int                               k{0};            ///< Link dimension.
    int                               lda{0};          ///< Leading dim of each A (row-major stride).
    int                               ldb{0};          ///< Leading dim of each B.
    int                               ldc{0};          ///< Leading dim of each C.
    char                              trans_a{'N'};    ///< BLAS transpose flag for A ('N' or 'T').
    char                              trans_b{'N'};    ///< BLAS transpose flag for B.
    std::complex<double>              alpha{1.0, 0.0}; ///< A*B prefactor (full complex; imag part used for complex tensors).
    std::complex<double>              beta{0.0, 0.0};  ///< C prefactor.
    int                               batch_count{0};  ///< Number of GEMMs fused into this call.
    BlasScalar                        scalar{BlasScalar::Double};
};

/**
 * @brief One shape class inside a @ref GroupedBatchedGemmDescriptor.
 *
 * The per-group counterpart of what @ref BatchedGemmDescriptor holds once.
 */
struct GemmGroup {
    int                  m{0};            ///< Rows of each C in this group (and of op(A)).
    int                  n{0};            ///< Cols of each C in this group (and of op(B)).
    int                  k{0};            ///< Link dimension.
    int                  lda{0};          ///< Leading dim of each A in this group.
    int                  ldb{0};          ///< Leading dim of each B.
    int                  ldc{0};          ///< Leading dim of each C.
    char                 trans_a{'N'};    ///< BLAS transpose flag for A ('N', 'T' or 'C').
    char                 trans_b{'N'};    ///< BLAS transpose flag for B.
    std::complex<double> alpha{1.0, 0.0}; ///< A*B prefactor (full complex; imag used for complex tensors).
    std::complex<double> beta{0.0, 0.0};  ///< C prefactor.
    int                  count{0};        ///< How many GEMMs this group holds.

    /// Where this group's members start in the node's flattened operand lists.
    int first{0};
};

/**
 * @brief Metadata for GroupedBatchedGemm nodes.
 *
 * Several differently shaped batches under one OpenMP region, which costs tens of microseconds to
 * enter (a DLPNO-MP2 iteration spent 45 ms on 16 ms of arithmetic across 754 batches). The node
 * holds the 2N inputs and the outputs in group order, indexed by @ref GemmGroup::first. @ref labels
 * names the groups so `einsums:graph:profile-groups` can time each.
 */
struct GroupedBatchedGemmDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "GroupedBatchedGemmDescriptor";
    std::vector<GemmGroup>            groups;   ///< One entry per shape class, in operand order.
    int                               total{0}; ///< Sum of every group's count.
    BlasScalar                        scalar{BlasScalar::Double};

    /// Human-readable name per group, parallel to @ref groups. Shape-derived
    /// when the capture API grouped the batch itself.
    std::vector<std::string> labels;

    /// Whether each member writes a block of a shared base, at an offset only the executor holds.
    /// The outputs then list each distinct base, so readers that map members to outputs (region
    /// rewrites) must decline a blocked batch.
    bool blocked{false};
};

/**
 * @brief Metadata for memory allocation/deallocation nodes.
 *
 * A tensor's lifetime boundaries, for MemoryPlanning; the graph owns the storage.
 */
struct AllocDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "AllocDescriptor";
    TensorId                          tensor_id{0};  ///< Which tensor this alloc/free refers to
    size_t                            size_bytes{0}; ///< Size of the allocation in bytes
    std::string                       tensor_name;   ///< Name for debugging
};

/**
 * @brief Metadata for GPU memory transfer nodes (HostToDevice / DeviceToHost).
 *
 * Inserted by TransferInsertion and pruned by TransferElimination.
 * The executor lambda performs the actual gpu::memcpy_* call.
 */
struct TransferDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "TransferDescriptor";
    TensorId                          tensor_id{0};  ///< Which tensor is being transferred
    size_t                            size_bytes{0}; ///< Number of bytes to transfer
};

/**
 * @brief Metadata for disk I/O nodes (DiskRead / DiskWrite).
 *
 * Stores the file path and dataset name for tensor serialization.
 */
struct DiskIODescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "DiskIODescriptor";
    std::string                       file_path;    ///< Path to the file (HDF5, binary, etc.)
    std::string                       dataset_name; ///< Dataset/key name within the file
    TensorId                          tensor_id{0}; ///< Which tensor is being read/written
    size_t                            size_bytes{0};
};

/**
 * @brief Metadata for Initialize nodes (zero fill, random fill, disk load).
 */
struct InitializeDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "InitializeDescriptor";
    TensorId                          tensor_id{0};
    InitKind                          kind{InitKind::Zero};
    std::string                       source_path; ///< File path for FromDisk initialization
};

/**
 * @brief Metadata for distributed communication nodes (Allreduce, Broadcast, etc.).
 */
struct CommDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "CommDescriptor";
    TensorId                          tensor_id{0};    ///< Tensor being communicated
    size_t                            size_bytes{0};   ///< Size of the data in bytes
    int                               root{0};         ///< Root rank (for Broadcast/Scatter)
    bool                              use_nccl{false}; ///< True if tensor is GPU-resident and NCCL available
};

/**
 * @brief Metadata for @ref OpKind::ElementTransform nodes whose kernel is NAMED.
 *
 * Applies the kernel registered under @ref op_name to every element of the node's one output (also
 * its one input). Lambda overloads record the kind with no descriptor, which cannot be saved;
 * ``Graph::serializability_report`` names them.
 */
struct ElementTransformDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "ElementTransformDescriptor";
    /// Name of the kernel in the process's element-op registry.
    std::string op_name;

    /// The parameter of a parameterized kernel (e.g. a drop threshold). Empty means the default its
    /// registration documents, so files without the field still read.
    std::optional<double> param;
};

EINSUMS_NAMESPACE_END(compute_graph)
