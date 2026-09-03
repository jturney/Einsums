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
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// NOTE: this header holds only the descriptors expressible in the
// ComputeGraphTypes tier -- inert data over Enums.hpp / Ids.hpp and std types,
// with concrete scalars (`double`, `std::complex<double>`).
//
// Descriptors that reference a type defined further up the stack live in
// `Einsums/ComputeGraph/Node.hpp` instead, and are NOT duplicated here:
//
//   EinsumDescriptor      needs packed_gemm::ContractionSpec (PackedGemm)
//   AxpbyDescriptor       needs PrefactorScalar, plus a shared_ptr<AxpbyParams>
//                         handle into live execution state
//   ScaleDescriptor       same: PrefactorScalar plus a live params handle
//   PermuteDescriptor     same: a live params handle
//   ElementwiseBinaryDescriptor  same (DirectProduct / DirectDivision)
//   Loop/ConditionalDescriptor  hold shared_ptr<Graph> and std::function
//   ViewDescriptor        needs ViewAxis
//
// So a descriptor missing from this file has probably not been written yet --
// check Node.hpp before concluding it does not exist.
//
// ScaleDescriptor and PermuteDescriptor USED to live here, on concrete
// `double` / `std::complex<double>` scalars. They moved when their executors
// started reading their prefactors from a shared, pass-rewritable params block
// (see ExecutorBuilder.hpp): that handle is a `shared_ptr` to a type holding
// PrefactorScalar, which is exactly the "further up the stack" condition above.

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
 * @brief The @ref BlasScalar tag naming @p T.
 *
 * Every producer of a descriptor carrying a `scalar` field has to answer the
 * same question, and answering it in place invites the four branches to drift
 * apart. @p T is one of the four types `blas::gemm_batch` accepts.
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
        return BlasScalar::ComplexDouble;
    }
}

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
    ///
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
 * @brief One shape class inside a @ref GroupedBatchedGemmDescriptor.
 *
 * Everything @ref BatchedGemmDescriptor holds once for a whole call is held
 * here per group, because that is exactly the difference between the two
 * nodes: a grouped call is several uniform batches issued under one OpenMP
 * region.
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
 * A GroupedBatchedGemm is a @ref BatchedGemmDescriptor that stopped insisting
 * every member agree on shape. It exists because entering an OpenMP region
 * costs tens of microseconds on a wide team, and a dependency level holding
 * many differently shaped batches used to pay that once per shape: measured on
 * a DLPNO-MP2 iteration, 754 batched calls whose arithmetic wanted 16 ms spent
 * 45. Collapsing them into one call made the time track the arithmetic again.
 *
 * The parent @ref Node stores the full 2N inputs (A_0, B_0, A_1, B_1, ...) and
 * the outputs in group order, so @ref GemmGroup::first indexes both.
 *
 * On observability. One node in place of many is one timing row in place of
 * many, and the per-shape rows are what made the investigations that produced
 * this node possible in the first place. So @ref labels names every group, and
 * the executor can be asked to time them individually; see
 * `einsums:graph:profile-groups`.
 */
struct GroupedBatchedGemmDescriptor {
    std::vector<GemmGroup> groups;   ///< One entry per shape class, in operand order.
    int                    total{0}; ///< Sum of every group's count.
    BlasScalar             scalar{BlasScalar::Double};

    /// Human-readable name per group, parallel to @ref groups. Shape-derived
    /// when the capture API grouped the batch itself.
    std::vector<std::string> labels;
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

/**
 * @brief Metadata for @ref OpKind::ElementTransform nodes whose kernel is NAMED.
 *
 * A named element transform applies the kernel registered under @ref op_name to
 * every element of its destination, which the parent @ref Node lists as its one
 * output (and, since the operation is a read-modify-write, as its one input).
 *
 * The lambda-taking ``cg::element_transform`` overloads record the same kind
 * with NO descriptor, because a closure is precisely what a descriptor cannot
 * hold. Both remain fully legal; only the named form can be written to a file,
 * and ``Graph::serializability_report`` names the anonymous ones individually
 * with the fix in the message.
 *
 * A bare string, so it belongs in this tier: the registry the name resolves
 * against lives further up the stack, but resolving it is the BUILDER's job and
 * nothing about the recorded node needs to know the registry exists.
 */
struct ElementTransformDescriptor {
    /// Name of the kernel in the process's element-op registry.
    std::string op_name;

    /// The policy number a PARAMETERIZED kernel is applied with, e.g. the drop
    /// threshold a guarded inverse square root compares against.
    ///
    /// Empty for an op that takes no parameter, and empty for a parameterized op
    /// the capture site said nothing about, which then runs with the default its
    /// registration documents. That is also how a file written before this field
    /// existed reads, which is the whole reason the default is part of the
    /// REGISTRATION rather than of the capture site: an absent key has to mean
    /// something a reader can look up.
    std::optional<double> param;
};

EINSUMS_NAMESPACE_END(compute_graph)
