//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraphTypes/EnumNames.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

class GraphAnchor;

/**
 * @brief Where a lifecycle node's tensor is declared, when that is a graph other than the node's.
 *
 * A loop body's scratch is materialized in the parent, and a parent's tensor can be materialized
 * inside a setup body. The node's own graph then holds only a copy of the tensor's handle,
 * registered when the tensor was first named there, and the copy need not carry the hooks the
 * declaration does. The node's executor therefore reads the declaration, through the declaring
 * graph's anchor, which follows that graph through moves.
 *
 * Held only in memory: the kinds that carry one are placed by the resource phase, which runs
 * again after a load, so no saved graph holds one.
 *
 * @versionadded{2.0.0}
 */
struct DeclaredTensor {
    std::shared_ptr<GraphAnchor const> graph; ///< The declaring graph; null when it is the node's own.
    TensorId                           id{0}; ///< The tensor's id in the declaring graph.
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
    /// Where the tensor is declared, for a Materialize or Free placed in another graph.
    DeclaredTensor declared;
    /// Set by MemoryPlanning: the tensor is attached at this offset of its graph's arena
    /// (Graph::arena) instead of being allocated, and its Free detaches it instead of releasing it.
    std::optional<std::size_t> arena_offset;
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
 * @brief What a DiskRead or DiskWrite node moves, which is what its executor is built from.
 *
 * @versionadded{2.0.0}
 */
enum class DiskIOMode : std::uint8_t {
    Caller,  ///< Captured through ``cg::read`` / ``cg::write`` with an executor the caller wrote; nothing rebuilds it.
    Whole,   ///< DiskWrite of the whole tensor into its write backing, after its last writer.
    Publish, ///< DiskWrite that publishes the backings of @ref DiskIODescriptor::members as one group.
    Window,  ///< A window of a backing, through @ref DiskIODescriptor::window, read into or written from the node's tensor.
};

struct BackingWindow;

/**
 * @brief Metadata for disk I/O nodes (DiskRead / DiskWrite).
 *
 * Stores the file path and dataset name for tensor serialization, and what the transfer moves,
 * which a pass that places one records so the executor can be built again from the node.
 */
struct DiskIODescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "DiskIODescriptor";
    std::string                       file_path;    ///< Path to the file (HDF5, binary, etc.)
    std::string                       dataset_name; ///< Dataset/key name within the file
    TensorId                          tensor_id{0}; ///< Which tensor is being read/written
    size_t                            size_bytes{0};
    /// The parameters the transfer's region is resolved from when it runs, for a transfer of a
    /// box a loop's iteration chooses. The scheduler orders the transfer after the nodes that
    /// write them, as it orders a parameter-bound view; empty for a fixed region.
    std::vector<std::string> bound_params;
    /// What the transfer moves.
    DiskIOMode mode{DiskIOMode::Caller};
    /// For @ref DiskIOMode::Publish: the tensors, in the node's graph, whose backings publish together.
    std::vector<TensorId> members;
    /// For @ref DiskIOMode::Window: which part of which backing. Held only in memory, since the
    /// passes that place a window run again after a load.
    std::shared_ptr<BackingWindow const> window;
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
    /// Where the tensor is declared, for an Initialize placed in another graph.
    DeclaredTensor declared;
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

/**
 * @brief Which extremum an @ref OpKind::ElementwiseMaxMin or @ref OpKind::ReduceAxes node takes.
 *
 * Both follow NumPy: a NaN anywhere in what is compared is the result, and of the two signed
 * zeros the maximum is +0 and the minimum -0.
 * @versionadded{2.0.0}
 */
enum class MaxMinOp : std::uint8_t {
    Max, ///< The larger: ``numpy.maximum``, ``numpy.max``.
    Min, ///< The smaller: ``numpy.minimum``, ``numpy.min``.
};

/// The spellings of @ref MaxMinOp, as a saved graph writes them.
inline constexpr EnumNames kMaxMinOpNames{std::array<std::pair<MaxMinOp, std::string_view>, 2>{{
                                              {MaxMinOp::Max, "max"},
                                              {MaxMinOp::Min, "min"},
                                          }},
                                          "max"};

/**
 * @brief Metadata for @ref OpKind::ElementwiseMaxMin: ``C = maximum(A, B)`` or ``minimum(A, B)``.
 *
 * The node's inputs are A and B and its one output is C, which it overwrites; C may be A or B
 * with the same layout. Real element types only.
 * @versionadded{2.0.0}
 */
struct ElementwiseMaxMinDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "ElementwiseMaxMinDescriptor";
    MaxMinOp                          op{MaxMinOp::Max}; ///< Which extremum.

    bool operator==(ElementwiseMaxMinDescriptor const &) const = default; ///< Field-wise.
};

/**
 * @brief Metadata for @ref OpKind::ReduceAxes: the maximum or minimum of A over some of its axes.
 *
 * The node's one input is A and its one output is the reduction, which it overwrites. The
 * output's axes are @ref kept, in that order; every other axis of A is folded away, and none of
 * those may be empty. Real element types only.
 * @versionadded{2.0.0}
 */
struct ReduceAxesDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "ReduceAxesDescriptor";
    MaxMinOp                          op{MaxMinOp::Max}; ///< Which extremum.
    std::vector<std::size_t>          kept;              ///< The axes of A that survive, ascending.

    bool operator==(ReduceAxesDescriptor const &) const = default; ///< Field-wise.
};

/**
 * @brief The associative operator an @ref OpKind::Scan node folds with.
 *
 * @ref ScanOp::Max and @ref ScanOp::Min are real-only and follow NumPy's ``maximum.accumulate``
 * and ``minimum.accumulate``: once a NaN is seen along the scan axis, every later element is NaN.
 * @versionadded{2.0.0}
 */
enum class ScanOp : std::uint8_t {
    Sum,  ///< Running sum; identity 0.
    Prod, ///< Running product; identity 1.
    Max,  ///< Running maximum; identity -inf.
    Min,  ///< Running minimum; identity +inf.
};

/// The spellings of @ref ScanOp, as the API takes them and a saved graph writes them.
inline constexpr EnumNames kScanOpNames{std::array<std::pair<ScanOp, std::string_view>, 4>{{
                                            {ScanOp::Sum, "sum"},
                                            {ScanOp::Prod, "prod"},
                                            {ScanOp::Max, "max"},
                                            {ScanOp::Min, "min"},
                                        }},
                                        "sum"};

/**
 * @brief Metadata for @ref OpKind::Scan: a prefix fold of A along one axis.
 *
 * The node's one input is A and its one output has A's shape, overwritten. Inclusive by default:
 * element k of a line is the fold of elements 0 through k. @ref exclusive starts each line at the
 * operator's identity and folds through k - 1; @ref reverse folds from the last element toward
 * the first.
 * @versionadded{2.0.0}
 */
struct ScanDescriptor {
    /// The descriptor's name: its identity inside an @ref OpData and in a saved graph.
    static constexpr std::string_view descriptor_name = "ScanDescriptor";
    ScanOp                            op{ScanOp::Sum}; ///< The operator.
    std::size_t                       axis{0};         ///< The axis scanned along.
    bool                              exclusive{};     ///< Shift by one, starting at the identity.
    bool                              reverse{};       ///< Fold from the last element toward the first.

    bool operator==(ScanDescriptor const &) const = default; ///< Field-wise.
};

EINSUMS_NAMESPACE_END(compute_graph)
