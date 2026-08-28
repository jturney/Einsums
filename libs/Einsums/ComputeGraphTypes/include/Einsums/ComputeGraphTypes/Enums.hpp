//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN()
namespace APIARY_MODULE("graph") compute_graph {

/**
 * @brief Identifies the kind of operation a node represents.
 *
 * Used by optimization passes for pattern matching. For example,
 * ScaleAbsorption looks for OpKind::Scale followed by a prefactor-bearing op.
 *
 * Categories:
 * - **TensorAlgebra**: Einsum, Permute, Transpose, ElementTransform, KhatriRao
 * - **BLAS-level**: Gemm, Gemv, Ger, Dot, Scale, Axpby, DirectProduct
 * - **LAPACK-level**: SVD, QR, Syev, Heev, Geev, Gesv, Getrf, Getrs, Invert, Det, Pow, etc.
 * - **Other**: HPTTPermute, Custom
 */
enum class APIARY_EXPOSE OpKind : std::uint8_t {
    // TensorAlgebra operations
    Einsum,           ///< Tensor contraction via tensor_algebra::einsum()
    Permute,          ///< Index reordering via tensor_algebra::permute()
    GroupedPermute,   ///< Many independent index reorderings, one per member
    Transpose,        ///< 2D transpose via tensor_algebra::transpose()
    ElementTransform, ///< Element-wise unary transform
    KhatriRao,        ///< Khatri-Rao product

    // LinearAlgebra - BLAS level
    BatchedGemm,           ///< Many independent GEMMs in one `gemm_batch` call
    GroupedBatchedGemm,    ///< Many independent GEMMs of DIFFERING shape in one `gemm_batch_grouped` call
    Gemm,                  ///< General matrix-matrix multiply (BLAS Level 3)
    Gemv,                  ///< General matrix-vector multiply (BLAS Level 2)
    Ger,                   ///< Rank-1 update (BLAS Level 2)
    Dot,                   ///< Dot product (returns scalar)
    GroupedDot,            ///< Many independent dot products, one per entry, in one node
    Scale,                 ///< Scalar multiplication of entire tensor
    Axpby,                 ///< Y = alpha * X + beta * Y
    GroupedAxpby,          ///< Many independent `Y = alpha*X + beta*Y`, one per entry, in one node
    GroupedSandwich,       ///< Many independent q-tiled dressed sandwich accumulations, one per entry
    GroupedGatherRotate,   ///< Many independent q-tiled gather-and-rotate blocks, one per entry
    DirectProduct,         ///< Element-wise (Hadamard) product
    DirectDivision,        ///< Element-wise (Hadamard) quotient
    GroupedDirectProduct,  ///< Many independent element-wise products, one per member
    GroupedDirectDivision, ///< Many independent element-wise quotients, one per member

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
    Getrs,         ///< Solve against an LU factorization
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
    Setup,       ///< Body computed once per bound problem, skipped by later replays

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

    // Tiled lowering
    TileGather,      ///< Copy a tiled tensor's tiles into one dense buffer
    TileScatter,     ///< Copy a dense buffer back into a tiled tensor's tiles
    TileElementwise, ///< Apply one elementwise operation to a whole list of tiles

    // Iterative-solver acceleration
    DiisStep, ///< Pulay DIIS extrapolation over an accelerator's (amplitude, step) pairs

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
    case OpKind::GroupedPermute:
        return "GroupedPermute";
    case OpKind::Transpose:
        return "Transpose";
    case OpKind::ElementTransform:
        return "ElementTransform";
    case OpKind::KhatriRao:
        return "KhatriRao";
    case OpKind::BatchedGemm:
        return "BatchedGemm";
    case OpKind::GroupedBatchedGemm:
        return "GroupedBatchedGemm";
    case OpKind::Gemm:
        return "Gemm";
    case OpKind::Gemv:
        return "Gemv";
    case OpKind::Ger:
        return "Ger";
    case OpKind::Dot:
        return "Dot";
    case OpKind::GroupedDot:
        return "GroupedDot";
    case OpKind::Scale:
        return "Scale";
    case OpKind::Axpby:
        return "Axpby";
    case OpKind::GroupedAxpby:
        return "GroupedAxpby";
    case OpKind::GroupedSandwich:
        return "GroupedSandwich";
    case OpKind::GroupedGatherRotate:
        return "GroupedGatherRotate";
    case OpKind::DirectProduct:
        return "DirectProduct";
    case OpKind::DirectDivision:
        return "DirectDivision";
    case OpKind::GroupedDirectProduct:
        return "GroupedDirectProduct";
    case OpKind::GroupedDirectDivision:
        return "GroupedDirectDivision";
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
    case OpKind::Getrs:
        return "Getrs";
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
    case OpKind::Setup:
        return "Setup";
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
    case OpKind::DiisStep:
        return "DiisStep";
    case OpKind::Custom:
        return "Custom";
    case OpKind::TileGather:
        return "TileGather";
    case OpKind::TileScatter:
        return "TileScatter";
    case OpKind::TileElementwise:
        return "TileElementwise";
    }
    return "Unknown";
}

/**
 * @brief The @ref OpKind spelled @p name, if there is one.
 * @param[in] name A spelling @ref op_kind_name produces.
 * @return The kind, or an empty optional when nothing is spelled that way.
 *
 * The reverse of @ref op_kind_name, and it exists because the saved-graph IR
 * writes op kinds BY NAME: @ref OpKind is a @c std::uint8_t
 * enum whose members are added in the middle by whoever adds an operation, so a
 * numeric value in a file would silently mean a different operation the next
 * time someone inserts a kind. An unresolvable name is an empty optional rather
 * than a fallback, so a loader can fail naming the string it could not resolve.
 *
 * Linear over the enumerators, which is what keeps the two functions from
 * drifting: there is one table (the ``switch`` in @ref op_kind_name) and this
 * asks it, rather than a second table that could disagree.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::optional<OpKind> op_kind_from_name(std::string_view name) {
    for (std::uint16_t value = 0; value <= static_cast<std::uint16_t>(OpKind::Custom); ++value) {
        auto const kind = static_cast<OpKind>(value);
        if (op_kind_name(kind) == name) {
            return kind;
        }
    }
    return std::nullopt;
}

/**
 * @brief The name of an allocation state, for diagnostics and the saved form.
 * @param[in] state The state to name.
 * @return A stable spelling ("materialized", "deferred").
 *
 * Distinct from @ref init_kind_name, which says what a tensor is FILLED with. This says
 * whether it has storage at all, which is what decides whether a bind may still reshape it.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::string_view alloc_state_name(AllocState state) noexcept {
    switch (state) {
    case AllocState::Materialized:
        return "materialized";
    case AllocState::Deferred:
        return "deferred";
    }
    return "materialized";
}

/**
 * @brief The @ref AllocState spelled @p name, if there is one.
 * @param[in] name A spelling @ref alloc_state_name produces.
 * @return The state, or an empty optional.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::optional<AllocState> alloc_state_from_name(std::string_view name) noexcept {
    for (auto const state : {AllocState::Materialized, AllocState::Deferred}) {
        if (alloc_state_name(state) == name) {
            return state;
        }
    }
    return std::nullopt;
}

/**
 * @brief The name of an initialization strategy, for diagnostics and the saved form.
 * @param[in] kind The strategy to name.
 * @return A stable spelling ("none", "zero", "random", "from_disk").
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::string_view init_kind_name(InitKind kind) noexcept {
    switch (kind) {
    case InitKind::None:
        return "none";
    case InitKind::Zero:
        return "zero";
    case InitKind::Random:
        return "random";
    case InitKind::FromDisk:
        return "from_disk";
    }
    return "none";
}

/**
 * @brief The @ref InitKind spelled @p name, if there is one.
 * @param[in] name A spelling @ref init_kind_name produces.
 * @return The strategy, or an empty optional.
 * @versionadded{2.0.0}
 */
[[nodiscard]] inline std::optional<InitKind> init_kind_from_name(std::string_view name) noexcept {
    for (auto const kind : {InitKind::None, InitKind::Zero, InitKind::Random, InitKind::FromDisk}) {
        if (init_kind_name(kind) == name) {
            return kind;
        }
    }
    return std::nullopt;
}

// ── OpKind classifier predicates ───────────────────────────────────────────
// Shared vocabulary for pass pattern-matching. Kept here, next to the enum and
// op_kind_name, so the membership of each category lives in ONE place: several
// passes and Graph::effective_io must agree on these sets, and hand-copied
// inline checks silently drift as new OpKinds are added.

/// @brief Lifecycle/bookkeeping kinds that produce no value of their own
///        (Alloc, Free, Materialize, Initialize). Passes counting value-producing
///        nodes or resolving readers/writers skip these; must stay in lockstep
///        with Graph::effective_io.
[[nodiscard]] inline bool is_lifecycle(OpKind kind) {
    return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
}

/// @brief Control-flow kinds carrying subgraphs (Conditional, Loop, Setup).
///
/// Membership is about STRUCTURE, not about how often a body runs: everything that
/// walks, expands, or refuses a node because it owns a sub-graph has to see all three.
/// A @ref OpKind::Setup body runs at most once per bound problem where a loop body runs
/// many times, and that difference belongs to the executor and to whichever pass reasons
/// about iteration counts, not to the question this predicate answers.
[[nodiscard]] inline bool is_control_flow(OpKind kind) {
    return kind == OpKind::Conditional || kind == OpKind::Loop || kind == OpKind::Setup;
}

/// @brief Host<->device transfer kinds (HostToDevice, DeviceToHost).
[[nodiscard]] inline bool is_transfer(OpKind kind) {
    return kind == OpKind::HostToDevice || kind == OpKind::DeviceToHost;
}

/// @brief Distributed collective kinds (Allreduce, Broadcast, Allgather, Scatter, Barrier).
[[nodiscard]] inline bool is_collective(OpKind kind) {
    return kind == OpKind::Allreduce || kind == OpKind::Broadcast || kind == OpKind::Allgather || kind == OpKind::Scatter ||
           kind == OpKind::Barrier;
}

/// @brief Infrastructure kinds that carry no einsum-style data dependency a
///        distribution/communication pass should reason about: collectives plus
///        materialize/initialize, host/device transfers, disk I/O, and control flow.
[[nodiscard]] inline bool is_infrastructure(OpKind kind) {
    return is_collective(kind) || is_transfer(kind) || is_control_flow(kind) || kind == OpKind::Materialize || kind == OpKind::Initialize ||
           kind == OpKind::DiskRead || kind == OpKind::DiskWrite;
}

} // namespace APIARY_MODULE("graph")compute_graph
EINSUMS_NAMESPACE_END()
