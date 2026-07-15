//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Tensor/PendingInit.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace einsums::compute_graph {

/**
 * @brief A tensor the graph can capture into a handle.
 *
 * Either a dense, in-core basic tensor (GeneralTensor, TensorView,
 * RuntimeTensor, ...) or a tile-wise sparse in-core tensor
 * (TiledRuntimeTensor). The latter no longer satisfies BasicTensorConcept (a
 * tiled tensor isn't a single-buffer tensor), but it still exposes the metadata
 * surface make_handle needs, name/rank/dim/stride/data (with data() null for
 * the multi-tile case), so we admit it explicitly here.
 */
template <typename D>
concept GraphCapturableTensor =
    CoreBasicTensorConcept<D> || (IsIncoreTensorV<std::remove_cvref_t<D>> && IsTiledTensorV<std::remove_cvref_t<D>> && requires(D t) {
        t.data();
        t.stride(0);
        t.strides();
        t.rank();
        t.name();
    });

/**
 * @brief Unique identifier for a tensor within a computation graph.
 *
 * Each tensor registered with a Graph receives a unique TensorId.
 * These IDs are used in Node::inputs and Node::outputs to express
 * data dependencies between operations.
 */
// TensorId, AllocState, InitKind, Residency are defined in ComputeGraphTypes module
// (included via Einsums/ComputeGraphTypes/Enums.hpp and Ids.hpp above)

/**
 * @brief Type-erased handle to a tensor, storing pointer and metadata.
 *
 * TensorHandle provides a uniform representation for tensors of any type
 * and rank within the computation graph. It stores:
 * - A void pointer to the actual Tensor object (not the raw data pointer)
 * - Dimensional metadata (rank, dims, strides, element size, dtype)
 * - A validation function to detect use-after-free at runtime
 *
 * @note TensorHandle does NOT own the tensor by default. The user is
 *       responsible for ensuring tensors outlive the graph, or should
 *       use Graph::create_tensor() for graph-owned intermediates.
 *
 * @see make_handle() to construct a TensorHandle from a typed tensor
 * @see Graph::create_tensor() for graph-owned tensor creation
 */
struct TensorHandle {
    void                   *tensor_ptr{nullptr};                     ///< Pointer to the Tensor object (not its data() pointer)
    void                   *data_ptr{nullptr};                       ///< Pointer to the raw data buffer (tensor.data())
    TensorId                id{0};                                   ///< Unique identifier assigned by Graph::register_tensor()
    std::string             name;                                    ///< Human-readable tensor name (copied from tensor at registration)
    size_t                  rank{0};                                 ///< Number of dimensions (e.g., 2 for a matrix)
    size_t                  element_size{0};                         ///< Size of one element in bytes (sizeof(ValueType))
    std::vector<size_t>     dims;                                    ///< Size of each dimension, in order
    std::vector<size_t>     strides;                                 ///< Stride of each dimension, in elements
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown}; ///< Element type enum for runtime dispatch
    bool                    is_intermediate{false};                  ///< True if this tensor is owned by the graph (from create_tensor())
    bool                    is_tiled{false};       ///< True if a tile-wise sparse tensor (no single contiguous data() buffer)
    bool                    is_distributed{false}; ///< True if this tensor is distributed across ranks
    bool                    is_replicated{true};   ///< True if distributed tensor is replicated on all ranks
    AllocState              alloc_state{AllocState::Materialized}; ///< Whether data is allocated (Materialized) or deferred
    InitKind                init_kind{InitKind::None};             ///< How to initialize after materialization
    Residency               residency{Residency::Host};            ///< Where the tensor data currently lives (updated by GPU passes)

    /// Type-erased function to allocate backing storage for a deferred tensor.
    /// Called by MaterializationPass. Null for already-materialized tensors.
    std::function<void()> materialize_fn;

    /// Live query: does the tensor currently have backing storage?
    /// ``alloc_state`` is a snapshot from registration time and goes stale if
    /// the user calls ``tensor.materialize()`` directly; execute-time
    /// validation uses this to avoid false "still deferred" diagnostics.
    /// Null means "assume materialized".
    std::function<bool()> is_materialized_fn;

    /// Attach caller-provided storage instead of allocating (type-erased
    /// Tensor::materialize_into). Set only for owning tensor types that
    /// support external storage; the MemoryPlanning arena requires it.
    std::function<void(void *)> materialize_into_fn;

    /// Type-erased function to release backing storage (free memory, return to deferred state).
    /// Set by make_handle() or declare_tensor(). Called by Free nodes from FreeInsertion pass.
    std::function<void()> release_fn;

    /// Type-erased function to zero the tensor data.
    /// Set by declare_tensor on Workspace/Pipeline/Graph. Called by Initialize nodes.
    std::function<void()> zero_fn;

    /// Type-erased function to fill tensor with random values.
    /// Set by declare_tensor on Workspace/Pipeline/Graph. Called by Initialize nodes.
    std::function<void()> random_fn;

    /// Type-erased begin/end local view for input slicing.
    /// begin_local_view_fn(dim, start, count) → opaque state token
    /// end_local_view_fn(token) → restores original state
    std::function<size_t(size_t, size_t, size_t)> begin_local_view_fn;
    std::function<void(size_t)>                   end_local_view_fn;

    /// Type-erased in-place allreduce sum across MPI ranks (synchronous).
    /// Set by make_handle(). Reads tensor data at execution time.
    std::function<void()> allreduce_sum_fn;

    /// Type-erased non-blocking allreduce (async). Returns a Request to wait on.
    /// Used by CommunicationScheduling to overlap communication with computation.
    std::function<comm::Request()> iallreduce_sum_fn;

    /// Type-erased function to resize a deferred tensor's dimensions before allocation.
    /// Called by MaterializationPass when distribution_info is set.
    /// Argument: vector of new local dimensions for this rank.
    std::function<void(std::vector<size_t> const &)> resize_deferred_fn;

    /// Type-erased function to set distribution metadata on the tensor after materialization.
    /// Args: (global_dims, local_offsets), enabling T.range(dim) and T.global(indices...).
    std::function<void(std::vector<size_t> const &, std::vector<size_t> const &)> set_distribution_fn;

    /// Distribution metadata (set by DistributionPlanningPass, read by MaterializationPass).
    /// Stores the index of the dimension to block-distribute as shared_ptr<size_t>.
    std::shared_ptr<void> distribution_info;

    /// Declared or inferred tensor symmetry.
    ///
    /// At registration time ``make_handle()`` reads the backing tensor's
    /// ``.symmetry()`` and stores a copy here. The ``SymmetryPropagation``
    /// pass may later infer additional symmetry and update this hint plus
    /// the backing tensor (via ``set_symmetry_fn``). ``nullptr`` means
    /// "no declared symmetry", so downstream dispatch falls through.
    std::shared_ptr<SymmetryDescriptor const> symmetry_hint;

    /// Type-erased setter that pushes a SymmetryDescriptor back to the
    /// backing tensor. Populated by ``make_handle()``. Called by the
    /// SymmetryPropagation pass so inferred symmetries take effect on the
    /// next ``graph.execute()`` through the rank-2 BLAS dispatch path.
    std::function<void(SymmetryDescriptor)> set_symmetry_fn;

    /**
     * @brief Swap the tensor's underlying data pointer with a new one.
     *
     * Used by the GPU executor to temporarily redirect a tensor to a device shadow
     * allocation. The function captures the tensor type and performs the swap.
     * Returns the previous data pointer so it can be restored after GPU execution.
     *
     * Set by make_handle(). Null for scalar handles.
     */
    std::function<void *(void *)> swap_data;

    /**
     * @brief Hash of the tensor name at registration time.
     *
     * Used by the runtime validation system to detect destroyed tensors.
     * Compared against the tensor's current name hash before execution.
     */
    size_t name_hash{0};

    /**
     * @brief Aliasing parent, set when this handle represents a non-owning
     *        view of another tensor.
     *
     * Populated by ``cg::view()`` (the @c View op): the slice's TensorHandle
     * has ``aliases == parent_id``. Storage allocation is the parent's
     * responsibility: the Alloc/Free passes skip handles with ``aliases``
     * set, the Lifetime pass extends the parent's live range to cover all
     * uses of any of its aliases, and the InplaceOptimization /
     * scheduling passes treat reads/writes through an alias as touching
     * the parent.
     *
     * ``0`` means "not an alias" (no parent, own your own storage).
     */
    TensorId aliases{0};

    /**
     * @brief Optional validation function to check if the tensor is still alive.
     *
     * Set automatically by make_handle() at registration time. The function captures
     * a reference to the original tensor and checks that its name hash still matches.
     * Returns true if the tensor appears valid, false if it may have been destroyed.
     *
     * @note This is a best-effort check, it catches most use-after-free cases but
     *       is not guaranteed to detect all memory corruption.
     */
    std::function<bool()> validator;

    /**
     * @brief Compute the total memory footprint of this tensor in bytes.
     * @return element_size * product(dims), or 0 if dims is empty.
     */
    [[nodiscard]] size_t total_bytes() const {
        if (dims.empty())
            return 0;
        return element_size * std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>{});
    }

    /**
     * @brief Check if the tensor's dimensions match the given dimensions.
     * @param[in] other_dims Dimensions to compare against.
     * @return True if dimensions are identical.
     */
    [[nodiscard]] bool dims_match(std::vector<size_t> const &other_dims) const { return dims == other_dims; }
};

/**
 * @brief Construct a TensorHandle from a typed tensor.
 *
 * Extracts metadata (name, rank, dims, strides, dtype) from the tensor and
 * sets up a validation function that can detect if the tensor is destroyed.
 *
 * @tparam TensorType Any type satisfying CoreBasicTensorConcept.
 * @param[in] tensor The tensor to create a handle for. The tensor must outlive the graph.
 * @param[in] id Initial ID (will be reassigned by Graph::register_tensor()).
 * @return A fully populated TensorHandle.
 *
 * @code
 * auto A = create_random_tensor<double>("A", 10, 10);
 * auto handle = make_handle(A, 0);
 * // handle.name == "A", handle.rank == 2, handle.dims == {10, 10}
 * @endcode
 */
template <GraphCapturableTensor TensorType>
TensorHandle make_handle(TensorType const &tensor, TensorId id) {
    TensorHandle h;
    h.tensor_ptr = const_cast<void *>(static_cast<void const *>(&tensor));

    // data_ptr may be nullptr for deferred (shell) tensors, that's expected.
    if constexpr (requires { tensor.is_materialized(); }) {
        h.data_ptr           = tensor.is_materialized() ? const_cast<void *>(static_cast<void const *>(tensor.data())) : nullptr;
        h.alloc_state        = tensor.is_materialized() ? AllocState::Materialized : AllocState::Deferred;
        auto const *live_ptr = &tensor;
        h.is_materialized_fn = [live_ptr]() { return live_ptr->is_materialized(); };
    } else {
        h.data_ptr = const_cast<void *>(static_cast<void const *>(tensor.data()));
    }
    h.id           = id;
    h.name         = tensor.name();
    h.rank         = detail::tensor_rank(tensor);
    h.element_size = sizeof(typename std::remove_cvref_t<TensorType>::ValueType);
    h.dtype        = packed_gemm::get_scalar_type<typename std::remove_cvref_t<TensorType>::ValueType>();
    h.dims.resize(h.rank);
    h.strides.resize(h.rank);
    for (size_t i = 0; i < h.rank; i++) {
        h.dims[i]    = tensor.dim(i);
        h.strides[i] = tensor.stride(i);
    }

    // Tile-wise sparse tensors (TiledRuntimeTensor) advertise themselves so the
    // graph can flag the handle and keep them out of dense buffer-level passes;
    // their data_ptr is null (multi-tile) since there is no single buffer.
    if constexpr (requires { tensor.is_tiled_tensor(); }) {
        h.is_tiled = tensor.is_tiled_tensor();
    }

    // swap_data: redirect the tensor's internal data pointer to a new buffer.
    // Returns the previous data pointer. Used for GPU shadow allocation swapping.
    // Owning tensors expose ``set_data``; non-owning views (TensorView) do not,
    // skip the lambda for view types so make_handle still type-checks.
    using CleanTensor = std::remove_cvref_t<TensorType>;
    using ValType     = typename CleanTensor::ValueType;
    auto *tensor_mut  = const_cast<CleanTensor *>(&tensor);
    if constexpr (requires(CleanTensor &t, ValType *p) { t.set_data(p); }) {
        h.swap_data = [tensor_mut](void *new_ptr) -> void * {
            void *old_ptr = tensor_mut->data();
            tensor_mut->set_data(static_cast<ValType *>(new_ptr));
            return old_ptr;
        };
    }

    // begin_local_view_fn / end_local_view_fn: temporarily restrict tensor to a slice.
    // Only available on owning tensors that expose the local-view machinery.
    if constexpr (requires(CleanTensor &t) {
                      typename CleanTensor::LocalViewState;
                      t.begin_local_view(size_t{}, size_t{}, size_t{});
                  }) {
        auto view_states      = std::make_shared<std::vector<typename CleanTensor::LocalViewState>>();
        h.begin_local_view_fn = [tensor_mut, view_states](size_t dim, size_t start, size_t count) -> size_t {
            auto   state = tensor_mut->begin_local_view(dim, start, count);
            size_t token = view_states->size();
            view_states->push_back(state);
            return token;
        };
        h.end_local_view_fn = [tensor_mut, view_states](size_t token) {
            if (token < view_states->size()) {
                tensor_mut->end_local_view((*view_states)[token]);
            }
        };
    }

    // release_fn: free backing storage, return to deferred-like state. Non-owning
    // views don't have anything to release.
    if constexpr (requires(CleanTensor &t) { t.release(); }) {
        h.release_fn = [tensor_mut]() { tensor_mut->release(); };
    }

    // materialize_fn: allocate backing storage for a deferred tensor. When a
    // workspace-declared tensor is later referenced by ops in some Graph's
    // capture, that Graph's tensor_map gets a fresh handle made by this
    // function: workspace's canonical handle (with its own ``materialize_fn``)
    // is *not* shared. Synthesizing the callback here means the Materialization
    // pass can hoist allocation of body-resident workspace tensors out of a
    // loop without having to look up workspace's canonical handle.
    //
    // Owning tensors expose ``materialize()``; TensorView / RuntimeTensorView
    // do not, the if-constexpr keeps the function type-erased for both. For
    // workspace-declared tensors, the workspace's own canonical handle also
    // sets ``materialize_fn`` (so ``Workspace::materialize_all()`` still works
    // unchanged), both closures end up calling the same ``ptr->materialize()``,
    // which is idempotent.
    if constexpr (requires(CleanTensor &t) { t.materialize(); }) {
        h.materialize_fn = [tensor_mut]() { tensor_mut->materialize(); };
    }

    // materialize_into_fn: place the tensor at caller-provided storage (the
    // MemoryPlanning arena). Only owning tensors with external-storage
    // support (Tensor::materialize_into) qualify.
    if constexpr (requires(CleanTensor &t, typename CleanTensor::ValueType *p) { t.materialize_into(p); }) {
        h.materialize_into_fn = [tensor_mut](void *ptr) {
            tensor_mut->materialize_into(static_cast<typename CleanTensor::ValueType *>(ptr));
        };
    }

    // Post-materialize init: if the tensor was tagged with a pending-init
    // policy at declaration time (e.g. by ``Workspace::declare_zero_tensor``),
    // propagate that to the handle so the Materialization pass knows to emit
    // an Initialize node alongside the Materialize. ``pending_init()`` lives
    // on the tensor itself, not on workspace's _handles vector, so the
    // information survives capture into bodies the workspace doesn't own.
    if constexpr (requires(CleanTensor const &t) { t.pending_init(); }) {
        switch (tensor.pending_init()) {
        case PendingInit::Zero:
            h.init_kind = InitKind::Zero;
            if constexpr (requires(CleanTensor &t) {
                              t.materialize();
                              t.zero();
                          }) {
                h.zero_fn = [tensor_mut]() {
                    tensor_mut->materialize();
                    tensor_mut->zero();
                };
            }
            break;
        case PendingInit::Random:
            h.init_kind = InitKind::Random;
            // Random fill matches Workspace::declare_random_*'s inline loop:
            // uniform on [-1, 1) using ``std::rand``. Done here so a
            // body-resident handle can self-initialize without consulting
            // workspace.
            if constexpr (requires(CleanTensor &t) {
                              t.materialize();
                              t.data();
                              t.size();
                          }) {
                h.random_fn = [tensor_mut]() {
                    tensor_mut->materialize();
                    auto *data = tensor_mut->data();
                    for (size_t idx = 0; idx < tensor_mut->size(); idx++) {
                        // NOLINTNEXTLINE(misc-predictable-rand)
                        data[idx] = static_cast<ValType>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
                    }
                };
            }
            break;
        case PendingInit::None:
            break;
        }
    }

    // Symmetry metadata: seed the handle from the backing tensor's current
    // descriptor (if any) and wire a setter that SymmetryPropagation can
    // use to push an inferred descriptor back to the tensor. The setter is
    // unconditional on the tensor type having a set_symmetry method;
    // CoreBasicTensorConcept types (GeneralTensor, TensorView, etc.) all
    // expose one or can be no-ops.
    if constexpr (requires { tensor.symmetry(); }) {
        if (auto const *desc = tensor.symmetry())
            h.symmetry_hint = std::make_shared<SymmetryDescriptor>(*desc);
    }
    if constexpr (requires(SymmetryDescriptor d) { tensor_mut->set_symmetry(std::move(d)); }) {
        h.set_symmetry_fn = [tensor_mut](SymmetryDescriptor d) { tensor_mut->set_symmetry(std::move(d)); };
    }

    // allreduce_sum_fn: in-place sum across MPI ranks. Uses the tensor's data() and size()
    // at execution time (after materialization). Safe for both pre-allocated and deferred tensors.
    h.allreduce_sum_fn = [tensor_mut]() {
        auto span = std::span<ValType>(tensor_mut->data(), tensor_mut->size());
        (void)comm::allreduce_inplace<ValType>(span, comm::ReduceOp::Sum);
    };

    // iallreduce_sum_fn: non-blocking version for async overlap.
    // Returns a Request that must be waited on before reading the result.
    h.iallreduce_sum_fn = [tensor_mut]() -> comm::Request {
        auto span   = std::span<ValType>(tensor_mut->data(), tensor_mut->size());
        auto result = comm::iallreduce_inplace<ValType>(span, comm::ReduceOp::Sum);
        if (result.has_value())
            return std::move(result.value());
        return comm::Request{}; // Fallback: default request (immediately complete)
    };

    h.name_hash = std::hash<std::string>{}(h.name);
    // Capture a liveness token (weak_ptr) so destruction is detected WITHOUT
    // dereferencing a possibly-freed tensor. Types without the token skip the
    // check (they were unchecked before too). `life_token` is empty for those,
    // and `has_life_token` gates the .expired() probe so an empty weak_ptr (which
    // reports expired) is never mistaken for a destroyed tensor.
    constexpr bool      has_life_token = requires { tensor_mut->liveness_token(); };
    std::weak_ptr<void> life_token;
    if constexpr (has_life_token) {
        life_token = tensor_mut->liveness_token();
    }
    h.validator = [tensor_mut, life_token, expected_hash = h.name_hash]() -> bool {
        try {
            if constexpr (has_life_token) {
                if (life_token.expired())
                    return false; // tensor destroyed, definitive, no UB
            }
            return std::hash<std::string>{}(tensor_mut->name()) == expected_hash;
        } catch (...) {
            return false;
        }
    };

    return h;
}

/**
 * @brief Construct a TensorHandle for a scalar value (rank-0 tensor).
 *
 * Scalars don't have dimensions or strides, but can participate in the graph
 * as inputs/outputs for operations like dot() or det() that return scalars.
 *
 * @tparam T An arithmetic type (double, float, int, etc.).
 * @param[in] scalar Pointer to the scalar value.
 * @param[in] id Initial ID (will be reassigned by Graph::register_tensor()).
 * @param[in] name Human-readable name for the scalar.
 * @return A TensorHandle with rank=0 and no dims/strides.
 */
template <typename T>
    requires(std::is_arithmetic_v<T> || IsComplexV<T>)
TensorHandle make_scalar_handle(T *scalar, TensorId id, std::string name = "scalar") {
    TensorHandle h;
    h.tensor_ptr   = scalar;
    h.id           = id;
    h.name         = std::move(name);
    h.rank         = 0;
    h.element_size = sizeof(T);
    h.dtype        = packed_gemm::get_scalar_type<T>();

    // Enable allreduce on scalar results from distributed computations
    h.allreduce_sum_fn = [scalar]() {
        auto span = std::span<T>(scalar, 1);
        (void)comm::allreduce_inplace<T>(span, comm::ReduceOp::Sum);
    };

    return h;
}

} // namespace einsums::compute_graph
