//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/// @file View.hpp
/// @brief Graph-aware non-owning views (zero-copy slices) of tensors.
///
/// ``cg::view<T, Rank>(parent, axes...)`` records a @ref OpKind::View node
/// and returns a reference to a graph-owned ``TensorView<T, Rank>`` that
/// downstream graph ops can consume. Each iteration the View executor
/// rebuilds the underlying view by:
///
/// 1. Resolving each axis's @ref BoundExpr against the active Pipeline's
///    @ref ParamTable.
/// 2. Computing the slice's data pointer = ``parent.data() + Σ offset_i *
///    parent.stride(i)`` and the per-axis extents.
/// 3. Re-emplacing a heap-stored ``std::optional<TensorView<T, Rank>>``
///    with the new pointer/dims/strides. The optional's storage address is
///    stable, so consuming ops continue to read through the same slot.
///
/// The view's ``TensorHandle::aliases`` points back to the parent so
/// passes (Alloc, Free, lifetime, in-place fusion, scheduling) can treat
/// reads/writes through the slice as touching the parent.
///
/// Current limitations, intentional and documented for follow-up:
///   - Slices are contiguous (step == 1); strided slicing is not supported.
///   - The typed, compile-time-rank ``cg::view<T, Rank>`` supports only
///     @ref ViewAxis::Kind::Full and @ref ViewAxis::Kind::Range axes and is
///     rank-preserving; calling it with a ``Drop`` axis throws at capture.
///     ``Drop`` (rank-reducing index) is supported on the runtime-rank path
///     (@ref view_runtime, or the Python ``view_indexed``), where the result
///     rank is the parent rank minus the number of Drop axes.
///   - Axis permutations (transpose-via-view) are available on both the typed
///     and runtime-rank paths via ``cg::permute_view``; a Drop axis and a
///     permutation cannot be combined in one call (chain two views instead).
///   - Single-node assumption: distributed-tensor parents work only if
///     the slice doesn't straddle a partition boundary; otherwise the
///     view's data pointer is undefined. This is enforced lazily; the
///     dispatch passes will gain a check in a follow-up.

namespace detail {

/// Heap-stored holder for a ``TensorView<T, Rank>`` that the View
/// executor re-emplaces each iteration. Lives in
/// ``Graph::_owned_tensors`` (graph-owned), giving the @ref TensorSlot
/// a stable address to capture.
template <typename T, size_t Rank>
struct ViewHolder {
    std::optional<TensorView<T, Rank>> view;

    /// @see bind_holder_liveness.
    std::shared_ptr<void> life{std::make_shared<char>()};
};

/// Runtime-rank counterpart to ViewHolder. Holds an ``std::optional<RuntimeTensorView<T>>``
/// whose address is stable across re-emplaces, so the executor can refresh
/// the slice's data pointer / dims / strides each iteration without
/// invalidating slots that downstream ops captured by reference.
///
/// Everything the executor reads lives here rather than in its closure, so
/// the closure is a single pointer: small enough for std::function's inline
/// buffer, which keeps the recording path (a DLPNO capture builds 18k of
/// these) off the heap for the lambda and off a second copy of the axes.
/// The two scratch vectors are the executor's working space, allocated once
/// at capture and re-filled each replay; before they existed every replay of
/// every view node paid two fresh vector allocations.
template <typename T>
struct RuntimeViewHolder {
    std::optional<RuntimeTensorView<T>> view;

    std::vector<ViewAxis>       axes;          ///< The slice recipe, moved in at capture.
    std::vector<size_t>         perm;          ///< Axis permutation (empty == identity).
    std::shared_ptr<ParamTable> params;        ///< Keeps the graph's table alive for resolve().
    TensorSlot                 *parent_slot{}; ///< Where the parent's current address lives.
    TensorSlot                 *slot{};        ///< Where this view publishes its address.

    std::vector<size_t> scratch_dims;    ///< Reused per replay.
    std::vector<size_t> scratch_strides; ///< Reused per replay.

    /// @see bind_holder_liveness.
    std::shared_ptr<void> life{std::make_shared<char>()};
};

/// Point a view handle's liveness check at its HOLDER rather than at the view
/// instance the holder happened to contain at capture.
///
/// ``make_handle`` captures a weak liveness token from the object it is handed,
/// which for a View node is the placeholder inside the holder. Every replay
/// re-emplaces that optional, destroying the instance whose token the handle
/// registered, so from the first replay onwards the token reads as expired.
/// Nothing noticed while ``execute()`` validated only before the first run, but
/// a modifying pass calls ``Graph::mark_sorted()``, which clears the executed
/// flag: the next ``execute()`` re-validates and reported a perfectly live,
/// graph-owned view as destroyed.
///
/// The holder is what the handle actually names. The graph owns it, it outlives
/// every replay, and it is destroyed exactly when the graph is, so its own token
/// is the stable one. The reference stays WEAK deliberately: a handle that
/// somehow outlived its graph then still reports "destroyed" instead of reading
/// freed memory. Liveness of the data behind the view is the PARENT's business,
/// and the parent carries its own handle and its own check.
template <typename Holder>
void bind_holder_liveness(TensorHandle &handle, Holder const *holder) {
    handle.validator = [life = std::weak_ptr<void>(holder->life)]() -> bool { return !life.expired(); };
}

} // namespace detail

namespace detail {

/// Shared body for the typed @ref cg::view and @ref cg::permute_view.
///
/// @p axis_vec has one @ref ViewAxis per parent rank (the slice/full of
/// result axis i). @p perm is the axis permutation: result axis i aliases
/// parent axis ``perm[i]`` (empty == identity). Records an
/// @ref OpKind::View node and returns a reference to the graph-owned,
/// parent-aliasing slice. (Runtime-rank counterpart: @ref view_runtime.)
template <typename T, size_t Rank, CoreBasicTensorConcept ParentT>
TensorView<T, Rank> &record_typed_view(ParentT &parent, std::vector<ViewAxis> const &axis_vec, std::vector<size_t> const &perm) {
    using Holder = ViewHolder<T, Rank>;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view called outside of capture");
    }

    for (auto const &ax : axis_vec) {
        if (ax.kind == ViewAxis::Kind::Drop)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: ViewAxis::Drop is not yet supported");
    }
    // ``perm`` (empty == identity) maps result axis i -> parent axis perm[i].
    // Validate it's a genuine permutation of [0, Rank).
    if (!perm.empty()) {
        if (perm.size() != Rank)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: permutation length ({}) must equal rank ({})", perm.size(), Rank);
        std::array<bool, Rank> seen{};
        for (size_t const p : perm) {
            if (p >= Rank || seen[p])
                EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: permutation must be a bijection of [0, {})", Rank);
            seen[p] = true;
        }
    }
    auto const parent_axis = [&perm](size_t i) -> size_t { return perm.empty() ? i : perm[i]; };

    // Allocate a holder on the heap. Owned by the graph so its address
    // is stable across the lifetime of every executor that captured it.
    auto *holder = new Holder;

    // At capture time we don't yet have a ParamTable to resolve Param
    // bounds against. Emplace a "full parent" view (permuted, if requested)
    // as a placeholder so the holder's view has a valid object, its address
    // is what downstream ops will capture, and the handle registered below
    // sees the correct post-permute dims. The first execute() re-emplaces
    // with the actual computed bounds.
    // Constant Range bounds get their real sliced extent in the placeholder so
    // downstream ops that read dims at capture time see correct sizes (see the
    // runtime-rank view_runtime for the rationale).
    auto const placeholder_dim = [&](size_t i, size_t p) -> size_t {
        auto const &ax = axis_vec[i];
        if (ax.kind == ViewAxis::Kind::Range && ax.lo.is_const() && ax.hi.is_const())
            return static_cast<size_t>(ax.hi.const_value() - ax.lo.const_value());
        return parent.dim(p);
    };
    // Offset the placeholder's base pointer by the slice's constant bounds, exactly as
    // ``view_runtime`` does. Without it the registration-time geometry on the handle is the
    // slice's EXTENTS at the PARENT's address, which is not any real region: every slice of
    // one parent registers at the same base, so a derivation that reads addresses sees them
    // as overlapping when they are disjoint. The hazard scan takes the ``View`` node's box in
    // preference to the handle's and so never asked, which is why this was harmless in
    // practice and wrong on paper; a region rewrite that consults a handle would inherit the
    // lie, so the two recorders agree here instead.
    std::ptrdiff_t ph_offset = 0;
    Stride<Rank>   parent_strides;
    Dim<Rank>      parent_dims;
    for (size_t i = 0; i < Rank; ++i) {
        size_t const p = parent_axis(i);
        if (auto const &ax = axis_vec[i]; ax.kind != ViewAxis::Kind::Full && ax.lo.is_const()) {
            ph_offset += ax.lo.const_value() * static_cast<std::ptrdiff_t>(parent.stride(p));
        }
        parent_dims[i]    = placeholder_dim(i, p);
        parent_strides[i] = parent.stride(p);
    }
    // A deferred parent has no base to offset from, and nullptr + n is undefined rather than
    // merely useless. Such a placeholder keeps the null base the executor re-emplaces anyway.
    T *ph_base = parent.data();
    if (ph_base != nullptr) {
        ph_base += ph_offset;
    }
    holder->view.emplace(ph_base, parent_dims, parent_strides);

    auto *graph = ctx.graph();
    if (graph == nullptr)
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: no active graph");

    // Hand ownership to the graph alongside other owned tensors.
    graph->adopt([holder]() { delete holder; });

    // Register the parent first so its TensorId is stable.
    auto [parent_id, parent_slot] = ctx.get_slot(parent);

    // Register the slice as a graph tensor. Its handle aliases the parent.
    TensorView<T, Rank> &slice_ref = holder->view.value();
    auto                 handle    = make_handle(slice_ref, 0);
    handle.is_intermediate         = true;
    handle.name                    = fmt::format("view({})", parent.name());
    handle.aliases                 = parent_id;
    bind_holder_liveness(handle, holder);

    TensorId const slice_id = graph->register_tensor(std::move(handle));
    auto          *slot     = ctx.create_slot_for(slice_ref, slice_id);

    // Build the View descriptor for passes to inspect.
    ViewDescriptor desc;
    desc.parent_id   = parent_id;
    desc.axes        = axis_vec;
    desc.result_rank = Rank;
    desc.permutation = perm;

    // Capture references to the resolved-at-execute-time pieces.
    auto params_ptr = ctx.params_ptr();

    auto label = fmt::format("view {} <- {}", handle.name, parent.name());

    auto executor = [holder, axis_vec, perm, parent_slot, params_ptr, slot]() {
        if (!params_ptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view executor: no ParamTable bound to graph");

        auto *parent_ptr = static_cast<ParentT *>(parent_slot->ptr);
        if (parent_ptr == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view executor: parent slot is null");
        if (parent_ptr->data() == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view executor: parent has no backing data (deferred?)");

        // Resolve bounds and compute slice pointer + dims.
        std::array<std::int64_t, Rank> offsets{};
        Dim<Rank>                      slice_dims;
        Stride<Rank>                   slice_strides;
        std::ptrdiff_t                 ptr_offset = 0;

        for (size_t i = 0; i < Rank; ++i) {
            // Result axis i maps to parent axis p; axis_vec[i] slices it.
            size_t const p   = perm.empty() ? i : perm[i];
            auto const  &ax  = axis_vec[i];
            slice_strides[i] = parent_ptr->stride(p);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                offsets[i]    = 0;
                slice_dims[i] = parent_ptr->dim(p);
                break;
            case ViewAxis::Kind::Range: {
                std::int64_t const lo = ax.lo.resolve(*params_ptr);
                std::int64_t const hi = ax.hi.resolve(*params_ptr);
                if (lo < 0 || hi < lo || hi > static_cast<std::int64_t>(parent_ptr->dim(p)))
                    EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::view: range out of parent dim");
                offsets[i]    = lo;
                slice_dims[i] = static_cast<size_t>(hi - lo);
                break;
            }
            case ViewAxis::Kind::Drop:
                EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: Drop axis not supported in v1");
            }
            ptr_offset += offsets[i] * static_cast<std::ptrdiff_t>(parent_ptr->stride(p));
        }

        // Re-emplace the view with the new pointer/dims/strides. Address
        // of the contained TensorView is stable across emplace.
        T *slice_data = parent_ptr->data() + ptr_offset;
        holder->view.emplace(slice_data, slice_dims, slice_strides);

        // Refresh the slot's view of the slice's data, in case any
        // consumer caches data_ptr instead of going through the slot's
        // tensor object each call.
        slot->ptr = &holder->view.value();
    };

    ctx.record(OpKind::View, std::move(label), {parent_id}, {slice_id}, std::move(executor), std::move(desc));

    return slice_ref;
}

} // namespace detail

/// @brief Record a non-owning slice of a tensor.
///
/// Records a @ref OpKind::View node and returns a graph-owned slice that
/// downstream operations can consume. @p parent must outlive the graph.
///
/// @tparam T          Element type.
/// @tparam Rank       Rank of the resulting view (currently must match parent's rank).
/// @tparam ParentT    Parent tensor type (any ``CoreBasicTensorConcept`` value).
/// @tparam Axes       Pack of @ref ViewAxis. Use ``ViewAxis::full()`` and
///                    ``ViewAxis::range(lo, hi)`` to construct.
/// @param[in] parent  The tensor being sliced.
/// @param[in] axes    One ViewAxis per parent rank.
/// @return Reference to a graph-owned ``TensorView<T, Rank>`` that subsequent
///         ``cg::*`` calls (einsum, gemm, …) can consume by value/reference.
template <typename T, size_t Rank, CoreBasicTensorConcept ParentT, typename... Axes>
TensorView<T, Rank> &view(ParentT &parent, Axes &&...axes) {
    static_assert(sizeof...(Axes) == Rank, "cg::view requires one ViewAxis per parent rank (rank-preserving slices only in v1)");
    static_assert(std::is_same_v<typename ParentT::ValueType, T>, "cg::view: T must match parent's ValueType");
    return detail::record_typed_view<T, Rank, ParentT>(parent, std::vector<ViewAxis>{ViewAxis(std::forward<Axes>(axes))...}, {});
}

/// Type-deducing overload that infers @c T and @c Rank from @p parent. The
/// codegen path emits this form so it doesn't have to spell out
/// ``view<typename T::ValueType, T::Rank>(...)`` boilerplate.
template <CoreBasicTensorConcept ParentT, typename... Axes>
    requires HasCompileTimeRank<ParentT>
TensorView<typename std::remove_cvref_t<ParentT>::ValueType, std::remove_cvref_t<ParentT>::Rank> &view(ParentT &parent, Axes &&...axes) {
    using T            = typename std::remove_cvref_t<ParentT>::ValueType;
    constexpr size_t R = std::remove_cvref_t<ParentT>::Rank;
    return view<T, R, ParentT>(parent, std::forward<Axes>(axes)...);
}

/// @brief Typed (compile-time rank) transpose / axis-permutation view.
///
/// Result axis i aliases parent axis ``perm[i]``; @p perm must be a
/// permutation of ``[0, Rank)``. Compile-time-rank counterpart to the
/// runtime ``cg::permute_view`` that backs Python's capture-aware ``.T``.
/// Records a graph-registered, parent-aliasing view that downstream typed
/// ``cg::*`` ops (einsum, gemm, …) can consume, unlike a raw
/// ``transpose_view``, whose shared data pointer is invisible to the graph.
/// @p Rank is deduced from the @p perm array (and asserted to equal the
/// parent's rank).
///
/// @code
///   with-capture:
///     auto &At = cg::permute_view(A, std::array<size_t, 2>{1, 0});  // A^T
///     cg::gemm(1.0, At, B, 0.0, C);
/// @endcode
template <CoreBasicTensorConcept ParentT, size_t Rank>
    requires HasCompileTimeRank<ParentT>
TensorView<typename std::remove_cvref_t<ParentT>::ValueType, Rank> &permute_view(ParentT &parent, std::array<size_t, Rank> const &perm) {
    using T = typename std::remove_cvref_t<ParentT>::ValueType;
    static_assert(Rank == std::remove_cvref_t<ParentT>::Rank, "cg::permute_view: perm length must equal the parent's rank");
    std::vector<ViewAxis> axes(Rank, ViewAxis::full());
    return detail::record_typed_view<T, Rank, ParentT>(parent, axes, std::vector<size_t>(perm.begin(), perm.end()));
}

/// @brief Runtime-rank counterpart to cg::view: record a non-owning slice
///        of a RuntimeTensor parent.
///
/// Unlike the typed ``cg::view``, this path honors ``Drop`` axes
/// (rank-reducing integer index): the result rank is the parent rank minus
/// the number of Drop axes. The optional trailing @p perm vector applies an
/// axis permutation (result axis i aliases parent axis ``perm[i]``); this is
/// the runtime-rank counterpart of ``cg::permute_view``, and the Python
/// binding ``einsums.graph.permute_view`` takes it as a list. A Drop axis
/// and a permutation cannot be combined in one call (chain two views
/// instead). Strides are inherited from the parent. The slice's rank is read
/// from the parent at capture time; the executor rebuilds the slice each
/// iteration by re-emplacing a heap-stored RuntimeTensorView with new
/// pointer / dims / strides.
template <RuntimeRankTensorConcept ParentT>
RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &view_runtime(ParentT &parent, std::vector<ViewAxis> axis_vec,
                                                                                  std::vector<size_t> perm = {}) {
    using T      = typename std::remove_cvref_t<ParentT>::ValueType;
    using Holder = detail::RuntimeViewHolder<T>;

    size_t const rank = parent.rank();
    if (axis_vec.size() != rank) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime requires one ViewAxis per parent rank ({} given, parent has rank {})",
                                axis_vec.size(), rank);
    }
    // Drop axes (rank-reducing integer index) remove a parent axis from the
    // result: result rank = parent rank - #Drop. axis_vec is parent-indexed in
    // this (perm-empty) mode. Drops and a permutation don't co-occur from the
    // Python paths (transpose = perm only; indexing = drop/range only).
    size_t n_drop = 0;
    for (auto const &ax : axis_vec)
        if (ax.kind == ViewAxis::Kind::Drop)
            ++n_drop;
    // ``perm`` (empty == identity) maps result axis i -> parent axis perm[i],
    // so a transpose/permutation becomes a graph-registered, parent-aliasing
    // view. Validate it's a genuine permutation of [0, rank).
    if (!perm.empty()) {
        if (perm.size() != rank)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: permutation length ({}) must equal parent rank ({})", perm.size(),
                                    rank);
        std::vector<bool> seen(rank, false);
        for (size_t const p : perm) {
            if (p >= rank || seen[p])
                EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: permutation must be a bijection of [0, {})", rank);
            seen[p] = true;
        }
    }
    if (n_drop > 0 && !perm.empty())
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: Drop axes cannot be combined with a permutation");
    size_t const result_rank = rank - n_drop;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime called outside of capture");
    }

    // The recipe moves into the holder and is read from there for the rest of
    // this function: the parameters are moved-from beyond this point.
    auto *holder         = new Holder;
    holder->axes         = std::move(axis_vec);
    holder->perm         = std::move(perm);
    auto const &axes_ref = holder->axes;
    auto const &perm_ref = holder->perm;

    auto const parent_axis = [&perm_ref](size_t i) -> size_t { return perm_ref.empty() ? i : perm_ref[i]; };

    // Placeholder: build a "full parent" view (permuted and/or with dropped
    // axes removed) so the holder's address is valid before the first
    // execute() runs and the registered handle below sees the correct
    // post-permute / post-drop rank and dims (ranges resolve at execute).
    // A Range axis with constant bounds gets its real sliced extent (hi - lo)
    // in the placeholder, not the full parent dim, downstream operators read
    // the captured view's dims/size at *capture* time (matmul's output alloc,
    // mean's divisor, elementwise same-shape checks), so a full-parent
    // placeholder would mis-size them. Param-bounded ranges aren't known until
    // execute and fall back to the parent dim.
    auto const placeholder_dim = [&](size_t i, size_t p) -> size_t {
        auto const &ax = axes_ref[i];
        if (ax.kind == ViewAxis::Kind::Range && ax.lo.is_const() && ax.hi.is_const())
            return static_cast<size_t>(ax.hi.const_value() - ax.lo.const_value());
        return parent.dim(p);
    };
    // Also apply the constant slice/drop offset to the placeholder data
    // pointer. Beyond matching the resolved view for const bounds, this gives
    // a view a data pointer distinct from its parent's, so get_slot doesn't
    // collide them, required for chained views (a slice of a slice) to
    // resolve the correct parent at execute.
    std::ptrdiff_t ph_offset      = 0;
    auto          &parent_dims    = holder->scratch_dims;
    auto          &parent_strides = holder->scratch_strides;
    parent_dims.reserve(result_rank);
    parent_strides.reserve(result_rank);
    for (size_t i = 0; i < rank; ++i) {
        auto const  &ax = axes_ref[i];
        size_t const p  = parent_axis(i);
        if (ax.kind != ViewAxis::Kind::Full && ax.lo.is_const())
            ph_offset += ax.lo.const_value() * static_cast<std::ptrdiff_t>(parent.stride(p));
        if (ax.kind == ViewAxis::Kind::Drop)
            continue; // dropped axes do not appear in the result
        parent_dims.push_back(placeholder_dim(i, p));
        parent_strides.push_back(parent.stride(p));
    }
    // Take the base pointer from the parent's IMPL, not from data(). For a
    // DEFERRED parent (Graph::declare_*_tensor) data() returns the empty
    // storage vector's pointer, i.e. nullptr, while impl() holds the shell's
    // sentinel. TensorImpl::dim() reports 0 for every axis of a null-pointer
    // tensor with nonzero size, so a nullptr base would give the placeholder
    // all-zero dims -- and capture-time einsum validation then rejects any
    // operation on a view of a deferred tensor ("index 'e' spans 0 elements").
    // For a materialized parent the two are the same pointer. The real address
    // is re-emplaced by the executor below once the parent is allocated.
    holder->view.emplace(::einsums::detail::TensorImpl<T>(const_cast<T *>(parent.impl().data()) + ph_offset, parent_dims, parent_strides));

    auto *graph = ctx.graph();
    if (graph == nullptr)
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: no active graph");

    graph->adopt([holder]() { delete holder; });

    auto [parent_id, parent_slot] = ctx.get_slot(parent);

    RuntimeTensorView<T> &slice_ref = holder->view.value();
    auto                  handle    = make_handle(slice_ref, 0);
    handle.is_intermediate          = true;
    handle.name                     = fmt::format("view_rt({})", parent.name());
    handle.aliases                  = parent_id;
    detail::bind_holder_liveness(handle, holder);

    TensorId const slice_id = graph->register_tensor(std::move(handle));

    ViewDescriptor desc;
    desc.parent_id   = parent_id;
    desc.axes        = axes_ref;
    desc.result_rank = result_rank;
    desc.permutation = perm_ref;

    holder->params      = ctx.params_ptr();
    holder->parent_slot = parent_slot;
    holder->slot        = ctx.create_slot_for(slice_ref, slice_id);

    auto label = fmt::format("view_rt {} <- {}", handle.name, parent.name());

    // The closure is one pointer on purpose: it fits std::function's inline
    // buffer, so recording a view does not heap-allocate for the executor,
    // and the axes are not copied a second time into it. Everything the
    // replay reads lives in the holder, whose lifetime the graph owns.
    auto executor = [holder]() {
        auto const &axes = holder->axes;
        auto const &perm = holder->perm;
        if (!holder->params)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: no ParamTable bound to graph");

        using ParentType = std::remove_cvref_t<ParentT>;
        auto *parent_ptr = static_cast<ParentType *>(holder->parent_slot->ptr);
        if (parent_ptr == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: parent slot is null");
        if (parent_ptr->data() == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: parent has no backing data (deferred?)");

        // result rank == #non-drop axes. Scratch is reused across replays;
        // its capacity survives the clear.
        auto &slice_dims    = holder->scratch_dims;
        auto &slice_strides = holder->scratch_strides;
        slice_dims.clear();
        slice_strides.clear();
        std::ptrdiff_t ptr_offset = 0;

        for (size_t i = 0; i < axes.size(); ++i) {
            // Result reads parent axis p (perm[i], or i for identity); axes[i]
            // slices/drops that axis. Drop contributes only an offset (no result axis).
            size_t const p  = perm.empty() ? i : perm[i];
            auto const  &ax = axes[i];
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                slice_dims.push_back(parent_ptr->dim(p));
                slice_strides.push_back(parent_ptr->stride(p));
                break;
            case ViewAxis::Kind::Range: {
                std::int64_t const lo = ax.lo.resolve(*holder->params);
                std::int64_t const hi = ax.hi.resolve(*holder->params);
                if (lo < 0 || hi < lo || hi > static_cast<std::int64_t>(parent_ptr->dim(p)))
                    EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::view_runtime: range out of parent dim");
                ptr_offset += lo * static_cast<std::ptrdiff_t>(parent_ptr->stride(p));
                slice_dims.push_back(static_cast<size_t>(hi - lo));
                slice_strides.push_back(parent_ptr->stride(p));
                break;
            }
            case ViewAxis::Kind::Drop: {
                std::int64_t const idx = ax.lo.resolve(*holder->params);
                if (idx < 0 || idx >= static_cast<std::int64_t>(parent_ptr->dim(p)))
                    EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::view_runtime: drop index out of parent dim");
                ptr_offset += idx * static_cast<std::ptrdiff_t>(parent_ptr->stride(p));
                break; // dropped axis: offset only, no result axis
            }
            }
        }

        T *slice_data = parent_ptr->data() + ptr_offset;
        holder->view.emplace(::einsums::detail::TensorImpl<T>(slice_data, slice_dims, slice_strides));
        holder->slot->ptr = &holder->view.value();
    };

    ctx.record(OpKind::View, std::move(label), {parent_id}, {slice_id}, std::move(executor), std::move(desc));

    return slice_ref;
}

/// @brief Python-friendly wrapper over @ref view_runtime.
///
/// Takes a list of ``(lo, hi)`` integer pairs, one per parent axis. A pair
/// with both values < 0 (the conventional ``(-1, -1)``) means "full axis";
/// any other pair becomes a ``[lo, hi)`` range slice.
///
/// Must be called inside a capture context. The returned view is owned by the
/// active graph and aliases the parent: writes through it land in the parent,
/// and the graph's optimization passes treat it as a dependency on the parent.
///
/// @code
///   with cg.capture(g):
///       C_occ = einsums.graph.view(C, [(-1, -1), (0, nocc)])  # C[:, :nocc]
///       einsums.linalg.gemm(2.0, C_occ, C_occ, 0.0, D, trans_b=True)
/// @endcode
template <RuntimeRankTensorConcept ParentT>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_RVP(reference)
APIARY_INSTANTIATE_AS("view", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("view", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("view", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("view", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
// A view of a view (e.g. (A.T)[1:]) must also stay graph-registered and
// parent-aliasing, or the slice falls back to a raw eager view with
// aliases == 0, invisible to the scheduler's alias resolution, so a
// reduction over it can be reordered ahead of an in-place write to the
// owning tensor.
APIARY_INSTANTIATE_AS("view", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("view", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("view", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("view", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &view_python(
        ParentT &parent, std::vector<std::pair<std::int64_t, std::int64_t>> const &ranges) {
    if (ranges.size() != parent.rank()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::view: ranges length ({}) must equal parent rank ({})", ranges.size(),
                                parent.rank());
    }
    std::vector<ViewAxis> axes;
    axes.reserve(ranges.size());
    for (auto const &[lo, hi] : ranges) {
        if (lo < 0 && hi < 0) {
            axes.push_back(ViewAxis::full());
        } else {
            axes.push_back(ViewAxis::range(lo, hi));
        }
    }
    return view_runtime(parent, std::move(axes));
}

/// @brief Python-friendly indexed view supporting rank-reducing integer indices.
///
/// One ``(kind, a, b)`` triple per parent axis:
///   * ``kind == 0``: full axis (``a``/``b`` ignored)
///   * ``kind == 1``: range ``[a, b)``
///   * ``kind == 2``: drop, indexing the axis at ``a`` and removing it from the result
/// Result rank = parent rank − (number of drop axes). This is what backs
/// capture-mode rank-reducing reads like ``A[i]`` / ``A[:, j]`` (the plain
/// ``view`` above only does full/range, i.e. rank-preserving slices).
template <RuntimeRankTensorConcept ParentT>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_RVP(reference)
APIARY_INSTANTIATE_AS("view_indexed", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
// Rank-reducing index of a view (e.g. (A.T)[0]), same rationale as ``view``.
APIARY_INSTANTIATE_AS("view_indexed", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("view_indexed", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &view_indexed_python(
        ParentT &parent, std::vector<std::tuple<int, std::int64_t, std::int64_t>> const &specs) {
    if (specs.size() != parent.rank()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::view_indexed: specs length ({}) must equal parent rank ({})", specs.size(),
                                parent.rank());
    }
    std::vector<ViewAxis> axes;
    axes.reserve(specs.size());
    for (auto const &[kind, a, b] : specs) {
        switch (kind) {
        case 0:
            axes.push_back(ViewAxis::full());
            break;
        case 1:
            axes.push_back(ViewAxis::range(a, b));
            break;
        case 2:
            axes.push_back(ViewAxis::drop(a));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::view_indexed: unknown axis kind {} (expected 0=full, 1=range, 2=drop)",
                                    kind);
        }
    }
    return view_runtime(parent, std::move(axes));
}

/// @brief Batched @ref view_indexed_python: record N views of one parent in one call.
///
/// Same per-axis ``(kind, a, b)`` triples as ``view_indexed`` - ``0`` full,
/// ``1`` range ``[a, b)``, ``2`` drop at ``a`` - one inner list per view, and
/// the views come back in the order they were asked for.
///
/// This exists for callers that build many views of the same parent in a
/// loop, which is the shape of every batched capture: slicing per-pair blocks
/// out of a pair store issues thousands of these, and the per-call Python and
/// pybind overhead - argument normalization, overload resolution, the
/// crossing itself - was about a third of what each view cost. One crossing
/// for N views leaves only the recording. The recorded graph is identical to
/// N ``view_indexed`` calls; nothing downstream can tell the difference.
template <RuntimeRankTensorConcept ParentT>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_RVP(reference)
APIARY_INSTANTIATE_AS("views", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("views", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("views", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("views", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("views", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("views", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("views", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("views", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    std::vector<RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> *> views_python(
        ParentT &parent, std::vector<std::vector<std::tuple<int, std::int64_t, std::int64_t>>> const &specs) {
    std::vector<RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> *> out;
    out.reserve(specs.size());
    for (auto const &spec : specs) {
        out.push_back(&view_indexed_python(parent, spec));
    }
    return out;
}

/// @brief Python-friendly transpose / axis-permutation view.
///
/// Records a graph View whose result axis ``i`` aliases parent axis
/// ``perm[i]`` (``perm`` must be a permutation of ``[0, rank)``). The returned
/// view is graph-owned and aliases the parent. Unlike the raw
/// ``RuntimeTensor.transpose_view()``, which shares the parent's data pointer
/// and is invisible to the graph, a transposed tensor here can be fed to captured
/// gemm/axpy/etc. ops. This is what backs the capture-aware ``A.T``.
///
/// @code
///   with cg.capture(g):
///       At = einsums.graph.permute_view(A, [1, 0])   # A.T, graph-registered
///       einsums.linalg.gemm(1.0, At, B, 0.0, C)
/// @endcode
template <RuntimeRankTensorConcept ParentT>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_RVP(reference)
APIARY_INSTANTIATE_AS("permute_view", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("permute_view", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("permute_view", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("permute_view", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("permute_view", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("permute_view", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("permute_view", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("permute_view", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &permute_view_python(ParentT                   &parent,
                                                                                             std::vector<size_t> const &perm) {
    if (perm.size() != parent.rank()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::permute_view: perm length ({}) must equal parent rank ({})", perm.size(),
                                parent.rank());
    }
    std::vector<ViewAxis> axes(parent.rank(), ViewAxis::full());
    return view_runtime(parent, std::move(axes), perm);
}

// ─────────────────────────────────────────────────────────────────────────────
// tile_view: graph-registered DENSE view of one tile of a TiledRuntimeTensor.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Record a dense, parent-aliasing view of a single tile.
///
/// The tiled capture-view primitive. A tile is a full-rank dense block, so
/// the returned ``RuntimeTensorView`` feeds the ENTIRE dense op surface -
/// einsum, gemm, permute, axpby, ... - inside capture, which is exactly how
/// tiled workflows iterate (per tile block, the shape TiledExpansion itself
/// lowers to). Element-granular slicing of a tiled tensor is deliberately
/// not offered: a slice crossing tile boundaries has no single buffer.
///
/// The tile is created (deferred, dims fixed by the grid) at capture if
/// absent - value-equivalent, since absent tiles are rigorous zeros and a
/// created tile materializes zeroed. Each replay the executor re-resolves
/// the tile from the LIVE parent and re-emplaces the view, so parents whose
/// storage moves between replays (graph-owned deferred scratch released by
/// FreeInsertion and re-materialized) stay correct.
///
/// The recorded ViewDescriptor carries one constant ``[c, c+1)`` range per
/// axis IN TILE UNITS: the disjointness-aware hazard scan then proves views
/// of distinct tiles non-conflicting, so per-tile writes to one output can
/// be scheduled concurrently. Units stay consistent because element-unit
/// boxes cannot appear on a tiled parent (the dense view entry points do
/// not accept one).
///
/// @code
///   with cg.capture(g):
///       for i in range(2):
///           a = cg.tile_view(A, [i, 0])
///           c = cg.tile_view(C, [i, 0])
///           einsums.einsum("ij <- ik ; kj", c, a, B, c_pf=0.0, ab_pf=1.0)
/// @endcode
template <typename ParentT>
    requires IsTiledTensorV<std::remove_cvref_t<ParentT>>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_RVP(reference)
APIARY_INSTANTIATE_AS("tile_view", einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("tile_view", einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("tile_view", einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("tile_view", einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &tile_view_python(ParentT &parent, std::vector<int> const &coord) {
    using T      = typename std::remove_cvref_t<ParentT>::ValueType;
    using Holder = detail::RuntimeViewHolder<T>;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::tile_view called outside of capture (use tensor.tile_view(coord) for eager access)");
    }

    size_t const rank = parent.rank();
    if (coord.size() != rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::tile_view: coord length ({}) must equal parent rank ({})", coord.size(), rank);
    }
    auto const &sizes = parent.tile_sizes();
    for (size_t i = 0; i < rank; ++i) {
        if (coord[i] < 0 || coord[i] >= static_cast<int>(sizes[i].size())) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::tile_view: coord[{}] = {} outside the {}-tile grid axis", i, coord[i],
                                    sizes[i].size());
        }
    }

    // Register the parent BEFORE creating the tile: make_handle derives the
    // handle's AllocState from is_materialized(), and adding a deferred tile
    // first would flip a perfectly usable parent (empty = vacuously
    // materialized; executors materialize tiles per replay) into "Deferred",
    // which execute()'s validation then rejects without a Materialization run.
    auto [parent_id, parent_slot] = ctx.get_slot(parent);

    // Fix the tile's existence and dims now: a deferred tile keeps its shape
    // through a sentinel pointer, exactly the placeholder capture-time
    // validation needs (the same trick the dense deferred-parent path uses).
    auto &tile = parent.tile(coord);

    auto *holder = new Holder();
    holder->view.emplace(tile.impl());

    auto *graph = ctx.graph();
    if (graph == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::tile_view: no active graph");
    }
    graph->adopt([holder]() { delete holder; });

    RuntimeTensorView<T> &slice_ref = holder->view.value();
    auto                  handle    = make_handle(slice_ref, 0);
    handle.is_intermediate          = true;
    handle.name                     = fmt::format("tile_view({})", parent.name());
    handle.aliases                  = parent_id;
    detail::bind_holder_liveness(handle, holder);

    TensorId const slice_id = graph->register_tensor(std::move(handle));
    auto          *slot     = ctx.create_slot_for(slice_ref, slice_id);

    // One constant [c, c+1) range per axis, in TILE units: the hazard scan's
    // disjointness boxes prove distinct tiles non-conflicting.
    ViewDescriptor desc;
    desc.parent_id   = parent_id;
    desc.result_rank = rank;
    desc.axes.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        desc.axes.push_back(ViewAxis::range(coord[i], coord[i] + 1));
    }

    auto label = fmt::format("tile_view [{}] <- {}", fmt::join(coord, ","), parent.name());

    using ParentType = std::remove_cvref_t<ParentT>;
    auto executor    = [holder, coord, parent_slot, slot]() {
        auto *parent_ptr = static_cast<ParentType *>(parent_slot->ptr);
        if (parent_ptr == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::tile_view executor: parent slot is null");
        }
        auto &t = parent_ptr->tile(coord);
        t.materialize();
        holder->view.emplace(t.impl());
        slot->ptr = &holder->view.value();
    };

    ctx.record(OpKind::View, std::move(label), {parent_id}, {slice_id}, std::move(executor), std::move(desc));

    return slice_ref;
}

// ─────────────────────────────────────────────────────────────────────────────
// write_param: explicit dataflow write into a Pipeline parameter.
// ─────────────────────────────────────────────────────────────────────────────
//
// Two forms:
//   1) From a graph scalar tensor → reads the value at execute time and stores
//      it in the active ParamTable under @p name. Adds a graph dependency on
//      the source so subsequent View ops referencing @p name execute in the
//      correct order.
//   2) From a free-standing callback → invokes the callback at execute time
//      and stores its return value. Useful when the value is computed by
//      external (non-graph) code; no graph dependency is created.

/// Write a parameter from a scalar variable produced by the graph.
///
/// The value of @p source (an arithmetic scalar variable) is read at execute
/// time and stored in ``params[name]``. The graph captures ``&source``, so
/// any upstream op that registers the same address as a scalar input (via
/// ``ctx.get_or_register_scalar``) becomes a true dataflow predecessor: the
/// scheduler orders the producer before ``write_param``, and ``write_param``
/// before any downstream View op that reads ``Param(name)``.
template <typename T>
    requires std::is_arithmetic_v<T>
void write_param(std::string name, T &source) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing())
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param called outside of capture");

    TensorId const source_id = ctx.get_or_register_scalar(&source, fmt::format("write_param_src({})", name));

    WriteParamDescriptor desc;
    desc.name        = name;
    desc.source_id   = source_id;
    desc.source_type = param_source_type<T>();

    auto label = fmt::format("write_param: {} <- scalar", name);

    // The executor no longer closes over ``source``. It reads the address the
    // graph holds for that scalar on every call, so repointing the handle moves
    // the read - which is the whole point of the source being a graph tensor
    // rather than a captured reference.
    //
    // Rank is keyed on the destination, and this node's destination is a
    // ParamTable entry that no tensor names: 0.
    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::WriteParam, packed_gemm::get_scalar_type<T>(), 0, op_data, *ctx.graph(),
                                     std::span<TensorId const>{&source_id, 1}, {});

    ctx.record(OpKind::WriteParam, std::move(label), {source_id}, {}, std::move(executor), std::move(op_data));
}

/// Write a parameter from a runtime scalar EXPRESSION evaluated at execute time.
///
/// No graph dependency on a tensor is created. A @ref BoundExpr::Param source
/// does create a parameter-ordering edge (see @ref param_reads), so a chain of
/// parameter writes is scheduled correctly; a literal and a callback name
/// nothing and are unordered against anything.
///
/// This is the named-callable rule applied to the last of the four closures a
/// graph can carry: a runtime scalar is a literal, a named parameter, or a
/// callback, with only the last unserializable. @c OpKind::WriteParam is a reconstructible KIND either
/// way, and @ref Graph::serializability_report names a callback-arm node
/// individually.
inline void write_param(std::string name, BoundExpr source) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing())
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param called outside of capture");

    char const *arm = source.is_const() ? "const" : (source.is_param() ? "param" : "callback");

    WriteParamDescriptor desc;
    desc.name        = name;
    desc.source_id   = 0;
    desc.source_expr = std::move(source);

    auto label = fmt::format("write_param: {} <- {}", name, arm);

    // Rank is keyed on the destination, and this node's destination is a
    // ParamTable entry that no tensor names: 0. The expression arm reads no
    // operand at all, so there is no dtype either.
    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::WriteParam, packed_gemm::ScalarType::Unknown, 0, op_data, *ctx.graph(), {}, {});

    ctx.record(OpKind::WriteParam, std::move(label), {}, {}, std::move(executor), std::move(op_data));
}

/// Write a parameter from an arbitrary callback evaluated at execute time.
///
/// A thin spelling of the @ref BoundExpr form above, kept because a bare lambda
/// cannot reach a ``BoundExpr`` parameter implicitly (that would be two
/// user-defined conversions). No graph dependency is created; the callback is
/// treated as opaque. Use the tensor-source form when the value is produced by
/// upstream graph ops and you want correct scheduling.
inline void write_param(std::string name, std::function<std::int64_t()> source_fn) {
    write_param(std::move(name), BoundExpr(std::move(source_fn)));
}

EINSUMS_NAMESPACE_END(compute_graph)
