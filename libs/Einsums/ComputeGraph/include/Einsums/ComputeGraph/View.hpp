//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace einsums::compute_graph {

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
/// **v1 limitations** (intentional, documented for follow-up):
///   - Only @ref ViewAxis::Kind::Full and @ref ViewAxis::Kind::Range are
///     supported. ``Drop`` (rank-reducing index) will be added later;
///     calling ``cg::view`` with a Drop axis throws at capture.
///   - Only rank-preserving slices: ``ResultRank == parent.rank()``.
///   - Constant strides — the slice inherits the parent's strides. No
///     stride remapping or transpose-via-view.
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
};

/// Runtime-rank counterpart to ViewHolder. Holds an ``std::optional<RuntimeTensorView<T>>``
/// whose address is stable across re-emplaces, so the executor can refresh
/// the slice's data pointer / dims / strides each iteration without
/// invalidating slots that downstream ops captured by reference.
template <typename T>
struct RuntimeViewHolder {
    std::optional<RuntimeTensorView<T>> view;
};

} // namespace detail

/// @brief Record a non-owning slice of a tensor.
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
    using Holder = detail::ViewHolder<T, Rank>;

    static_assert(sizeof...(Axes) == Rank, "cg::view requires one ViewAxis per parent rank (rank-preserving slices only in v1)");
    static_assert(std::is_same_v<typename ParentT::ValueType, T>, "cg::view: T must match parent's ValueType");

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view called outside of capture");
    }

    // Collect axes as a vector so we can do runtime work with them.
    std::vector<ViewAxis> axis_vec{ViewAxis(std::forward<Axes>(axes))...};
    for (auto const &ax : axis_vec) {
        if (ax.kind == ViewAxis::Kind::Drop)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: ViewAxis::Drop is not yet supported");
    }

    // Allocate a holder on the heap. Owned by the graph so its address
    // is stable across the lifetime of every executor that captured it.
    auto *holder = new Holder;

    // At capture time we don't yet have a ParamTable to resolve Param
    // bounds against. Emplace a "full parent" view as a placeholder so
    // the holder's view has a valid object — its address is what
    // downstream ops will capture. The first execute() will re-emplace
    // with the actual computed bounds.
    Stride<Rank> parent_strides;
    Dim<Rank>    parent_dims;
    for (size_t i = 0; i < Rank; ++i) {
        parent_dims[i]    = parent.dim(i);
        parent_strides[i] = parent.stride(i);
    }
    holder->view.emplace(parent.data(), parent_dims, parent_strides);

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

    TensorId const slice_id = graph->register_tensor(std::move(handle));
    auto          *slot     = ctx.create_slot_for(slice_ref, slice_id);

    // Build the View descriptor for passes to inspect.
    ViewDescriptor desc;
    desc.parent_id   = parent_id;
    desc.axes        = axis_vec;
    desc.result_rank = Rank;

    // Capture references to the resolved-at-execute-time pieces.
    auto params_ptr = ctx.params_ptr();

    auto label = fmt::format("view {} <- {}", handle.name, parent.name());

    auto executor = [holder, axis_vec, parent_slot, params_ptr, slot]() {
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
            auto const &ax   = axis_vec[i];
            slice_strides[i] = parent_ptr->stride(i);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                offsets[i]    = 0;
                slice_dims[i] = parent_ptr->dim(i);
                break;
            case ViewAxis::Kind::Range: {
                std::int64_t const lo = ax.lo.resolve(*params_ptr);
                std::int64_t const hi = ax.hi.resolve(*params_ptr);
                if (lo < 0 || hi < lo || hi > static_cast<std::int64_t>(parent_ptr->dim(i)))
                    EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::view: range out of parent dim");
                offsets[i]    = lo;
                slice_dims[i] = static_cast<size_t>(hi - lo);
                break;
            }
            case ViewAxis::Kind::Drop:
                EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view: Drop axis not supported in v1");
            }
            ptr_offset += offsets[i] * static_cast<std::ptrdiff_t>(parent_ptr->stride(i));
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

/// Type-deducing overload — infers @c T and @c Rank from @p parent. The
/// codegen path emits this form so it doesn't have to spell out
/// ``view<typename T::ValueType, T::Rank>(...)`` boilerplate.
template <CoreBasicTensorConcept ParentT, typename... Axes>
    requires HasCompileTimeRank<ParentT>
TensorView<typename std::remove_cvref_t<ParentT>::ValueType, std::remove_cvref_t<ParentT>::Rank> &view(ParentT &parent, Axes &&...axes) {
    using T            = typename std::remove_cvref_t<ParentT>::ValueType;
    constexpr size_t R = std::remove_cvref_t<ParentT>::Rank;
    return view<T, R, ParentT>(parent, std::forward<Axes>(axes)...);
}

/// @brief Runtime-rank counterpart to cg::view: record a non-owning slice
///        of a RuntimeTensor parent.
///
/// Same v1 limitations as the typed view (rank-preserving, no Drop axis,
/// constant strides inherited from parent). The slice's rank is read from
/// the parent at capture time; the executor rebuilds the slice each
/// iteration by re-emplacing a heap-stored RuntimeTensorView with new
/// pointer / dims / strides.
template <RuntimeRankTensorConcept ParentT>
RuntimeTensorView<typename std::remove_cvref_t<ParentT>::ValueType> &view_runtime(ParentT &parent, std::vector<ViewAxis> const &axis_vec) {
    using T      = typename std::remove_cvref_t<ParentT>::ValueType;
    using Holder = detail::RuntimeViewHolder<T>;

    size_t const rank = parent.rank();
    if (axis_vec.size() != rank) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime requires one ViewAxis per parent rank ({} given, parent has rank {})",
                                axis_vec.size(), rank);
    }
    for (auto const &ax : axis_vec) {
        if (ax.kind == ViewAxis::Kind::Drop)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: ViewAxis::Drop is not yet supported");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime called outside of capture");
    }

    auto *holder = new Holder;

    // Placeholder: build a "full parent" view so the holder's address is
    // valid before the first execute() runs.
    std::vector<size_t> parent_dims(rank), parent_strides(rank);
    for (size_t i = 0; i < rank; ++i) {
        parent_dims[i]    = parent.dim(i);
        parent_strides[i] = parent.stride(i);
    }
    holder->view.emplace(::einsums::detail::TensorImpl<T>(const_cast<T *>(parent.data()), parent_dims, parent_strides));

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

    TensorId const slice_id = graph->register_tensor(std::move(handle));
    auto          *slot     = ctx.create_slot_for(slice_ref, slice_id);

    ViewDescriptor desc;
    desc.parent_id   = parent_id;
    desc.axes        = axis_vec;
    desc.result_rank = rank;

    auto params_ptr = ctx.params_ptr();
    auto label      = fmt::format("view_rt {} <- {}", handle.name, parent.name());

    auto executor = [holder, axis_vec, parent_slot, params_ptr, slot, rank]() {
        if (!params_ptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: no ParamTable bound to graph");

        using ParentType = std::remove_cvref_t<ParentT>;
        auto *parent_ptr = static_cast<ParentType *>(parent_slot->ptr);
        if (parent_ptr == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: parent slot is null");
        if (parent_ptr->data() == nullptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime executor: parent has no backing data (deferred?)");

        std::vector<size_t> slice_dims(rank), slice_strides(rank);
        std::ptrdiff_t      ptr_offset = 0;

        for (size_t i = 0; i < rank; ++i) {
            auto const &ax   = axis_vec[i];
            slice_strides[i] = parent_ptr->stride(i);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                slice_dims[i] = parent_ptr->dim(i);
                break;
            case ViewAxis::Kind::Range: {
                std::int64_t const lo = ax.lo.resolve(*params_ptr);
                std::int64_t const hi = ax.hi.resolve(*params_ptr);
                if (lo < 0 || hi < lo || hi > static_cast<std::int64_t>(parent_ptr->dim(i)))
                    EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::view_runtime: range out of parent dim");
                ptr_offset += lo * static_cast<std::ptrdiff_t>(parent_ptr->stride(i));
                slice_dims[i] = static_cast<size_t>(hi - lo);
                break;
            }
            case ViewAxis::Kind::Drop:
                EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::view_runtime: Drop axis not supported in v1");
            }
        }

        T *slice_data = parent_ptr->data() + ptr_offset;
        holder->view.emplace(::einsums::detail::TensorImpl<T>(slice_data, slice_dims, slice_strides));
        slot->ptr = &holder->view.value();
    };

    ctx.record(OpKind::View, std::move(label), {parent_id}, {slice_id}, std::move(executor), std::move(desc));

    return slice_ref;
}

// ─────────────────────────────────────────────────────────────────────────────
// write_param — explicit dataflow write into a Pipeline parameter.
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

/// Write a parameter from a scalar tensor produced by the graph.
///
/// The scalar's value is read at execute time and stored in
/// ``params[name]``. Adds an explicit dataflow edge from @p source's
/// producer to all downstream View ops that read ``Param(name)``.
template <typename T>
    requires std::is_arithmetic_v<T>
void write_param(std::string name, T &source) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing())
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param called outside of capture");

    TensorId const source_id  = ctx.get_or_register_scalar(&source, fmt::format("write_param_src({})", name));
    auto           params_ptr = ctx.params_ptr();

    WriteParamDescriptor desc;
    desc.name      = name;
    desc.source_id = source_id;

    auto label = fmt::format("write_param: {} <- scalar", name);

    auto executor = [name, &source, params_ptr]() {
        if (!params_ptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param executor: no ParamTable bound to graph");
        params_ptr->set(name, static_cast<std::int64_t>(source));
    };

    ctx.record(OpKind::WriteParam, std::move(label), {source_id}, {}, std::move(executor), std::move(desc));
}

/// Write a parameter from an arbitrary callback evaluated at execute time.
///
/// No graph dependency is created — the callback is treated as opaque. Use
/// the tensor-source form when the value is produced by upstream graph ops
/// and you want correct scheduling.
inline void write_param(std::string name, std::function<std::int64_t()> source_fn) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing())
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param called outside of capture");

    auto params_ptr = ctx.params_ptr();

    WriteParamDescriptor desc;
    desc.name      = name;
    desc.source_id = 0;
    desc.source_fn = source_fn;

    auto label = fmt::format("write_param: {} <- callback", name);

    auto executor = [name, source_fn = std::move(source_fn), params_ptr]() {
        if (!params_ptr)
            EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::write_param executor: no ParamTable bound to graph");
        params_ptr->set(name, source_fn());
    };

    ctx.record(OpKind::WriteParam, std::move(label), {}, {}, std::move(executor), std::move(desc));
}

} // namespace einsums::compute_graph
