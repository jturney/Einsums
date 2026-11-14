//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "LifecycleExecutors.hpp"

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ArenaBlock.hpp"
#include "Passes/BackingFill.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

TensorHandle const &LifecycleTarget::handle() const {
    return graph->graph().tensor(id);
}

TensorHandle const *LifecycleTarget::graph_owned() const {
    TensorHandle const &now = handle();
    return now.caller_bound ? nullptr : &now;
}

namespace {

/// The executor that fills a backed tensor whole from its backing, after its Materialize.
///
/// Reads the tensor's storage through its type's @ref TensorOps::impl at run time rather than any
/// pointer known now, since the memory planner may place it anywhere. The backing writes its
/// storage's column-major order, and a tensor laid out otherwise (row-major, as Python declares
/// one) is filled through a scatter by its strides.
std::function<void()> load_whole(LifecycleTarget target) {
    TensorHandle const &handle = target.handle();
    if (!handle.backing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): tensor '{}' is initialized from a backing but carries none",
                                handle.name);
    }
    if (!handle.has_impl()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): backed tensor '{}' has no storage accessor to fill",
                                handle.name);
    }
    return [target = std::move(target)]() {
        TensorHandle const *now = target.graph_owned();
        if (now == nullptr || !now->backing) {
            return;
        }
        dispatch_scalar_type(now->dtype, [&]<typename T>(T /*tag*/) {
            auto *impl = static_cast<einsums::detail::TensorImpl<T> *>(now->impl());
            if (impl == nullptr || (impl->data() == nullptr && impl->size() != 0)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error, "backed tensor '{}' has no storage to fill from {}", now->name,
                                        now->backing->describe());
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(now->backing->dims.size());
            for (std::size_t const extent : now->backing->dims) {
                ranges.emplace_back(0, extent);
            }
            passes::detail::fill_from_backing(*now->backing, impl->data(), *impl, ranges);
        });
    };
}

/// Resize a distributed tensor to this rank's partition before it is allocated. Only for a
/// tensor this graph declared; see TensorHandle::resizable.
void resize_to_partition(TensorHandle const &handle, void *object) {
    TensorOps const &ops = *handle.ops;
    if (!handle.resizable || !handle.is_distributed || handle.is_replicated || !handle.distribution_info ||
        ops.resize_deferred == nullptr) {
        return;
    }
    auto const desc       = std::static_pointer_cast<comm::DistributionDescriptor>(handle.distribution_info);
    int const  rank       = comm::world_rank();
    auto const local_dims = desc->local_dims_for(rank);
    ops.resize_deferred(object, local_dims);

    if (ops.set_distribution != nullptr) {
        std::vector<size_t> offsets(desc->dim_to_axis.size());
        for (size_t d = 0; d < desc->dim_to_axis.size(); d++) {
            auto [start, end] = desc->local_range(d, rank);
            offsets[d]        = start;
        }
        ops.set_distribution(object, desc->global_dims, offsets);
    }
}

} // namespace

std::function<void()> build_materialize(LifecycleTarget target, std::shared_ptr<RunContext> context) {
    return [target = std::move(target), context = std::move(context)]() {
        TensorHandle const *handle = target.graph_owned();
        if (handle == nullptr) {
            return;
        }
        TensorOps const &ops         = *handle->ops;
        void *const      object      = handle->live_ptr();
        bool const       held_before = ops.is_materialized == nullptr || ops.is_materialized(object);
        resize_to_partition(*handle, object);
        if (ops.materialize != nullptr) {
            ops.materialize(object);
        }
        if (context && !held_before && ops.is_materialized(object)) {
            context->allocated(static_cast<std::uint64_t>(handle->total_bytes()));
        }
    };
}

std::function<void()> build_free(LifecycleTarget target, std::shared_ptr<RunContext> context) {
    // No logging here: this fires once per freed tensor per REPLAY.
    return [target = std::move(target), context = std::move(context)]() {
        TensorHandle const *handle = target.graph_owned();
        if (handle == nullptr) {
            return;
        }
        TensorOps const &ops         = *handle->ops;
        void *const      object      = handle->live_ptr();
        bool const       held_before = ops.is_materialized != nullptr && ops.is_materialized(object);
        if (ops.release != nullptr) {
            ops.release(object);
        }
        if (context && held_before && !ops.is_materialized(object)) {
            context->released(static_cast<std::uint64_t>(handle->total_bytes()));
        }
    };
}

std::function<void()> build_attach(LifecycleTarget target, std::shared_ptr<ArenaBlock> arena, std::size_t offset) {
    return [target = std::move(target), arena = std::move(arena), offset]() {
        if (TensorHandle const *handle = target.graph_owned(); handle != nullptr) {
            handle->ops->materialize_into(handle->live_ptr(), static_cast<void *>(arena->data + offset));
        }
    };
}

std::function<void()> build_detach(LifecycleTarget target) {
    return [target = std::move(target)]() {
        if (TensorHandle const *handle = target.graph_owned(); handle != nullptr) {
            handle->ops->release(handle->live_ptr());
        }
    };
}

std::function<void()> build_initialize(InitKind init, LifecycleTarget target) {
    TensorHandle const &handle = target.handle();
    switch (init) {
    case InitKind::Zero:
        if (handle.ops->fill_zero == nullptr) {
            return [] {};
        }
        return [target = std::move(target)]() {
            if (TensorHandle const *now = target.graph_owned(); now != nullptr) {
                now->ops->fill_zero(now->live_ptr());
            }
        };
    case InitKind::Random:
        if (handle.ops->fill_random == nullptr) {
            return [] {};
        }
        return [target = std::move(target)]() {
            if (TensorHandle const *now = target.graph_owned(); now != nullptr) {
                now->ops->fill_random(now->live_ptr());
            }
        };
    case InitKind::Custom:
        if (!handle.custom_init) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): tensor '{}' has a custom initializer and no function",
                                    handle.name);
        }
        return [target = std::move(target)]() {
            if (TensorHandle const *now = target.graph_owned(); now != nullptr && now->custom_init) {
                now->custom_init(now->live_ptr());
            }
        };
    case InitKind::FromDisk:
        return load_whole(std::move(target));
    case InitKind::None:
        break;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "build_executor(Initialize): tensor '{}' has an Initialize node that initializes nothing", handle.name);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
