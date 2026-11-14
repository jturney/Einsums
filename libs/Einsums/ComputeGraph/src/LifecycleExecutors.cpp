//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "LifecycleExecutors.hpp"

#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
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

namespace {

/// The executor that fills a backed tensor whole from its backing, after its Materialize.
///
/// Reads the tensor's storage through its type's @ref TensorOps::impl at run time rather than any
/// pointer known now, since the memory planner may place it anywhere. The backing writes its
/// storage's column-major order, and a tensor laid out otherwise (row-major, as Python declares
/// one) is filled through a scatter by its strides.
std::function<void()> load_whole(TensorHandle const &handle) {
    if (!handle.backing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): tensor '{}' is initialized from a backing but carries none",
                                handle.name);
    }
    if (!handle.has_impl()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): backed tensor '{}' has no storage accessor to fill",
                                handle.name);
    }
    return [backing = handle.backing, ops = handle.ops, object = handle.live_ptr(), dtype = handle.dtype, name = handle.name]() {
        dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            auto *impl = static_cast<einsums::detail::TensorImpl<T> *>(ops->impl(object));
            if (impl == nullptr || (impl->data() == nullptr && impl->size() != 0)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error, "backed tensor '{}' has no storage to fill from {}", name, backing->describe());
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(backing->dims.size());
            for (std::size_t const extent : backing->dims) {
                ranges.emplace_back(0, extent);
            }
            passes::detail::fill_from_backing(*backing, impl->data(), *impl, ranges);
        });
    };
}

} // namespace

std::function<void()> build_materialize(TensorHandle const &handle, std::shared_ptr<RunContext> context) {
    // Resizing for a distribution is this graph's to do only for a tensor it declared; see
    // TensorHandle::resizable.
    auto const resize_fn   = handle.resizable ? handle.ops->resize_deferred : nullptr;
    auto const set_dist_fn = handle.resizable ? handle.ops->set_distribution : nullptr;
    return
        [ops = handle.ops, object = handle.live_ptr(), resize_fn, set_dist_fn, is_dist = handle.is_distributed && !handle.is_replicated,
         dist_info = handle.distribution_info, bytes = static_cast<std::uint64_t>(handle.total_bytes()), context = std::move(context)]() {
            auto const held        = [ops, object] { return ops->is_materialized(object); };
            bool const held_before = ops->is_materialized != nullptr ? held() : true;
            if (is_dist && resize_fn != nullptr && dist_info) {
                auto const desc       = std::static_pointer_cast<comm::DistributionDescriptor>(dist_info);
                int const  rank       = comm::world_rank();
                auto const local_dims = desc->local_dims_for(rank);
                resize_fn(object, local_dims);

                if (set_dist_fn != nullptr) {
                    std::vector<size_t> offsets(desc->dim_to_axis.size());
                    for (size_t d = 0; d < desc->dim_to_axis.size(); d++) {
                        auto [start, end] = desc->local_range(d, rank);
                        offsets[d]        = start;
                    }
                    set_dist_fn(object, desc->global_dims, offsets);
                }
            }
            if (ops->materialize != nullptr) {
                ops->materialize(object);
            }
            if (context && !held_before && ops->is_materialized != nullptr && held()) {
                context->allocated(bytes);
            }
        };
}

std::function<void()> build_free(TensorHandle const &handle, std::shared_ptr<RunContext> context) {
    // No logging here: this fires once per freed tensor per REPLAY.
    return [ops = handle.ops, object = handle.live_ptr(), bytes = static_cast<std::uint64_t>(handle.total_bytes()),
            context = std::move(context)]() {
        bool const held_before = ops->is_materialized != nullptr && ops->is_materialized(object);
        if (ops->release != nullptr) {
            ops->release(object);
        }
        if (context && held_before && !ops->is_materialized(object)) {
            context->released(bytes);
        }
    };
}

std::function<void()> build_attach(TensorHandle const &handle, std::shared_ptr<ArenaBlock> arena, std::size_t offset) {
    return [arena = std::move(arena), offset, attach = handle.ops->materialize_into, object = handle.live_ptr()]() {
        attach(object, static_cast<void *>(arena->data + offset));
    };
}

std::function<void()> build_detach(TensorHandle const &handle) {
    return [release = handle.ops->release, object = handle.live_ptr()]() { release(object); };
}

std::function<void()> build_initialize(InitKind init, TensorHandle const &handle) {
    switch (init) {
    case InitKind::Zero:
        if (handle.ops->fill_zero == nullptr) {
            return [] {};
        }
        return [fill = handle.ops->fill_zero, object = handle.live_ptr()]() { fill(object); };
    case InitKind::Random:
        if (handle.ops->fill_random == nullptr) {
            return [] {};
        }
        return [fill = handle.ops->fill_random, object = handle.live_ptr()]() { fill(object); };
    case InitKind::Custom:
        if (!handle.custom_init) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): tensor '{}' has a custom initializer and no function",
                                    handle.name);
        }
        return [fill = handle.custom_init, object = handle.live_ptr()]() { fill(object); };
    case InitKind::FromDisk:
        return load_whole(handle);
    case InitKind::None:
        break;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "build_executor(Initialize): tensor '{}' has an Initialize node that initializes nothing", handle.name);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
