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
/// Reads the tensor's storage through @ref TensorHandle::impl_fn at run time rather than any
/// pointer known now, since the memory planner may place it anywhere. The backing writes its
/// storage's column-major order, and a tensor laid out otherwise (row-major, as Python declares
/// one) is filled through a scatter by its strides.
std::function<void()> load_whole(TensorHandle const &handle) {
    if (!handle.backing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): tensor '{}' is initialized from a backing but carries none",
                                handle.name);
    }
    if (!handle.impl_fn) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(Initialize): backed tensor '{}' has no storage accessor to fill",
                                handle.name);
    }
    return [backing = handle.backing, impl_fn = handle.impl_fn, dtype = handle.dtype, name = handle.name]() {
        dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            auto *impl = static_cast<einsums::detail::TensorImpl<T> *>(impl_fn());
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
    return [mat_fn = handle.materialize_fn, resize_fn = handle.resize_deferred_fn, set_dist_fn = handle.set_distribution_fn,
            is_dist = handle.is_distributed && !handle.is_replicated, dist_info = handle.distribution_info,
            held_fn = handle.is_materialized_fn, bytes = static_cast<std::uint64_t>(handle.total_bytes()), context = std::move(context)]() {
        bool const held_before = held_fn ? held_fn() : true;
        if (is_dist && resize_fn && dist_info) {
            auto const desc       = std::static_pointer_cast<comm::DistributionDescriptor>(dist_info);
            int const  rank       = comm::world_rank();
            auto const local_dims = desc->local_dims_for(rank);
            resize_fn(local_dims);

            if (set_dist_fn) {
                std::vector<size_t> offsets(desc->dim_to_axis.size());
                for (size_t d = 0; d < desc->dim_to_axis.size(); d++) {
                    auto [start, end] = desc->local_range(d, rank);
                    offsets[d]        = start;
                }
                set_dist_fn(desc->global_dims, offsets);
            }
        }
        if (mat_fn) {
            mat_fn();
        }
        if (context && !held_before && held_fn && held_fn()) {
            context->allocated(bytes);
        }
    };
}

std::function<void()> build_free(TensorHandle const &handle, std::shared_ptr<RunContext> context) {
    // No logging here: this fires once per freed tensor per REPLAY.
    return [rel_fn = handle.release_fn, held_fn = handle.is_materialized_fn, bytes = static_cast<std::uint64_t>(handle.total_bytes()),
            context = std::move(context)]() {
        bool const held_before = held_fn && held_fn();
        if (rel_fn) {
            rel_fn();
        }
        if (context && held_before && !held_fn()) {
            context->released(bytes);
        }
    };
}

std::function<void()> build_attach(TensorHandle const &handle, std::shared_ptr<ArenaBlock> arena, std::size_t offset) {
    return [arena = std::move(arena), offset, attach = handle.materialize_into_fn]() { attach(static_cast<void *>(arena->data + offset)); };
}

std::function<void()> build_detach(TensorHandle const &handle) {
    return [rel = handle.release_fn]() { rel(); };
}

std::function<void()> build_initialize(InitKind init, TensorHandle const &handle) {
    switch (init) {
    case InitKind::Zero:
        return [zero_fn = handle.zero_fn]() {
            if (zero_fn) {
                zero_fn();
            }
        };
    case InitKind::Random:
        return [random_fn = handle.random_fn]() {
            if (random_fn) {
                random_fn();
            }
        };
    case InitKind::FromDisk:
        return load_whole(handle);
    case InitKind::None:
        break;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "build_executor(Initialize): tensor '{}' has an Initialize node that initializes nothing", handle.name);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
