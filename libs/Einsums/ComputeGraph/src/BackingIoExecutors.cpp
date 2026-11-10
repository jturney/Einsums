//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "BackingIoExecutors.hpp"

#include <Einsums/ComputeGraph/Detail/RunContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Observer.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "Passes/BackingFill.hpp"
#include "Passes/LoopEmission.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

std::function<void()> build_store_whole(TensorHandle const &handle) {
    if (!handle.backing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(DiskWrite): tensor '{}' is stored whole into a backing but carries none",
                                handle.name);
    }
    return [backing = handle.backing, impl_fn = handle.impl_fn, dtype = handle.dtype, name = handle.name]() {
        dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            auto const *impl = static_cast<einsums::detail::TensorImpl<T> const *>(impl_fn());
            if (impl == nullptr || (impl->data() == nullptr && impl->size() != 0)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error, "write-backed tensor '{}' has no storage to store into {}", name,
                                        backing->describe());
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(backing->dims.size());
            for (std::size_t const extent : backing->dims) {
                ranges.emplace_back(0, extent);
            }
            passes::detail::store_to_backing(*backing, impl->data(), *impl, ranges);
        });
    };
}

std::function<void()> build_publish(std::vector<std::shared_ptr<TensorBacking const>> backings) {
    if (backings.empty()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor(DiskWrite): a publish names no backing");
    }
    return [backings = std::move(backings)]() {
        std::vector<TensorBacking const *> members;
        members.reserve(backings.size());
        for (auto const &backing : backings) {
            members.push_back(backing.get());
        }
        members.front()->publish(members);
        for (TensorBacking const *member : members) {
            if (member->counters) {
                member->counters->publishes.fetch_add(1, std::memory_order_relaxed);
                if (member->counters->context) {
                    notify_io(*member->counters->context, EventKind::Publish, member->counters->tensor, *member, 0);
                }
            }
        }
    };
}

namespace {

/// A box: the region the window's axes select, resolved against the parameters when it runs.
std::function<void()> build_box(bool store, TensorHandle const &handle, std::shared_ptr<BackingWindow const> window,
                                std::shared_ptr<ParamTable> params) {
    return [window = std::move(window), impl_fn = handle.impl_fn, params = std::move(params), store]() {
        auto const &backing = window->backing;
        auto const &axes    = window->axes;
        auto const &name    = window->name;
        if (!params) {
            EINSUMS_THROW_EXCEPTION(std::logic_error, "SlabStreaming: the box of '{}' has no parameter table", name);
        }
        dispatch_scalar_type(backing->dtype, [&]<typename T>(T) {
            auto *impl = static_cast<einsums::detail::TensorImpl<T> *>(impl_fn());
            if (impl == nullptr || (impl->data() == nullptr && impl->size() != 0)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error, "SlabStreaming: the box of '{}' has no storage", name);
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(axes.size());
            std::size_t kept_axis = 0;
            for (std::size_t p = 0; p < axes.size(); ++p) {
                auto const        &axis   = axes[p];
                std::int64_t const extent = static_cast<std::int64_t>(backing->dims[p]);
                std::int64_t       lo     = 0;
                std::int64_t       hi     = extent;
                if (axis.kind == ViewAxis::Kind::Drop) {
                    lo = axis.lo.resolve(*params);
                    hi = lo + 1;
                } else if (axis.kind == ViewAxis::Kind::Range) {
                    lo = axis.lo.resolve(*params);
                    hi = axis.hi.resolve(*params);
                }
                bool const fits_buffer =
                    axis.kind == ViewAxis::Kind::Drop || static_cast<std::size_t>(hi - lo) == impl->dim(static_cast<int>(kept_axis));
                if (lo < 0 || hi < lo || hi > extent || !fits_buffer) {
                    EINSUMS_THROW_EXCEPTION(std::logic_error, "SlabStreaming: the box of '{}' on axis {} is [{}, {}), which does not fit",
                                            name, p, lo, hi);
                }
                if (axis.kind != ViewAxis::Kind::Drop) {
                    ++kept_axis;
                }
                ranges.emplace_back(static_cast<std::size_t>(lo), static_cast<std::size_t>(hi));
            }
            if (store) {
                passes::detail::store_to_backing(*backing, impl->data(), *impl, ranges);
            } else {
                passes::detail::fill_from_backing(*backing, impl->data(), *impl, ranges);
            }
        });
    };
}

/// A slab: a run of the backing's last axis that advances with the loop. Each bound follows the
/// loop with a cursor of its own, as a parameter-bound view's would, and a fresh cursor starts
/// where a finished sweep leaves one, since the decode is periodic in the slice count.
std::function<void()> build_slab(bool store, TensorHandle const &handle, std::shared_ptr<BackingWindow const> window) {
    auto const &slab  = *window->slab;
    auto        first = passes::detail::slice_bound(std::make_shared<passes::detail::TileCursor>(), 1, 0, 1, slab.count, slab.step,
                                                    static_cast<std::int64_t>(slab.offset));
    auto        last  = passes::detail::slice_bound(std::make_shared<passes::detail::TileCursor>(), 1, 0, 1, slab.count, slab.step,
                                                    static_cast<std::int64_t>(slab.offset) + static_cast<std::int64_t>(slab.length));
    return [window = std::move(window), impl_fn = handle.impl_fn, dtype = handle.dtype, first = std::move(first), last = std::move(last),
            store]() {
        auto const &backing = window->backing;
        auto const &dims    = backing->dims;
        dispatch_scalar_type(dtype, [&]<typename T>(T) {
            auto *impl = static_cast<einsums::detail::TensorImpl<T> *>(impl_fn());
            if (impl == nullptr || (impl->data() == nullptr && impl->size() != 0)) {
                EINSUMS_THROW_EXCEPTION(std::logic_error, "SlabStreaming: the slab of '{}' has no storage", window->name);
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(dims.size());
            for (std::size_t p = 0; p + 1 < dims.size(); ++p) {
                ranges.emplace_back(0, dims[p]);
            }
            auto const start = static_cast<std::size_t>(first());
            auto const end   = static_cast<std::size_t>(last());
            ranges.emplace_back(start, end);
            if (store) {
                passes::detail::store_to_backing(*backing, impl->data(), *impl, ranges);
            } else {
                passes::detail::fill_from_backing(*backing, impl->data(), *impl, ranges);
            }
        });
    };
}

} // namespace

std::function<void()> build_window(bool store, TensorHandle const &handle, std::shared_ptr<BackingWindow const> window,
                                   std::shared_ptr<ParamTable> params) {
    if (!window || !window->backing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor({}): transfer of '{}' names no backing window",
                                store ? "DiskWrite" : "DiskRead", handle.name);
    }
    if (window->slab.has_value() == !window->axes.empty()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor({}): the window of '{}' must be exactly one of a box and a slab",
                                store ? "DiskWrite" : "DiskRead", window->name);
    }
    if (!handle.impl_fn) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "build_executor({}): '{}' has no storage accessor", store ? "DiskWrite" : "DiskRead",
                                handle.name);
    }
    if (window->slab.has_value()) {
        return build_slab(store, handle, std::move(window));
    }
    return build_box(store, handle, std::move(window), std::move(params));
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
