//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file BackingFill.hpp
 * @brief Fill a tensor's storage from a backing, whatever the tensor's layout.
 *
 * Private to the pass sources. A backing writes a slab densely in column-major order, the order
 * its storage keeps. A tensor the library declares follows @c --einsums:row-major, which Python
 * turns on, so the tensor a slab lands in is not always laid out that way. A dense column-major
 * tensor takes the read directly; any other is read into scratch and scattered by its strides, so
 * every element lands where the tensor indexes it.
 */

#include <Einsums/ComputeGraph/Detail/RunContext.hpp>
#include <Einsums/ComputeGraph/Observer.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::detail)

/// Count a read of @p bytes from @p backing, for Graph::last_run, and tell an attached observer.
inline void count_read(TensorBacking const &backing, std::size_t bytes) noexcept {
    if (backing.counters) {
        backing.counters->reads.fetch_add(1, std::memory_order_relaxed);
        backing.counters->bytes_read.fetch_add(bytes, std::memory_order_relaxed);
        if (backing.counters->context) {
            compute_graph::detail::notify_io(*backing.counters->context, EventKind::Read, backing.counters->tensor, backing, bytes);
        }
    }
}

/// Count a write of @p bytes to @p backing, for Graph::last_run, and tell an attached observer.
inline void count_write(TensorBacking const &backing, std::size_t bytes) noexcept {
    if (backing.counters) {
        backing.counters->writes.fetch_add(1, std::memory_order_relaxed);
        backing.counters->bytes_written.fetch_add(bytes, std::memory_order_relaxed);
        if (backing.counters->context) {
            compute_graph::detail::notify_io(*backing.counters->context, EventKind::Write, backing.counters->tensor, backing, bytes);
        }
    }
}

/**
 * @brief Read the slab @p ranges of @p backing into a tensor's storage.
 * @param[in] backing The storage to read.
 * @param[out] data The tensor's first element, writable.
 * @param[in] layout The tensor's extents and strides, which must be the slab's extents.
 * @param[in] ranges Per axis, the range of the stored tensor to read.
 */
template <typename T>
void fill_from_backing(TensorBacking const &backing, T *data, einsums::detail::TensorImpl<T> const &layout,
                       std::vector<std::pair<std::size_t, std::size_t>> const &ranges) {
    std::size_t const count = layout.size();
    if (count == 0) {
        return;
    }
    if (layout.is_contiguous() && layout.is_column_major()) {
        backing.read_slab(data, ranges);
        count_read(backing, count * sizeof(T));
        return;
    }
    std::vector<T> scratch(count);
    backing.read_slab(scratch.data(), ranges);
    count_read(backing, count * sizeof(T));
    std::size_t const        rank = layout.rank();
    std::vector<std::size_t> index(rank, 0);
    for (std::size_t n = 0; n < count; ++n) {
        std::size_t offset = 0;
        for (std::size_t d = 0; d < rank; ++d) {
            offset += index[d] * layout.stride(d);
        }
        data[offset] = scratch[n];
        // Column-major: the first index runs fastest, which is the order the scratch holds.
        for (std::size_t d = 0; d < rank; ++d) {
            if (++index[d] < layout.dim(d)) {
                break;
            }
            index[d] = 0;
        }
    }
}

/**
 * @brief Write a tensor's storage to the slab @p ranges of @p backing.
 *
 * The mirror of @ref fill_from_backing: a dense column-major tensor is written directly, any
 * other is gathered by its strides into scratch in the storage's column-major order first.
 *
 * @param[in] backing The storage to write.
 * @param[in] data The tensor's first element.
 * @param[in] layout The tensor's extents and strides, which must be the slab's extents.
 * @param[in] ranges Per axis, the range of the stored tensor to write.
 */
template <typename T>
void store_to_backing(TensorBacking const &backing, T const *data, einsums::detail::TensorImpl<T> const &layout,
                      std::vector<std::pair<std::size_t, std::size_t>> const &ranges) {
    std::size_t const count = layout.size();
    if (count == 0) {
        return;
    }
    if (layout.is_contiguous() && layout.is_column_major()) {
        backing.write_slab(data, ranges);
        count_write(backing, count * sizeof(T));
        return;
    }
    std::vector<T>           scratch(count);
    std::size_t const        rank = layout.rank();
    std::vector<std::size_t> index(rank, 0);
    for (std::size_t n = 0; n < count; ++n) {
        std::size_t offset = 0;
        for (std::size_t d = 0; d < rank; ++d) {
            offset += index[d] * layout.stride(d);
        }
        scratch[n] = data[offset];
        for (std::size_t d = 0; d < rank; ++d) {
            if (++index[d] < layout.dim(d)) {
                break;
            }
            index[d] = 0;
        }
    }
    backing.write_slab(scratch.data(), ranges);
    count_write(backing, count * sizeof(T));
}

EINSUMS_NAMESPACE_END(compute_graph::passes::detail)
