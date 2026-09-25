//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Runtime.hpp>

#include <cstddef>
#include <memory>
#include <unordered_map>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

struct DeviceFree {
    void operator()(void *ptr) const noexcept { gpu::device_free(ptr); }
};

// One owned device buffer, so replacing, erasing, destroying and move-assigning all free what they
// drop. DeviceShadowMap used to hold raw pointers with a defaulted move-assignment, which overwrote
// a non-empty map's buffers without freeing them.
struct DeviceShadow {
    std::unique_ptr<void, DeviceFree> ptr;
    size_t                            bytes{0}; // what ptr holds; zero when the allocation failed
};

} // namespace detail

/**
 * @brief Manages device (GPU) shadow allocations for tensors.
 *
 * When the graph executor needs to run GPU nodes, it creates device shadows
 * for each tensor used by GPU operations. This class tracks the mapping
 * from TensorId to device pointer and handles allocation/deallocation.
 *
 * On mock backend, device_malloc returns real heap memory, so shadows are
 * genuine separate buffers that exercise the full H2D → compute → D2H path.
 */
class DeviceShadowMap {
  public:
    /// Allocate a device shadow for a tensor if not already allocated.
    /// Returns the device pointer, or nullptr when the device is out of memory.
    void *ensure(TensorId tid, size_t bytes) {
        auto it = _shadows.find(tid);
        if (it != _shadows.end()) {
            // First writer wins on size, and the two call sites disagree about
            // what to pass: the HostToDevice path uses TransferDescriptor::
            // size_bytes while the GPU-node path uses TensorHandle::total_bytes().
            // If those ever diverge, whichever ran first silently sizes the
            // allocation and the other overruns it. Grow rather than truncate,
            // since a too-small device buffer is a heap corruption on the device.
            if (bytes > it->second.bytes) {
                it->second = allocate(bytes);
            }
            return it->second.ptr.get();
        }
        return _shadows.emplace(tid, allocate(bytes)).first->second.ptr.get();
    }

    /// Get the device pointer for a tensor, or nullptr if not allocated.
    [[nodiscard]] void *get(TensorId tid) const {
        auto it = _shadows.find(tid);
        return it != _shadows.end() ? it->second.ptr.get() : nullptr;
    }

    /// Check if a shadow exists for the given tensor.
    [[nodiscard]] bool has(TensorId tid) const { return _shadows.contains(tid); }

    /// Free all device shadows.
    void free_all() { _shadows.clear(); }

    /// Number of allocated shadows.
    [[nodiscard]] size_t size() const { return _shadows.size(); }

  private:
    static detail::DeviceShadow allocate(size_t bytes) {
        auto result = gpu::device_malloc(bytes);
        if (!result) {
            return {};
        }
        return {.ptr = std::unique_ptr<void, detail::DeviceFree>{result.value()}, .bytes = bytes};
    }

    std::unordered_map<TensorId, detail::DeviceShadow> _shadows;
};

EINSUMS_NAMESPACE_END(compute_graph)
