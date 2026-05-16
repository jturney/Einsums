//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/GPU/Runtime.hpp>

#include <cstddef>
#include <unordered_map>

namespace einsums::compute_graph {

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
    DeviceShadowMap() = default;
    ~DeviceShadowMap() { free_all(); }

    DeviceShadowMap(DeviceShadowMap const &)            = delete;
    DeviceShadowMap &operator=(DeviceShadowMap const &) = delete;
    DeviceShadowMap(DeviceShadowMap &&)                 = default;
    DeviceShadowMap &operator=(DeviceShadowMap &&)      = default;

    /// Allocate a device shadow for a tensor if not already allocated.
    /// Returns the device pointer.
    void *ensure(TensorId tid, size_t bytes) {
        auto it = _shadows.find(tid);
        if (it != _shadows.end())
            return it->second.ptr;

        auto  result  = gpu::device_malloc(bytes);
        void *ptr     = result ? result.value() : nullptr;
        _shadows[tid] = {.ptr = ptr, .bytes = bytes};
        return ptr;
    }

    /// Get the device pointer for a tensor, or nullptr if not allocated.
    [[nodiscard]] void *get(TensorId tid) const {
        auto it = _shadows.find(tid);
        return it != _shadows.end() ? it->second.ptr : nullptr;
    }

    /// Check if a shadow exists for the given tensor.
    [[nodiscard]] bool has(TensorId tid) const { return _shadows.find(tid) != _shadows.end(); }

    /// Free all device shadows.
    void free_all() {
        for (auto &[tid, shadow] : _shadows) {
            if (shadow.ptr) {
                gpu::device_free(shadow.ptr);
                shadow.ptr = nullptr;
            }
        }
        _shadows.clear();
    }

    /// Number of allocated shadows.
    [[nodiscard]] size_t size() const { return _shadows.size(); }

  private:
    struct Shadow {
        void  *ptr{nullptr};
        size_t bytes{0};
    };
    std::unordered_map<TensorId, Shadow> _shadows;
};

} // namespace einsums::compute_graph
