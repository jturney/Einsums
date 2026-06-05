//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums::gpu {

// ===========================================================================
// DeviceAllocator tag — used by GeneralTensor to select DeviceVector storage.
// ===========================================================================

/// Tag type indicating device memory allocation.
/// When used as the Alloc parameter of GeneralTensor, the storage switches
/// from std::vector<T, Alloc> to DeviceVector<T>.
template <typename T>
struct DeviceAllocator {
    using value_type = T; // NOLINT(readability-identifier-naming)
};

/// Trait to detect device allocators.
template <typename Alloc>
struct IsDeviceAllocator : std::false_type {};

template <typename T>
struct IsDeviceAllocator<DeviceAllocator<T>> : std::true_type {};

template <typename Alloc>
inline constexpr bool IsDeviceAllocatorV = IsDeviceAllocator<Alloc>::value;

// ===========================================================================
// DeviceVector<T> — minimal vector-like container backed by GPU memory.
//
// Supports only bulk operations: data(), size(), resize(), copy, move.
// No element access (operator[], begin/end, at) — device memory is not
// host-accessible. Use gpu::memcpy_device_to_host to read elements.
//
// On the mock backend, uses std::malloc (same address space) so tests work.
// ===========================================================================

template <typename T>
class DeviceVector {
  public:
    using value_type = T; // NOLINT(readability-identifier-naming)

    /// Default constructor — empty vector.
    DeviceVector() noexcept = default;

    /// Construct with a given size. Memory is allocated but NOT initialized.
    explicit DeviceVector(size_t count) : _size(count) {
        if (count > 0) {
            _ptr = static_cast<T *>(device_malloc(count * sizeof(T)));
        }
    }

    /// Copy constructor — allocates new device memory and copies data.
    DeviceVector(DeviceVector const &other) : _size(other._size) {
        if (_size > 0) {
            _ptr = static_cast<T *>(device_malloc(_size * sizeof(T)));
            memcpy_device_to_device(_ptr, other._ptr, _size * sizeof(T));
        }
    }

    /// Move constructor — takes ownership, leaves other empty.
    DeviceVector(DeviceVector &&other) noexcept : _ptr(other._ptr), _size(other._size) {
        other._ptr  = nullptr;
        other._size = 0;
    }

    /// Destructor — frees device memory.
    ~DeviceVector() {
        if (_ptr != nullptr) {
            device_free(_ptr);
        }
    }

    /// Copy assignment — deep copy of device data.
    DeviceVector &operator=(DeviceVector const &other) {
        if (this != &other) {
            if (_size != other._size) {
                if (_ptr != nullptr) {
                    device_free(_ptr);
                    _ptr = nullptr;
                }
                _size = other._size;
                if (_size > 0) {
                    _ptr = static_cast<T *>(device_malloc(_size * sizeof(T)));
                }
            }
            if (_size > 0) {
                memcpy_device_to_device(_ptr, other._ptr, _size * sizeof(T));
            }
        }
        return *this;
    }

    /// Move assignment.
    DeviceVector &operator=(DeviceVector &&other) noexcept {
        if (this != &other) {
            if (_ptr != nullptr) {
                device_free(_ptr);
            }
            _ptr        = other._ptr;
            _size       = other._size;
            other._ptr  = nullptr;
            other._size = 0;
        }
        return *this;
    }

    /// Get raw device pointer to data.
    [[nodiscard]] T *data() noexcept { return _ptr; }

    /// Get raw device pointer to data (const).
    [[nodiscard]] T const *data() const noexcept { return _ptr; }

    /// Get number of elements.
    [[nodiscard]] size_t size() const noexcept { return _size; }

    /// Check if empty.
    [[nodiscard]] bool empty() const noexcept { return _size == 0; }

    /// Resize the vector. If size changes, old data is lost (not preserved).
    /// This matches the behavior needed by GeneralTensor which always
    /// reinitializes after resize.
    void resize(size_t new_size) {
        if (new_size == _size) {
            return;
        }
        if (_ptr != nullptr) {
            device_free(_ptr);
            _ptr = nullptr;
        }
        _size = new_size;
        if (new_size > 0) {
            auto result = device_malloc(new_size * sizeof(T));
            _ptr        = result ? static_cast<T *>(result.value()) : nullptr;
        }
    }

  private:
    T     *_ptr{nullptr};
    size_t _size{0};
};

} // namespace einsums::gpu
