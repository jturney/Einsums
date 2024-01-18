//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <cstddef>
#include <memory>

namespace einsums {

namespace detail {
auto EINSUMS_EXPORT allocate_aligned_memory(size_t align, size_t size) -> void *;
void EINSUMS_EXPORT deallocate_aligned_memory(void *ptr) noexcept;
} // namespace detail

template <typename T, size_t Align = 32>
class AlignedAllocator;

template <size_t Align>
class AlignedAllocator<void, Align> {
  public:
    using pointer       = void *;
    using const_pointer = void const *;
    using value_type    = void;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template <typename T, size_t Align>
class AlignedAllocator {
  public:
    using value_type      = T;
    using pointer         = T *;
    using const_pointer   = T const *;
    using reference       = T &;
    using const_reference = T const &;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

  public:
    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(AlignedAllocator<U, Align> const &) noexcept {}

    [[nodiscard]] auto max_size() const noexcept -> size_type { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    auto address(reference x) const noexcept -> pointer { return std::addressof(x); }

    auto address(const_reference x) const noexcept -> const_pointer { return std::addressof(x); }

    auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) -> pointer {
        auto const alignment = static_cast<size_type>(Align);
        void      *ptr       = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { return detail::deallocate_aligned_memory(p); }

    template <class U, class... Args>
    void construct(U *p, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, size_t Align>
class AlignedAllocator<T const, Align> {
  public:
    using value_type      = T;
    using pointer         = T const *;
    using const_pointer   = T const *;
    using reference       = T const &;
    using const_reference = T const &;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

  public:
    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(AlignedAllocator<U, Align> const &) noexcept {}

    [[nodiscard]] auto max_size() const noexcept -> size_type { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    auto address(const_reference x) const noexcept -> const_pointer { return std::addressof(x); }

    auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) -> pointer {
        auto const alignment = static_cast<size_type>(Align);
        void      *ptr       = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { return detail::deallocate_aligned_memory(p); }

    template <class U, class... Args>
    void construct(U *p, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline auto operator==(AlignedAllocator<T, TAlign> const &, AlignedAllocator<U, UAlign> const &) noexcept -> bool {
    return TAlign == UAlign;
}

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline auto operator!=(AlignedAllocator<T, TAlign> const &, AlignedAllocator<U, UAlign> const &) noexcept -> bool {
    return TAlign != UAlign;
}

} // namespace einsums