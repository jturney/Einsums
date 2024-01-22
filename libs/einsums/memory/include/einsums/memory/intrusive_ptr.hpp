//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/memory/config/defines.hpp>
#include <einsums/memory/detail/sp_convertible.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <type_traits>

namespace einsums::memory {

template <typename T>
struct intrusive_ptr {
  private:
    using this_type = intrusive_ptr;

  public:
    using element_type = T;

    constexpr intrusive_ptr() noexcept = default;

    intrusive_ptr(T *p, bool add_ref = true) : _px(p) {
        if (_px != nullptr && add_ref)
            intrusive_ptr_add_ref(_px);
    }

    template <typename U, typename Enable = std::enable_if_t<memory::detail::sp_convertible_v<U, T>>>
    intrusive_ptr(intrusive_ptr<U> const &rhs) : _px(rhs.get()) {
        if (_px != nullptr)
            intrusive_ptr_add_ref(_px);
    }

#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 120000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuse-after-free"
#endif
    intrusive_ptr(intrusive_ptr const &rhs) : _px(rhs._px) {
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 120000
#    pragma GCC diagnostic pop
#endif
        if (_px != nullptr)
            intrusive_ptr_add_ref(_px);
    }

    ~intrusive_ptr() {
        if (_px != nullptr)
            intrusive_ptr_release(_px);
    }

    template <typename U>
    auto operator=(intrusive_ptr<U> const &rhs) -> intrusive_ptr & {
        this_type(rhs).swap(*this);
        return *this;
    }

    // Move support
    constexpr intrusive_ptr(intrusive_ptr &&rhs) noexcept : _px(rhs._px) { rhs._px = nullptr; }

    auto operator=(intrusive_ptr &&rhs) noexcept -> intrusive_ptr & {
        this_type(static_cast<intrusive_ptr &&>(rhs)).swap(*this);
        return *this;
    }

    template <typename U>
    friend class intrusive_ptr;

    template <typename U, typename Enable = std::enable_if_t<memory::detail::sp_convertible_v<U, T>>>
    constexpr intrusive_ptr(intrusive_ptr<U> &&rhs) noexcept : _px(rhs.px) {
        rhs.px = nullptr;
    }

    template <typename U>
    auto operator=(intrusive_ptr<U> &&rhs) noexcept -> intrusive_ptr & {
        this_type(static_cast<intrusive_ptr<U> &&>(rhs)).swap(*this);
        return *this;
    }

    // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
    auto operator=(intrusive_ptr const &rhs) noexcept -> intrusive_ptr & {
        this_type(rhs).swap(*this);
        return *this;
    }

    auto operator=(T *rhs) noexcept -> intrusive_ptr & {
        this_type(rhs).swap(*this);
        return *this;
    }

    void reset() noexcept { this_type().swap(*this); }

    void reset(T *rhs) noexcept { this_type(rhs).swap(*this); }

    void reset(T *rhs, bool add_ref) noexcept { this_type(rhs, add_ref).swap(*this); }

    constexpr auto get() const noexcept -> T * { return _px; }

    constexpr auto detach() noexcept -> T * {
        T *ret = _px;
        _px    = nullptr;
        return ret;
    }

    auto operator*() const noexcept -> T & {
        PIKA_ASSERT(_px != nullptr);
        return *_px;
    }

    auto operator->() const noexcept -> T * {
        PIKA_ASSERT(_px != nullptr);
        return _px;
    }

    explicit constexpr operator bool() const noexcept { return _px != nullptr; }

    constexpr void swap(intrusive_ptr &rhs) noexcept {
        T *tmp  = _px;
        _px     = rhs._px;
        rhs._px = tmp;
    }

  private:
    T *_px = nullptr;
};

template <typename T, typename U>
inline constexpr auto operator==(intrusive_ptr<T> const &a, intrusive_ptr<U> const &b) noexcept -> bool {
    return a.get() == b.get();
}

template <typename T, typename U>
inline constexpr auto operator!=(intrusive_ptr<T> const &a, intrusive_ptr<U> const &b) noexcept -> bool {
    return a.get() != b.get();
}

template <typename T, typename U>
inline constexpr auto operator==(intrusive_ptr<T> const &a, U *b) noexcept -> bool {
    return a.get() == b;
}

template <typename T, typename U>
inline constexpr auto operator!=(intrusive_ptr<T> const &a, U *b) noexcept -> bool {
    return a.get() != b;
}

template <typename T, typename U>
inline constexpr auto operator==(T *a, intrusive_ptr<U> const &b) noexcept -> bool {
    return a == b.get();
}

template <typename T, typename U>
inline constexpr auto operator!=(T *a, intrusive_ptr<U> const &b) noexcept -> bool {
    return a != b.get();
}

template <typename T>
inline constexpr auto operator==(intrusive_ptr<T> const &p, std::nullptr_t) noexcept -> bool {
    return p.get() == nullptr;
}

template <typename T>
inline constexpr auto operator==(std::nullptr_t, intrusive_ptr<T> const &p) noexcept -> bool {
    return p.get() == nullptr;
}

template <typename T>
inline constexpr auto operator!=(intrusive_ptr<T> const &p, std::nullptr_t) noexcept -> bool {
    return p.get() != nullptr;
}

template <typename T>
inline constexpr auto operator!=(std::nullptr_t, intrusive_ptr<T> const &p) noexcept -> bool {
    return p.get() != nullptr;
}

template <typename T>
inline constexpr auto operator<(intrusive_ptr<T> const &a, intrusive_ptr<T> const &b) noexcept -> bool {
    return std::less<T *>{}(a.get(), b.get());
}

template <typename T>
void swap(intrusive_ptr<T> &lhs, intrusive_ptr<T> &rhs) noexcept {
    lhs.swap(rhs);
}

// mem_fn support
template <typename T>
constexpr auto get_pointer(intrusive_ptr<T> const &p) noexcept -> T * {
    return p.get();
}

// pointer casts
template <typename T, typename U>
constexpr auto static_pointer_cast(intrusive_ptr<U> const &p) -> intrusive_ptr<T> {
    return static_cast<T *>(p.get());
}

template <typename T, typename U>
constexpr auto const_pointer_cast(intrusive_ptr<U> const &p) -> intrusive_ptr<T> {
    return const_cast<T *>(p.get());
}

template <typename T, typename U>
auto dynamic_pointer_cast(intrusive_ptr<U> const &p) -> intrusive_ptr<T> {
    return dynamic_cast<T *>(p.get());
}

template <typename T, typename U>
constexpr auto static_pointer_cast(intrusive_ptr<U> &&p) noexcept -> intrusive_ptr<T> {
    return intrusive_ptr<T>(static_cast<T *>(p.detach()), false);
}

template <typename T, typename U>
constexpr auto const_pointer_cast(intrusive_ptr<U> &&p) noexcept -> intrusive_ptr<T> {
    return intrusive_ptr<T>(const_cast<T *>(p.detach()), false);
}

template <typename T, typename U>
auto dynamic_pointer_cast(intrusive_ptr<U> &&p) noexcept -> intrusive_ptr<T> {
    T *p2 = dynamic_cast<T *>(p.get());

    intrusive_ptr<T> r(p2, false);

    if (p2)
        p.detach();

    return r;
}

// operator<<
template <typename Y>
auto operator<<(std::ostream &os, intrusive_ptr<Y> const &p) -> std::ostream & {
    os << p.get();
    return os;
}

} // namespace einsums::memory

namespace einsums {

using einsums::memory::intrusive_ptr;

using einsums::memory::get_pointer;

using einsums::memory::const_pointer_cast;
using einsums::memory::dynamic_pointer_cast;
using einsums::memory::static_pointer_cast;

} // namespace einsums

namespace std {

// support hashing
template <typename T>
struct hash<einsums::memory::intrusive_ptr<T>> {
    using result_type = std::size_t;

    auto operator()(einsums::memory::intrusive_ptr<T> const &p) const noexcept -> result_type {
        return hash<T *>{}(p.get());
    }
};

} // namespace std