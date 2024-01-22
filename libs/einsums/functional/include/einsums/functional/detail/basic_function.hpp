//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/functional/detail/empty_function.hpp>
#include <einsums/functional/detail/vtable/function_vtable.hpp>
#include <einsums/functional/detail/vtable/vtable.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace einsums::util::detail {

static std::size_t const function_storage_size = 3 * sizeof(void *);

struct EINSUMS_EXPORT function_base {
  private:
    using vtable = function_base_vtable;

  protected:
    vtable const *vptr;
    void         *object;
    union {
        char                  storage_init;
        mutable unsigned char storage[function_storage_size];
    };

  public:
    constexpr explicit function_base(function_base_vtable const *empty_vptr) noexcept
        : vptr(empty_vptr), object(nullptr), storage_init() {}

    function_base(function_base const &other, vtable const *empty_vptr);
    function_base(function_base &&other, vtable const *empty_vptr) noexcept;
    ~function_base();

    void op_assign(function_base const &other, vtable const *empty_vtable);
    void op_assign(function_base &&other, vtable const *empty_vtable) noexcept;

    void destroy() noexcept;
    void reset(vtable const *empty_vptr) noexcept;
    void swap(function_base &f) noexcept;

    bool empty() const noexcept { return object == nullptr; }

    explicit operator bool() const noexcept { return !empty(); }

    auto get_function_address() const -> std::size_t;
    auto get_function_annotation() const -> char const *;
    auto get_function_annotation_itt() const -> util::itt::string_handle;
};

template <typename F>
constexpr auto is_empty_function(F *fp) noexcept -> bool {
    return fp == nullptr;
}

template <typename T, typename C>
constexpr auto is_empty_function(T C::*mp) noexcept -> bool {
    return mp == nullptr;
}

inline auto is_empty_function_impl(function_base const *f) noexcept -> bool {
    return f->empty();
}

inline constexpr auto is_empty_function_impl(...) noexcept -> bool {
    return false;
}

////////////////////////////////////////////////////////////////////////////////

template <typename Sig, bool Copyable>
struct basic_function;

template <bool Copyable, typename R, typename... Ts>
struct basic_function<R(Ts...), Copyable> : public function_base {
  private:
    using base_type = function_base;
    using vtable    = function_vtable<R(Ts...), Copyable>;

  public:
    constexpr basic_function() noexcept : base_type(get_empty_vtable()) {}

    basic_function(basic_function const &other) : base_type(other, get_empty_vtable()) {}

    basic_function(basic_function &&other) noexcept : base_type(EINSUMS_MOVE(other), get_empty_vtable()) {}

    basic_function &operator=(basic_function const &other) {
        base_type::op_assign(other, get_empty_vtable());
        return *this;
    }

    basic_function &operator=(basic_function &&other) noexcept {
        base_type::op_assign(EINSUMS_MOVE(other), get_empty_vtable());
        return *this;
    }

    void assign(std::nullptr_t) noexcept { base_type::reset(get_empty_vtable()); }

    template <typename F>
    void assign(F &&f) {
        using T = std::decay_t<F>;
        static_assert(!Copyable || std::is_constructible_v<T, T const &>, "F shall be CopyConstructible");

        if (!detail::is_empty_function(f)) {
            vtable const *f_vptr = get_vtable<T>();
            void         *buffer = nullptr;
            if (vptr == f_vptr) {
                EINSUMS_ASSERT(object != nullptr);
                // reuse object storage
                buffer = object;
                vtable::template get<T>(object).~T();
            } else {
                destroy();
                vptr   = f_vptr;
                buffer = vtable::template allocate<T>(storage, function_storage_size);
            }
            object = ::new (buffer) T(EINSUMS_FORWARD(F, f));
        } else {
            base_type::reset(get_empty_vtable());
        }
    }

    void reset() noexcept { base_type::reset(get_empty_vtable()); }

    using base_type::empty;
    using base_type::swap;
    using base_type::operator bool;

    template <typename T>
    auto target() noexcept -> T * {
        using TD = std::remove_cv_t<T>;
        static_assert(std::is_invocable_r_v<R, TD &, Ts...>, "T shall be Callable with the function signature");

        vtable const *f_vptr = get_vtable<TD>();
        if (vptr != f_vptr || empty())
            return nullptr;

        return &vtable::template get<TD>(object);
    }

    template <typename T>
    auto target() const noexcept -> T const * {
        using TD = std::remove_cv_t<T>;
        static_assert(std::is_invocable_r_v<R, TD &, Ts...>, "T shall be Callable with the function signature");

        vtable const *f_vptr = get_vtable<TD>();
        if (vptr != f_vptr || empty())
            return nullptr;

        return &vtable::template get<TD>(object);
    }

    EINSUMS_FORCEINLINE auto operator()(Ts... vs) const -> R {
        auto const *vptr = static_cast<vtable const *>(base_type::vptr);
        return vptr->invoke(object, EINSUMS_FORWARD(Ts, vs)...);
    }

    using base_type::get_function_address;
    using base_type::get_function_annotation;
    using base_type::get_function_annotation_itt;

  private:
    static constexpr auto get_empty_vtable() noexcept -> vtable const * {
        return detail::get_empty_function_vtable<R(Ts...)>();
    }

    template <typename T>
    static auto get_vtable() noexcept -> vtable const * {
        return detail::get_vtable<vtable, T>();
    }

  protected:
    using base_type::object;
    using base_type::storage;
    using base_type::vptr;
};

} // namespace einsums::util::detail