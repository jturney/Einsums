//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums::util::detail {

template <typename T>
struct construct_vtable {};

template <typename VTable, typename T>
struct vtables {
    static constexpr VTable instance = detail::construct_vtable<T>();
};

template <typename VTable, typename T>
constexpr VTable vtables<VTable, T>::instance;

template <typename VTable, typename T>
constexpr VTable const *get_vtable() noexcept {
    static_assert(!std::is_reference_v<T>, "T shall have no ref-qualifiers");

    return &vtables<VTable, T>::instance;
}

struct vtable {
    template <typename T>
    static T &get(void *obj) noexcept {
        return *reinterpret_cast<T *>(obj);
    }

    template <typename T>
    static T const &get(void const *obj) noexcept {
        return *reinterpret_cast<T const *>(obj);
    }

    template <typename T>
    static void *allocate(void *storage, std::size_t storage_size) {
        using storage_t = std::aligned_storage_t<sizeof(T), alignof(T)>;

        if (sizeof(T) > storage_size) {
            return new storage_t;
        }
        return storage;
    }

    template <typename T>
    static void _deallocate(void *obj, std::size_t storage_size, bool destroy) {
        using storage_t = std::aligned_storage_t<sizeof(T), alignof(T)>;

        if (destroy) {
            get<T>(obj).~T();
        }

        if (sizeof(T) > storage_size) {
            delete static_cast<storage_t *>(obj);
        }
    }
    void (*deallocate)(void *, std::size_t storage_size, bool);

    template <typename T>
    constexpr vtable(construct_vtable<T>) noexcept : deallocate(&vtable::template _deallocate<T>) {}
};
} // namespace einsums::util::detail