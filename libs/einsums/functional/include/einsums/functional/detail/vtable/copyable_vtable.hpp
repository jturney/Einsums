//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/functional/detail/vtable/vtable.hpp>

#include <cstddef>
#include <new>

namespace einsums::util::detail {

struct copyable_vtable {
    template <typename T>
    static void *_copy(void *storage, std::size_t storage_size, void const *src, bool destroy) {
        if (destroy)
            vtable::get<T>(storage).~T();

        void *buffer = vtable::allocate<T>(storage, storage_size);
        return ::new (buffer) T(vtable::get<T>(src));
    }
    void *(*copy)(void *, std::size_t, void const *, bool);

    constexpr copyable_vtable(std::nullptr_t) noexcept : copy(nullptr) {}

    template <typename T>
    constexpr copyable_vtable(construct_vtable<T>) noexcept : copy(&copyable_vtable::template _copy<T>) {}
};

} // namespace einsums::util::detail