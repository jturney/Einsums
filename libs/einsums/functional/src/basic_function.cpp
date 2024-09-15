//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/functional/detail/basic_function.hpp>
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
///////////////////////////////////////////////////////////////////////////
function_base::function_base(function_base const &other, vtable const * /* empty_vtable */) : vptr(other.vptr), object(other.object) {
    if (other.object != nullptr) {
        object = vptr->copy(storage, detail::function_storage_size, other.object, /*destroy*/ false);
    }
}

function_base::function_base(function_base &&other, vtable const *empty_vptr) noexcept : vptr(other.vptr), object(other.object) {
    if (object == &other.storage) {
        std::memcpy(storage, other.storage, function_storage_size);
        object = &storage;
    }
    other.vptr   = empty_vptr;
    other.object = nullptr;
}

function_base::~function_base() {
    destroy();
}

void function_base::op_assign(function_base const &other, vtable const * /* empty_vtable */) {
    if (vptr == other.vptr) {
        if (this != &other && object) {
            EINSUMS_ASSERT(other.object != nullptr);
            // reuse object storage
            object = vptr->copy(object, std::size_t(-1), other.object, /*destroy*/ true);
        }
    } else {
        destroy();
        vptr = other.vptr;
        if (other.object != nullptr) {
            object = vptr->copy(storage, detail::function_storage_size, other.object, /*destroy*/ false);
        } else {
            object = nullptr;
        }
    }
}

void function_base::op_assign(function_base &&other, vtable const *empty_vtable) noexcept {
    if (this != &other) {
        swap(other);
        other.reset(empty_vtable);
    }
}

void function_base::destroy() noexcept {
    if (object != nullptr) {
        vptr->deallocate(object, function_storage_size,
                         /*destroy*/ true);
    }
}

void function_base::reset(vtable const *empty_vptr) noexcept {
    destroy();
    vptr   = empty_vptr;
    object = nullptr;
}

void function_base::swap(function_base &f) noexcept {
    std::swap(vptr, f.vptr);
    std::swap(object, f.object);
    std::swap(storage, f.storage);
    if (object == &f.storage)
        object = &storage;
    if (f.object == &storage)
        f.object = &f.storage;
}

std::size_t function_base::get_function_address() const {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    return vptr->get_function_address(object);
#else
    return 0;
#endif
}

char const *function_base::get_function_annotation() const {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    return vptr->get_function_annotation(object);
#else
    return nullptr;
#endif
}

} // namespace einsums::util::detail
