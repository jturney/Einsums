//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/detail/empty_function.hpp>
#include <einsums/functional/detail/vtable/vtable.hpp>
#include <einsums/functional/invoke.hpp>
#include <einsums/functional/traits/get_function_address.hpp>
#include <einsums/functional/traits/get_function_annotation.hpp>
#include <einsums/type_support/void_guard.hpp>

#include <cstddef>

namespace einsums::util::detail {

struct trivial_empty_function;

///////////////////////////////////////////////////////////////////////////
struct callable_info_vtable {
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    template <typename T>
    EINSUMS_FORCEINLINE static std::size_t _get_function_address(void *f) {
        return einsums::detail::get_function_address<T>::call(vtable::get<T>(f));
    }
    std::size_t (*get_function_address)(void *);

    template <typename T>
    EINSUMS_FORCEINLINE static char const *_get_function_annotation(void *f) {
        return einsums::detail::get_function_annotation<T>::call(vtable::get<T>(f));
    }
    char const *(*get_function_annotation)(void *);

#    if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
    template <typename T>
    EINSUMS_FORCEINLINE static util::itt::string_handle _get_function_annotation_itt(void *f) {
        return einsums::detail::get_function_annotation_itt<T>::call(vtable::get<T>(f));
    }
    util::itt::string_handle (*get_function_annotation_itt)(void *);
#    endif
#endif

    template <typename T>
    constexpr callable_info_vtable(construct_vtable<T>) noexcept
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        : get_function_address(&callable_info_vtable::template _get_function_address<T>),
          get_function_annotation(&callable_info_vtable::template _get_function_annotation<T>)
#    if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
          ,
          get_function_annotation_itt(&callable_info_vtable::template _get_function_annotation_itt<T>)
#    endif
#endif
    {
    }

    constexpr callable_info_vtable(construct_vtable<trivial_empty_function>) noexcept
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        : get_function_address(nullptr), get_function_annotation(nullptr)
#    if EINSUMS_HAVE_ITTNOTIFY != 0 && !defined(EINSUMS_HAVE_APEX)
          ,
          get_function_annotation_itt(nullptr)
#    endif
#endif
    {
    }
};

///////////////////////////////////////////////////////////////////////////
template <typename Sig>
struct callable_vtable;

template <typename R, typename... Ts>
struct callable_vtable<R(Ts...)> {
    template <typename T>
    EINSUMS_FORCEINLINE static R _invoke(void *f, Ts &&...vs) {
        return EINSUMS_INVOKE_R(R, vtable::get<T>(f), EINSUMS_FORWARD(Ts, vs)...);
    }
    R (*invoke)(void *, Ts &&...);

    template <typename T>
    constexpr callable_vtable(construct_vtable<T>) noexcept : invoke(&callable_vtable::template _invoke<T>) {}

    static R _empty_invoke(void *, Ts &&...) { return throw_bad_function_call<R>(); }

    constexpr callable_vtable(construct_vtable<trivial_empty_function>) noexcept
        : invoke(&callable_vtable::_empty_invoke) {}
};

} // namespace einsums::util::detail