//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/detail/vtable/vtable.hpp>

namespace einsums::util::detail {

struct trivial_empty_function {};

[[noreturn]] EINSUMS_EXPORT void throw_bad_function_call();

template <typename R>
[[noreturn]] inline auto throw_bad_function_call() -> R {
    throw_bad_function_call();
}

template <typename Sig, bool Copyable>
struct function_vtable;

// NOTE: nvcc (at least CUDA 9.2 and 10.1) fails with an internal compiler error so we use a fallback
#if !defined(EINSUMS_HAVE_CUDA)
template <typename Sig>
constexpr auto get_empty_function_vtable() noexcept -> function_vtable<Sig, true> const * {
    return &vtables<function_vtable<Sig, true>, trivial_empty_function>::instance;
}
#else
template <typename Sig>
function_vtable<Sig, true> const *get_empty_function_vtable() noexcept {
    static function_vtable<Sig, true> const empty_vtable = detail::construct_vtable<trivial_empty_function>();
    return &empty_vtable;
}
#endif

} // namespace einsums::util::detail