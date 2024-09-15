//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/assert.hpp>
#include <einsums/concepts/has_member_xxx.hpp>

#include <type_traits>

namespace einsums::util::detail {

EINSUMS_HAS_MEMBER_XXX_TRAIT_DEF(owns_lock)

template <typename Lock>
constexpr void assert_owns_lock(Lock const &, int) noexcept {
}

template <typename Lock>
constexpr void assert_doesnt_own_lock(Lock const &, int) noexcept {
}

template <typename Lock>
    requires(has_owns_lock_v<Lock>)
void assert_owns_lock([[maybe_unused]] Lock const &l, long) noexcept {
    EINSUMS_ASSERT(l.owns_lock());
}

template <typename Lock>
    requires(has_owns_lock_v<Lock>)
void assert_doesnt_own_lock([[maybe_unused]] Lock const &l, long) noexcept {
    EINSUMS_ASSERT(!l.owns_lock());
}

} // namespace einsums::util::detail

#define EINSUMS_ASSERT_OWNS_LOCK(l) ::einsums::util::detail::assert_owns_lock(l, 0L)

#define EINSUMS_ASSERT_DOESNT_OWN_LOCK(l) ::einsums::util::detail::assert_doesnt_own_lock(l, 0L)
