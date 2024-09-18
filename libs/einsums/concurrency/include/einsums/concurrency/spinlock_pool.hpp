//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/concurrency/cache_line_data.hpp>
#include <einsums/concurrency/detail/fibhash.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/thread_support/spinlock.hpp>

#include <cstddef>

namespace einsums::concurrency::detail {

template <typename Tag, std::size_t N = EINSUMS_HAVE_SPINLOCK_POOL_NUM>
struct spinlock_pool {
    static ::einsums::detail::spinlock &spinlock_for(void const *pv) {
        std::size_t i = einsums::detail::fibhash<N>(reinterpret_cast<std::size_t>(pv));
        return _pool[i].data_;
    }

  private:
    static einsums::concurrency::detail::cache_aligned_data<::einsums::detail::spinlock> _pool[N];
};

template <typename Tag, std::size_t N>
einsums::concurrency::detail::cache_aligned_data<::einsums::detail::spinlock> spinlock_pool<Tag, N>::_pool[N];

} // namespace einsums::concurrency::detail