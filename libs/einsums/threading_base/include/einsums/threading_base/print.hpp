//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/debugging/print.hpp>
#include <einsums/threading_base/thread_data.hpp>

#include <iosfwd>

// ------------------------------------------------------------
/// \cond NODETAIL
namespace einsums::debug::detail {
// ------------------------------------------------------------------
// safely dump thread pointer/description
// ------------------------------------------------------------------
template <typename T>
struct threadinfo;

// ------------------------------------------------------------------
// safely dump thread pointer/description
// ------------------------------------------------------------------
template <>
struct threadinfo<threads::detail::thread_data *> {
    constexpr explicit threadinfo(threads::detail::thread_data const *v) : data(v) {}

    threads::detail::thread_data const *data;

    EINSUMS_EXPORT friend std::ostream &operator<<(std::ostream &os, threadinfo const &d);
};

template <>
struct threadinfo<threads::detail::thread_id_type *> {
    constexpr explicit threadinfo(threads::detail::thread_id_type const *v) : data(v) {}

    threads::detail::thread_id_type const *data;

    EINSUMS_EXPORT friend std::ostream &operator<<(std::ostream &os, threadinfo const &d);
};

template <>
struct threadinfo<threads::detail::thread_id_ref_type *> {
    constexpr explicit threadinfo(threads::detail::thread_id_ref_type const *v) : data(v) {}

    threads::detail::thread_id_ref_type const *data;

    EINSUMS_EXPORT friend std::ostream &operator<<(std::ostream &os, threadinfo const &d);
};

template <>
struct threadinfo<einsums::threads::detail::thread_init_data> {
    constexpr explicit threadinfo(einsums::threads::detail::thread_init_data const &v) : data(v) {}

    einsums::threads::detail::thread_init_data const &data;

    EINSUMS_EXPORT friend std::ostream &operator<<(std::ostream &os, threadinfo const &d);
};
} // namespace einsums::debug::detail
/// \endcond
