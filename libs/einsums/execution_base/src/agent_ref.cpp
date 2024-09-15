
//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/execution_base/agent_ref.hpp>
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
#    include <einsums/lock_registration/detail/register_locks.hpp>
#endif
#include <einsums/execution_base/this_thread.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <string>

namespace einsums::execution::detail {

void agent_ref::yield(const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    // verify that there are no more registered locks for this OS-thread
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    util::verify_no_locks();
#endif
    impl_->yield(desc);
}

void agent_ref::yield_k(std::size_t k, const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    // verify that there are no more registered locks for this OS-thread
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    util::verify_no_locks();
#endif
    impl_->yield_k(k, desc);
}

void agent_ref::spin_k(std::size_t k, const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    // verify that there are no more registered locks for this OS-thread
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    util::verify_no_locks();
#endif
    impl_->spin_k(k, desc);
}

void agent_ref::suspend(const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    // verify that there are no more registered locks for this OS-thread
#ifdef EINSUMS_HAVE_VERIFY_LOCKS
    util::verify_no_locks();
#endif
    impl_->suspend(desc);
}

void agent_ref::resume(const char *desc) {
    EINSUMS_ASSERT(*this != einsums::execution::this_thread::detail::agent());
    impl_->resume(desc);
}

void agent_ref::abort(const char *desc) {
    EINSUMS_ASSERT(*this != einsums::execution::this_thread::detail::agent());
    impl_->abort(desc);
}

void agent_ref::sleep_for(einsums::chrono::steady_duration const &sleep_duration, const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    impl_->sleep_for(sleep_duration, desc);
}

void agent_ref::sleep_until(einsums::chrono::steady_time_point const &sleep_time, const char *desc) {
    EINSUMS_ASSERT(*this == einsums::execution::this_thread::detail::agent());
    impl_->sleep_until(sleep_time, desc);
}

std::string format(einsums::execution::detail::agent_ref const &a) {
    return fmt::format("agent_ref{{{}}}", a.impl_->description());
}
} // namespace einsums::execution::detail
