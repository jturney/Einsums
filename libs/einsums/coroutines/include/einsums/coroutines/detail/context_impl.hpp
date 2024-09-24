//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) && !defined(__bgq__) && !defined(__powerpc__) &&    \
    !defined(__s390x__) && !defined(__arm__) && !defined(arm64) && !defined(__arm64) && !defined(__arm64__) && !defined(__aarch64__)

#    include <einsums/coroutines/detail/context_linux_x86.hpp>
namespace einsums::threads::coroutines::detail {
template <typename CoroutineImpl>
using default_context_impl = lx::x86_linux_context_impl<CoroutineImpl>;
} // namespace einsums::threads::coroutines::detail

#elif defined(_POSIX_VERSION) || defined(__bgq__) || defined(__powerpc__) || defined(__s390x__) || defined(__arm__) || defined(arm64) ||   \
    defined(__arm64) || defined(__arm64__) || defined(__aarch64__)

#    include <einsums/coroutines/detail/context_posix.hpp>
namespace einsums {
namespace threads {
namespace coroutines {
namespace detail {
template <typename CoroutineImpl>
using default_context_impl = posix::ucontext_context_impl<CoroutineImpl>;
}
} // namespace coroutines
} // namespace threads
} // namespace einsums

#elif defined(PIKA_HAVE_FIBER_BASED_COROUTINES)

#    include <einsums/coroutines/detail/context_windows_fibers.hpp>
namespace einsums {
namespace threads {
namespace coroutines {
namespace detail {
template <typename CoroutineImpl>
using default_context_impl = windows::fibers_context_impl<CoroutineImpl>;
}
} // namespace coroutines
} // namespace threads
} // namespace einsums

#else

#    error No default context switching implementation available for this system.

#endif // PIKA_HAVE_BOOST_CONTEXT
