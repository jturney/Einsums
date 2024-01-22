//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/coroutines/detail/context_impl.hpp>

// The preprocessor conditions below are kept in sync with those used in
// context_impl.hpp

#if (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) && !defined(__bgq__) &&         \
    !defined(__powerpc__) && !defined(__s390x__) && !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)

// left empty on purpose

#elif defined(_POSIX_VERSION) || defined(__bgq__) || defined(__powerpc__) || defined(__s390x__) || defined(__arm__) || \
    defined(__arm64__) || defined(__aarch64__)

#    include <cstddef>

namespace einsums {
namespace threads {
namespace coroutines {
namespace detail {
namespace posix {

std::ptrdiff_t ucontext_context_impl_base::default_stack_size = SIGSTKSZ;
}
} // namespace detail
} // namespace coroutines
} // namespace threads
} // namespace einsums

#elif defined(EINSUMS_HAVE_FIBER_BASED_COROUTINES)

// left empty on purpose

#else

#    error No default_context_impl available for this system

#endif
