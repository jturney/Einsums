//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>

/*
   ContextImpl concept documentation.
   A ContextImpl holds a context plus its stack.
   - ContextImpl must have constructor with the following signature:

     template<typename Functor>
     ContextImpl(Functor f, std::ptrdiff_t stack_size);

     Preconditions:
     f is a generic function object (support for function pointers
     is not required.
     stack_size is the size of the stack allocated
     for the context. The stack_size is only an hint. If stack_size is -1
     it should default to a sensible value.
     Postconditions:
     f is bound to this context in its own stack.
     When the context is activated with swap_context for the first time
     f is entered.

   - ContextImpl destructor must properly dispose of the stack and perform
     any other clean up action required.

   - ContextImpl is not required to be DefaultConstructible nor Copyable.

   - ContextImpl must expose the following typedef:

     using context_impl_base = unspecified-type;

     ContextImpl must be convertible to context_impl_base.
     context_impl_base must have conform to the ContextImplBase concept:

     - DefaultConstructible. A default constructed ContextImplBase is in
       an initialized state.
       A ContextImpl is an initialized ContextImplBase.

     - ContextImplBase is Copyable. A copy of a ContextImplBase holds
       The same information as the original. Once a ContextImplBase
       is used as an argument to swap_context, all its copies become stale.
       (that is, only one copy of ContextImplBase can be used).
       A ContextImpl cannot be sliced by copying it to a ContextImplBase.

     - void swap_context(ContextImplBase& from,
                         const ContextImplBase& to)

       This free function is called unqualified (via ADL).
       Preconditions:
       Its 'to' argument must be an initialized ContextImplBase.
       Its 'from' argument may be an uninitialized ContextImplBase that
       will be initialized by a swap_context.
       Postconditions:
       The current context is saved in the 'from' context, and the
       'to' context is restored.
       It is undefined behavior if the 'to' argument is an invalid
       (uninitialized) swap context.

     A ContextImplBase is meant to be used when an empty temporary
     context is needed to store the current context before
     restoring a ContextImpl and no current context is
     available. It could be possible to simply have ContextImpl
     default constructible, but on destruction it would need to
     check if a stack has been allocated and would slow down the
     fast invocation path. Also a stack-full context could not be
     make copyable.
*/

#if (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) && !defined(__bgq__) &&         \
    !defined(__powerpc__) && !defined(__s390x__) && !defined(__arm__) && !defined(arm64) && !defined(__arm64) &&       \
    !defined(__arm64__) && !defined(__aarch64__)

#    include <einsums/coroutines/detail/context_linux_x86.hpp>
namespace einsums::threads::coroutines::detail {
template <typename CoroutineImpl>
using default_context_impl = lx::x86_linux_context_impl<CoroutineImpl>;
} // namespace einsums::threads::coroutines::detail

#elif defined(_POSIX_VERSION) || defined(__bgq__) || defined(__powerpc__) || defined(__s390x__) || defined(__arm__) || \
    defined(arm64) || defined(__arm64) || defined(__arm64__) || defined(__aarch64__)

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

#elif defined(EINSUMS_HAVE_FIBER_BASED_COROUTINES)

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

#endif // EINSUMS_HAVE_BOOST_CONTEXT
