//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/schedulers/local_priority_queue_scheduler.hpp>
#include <einsums/schedulers/local_queue_scheduler.hpp>
#include <einsums/schedulers/shared_priority_queue_scheduler.hpp>
#include <einsums/schedulers/static_priority_queue_scheduler.hpp>
#include <einsums/schedulers/static_queue_scheduler.hpp>
#include <einsums/thread_pools/scheduled_thread_pool.hpp>
#include <einsums/thread_pools/scheduled_thread_pool_impl.hpp>

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread pools of our choice
template class EINSUMS_EXPORT einsums::threads::detail::local_queue_scheduler<>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<einsums::threads::detail::local_queue_scheduler<>>;

template class EINSUMS_EXPORT einsums::threads::detail::static_queue_scheduler<>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<einsums::threads::detail::static_queue_scheduler<>>;

template class EINSUMS_EXPORT einsums::threads::detail::local_priority_queue_scheduler<>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_fifo>>;

template class EINSUMS_EXPORT einsums::threads::detail::static_priority_queue_scheduler<>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<einsums::threads::detail::static_priority_queue_scheduler<>>;
#if defined(EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT)
template class EINSUMS_EXPORT einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_lifo>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_lifo>>;
#endif

#if defined(EINSUMS_HAVE_CXX11_STD_ATOMIC_128BIT)
template class EINSUMS_EXPORT
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_abp_fifo>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_abp_fifo>>;
template class EINSUMS_EXPORT
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_abp_lifo>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<
    einsums::threads::detail::local_priority_queue_scheduler<std::mutex, einsums::threads::detail::lockfree_abp_lifo>>;
#endif

template class EINSUMS_EXPORT einsums::threads::detail::shared_priority_queue_scheduler<>;
template class EINSUMS_EXPORT einsums::threads::detail::scheduled_thread_pool<einsums::threads::detail::shared_priority_queue_scheduler<>>;
