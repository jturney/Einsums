//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/threading_base/thread_pool_base.hpp>
#include <einsums/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace einsums::resource {
///////////////////////////////////////////////////////////////////////////
/// Return the number of thread pools currently managed by the
/// \a resource_partitioner
EINSUMS_EXPORT std::size_t get_num_thread_pools();

/// Return the number of threads in all thread pools currently
/// managed by the \a resource_partitioner
EINSUMS_EXPORT std::size_t get_num_threads();

/// Return the number of threads in the given thread pool currently
/// managed by the \a resource_partitioner
EINSUMS_EXPORT std::size_t get_num_threads(std::string const &pool_name);

/// Return the number of threads in the given thread pool currently
/// managed by the \a resource_partitioner
EINSUMS_EXPORT std::size_t get_num_threads(std::size_t pool_index);

/// Return the internal index of the pool given its name.
EINSUMS_EXPORT std::size_t get_pool_index(std::string const &pool_name);

/// Return the name of the pool given its internal index
EINSUMS_EXPORT std::string const &get_pool_name(std::size_t pool_index);

/// Return the name of the pool given its name
EINSUMS_EXPORT einsums::threads::detail::thread_pool_base &get_thread_pool(std::string const &pool_name);

/// Return the thread pool given its internal index
EINSUMS_EXPORT einsums::threads::detail::thread_pool_base &get_thread_pool(std::size_t pool_index);

/// Return true if the pool with the given name exists
EINSUMS_EXPORT bool pool_exists(std::string const &pool_name);

/// Return true if the pool with the given index exists
EINSUMS_EXPORT bool pool_exists(std::size_t pool_index);
} // namespace einsums::resource
