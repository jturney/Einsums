//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/modules/resource_partitioner.hpp>
#include <einsums/modules/thread_manager.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/thread_pool_helpers.hpp>
#include <einsums/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace einsums::resource {
std::size_t get_num_thread_pools() {
    return get_partitioner().get_num_pools();
}

std::size_t get_num_threads() {
    return get_partitioner().get_num_threads();
}

std::size_t get_num_threads(std::string const &pool_name) {
    return get_partitioner().get_num_threads(pool_name);
}

std::size_t get_num_threads(std::size_t pool_index) {
    return get_partitioner().get_num_threads(pool_index);
}

std::size_t get_pool_index(std::string const &pool_name) {
    return get_partitioner().get_pool_index(pool_name);
}

std::string const &get_pool_name(std::size_t pool_index) {
    return get_partitioner().get_pool_name(pool_index);
}

einsums::threads::detail::thread_pool_base &get_thread_pool(std::string const &pool_name) {
    return einsums::detail::get_runtime().get_thread_manager().get_pool(pool_name);
}

einsums::threads::detail::thread_pool_base &get_thread_pool(std::size_t pool_index) {
    return get_thread_pool(get_pool_name(pool_index));
}

bool pool_exists(std::string const &pool_name) {
    return einsums::detail::get_runtime().get_thread_manager().pool_exists(pool_name);
}

bool pool_exists(std::size_t pool_index) {
    return einsums::detail::get_runtime().get_thread_manager().pool_exists(pool_index);
}
} // namespace einsums::resource
