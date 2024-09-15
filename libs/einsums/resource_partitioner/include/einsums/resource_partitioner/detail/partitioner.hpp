//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/affinity/affinity_data.hpp>
#include <einsums/assert.hpp>
#include <einsums/ini/ini.hpp>
#include <einsums/resource_partitioner/partitioner.hpp>
#include <einsums/threading_base/scheduler_mode.hpp>
#include <einsums/topology/cpu_mask.hpp>
#include <einsums/topology/topology.hpp>

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace einsums::resource::detail {

// Structure used to encapsulate all characteristics of thread_pools as
// specified by the user in int main().
struct init_pool_data {
    // mechanism for adding resources (zero-based index)
    void add_resource(std::size_t pu_index, bool exclusive, std::size_t num_threads);

    void print_pool(std::ostream &) const;

    void assign_pu(std::size_t virt_core);
    void unassign_pu(std::size_t virt_core);

    bool pu_is_exclusive(std::size_t virt_core) const;
    bool pu_is_assigned(std::size_t virt_core) const;

    void assign_first_core(std::size_t first_core);

    friend class resource::detail::partitioner;

    // counter ... overall, in all the thread pools
    static std::size_t num_threads_overall;

  private:
    init_pool_data(const std::string &name, scheduling_policy policy, einsums::threads::scheduler_mode mode);

    init_pool_data(std::string const &name, scheduler_function create_func, einsums::threads::scheduler_mode mode);

    std::string       pool_name_;
    scheduling_policy scheduling_policy_;

    // PUs this pool is allowed to run on
    std::vector<threads::detail::mask_type> assigned_pus_; // mask

    // pu index/exclusive/assigned
    std::vector<std::tuple<std::size_t, bool, bool>> assigned_pu_nums_;

    // counter for number of threads bound to this pool
    std::size_t                      num_threads_;
    einsums::threads::scheduler_mode mode_;
    scheduler_function               create_function_;
};

} // namespace einsums::resource::detail