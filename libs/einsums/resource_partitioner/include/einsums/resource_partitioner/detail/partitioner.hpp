//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once


#include <einsums/config.hpp>
#include <einsums/affinity/affinity_data.hpp>
#include <einsums/assert.hpp>
#include <einsums/concurrency/spinlock.hpp>
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
    ///////////////////////////////////////////////////////////////////////////
    // structure used to encapsulate all characteristics of thread_pools
    // as specified by the user in int main()
    class init_pool_data
    {
    public:
        // mechanism for adding resources (zero-based index)
        void add_resource(std::size_t pu_index, bool exclusive, std::size_t num_threads);

        void print_pool(std::ostream&) const;

        void assign_pu(std::size_t virt_core);
        void unassign_pu(std::size_t virt_core);

        bool pu_is_exclusive(std::size_t virt_core) const;
        bool pu_is_assigned(std::size_t virt_core) const;

        void assign_first_core(std::size_t first_core);

        friend class resource::detail::partitioner;

        // counter ... overall, in all the thread pools
        static std::size_t num_threads_overall;

    private:
        init_pool_data(
            const std::string& name, scheduling_policy policy, einsums::threads::scheduler_mode mode);

        init_pool_data(std::string const& name, scheduler_function create_func,
            einsums::threads::scheduler_mode mode);

        std::string pool_name_;
        scheduling_policy scheduling_policy_;

        // PUs this pool is allowed to run on
        std::vector<threads::detail::mask_type> assigned_pus_;    // mask

        // pu index/exclusive/assigned
        std::vector<std::tuple<std::size_t, bool, bool>> assigned_pu_nums_;

        // counter for number of threads bound to this pool
        std::size_t num_threads_;
        einsums::threads::scheduler_mode mode_;
        scheduler_function create_function_;
    };

    ///////////////////////////////////////////////////////////////////////
    class partitioner
    {
        using mutex_type = einsums::concurrency::detail::spinlock;

    public:
        partitioner();
        ~partitioner();

        void print_init_pool_data(std::ostream&) const;

        // create a thread_pool
        void create_thread_pool(std::string const& name,
            scheduling_policy sched = scheduling_policy::unspecified,
            einsums::threads::scheduler_mode = einsums::threads::scheduler_mode::default_mode);

        // create a thread_pool with a callback function for creating a custom
        // scheduler
        void create_thread_pool(std::string const& name, scheduler_function scheduler_creation);

        // Functions to add processing units to thread pools via
        // the pu/core/socket API
        void add_resource(
            einsums::resource::pu const& p, std::string const& pool_name, std::size_t num_threads = 1)
        {
            add_resource(p, pool_name, true, num_threads);
        }
        void add_resource(einsums::resource::pu const& p, std::string const& pool_name, bool exclusive,
            std::size_t num_threads = 1);
        void add_resource(const std::vector<einsums::resource::pu>& pv, std::string const& pool_name,
            bool exclusive = true);
        void add_resource(
            const einsums::resource::core& c, std::string const& pool_name, bool exclusive = true);
        void add_resource(const std::vector<einsums::resource::core>& cv, std::string const& pool_name,
            bool exclusive = true);
        void add_resource(
            const einsums::resource::socket& nd, std::string const& pool_name, bool exclusive = true);
        void add_resource(const std::vector<einsums::resource::socket>& ndv,
            std::string const& pool_name, bool exclusive = true);

        einsums::detail::affinity_data const& get_affinity_data() const { return affinity_data_; }

        // Does initialization of all resources and internal data of the
        // resource partitioner called in einsums_init
        void configure_pools();

        // returns the number of threads(pus) requested
        // by the user at startup.
        // This should not be called before the RP has parsed the config and
        // assigned affinity data
        std::size_t threads_needed()
        {
            EINSUMS_ASSERT(affinity_data_.get_num_pus_needed() != std::size_t(-1));
            return affinity_data_.get_num_pus_needed();
        }

        ////////////////////////////////////////////////////////////////////////
        scheduling_policy which_scheduler(std::string const& pool_name);
        threads::detail::topology& get_topology() const;

        std::size_t get_num_pools() const;

        std::size_t get_num_threads() const;
        std::size_t get_num_threads(std::string const& pool_name) const;
        std::size_t get_num_threads(std::size_t pool_index) const;

        einsums::threads::scheduler_mode get_scheduler_mode(std::size_t pool_index) const;

        std::string const& get_pool_name(std::size_t index) const;
        std::size_t get_pool_index(std::string const& pool_name) const;

        std::size_t get_pu_num(std::size_t global_thread_num);
        threads::detail::mask_cref_type get_pu_mask(std::size_t global_thread_num) const;

        void init(resource::partitioner_mode rpmode, einsums::detail::section cfg,
            einsums::detail::affinity_data affinity_data);

        scheduler_function get_pool_creator(size_t index) const;

        std::vector<socket> const& sockets() const { return sockets_; }

        std::size_t assign_cores(std::size_t first_core);

        // manage dynamic footprint of pools
        void assign_pu(std::string const& pool_name, std::size_t virt_core);
        void unassign_pu(std::string const& pool_name, std::size_t virt_core);

        std::size_t shrink_pool(std::string const& pool_name,
            util::detail::function<void(std::size_t)> const& remove_pu);
        std::size_t expand_pool(
            std::string const& pool_name, util::detail::function<void(std::size_t)> const& add_pu);

        void set_default_pool_name(const std::string& name)
        {
            initial_thread_pools_[0].pool_name_ = name;
        }

        const std::string& get_default_pool_name() const
        {
            return initial_thread_pools_[0].pool_name_;
        }

    private:
        ////////////////////////////////////////////////////////////////////////
        void fill_topology_vectors();
        bool pu_exposed(std::size_t pid);

        ////////////////////////////////////////////////////////////////////////
        // called in einsums_init run_or_start
        void setup_pools();
        void setup_schedulers();
        void reconfigure_affinities();
        void reconfigure_affinities_locked();
        bool check_empty_pools() const;

        // helper functions
        detail::init_pool_data const& get_pool_data(
            std::unique_lock<mutex_type>& l, std::size_t pool_index) const;

        // has to be private because pointers become invalid after data member
        // thread_pools_ is resized we don't want to allow the user to use it
        detail::init_pool_data const& get_pool_data(
            std::unique_lock<mutex_type>& l, std::string const& pool_name) const;
        detail::init_pool_data& get_pool_data(
            std::unique_lock<mutex_type>& l, std::string const& pool_name);

        void set_scheduler(scheduling_policy sched, std::string const& pool_name);

        ////////////////////////////////////////////////////////////////////////
        // counter for instance numbers
        static std::atomic<int> instance_number_counter_;

        // holds all of the command line switches
        einsums::detail::section rtcfg_;
        std::size_t first_core_;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        mutable mutex_type mtx_;
        std::vector<detail::init_pool_data> initial_thread_pools_;

        // reference to the topology and affinity data
        einsums::detail::affinity_data affinity_data_;

        // contains the internal topology back-end used to add resources to
        // initial_thread_pools
        std::vector<socket> sockets_;

        // store policy flags determining the general behavior of the
        // resource_partitioner
        resource::partitioner_mode mode_;

        // topology information
        threads::detail::topology& topo_;

        threads::scheduler_mode default_scheduler_mode_;
    };
}    // namespace einsums::resource::detail
