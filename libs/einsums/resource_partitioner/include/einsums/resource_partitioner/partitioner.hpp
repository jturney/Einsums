//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/function.hpp>
#include <einsums/ini/ini.hpp>
#include <einsums/resource_partitioner/detail/create_partitioner.hpp>
#include <einsums/resource_partitioner/partitioner_fwd.hpp>
#include <einsums/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace einsums::resource {
///////////////////////////////////////////////////////////////////////////
class pu {
    static constexpr const std::size_t invalid_pu_id = std::size_t(-1);

  public:
    explicit pu(std::size_t id = invalid_pu_id, core *core = nullptr, std::size_t thread_occupancy = 0)
        : _id(id), _core(core), _thread_occupancy(thread_occupancy), _thread_occupancy_count(0) {}

    std::size_t id() const { return _id; }

  private:
    friend class core;
    friend class socket;
    friend class resource::detail::partitioner;

    std::vector<pu> pus_sharing_core();
    std::vector<pu> pus_sharing_socket();

    std::size_t _id;
    core       *_core;

    // indicates the number of threads that should run on this PU
    //  0: this PU is not exposed by the affinity bindings
    //  1: normal occupancy
    // >1: oversubscription
    std::size_t _thread_occupancy;

    // counts number of threads bound to this PU
    mutable std::size_t _thread_occupancy_count;
};

class core {
    static constexpr const std::size_t invalid_core_id = std::size_t(-1);

  public:
    explicit core(std::size_t id = invalid_core_id, socket *socket = nullptr) : _id(id), _socket(socket) {}

    std::vector<pu> const &pus() const { return _pus; }
    std::size_t            id() const { return _id; }

  private:
    std::vector<core> cores_sharing_socket();

    friend class pu;
    friend class socket;
    friend class resource::detail::partitioner;

    std::size_t     _id;
    socket         *_socket;
    std::vector<pu> _pus;
};

class socket {
    static constexpr const std::size_t invalid_socket_id = std::size_t(-1);

  public:
    explicit socket(std::size_t id = invalid_socket_id) : _id(id) {}

    std::vector<core> const &cores() const { return _cores; }
    std::size_t              id() const { return _id; }

  private:
    friend class pu;
    friend class core;
    friend class resource::detail::partitioner;

    std::size_t       _id;
    std::vector<core> _cores;
};

///////////////////////////////////////////////////////////////////////////
namespace detail {
::einsums::resource::partitioner make_partitioner(resource::partitioner_mode rpmode, einsums::detail::section rtcfg,
                                                  einsums::detail::affinity_data affinity_data);
}

class partitioner {
  private:
    friend ::einsums::resource::partitioner detail::make_partitioner(resource::partitioner_mode rpmode, einsums::detail::section rtcfg,
                                                                     einsums::detail::affinity_data affinity_data);

    partitioner(resource::partitioner_mode rpmode, einsums::detail::section rtcfg, einsums::detail::affinity_data affinity_data);

  public:
    ///////////////////////////////////////////////////////////////////////
    // Create one of the predefined thread pools
    EINSUMS_EXPORT void create_thread_pool(std::string const &name, scheduling_policy sched = scheduling_policy::unspecified,
                                           einsums::threads::scheduler_mode = einsums::threads::scheduler_mode::default_mode);

    // Create a custom thread pool with a callback function
    EINSUMS_EXPORT void create_thread_pool(std::string const &name, scheduler_function scheduler_creation);

    // allow the default pool to be renamed to something else
    EINSUMS_EXPORT void set_default_pool_name(std::string const &name);

    EINSUMS_EXPORT const std::string &get_default_pool_name() const;

    ///////////////////////////////////////////////////////////////////////
    // Functions to add processing units to thread pools via
    // the pu/core/socket API
    void add_resource(einsums::resource::pu const &p, std::string const &pool_name, std::size_t num_threads = 1) {
        add_resource(p, pool_name, true, num_threads);
    }
    EINSUMS_EXPORT void add_resource(einsums::resource::pu const &p, std::string const &pool_name, bool exclusive,
                                     std::size_t num_threads = 1);
    EINSUMS_EXPORT void add_resource(std::vector<einsums::resource::pu> const &pv, std::string const &pool_name, bool exclusive = true);
    EINSUMS_EXPORT void add_resource(einsums::resource::core const &c, std::string const &pool_name, bool exclusive = true);
    EINSUMS_EXPORT void add_resource(std::vector<einsums::resource::core> &cv, std::string const &pool_name, bool exclusive = true);
    EINSUMS_EXPORT void add_resource(einsums::resource::socket const &nd, std::string const &pool_name, bool exclusive = true);
    EINSUMS_EXPORT void add_resource(std::vector<einsums::resource::socket> const &ndv, std::string const &pool_name,
                                     bool exclusive = true);

    // Access all available sockets
    EINSUMS_EXPORT std::vector<socket> const &sockets() const;

    // Returns the threads requested at startup --einsums:threads=cores
    // for example will return the number actually created
    EINSUMS_EXPORT std::size_t get_number_requested_threads();

    // return the topology object managed by the internal partitioner
    EINSUMS_EXPORT einsums::threads::detail::topology const &get_topology() const;

    // Does initialization of all resources and internal data of the
    // resource partitioner called in einsums_init
    EINSUMS_EXPORT void configure_pools();

  private:
    detail::partitioner &_partitioner;
};

} // namespace einsums::resource
