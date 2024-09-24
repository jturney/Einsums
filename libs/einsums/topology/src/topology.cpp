//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/topology/cpu_mask.hpp>
#include <einsums/topology/topology.hpp>
#include <einsums/type_support/unused.hpp>
#include <einsums/util/ios_flags_saver.hpp>

#include <fmt/ostream.h>

#include <cstddef>
#include <errno.h>
#include <hwloc.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if HWLOC_API_VERSION < 0x0001'0b00
#    define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

#if defined(_POSIX_VERSION)
#    include <sys/resource.h>
#    include <sys/syscall.h>
#endif

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

namespace einsums::threads::detail {

void write_to_log(char const *valuename, std::size_t value) {
    EINSUMS_LOG(debug, "topology: {}: {}", valuename, value);
}

void write_to_log_mask(char const *valuename, mask_cref_type value) {
    EINSUMS_LOG(debug, "topology: {}: {}", valuename, einsums::threads::detail::to_string(value));
}

void write_to_log(char const *valuename, std::vector<std::size_t> const &values) {
    EINSUMS_LOG(debug, "topology: {}s, size: {}", valuename, values.size());

    std::size_t i = 0;
    for (std::size_t value : values) {
        EINSUMS_LOG(debug, "topology: {}({}): {}", valuename, i++, value);
    }
}

void write_to_log_mask(char const *valuename, std::vector<mask_type> const &values) {
    EINSUMS_LOG(debug, "topology: {}s, size: {}", valuename, values.size());

    std::size_t i = 0;
    for (mask_cref_type value : values) {
        EINSUMS_LOG(debug, "topology: {}({}): {}", valuename, i++, einsums::threads::detail::to_string(value));
    }
}

std::size_t get_index(hwloc_obj_t obj) {
    if (obj->logical_index == ~0x0u) {
        return static_cast<std::size_t>(obj->os_index);
    }
    return static_cast<std::size_t>(obj->logical_index);
}

hwloc_obj_t adjust_node_obj(hwloc_obj_t node) noexcept {
#if HWLOC_API_VERSION >= 0x0002'0000
    // www.open-mpi.org/projects/hwloc/doc/hwloc-v2.0.0-letter.pdf:
    // Starting with hwloc v2.0, NUMA nodes are not in the main tree
    // anymore. They are attached under objects as Memory Children
    // on the side of normal children.
    while (hwloc_obj_type_is_memory(node->type))
        node = node->parent;
    EINSUMS_ASSERT(node);
#endif
    return node;
}

std::size_t get_memory_page_size_impl() {
#if defined(EINSUMS_HAVE_UNISTD_H)
    return sysconf(_SC_PAGE_SIZE);
#elif defined(EINSUMS_WINDOWS)
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    return static_cast<std::size_t>(systemInfo.dwPageSize);
#else
    return 4096;
#endif
}

std::size_t topology::_memory_page_size = get_memory_page_size_impl();

std::ostream &operator<<(std::ostream &os, hwloc_bitmap_wrapper const *bmp) {
    char buffer[256];
    hwloc_bitmap_snprintf(buffer, 256, bmp->_bmp);
    os << buffer;
    return os;
}

bool topology::reduce_thread_priority(error_code &ec) const {
    EINSUMS_UNUSED(ec);
#ifdef EINSUMS_HAVE_NICE_THREADLEVEL
#    if defined(__linux__) && !defined(__ANDROID__) && !defined(__bgq__)
    pid_t tid;
    tid = syscall(SYS_gettid);
    if (setpriority(PRIO_PROCESS, tid, 19)) {
        EINSUMS_THROWS_IF(ec, einsums::error::no_success, "topology::reduce_thread_priority", "setpriority returned an error");
        return false;
    }
#    elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__)

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST)) {
        EINSUMS_THROWS_IF(ec, einsums::error::no_success, "topology::reduce_thread_priority", "SetThreadPriority returned an error");
        return false;
    }
#    elif defined(__bgq__)
    ThreadPriority_Low();
#    endif
#endif
    return true;
}

topology &get_topology() {
    static topology topo;
    return topo;
}

static struct init_topology_t {
    init_topology_t() { get_topology(); }
} init_topology{};

#if !defined(EINSUMS_MAX_CPU_COUNT)
mask_type topology::empty_mask = mask_type(hardware_concurrency());
#else
mask_type topology::empty_mask = mask_type();
#endif

topology::topology() : topo(nullptr), _use_pus_as_cores(false), _machine_affinity_mask(0), _main_thread_affinity_mask(0) {
    int err = hwloc_topology_init(&topo);
    if (err != 0) {
        EINSUMS_THROW_EXCEPTION(error::no_success, "Failed to initialize hwloc topology");
    }

    err = hwloc_topology_load(topo);
    if (err != 0) {
        EINSUMS_THROW_EXCEPTION(error::no_success, "Failed to load hwloc topology");
    }

    init_num_of_pus();

    _socket_numbers.reserve(_num_of_pus);
    _numa_node_numbers.reserve(_num_of_pus);
    _core_numbers.reserve(_num_of_pus);

    // Initialize each set of data entirely, as some of the initialization
    // routines rely on access to other pieces of topology data. The
    // compiler will optimize the loops where possible anyways.

    std::size_t num_of_sockets = get_number_of_sockets();
    if (num_of_sockets == 0)
        num_of_sockets = 1;

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        std::size_t socket = init_socket_number(i);
        EINSUMS_ASSERT(socket < num_of_sockets);
        _socket_numbers.push_back(socket);
    }

    std::size_t num_of_nodes = get_number_of_numa_nodes();
    if (num_of_nodes == 0)
        num_of_nodes = 1;

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        std::size_t numa_node = init_numa_node_number(i);
        EINSUMS_ASSERT(numa_node < num_of_nodes);
        _numa_node_numbers.push_back(numa_node);
    }

    std::size_t num_of_cores = get_number_of_cores();
    if (num_of_cores == 0)
        num_of_cores = 1;

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        std::size_t core_number = init_core_number(i);
        EINSUMS_ASSERT(core_number < num_of_cores);
        _core_numbers.push_back(core_number);
    }

    _machine_affinity_mask = init_machine_affinity_mask();
    _socket_affinity_masks.reserve(_num_of_pus);
    _numa_node_affinity_masks.reserve(_num_of_pus);
    _core_affinity_masks.reserve(_num_of_pus);
    _thread_affinity_masks.reserve(_num_of_pus);

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        _socket_affinity_masks.push_back(init_socket_affinity_mask(i));
    }

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        _numa_node_affinity_masks.push_back(init_numa_node_affinity_mask(i));
    }

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        _core_affinity_masks.push_back(init_core_affinity_mask(i));
    }

    for (std::size_t i = 0; i < _num_of_pus; ++i) {
        _thread_affinity_masks.push_back(init_thread_affinity_mask(i));
    }

    // We assume here that the topology object is created in a global constructor on the main
    // thread (get_cpubind_mask returns the mask of the current thread).
    _main_thread_affinity_mask = get_cpubind_mask();
}

void topology::write_to_log() const {
    std::size_t num_of_sockets = get_number_of_sockets();
    if (num_of_sockets == 0)
        num_of_sockets = 1;
    detail::write_to_log("num_sockets", num_of_sockets);

    std::size_t num_of_nodes = get_number_of_numa_nodes();
    if (num_of_nodes == 0)
        num_of_nodes = 1;
    detail::write_to_log("num_of_nodes", num_of_nodes);

    std::size_t num_of_cores = get_number_of_cores();
    if (num_of_cores == 0)
        num_of_cores = 1;
    detail::write_to_log("num_of_cores", num_of_cores);

    detail::write_to_log("num_of_pus", _num_of_pus);

    detail::write_to_log("socket_number", _socket_numbers);
    detail::write_to_log("numa_node_number", _numa_node_numbers);
    detail::write_to_log("core_number", _core_numbers);

    detail::write_to_log_mask("machine_affinity_mask", _machine_affinity_mask);

    detail::write_to_log_mask("socket_affinity_mask", _socket_affinity_masks);
    detail::write_to_log_mask("numa_node_affinity_mask", _numa_node_affinity_masks);
    detail::write_to_log_mask("core_affinity_mask", _core_affinity_masks);
    detail::write_to_log_mask("thread_affinity_mask", _thread_affinity_masks);
}

topology::~topology() {
    if (topo)
        hwloc_topology_destroy(topo);
}

std::size_t topology::get_pu_number(std::size_t num_core, std::size_t num_pu, error_code &ec) const {
    std::unique_lock<mutex_type> lk(topo_mutex);

    int  num_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    bool use_pus   = false;

    // If num_cores is smaller 0, we have an error, it should never be zero
    // either to avoid division by zero, we should always have at least one
    // core
    if (num_core <= 0) {
        // on some platforms, hwloc can't report the number of cores (BSD),
        // fall back to report the number of PUs instead
        num_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
        if (num_cores <= 0) {
            EINSUMS_THROWS_IF(ec, error::no_success, "Failed to get number of cores");
            return std::size_t(-1);
        }
        use_pus = true;
    }
    num_core %= num_cores;

    hwloc_obj_t core_obj;
    if (!use_pus) {
        core_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));

        num_pu %= core_obj->arity;
        return std::size_t(core_obj->children[num_pu]->logical_index);
    }

    core_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, static_cast<unsigned>(num_core));

    return std::size_t(core_obj->logical_index);
}

mask_cref_type topology::get_machine_affinity_mask(error_code& ec) const
{
    if (&ec != &throws) ec = make_success_code();

    return _machine_affinity_mask;
}

mask_cref_type topology::get_socket_affinity_mask(std::size_t num_thread, einsums::error_code &ec) const {
    std::size_t num_pu = num_thread % _num_of_pus;

    if (num_pu < _socket_affinity_masks.size()) {
        if (&ec != &throws)
            ec = make_success_code();

        return _socket_affinity_masks[num_pu];
    }

    EINSUMS_THROWS_IF(ec, error::bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

mask_cref_type topology::get_numa_node_affinity_mask(std::size_t num_thread, error_code &ec) const {
    std::size_t num_pu = num_thread % _num_of_pus;

    if (num_pu < _numa_node_affinity_masks.size()) {
        if (&ec != &throws)
            ec = make_success_code();

        return _numa_node_affinity_masks[num_pu];
    }

    EINSUMS_THROWS_IF(ec, error::bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

mask_cref_type topology::get_core_affinity_mask(std::size_t num_thread, error_code &ec) const {
    std::size_t num_pu = num_thread % _num_of_pus;

    if (num_pu < _core_affinity_masks.size()) {
        if (&ec != &throws)
            ec = make_success_code();

        return _core_affinity_masks[num_pu];
    }

    EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

mask_cref_type topology::get_thread_affinity_mask(std::size_t num_thread, error_code &ec) const {
    std::size_t num_pu = num_thread % _num_of_pus;

    if (num_pu < _thread_affinity_masks.size()) {
        if (&ec != &throws)
            ec = make_success_code();

        return _thread_affinity_masks[num_pu];
    }

    EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

///////////////////////////////////////////////////////////////////////////
void topology::set_thread_affinity_mask(mask_cref_type mask, error_code &ec) const {
#if !defined(__APPLE__)
    // setting thread affinities is not supported by OSX
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);

    for (std::size_t i = 0; i != mask_size(mask); ++i) {
        if (test(mask, i)) {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, unsigned(i));
            EINSUMS_ASSERT(i == detail::get_index(pu_obj));
            hwloc_bitmap_set(cpuset, static_cast<unsigned int>(pu_obj->os_index));
        }
    }

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        if (hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_STRICT | HWLOC_CPUBIND_THREAD)) {
            // Strict binding not supported or failed, try weak binding.
            if (hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD)) {
                std::unique_ptr<char[]> buffer(new char[1024]);

                hwloc_bitmap_snprintf(buffer.get(), 1024, cpuset);
                hwloc_bitmap_free(cpuset);

                EINSUMS_THROWS_IF(ec, einsums::error::kernel_error, "failed to set thread affinity mask ({}) for cpuset {}",
                                  einsums::threads::detail::to_string(mask), buffer.get());
                return;
            }
        }
    }
#    if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    sleep(0); // Allow the OS to pick up the change.
#    endif
    hwloc_bitmap_free(cpuset);
#else
    EINSUMS_UNUSED(mask);
#endif // __APPLE__

    if (&ec != &throws)
        ec = make_success_code();
} // }}}

///////////////////////////////////////////////////////////////////////////
mask_type topology::get_thread_affinity_mask_from_lva(void const *lva, error_code &ec) const { // {{{
    if (&ec != &throws)
        ec = make_success_code();

    hwloc_membind_policy_t policy  = ::HWLOC_MEMBIND_DEFAULT;
    hwloc_nodeset_t        nodeset = hwloc_bitmap_alloc();

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        int                          ret =
#if HWLOC_API_VERSION >= 0x0001'0b06
            hwloc_get_area_membind(topo, lva, 1, nodeset, &policy, HWLOC_MEMBIND_BYNODESET);
#else
            hwloc_get_area_membind_nodeset(topo, lva, 1, nodeset, &policy, 0);
#endif

        if (-1 != ret) {
            hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
            hwloc_cpuset_from_nodeset(topo, cpuset, nodeset);
            lk.unlock();

            hwloc_bitmap_free(nodeset);

            mask_type mask = mask_type();
            resize(mask, get_number_of_pus());

            int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
            for (unsigned int i = 0; std::size_t(i) != _num_of_pus; ++i) {
                hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
                unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
                if (hwloc_bitmap_isset(cpuset, idx) != 0)
                    set(mask, detail::get_index(pu_obj));
            }

            hwloc_bitmap_free(cpuset);
            return mask;
        } else {
            std::string errstr = std::strerror(errno);

            lk.unlock();
            EINSUMS_THROW_EXCEPTION(einsums::error::no_success, "failed calling 'hwloc_get_area_membind_nodeset', reported error: {}",
                                    errstr);
        }
    }

    hwloc_bitmap_free(nodeset);
    return empty_mask;
} // }}}

std::size_t topology::init_numa_node_number(std::size_t num_thread) {
#if HWLOC_API_VERSION >= 0x0002'0000
    if (std::size_t(-1) == num_thread)
        return std::size_t(-1);

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    hwloc_obj_t obj;
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
        EINSUMS_ASSERT(num_pu == detail::get_index(obj));
    }

    hwloc_obj_t tmp = nullptr;
    while ((tmp = hwloc_get_next_obj_by_type(topo, HWLOC_OBJ_NUMANODE, tmp)) != nullptr) {
        if (hwloc_bitmap_intersects(tmp->cpuset, obj->cpuset)) {
            /* tmp matches, use it */
            return tmp->logical_index;
        }
    }
    return 0;
#else
    return init_node_number(num_thread, HWLOC_OBJ_NODE);
#endif
}

std::size_t topology::init_node_number(std::size_t num_thread, hwloc_obj_type_t type) { // {{{
    if (std::size_t(-1) == num_thread)
        return std::size_t(-1);

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    {
        hwloc_obj_t obj;

        {
            std::unique_lock<mutex_type> lk(topo_mutex);
            obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
            EINSUMS_ASSERT(num_pu == detail::get_index(obj));
        }

        while (obj) {
            if (hwloc_compare_types(obj->type, type) == 0) {
                return detail::get_index(obj);
            }
            obj = obj->parent;
        }
    }

    return 0;
} // }}}

void topology::extract_node_mask(hwloc_obj_t parent, mask_type &mask) const { // {{{
    hwloc_obj_t obj;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_next_child(topo, parent, nullptr);
    }

    while (obj) {
        if (hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0) {
            do {
                set(mask, detail::get_index(obj)); //-V106
                {
                    std::unique_lock<mutex_type> lk(topo_mutex);
                    obj = hwloc_get_next_child(topo, parent, obj);
                }
            } while (obj != nullptr && hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0);
            return;
        }

        extract_node_mask(obj, mask);

        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_next_child(topo, parent, obj);
    }
} // }}}

std::size_t topology::extract_node_count(hwloc_obj_t parent, hwloc_obj_type_t type, std::size_t count) const { // {{{
    hwloc_obj_t obj;

    if (parent == nullptr) {
        return count;
    }

    if (hwloc_compare_types(type, parent->type) == 0) {
        return count;
    }

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_next_child(topo, parent, nullptr);
    }

    while (obj) {
        if (hwloc_compare_types(type, obj->type) == 0) {
            /*
            do {
                ++count;
                {
                    std::unique_lock<mutex_type> lk(topo_mutex);
                    obj = hwloc_get_next_child(topo, parent, obj);
                }
            } while (obj != nullptr && hwloc_compare_types(type, obj->type) == 0);
            return count;
            */
            ++count;
        }

        count = extract_node_count(obj, type, count);

        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_next_child(topo, parent, obj);
    }

    return count;
} // }}}

std::size_t topology::get_number_of_sockets() const {
    int nobjs = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_SOCKET);
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type failed");
        return std::size_t(nobjs);
    }
    return std::size_t(nobjs);
}

std::size_t topology::get_number_of_numa_nodes() const {
    int nobjs = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type failed");
        return std::size_t(nobjs);
    }
    return std::size_t(nobjs);
}

std::size_t topology::get_number_of_cores() const {
    int nobjs = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);

    // If num_cores is smaller 0, we have an error
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type(HWLOC_OBJ_CORE) failed");
        return std::size_t(nobjs);
    } else if (0 == nobjs) {
        // some platforms report zero cores but might still report the
        // number of PUs
        nobjs = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
        if (0 > nobjs) {
            EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type(HWLOC_OBJ_PU) failed");
            return std::size_t(nobjs);
        }
    }

    // the number of reported cores/pus should never be zero either to
    // avoid division by zero, we should always have at least one core
    if (0 == nobjs) {
        EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type reports zero cores/pus");
        return std::size_t(nobjs);
    }

    return std::size_t(nobjs);
}

std::size_t topology::get_number_of_socket_pus(std::size_t num_socket) const {
    hwloc_obj_t socket_obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        socket_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));
        std::size_t pu_count = 0;
        return extract_node_count(socket_obj, HWLOC_OBJ_PU, pu_count);
    }

    return _num_of_pus;
}

std::size_t topology::get_number_of_numa_node_pus(std::size_t numa_node) const {
    hwloc_obj_t node_obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        node_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(node_obj));
        std::size_t pu_count = 0;
        node_obj             = detail::adjust_node_obj(node_obj);
        return extract_node_count(node_obj, HWLOC_OBJ_PU, pu_count);
    }

    return _num_of_pus;
}

std::size_t topology::get_number_of_core_pus(std::size_t core) const {
    hwloc_obj_t core_obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        core_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, static_cast<unsigned>(core));
    }

    if (!_use_pus_as_cores && core_obj) {
        EINSUMS_ASSERT(core == detail::get_index(core_obj));
        std::size_t pu_count = 0;
        return extract_node_count(core_obj, HWLOC_OBJ_PU, pu_count);
    }

    return std::size_t(1);
}

std::size_t topology::get_number_of_socket_cores(std::size_t num_socket) const {
    hwloc_obj_t socket_obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        socket_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));
        std::size_t pu_count = 0;
        return extract_node_count(socket_obj, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, pu_count);
    }

    return get_number_of_cores();
}

std::size_t topology::get_number_of_numa_node_cores(std::size_t numa_node) const {
    hwloc_obj_t node_obj = nullptr;
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        node_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(node_obj));
        std::size_t pu_count = 0;
        node_obj             = detail::adjust_node_obj(node_obj);
        return extract_node_count(node_obj, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, pu_count);
    }

    return get_number_of_cores();
}

hwloc_bitmap_ptr topology::cpuset_to_nodeset(mask_cref_type mask) const {
    hwloc_bitmap_t cpuset  = mask_to_bitmap(mask, HWLOC_OBJ_PU);
    hwloc_bitmap_t nodeset = hwloc_bitmap_alloc();
#if HWLOC_API_VERSION >= 0x0002'0000
    hwloc_cpuset_to_nodeset(topo, cpuset, nodeset);
#else
    hwloc_cpuset_to_nodeset_strict(topo, cpuset, nodeset);
#endif
    hwloc_bitmap_free(cpuset);
    return std::make_shared<hwloc_bitmap_wrapper>(nodeset);
}

void print_info(std::ostream &os, hwloc_obj_t obj, char const *name, bool comma) {
    if (comma)
        os << ", ";
    os << name;

    if (obj->logical_index != ~0x0u)
        os << "L#" << obj->logical_index;
    if (obj->os_index != ~0x0u)
        os << "(P#" << obj->os_index << ")";
}

void print_info(std::ostream &os, hwloc_obj_t obj, bool comma = false) {
    switch (obj->type) {
    case HWLOC_OBJ_PU:
        print_info(os, obj, "PU ", comma);
        break;

    case HWLOC_OBJ_CORE:
        print_info(os, obj, "Core ", comma);
        break;

    case HWLOC_OBJ_SOCKET:
        print_info(os, obj, "Socket ", comma);
        break;

    case HWLOC_OBJ_NODE:
        print_info(os, obj, "NUMANode ", comma);
        break;

    default:
        break;
    }
}

void topology::print_affinity_mask(std::ostream &os, std::size_t num_thread, mask_cref_type m, const std::string &pool_name) const {
    einsums::detail::ios_flags_saver ifs(os);
    bool                             first = true;

    if (!threads::detail::any(m)) {
        os << std::setw(4) << num_thread << ": thread binding disabled" << ", on pool \"" << pool_name << "\"" << std::endl;
        return;
    }

    for (std::size_t i = 0; i != _num_of_pus; ++i) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, unsigned(i));
        if (!obj) {
            EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "object not found");
            return;
        }

        if (!test(m, detail::get_index(obj))) //-V106
            continue;

        if (first) {
            first = false;
            os << std::setw(4) << num_thread << ": "; //-V112 //-V128
        } else {
            os << "      ";
        }

        detail::print_info(os, obj);

        while (obj->parent) {
            detail::print_info(os, obj->parent, true);
            obj = obj->parent;
        }

        os << ", on pool \"" << pool_name << "\"";

        os << std::endl;
    }
}

mask_type topology::init_machine_affinity_mask() const { // {{{
    mask_type machine_affinity_mask = mask_type();
    resize(machine_affinity_mask, get_number_of_pus());

    hwloc_obj_t machine_obj;
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        machine_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_MACHINE, 0);
    }
    if (machine_obj) {
        extract_node_mask(machine_obj, machine_affinity_mask);
        return machine_affinity_mask;
    }

    EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "failed to initialize machine affinity mask");
    return empty_mask;
} // }}}

mask_type topology::init_socket_affinity_mask_from_socket(std::size_t num_socket) const {
    // If we have only one or no socket, the socket affinity mask
    // spans all processors
    if (std::size_t(-1) == num_socket)
        return _machine_affinity_mask;

    hwloc_obj_t socket_obj = nullptr;
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        socket_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));

        mask_type socket_affinity_mask = mask_type();
        resize(socket_affinity_mask, get_number_of_pus());

        extract_node_mask(socket_obj, socket_affinity_mask);
        return socket_affinity_mask;
    }

    return _machine_affinity_mask;
}

mask_type topology::init_numa_node_affinity_mask_from_numa_node(std::size_t numa_node) const {
    // If we have only one or no NUMA domain, the NUMA affinity mask
    // spans all processors
    if (std::size_t(-1) == numa_node) {
        return _machine_affinity_mask;
    }

    hwloc_obj_t numa_node_obj = nullptr;
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        numa_node_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (numa_node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(numa_node_obj));
        mask_type node_affinity_mask = mask_type();
        resize(node_affinity_mask, get_number_of_pus());

        numa_node_obj = detail::adjust_node_obj(numa_node_obj);
        extract_node_mask(numa_node_obj, node_affinity_mask);
        return node_affinity_mask;
    }

    return _machine_affinity_mask;
}

mask_type topology::init_core_affinity_mask_from_core(std::size_t core, mask_cref_type default_mask) const {
    if (std::size_t(-1) == core) {
        return default_mask;
    }

    hwloc_obj_t core_obj = nullptr;

    std::size_t num_core = (core + core_offset) % get_number_of_cores();

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        core_obj = hwloc_get_obj_by_type(topo, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));
    }

    if (core_obj) {
        EINSUMS_ASSERT(num_core == detail::get_index(core_obj));
        mask_type core_affinity_mask = mask_type();
        resize(core_affinity_mask, get_number_of_pus());

        extract_node_mask(core_obj, core_affinity_mask);
        return core_affinity_mask;
    }

    return default_mask;
}

mask_type topology::init_thread_affinity_mask(std::size_t num_thread) const {

    if (std::size_t(-1) == num_thread) {
        return get_core_affinity_mask(num_thread);
    }

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    hwloc_obj_t obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
    }

    if (!obj) {
        return get_core_affinity_mask(num_thread);
    }

    EINSUMS_ASSERT(num_pu == detail::get_index(obj));
    mask_type mask = mask_type();
    resize(mask, get_number_of_pus());

    set(mask, detail::get_index(obj)); //-V106

    return mask;
}

mask_type topology::init_thread_affinity_mask(std::size_t num_core, std::size_t num_pu) const {
    hwloc_obj_t obj = nullptr;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        int                          num_cores = hwloc_get_nbobjs_by_type(topo, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);

        // If num_cores is smaller 0, we have an error, it should never be zero
        // either to avoid division by zero, we should always have at least one
        // core
        if (num_cores <= 0) {
            EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_get_nbobjs_by_type failed");
            return empty_mask;
        }

        num_core = (num_core + core_offset) % std::size_t(num_cores);
        obj      = hwloc_get_obj_by_type(topo, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));
    }

    if (!obj)
        return empty_mask; // get_core_affinity_mask(num_thread, false);

    EINSUMS_ASSERT(num_core == detail::get_index(obj));

    mask_type mask = mask_type();
    resize(mask, get_number_of_pus());

    if (_use_pus_as_cores) {
        set(mask, detail::get_index(obj)); //-V106
    } else {
        num_pu %= obj->arity;                                //-V101 //-V104
        set(mask, detail::get_index(obj->children[num_pu])); //-V106
    }

    return mask;
}

///////////////////////////////////////////////////////////////////////////
void topology::init_num_of_pus() {
    _num_of_pus       = 1;
    _use_pus_as_cores = false;

    {
        std::unique_lock<mutex_type> lk(topo_mutex);

        // on some platforms, hwloc can't report the number of cores (BSD),
        // in this case we use PUs as cores
        if (hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE) <= 0) {
            _use_pus_as_cores = true;
        }

        int num_of_pus = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
        if (num_of_pus > 0) {
            _num_of_pus = static_cast<std::size_t>(num_of_pus);
        }
    }
}

std::size_t topology::get_number_of_pus() const {
    return _num_of_pus;
}

///////////////////////////////////////////////////////////////////////////
mask_type topology::get_cpubind_mask_main_thread(error_code &) const {
    return _main_thread_affinity_mask;
}

void topology::set_cpubind_mask_main_thread(mask_type mask, error_code &ec) {
    auto const concurrency = hardware_concurrency();
    auto const size        = mask_size(mask);

    // If the given mask is smaller than the hardware concurrency, we simply resize it to
    // contain hardware concurrency bits.
    if (size < concurrency) {
        resize(mask, concurrency);
    }
    // If the given mask is larger than the hardware concurrency, we may still be able to
    // use it if the bits past hardware concurrency are unset. We mask shift away the bits
    // that are allowed to be set and check if there are any remaining bits set.
    else if (mask_size(mask) > concurrency && any(mask >> concurrency)) {
        EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter, "CPU mask ({}) has bits set past the hardware concurrency of the system ({})",
                          fmt::streamed(mask), concurrency);
    }

    if (!any(mask)) {
        EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter,
                          "CPU mask is empty ({}), make sure it has at least one bit set through "
                          "EINSUMS_PROCESS_MASK or --einsums:process-mask",
                          fmt::streamed(mask));
    }

    // The mask is assumed to use physical/OS indices (as returned by e.g. hwloc-bind --get
    // --taskset or taskset --pid) while einsums deals with logical indices from this point
    // onwards. We convert the mask from physical indices to logical indices before storing it.
    mask_type logical_mask{};
    resize(logical_mask, get_number_of_pus());

#if !defined(__APPLE__)
    {
        std::unique_lock<mutex_type> lk(topo_mutex);

        int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != get_number_of_pus(); ++i) {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            EINSUMS_ASSERT(i == detail::get_index(pu_obj));
            if (test(mask, idx)) {
                set(logical_mask, detail::get_index(pu_obj));
            }
        }
    }
#endif // __APPLE__

    _main_thread_affinity_mask = std::move(logical_mask);
}

mask_type topology::get_cpubind_mask(error_code &ec) const {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    mask_type mask = mask_type();
    resize(mask, get_number_of_pus());

#if !defined(__APPLE__)
    {
        std::unique_lock<mutex_type> lk(topo_mutex);
        if (hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD)) {
            hwloc_bitmap_free(cpuset);
            EINSUMS_THROWS_IF(ec, einsums::error::kernel_error, "hwloc_get_cpubind failed");
            return empty_mask;
        }

        int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != _num_of_pus; ++i) //-V104
        {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            if (hwloc_bitmap_isset(cpuset, idx) != 0)
                set(mask, detail::get_index(pu_obj));
        }
    }
#endif // __APPLE__

    hwloc_bitmap_free(cpuset);

    if (&ec != &throws)
        ec = make_success_code();

    return mask;
}

mask_type topology::get_cpubind_mask(std::thread &handle, error_code &ec) const {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    mask_type mask = mask_type();
    resize(mask, get_number_of_pus());

    {
        std::unique_lock<mutex_type> lk(topo_mutex);
#if defined(EINSUMS_MINGW)
        if (hwloc_get_thread_cpubind(topo, pthread_gethandle(handle.native_handle()), cpuset, HWLOC_CPUBIND_THREAD))
#else
        if (hwloc_get_thread_cpubind(topo, handle.native_handle(), cpuset, HWLOC_CPUBIND_THREAD))
#endif
        {
            hwloc_bitmap_free(cpuset);
            EINSUMS_THROWS_IF(ec, einsums::error::kernel_error, "hwloc_get_cpubind failed");
            return empty_mask;
        }

        int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != _num_of_pus; ++i) //-V104
        {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            if (hwloc_bitmap_isset(cpuset, idx) != 0)
                set(mask, detail::get_index(pu_obj));
        }
    }

    hwloc_bitmap_free(cpuset);

    if (&ec != &throws)
        ec = make_success_code();

    return mask;
}

///////////////////////////////////////////////////////////////////////////
/// This is equivalent to malloc(), except that it tries to allocate
/// page-aligned memory from the OS.
void *topology::allocate(std::size_t len) const {
    return hwloc_alloc(topo, len);
}

///////////////////////////////////////////////////////////////////////////
/// Allocate some memory on NUMA memory nodes specified by nodeset
/// as specified by the hwloc hwloc_alloc_membind_nodeset call
void *topology::allocate_membind(std::size_t len, hwloc_bitmap_ptr bitmap, hwloc_membind_policy policy, int flags) const {
    return
#if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_alloc_membind(topo, len, bitmap->get_bmp(), (hwloc_membind_policy_t)(policy), flags | HWLOC_MEMBIND_BYNODESET);
#else
        hwloc_alloc_membind_nodeset(topo, len, bitmap->get_bmp(), (hwloc_membind_policy_t)(policy), flags);
#endif
}

bool topology::set_area_membind_nodeset(const void *addr, std::size_t len, void *nodeset) const {
#if !defined(__APPLE__)
    hwloc_membind_policy_t policy = ::HWLOC_MEMBIND_BIND;
    hwloc_nodeset_t        ns     = reinterpret_cast<hwloc_nodeset_t>(nodeset);

    int ret =
#    if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_set_area_membind(topo, addr, len, ns, policy, HWLOC_MEMBIND_BYNODESET);
#    else
        hwloc_set_area_membind_nodeset(topo, addr, len, ns, policy, 0);
#    endif

    if (ret < 0) {
        std::string msg = std::strerror(errno);
        if (errno == ENOSYS)
            msg = "the action is not supported";
        if (errno == EXDEV)
            msg = "the binding cannot be enforced";
        EINSUMS_THROW_EXCEPTION(einsums::error::kernel_error, "hwloc_set_area_membind_nodeset failed : {}", msg);
        return false;
    }
#else
    EINSUMS_UNUSED(addr);
    EINSUMS_UNUSED(len);
    EINSUMS_UNUSED(nodeset);
#endif
    return true;
}

namespace {
hwloc_bitmap_wrapper &bitmap_storage() {
    static thread_local hwloc_bitmap_wrapper bitmap_storage_(nullptr);

    return bitmap_storage_;
}
} // namespace

threads::detail::mask_type topology::get_area_membind_nodeset(const void *addr, std::size_t len) const {
    hwloc_bitmap_wrapper &nodeset = bitmap_storage();
    if (!nodeset) {
        nodeset.reset(hwloc_bitmap_alloc());
    }

    //
    hwloc_membind_policy_t policy;
    hwloc_nodeset_t        ns = reinterpret_cast<hwloc_nodeset_t>(nodeset.get_bmp());

    if (
#if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_get_area_membind(topo, addr, len, ns, &policy, HWLOC_MEMBIND_BYNODESET)
#else
        hwloc_get_area_membind_nodeset(topo, addr, len, ns, &policy, 0)
#endif
        == -1) {
        EINSUMS_THROW_EXCEPTION(error::kernel_error, "hwloc_get_area_membind_nodeset failed");
        return bitmap_to_mask(ns, HWLOC_OBJ_MACHINE);
    }

    return bitmap_to_mask(ns, HWLOC_OBJ_NUMANODE);
}

int topology::get_numa_domain(const void *addr) const {
#if HWLOC_API_VERSION >= 0x0001'0b06
    hwloc_bitmap_wrapper &nodeset = bitmap_storage();
    if (!nodeset) {
        nodeset.reset(hwloc_bitmap_alloc());
    }

    //
    hwloc_nodeset_t ns = reinterpret_cast<hwloc_nodeset_t>(nodeset.get_bmp());

    int ret = hwloc_get_area_memlocation(topo, addr, 1, ns, HWLOC_MEMBIND_BYNODESET);
    if (ret < 0) {
#    if defined(__FreeBSD__)
        // on some platforms this API is not supported (e.g. FreeBSD)
        return 0;
#    else
        std::string msg(strerror(errno));
        EINSUMS_THROW_EXCEPTION(error::kernel_error, "hwloc_get_area_memlocation failed {}", msg);
        return -1;
#    endif
    }

    threads::detail::mask_type mask = bitmap_to_mask(ns, HWLOC_OBJ_NUMANODE);
    return static_cast<int>(threads::detail::find_first(mask));
#else
    EINSUMS_UNUSED(addr);
    return 0;
#endif
}

/// Free memory that was previously allocated by allocate
void topology::deallocate(void *addr, std::size_t len) const {
    hwloc_free(topo, addr, len);
}

///////////////////////////////////////////////////////////////////////////
hwloc_bitmap_t topology::mask_to_bitmap(mask_cref_type mask, hwloc_obj_type_t htype) const {
    hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
    hwloc_bitmap_zero(bitmap);
    //
    int const depth = hwloc_get_type_or_below_depth(topo, htype);

    for (std::size_t i = 0; i != mask_size(mask); ++i) {
        if (test(mask, i)) {
            hwloc_obj_t const hw_obj = hwloc_get_obj_by_depth(topo, depth, unsigned(i));
            EINSUMS_ASSERT(i == detail::get_index(hw_obj));
            hwloc_bitmap_set(bitmap, static_cast<unsigned int>(hw_obj->os_index));
        }
    }
    return bitmap;
}

///////////////////////////////////////////////////////////////////////////
mask_type topology::bitmap_to_mask(hwloc_bitmap_t bitmap, hwloc_obj_type_t htype) const {
    mask_type mask = mask_type();
    resize(mask, get_number_of_pus());
    std::size_t num = hwloc_get_nbobjs_by_type(topo, htype);
    //
    int const pu_depth = hwloc_get_type_or_below_depth(topo, htype);
    for (unsigned int i = 0; std::size_t(i) != num; ++i) //-V104
    {
        hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
        unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
        if (hwloc_bitmap_isset(bitmap, idx) != 0)
            set(mask, detail::get_index(pu_obj));
    }
    return mask;
}

///////////////////////////////////////////////////////////////////////////
void topology::print_mask_vector(std::ostream &os, std::vector<mask_type> const &v) const {
    std::size_t s = v.size();
    if (s == 0) {
        os << "(empty)\n";
        return;
    }

    for (std::size_t i = 0; i != s; i++) {
        os << einsums::threads::detail::to_string(v[i]) << "\n";
    }
    os << "\n";
}

void topology::print_vector(std::ostream &os, std::vector<std::size_t> const &v) const {
    std::size_t s = v.size();
    if (s == 0) {
        os << "(empty)\n";
        return;
    }

    os << v[0];
    for (std::size_t i = 1; i != s; i++) {
        os << ", " << std::dec << v[i];
    }
    os << "\n";
}

void topology::print_hwloc(std::ostream &os) const {
    os << "[HWLOC topology info] number of ...\n"
       << std::dec << "number of sockets     : " << get_number_of_sockets() << "\n"
       << "number of numa nodes  : " << get_number_of_numa_nodes() << "\n"
       << "number of cores       : " << get_number_of_cores() << "\n"
       << "number of PUs         : " << get_number_of_pus() << "\n"
       << "hardware concurrency  : " << hardware_concurrency() << "\n"
       << std::endl;
    //! -------------------------------------- topology (affinity masks)
    os << "[HWLOC topology info] affinity masks :\n"
       << "machine               : \n"
       << einsums::threads::detail::to_string(_machine_affinity_mask) << "\n";

    os << "socket                : \n";
    print_mask_vector(os, _socket_affinity_masks);
    os << "numa node             : \n";
    print_mask_vector(os, _numa_node_affinity_masks);
    os << "core                  : \n";
    print_mask_vector(os, _core_affinity_masks);
    os << "PUs (/threads)        : \n";
    print_mask_vector(os, _thread_affinity_masks);

    //! -------------------------------------- topology (numbers)
    os << "[HWLOC topology info] resource numbers :\n";
    os << "socket                : \n";
    print_vector(os, _socket_numbers);
    os << "numa node             : \n";
    print_vector(os, _numa_node_numbers);
    os << "core                  : \n";
    print_vector(os, _core_numbers);
    // os << "PUs (/threads)        : \n";
    // print_vector(os, pu_numbers_);
}

///////////////////////////////////////////////////////////////////////////
struct hw_concurrency {
    hw_concurrency() noexcept : num_of_cores_(get_topology().get_number_of_pus()) {
        if (num_of_cores_ == 0)
            num_of_cores_ = 1;
    }

    std::size_t num_of_cores_;
};

unsigned int hardware_concurrency() noexcept {
    static detail::hw_concurrency hwc;
    return static_cast<unsigned int>(hwc.num_of_cores_);
}
} // namespace einsums::threads::detail
