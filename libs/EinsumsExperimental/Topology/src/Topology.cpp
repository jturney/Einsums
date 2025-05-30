//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Topology/CPUMask.hpp>
#include <Einsums/Topology/Topology.hpp>
#include <Einsums/TypeSupport/Unused.hpp>

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

#if defined(__ANDROID__) && defined(ANDROID)
#    include <cpu-features.h>
#endif

#if defined(_POSIX_VERSION)
#    include <sys/resource.h>
#    include <sys/syscall.h>
#endif

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

namespace einsums::topology::detail {

void write_to_log(char const *valuename, std::size_t value) {
    EINSUMS_LOG_DEBUG("topology: {}: {}", valuename, value);
}

void write_to_log_mask(char const *valuename, MaskCRefType value) {
    EINSUMS_LOG_DEBUG("topology: {}: {}", valuename, to_string(value));
}

void write_to_log(char const *valuename, std::vector<std::size_t> const &values) {
    EINSUMS_LOG_DEBUG("topology: {}s, size: {}", valuename, values.size());

    std::size_t i = 0;
    for (std::size_t value : values) {
        EINSUMS_LOG_DEBUG("topology: {}({}): {}", valuename, i++, value);
    }
}

void write_to_log_mask(char const *valuename, std::vector<MaskType> const &values) {
    EINSUMS_LOG_DEBUG("topology: {}s, size: {}", valuename, values.size());

    std::size_t i = 0;
    for (MaskCRefType value : values) {
        EINSUMS_LOG_DEBUG("topology: {}({}): {}", valuename, i++, to_string(value));
    }
}

auto get_index(hwloc_obj_t obj) -> std::size_t {
    // On Windows logical_index is always -1
    if (obj->logical_index == ~0x0u)
        return static_cast<std::size_t>(obj->os_index);

    return static_cast<std::size_t>(obj->logical_index);
}

auto adjust_node_obj(hwloc_obj_t node) noexcept -> hwloc_obj_t {
#if HWLOC_API_VERSION >= 0x0002'0000
    while (hwloc_obj_type_is_memory(node->type))
        node = node->parent;
    EINSUMS_ASSERT(node);
#endif
    return node;
}

auto get_memory_page_size_impl() -> std::size_t {
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

std::size_t Topology::_memory_page_size = get_memory_page_size_impl();

auto operator<<(std::ostream &os, HwlocBitmapWrapper const *bitmap) -> std::ostream & {
    char buffer[256];
    hwloc_bitmap_snprintf(buffer, 256, bitmap->_bitmap);
    os << buffer;
    return os;
}

auto Topology::reduce_thread_priority() -> bool {
#ifdef EINSUMS_HAVE_NICE_THREADLEVEL
#    if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
    if (setpriority(PRIO_PROCESS, tid, 19)) {
        EINSUMS_THROW_EXCEPTION(system_error, "setpriority returned an error");
        return false;
    }
#    elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
    if (!SetTHreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST)) {
        EINSUMS_THROW_EXCEPTION(system_error, "SetThreadPriority returned an error");
        return false;
    }
#    endif
#endif
    return true;
}

#if !defined(EINSUMS_HAVE_MAX_CPU_COUNT)
MaskType Topologoy::empty_mask = MaskType(hardware_concurrency());
#else
MaskType Topology::empty_mask = MaskType();
#endif

EINSUMS_SINGLETON_IMPL(Topology);

Topology::Topology() : _topology(nullptr), _use_pus_as_cores(false), _machine_affinity_mask(0), _main_thread_affinity_mask(0) {
    int err = hwloc_topology_init(&_topology);
    if (err != 0) {
        EINSUMS_THROW_EXCEPTION(no_success, "Failed to init hwloc topology");
    }

    err = hwloc_topology_set_flags(_topology,
#if HWLOC_API_VERSION < 0x0002'0000
                                   HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM
#else
                                   HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED
#endif
    );
    if (err != 0) {
        EINSUMS_THROW_EXCEPTION(no_success, "Failed to set HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED flag for hwloc topology");
    }

    err = hwloc_topology_load(_topology);
    if (err != 0) {
        EINSUMS_THROW_EXCEPTION(no_success, "Failed to load hwloc topology");
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

void Topology::write_to_log() {
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

Topology::~Topology() {
    if (_topology)
        hwloc_topology_destroy(_topology);
}

auto Topology::get_pu_number(std::size_t num_core, std::size_t num_pu) -> std::size_t {
    std::unique_lock lk(_topology_mutex);

    int  num_cores = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_CORE);
    bool use_pus   = false;

    if (num_cores <= 0) {
        // on some platforms, hwloc can't report the number of cores (BSD)
        // fall back to the number of PUs instead
        num_cores = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_PU);
        if (num_cores <= 0) {
            EINSUMS_THROW_EXCEPTION(no_success, "Failed to get number of cores");
            return static_cast<std::size_t>(-1);
        }
        use_pus = true;
    }
    num_core %= num_cores;

    hwloc_obj_t core_obj;
    if (!use_pus) {
        core_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));

        num_pu %= core_obj->arity;
        return static_cast<std::size_t>(core_obj->children[num_pu]->logical_index);
    }

    core_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_PU, static_cast<unsigned>(num_core));

    return static_cast<std::size_t>(core_obj->logical_index);
}

auto Topology::get_machine_affinity_mask() const -> MaskCRefType {
    return _machine_affinity_mask;
}

auto Topology::get_socket_affinity_mask(std::size_t num_thread) const -> MaskCRefType {
    if (std::size_t const num_pu = num_thread % _num_of_pus; num_pu < _socket_affinity_masks.size()) {
        return _socket_affinity_masks[num_pu];
    }

    EINSUMS_THROW_EXCEPTION(bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

auto Topology::get_numa_node_affinity_mask(std::size_t num_thread) const -> MaskCRefType {
    if (std::size_t const num_pu = num_thread % _num_of_pus; num_pu < _numa_node_affinity_masks.size()) {
        return _numa_node_affinity_masks[num_pu];
    }

    EINSUMS_THROW_EXCEPTION(bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

auto Topology::get_core_affinity_mask(std::size_t num_thread) const -> MaskCRefType {
    if (std::size_t const num_pu = num_thread % _num_of_pus; num_pu < _core_affinity_masks.size()) {
        return _core_affinity_masks[num_pu];
    }

    EINSUMS_THROW_EXCEPTION(bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

auto Topology::get_thread_affinity_mask(std::size_t num_thread) const -> MaskCRefType {

    if (std::size_t const num_pu = num_thread % _num_of_pus; num_pu < _thread_affinity_masks.size()) {
        return _thread_affinity_masks[num_pu];
    }

    EINSUMS_THROW_EXCEPTION(bad_parameter, "thread number {} is out of range", num_thread);
    return empty_mask;
}

///////////////////////////////////////////////////////////////////////////
void Topology::set_thread_affinity_mask(MaskCRefType mask) {

#if !defined(__APPLE__)
    // setting thread affinities is not supported by OSX
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    int const pu_depth = hwloc_get_type_or_below_depth(_topology, HWLOC_OBJ_PU);

    for (std::size_t i = 0; i != mask_size(mask); ++i) {
        if (test(mask, i)) {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, unsigned(i));
            EINSUMS_ASSERT(i == detail::get_index(pu_obj));
            hwloc_bitmap_set(cpuset, static_cast<unsigned int>(pu_obj->os_index));
        }
    }

    {
        std::unique_lock lk(_topology_mutex);
        if (hwloc_set_cpubind(_topology, cpuset, HWLOC_CPUBIND_STRICT | HWLOC_CPUBIND_THREAD)) {
            // Strict binding not supported or failed, try weak binding.
            if (hwloc_set_cpubind(_topology, cpuset, HWLOC_CPUBIND_THREAD)) {
                std::unique_ptr<char[]> buffer(new char[1024]);

                hwloc_bitmap_snprintf(buffer.get(), 1024, cpuset);
                hwloc_bitmap_free(cpuset);

                EINSUMS_THROW_EXCEPTION(system_error, "failed to set thread affinity mask ({}) for cpuset {}", to_string(mask),
                                        buffer.get());
                return;
            }
        }
    }
#    if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    sleep(0); // Allow the OS to pick up the change.
#    endif
    hwloc_bitmap_free(cpuset);
#else
    PIKA_UNUSED(mask);
#endif // __APPLE__
}

///////////////////////////////////////////////////////////////////////////
auto Topology::get_thread_affinity_mask_from_lva(void const *lva) -> MaskType {
    hwloc_membind_policy_t policy  = ::HWLOC_MEMBIND_DEFAULT;
    hwloc_nodeset_t        nodeset = hwloc_bitmap_alloc();

    {
        std::unique_lock lk(_topology_mutex);
        int              ret =
#if HWLOC_API_VERSION >= 0x0001'0b06
            hwloc_get_area_membind(_topology, lva, 1, nodeset, &policy, HWLOC_MEMBIND_BYNODESET);
#else
            hwloc_get_area_membind_nodeset(_topology, lva, 1, nodeset, &policy, 0);
#endif

        if (-1 != ret) {
            hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
            hwloc_cpuset_from_nodeset(_topology, cpuset, nodeset);
            lk.unlock();

            hwloc_bitmap_free(nodeset);

            MaskType mask = MaskType();
            resize(mask, get_number_of_pus());

            int const pu_depth = hwloc_get_type_or_below_depth(_topology, HWLOC_OBJ_PU);
            for (unsigned int i = 0; std::size_t(i) != _num_of_pus; ++i) {
                hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, i);
                unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
                if (hwloc_bitmap_isset(cpuset, idx) != 0)
                    set(mask, detail::get_index(pu_obj));
            }

            hwloc_bitmap_free(cpuset);
            return mask;
        } else {
            std::string errstr = std::strerror(errno);

            lk.unlock();
            EINSUMS_THROW_EXCEPTION(no_success, "failed calling 'hwloc_get_area_membind_nodeset', reported error: {}", errstr);
        }
    }

    hwloc_bitmap_free(nodeset);
    return empty_mask;
}

auto Topology::init_numa_node_number(std::size_t num_thread) -> std::size_t {
#if HWLOC_API_VERSION >= 0x0002'0000
    if (std::size_t(-1) == num_thread)
        return std::size_t(-1);

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    hwloc_obj_t obj;
    {
        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
        EINSUMS_ASSERT(num_pu == detail::get_index(obj));
    }

    hwloc_obj_t tmp = nullptr;
    while ((tmp = hwloc_get_next_obj_by_type(_topology, HWLOC_OBJ_NUMANODE, tmp)) != nullptr) {
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

auto Topology::init_node_number(std::size_t num_thread, hwloc_obj_type_t type) -> std::size_t {
    if (std::size_t(-1) == num_thread)
        return std::size_t(-1);

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    {
        hwloc_obj_t obj;

        {
            std::unique_lock lk(_topology_mutex);
            obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
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
}

void Topology::extract_node_mask(hwloc_obj_t parent, MaskType &mask) {
    hwloc_obj_t obj;

    {
        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_next_child(_topology, parent, nullptr);
    }

    while (obj) {
        if (hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0) {
            do {
                set(mask, detail::get_index(obj)); //-V106
                {
                    std::unique_lock lk(_topology_mutex);
                    obj = hwloc_get_next_child(_topology, parent, obj);
                }
            } while (obj != nullptr && hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0);
            return;
        }

        extract_node_mask(obj, mask);

        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_next_child(_topology, parent, obj);
    }
}

auto Topology::extract_node_count(hwloc_obj_t parent, hwloc_obj_type_t type, std::size_t count) -> std::size_t {
    hwloc_obj_t obj;

    if (parent == nullptr) {
        return count;
    }

    if (hwloc_compare_types(type, parent->type) == 0) {
        return count;
    }

    {
        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_next_child(_topology, parent, nullptr);
    }

    while (obj) {
        if (hwloc_compare_types(type, obj->type) == 0) {
            /*
            do {
                ++count;
                {
                    std::unique_lock lk(_topology_mutex);
                    obj = hwloc_get_next_child(_topology, parent, obj);
                }
            } while (obj != nullptr && hwloc_compare_types(type, obj->type) == 0);
            return count;
            */
            ++count;
        }

        count = extract_node_count(obj, type, count);

        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_next_child(_topology, parent, obj);
    }

    return count;
}

auto Topology::get_number_of_sockets() const -> std::size_t {
    int const nobjs = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_SOCKET);
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type failed");
        return std::size_t(nobjs);
    }
    return std::size_t(nobjs);
}

auto Topology::get_number_of_numa_nodes() const -> std::size_t {
    int const nobjs = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_NUMANODE);
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type failed");
        return std::size_t(nobjs);
    }
    return std::size_t(nobjs);
}

std::size_t Topology::get_number_of_cores() const {
    int nobjs = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_CORE);

    // If num_cores is smaller 0, we have an error
    if (0 > nobjs) {
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type(HWLOC_OBJ_CORE) failed");
        return std::size_t(nobjs);
    } else if (0 == nobjs) {
        // some platforms report zero cores but might still report the
        // number of PUs
        nobjs = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_PU);
        if (0 > nobjs) {
            EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type(HWLOC_OBJ_PU) failed");
            return std::size_t(nobjs);
        }
    }

    // the number of reported cores/pus should never be zero either to
    // avoid division by zero, we should always have at least one core
    if (0 == nobjs) {
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type reports zero cores/pus");
        return std::size_t(nobjs);
    }

    return std::size_t(nobjs);
}

std::size_t Topology::get_number_of_socket_pus(std::size_t num_socket) {
    hwloc_obj_t socket_obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        socket_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));
        std::size_t pu_count = 0;
        return extract_node_count(socket_obj, HWLOC_OBJ_PU, pu_count);
    }

    return _num_of_pus;
}

std::size_t Topology::get_number_of_numa_node_pus(std::size_t numa_node) {
    hwloc_obj_t node_obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        node_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(node_obj));
        std::size_t pu_count = 0;
        node_obj             = detail::adjust_node_obj(node_obj);
        return extract_node_count(node_obj, HWLOC_OBJ_PU, pu_count);
    }

    return _num_of_pus;
}

std::size_t Topology::get_number_of_core_pus(std::size_t core) {
    hwloc_obj_t core_obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        core_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_CORE, static_cast<unsigned>(core));
    }

    if (!_use_pus_as_cores && core_obj) {
        EINSUMS_ASSERT(core == detail::get_index(core_obj));
        std::size_t pu_count = 0;
        return extract_node_count(core_obj, HWLOC_OBJ_PU, pu_count);
    }

    return std::size_t(1);
}

std::size_t Topology::get_number_of_socket_cores(std::size_t num_socket) {
    hwloc_obj_t socket_obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        socket_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));
        std::size_t pu_count = 0;
        return extract_node_count(socket_obj, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, pu_count);
    }

    return get_number_of_cores();
}

std::size_t Topology::get_number_of_numa_node_cores(std::size_t numa_node) {
    hwloc_obj_t node_obj = nullptr;
    {
        std::unique_lock lk(_topology_mutex);
        node_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(node_obj));
        std::size_t pu_count = 0;
        node_obj             = detail::adjust_node_obj(node_obj);
        return extract_node_count(node_obj, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, pu_count);
    }

    return get_number_of_cores();
}

HwlocBitmapPtr Topology::cpuset_to_nodeset(MaskCRefType mask) {
    hwloc_bitmap_t cpuset  = mask_to_bitmap(mask, HWLOC_OBJ_PU);
    hwloc_bitmap_t nodeset = hwloc_bitmap_alloc();
#if HWLOC_API_VERSION >= 0x0002'0000
    hwloc_cpuset_to_nodeset(_topology, cpuset, nodeset);
#else
    hwloc_cpuset_to_nodeset_strict(_topology, cpuset, nodeset);
#endif
    hwloc_bitmap_free(cpuset);
    return std::make_shared<HwlocBitmapWrapper>(nodeset);
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

void Topology::print_affinity_mask(std::ostream &os, std::size_t num_thread, MaskCRefType m, std::string const &pool_name) const {
    bool first = true;

    if (!any(m)) {
        os << std::setw(4) << num_thread << ": thread binding disabled" << ", on pool \"" << pool_name << "\"" << std::endl;
        return;
    }

    for (std::size_t i = 0; i != _num_of_pus; ++i) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_PU, unsigned(i));
        if (!obj) {
            EINSUMS_THROW_EXCEPTION(system_error, "object not found");
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

MaskType Topology::init_machine_affinity_mask() {
    MaskType machine_affinity_mask = MaskType();
    resize(machine_affinity_mask, get_number_of_pus());

    hwloc_obj_t machine_obj;
    {
        std::unique_lock lk(_topology_mutex);
        machine_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_MACHINE, 0);
    }
    if (machine_obj) {
        extract_node_mask(machine_obj, machine_affinity_mask);
        return machine_affinity_mask;
    }

    EINSUMS_THROW_EXCEPTION(system_error, "failed to initialize machine affinity mask");
    return empty_mask;
}

MaskType Topology::init_socket_affinity_mask_from_socket(std::size_t num_socket) {
    // If we have only one or no socket, the socket affinity mask
    // spans all processors
    if (std::size_t(-1) == num_socket)
        return _machine_affinity_mask;

    hwloc_obj_t socket_obj = nullptr;
    {
        std::unique_lock lk(_topology_mutex);
        socket_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
    }

    if (socket_obj) {
        EINSUMS_ASSERT(num_socket == detail::get_index(socket_obj));

        MaskType socket_affinity_mask = MaskType();
        resize(socket_affinity_mask, get_number_of_pus());

        extract_node_mask(socket_obj, socket_affinity_mask);
        return socket_affinity_mask;
    }

    return _machine_affinity_mask;
}

MaskType Topology::init_numa_node_affinity_mask_from_numa_node(std::size_t numa_node) {
    // If we have only one or no NUMA domain, the NUMA affinity mask
    // spans all processors
    if (std::size_t(-1) == numa_node) {
        return _machine_affinity_mask;
    }

    hwloc_obj_t numa_node_obj = nullptr;
    {
        std::unique_lock lk(_topology_mutex);
        numa_node_obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
    }

    if (numa_node_obj) {
        EINSUMS_ASSERT(numa_node == detail::get_index(numa_node_obj));
        MaskType node_affinity_mask = MaskType();
        resize(node_affinity_mask, get_number_of_pus());

        numa_node_obj = detail::adjust_node_obj(numa_node_obj);
        extract_node_mask(numa_node_obj, node_affinity_mask);
        return node_affinity_mask;
    }

    return _machine_affinity_mask;
}

MaskType Topology::init_core_affinity_mask_from_core(std::size_t core, MaskCRefType default_mask) {
    if (std::size_t(-1) == core) {
        return default_mask;
    }

    hwloc_obj_t core_obj = nullptr;

    std::size_t num_core = (core + core_offset) % get_number_of_cores();

    {
        std::unique_lock lk(_topology_mutex);
        core_obj = hwloc_get_obj_by_type(_topology, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));
    }

    if (core_obj) {
        EINSUMS_ASSERT(num_core == detail::get_index(core_obj));
        MaskType core_affinity_mask = MaskType();
        resize(core_affinity_mask, get_number_of_pus());

        extract_node_mask(core_obj, core_affinity_mask);
        return core_affinity_mask;
    }

    return default_mask;
}

MaskType Topology::init_thread_affinity_mask(std::size_t num_thread) {

    if (std::size_t(-1) == num_thread) {
        return get_core_affinity_mask(num_thread);
    }

    std::size_t num_pu = (num_thread + pu_offset) % _num_of_pus;

    hwloc_obj_t obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        obj = hwloc_get_obj_by_type(_topology, HWLOC_OBJ_PU, static_cast<unsigned>(num_pu));
    }

    if (!obj) {
        return get_core_affinity_mask(num_thread);
    }

    EINSUMS_ASSERT(num_pu == detail::get_index(obj));
    MaskType mask = MaskType();
    resize(mask, get_number_of_pus());

    set(mask, detail::get_index(obj)); //-V106

    return mask;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
MaskType Topology::init_thread_affinity_mask(std::size_t num_core, std::size_t num_pu)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    hwloc_obj_t obj = nullptr;

    {
        std::unique_lock lk(_topology_mutex);
        int              num_cores = hwloc_get_nbobjs_by_type(_topology, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);

        // If num_cores is smaller 0, we have an error, it should never be zero
        // either to avoid division by zero, we should always have at least one
        // core
        if (num_cores <= 0) {
            EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_nbobjs_by_type failed");
            return empty_mask;
        }

        num_core = (num_core + core_offset) % std::size_t(num_cores);
        obj      = hwloc_get_obj_by_type(_topology, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));
    }

    if (!obj)
        return empty_mask; // get_core_affinity_mask(num_thread, false);

    EINSUMS_ASSERT(num_core == detail::get_index(obj));

    MaskType mask = MaskType();
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
void Topology::init_num_of_pus() {
    _num_of_pus       = 1;
    _use_pus_as_cores = false;

    {
        std::unique_lock lk(_topology_mutex);

        // on some platforms, hwloc can't report the number of cores (BSD),
        // in this case we use PUs as cores
        if (hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_CORE) <= 0) {
            _use_pus_as_cores = true;
        }

        int num_of_pus = hwloc_get_nbobjs_by_type(_topology, HWLOC_OBJ_PU);
        if (num_of_pus > 0) {
            _num_of_pus = static_cast<std::size_t>(num_of_pus);
        }
    }
}

std::size_t Topology::get_number_of_pus() const {
    return _num_of_pus;
}

///////////////////////////////////////////////////////////////////////////
MaskType Topology::get_cpubind_mask_main_thread() const {
    return _main_thread_affinity_mask;
}

void Topology::set_cpubind_mask_main_thread(MaskType mask) {
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
        EINSUMS_THROW_EXCEPTION(bad_parameter, "CPU mask ({}) has bits set past the hardware concurrency of the system ({})",
                                fmt::streamed(mask), concurrency);
    }

    if (!any(mask)) {
        EINSUMS_THROW_EXCEPTION(bad_parameter,
                                "CPU mask is empty ({}), make sure it has at least one bit set through "
                                "EINSUMS_PROCESS_MASK or --einsums:process-mask",
                                fmt::streamed(mask));
    }

    // The mask is assumed to use physical/OS indices (as returned by e.g. hwloc-bind --get
    // --taskset or taskset --pid) while einsums deals with logical indices from this point
    // onwards. We convert the mask from physical indices to logical indices before storing it.
    MaskType logical_mask{};
    resize(logical_mask, get_number_of_pus());

#if !defined(__APPLE__)
    {
        std::unique_lock lk(_topology_mutex);

        int const pu_depth = hwloc_get_type_or_below_depth(_topology, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != get_number_of_pus(); ++i) {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            EINSUMS_ASSERT(i == detail::get_index(pu_obj));
            EINSUMS_ASSERT(idx < mask_size(mask));
            EINSUMS_ASSERT(detail::get_index(pu_obj) < mask_size(logical_mask));
            if (test(mask, idx)) {
                set(logical_mask, detail::get_index(pu_obj));
            }
        }
    }
#endif // __APPLE__

    _main_thread_affinity_mask = std::move(logical_mask);
}

MaskType Topology::get_cpubind_mask() {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    MaskType mask = MaskType();
    resize(mask, get_number_of_pus());

#if !defined(__APPLE__)
    {
        std::unique_lock lk(_topology_mutex);
        if (hwloc_get_cpubind(_topology, cpuset, HWLOC_CPUBIND_THREAD)) {
            hwloc_bitmap_free(cpuset);
            EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_cpubind failed");
            return empty_mask;
        }

        int const pu_depth = hwloc_get_type_or_below_depth(_topology, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != _num_of_pus; ++i) //-V104
        {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            if (hwloc_bitmap_isset(cpuset, idx) != 0)
                set(mask, detail::get_index(pu_obj));
        }
    }
#endif // __APPLE__

    hwloc_bitmap_free(cpuset);

    return mask;
}

MaskType Topology::get_cpubind_mask(std::thread &handle) {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    MaskType mask = MaskType();
    resize(mask, get_number_of_pus());

    {
        std::unique_lock lk(_topology_mutex);
#if defined(PIKA_MINGW)
        if (hwloc_get_thread_cpubind(_topology, pthread_gethandle(handle.native_handle()), cpuset, HWLOC_CPUBIND_THREAD))
#else
        if (hwloc_get_thread_cpubind(_topology, handle.native_handle(), cpuset, HWLOC_CPUBIND_THREAD))
#endif
        {
            hwloc_bitmap_free(cpuset);
            EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_cpubind failed");
            return empty_mask;
        }

        int const pu_depth = hwloc_get_type_or_below_depth(_topology, HWLOC_OBJ_PU);
        for (unsigned int i = 0; i != _num_of_pus; ++i) //-V104
        {
            hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, i);
            unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
            if (hwloc_bitmap_isset(cpuset, idx) != 0)
                set(mask, detail::get_index(pu_obj));
        }
    }

    hwloc_bitmap_free(cpuset);

    return mask;
}

///////////////////////////////////////////////////////////////////////////
/// This is equivalent to malloc(), except that it tries to allocate
/// page-aligned memory from the OS.
void *Topology::allocate(std::size_t len) const {
    return hwloc_alloc(_topology, len);
}

///////////////////////////////////////////////////////////////////////////
/// Allocate some memory on NUMA memory nodes specified by nodeset
/// as specified by the hwloc hwloc_alloc_membind_nodeset call
void *Topology::allocate_membind(std::size_t len, HwlocBitmapPtr bitmap, HwlocMembindPolicy policy, int flags) const {
    return
#if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_alloc_membind(_topology, len, bitmap->bitmap(), (hwloc_membind_policy_t)(policy), flags | HWLOC_MEMBIND_BYNODESET);
#else
        hwloc_alloc_membind_nodeset(_topology, len, bitmap->get_bmp(), (hwloc_membind_policy_t)(policy), flags);
#endif
}

bool Topology::set_area_membind_nodeset(void const *addr, std::size_t len, void *nodeset) const {
#if !defined(__APPLE__)
    hwloc_membind_policy_t policy = ::HWLOC_MEMBIND_BIND;
    hwloc_nodeset_t        ns     = reinterpret_cast<hwloc_nodeset_t>(nodeset);

    int ret =
#    if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_set_area_membind(_topology, addr, len, ns, policy, HWLOC_MEMBIND_BYNODESET);
#    else
        hwloc_set_area_membind_nodeset(_topology, addr, len, ns, policy, 0);
#    endif

    if (ret < 0) {
        std::string msg = std::strerror(errno);
        if (errno == ENOSYS)
            msg = "the action is not supported";
        if (errno == EXDEV)
            msg = "the binding cannot be enforced";
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_set_area_membind_nodeset failed : {}", msg);
        return false;
    }
#else
    PIKA_UNUSED(addr);
    PIKA_UNUSED(len);
    PIKA_UNUSED(nodeset);
#endif
    return true;
}

namespace {
HwlocBitmapWrapper &bitmap_storage() {
    static thread_local HwlocBitmapWrapper bitmap_storage_(nullptr);

    return bitmap_storage_;
}
} // namespace

MaskType Topology::get_area_membind_nodeset(void const *addr, std::size_t len) {
    HwlocBitmapWrapper &nodeset = bitmap_storage();
    if (!nodeset) {
        nodeset.reset(hwloc_bitmap_alloc());
    }

    //
    hwloc_membind_policy_t policy;
    hwloc_nodeset_t        ns = reinterpret_cast<hwloc_nodeset_t>(nodeset.bitmap());

    if (
#if HWLOC_API_VERSION >= 0x0001'0b06
        hwloc_get_area_membind(_topology, addr, len, ns, &policy, HWLOC_MEMBIND_BYNODESET)
#else
        hwloc_get_area_membind_nodeset(_topology, addr, len, ns, &policy, 0)
#endif
        == -1) {
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_area_membind_nodeset failed");
        return bitmap_to_mask(ns, HWLOC_OBJ_MACHINE);
    }

    return bitmap_to_mask(ns, HWLOC_OBJ_NUMANODE);
}

int Topology::get_numa_domain(void const *addr) {
#if HWLOC_API_VERSION >= 0x0001'0b06
    HwlocBitmapWrapper &nodeset = bitmap_storage();
    if (!nodeset) {
        nodeset.reset(hwloc_bitmap_alloc());
    }

    //
    hwloc_nodeset_t ns = reinterpret_cast<hwloc_nodeset_t>(nodeset.bitmap());

    int ret = hwloc_get_area_memlocation(_topology, addr, 1, ns, HWLOC_MEMBIND_BYNODESET);
    if (ret < 0) {
#    if defined(__FreeBSD__)
        // on some platforms this API is not supported (e.g. FreeBSD)
        return 0;
#    else
        std::string msg(strerror(errno));
        EINSUMS_THROW_EXCEPTION(system_error, "hwloc_get_area_memlocation failed {}", msg);
        return -1;
#    endif
    }

    MaskType mask = bitmap_to_mask(ns, HWLOC_OBJ_NUMANODE);
    return static_cast<int>(detail::find_first(mask));
#else
    EINSUMS_UNUSED(addr);
    return 0;
#endif
}

/// Free memory that was previously allocated by allocate
void Topology::deallocate(void *addr, std::size_t len) const {
    hwloc_free(_topology, addr, len);
}

///////////////////////////////////////////////////////////////////////////
hwloc_bitmap_t Topology::mask_to_bitmap(MaskCRefType mask, hwloc_obj_type_t htype) const {
    hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
    hwloc_bitmap_zero(bitmap);
    //
    int const depth = hwloc_get_type_or_below_depth(_topology, htype);

    for (std::size_t i = 0; i != mask_size(mask); ++i) {
        if (test(mask, i)) {
            hwloc_obj_t const hw_obj = hwloc_get_obj_by_depth(_topology, depth, unsigned(i));
            EINSUMS_ASSERT(i == detail::get_index(hw_obj));
            hwloc_bitmap_set(bitmap, static_cast<unsigned int>(hw_obj->os_index));
        }
    }
    return bitmap;
}

///////////////////////////////////////////////////////////////////////////
MaskType Topology::bitmap_to_mask(hwloc_bitmap_t bitmap, hwloc_obj_type_t htype) {
    MaskType mask = MaskType();
    resize(mask, get_number_of_pus());
    std::size_t num = hwloc_get_nbobjs_by_type(_topology, htype);
    //
    int const pu_depth = hwloc_get_type_or_below_depth(_topology, htype);
    for (unsigned int i = 0; std::size_t(i) != num; ++i) //-V104
    {
        hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(_topology, pu_depth, i);
        unsigned          idx    = static_cast<unsigned>(pu_obj->os_index);
        if (hwloc_bitmap_isset(bitmap, idx) != 0)
            set(mask, detail::get_index(pu_obj));
    }
    return mask;
}

///////////////////////////////////////////////////////////////////////////
void Topology::print_mask_vector(std::ostream &os, std::vector<MaskType> const &v) {
    std::size_t s = v.size();
    if (s == 0) {
        os << "(empty)\n";
        return;
    }

    for (std::size_t i = 0; i != s; i++) {
        os << detail::to_string(v[i]) << "\n";
    }
    os << "\n";
}

void Topology::print_vector(std::ostream &os, std::vector<std::size_t> const &v) {
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

void Topology::print_hwloc(std::ostream &os) {
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
       << detail::to_string(_machine_affinity_mask) << "\n";

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
    hw_concurrency() noexcept
#if defined(__ANDROID__) && defined(ANDROID)
        : num_of_cores_(::android_getCpuCount())
#else
        : num_of_cores_(Topology::get_singleton().get_number_of_pus())
#endif
    {
        if (num_of_cores_ == 0)
            num_of_cores_ = 1;
    }

    std::size_t num_of_cores_;
};

unsigned int hardware_concurrency() noexcept {
    static detail::hw_concurrency hwc;
    return static_cast<unsigned int>(hwc.num_of_cores_);
}
} // namespace einsums::topology::detail
