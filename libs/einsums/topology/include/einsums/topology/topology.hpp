//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/modules/errors.hpp>
#include <einsums/topology/cpu_mask.hpp>

#include <cstddef>
#include <hwloc.h>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace einsums::threads::detail {

struct hwloc_bitmap_wrapper {
    EINSUMS_NON_COPYABLE(hwloc_bitmap_wrapper);

    hwloc_bitmap_wrapper() : _bmp(nullptr) {}

    hwloc_bitmap_wrapper(void *bmp) : _bmp(reinterpret_cast<hwloc_bitmap_t>(bmp)) {}

    ~hwloc_bitmap_wrapper() { hwloc_bitmap_free(_bmp); }

    void reset(hwloc_bitmap_t bmp) {
        if (_bmp)
            hwloc_bitmap_free(_bmp);
        _bmp = bmp;
    }

    explicit operator bool() const noexcept { return _bmp != nullptr; }

    hwloc_bitmap_t get_bmp() const noexcept { return _bmp; }

    // stringify the bitmp using hwloc
    friend EINSUMS_EXPORT std::ostream &operator<<(std::ostream &os, hwloc_bitmap_wrapper const *bmp);

  private:
    hwloc_bitmap_t _bmp{nullptr};
};

using hwloc_bitmap_ptr = std::shared_ptr<hwloc_bitmap_wrapper>;

enum hwloc_membind_policy : int {
    membind_default    = HWLOC_MEMBIND_DEFAULT,
    membind_firsttouch = HWLOC_MEMBIND_FIRSTTOUCH,
    membind_bind       = HWLOC_MEMBIND_BIND,
    membind_interleave = HWLOC_MEMBIND_INTERLEAVE,
#if HWLOC_API_VERSION < 0x0002'0000
    membind_replicate = HWLOC_MEMBIND_REPLICATE,
#endif
    membind_nexttouch = HWLOC_MEMBIND_NEXTTOUCH,
    membind_mixed     = HWLOC_MEMBIND_MIXED,
    // special einsums addition
    membind_user = HWLOC_MEMBIND_MIXED + 256
};

struct EINSUMS_EXPORT topology {
    topology();
    ~topology();

    /// \brief Return the Socket number of the processing unit the
    ///        given thread is running on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    std::size_t get_socket_number(std::size_t num_thread, error_code & /*ec*/ = throws) const {
        return _socket_numbers[num_thread % _num_of_pus];
    }

    /// \brief Return the NUMA node number of the processing unit the
    ///        given thread is running on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    std::size_t get_numa_node_number(std::size_t num_thread, error_code & /*ec*/ = throws) const {
        return _numa_node_numbers[num_thread % _num_of_pus];
    }

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit available to the application.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_cref_type get_machine_affinity_mask(error_code &ec = throws) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit available to the given thread inside
    ///        the socket it is running on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_cref_type get_socket_affinity_mask(std::size_t num_thread, error_code &ec = throws) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit available to the given thread inside
    ///        the NUMA domain it is running on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_cref_type get_numa_node_affinity_mask(std::size_t num_thread, error_code &ec = throws) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit associated with the given NUMA node.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_type get_numa_node_affinity_mask_from_numa_node(std::size_t num_node) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit available to the given thread inside
    ///        the core it is running on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_cref_type get_core_affinity_mask(std::size_t num_thread, error_code &ec = throws) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit available to the given thread.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_cref_type get_thread_affinity_mask(std::size_t num_thread, error_code &ec = throws) const;

    /// \brief Use the given bit mask to set the affinity of the given
    ///        thread. Each set bit corresponds to a processing unit the
    ///        thread will be allowed to run on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    ///
    /// \note  Use this function on systems where the affinity must be
    ///        set from inside the thread itself.
    void set_thread_affinity_mask(mask_cref_type mask, error_code &ec = throws) const;

    /// \brief Return a bit mask where each set bit corresponds to a
    ///        processing unit co-located with the memory the given
    ///        address is currently allocated on.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    mask_type get_thread_affinity_mask_from_lva(void const *lva, error_code &ec = throws) const;

    /// \brief Prints the \param m to os in a human readable form
    void print_affinity_mask(std::ostream &os, std::size_t num_thread, mask_cref_type m, const std::string &pool_name) const;

    /// \brief Reduce thread priority of the current thread.
    ///
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    bool reduce_thread_priority(error_code &ec = throws) const;

    /// \brief Return the number of available NUMA domains
    std::size_t get_number_of_sockets() const;

    /// \brief Return the number of available NUMA domains
    std::size_t get_number_of_numa_nodes() const;

    /// \brief Return the number of available cores
    std::size_t get_number_of_cores() const;

    /// \brief Return the number of available hardware processing units
    std::size_t get_number_of_pus() const;

    /// \brief Return number of cores in given numa domain
    std::size_t get_number_of_numa_node_cores(std::size_t numa) const;

    /// \brief Return number of processing units in a given numa domain
    std::size_t get_number_of_numa_node_pus(std::size_t numa) const;

    /// \brief Return number of processing units in a given socket
    std::size_t get_number_of_socket_pus(std::size_t socket) const;

    /// \brief Return number of processing units in given core
    std::size_t get_number_of_core_pus(std::size_t core) const;

    /// \brief Return number of cores units in given socket
    std::size_t get_number_of_socket_cores(std::size_t socket) const;

    std::size_t get_core_number(std::size_t num_thread, error_code & /*ec*/ = throws) const {
        return _core_numbers[num_thread % _num_of_pus];
    }

    std::size_t get_pu_number(std::size_t num_core, std::size_t num_pu, error_code &ec = throws) const;

    mask_type get_cpubind_mask_main_thread(error_code &ec = throws) const;
    void      set_cpubind_mask_main_thread(mask_type, error_code &ec = throws);
    mask_type get_cpubind_mask(error_code &ec = throws) const;
    mask_type get_cpubind_mask(std::thread &handle, error_code &ec = throws) const;

    /// convert a cpu mask into a numa node mask in hwloc bitmap form
    hwloc_bitmap_ptr cpuset_to_nodeset(mask_cref_type cpuset) const;

    void write_to_log() const;

    /// This is equivalent to malloc(), except that it tries to allocate
    /// page-aligned memory from the OS.
    void *allocate(std::size_t len) const;

    /// allocate memory with binding to a numa node set as
    /// specified by the policy and flags (see hwloc docs)
    void *allocate_membind(std::size_t len, hwloc_bitmap_ptr bitmap, hwloc_membind_policy policy, int flags) const;

    threads::detail::mask_type get_area_membind_nodeset(const void *addr, std::size_t len) const;

    bool set_area_membind_nodeset(const void *addr, std::size_t len, void *nodeset) const;

    int get_numa_domain(const void *addr) const;

    /// Free memory that was previously allocated by allocate
    void deallocate(void *addr, std::size_t len) const;

    void print_vector(std::ostream &os, std::vector<std::size_t> const &v) const;
    void print_mask_vector(std::ostream &os, std::vector<mask_type> const &v) const;
    void print_hwloc(std::ostream &) const;

    mask_type init_socket_affinity_mask_from_socket(std::size_t num_socket) const;
    mask_type init_numa_node_affinity_mask_from_numa_node(std::size_t num_numa_node) const;
    mask_type init_core_affinity_mask_from_core(std::size_t num_core, mask_cref_type default_mask = empty_mask) const;
    mask_type init_thread_affinity_mask(std::size_t num_thread) const;
    mask_type init_thread_affinity_mask(std::size_t num_core, std::size_t num_pu) const;

    hwloc_bitmap_t mask_to_bitmap(mask_cref_type mask, hwloc_obj_type_t htype) const;
    mask_type      bitmap_to_mask(hwloc_bitmap_t bitmap, hwloc_obj_type_t htype) const;

  private:
    std::size_t init_node_number(std::size_t num_thread, hwloc_obj_type_t type);

    std::size_t init_socket_number(std::size_t num_thread) { return init_node_number(num_thread, HWLOC_OBJ_SOCKET); }

    std::size_t init_numa_node_number(std::size_t num_thread);

    std::size_t init_core_number(std::size_t num_thread) {
        return init_node_number(num_thread, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);
    }

    void extract_node_mask(hwloc_obj_t parent, mask_type &mask) const;

    std::size_t extract_node_count(hwloc_obj_t parent, hwloc_obj_type_t type, std::size_t count) const;

    mask_type init_machine_affinity_mask() const;
    mask_type init_socket_affinity_mask(std::size_t num_thread) const {
        return init_socket_affinity_mask_from_socket(get_socket_number(num_thread));
    }

    mask_type init_numa_node_affinity_mask(std::size_t num_thread) const {
        return init_numa_node_affinity_mask_from_numa_node(get_numa_node_number(num_thread));
    }

    mask_type init_core_affinity_mask(std::size_t num_thread) const {
        mask_type default_mask = _numa_node_affinity_masks[num_thread];
        return init_core_affinity_mask_from_core(get_core_number(num_thread), default_mask);
    }

    void init_num_of_pus();

    hwloc_topology_t topo;

    static mask_type   empty_mask;
    static std::size_t _memory_page_size;
    friend std::size_t get_memory_page_size();

    static constexpr std::size_t pu_offset   = 0;
    static constexpr std::size_t core_offset = 0;

    std::size_t _num_of_pus;
    bool        _use_pus_as_cores;

    using mutex_type = std::mutex;
    mutable mutex_type topo_mutex;

    // Number masks:
    // Vectors of non-negative integers indicated which architecture objects each PU belongs to. For example, numa_node_numbers[0] indicates
    // which numa node number PU #0 (zero-based index) belong to
    std::vector<std::size_t> _socket_numbers;
    std::vector<std::size_t> _numa_node_numbers;
    std::vector<std::size_t> _core_numbers;

    // Affinity masks: vectors of bitmasks
    // - Length of the vector: number of PUs of the machine
    // - Elements of the vector:
    // Bitmasks of length equal to the number of PUs of the machine.
    // The bitmasks indicate which PUs belong to which resource.
    // For example, core_affinity_masks[0] is a bitmask, where the
    // elements = 1 indicate the PUs that belong to the core on which
    // PU #0 (zero-based index) lies.
    mask_type              _machine_affinity_mask;
    std::vector<mask_type> _socket_affinity_masks;
    std::vector<mask_type> _numa_node_affinity_masks;
    std::vector<mask_type> _core_affinity_masks;
    std::vector<mask_type> _thread_affinity_masks;
    mask_type              _main_thread_affinity_mask;
};

///////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT topology &get_topology();

[[nodiscard]] EINSUMS_EXPORT unsigned int hardware_concurrency() noexcept;

///////////////////////////////////////////////////////////////////////////
// abstract away memory page size, calls to system functions are
// expensive, so return a value initialized at startup
inline std::size_t get_memory_page_size() {
    return einsums::threads::detail::topology::_memory_page_size;
}

} // namespace einsums::threads::detail