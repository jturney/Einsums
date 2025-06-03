//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Hardware/CPUMask.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <hwloc.h>

namespace einsums::hardware {

struct HwlocBitmapWrapper {
    EINSUMS_NON_COPYABLE(HwlocBitmapWrapper);

    HwlocBitmapWrapper() : _bitmap(nullptr) {}

    HwlocBitmapWrapper(void *bitmap) : _bitmap(static_cast<hwloc_bitmap_t>(bitmap)) {}

    ~HwlocBitmapWrapper() { hwloc_bitmap_free(_bitmap); }

    void reset(hwloc_bitmap_t bitmap) {
        if (_bitmap)
            hwloc_bitmap_free(_bitmap);
        _bitmap = bitmap;
    }

    explicit operator bool() const noexcept { return _bitmap != nullptr; }

    hwloc_bitmap_t bitmap() const noexcept { return _bitmap; }

    friend EINSUMS_EXPORT std::ostream &operator<<(std::ostream &os, HwlocBitmapWrapper const *bitmap);

  private:
    hwloc_bitmap_t _bitmap;
};

using HwlocBitmapPtr = std::shared_ptr<HwlocBitmapWrapper>;

/// \brief See hwloc documentation for the corresponding enum HWLOC_MEMBIND_XXX
enum HwlocMembindPolicy : int {
    MembindDefault    = HWLOC_MEMBIND_DEFAULT,
    MembindFirstTouch = HWLOC_MEMBIND_FIRSTTOUCH,
    MembindBind       = HWLOC_MEMBIND_BIND,
    MembindInterleave = HWLOC_MEMBIND_INTERLEAVE,
#if HWLOC_API_VERSION < 0x0002'0000
    MembindReplicate = HWLOC_MEMBIND_REPLICATE,
#endif
    MembindNextTouch = HWLOC_MEMBIND_NEXTTOUCH,
    MembindMixed     = HWLOC_MEMBIND_MIXED,
    MembindUser      = HWLOC_MEMBIND_MIXED + 256
};

struct EINSUMS_EXPORT Topology {

    EINSUMS_SINGLETON_DEF(Topology);

  private:
    Topology(); // needed by EINSUMS_SINGLETON_DEF

  public:
    ~Topology();

    /// Return the socket number of the processing unit the given thread is running on.
    std::size_t get_socket_number(std::size_t num_thread) const { return _socket_numbers[num_thread % _num_of_pus]; }

    /// Return the NUMA node number of the processing unit the given thread is running on.
    std::size_t get_numa_node_number(std::size_t num_thread) const { return _numa_node_numbers[num_thread % _num_of_pus]; }

    /// Return a bit mask where each set bit corresponds to a processing unit available to the application.
    MaskCRefType get_machine_affinity_mask() const;

    /// Return a bit mask where each set bit corresponds to a processing unit available to the given thread inside the socket it is running
    /// on.
    MaskCRefType get_socket_affinity_mask(std::size_t num_thread) const;

    /// Return a bit mask where each set bit corresponds to a processing unit available to the given thread inside the NUMA domain it is
    /// running on.
    MaskCRefType get_numa_node_affinity_mask(std::size_t num_thread) const;

    /// Return a bit mask where each set bit corresponds to a processing unit associated with the given NUMA node.
    MaskType get_numa_node_affinity_mask_from_numa_node(std::size_t num_node);

    /// Return a bit mask where each set bit corresponds to a processing unit available to the given thread inside the core it is running
    /// on.
    MaskCRefType get_core_affinity_mask(std::size_t num_thread) const;

    /// Return a bit mask where each set bit corresponds to a processing unit available to the given thread
    MaskCRefType get_thread_affinity_mask(std::size_t num_thread) const;

    /// Use the given bit mask to set the affinity of the given thread. Each bit corresponds to a processing unit the thread will be allowed
    /// to run on.
    void set_thread_affinity_mask(MaskCRefType mask);

    /// Return a bit mask where each set bit corresponds to a processing unit co-located with the memory the given address is currently
    /// allocated on.
    MaskType get_thread_affinity_mask_from_lva(void const *lva);

    /// Prints the \param m to os in a human-readable form
    void print_affinity_mask(std::ostream &os, std::size_t num_thread, MaskCRefType m, std::string const &pool_name) const;

    /// Reduce the thread priority of the current thread.
    bool reduce_thread_priority();

    /// Return the number of available NUMA domains
    std::size_t get_number_of_sockets() const;

    /// Returns the number of available NUMA domains
    std::size_t get_number_of_numa_nodes() const;

    /// Return the number of available cores
    std::size_t get_number_of_cores() const;

    /// Return the number of available hardware processing units
    std::size_t get_number_of_pus() const;

    /// Return number of cores in given NUMA domain
    std::size_t get_number_of_numa_node_cores(std::size_t numa);

    /// \brief Return number of processing units in a given numa domain
    std::size_t get_number_of_numa_node_pus(std::size_t numa);

    /// \brief Return number of processing units in a given socket
    std::size_t get_number_of_socket_pus(std::size_t socket);

    /// \brief Return number of processing units in given core
    std::size_t get_number_of_core_pus(std::size_t core);

    /// \brief Return number of cores units in given socket
    std::size_t get_number_of_socket_cores(std::size_t socket);

    std::size_t get_core_number(std::size_t num_thread) const { return _core_numbers[num_thread % _num_of_pus]; }

    std::size_t get_pu_number(std::size_t num_core, std::size_t num_pu);

    MaskType get_cpubind_mask_main_thread() const;
    void     set_cpubind_mask_main_thread(MaskType);
    MaskType get_cpubind_mask();
    MaskType get_cpubind_mask(std::thread &handle);

    /// convert a cpu mask into a numa node mask in hwloc bitmap form
    HwlocBitmapPtr cpuset_to_nodeset(MaskCRefType cpuset) const;

    void write_to_log() const;

    /// This is equivalent to malloc(), except that it tries to allocate
    /// page-aligned memory from the OS.
    void *allocate(std::size_t len) const;

    /// allocate memory with binding to a numa node set as
    /// specified by the policy and flags (see hwloc docs)
    void *allocate_membind(std::size_t len, HwlocBitmapPtr bitmap, HwlocMembindPolicy policy, int flags) const;

    MaskType get_area_membind_nodeset(void const *addr, std::size_t len);

    bool set_area_membind_nodeset(void const *addr, std::size_t len, void *nodeset) const;

    int get_numa_domain(void const *addr);

    /// Free memory that was previously allocated by allocate
    void deallocate(void *addr, std::size_t len) const;

    void print_vector(std::ostream &os, std::vector<std::size_t> const &v);
    void print_mask_vector(std::ostream &os, std::vector<MaskType> const &v);
    void print_hwloc(std::ostream &);

    MaskType init_socket_affinity_mask_from_socket(std::size_t num_socket);
    MaskType init_numa_node_affinity_mask_from_numa_node(std::size_t num_numa_node);
    MaskType init_core_affinity_mask_from_core(std::size_t num_core, MaskCRefType default_mask = empty_mask);
    MaskType init_thread_affinity_mask(std::size_t num_thread);
    MaskType init_thread_affinity_mask(std::size_t num_core, std::size_t num_pu);

    hwloc_bitmap_t mask_to_bitmap(MaskCRefType mask, hwloc_obj_type_t htype) const;
    MaskType       bitmap_to_mask(hwloc_bitmap_t bitmap, hwloc_obj_type_t htype);

  private:
    static MaskType    empty_mask;
    static std::size_t _memory_page_size;
    friend std::size_t get_memory_page_size();

    std::size_t init_node_number(std::size_t num_thread, hwloc_obj_type_t type);

    std::size_t init_socket_number(std::size_t num_thread) { return init_node_number(num_thread, HWLOC_OBJ_SOCKET); }

    std::size_t init_numa_node_number(std::size_t num_thread);

    std::size_t init_core_number(std::size_t num_thread) {
        return init_node_number(num_thread, _use_pus_as_cores ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);
    }

    void extract_node_mask(hwloc_obj_t parent, MaskType &mask);

    std::size_t extract_node_count(hwloc_obj_t parent, hwloc_obj_type_t type, std::size_t count);

    MaskType init_machine_affinity_mask();
    MaskType init_socket_affinity_mask(std::size_t num_thread) {
        return init_socket_affinity_mask_from_socket(get_socket_number(num_thread));
    }

    MaskType init_numa_node_affinity_mask(std::size_t num_thread) {
        return init_numa_node_affinity_mask_from_numa_node(get_numa_node_number(num_thread));
    }

    MaskType init_core_affinity_mask(std::size_t num_thread) {
        MaskType default_mask = _numa_node_affinity_masks[num_thread];
        return init_core_affinity_mask_from_core(get_core_number(num_thread), default_mask);
    }

    void init_num_of_pus();

    hwloc_topology_t _topology;
    std::mutex       _topology_mutex;

    static constexpr std::size_t pu_offset   = 0;
    static constexpr std::size_t core_offset = 0;

    std::size_t _num_of_pus;
    bool        _use_pus_as_cores;

    // Number masks:
    // Vectors of non-negative integers
    // Indicating which architecture object each PU belongs to.
    // For example, numa_node_numbers[0] indicates which numa node
    // number PU #0 (zero-based index) belongs to
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
    MaskType              _machine_affinity_mask;
    std::vector<MaskType> _socket_affinity_masks;
    std::vector<MaskType> _numa_node_affinity_masks;
    std::vector<MaskType> _core_affinity_masks;
    std::vector<MaskType> _thread_affinity_masks;
    MaskType              _main_thread_affinity_mask;
};

[[nodiscard]] EINSUMS_EXPORT unsigned int hardware_concurrency() noexcept;

///////////////////////////////////////////////////////////////////////////
// abstract away memory page size, calls to system functions are
// expensive, so return a value initialized at startup
inline std::size_t get_memory_page_size() {
    return Topology::_memory_page_size;
}

} // namespace einsums::hardware