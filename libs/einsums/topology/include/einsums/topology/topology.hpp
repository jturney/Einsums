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
#include <string>
#include <thread>
#include <vector>

namespace einsums::threads::detail {

struct einsums_hwloc_bitmap_wrapper {
  private:
    hwloc_bitmap_t _bmp;

  public:
    EINSUMS_NON_COPYABLE(einsums_hwloc_bitmap_wrapper);

    einsums_hwloc_bitmap_wrapper() : _bmp(nullptr) {}

    einsums_hwloc_bitmap_wrapper(void *bmp) : _bmp(reinterpret_cast<hwloc_bitmap_t>(bmp)) {}

    ~einsums_hwloc_bitmap_wrapper() { hwloc_bitmap_free(_bmp); }

    void reset(hwloc_bitmap_t bmp) {
        if (_bmp)
            hwloc_bitmap_free(_bmp);
        _bmp = bmp;
    }

    explicit operator bool() const noexcept { return _bmp != nullptr; }

    [[nodiscard]] auto get_bmp() const noexcept -> hwloc_bitmap_t { return _bmp; }

    friend EINSUMS_EXPORT auto operator<<(std::ostream &os, einsums_hwloc_bitmap_wrapper const *bmp) -> std::ostream &;
};

using hwloc_bitmap_ptr = std::shared_ptr<einsums_hwloc_bitmap_wrapper>;

enum einsums_hwloc_membind_policy : int {
    membind_default    = HWLOC_MEMBIND_DEFAULT,
    membind_firsttouch = HWLOC_MEMBIND_FIRSTTOUCH,
    membind_bind       = HWLOC_MEMBIND_BIND,
    membind_interleave = HWLOC_MEMBIND_INTERLEAVE,
#if HWLOC_API_VERSION < 0x00020000
    membind_replicate = HWLOC_MEMBIND_REPLICATE,
#endif
    membind_nexttouch = HWLOC_MEMBIND_NEXTTOUCH,
    membind_mixed     = HWLOC_MEMBIND_MIXED,
    membind_user      = HWLOC_MEMBIND_MIXED + 256
};

#include <einsums/config/warnings_prefix.hpp>

struct EINSUMS_EXPORT topology {
  private:
    static mask_type   empty_mask;
    static std::size_t memory_page_size_;
    friend std::size_t get_memory_page_size();

    std::size_t init_node_number(std::size_t num_thread, hwloc_obj_type_t type);

    std::size_t init_socket_number(std::size_t num_thread) { return init_node_number(num_thread, HWLOC_OBJ_SOCKET); }

    std::size_t init_numa_node_number(std::size_t num_thread);

    std::size_t init_core_number(std::size_t num_thread) {
        return init_node_number(num_thread, use_pus_as_cores_ ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);
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
        mask_type default_mask = numa_node_affinity_masks_[num_thread];
        return init_core_affinity_mask_from_core(get_core_number(num_thread), default_mask);
    }

    void init_num_of_pus();

    hwloc_topology_t topo;

    static constexpr std::size_t pu_offset   = 0;
    static constexpr std::size_t core_offset = 0;

    std::size_t num_of_pus_;
    bool        use_pus_as_cores_;

    using mutex_type = pika::concurrency::detail::spinlock;
    mutable mutex_type topo_mtx;

    // Number masks:
    // Vectors of non-negative integers
    // Indicating which architecture object each PU belongs to.
    // For example, numa_node_numbers[0] indicates which numa node
    // number PU #0 (zero-based index) belongs to
    std::vector<std::size_t> socket_numbers_;
    std::vector<std::size_t> numa_node_numbers_;
    std::vector<std::size_t> core_numbers_;

    // Affinity masks: vectors of bitmasks
    // - Length of the vector: number of PUs of the machine
    // - Elements of the vector:
    // Bitmasks of length equal to the number of PUs of the machine.
    // The bitmasks indicate which PUs belong to which resource.
    // For example, core_affinity_masks[0] is a bitmask, where the
    // elements = 1 indicate the PUs that belong to the core on which
    // PU #0 (zero-based index) lies.
    mask_type              machine_affinity_mask_;
    std::vector<mask_type> socket_affinity_masks_;
    std::vector<mask_type> numa_node_affinity_masks_;
    std::vector<mask_type> core_affinity_masks_;
    std::vector<mask_type> thread_affinity_masks_;
    mask_type              main_thread_affinity_mask_;

  public:
};

#include <einsums/config/warnings_suffix.hpp>

} // namespace einsums::threads::detail