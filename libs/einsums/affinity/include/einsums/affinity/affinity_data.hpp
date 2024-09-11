//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/topology/topology.hpp>

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace einsums::detail {

/// Structure holding information related to thread affinity selection
/// for the shepherd threads of this instance.
struct EINSUMS_EXPORT affinity_data {
    affinity_data();
    ~affinity_data();

    void init(std::size_t num_threads = 1, std::size_t max_cores = 1, std::size_t pu_offset = 0, std::size_t pu_step = 1,
              std::size_t used_cores = 0, std::string affinity_domain = "pu", std::string const &affinity_description = "balanced",
              bool use_process_mask = true);

    void set_num_threads(size_t num_threads) { _num_threads = num_threads; }

    void set_affinity_masks(std::vector<threads::detail::mask_type> const &affinity_masks) { _affinity_masks = affinity_masks; }
    void set_affinity_masks(std::vector<threads::detail::mask_type> &&affinity_masks) { _affinity_masks = std::move(affinity_masks); }

    std::size_t get_num_threads() const { return _num_threads; }

    bool using_process_mask() const noexcept { return _use_process_mask; }

    threads::detail::mask_cref_type get_pu_mask(threads::detail::topology const &topo, std::size_t num_thread) const;

    threads::detail::mask_type get_used_pus_mask(threads::detail::topology const &topo, std::size_t pu_num) const;
    std::size_t                get_thread_occupancy(threads::detail::topology const &topo, std::size_t pu_num) const;

    std::size_t get_pu_num(std::size_t num_thread) const {
        EINSUMS_ASSERT(num_thread < _pu_nums.size());
        return _pu_nums[num_thread];
    }
    void set_pu_nums(std::vector<std::size_t> const &pu_nums) { _pu_nums = pu_nums; }
    void set_pu_nums(std::vector<std::size_t> &&pu_nums) { _pu_nums = std::move(pu_nums); }

    void add_punit(std::size_t virt_core, std::size_t thread_num);
    void init_cached_pu_nums(std::size_t hardware_concurrency);

    std::size_t get_num_pus_needed() const { return _num_pus_needed; }

  protected:
    std::size_t get_pu_num(std::size_t num_thread, std::size_t hardware_concurrency) const;

  private:
    std::size_t                             _num_threads; ///< number of processing units managed
    std::size_t                             _pu_offset;   ///< offset of the first processing unit to use
    std::size_t                             _pu_step;     ///< step between processing units
    std::size_t                             _used_cores;
    std::string                             _affinity_domain;
    std::vector<threads::detail::mask_type> _affinity_masks;
    std::vector<std::size_t>                _pu_nums;
    threads::detail::mask_type              _no_affinity;      ///< mask of processing units which have no affinity
    bool                                    _use_process_mask; ///< use the process CPU mask to limit the available PUs
    std::size_t                             _num_pus_needed;
};

} // namespace einsums::detail