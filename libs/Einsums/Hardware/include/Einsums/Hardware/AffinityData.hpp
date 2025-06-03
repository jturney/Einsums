//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Hardware/Topology.hpp>

namespace einsums::hardware {

struct EINSUMS_EXPORT AffinityData {
    AffinityData();
    ~AffinityData();

    void init(std::size_t num_threads = 1, std::size_t max_cores = 1, std::size_t pu_offset = 0, std::size_t pu_step = 1,
              std::size_t used_cores = 0, std::string affinity_domain = "pu", std::string const &affinity_description = "balanced",
              bool use_process_mask = true);

    void set_num_threads(std::size_t num_threads) { _num_threads = num_threads; }

    void set_affinity_masks(std::vector<MaskType> const &affinity_masks) { _affinity_masks = affinity_masks; }
    void set_affinity_masks(std::vector<MaskType> &&affinity_masks) { _affinity_masks = std::move(affinity_masks); }

    std::size_t get_num_threads() const { return _num_threads; }

    bool use_process_mask() const noexcept { return _use_process_mask; }

    MaskCRefType get_pu_mask(Topology const &topo, std::size_t num_thread) const;

    MaskType get_used_pus_mask(Topology const &topo, std::size_t pu_num) const;

    std::size_t get_thread_occupancy(Topology const &topo, std::size_t pu_num) const;

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
    std::size_t              _num_threads{};
    std::size_t              _pu_offset{};
    std::size_t              _pu_step{};
    std::size_t              _used_cores{};
    std::string              _affinity_domain;
    std::vector<MaskType>    _affinity_masks;
    std::vector<std::size_t> _pu_nums;
    MaskType                 _no_affinity{};
    bool                     _use_process_mask{};
    std::size_t              _num_pus_needed{};
};

} // namespace einsums::hardware