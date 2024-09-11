//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/affinity/affinity_data.hpp>
#include <einsums/affinity/parse_affinity_options.hpp>
#include <einsums/assert.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/topology/cpu_mask.hpp>
#include <einsums/topology/topology.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace einsums::detail {

inline std::size_t count_initialized(std::vector<threads::detail::mask_type> const &masks) {
    std::size_t count = 0;
    for (threads::detail::mask_cref_type m : masks) {
        if (threads::detail::any(m))
            ++count;
    }

    return count;
}

affinity_data::affinity_data()
    : _num_threads(0), _pu_offset(static_cast<std::size_t>(-1)), _pu_step(1), _used_cores(0), _affinity_domain("pu"), _affinity_masks(),
      _pu_nums(), _no_affinity(), _use_process_mask(true), _num_pus_needed(0) {
    threads::detail::resize(_no_affinity, threads::detail::hardware_concurrency());
}

affinity_data::~affinity_data() = default;

void affinity_data::init(std::size_t num_threads, std::size_t max_cores, std::size_t pu_offset, std::size_t pu_step, std::size_t used_cores,
                         std::string affinity_domain, std::string const &affinity_description, bool use_process_mask) {
#if defined(__APPLE__)
    use_process_mask = false;
#endif

    _use_process_mask          = use_process_mask;
    _num_threads               = num_threads;
    std::size_t num_system_pus = threads::detail::hardware_concurrency();

    if (pu_offset == static_cast<std::size_t>(-1)) {
        _pu_offset = 0;
    } else {
        _pu_offset = pu_offset;
    }

    if (num_system_pus > 1) {
        _pu_step = pu_step % num_system_pus;
    }

    _affinity_domain = std::move(affinity_domain);
    _pu_nums.clear();

    init_cached_pu_nums(num_system_pus);

    auto const &topo = threads::detail::get_topology();

    if (affinity_description == "none") {
        // don't use any affinity for any of the os-threads
        threads::detail::resize(_no_affinity, num_system_pus);
        for (std::size_t i = 0; i != _num_threads; ++i) {
            threads::detail::set(_no_affinity, get_pu_num(i));
        }
    } else if (!affinity_description.empty()) {
        _affinity_masks.clear();
        _affinity_masks.resize(_num_threads, threads::detail::mask_type{});

        for (std::size_t i = 0; i != _num_threads; ++i) {
            threads::detail::resize(_affinity_masks[i], num_system_pus);
        }

        parse_affinity_options(affinity_description, _affinity_masks, used_cores, max_cores, _num_threads, _pu_nums, _use_process_mask);

        if (std::size_t num_initialized = count_initialized(_affinity_masks); num_initialized != _num_threads) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter,
                                    "The number of OS threads requested ({}) does not match the number of threads "
                                    "to bind ({})",
                                    _num_threads, num_initialized);
        }
    } else if (pu_offset == static_cast<std::size_t>(-1)) {
        // calculate the pu offset based on the used cores, but only if its
        // not explicitly specified
        for (std::size_t num_core = 0; num_core != used_cores; ++num_core) {
            _pu_offset += topo.get_number_of_core_pus(num_core);
        }
    }

    if (_used_cores == 0) {
        _used_cores = used_cores;
    }

    _pu_offset %= num_system_pus;

    std::vector<std::size_t> cores;
    cores.reserve(_num_threads);
    for (std::size_t i = 0; i != _num_threads; ++i) {
        std::size_t add_me = topo.get_core_number(get_pu_num(i));
        cores.push_back(add_me);
    }

    std::ranges::sort(cores);
    auto it = std::ranges::unique(cores).begin();
    cores.erase(it, cores.end());

    std::size_t num_unique_cores = cores.size();

    _num_pus_needed = std::max(num_unique_cores, max_cores);
}

threads::detail::mask_cref_type affinity_data::get_pu_mask(threads::detail::topology const &topo, std::size_t global_thread_num) const {
    // --einsums:bind=none disables all affinity
    if (threads::detail::test(_no_affinity, global_thread_num)) {
        static threads::detail::mask_type m = threads::detail::mask_type();
        threads::detail::resize(m, threads::detail::hardware_concurrency());
        return m;
    }

    // if we have individual, predefined affinity masks, return those
    if (!_affinity_masks.empty())
        return _affinity_masks[global_thread_num];

    // otherwise return mask based on affinity domain
    std::size_t pu_num = get_pu_num(global_thread_num);
    if (0 == std::string("pu").find(_affinity_domain)) {
        // The affinity domain is 'processing unit', just convert the
        // pu-number into a bit-mask.
        return topo.get_thread_affinity_mask(pu_num);
    }
    if (0 == std::string("core").find(_affinity_domain)) {
        // The affinity domain is 'core', return a bit mask corresponding
        // to all processing units of the core containing the given
        // pu_num.
        return topo.get_core_affinity_mask(pu_num);
    }
    if (0 == std::string("socket").find(_affinity_domain)) {
        // The affinity domain is 'socket', return a bit mask corresponding
        // to all processing units of the socket containing the
        // given pu_num.
        return topo.get_socket_affinity_mask(pu_num);
    }

    // The affinity domain is 'machine', return a bit mask corresponding
    // to all processing units of the machine.
    EINSUMS_ASSERT(0 == std::string("machine").find(_affinity_domain));
    return topo.get_machine_affinity_mask();
}

threads::detail::mask_type affinity_data::get_used_pus_mask(threads::detail::topology const &topo, std::size_t pu_num) const {
    threads::detail::mask_type ret = threads::detail::mask_type();
    threads::detail::resize(ret, threads::detail::hardware_concurrency());

    // --einsums:bind=none disables all affinity
    if (threads::detail::test(_no_affinity, pu_num)) {
        threads::detail::set(ret, pu_num);
        return ret;
    }

    for (std::size_t thread_num = 0; thread_num < _num_threads; ++thread_num) {
        ret |= get_pu_mask(topo, thread_num);
    }

    return ret;
}

std::size_t affinity_data::get_thread_occupancy(threads::detail::topology const &topo, std::size_t pu_num) const {
    std::size_t count = 0;
    if (threads::detail::test(_no_affinity, pu_num)) {
        ++count;
    } else {
        threads::detail::mask_type pu_mask = threads::detail::mask_type();

        threads::detail::resize(pu_mask, threads::detail::hardware_concurrency());
        threads::detail::set(pu_mask, pu_num);

        for (std::size_t num_thread = 0; num_thread < _num_threads; ++num_thread) {
            threads::detail::mask_cref_type affinity_mask = get_pu_mask(topo, num_thread);
            if (threads::detail::any(pu_mask & affinity_mask))
                ++count;
        }
    }
    return count;
}

// means of adding a processing unit after initialization
void affinity_data::add_punit(std::size_t virt_core, std::size_t thread_num) {
    std::size_t num_system_pus = threads::detail::hardware_concurrency();

    // initialize affinity_masks and set the mask for the given virt_core
    if (_affinity_masks.empty()) {
        _affinity_masks.resize(_num_threads);
        for (std::size_t i = 0; i != _num_threads; ++i)
            threads::detail::resize(_affinity_masks[i], num_system_pus);
    }
    threads::detail::set(_affinity_masks[virt_core], thread_num);

    // find first used pu, which is then stored as the pu_offset
    auto first_pu = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i != _num_threads; ++i) {
        std::size_t first = threads::detail::find_first(_affinity_masks[i]);
        first_pu          = (std::min)(first_pu, first);
    }
    if (first_pu != static_cast<std::size_t>(-1))
        _pu_offset = first_pu;

    init_cached_pu_nums(num_system_pus);
}

void affinity_data::init_cached_pu_nums(std::size_t hardware_concurrency) {
    if (_pu_nums.empty()) {
        _pu_nums.resize(_num_threads);
        for (std::size_t i = 0; i != _num_threads; ++i) {
            _pu_nums[i] = get_pu_num(i, hardware_concurrency);
        }
    }
}

std::size_t affinity_data::get_pu_num(std::size_t num_thread, std::size_t hardware_concurrency) const {
    // The offset shouldn't be larger than the number of available
    // processing units.
    EINSUMS_ASSERT(_pu_offset < hardware_concurrency);

    // The distance between assigned processing units shouldn't be zero
    EINSUMS_ASSERT(_pu_step > 0 && _pu_step <= hardware_concurrency);

    // We 'scale' the thread number to compute the corresponding
    // processing unit number.
    //
    // The base line processing unit number is computed from the given
    // pu-offset and pu-step.
    std::size_t num_pu = _pu_offset + _pu_step * num_thread;

    // We add an additional offset, which allows to 'roll over' if the
    // pu number would get larger than the number of available
    // processing units. Note that it does not make sense to 'roll over'
    // farther than the given pu-step.
    std::size_t offset = (num_pu / hardware_concurrency) % _pu_step;

    // The resulting pu number has to be smaller than the available
    // number of processing units.
    return (num_pu + offset) % hardware_concurrency;
}

} // namespace einsums::detail