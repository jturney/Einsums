//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Assert.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/Hardware/ParseAffinityOptions.hpp>
#include <Einsums/Hardware/Topology.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hwloc.h>
#include <string>
#include <tuple>
#include <vector>

namespace einsums::hardware {

using BoundsType = std::vector<std::int64_t>;

void parse_mappings(std::string const &spec, Mapping &mapping) {
    if (spec == "compact") {
        mapping = Mapping::Compact;
    } else if (spec == "scatter") {
        mapping = Mapping::Scatter;
    } else if (spec == "balanced") {
        mapping = Mapping::Balanced;
    } else if (spec == "numa-balanced") {
        mapping = Mapping::NumaBalanced;
    } else {
        EINSUMS_THROW_EXCEPTION(bad_parameter, "failed to parse affinity specification: \"{}\"", spec);
    }
}

using MaskInfo = std::tuple<std::size_t, MaskType>;

inline std::size_t get_index(MaskInfo const &smi) {
    return std::get<0>(smi);
}
inline MaskCRefType get_mask(MaskInfo const &smi) {
    return std::get<1>(smi);
}

std::vector<MaskInfo> extract_socket_masks(Topology &t, BoundsType const &b) {
    std::vector<MaskInfo> masks;
    for (std::int64_t index : b) {
        masks.push_back(
            std::make_tuple(static_cast<std::size_t>(index), t.init_socket_affinity_mask_from_socket(static_cast<std::size_t>(index))));
    }
    return masks;
}

bool pu_in_process_mask(bool use_process_mask, Topology &t, std::size_t num_core, std::size_t num_pu) {
    if (!use_process_mask) {
        return true;
    }

    MaskType proc_mask = t.get_cpubind_mask_main_thread();
    MaskType pu_mask   = t.init_thread_affinity_mask(num_core, num_pu);

    return bit_and(proc_mask, pu_mask);
}

void check_num_threads(bool use_process_mask, Topology &t, std::size_t num_threads) {
    if (use_process_mask) {
        MaskType    proc_mask         = t.get_cpubind_mask_main_thread();
        std::size_t num_pus_proc_mask = count(proc_mask);

        if (num_threads > num_pus_proc_mask) {
            EINSUMS_THROW_EXCEPTION(bad_parameter,
                                    "specified number of threads ({}) is larger than number of processing units "
                                    "available in process mask ({})",
                                    num_threads, num_pus_proc_mask);
        }
    } else {
        std::size_t num_threads_available = hardware_concurrency();

        if (num_threads > num_threads_available) {
            EINSUMS_THROW_EXCEPTION(bad_parameter,
                                    "specified number of threads ({}) is larger than number of available "
                                    "processing units ({})",
                                    num_threads, num_threads_available);
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void decode_compact_distribution(Topology &t, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t max_cores,
                                 std::vector<std::size_t> &num_pus, bool use_process_mask)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    std::size_t num_threads = affinities.size();

    check_num_threads(use_process_mask, t, num_threads);

    if (use_process_mask) {
        used_cores = 0;
        max_cores  = t.get_number_of_cores();
    }

    std::size_t num_cores = std::min(max_cores, t.get_number_of_cores());
    num_pus.resize(num_threads);

    for (std::size_t num_thread = 0; num_thread < num_threads; /**/) {
        for (std::size_t num_core = 0; num_core < num_cores; num_core++) {
            std::size_t num_core_pus = t.get_number_of_core_pus(num_core + used_cores);
            for (std::size_t num_pu = 0; num_pu < num_core_pus; ++num_pu) {
                if (!pu_in_process_mask(use_process_mask, t, num_core, num_pu)) {
                    continue;
                }

                if (any(affinities[num_thread])) {
                    EINSUMS_THROW_EXCEPTION(bad_parameter, "affinity mask for thread {} has already been set", num_thread);
                }

                num_pus[num_thread]    = t.get_pu_number(num_core + used_cores, num_pu);
                affinities[num_thread] = t.init_thread_affinity_mask(num_core + used_cores, num_pu);

                if (++num_thread == num_threads)
                    return;
            }
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void decode_scatter_distribution(Topology &t, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t max_cores,
                                 std::vector<std::size_t> &num_pus, bool use_process_mask)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    std::size_t num_threads = affinities.size();

    check_num_threads(use_process_mask, t, num_threads);

    if (use_process_mask) {
        used_cores = 0;
        max_cores  = t.get_number_of_cores();
    }

    std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

    std::vector<std::size_t> next_pu_index(num_cores, 0);
    num_pus.resize(num_threads);

    for (std::size_t num_thread = 0; num_thread < num_threads; /**/) {
        for (std::size_t num_core = 0; num_core < num_cores; ++num_core) {
            if (any(affinities[num_thread])) {
                EINSUMS_THROW_EXCEPTION(bad_parameter, "affinity mask for thread {} has already been set", num_thread);
                return;
            }

            std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
            std::size_t pu_index     = next_pu_index[num_core];
            bool        use_pu       = false;

            // Find the next PU on this core which is in the process mask
            while (pu_index < num_core_pus) {
                use_pu = pu_in_process_mask(use_process_mask, t, num_core, pu_index);
                ++pu_index;

                if (use_pu) {
                    break;
                }
            }

            next_pu_index[num_core] = pu_index;

            if (!use_pu) {
                continue;
            }

            num_pus[num_thread]    = t.get_pu_number(num_core + used_cores, next_pu_index[num_core] - 1);
            affinities[num_thread] = t.init_thread_affinity_mask(num_core + used_cores, next_pu_index[num_core] - 1);

            if (++num_thread == num_threads)
                return;
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void decode_balanced_distribution(Topology &t, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t max_cores,
                                  std::vector<std::size_t> &num_pus, bool use_process_mask)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    std::size_t num_threads = affinities.size();

    check_num_threads(use_process_mask, t, num_threads);

    if (use_process_mask) {
        used_cores = 0;
        max_cores  = t.get_number_of_cores();
    }

    std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

    std::vector<std::size_t>              num_pus_cores(num_cores, 0);
    std::vector<std::size_t>              next_pu_index(num_cores, 0);
    std::vector<std::vector<std::size_t>> pu_indexes(num_cores);
    num_pus.resize(num_threads);

    // At first, calculate the number of used pus per core.
    // This needs to be done to make sure that we occupy all the available
    // cores
    for (std::size_t num_thread = 0; num_thread < num_threads; /**/) {
        for (std::size_t num_core = 0; num_core < num_cores; ++num_core) {
            std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
            std::size_t pu_index     = next_pu_index[num_core];
            bool        use_pu       = false;

            // Find the next PU on this core which is in the process mask
            while (pu_index < num_core_pus) {
                use_pu = pu_in_process_mask(use_process_mask, t, num_core, pu_index);
                ++pu_index;

                if (use_pu) {
                    break;
                }
            }

            next_pu_index[num_core] = pu_index;

            if (!use_pu) {
                continue;
            }

            pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

            num_pus_cores[num_core]++;
            if (++num_thread == num_threads)
                break;
        }
    }

    // Iterate over the cores and assigned pus per core. this additional
    // loop is needed so that we have consecutive worker thread numbers
    std::size_t num_thread = 0;
    for (std::size_t num_core = 0; num_core < num_cores; ++num_core) {
        for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core]; ++num_pu) {
            if (any(affinities[num_thread])) {
                EINSUMS_THROW_EXCEPTION(bad_parameter, "affinity mask for thread {} has already been set", num_thread);
                return;
            }

            num_pus[num_thread]    = t.get_pu_number(num_core + used_cores, pu_indexes[num_core][num_pu]);
            affinities[num_thread] = t.init_thread_affinity_mask(num_core + used_cores, pu_indexes[num_core][num_pu]);
            ++num_thread;
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void decode_numabalanced_distribution(Topology &t, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t /*max_cores*/,
                                      std::vector<std::size_t> &num_pus, bool use_process_mask)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    std::size_t num_threads = affinities.size();

    check_num_threads(use_process_mask, t, num_threads);

    if (use_process_mask) {
        used_cores = 0;
    }

    num_pus.resize(num_threads);

    // sockets
    std::size_t              num_sockets = (std::max)(std::size_t(1), t.get_number_of_sockets());
    std::vector<std::size_t> num_cores_socket(num_sockets, 0);
    std::vector<std::size_t> num_pus_socket(num_sockets, 0);
    std::vector<std::size_t> num_threads_socket(num_sockets, 0);
    for (std::size_t n = 0; n < num_sockets; ++n) {
        num_cores_socket[n] = t.get_number_of_socket_cores(n);
    }

    std::size_t core_offset = 0;
    std::size_t pus_t       = 0;
    for (std::size_t n = 0; n < num_sockets; ++n) {
        for (std::size_t num_core = 0; num_core < num_cores_socket[n]; ++num_core) {
            std::size_t num_pus = t.get_number_of_core_pus(num_core + core_offset);
            for (std::size_t num_pu = 0; num_pu < num_pus; ++num_pu) {
                if (pu_in_process_mask(use_process_mask, t, num_core + core_offset, num_pu)) {
                    ++num_pus_socket[n];
                }
            }
        }

        pus_t += num_pus_socket[n];
        core_offset += num_cores_socket[n];
    }

    // how many threads should go on each domain
    std::size_t pus_t2 = 0;
    for (std::size_t n = 0; n < num_sockets; ++n) {
        std::size_t temp =
            static_cast<std::size_t>(std::round(static_cast<double>(num_threads * num_pus_socket[n]) / static_cast<double>(pus_t)));

        // due to rounding up, we might have too many threads
        if ((pus_t2 + temp) > num_threads)
            temp = num_threads - pus_t2;
        pus_t2 += temp;
        num_threads_socket[n] = temp;
    }

    // assign threads to cores on each socket
    std::size_t num_thread = 0;
    core_offset            = 0;
    for (std::size_t n = 0; n < num_sockets; ++n) {
        std::vector<std::size_t>              num_pus_cores(num_cores_socket[n], 0);
        std::vector<std::size_t>              next_pu_index(num_cores_socket[n], 0);
        std::vector<std::vector<std::size_t>> pu_indexes(num_cores_socket[n]);

        // iterate once and count pus/core
        for (std::size_t num_thread_socket = 0; num_thread_socket < num_threads_socket[n];
             /**/) {
            for (std::size_t num_core = 0; num_core < num_cores_socket[n]; ++num_core) {
                std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
                std::size_t pu_index     = next_pu_index[num_core];
                bool        use_pu       = false;

                // Find the next PU on this core which is in the process mask
                while (pu_index < num_core_pus) {
                    use_pu = pu_in_process_mask(use_process_mask, t, num_core + core_offset, pu_index);
                    ++pu_index;

                    if (use_pu) {
                        break;
                    }
                }

                next_pu_index[num_core] = pu_index;

                if (!use_pu) {
                    continue;
                }

                pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

                num_pus_cores[num_core]++;
                if (++num_thread_socket == num_threads_socket[n])
                    break;
            }
        }

        // Iterate over the cores and assigned pus per core. this additional
        // loop is needed so that we have consecutive worker thread numbers
        for (std::size_t num_core = 0; num_core < num_cores_socket[n]; ++num_core) {
            for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core]; ++num_pu) {
                if (any(affinities[num_thread])) {
                    EINSUMS_THROW_EXCEPTION(bad_parameter, "affinity mask for thread {} has already been set", num_thread);
                    return;
                }
                num_pus[num_thread]    = t.get_pu_number(num_core + used_cores, pu_indexes[num_core][num_pu]);
                affinities[num_thread] = t.init_thread_affinity_mask(num_core + used_cores + core_offset, pu_indexes[num_core][num_pu]);
                ++num_thread;
            }
        }
        core_offset += num_cores_socket[n];
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void decode_distribution(Mapping d, Topology &t, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t max_cores,
                         std::size_t num_threads, std::vector<std::size_t> &num_pus, bool use_process_mask)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    affinities.resize(num_threads);
    switch (d) {
    case Mapping::Compact:
        decode_compact_distribution(t, affinities, used_cores, max_cores, num_pus, use_process_mask);
        break;

    case Mapping::Scatter:
        decode_scatter_distribution(t, affinities, used_cores, max_cores, num_pus, use_process_mask);
        break;

    case Mapping::Balanced:
        decode_balanced_distribution(t, affinities, used_cores, max_cores, num_pus, use_process_mask);
        break;

    case Mapping::NumaBalanced:
        decode_numabalanced_distribution(t, affinities, used_cores, max_cores, num_pus, use_process_mask);
        break;

    default:
        EINSUMS_ASSERT(false);
    }
}

void parse_affinity_options(std::string const &spec, std::vector<MaskType> &affinities, std::size_t used_cores, std::size_t max_cores,
                            std::size_t num_threads, std::vector<std::size_t> &num_pus, bool use_process_mask) {
    Mapping mappings;
    parse_mappings(spec, mappings);

    Topology &t = Topology::get_singleton();

    decode_distribution(mappings, t, affinities, used_cores, max_cores, num_threads, num_pus, use_process_mask);
}

} // namespace einsums::hardware