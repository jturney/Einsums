//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/PlacementPolicy.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

PlacementPolicy::~PlacementPolicy() = default;

Place CheapestPlacement::choose(PlacementQuery const &query) const {
    if (!query.eligible.device(0) || !query.host_us || !query.device_us) {
        return query.planned;
    }
    std::size_t to_host   = 0; // inputs only the device copy of which is current
    std::size_t to_device = 0; // inputs only the host copy of which is current
    for (PlacementOperand const &operand : query.inputs) {
        if (operand.valid == Validity::Device) {
            to_host += operand.bytes;
        } else if (operand.valid == Validity::Host) {
            to_device += operand.bytes;
        }
    }
    double const launch       = _model.gpu.gpu_launch_latency_us;
    double const host_total   = *query.host_us + _model.estimate_transfer_time_us(to_host);
    double const device_total = *query.device_us + launch + _model.estimate_transfer_time_us(to_device);

    if (device_total < host_total) {
        return Place::device(0);
    }
    if (query.run_gain_us && query.run_boundary_bytes &&
        *query.run_gain_us > _model.estimate_transfer_time_us(*query.run_boundary_bytes) + launch) {
        return Place::device(0);
    }
    return device_total == host_total ? query.planned : Place::host();
}

std::vector<std::string> const &builtin_placement_policy_names() {
    static std::vector<std::string> const names{"planned", "host", "device", "cheapest"};
    return names;
}

PlacementPolicy const &builtin_placement_policy(std::string_view value) {
    static PlannedPlacement const        planned;
    static HostPlacement const           host;
    static EligibleDevicePlacement const device;
    if (value == "planned") {
        return planned;
    }
    if (value == "host") {
        return host;
    }
    if (value == "device") {
        return device;
    }
    if (value == "cheapest") {
        // Built on first use only: detecting the cost model reads the machine profile, which a
        // process that never asks for this policy should not pay for.
        static CheapestPlacement const cheapest{CostModel::detect_default()};
        return cheapest;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "einsums:graph:placement: '{}' is not a placement policy; the valid values are {}",
                            value, "planned, host, device, cheapest");
}

EINSUMS_NAMESPACE_END(compute_graph)
