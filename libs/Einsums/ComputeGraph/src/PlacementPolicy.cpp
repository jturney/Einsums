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

std::vector<std::string> const &builtin_placement_policy_names() {
    static std::vector<std::string> const names{"planned", "host", "device"};
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
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "einsums:graph:placement: '{}' is not a placement policy; the valid values are {}",
                            value, "planned, host, device");
}

EINSUMS_NAMESPACE_END(compute_graph)
