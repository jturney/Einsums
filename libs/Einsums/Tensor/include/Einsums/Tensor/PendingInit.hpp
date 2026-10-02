//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <cstdint>

EINSUMS_NAMESPACE_BEGIN()

/// What kind of post-materialization initialization a deferred tensor wants.
///
/// Set at declaration (e.g. ``Workspace::declare_zero_tensor``) and carried by ``make_handle`` into
/// the graph, where Materialization emits an Initialize node. A narrower twin of
/// ``compute_graph::InitKind``, which this module sits below.
enum class PendingInit : std::uint8_t {
    None,
    Zero,
    Random,
};

EINSUMS_NAMESPACE_END()
