//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace einsums {

/// What kind of post-materialization initialization a deferred tensor wants.
///
/// Set at declaration time (e.g. by ``Workspace::declare_zero_tensor``) and
/// consumed by ``make_handle`` so the init metadata flows through capture
/// into a ``Graph`` that the workspace doesn't directly own — the
/// loop-aware Materialization pass then emits the corresponding
/// Initialize node alongside Materialize.
///
/// Intentionally a parallel enum to ``compute_graph::InitKind`` (same
/// values, narrower set): the Tensor module sits below ComputeGraph in
/// the dep chain so it can't reference InitKind directly.
/// ``make_handle`` maps one to the other.
enum class PendingInit : std::uint8_t {
    None,
    Zero,
    Random,
};

} // namespace einsums
