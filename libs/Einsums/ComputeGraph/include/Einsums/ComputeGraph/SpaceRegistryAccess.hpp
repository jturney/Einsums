//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// The one process-wide @ref einsums::compute_graph::SpaceRegistry.
///
/// @ref einsums::compute_graph::SpaceRegistry itself lives in ComputeGraphTypes and is entirely
/// header-inline, which is right for a type: anyone can own a registry, and tests want their own.
/// The process-global instance cannot follow it there. A function-local static defined in a header
/// gets one instance per binary that includes it, so libEinsums and the Python `_core` extension
/// would each hold their own registry and a space registered on one side would be missing on the
/// other. Declaring it here and defining it in the ComputeGraph library gives it exactly one home.
///
/// Ids from this registry are handles into it and mean nothing against any other registry.

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief The process-wide space registry.
 * @return The single registry shared by every graph in this process.
 *
 * Register spaces and declare their relations during startup or setup; queries then run from
 * anywhere. The registry's own mutex makes the concurrent case safe rather than fast.
 */
[[nodiscard]] APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RVP(reference) EINSUMS_EXPORT SpaceRegistry &global_space_registry();

EINSUMS_NAMESPACE_END(compute_graph)
