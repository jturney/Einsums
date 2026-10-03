//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.h>

#include <cstdint>

// Two symbols every collector exports under these names, whatever its version, so that a copy
// loaded later can find the one already loaded. Never change their names, types or meaning.
extern "C" {
/// Present in every collector; its value is not read, only whether the symbol exists.
WAGGLE_C_EXPORT extern uint32_t const waggle_collector_marker_v1;

/// Tell this collector that another copy, at @p path (terminated) and built with interface
/// @p major . @p minor, found it and switched itself off.
WAGGLE_C_EXPORT void waggle_note_duplicate_v1(char const *path, uint32_t major, uint32_t minor);
}

WAGGLE_NAMESPACE_BEGIN

/// Whether this copy of the collector found another one already loaded and switched itself off.
/// Decided once, while the library loads, before any client can call in.
auto collector_inactive() noexcept -> bool;

WAGGLE_NAMESPACE_END
