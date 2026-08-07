//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(util)

/// Tries to break an attached debugger, if not supported a loop is
/// invoked which gives enough time to attach a debugger manually.
/// @versionadded{1.0.0}
EINSUMS_EXPORT void attach_debugger();

EINSUMS_NAMESPACE_END(util)