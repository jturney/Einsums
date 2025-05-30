//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Alias.hpp>
#include <Einsums/Config/BranchHints.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Debug.hpp>
#include <Einsums/Config/Defines.hpp>
#include <Einsums/Config/Deleted.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Types.hpp>
#include <Einsums/Config/Version.hpp>

#if !defined(EINSUMS_ZERO)
#    define EINSUMS_ZERO (1.0e-10)
#endif
