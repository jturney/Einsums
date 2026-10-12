//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Detection shared by the module's sources, not part of its interface.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>

#include <filesystem>

EINSUMS_NAMESPACE_BEGIN(hardware::detail)

/// Cache sizes as the hardware reports them, before any override option is applied.
CacheSizes detected_cache_sizes();

/// Where measured constants are kept between runs: option::CacheDir, else the platform cache
/// location, else empty (then nothing is kept).
std::filesystem::path cache_directory();

/// Time an empty parallel region at the current team size: warm team, best of several trials.
double measure_omp_region_cost_ns();

EINSUMS_NAMESPACE_END(hardware::detail)
