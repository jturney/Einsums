#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The SIMD library itself is Stripes (cmake/Einsums_SetupStripes.cmake). This module is what is
# Einsums' own about it: the --einsums:simd:arch option, and handing that and Einsums' logging to
# Stripes when the runtime starts.

set(SIMDHeaders Einsums/SIMD/Options.hpp)

set(SIMDSources Options.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums SIMD
  SOURCES ${SIMDSources}
  HEADERS ${SIMDHeaders}
  DEPENDENCIES Stripes::stripes
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging Einsums_Options
)
