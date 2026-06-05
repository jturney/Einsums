#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HPTTHeaders
    Einsums/HPTT/ComputeNode.hpp
    Einsums/HPTT/HPTT.hpp
    Einsums/HPTT/HPTTTypes.hpp
    Einsums/HPTT/Macros.hpp
    Einsums/HPTT/Plan.hpp
    Einsums/HPTT/Transpose.hpp
    Einsums/HPTT/Utils.hpp
    Einsums/HPTT/Files.hpp
)

set(HPTTSources HPTT.cpp Plan.cpp Transpose.cpp Utils.cpp Primes.hpp Files.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums HPTT
  SOURCES ${HPTTSources}
  HEADERS ${HPTTHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Assertion Einsums_BLASBase Einsums_Errors Einsums_Logging Einsums_SIMD
  CMAKE_SUBDIRS examples tests
)
