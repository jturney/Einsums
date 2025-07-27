#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(WIN32)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HPTTHeaders
    Einsums/HPTT/ComputeNode.hpp
    Einsums/HPTT/HPTT.hpp
    Einsums/HPTT/HPTTTypes.hpp
    Einsums/HPTT/Macros.hpp
    Einsums/HPTT/Plan.hpp
    Einsums/HPTT/Transpose.hpp
    Einsums/HPTT/Utils.hpp
)

set(HPTTSources HPTT.cpp Plan.cpp Transpose.cpp Utils.cpp Primes.hpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums HPTT
  SOURCES ${HPTTSources}
  HEADERS ${HPTTHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_BLASBase Einsums_Config
  CMAKE_SUBDIRS examples tests
)
