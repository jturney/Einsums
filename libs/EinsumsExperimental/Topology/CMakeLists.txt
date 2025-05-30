#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TopologyHeaders Einsums/Topology/CPUMask.hpp Einsums/Topology/Topology.hpp)

set(TopologySources Topology.cpp CPUMask.cpp InitModule.hpp ModuleVars.hpp InitModule.cpp ModuleVars.cpp)

include(Einsums_AddModule)
einsums_add_module(
  EinsumsExperimental Topology
  SOURCES ${TopologySources}
  HEADERS ${TopologyHeaders}
  DEPENDENCIES Einsums einsums_internal::hwloc
  MODULE_DEPENDENCIES Einsums_Assertion Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME EinsumsExperimental
)


