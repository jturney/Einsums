#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(topology_headers einsums/topology/cpu_mask.hpp einsums/topology/topology.hpp)

set(topology_sources cpu_mask.cpp topology.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums topology
  GLOBAL_HEADER_GEN ON
  SOURCES ${topology_sources}
  HEADERS ${topology_headers}
  DEPENDENCIES einsums_internal::hwloc
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_errors
        einsums_logging
        einsums_type_support
        einsums_util
  CMAKE_SUBDIRS examples tests
)
