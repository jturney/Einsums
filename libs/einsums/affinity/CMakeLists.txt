#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(affinity_headers
        einsums/affinity/affinity_data.hpp
        einsums/affinity/parse_affinity_options.hpp
)

set(affinity_sources
        affinity_data.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums affinity
  GLOBAL_HEADER_GEN ON
  SOURCES ${affinity_sources}
  HEADERS ${affinity_headers}
  DEPENDENCIES
        einsums_internal::hwloc
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_errors
        einsums_topology
        einsums_util
  CMAKE_SUBDIRS examples tests
)
