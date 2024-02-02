# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(topology_headers einsums/topology/cpu_mask.hpp)

set(topology_sources cpu_mask.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums topology
    GLOBAL_HEADER_GEN ON
    SOURCES ${topology_sources}
    HEADERS ${topology_headers}
    MODULE_DEPENDENCIES einsums_assertion einsums_config einsums_errors einsums_string_util einsums_type_support
    DEPENDENCIES einsums_internal::hwloc
    CMAKE_SUBDIRS examples tests
)
