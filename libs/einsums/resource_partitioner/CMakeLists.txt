#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(resource_partitioner_headers
        einsums/resource_partitioner/partitioner.hpp einsums/resource_partitioner/partitioner_fwd.hpp
        einsums/resource_partitioner/detail/create_partitioner.hpp
        einsums/resource_partitioner/detail/partitioner.hpp
)

set(resource_partitioner_sources detail_partitioner.cpp partitioner.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums resource_partitioner
        GLOBAL_HEADER_GEN ON
        SOURCES ${resource_partitioner_sources}
        HEADERS ${resource_partitioner_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_affinity einsums_config einsums_runtime_configuration einsums_errors
        einsums_ini einsums_threading_base einsums_thread_pools
        CMAKE_SUBDIRS examples tests
)
