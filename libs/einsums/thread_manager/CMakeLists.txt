#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_manager_headers einsums/modules/thread_manager.hpp einsums/thread_manager/thread_manager_fwd.hpp)

set(thread_manager_sources thread_manager.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums thread_manager
        GLOBAL_HEADER_GEN OFF
        SOURCES ${thread_manager_sources}
        HEADERS ${thread_manager_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        einsums_program_options
        einsums_resource_partitioner
        einsums_runtime_configuration
        einsums_errors
        einsums_logging
        einsums_schedulers
        einsums_thread_pools
        CMAKE_SUBDIRS examples tests
)
