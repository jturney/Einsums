#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_pools_headers
        einsums/thread_pools/scheduled_thread_pool.hpp
        einsums/thread_pools/scheduled_thread_pool_impl.hpp einsums/thread_pools/scheduling_loop.hpp
)

set(thread_pools_sources scheduled_thread_pool.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums thread_pools
        GLOBAL_HEADER_GEN ON
        SOURCES ${thread_pools_sources}
        HEADERS ${thread_pools_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_debugging
        einsums_errors
        einsums_itt_notify
        einsums_logging
        einsums_schedulers
        CMAKE_SUBDIRS examples tests
)
