#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(synchronization_headers
        einsums/synchronization/async_rw_mutex.hpp
        einsums/synchronization/barrier.hpp
        einsums/synchronization/condition_variable.hpp
        einsums/synchronization/counting_semaphore.hpp
        einsums/synchronization/detail/condition_variable.hpp
        einsums/synchronization/detail/counting_semaphore.hpp
        einsums/synchronization/detail/sliding_semaphore.hpp
        einsums/synchronization/event.hpp
        einsums/synchronization/latch.hpp
        einsums/synchronization/lock_types.hpp
        einsums/synchronization/mutex.hpp
        einsums/synchronization/no_mutex.hpp
        einsums/synchronization/once.hpp
        einsums/synchronization/recursive_mutex.hpp
        einsums/synchronization/sliding_semaphore.hpp
        einsums/synchronization/stop_token.hpp
)

set(synchronization_sources barrier.cpp detail/condition_variable.cpp detail/counting_semaphore.cpp
        detail/sliding_semaphore.cpp mutex.cpp stop_token.cpp
)

include(Einsums_AddModule)
einsums_add_module(
        einsums synchronization
        GLOBAL_HEADER_GEN ON
        SOURCES ${synchronization_sources}
        HEADERS ${synchronization_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        einsums_config
        einsums_assertion
        einsums_execution_base
        einsums_concurrency
        einsums_coroutines
        einsums_datastructures
        einsums_errors
        einsums_functional
        #    einsums_hashing
        einsums_itt_notify
        einsums_memory
        einsums_lock_registration
        einsums_logging
        einsums_memory
        einsums_threading_base
        einsums_thread_support
        einsums_topology
        einsums_timing
        CMAKE_SUBDIRS examples tests
)
