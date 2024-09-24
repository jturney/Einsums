#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_headers einsums/threading/jthread.hpp einsums/threading/thread.hpp)

set(threading_sources thread.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums threading
        GLOBAL_HEADER_GEN ON
        SOURCES ${threading_sources}
        HEADERS ${threading_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_errors einsums_lock_registration einsums_memory einsums_synchronization
        CMAKE_SUBDIRS examples tests
)
