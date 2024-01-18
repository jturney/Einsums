# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_base_headers einsums/threading_base/SchedulerState.hpp)

set(threading_base_sources)

include(einsums_add_module)
einsums_add_module(
    einsums threading_base
    GLOBAL_HEADER_GEN ON
    SOURCES ${threading_base_sources}
    HEADERS ${threading_base_headers}
    MODULE_DEPENDENCIES einsums_assertion einsums_config
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
