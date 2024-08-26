#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(backtrace_headers)

set(backtrace_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums backtrace
        GLOBAL_HEADER_GEN ON
        SOURCES ${backtrace_sources}
        HEADERS ${backtrace_headers}
        DEPENDENCIES
        CMAKE_SUBDIRS examples tests
)
