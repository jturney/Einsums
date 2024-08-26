#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(print_headers einsums/print.hpp)

set(print_sources print.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums print
        GLOBAL_HEADER_GEN OFF
        SOURCES ${print_sources}
        HEADERS ${print_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_config
        CMAKE_SUBDIRS examples tests
)
