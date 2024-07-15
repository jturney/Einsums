#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(config_headers)

set(config_sources)

include(Einsums_AddModule)
einsums_add_module(
    einsums config CONFIG_FILES
    GLOBAL_HEADER_GEN OFF
    SOURCES ${config_sources}
    HEADERS ${config_headers}
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
