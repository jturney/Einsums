# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(type_support_headers)

set(type_support_sources)

include(einsums_add_module)
einsums_add_module(
    einsums concepts
    GLOBAL_HEADER_GEN ON
    SOURCES ${type_support_sources}
    HEADERS ${type_support_headers}
    MODULE_DEPENDENCIES einsums_preprocessor
    CMAKE_SUBDIRS examples tests
)
