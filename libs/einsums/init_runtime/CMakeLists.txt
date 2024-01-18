# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_headers)

set(init_runtime_sources)

include(einsums_add_module)
einsums_add_module(
    einsums init_runtime
    GLOBAL_HEADER_GEN ON
    SOURCES ${init_runtime_sources}
    HEADERS ${init_runtime_headers}
    MODULE_DEPENDENCIES einsums_type_support
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
