# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_headers einsums/version.hpp)

# cmake-format: off
set(version_compat_headers)
# cmake-format: on

set(version_sources version.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums version
    GLOBAL_HEADER_GEN OFF
    SOURCES ${version_sources}
    HEADERS ${version_headers}
    MODULE_DEPENDENCIES einsums_config
    CMAKE_SUBDIRS examples tests
)
