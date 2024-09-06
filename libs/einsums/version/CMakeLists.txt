#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_headers einsums/version.hpp)

set(version_sources version.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums version
  GLOBAL_HEADER_GEN OFF
  SOURCES ${version_sources}
  HEADERS ${version_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_preprocessor
  CMAKE_SUBDIRS examples tests
)

target_include_directories(einsums_version PRIVATE $<BUILD_INTERFACE:${EINSUMS_BINARY_DIR}>)
