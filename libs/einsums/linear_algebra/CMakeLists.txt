#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(linear_algebra_headers)

set(linear_algebra_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums linear_algebra
  GLOBAL_HEADER_GEN ON
  SOURCES ${linear_algebra_sources}
  HEADERS ${linear_algebra_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
