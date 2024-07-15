#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_headers)

set(runtime_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums runtime
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_sources}
  HEADERS ${runtime_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
