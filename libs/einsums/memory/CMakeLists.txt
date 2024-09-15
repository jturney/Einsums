#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(memory_headers einsums/memory/intrusive_ptr.hpp einsums/memory/detail/sp_convertible.hpp)

set(memory_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums memory
  GLOBAL_HEADER_GEN ON
  SOURCES ${memory_sources}
  HEADERS ${memory_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    einsums_config
    einsums_assertion
  CMAKE_SUBDIRS examples tests
)
