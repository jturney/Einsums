#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(filesystem_headers einsums/filesystem.hpp)

set(filesystem_sources filesystem.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums filesystem
  GLOBAL_HEADER_GEN OFF
  SOURCES ${filesystem_sources}
  HEADERS ${filesystem_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
  CMAKE_SUBDIRS examples tests
)
