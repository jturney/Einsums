#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(logging_headers einsums/logging.hpp)

set(logging_sources logging.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums logging
  GLOBAL_HEADER_GEN OFF
  SOURCES ${logging_sources}
  HEADERS ${logging_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_assertion einsums_config einsums_string_util
  CMAKE_SUBDIRS examples tests
)
