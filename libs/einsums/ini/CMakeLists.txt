#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ini_headers einsums/ini/ini.hpp)

set(ini_sources ini.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums ini
  GLOBAL_HEADER_GEN ON
  SOURCES ${ini_sources}
  HEADERS ${ini_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_assertion
        einsums_errors
        einsums_string_util
  CMAKE_SUBDIRS examples tests
)
