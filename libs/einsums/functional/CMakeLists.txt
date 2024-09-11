#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(functional_headers einsums/functional/bind_back.hpp)

set(functional_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums functional
  GLOBAL_HEADER_GEN ON
  SOURCES ${functional_sources}
  HEADERS ${functional_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
