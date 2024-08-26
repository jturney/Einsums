#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(assertion_headers einsums/assert.hpp einsums/assertion/evaluate_assert.hpp)

set(assertion_sources assertion.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums assertion
  GLOBAL_HEADER_GEN OFF
  SOURCES ${assertion_sources}
  HEADERS ${assertion_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_preprocessor
  CMAKE_SUBDIRS examples tests
)
