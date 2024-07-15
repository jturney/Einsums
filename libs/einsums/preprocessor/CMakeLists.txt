#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(preprocessor_headers)

set(preprocessor_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums preprocessor
  GLOBAL_HEADER_GEN ON
  SOURCES ${preprocessor_sources}
  HEADERS ${preprocessor_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
