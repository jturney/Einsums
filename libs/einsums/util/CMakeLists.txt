#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(util_headers einsums/util/are_all_convertible.hpp einsums/util/arguments.hpp einsums/util/type_name.hpp)

set(util_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums util
  GLOBAL_HEADER_GEN OFF
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
